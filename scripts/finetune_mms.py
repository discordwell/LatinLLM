#!/usr/bin/env python3
"""
LatinLLM MMS Fine-Tuning Script
Fine-tunes MMS adapter layers for Latin ASR.

Key approach:
1. Load MMS with Italian adapter as starting point (closest to Latin)
2. Create Latin vocabulary from macronized transcripts
3. Initialize Latin adapter, freeze base model
4. Train only adapter (~2.5M params)
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Base paths - computed relative to this script's location
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure model directory exists
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_id: str = "facebook/mms-1b-all"
    source_language: str = "ita"  # Italian as starting point (closest to Latin)
    target_language: str = "lat"  # Latin (custom)

    # Training (based on HuggingFace MMS adapter best practices)
    num_epochs: int = 4
    batch_size: int = 2  # Smaller batch for MPS memory
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    learning_rate: float = 1e-3  # Higher LR for adapter training (HF recommended)
    lm_head_lr: float = 1e-3  # Same LR for lm_head
    warmup_ratio: float = 0.1
    weight_decay: float = 0.005
    lm_head_weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    freeze_lm_head: bool = False
    # Two-phase: Train lm_head FIRST (adapters frozen), then unfreeze adapters
    warmup_epochs_lm_head_only: int = 1

    # Memory optimization
    fp16: bool = True
    gradient_checkpointing: bool = False  # Disabled - can cause issues on MPS
    max_audio_length: float = 10.0  # seconds (shorter for stability)

    # Output
    output_dir: str = str(MODELS_DIR / "mms-latin")
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    max_steps: int = -1  # Set >0 to cap total optimizer steps (debugging)

    # Data
    train_file: str = str(PROCESSED_DIR / "train.json")
    val_file: str = str(PROCESSED_DIR / "validation.json")
    vocab_file: str = str(PROCESSED_DIR / "vocab.json")

    # Orthography
    orthography: str = "classical"


class LatinASRDataset(Dataset):
    """Dataset for Latin ASR training."""

    def __init__(
        self,
        data_file: str,
        processor,
        max_audio_length: float = 30.0,
        sample_rate: int = 16000
    ):
        self.processor = processor
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate

        # Load data
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load audio using soundfile (avoids TorchCodec dependency)
        import soundfile as sf
        audio_path = item["audio"]
        waveform, sr = sf.read(audio_path, dtype="float32")

        # Resample if needed
        if sr != self.sample_rate:
            waveform_tensor = torch.tensor(waveform).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform_tensor).squeeze().numpy()

        # Convert to mono if stereo
        if len(waveform.shape) > 1 and waveform.shape[1] == 2:
            waveform = waveform.mean(axis=1)

        # Truncate if too long
        max_samples = int(self.max_audio_length * self.sample_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        # Get text
        text = item["sentence"]

        return {
            "input_values": waveform,  # Key required by Trainer's LengthGroupedSampler
            "text": text
        }


_DEBUG_BATCH_COUNT = [0]

def collate_fn(batch, processor, latin_vocab):
    """Collate function for DataLoader.

    CRITICAL: Uses latin_vocab for label encoding, NOT processor.tokenizer.
    The processor.tokenizer is Italian (183 classes) but our lm_head is Latin (39 classes).
    Using Italian tokenizer causes label IDs > 38, which breaks CTC loss.
    """
    # Process audio
    audio_arrays = [item["input_values"] for item in batch]
    texts = [item["text"] for item in batch]

    # Check for NaN/Inf in input audio
    for i, arr in enumerate(audio_arrays):
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"Warning: NaN/Inf in audio sample {i}, replacing with zeros")
            audio_arrays[i] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Process inputs without attention mask (workaround for MPS boolean indexing issue)
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=False
    )

    # Ensure float32 dtype
    inputs["input_values"] = inputs["input_values"].float()

    # CRITICAL FIX: Encode labels using LATIN vocab, not Italian tokenizer
    # The Italian tokenizer maps 'ā'->129, 'ī'->84, etc. but our lm_head only has 39 outputs
    # We must use Latin vocab: 'ā'->22, 'ī'->17, etc.
    pad_id = latin_vocab.get("<pad>", 0)
    unk_id = latin_vocab.get("<unk>", 3)
    latin_vocab_size = len(latin_vocab)

    # Convert each text to Latin token IDs
    # CRITICAL: Encode spaces as pipe '|' (token 4), not literal space (token 5)
    # The Wav2Vec2 decoder expects pipe as word delimiter and converts it to space
    word_delimiter_id = latin_vocab.get('|', 4)

    label_ids_list = []
    for text in texts:
        # Convert text to lowercase and encode character by character
        text_lower = text.lower()
        ids = []
        for char in text_lower:
            if char == ' ':
                # Use pipe for word boundary (CTC convention)
                ids.append(word_delimiter_id)
            else:
                ids.append(latin_vocab.get(char, unk_id))
        label_ids_list.append(ids)

    # Pad to same length
    max_len = max(len(ids) for ids in label_ids_list)
    padded_labels = []
    for ids in label_ids_list:
        padded = ids + [pad_id] * (max_len - len(ids))
        padded_labels.append(padded)

    inputs["labels"] = torch.tensor(padded_labels, dtype=torch.long)

    # Debug: print label info for first few batches
    _DEBUG_BATCH_COUNT[0] += 1

    if _DEBUG_BATCH_COUNT[0] <= 3:
        print(f"\n[DEBUG batch {_DEBUG_BATCH_COUNT[0]}]")
        print(f"  Input shape: {inputs['input_values'].shape}")
        print(f"  Labels shape: {inputs['labels'].shape}")
        print(f"  Labels min/max: {inputs['labels'].min().item()}/{inputs['labels'].max().item()}")
        print(f"  Latin vocab size: {latin_vocab_size}")
        print(f"  Sample text: {texts[0][:50]}")
        print(f"  Sample labels: {inputs['labels'][0][:20].tolist()}")
        # Verify all labels are in valid range
        if inputs['labels'].max().item() >= latin_vocab_size:
            print(f"  ERROR: Label {inputs['labels'].max().item()} >= vocab size {latin_vocab_size}!")
        else:
            print(f"  ✓ All labels in valid range [0, {latin_vocab_size-1}]")

    # Replace padding with -100 for loss computation
    inputs["labels"] = inputs["labels"].masked_fill(
        inputs["labels"] == pad_id, -100
    )

    return inputs


def create_latin_vocabulary(vocab_file: str) -> Dict[str, int]:
    """Load or create Latin vocabulary."""
    if Path(vocab_file).exists():
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle both formats (direct vocab or nested)
            if "vocabulary" in data:
                return data["vocabulary"]
            return data

    # Default Latin character set (Classical with macrons)
    chars = list("abcdefghijklmnopqrstuvwxyz āēīōūȳ'")
    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4,
    }
    for c in chars:
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def setup_model_and_processor(config: TrainingConfig):
    """Initialize model and processor for fine-tuning."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
    from transformers import Wav2Vec2FeatureExtractor

    print(f"Loading MMS model: {config.model_id}")
    print(f"Source adapter: {config.source_language}")

    # Load Latin vocabulary FIRST (needed for model config)
    vocab = create_latin_vocabulary(config.vocab_file)
    print(f"Latin vocab ({len(vocab)} tokens): {vocab}")

    # Create tokenizer with Latin vocabulary
    vocab_dir = Path(config.output_dir) / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = vocab_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|"
    )

    # Load model with CRITICAL CTC fixes (per HuggingFace best practices)
    use_fp16 = config.fp16 and torch.cuda.is_available()
    model = Wav2Vec2ForCTC.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        # CRITICAL CTC fixes
        ctc_loss_reduction="mean",      # Use mean, not sum
        ctc_zero_infinity=True,         # Zero out infinite losses (prevents NaN)
        # Disable all dropout for small datasets
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        # New vocabulary size
        vocab_size=len(vocab),
        pad_token_id=tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,   # Allow lm_head resize
    )
    print(f"Model loaded with ctc_zero_infinity=True, ctc_loss_reduction='mean', all dropouts=0.0")

    # Disable masking - causes MPS crashes with boolean indexing
    model.config.mask_time_prob = 0.0
    model.config.mask_feature_prob = 0.0

    # Load Italian adapter as starting point for acoustic features
    model.load_adapter(config.source_language)
    print(f"Loaded Italian adapter as base")

    # CRITICAL FIX: load_adapter() overwrites lm_head with Italian vocab (183 tokens)
    # We need to replace it with a fresh lm_head for Latin vocab (39 tokens)
    import torch.nn as nn
    hidden_size = model.config.hidden_size  # 1280

    # Create new lm_head for Latin vocab
    model.lm_head = nn.Linear(hidden_size, len(vocab), bias=True)

    # Initialize with Xavier (as recommended for CTC output layers)
    nn.init.xavier_uniform_(model.lm_head.weight, gain=1.0)
    model.lm_head.bias.data.zero_()

    # CRITICAL: Set negative blank bias to prevent CTC collapse
    # Blank token (index 0) should be discouraged early in training
    model.lm_head.bias.data[0] = -5.0

    # Update config to match new vocab size
    model.config.vocab_size = len(vocab)

    print(f"Reinitialized lm_head for Latin vocab:")
    print(f"  Shape: {model.lm_head.weight.shape}")
    print(f"  Blank (pad) bias: {model.lm_head.bias.data[0].item():.2f}")

    # Feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    # Freeze base model, only train adapter and new head
    model.freeze_base_model()

    # IMPORTANT: Unfreeze adapter layers (freeze_base_model freezes them too)
    adapter_count = 0
    for name, param in model.named_parameters():
        if 'adapter' in name.lower():
            param.requires_grad = True
            adapter_count += 1
    print(f"Unfroze {adapter_count} adapter layer parameters")

    # Optionally freeze lm_head to only train adapter (for stability)
    if config.freeze_lm_head:
        print("FREEZING lm_head - only training adapter layers")
        for param in model.lm_head.parameters():
            param.requires_grad = False

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, processor, vocab


def compute_metrics(pred, processor):
    """Compute WER and CER metrics."""
    from evaluate import load

    wer_metric = load("wer")
    cer_metric = load("cer")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 with pad token
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # Filter empty strings
    pred_str = [p if p else " " for p in pred_str]
    label_str = [l if l else " " for l in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


from transformers import Trainer

class CTCTrainer(Trainer):
    """Custom Trainer that computes CTC loss manually to avoid label validation issues on MPS."""

    _last_valid_loss = 5.0  # Track last valid loss for recovery
    _nan_reported = False
    _nan_loss_reported = False

    def _log_nan(self, kind: str):
        if kind == "logits" and self._nan_reported:
            return
        if kind == "loss" and self._nan_loss_reported:
            return
        if kind == "logits":
            self._nan_reported = True
        if kind == "loss":
            self._nan_loss_reported = True
        step = getattr(getattr(self, "state", None), "global_step", "n/a")
        epoch = getattr(getattr(self, "state", None), "epoch", "n/a")
        print(f"WARNING: NaN/Inf in {kind} at step {step}, epoch {epoch}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import torch.nn.functional as F

        labels = inputs.pop("labels", None)
        outputs = model(**inputs)

        if labels is not None:
            logits = outputs.logits  # (batch, time, vocab)

            # Check for NaN in logits before proceeding
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                self._log_nan("logits")
                loss = torch.tensor(self._last_valid_loss, device=logits.device, requires_grad=True)
                return (loss, outputs) if return_outputs else loss

            # Clamp logits to prevent extreme values causing NaN in log_softmax
            logits = logits.clamp(min=-50, max=50)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (time, batch, vocab)

            # Check for NaN in log_probs
            if torch.isnan(log_probs).any():
                self._log_nan("logits")
                loss = torch.tensor(self._last_valid_loss, device=logits.device, requires_grad=True)
                return (loss, outputs) if return_outputs else loss

            # Get input lengths (all same length since no attention mask)
            input_lengths = torch.full(
                (logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device
            )

            # Get target lengths (count non -100 values)
            target_lengths = (labels != -100).sum(dim=1)

            # Replace -100 with 0 for CTC loss (it ignores based on target_lengths)
            labels_for_ctc = labels.clone()
            labels_for_ctc[labels_for_ctc == -100] = 0

            # Compute CTC loss
            loss = F.ctc_loss(
                log_probs,
                labels_for_ctc,
                input_lengths,
                target_lengths,
                blank=0,  # pad token is blank
                reduction="mean",
                zero_infinity=True  # Handle inf gracefully
            )

            # Replace NaN/Inf loss with last valid loss
            if torch.isnan(loss) or torch.isinf(loss):
                self._log_nan("loss")
                print(f"WARNING: Loss is {loss.item()}, using last valid loss {self._last_valid_loss}")
                loss = torch.tensor(self._last_valid_loss, device=logits.device, requires_grad=True)
            elif loss.item() > 0.01:  # Only update if loss is meaningful
                CTCTrainer._last_valid_loss = loss.item()
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


from transformers import TrainerCallback

class UnfreezeAdaptersCallback(TrainerCallback):
    """Callback to unfreeze adapter layers after warmup epochs.

    Two-phase training (per HuggingFace best practices):
    - Phase 1: Train lm_head only (adapters frozen)
    - Phase 2: Unfreeze adapters and train everything
    """

    def __init__(self, warmup_epochs: int, adapter_params: list):
        self.warmup_epochs = warmup_epochs
        self.adapter_params = adapter_params
        self.unfrozen = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not self.unfrozen and state.epoch >= self.warmup_epochs:
            print(f"\n>>> UNFREEZING ADAPTERS at epoch {state.epoch:.1f} <<<")
            for param in self.adapter_params:
                param.requires_grad = True
            self.unfrozen = True


def train(config: TrainingConfig):
    """Main training function with two-phase training."""
    from transformers import TrainingArguments
    from functools import partial

    print("=" * 70)
    print("LatinLLM MMS Fine-Tuning (Two-Phase)")
    print("=" * 70)
    print()

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Setup model
    model, processor, vocab = setup_model_and_processor(config)

    # Move to device (ensure float32 for MPS)
    if device == "cuda":
        model = model.cuda()
    elif device == "mps":
        model = model.float()  # Ensure float32 before moving to MPS
        model = model.to("mps")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = LatinASRDataset(
        config.train_file,
        processor,
        max_audio_length=config.max_audio_length
    )

    val_dataset = LatinASRDataset(
        config.val_file,
        processor,
        max_audio_length=config.max_audio_length
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        group_by_length=False,  # Disabled - custom dataset format
        remove_unused_columns=False,  # Keep 'text' column for our collate_fn
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="no",  # Disable eval - causes NaN on MPS
        save_strategy="epoch",  # Only save at end of epoch
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,  # Base LR (overridden by custom optimizer)
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        fp16=False,  # Disable fp16 entirely - causes issues on MPS
        gradient_checkpointing=False,  # Disable - causes issues on MPS
        save_total_limit=2,
        load_best_model_at_end=False,  # Disabled - uses eval which causes NaN
        push_to_hub=False,
        report_to="none",
        dataloader_num_workers=0,  # Disable multiprocessing - MPS issues
        dataloader_pin_memory=False,  # MPS doesn't support pinned memory
    )

    # Create custom optimizer with parameter groups (different LRs for lm_head vs adapters)
    # CRITICAL FIX: New lm_head needs much higher LR than pre-trained adapter layers
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    lm_head_params = []
    adapter_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lm_head" in name:
                lm_head_params.append(param)
            else:
                adapter_params.append(param)

    # TWO-PHASE TRAINING (HuggingFace best practice): Train lm_head FIRST
    # Phase 1: Freeze adapters, train only lm_head (learns Latin output mapping)
    # Phase 2: Unfreeze adapters to fine-tune acoustic features
    if config.warmup_epochs_lm_head_only > 0:
        print(f"\nTWO-PHASE TRAINING: Freezing ADAPTERS for first {config.warmup_epochs_lm_head_only} epoch(s)")
        print("  Phase 1: Train lm_head only (adapters frozen)")
        print("  Phase 2: Unfreeze adapters (train everything)")
        for param in adapter_params:
            param.requires_grad = False

    optimizer_grouped_parameters = [
        {
            "params": lm_head_params,
            "lr": config.lm_head_lr,
            "weight_decay": config.lm_head_weight_decay,
            "name": "lm_head"
        },
        {
            "params": adapter_params,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "name": "adapter"
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    # Calculate total training steps for scheduler
    num_training_steps = (
        len(train_dataset) // config.batch_size // config.gradient_accumulation_steps
    ) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print(f"\nOptimizer setup:")
    print(f"  lm_head params: {len(lm_head_params)}, lr={config.lm_head_lr}, wd={config.lm_head_weight_decay}")
    print(f"  adapter params: {len(adapter_params)}, lr={config.learning_rate}, wd={config.weight_decay}")
    print(f"  Total steps: {num_training_steps}, warmup: {num_warmup_steps}")

    # Setup callbacks for two-phase training
    callbacks = []
    if config.warmup_epochs_lm_head_only > 0:
        callbacks.append(UnfreezeAdaptersCallback(
            warmup_epochs=config.warmup_epochs_lm_head_only,
            adapter_params=adapter_params
        ))

    # Create Trainer (using custom CTCTrainer to avoid label validation issues on MPS)
    trainer = CTCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        data_collator=partial(collate_fn, processor=processor, latin_vocab=vocab),
        compute_metrics=partial(compute_metrics, processor=processor),
        optimizers=(optimizer, scheduler),  # Pass custom optimizer
        callbacks=callbacks,
    )

    # Train
    print("\nStarting training...")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rates: adapter={config.learning_rate}, lm_head={config.lm_head_lr}")
    if config.warmup_epochs_lm_head_only > 0:
        print(f"Two-phase training: lm_head-only for {config.warmup_epochs_lm_head_only} epoch(s), then full training")
    print()

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    # Save training config
    config_path = Path(config.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)

    print(f"\nModel saved to: {config.output_dir}")

    # Skip HuggingFace evaluation (causes NaN on MPS)
    # Run benchmark.py instead for proper evaluation
    print("\nTraining complete. Run benchmark.py to evaluate the model.")

    return trainer, {}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune MMS for Latin")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for adapter layers")
    parser.add_argument("--lm-head-lr", type=float, default=1e-4,
                       help="Learning rate for lm_head")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR / "mms-latin"),
                       help="Output directory")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--source-lang", type=str, default="ita",
                       help="Source language adapter (default: ita/Italian)")
    parser.add_argument("--orthography", type=str, default="classical",
                       choices=["classical", "common"],
                       help="Target orthography")
    parser.add_argument("--freeze-lm-head", action="store_true",
                       help="Freeze lm_head and only train adapter layers")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                       help="Epochs to train lm_head only before unfreezing adapters")
    parser.add_argument("--max-steps", type=int, default=-1,
                       help="Cap total optimizer steps (debugging; default: no cap)")
    args = parser.parse_args()

    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lm_head_lr=args.lm_head_lr,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        fp16=not args.no_fp16,
        source_language=args.source_lang,
        orthography=args.orthography,
        freeze_lm_head=args.freeze_lm_head,
        warmup_epochs_lm_head_only=args.warmup_epochs,
        max_steps=args.max_steps
    )

    # Check data exists
    if not Path(config.train_file).exists():
        print(f"Error: Training data not found at {config.train_file}")
        print("Run prepare_dataset.py first.")
        return

    train(config)


if __name__ == "__main__":
    main()
