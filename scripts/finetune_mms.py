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

    # Training
    num_epochs: int = 5
    batch_size: int = 2  # Smaller batch for MPS memory
    gradient_accumulation_steps: int = 16  # Effective batch = 32
    learning_rate: float = 1e-3  # Higher LR for adapter-only training
    warmup_ratio: float = 0.1  # 10% of training for warmup
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient clipping to prevent explosion

    # Memory optimization
    fp16: bool = True
    gradient_checkpointing: bool = True
    max_audio_length: float = 10.0  # seconds (shorter for stability)

    # Output
    output_dir: str = str(MODELS_DIR / "mms-latin")
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10

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

def collate_fn(batch, processor):
    """Collate function for DataLoader."""
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

    # Process labels (text to token IDs) using tokenizer directly
    labels = processor.tokenizer(
        texts,
        return_tensors="pt",
        padding=True
    )

    inputs["labels"] = labels.input_ids

    # Debug: print label info for first few batches
    _DEBUG_BATCH_COUNT[0] += 1
    vocab_size = len(processor.tokenizer)

    # Always clamp labels to valid range (safety measure for MPS compatibility)
    # Use vocab_size - 1 as max to ensure labels < vocab_size
    inputs['labels'] = inputs['labels'].clamp(min=0, max=vocab_size-1)

    if _DEBUG_BATCH_COUNT[0] <= 3:
        print(f"\n[DEBUG batch {_DEBUG_BATCH_COUNT[0]}]")
        print(f"  Input shape: {inputs['input_values'].shape}")
        print(f"  Labels shape: {inputs['labels'].shape}")
        print(f"  Labels min/max: {inputs['labels'].min().item()}/{inputs['labels'].max().item()}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Sample text: {texts[0][:50]}")
        print(f"  Sample labels: {inputs['labels'][0][:20].tolist()}")

    # Replace padding with -100 for loss computation
    inputs["labels"] = inputs["labels"].masked_fill(
        inputs["labels"] == processor.tokenizer.pad_token_id, -100
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

    # Load model with Italian adapter (fp16 only on CUDA)
    use_fp16 = config.fp16 and torch.cuda.is_available()
    model = Wav2Vec2ForCTC.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16 if use_fp16 else torch.float32
    )

    # Enable SpecAugment for regularization on small dataset
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05

    # Load Italian adapter as starting point
    model.load_adapter(config.source_language)

    # Load Latin vocabulary
    vocab = create_latin_vocabulary(config.vocab_file)

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

    # Update model config for new vocabulary size
    model.config.vocab_size = len(vocab)

    # Reinitialize output layer for new vocabulary
    model.lm_head = torch.nn.Linear(
        model.config.hidden_size,
        len(vocab),
        bias=True
    )
    # Initialize with standard gain (0.1 was too small, causing vanishing gradients)
    torch.nn.init.xavier_uniform_(model.lm_head.weight, gain=1.0)
    torch.nn.init.zeros_(model.lm_head.bias)

    # Freeze base model, only train adapter and new head
    model.freeze_base_model()

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


def train(config: TrainingConfig):
    """Main training function."""
    from transformers import Trainer, TrainingArguments
    from functools import partial

    print("=" * 70)
    print("LatinLLM MMS Fine-Tuning")
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
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        num_train_epochs=config.num_epochs,
        fp16=config.fp16 and device == "cuda",
        gradient_checkpointing=config.gradient_checkpointing,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        data_collator=partial(collate_fn, processor=processor),
        compute_metrics=partial(compute_metrics, processor=processor),
    )

    # Train
    print("\nStarting training...")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
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

    # Final evaluation
    print("\nFinal evaluation...")
    results = trainer.evaluate()
    print(f"WER: {results.get('eval_wer', 'N/A'):.2%}")
    print(f"CER: {results.get('eval_cer', 'N/A'):.2%}")

    return trainer, results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune MMS for Latin")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR / "mms-latin"),
                       help="Output directory")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--source-lang", type=str, default="ita",
                       help="Source language adapter (default: ita/Italian)")
    parser.add_argument("--orthography", type=str, default="classical",
                       choices=["classical", "common"],
                       help="Target orthography")
    args = parser.parse_args()

    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        fp16=not args.no_fp16,
        source_language=args.source_lang,
        orthography=args.orthography
    )

    # Check data exists
    if not Path(config.train_file).exists():
        print(f"Error: Training data not found at {config.train_file}")
        print("Run prepare_dataset.py first.")
        return

    train(config)


if __name__ == "__main__":
    main()
