#!/usr/bin/env python3
"""Diagnose warmstart issues by checking vocab mapping and weight initialization."""
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def main():
    print("=" * 70)
    print("Warmstart Diagnostic (Testing Fixes)")
    print("=" * 70)

    # Load MMS model with Italian adapter
    print("\n1. Loading MMS model with Italian adapter...")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    model.load_adapter("ita")

    # Get Italian vocab from actual tokenizer
    print("\n2. Loading Italian vocabulary from MMS tokenizer...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
    processor.tokenizer.set_target_lang("ita")
    italian_vocab = processor.tokenizer.get_vocab()

    print(f"   Italian vocab size: {len(italian_vocab)}")

    # Load Latin vocab
    print("\n3. Loading Latin vocabulary...")
    vocab_file = PROJECT_ROOT / "data" / "processed" / "vocab.json"
    with open(vocab_file, 'r') as f:
        data = json.load(f)
        latin_vocab = data.get('vocabulary', data)

    print(f"   Latin vocab size: {len(latin_vocab)}")

    # Check Italian blank bias
    print("\n4. Italian lm_head bias values:")
    italian_bias = model.lm_head.bias.data
    italian_weight = model.lm_head.weight.data
    print(f"   Blank (idx 0) bias: {italian_bias[0].item():.4f}")

    # Simulate warmstart WITH THE FIX
    print("\n5. Simulating warmstart WITH FIX...")
    new_lm_head = torch.nn.Linear(model.config.hidden_size, len(latin_vocab), bias=True)
    torch.nn.init.xavier_uniform_(new_lm_head.weight, gain=0.1)
    new_lm_head.bias.data.zero_()

    # Copy blank bias FIRST
    italian_blank_bias = italian_bias[0].item()
    if italian_blank_bias < 0:
        target_blank_bias = italian_blank_bias
        print(f"   Copying Italian blank bias: {italian_blank_bias:.4f}")
    else:
        target_blank_bias = -5.0
        print(f"   Setting blank bias to -5.0 (Italian was positive: {italian_blank_bias:.4f})")

    new_lm_head.bias.data[0] = target_blank_bias

    # Copy shared character weights (SKIP special tokens)
    copied_count = 0
    with torch.no_grad():
        for latin_char, latin_idx in latin_vocab.items():
            # SKIP special tokens - their bias is set separately
            if latin_char in ['<pad>', '<s>', '</s>', '<unk>']:
                continue
            if latin_char in italian_vocab:
                italian_idx = italian_vocab[latin_char]
                if italian_idx < italian_weight.shape[0]:
                    new_lm_head.weight.data[latin_idx] = italian_weight[italian_idx]
                    new_lm_head.bias.data[latin_idx] = italian_bias[italian_idx]
                    copied_count += 1

    # RESTORE blank bias (in case it was accidentally overwritten)
    new_lm_head.bias.data[0] = target_blank_bias

    print(f"   Copied {copied_count}/{len(latin_vocab)} character weights")

    print("\n6. VERIFICATION - Latin lm_head after warmstart:")
    print(f"   Bias[0] (blank): {new_lm_head.bias.data[0].item():.4f}")
    if new_lm_head.bias.data[0].item() < -1.0:
        print("   ✓ BLANK BIAS FIX WORKING!")
    else:
        print("   ✗ ERROR: Blank bias is not negative enough!")

    print(f"   Bias range: [{new_lm_head.bias.data.min().item():.4f}, {new_lm_head.bias.data.max().item():.4f}]")

    # Quick inference test
    print("\n7. Quick inference test...")
    model.lm_head = new_lm_head
    model.config.vocab_size = len(latin_vocab)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 16000)  # 1 second of audio
    with torch.no_grad():
        outputs = model(dummy_input)
        logits = outputs.logits

    # Check prediction distribution
    predicted_ids = torch.argmax(logits[0], dim=-1)
    blank_pct = (predicted_ids == 0).sum().item() / len(predicted_ids) * 100
    print(f"   Blank prediction %: {blank_pct:.1f}%")
    if blank_pct < 99:
        print("   ✓ Model is not collapsing to all blanks!")
    else:
        print("   ✗ WARNING: Model may be collapsing to blanks")

    unique_ids, counts = predicted_ids.unique(return_counts=True)
    print(f"   Unique predicted tokens: {unique_ids.tolist()[:10]}...")

    print("\n" + "=" * 70)
    print("Diagnostic complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
