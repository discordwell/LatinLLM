#!/usr/bin/env python3
"""
LatinLLM Dataset Preparation Script
Downloads and prepares the Ken-Z/Latin-Audio dataset from Hugging Face.

Dataset: https://huggingface.co/datasets/Ken-Z/Latin-Audio
- ~73 hours of Classical Latin audio
- Macronized transcriptions
- CC-BY-4.0 license
"""

import os
import sys
from pathlib import Path
import json
import random
from typing import Dict, List, Tuple, Any

# Base paths - computed relative to this script's location
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Import orthography module
from orthography import normalize_text, build_ctc_vocabulary


def download_dataset(num_samples: int = None, cache_dir: str = None, streaming: bool = False):
    """
    Download the Latin-Audio dataset from Hugging Face.

    Args:
        num_samples: Number of samples to download (None for all)
        cache_dir: Directory to cache the dataset
        streaming: Use streaming mode (avoids rate limits)

    Returns:
        HuggingFace dataset object or list of samples
    """
    from datasets import load_dataset

    print("Downloading Ken-Z/Latin-Audio dataset from Hugging Face...")

    if cache_dir is None:
        cache_dir = str(DATA_DIR / "cache")

    if streaming or num_samples:
        # Use streaming mode to avoid rate limits and download only what we need
        print(f"Using streaming mode to fetch {num_samples or 'all'} samples...")
        dataset = load_dataset(
            "Ken-Z/Latin-Audio",
            split="train",
            streaming=True,
            cache_dir=cache_dir
        )

        # Collect samples
        samples = []
        for i, item in enumerate(dataset):
            if num_samples and i >= num_samples:
                break
            samples.append(item)
            if (i + 1) % 50 == 0:
                print(f"  Fetched {i + 1} samples...")

        print(f"Loaded {len(samples)} samples")
        return samples
    else:
        print("This may take a while for the first download (~5GB)...")
        dataset = load_dataset(
            "Ken-Z/Latin-Audio",
            split="train",
            cache_dir=cache_dir
        )
        print(f"Loaded {len(dataset)} samples")
        return dataset


def prepare_samples(
    dataset,
    output_dir: Path,
    orthography: str = "classical",
    min_duration: float = 1.0,
    max_duration: float = 30.0
) -> List[Dict[str, Any]]:
    """
    Prepare samples from the dataset.

    Args:
        dataset: HuggingFace dataset or list of samples
        output_dir: Directory to save audio files
        orthography: Target orthography ("classical" or "common")
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds

    Returns:
        List of prepared sample dictionaries
    """
    import soundfile as sf
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []

    # Handle both dataset and list inputs
    data_iter = dataset if isinstance(dataset, list) else dataset

    print(f"Preparing samples with {orthography} orthography...")

    for i, item in enumerate(data_iter):
        # Get audio
        audio = item["audio"]
        waveform = np.array(audio["array"], dtype=np.float32)
        sample_rate = audio["sampling_rate"]

        # Calculate duration
        duration = len(waveform) / sample_rate

        # Filter by duration
        if duration < min_duration or duration > max_duration:
            continue

        # Get and normalize transcription
        text = item["transcription"]
        normalized = normalize_text(text, target_orthography=orthography)

        if not normalized.strip():
            continue

        # Save audio file
        audio_filename = f"sample_{i:06d}.wav"
        audio_path = output_dir / audio_filename

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import torchaudio
            import torch
            waveform_tensor = torch.tensor(waveform).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform_tensor = resampler(waveform_tensor)
            waveform = waveform_tensor.squeeze().numpy()
            sample_rate = 16000

        sf.write(str(audio_path), waveform, sample_rate)

        samples.append({
            "audio": str(audio_path),
            "sentence": normalized,
            "original_text": text,
            "duration": duration,
            "orthography": orthography
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} samples, kept {len(samples)}")

    print(f"Prepared {len(samples)} samples")
    return samples


def split_data(
    samples: List[Dict],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples into train/val/test sets."""
    random.seed(seed)
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = samples[:train_end]
    val = samples[train_end:val_end]
    test = samples[val_end:]

    return train, val, test


def save_splits(train, val, test, output_dir: Path, orthography: str):
    """Save data splits to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("validation", val), ("test", test)]:
        filepath = output_dir / f"{name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} samples to {filepath}")

    # Build and save vocabulary
    all_texts = [s["sentence"] for s in train + val + test]
    vocab = build_ctc_vocabulary(all_texts, orthography=orthography)

    vocab_data = {
        "vocabulary": vocab,
        "orthography": orthography,
        "size": len(vocab)
    }
    vocab_file = output_dir / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary ({len(vocab)} chars) to {vocab_file}")

    # Save metadata
    metadata = {
        "orthography": orthography,
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "total_samples": len(train) + len(val) + len(test),
        "vocab_size": len(vocab),
        "source": "Ken-Z/Latin-Audio (Hugging Face)"
    }
    metadata_file = output_dir / "dataset_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Latin-Audio dataset")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to use (default: all)")
    parser.add_argument("--orthography", type=str, default="classical",
                       choices=["classical", "common"],
                       help="Target orthography (default: classical)")
    parser.add_argument("--min-duration", type=float, default=1.0,
                       help="Minimum audio duration in seconds")
    parser.add_argument("--max-duration", type=float, default=30.0,
                       help="Maximum audio duration in seconds")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for splitting")
    args = parser.parse_args()

    print("=" * 70)
    print("LatinLLM Dataset Preparation")
    print("=" * 70)
    print()

    # Download dataset
    dataset = download_dataset(num_samples=args.num_samples)

    # Prepare samples
    audio_dir = PROCESSED_DIR / "segments"
    samples = prepare_samples(
        dataset,
        audio_dir,
        orthography=args.orthography,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )

    if not samples:
        print("No samples prepared!")
        return

    # Split data
    print(f"\nSplitting data (90/5/5)...")
    train, val, test = split_data(samples, seed=args.seed)
    print(f"  Train: {len(train)}")
    print(f"  Validation: {len(val)}")
    print(f"  Test: {len(test)}")

    # Save splits
    print("\nSaving data splits...")
    save_splits(train, val, test, PROCESSED_DIR, args.orthography)

    # Summary
    total_duration = sum(s["duration"] for s in samples)
    print("\n" + "=" * 70)
    print("Dataset Preparation Complete!")
    print("=" * 70)
    print(f"\nTotal samples: {len(samples)}")
    print(f"Total duration: {total_duration/3600:.1f} hours")
    print(f"Orthography: {args.orthography}")
    print(f"\nOutput: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
