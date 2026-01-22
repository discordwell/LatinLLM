#!/usr/bin/env python3
"""
LatinLLM UDHR Benchmark Script
Quick validation using a Latin test sample.

Uses Italian MMS adapter as baseline (closest living descendant of Latin).

Usage:
    python scripts/benchmark_udhr.py
"""

import os
import sys
from pathlib import Path
import json

import torch
import numpy as np

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
UDHR_DIR = DATA_DIR / "test" / "udhr_sample"

# Import orthography module
from orthography import to_common, normalize_text


def load_audio(audio_path: Path, target_sr: int = 16000):
    """Load and preprocess audio."""
    import torchaudio

    waveform, sr = torchaudio.load(str(audio_path))

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze().numpy(), target_sr


def transcribe_mms_italian(waveform: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe using MMS with Italian adapter."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print("Loading MMS model (Italian adapter)...")

    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")

    # Set to Italian (closest to Latin)
    processor.tokenizer.set_target_lang("ita")
    model.load_adapter("ita")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Running inference on {device}...")

    # Process audio
    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription


def compute_metrics(reference: str, hypothesis: str):
    """Compute WER and CER."""
    from jiwer import wer, cer

    ref = reference.strip()
    hyp = hypothesis.strip()

    if not ref:
        return (1.0, 1.0) if hyp else (0.0, 0.0)
    if not hyp:
        return (1.0, 1.0)

    return wer(ref, hyp), cer(ref, hyp)


def download_audio_if_needed():
    """Download UDHR audio if not present."""
    audio_path = UDHR_DIR / "audio.mp3"

    if audio_path.exists():
        return audio_path

    print("UDHR audio not found. Downloading...")

    url = "https://archive.org/download/LatinMp3_242/latinaudacity.mp3"

    import urllib.request
    UDHR_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(audio_path))

    print(f"Downloaded to {audio_path}")
    return audio_path


def main():
    print("=" * 70)
    print("LatinLLM UDHR Benchmark")
    print("=" * 70)
    print()

    # Ensure audio exists
    audio_path = download_audio_if_needed()

    # Load reference transcript
    ref_file = UDHR_DIR / "reference.txt"
    if not ref_file.exists():
        print(f"Error: Reference file not found at {ref_file}")
        return

    with open(ref_file, "r", encoding="utf-8") as f:
        reference = f.read().strip()

    print(f"Reference text (Classical): {reference[:60]}...")
    print()

    # Load audio
    print(f"Loading audio from {audio_path}...")
    waveform, sr = load_audio(audio_path)
    duration = len(waveform) / sr
    print(f"Audio duration: {duration:.1f}s")
    print()

    # Transcribe with MMS Italian
    transcription = transcribe_mms_italian(waveform, sr)

    # Normalize both
    ref_classical = normalize_text(reference, target_orthography="classical")
    ref_common = to_common(ref_classical)

    hyp_classical = normalize_text(transcription, target_orthography="classical")
    hyp_common = to_common(hyp_classical)

    # Compute metrics for both orthographies
    wer_classical, cer_classical = compute_metrics(ref_classical, hyp_classical)
    wer_common, cer_common = compute_metrics(ref_common, hyp_common)

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Metric':<20} {'Classical':>15} {'Common':>15}")
    print("-" * 50)
    print(f"{'WER':<20} {wer_classical:>15.1%} {wer_common:>15.1%}")
    print(f"{'CER':<20} {cer_classical:>15.1%} {cer_common:>15.1%}")
    print("-" * 50)
    print()
    print("Reference (Classical):")
    print(f"  {ref_classical}")
    print()
    print("Hypothesis (normalized):")
    print(f"  {hyp_classical}")
    print()

    # Save results
    results = {
        "model": "MMS Baseline (Italian)",
        "audio_file": str(audio_path),
        "duration_seconds": duration,
        "wer_classical": wer_classical,
        "wer_common": wer_common,
        "cer_classical": cer_classical,
        "cer_common": cer_common,
        "reference_classical": ref_classical,
        "reference_common": ref_common,
        "hypothesis_classical": hyp_classical,
        "hypothesis_common": hyp_common
    }

    results_file = UDHR_DIR / "benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {results_file}")
    print()
    print("Note: This test uses classroom audio that may not match the reference text.")
    print("For accurate benchmarking, use matched audio-transcript pairs from the dataset.")


if __name__ == "__main__":
    main()
