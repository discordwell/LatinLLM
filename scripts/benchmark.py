#!/usr/bin/env python3
"""
LatinLLM Benchmark Script
Evaluates ASR models on Latin audio using test samples.

Uses Italian as the MMS baseline proxy (closest Romance language to Latin).

Usage:
    python scripts/benchmark.py --num-samples 100
    python scripts/benchmark.py --model path/to/model
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import numpy as np

# Base paths - computed relative to this script's location
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Import orthography module
from orthography import to_common, normalize_text


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    num_samples: int
    wer_classical: float
    wer_common: float
    cer_classical: float
    cer_common: float
    sample_predictions: List[Dict[str, str]]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    from jiwer import wer
    if not reference.strip():
        return 1.0 if hypothesis.strip() else 0.0
    if not hypothesis.strip():
        return 1.0
    return wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    from jiwer import cer
    if not reference.strip():
        return 1.0 if hypothesis.strip() else 0.0
    if not hypothesis.strip():
        return 1.0
    return cer(reference, hypothesis)


def load_test_data(test_file: Path, num_samples: int = None) -> List[Dict]:
    """Load test samples."""
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]

    return data


def transcribe_mms_baseline(audio_path: str, sample_rate: int = 16000) -> str:
    """
    Transcribe using MMS baseline with Italian adapter.
    Italian is the closest living language to Latin.
    """
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torchaudio

    # Load model (cached after first call)
    if not hasattr(transcribe_mms_baseline, "model"):
        print("Loading MMS baseline (Italian adapter)...")
        transcribe_mms_baseline.model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
        transcribe_mms_baseline.processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
        transcribe_mms_baseline.processor.tokenizer.set_target_lang("ita")
        transcribe_mms_baseline.model.load_adapter("ita")
        transcribe_mms_baseline.model.eval()
        transcribe_mms_baseline.device = "cuda" if torch.cuda.is_available() else "cpu"
        transcribe_mms_baseline.model.to(transcribe_mms_baseline.device)

    model = transcribe_mms_baseline.model
    processor = transcribe_mms_baseline.processor
    device = transcribe_mms_baseline.device

    # Load audio using soundfile backend (avoids TorchCodec dependency)
    import soundfile as sf
    waveform_np, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.tensor(waveform_np).unsqueeze(0)  # Add channel dim

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    waveform = waveform.squeeze().numpy()

    # Process
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription


def transcribe_finetuned(audio_path: str, model_path: str) -> str:
    """Transcribe using fine-tuned model."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torchaudio

    # Load model (cached)
    cache_key = f"finetuned_{model_path}"
    if not hasattr(transcribe_finetuned, cache_key):
        print(f"Loading fine-tuned model from {model_path}...")
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        setattr(transcribe_finetuned, cache_key, (model, processor, device))

    model, processor, device = getattr(transcribe_finetuned, cache_key)

    # Load and process audio using soundfile backend
    import soundfile as sf
    waveform_np, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.tensor(waveform_np).unsqueeze(0)  # Add channel dim

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    waveform = waveform.squeeze().numpy()

    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription


def evaluate_model(
    test_data: List[Dict],
    transcribe_fn,
    model_name: str
) -> BenchmarkResult:
    """Evaluate a model on test data."""
    wer_classical_list = []
    wer_common_list = []
    cer_classical_list = []
    cer_common_list = []
    sample_predictions = []

    print(f"\nEvaluating {model_name} on {len(test_data)} samples...")

    for i, item in enumerate(test_data):
        # Get reference
        ref_classical = normalize_text(item["sentence"], target_orthography="classical")
        ref_common = to_common(ref_classical)

        # Transcribe
        try:
            prediction = transcribe_fn(item["audio"])
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            prediction = ""

        # Normalize prediction
        pred_classical = normalize_text(prediction, target_orthography="classical")
        pred_common = to_common(pred_classical)

        # Compute metrics
        wer_c = compute_wer(ref_classical, pred_classical)
        wer_p = compute_wer(ref_common, pred_common)
        cer_c = compute_cer(ref_classical, pred_classical)
        cer_p = compute_cer(ref_common, pred_common)

        wer_classical_list.append(wer_c)
        wer_common_list.append(wer_p)
        cer_classical_list.append(cer_c)
        cer_common_list.append(cer_p)

        # Store sample (first 10 only)
        if len(sample_predictions) < 10:
            sample_predictions.append({
                "reference": ref_classical,
                "prediction": pred_classical
            })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(test_data)}")

    return BenchmarkResult(
        model_name=model_name,
        num_samples=len(test_data),
        wer_classical=np.mean(wer_classical_list),
        wer_common=np.mean(wer_common_list),
        cer_classical=np.mean(cer_classical_list),
        cer_common=np.mean(cer_common_list),
        sample_predictions=sample_predictions
    )


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results."""
    print("\n" + "=" * 70)
    print("LATIN BENCHMARK RESULTS")
    print("=" * 70)

    print("\n" + "-" * 70)
    print(f"{'Model':<30} {'WER (Cls)':>10} {'WER (Com)':>10} {'CER (Cls)':>10} {'CER (Com)':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r.model_name:<30} {r.wer_classical:>10.1%} {r.wer_common:>10.1%} "
              f"{r.cer_classical:>10.1%} {r.cer_common:>10.1%}")

    print("-" * 70)

    # Show sample predictions
    print("\nSample Predictions (first result):")
    if results and results[0].sample_predictions:
        for j, sample in enumerate(results[0].sample_predictions[:3]):
            print(f"\n  [{j+1}]")
            print(f"  Ref: {sample['reference'][:80]}...")
            print(f"  Hyp: {sample['prediction'][:80]}...")


def save_results(results: List[BenchmarkResult], output_path: Path):
    """Save results to JSON."""
    output = {
        "benchmark": "Latin ASR",
        "results": []
    }

    for r in results:
        output["results"].append({
            "model": r.model_name,
            "num_samples": r.num_samples,
            "wer_classical": r.wer_classical,
            "wer_common": r.wer_common,
            "cer_classical": r.cer_classical,
            "cer_common": r.cer_common,
            "sample_predictions": r.sample_predictions
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LatinLLM Benchmark")
    parser.add_argument("--num-samples", type=int, default=50,
                       help="Number of test samples to evaluate")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to fine-tuned model")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline evaluation")
    parser.add_argument("--test-file", type=str, default=None,
                       help="Path to test JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("LatinLLM Benchmark")
    print("=" * 70)

    # Find test file
    if args.test_file:
        test_file = Path(args.test_file)
    else:
        test_file = PROCESSED_DIR / "test.json"

    if not test_file.exists():
        print(f"Error: Test file not found at {test_file}")
        print("Run prepare_dataset.py first to create test data.")
        return

    # Load test data
    test_data = load_test_data(test_file, args.num_samples)
    print(f"Loaded {len(test_data)} test samples")

    results = []

    # Baseline evaluation
    if not args.skip_baseline:
        print("\n--- MMS Baseline (Italian) ---")
        result = evaluate_model(
            test_data,
            transcribe_mms_baseline,
            "MMS Baseline (Italian)"
        )
        results.append(result)

    # Fine-tuned model evaluation
    model_path = args.model or (MODELS_DIR / "mms-latin")
    if Path(model_path).exists():
        print(f"\n--- Fine-tuned Model ---")
        result = evaluate_model(
            test_data,
            lambda x: transcribe_finetuned(x, str(model_path)),
            "MMS Fine-tuned"
        )
        results.append(result)
    elif args.model:
        print(f"Error: Model not found at {model_path}")
    else:
        print(f"\nNo fine-tuned model at {model_path}")
        print("Train one with: python scripts/finetune_mms.py")

    # Print and save results
    if results:
        print_results(results)
        output_path = PROCESSED_DIR / "benchmark_results.json"
        save_results(results, output_path)


if __name__ == "__main__":
    main()
