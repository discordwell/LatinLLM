# LatinLLM

Latin Automatic Speech Recognition using fine-tuned MMS (Massively Multilingual Speech).

## Overview

LatinLLM fine-tunes Meta's MMS model for Classical Latin speech recognition. It uses Italian as a starting point (the closest living descendant of Latin) and trains on the [Ken-Z/Latin-Audio](https://huggingface.co/datasets/Ken-Z/Latin-Audio) dataset (~73 hours of macronized Latin audio).

## Features

- **Orthography Support**: Classical Latin (with macrons: ā, ē, ī, ō, ū) and Common Latin (without macrons)
- **Pre-built Dataset**: Uses Hugging Face's Latin-Audio dataset with macronized transcriptions
- **Efficient Fine-tuning**: Only trains adapter layers (~2.5M parameters)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download and prepare dataset (uses a subset by default)
python scripts/prepare_dataset.py --num-samples 1000

# Run baseline benchmark (Italian adapter)
python scripts/benchmark.py --num-samples 50

# Fine-tune for Latin
python scripts/finetune_mms.py --epochs 10

# Evaluate fine-tuned model
python scripts/benchmark.py
```

## Project Structure

```
LatinLLM/
├── scripts/
│   ├── orthography.py      # Classical/Common Latin conversions
│   ├── prepare_dataset.py  # Download & prepare Latin-Audio
│   ├── benchmark.py        # Evaluate models
│   └── finetune_mms.py     # Fine-tune MMS for Latin
├── data/
│   ├── processed/          # Prepared training data
│   └── cache/              # HuggingFace dataset cache
├── models/                 # Trained models
└── tests/                  # Test suite
```

## Orthography

Latin has two main written forms:

| System | Example | Description |
|--------|---------|-------------|
| Classical | "Arma virumque canō" | Macrons mark long vowels |
| Common | "Arma virumque cano" | No diacritics |

The dataset uses Classical orthography with macrons, which preserves vowel length information important for Latin poetry and pronunciation.

## Data Source

- **Dataset**: [Ken-Z/Latin-Audio](https://huggingface.co/datasets/Ken-Z/Latin-Audio) (Vox Classica)
- **Size**: ~73 hours, 8,400+ samples
- **Sources**: Dickinson College, Nuntii Latini, LibriVox, YouTube educators
- **License**: CC-BY-4.0

## Baseline

Using MMS with Italian adapter (untrained on Latin):
- Expected WER: 60-80%
- Expected CER: 20-40%

After fine-tuning, expect significant improvements similar to the Tironiculum project (4.13% WER).

## License

MIT
