#!/usr/bin/env python3
"""Quick test to verify predictions contain spaces."""
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import json
import os

# Load Latin vocab
with open('data/processed/vocab.json', 'r') as f:
    vocab_data = json.load(f)
    latin_vocab = vocab_data['vocabulary']

# Create reverse mapping
id_to_char = {v: k for k, v in latin_vocab.items()}

# Load model
model_path = 'models/mms-latin-space-fix'
model = Wav2Vec2ForCTC.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained('facebook/mms-1b-all')
model.eval()

# Load test audio files from processed segments
test_files = [
    'data/processed/segments/sample_022966.wav',  # "dē vīrō corōnātō"
    'data/processed/segments/sample_005714.wav',  # "quid est catilīna"
]

test_references = [
    "dē vīrō corōnātō",
    "quid est catilīna",
]

print('Testing predictions with space fix model:')
print('=' * 60)

for idx, audio_path in enumerate(test_files):
    if not os.path.exists(audio_path):
        print(f'File not found: {audio_path}')
        continue

    waveform, sr = librosa.load(audio_path, sr=16000)

    inputs = processor(waveform, sampling_rate=16000, return_tensors='pt')

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)[0]

    # Decode manually using Latin vocab
    pred_chars = []
    prev_id = -1
    for id in predicted_ids.tolist():
        if id != prev_id and id != 0:  # Skip repeats and pad
            char = id_to_char.get(id, '?')
            if char == '|':
                pred_chars.append(' ')
            else:
                pred_chars.append(char)
        prev_id = id

    prediction = ''.join(pred_chars)
    reference = test_references[idx] if idx < len(test_references) else "N/A"
    print(f'File: {os.path.basename(audio_path)}')
    print(f'Reference:  {reference}')
    print(f'Prediction: {prediction}')
    has_spaces = ' ' in prediction
    print(f'Has spaces: {has_spaces}')
    print()
