"""
LatinLLM Test Configuration
Pytest fixtures for test suite.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np

# Add scripts directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def sample_classical_text() -> str:
    """Sample text in Classical Latin (with macrons)."""
    return "Arma virumque canō, Trōiae quī prīmus ab ōrīs"


@pytest.fixture
def sample_common_text() -> str:
    """Sample text in Common Latin (without macrons)."""
    return "Arma virumque cano, Troiae qui primus ab oris"


@pytest.fixture
def aeneid_classical() -> str:
    """First lines of Virgil's Aeneid in Classical orthography."""
    return """Arma virumque canō, Trōiae quī prīmus ab ōrīs
Ītaliam, fātō profugus, Lāvīniaque vēnit
lītora, multum ille et terrīs iactātus et altō
vī superum saevae memorem Iūnōnis ob īram."""


@pytest.fixture
def aeneid_common() -> str:
    """First lines of Virgil's Aeneid in Common orthography."""
    return """Arma virumque cano, Troiae qui primus ab oris
Italiam, fato profugus, Laviniaque venit
litora, multum ille et terris iactatus et alto
vi superum saevae memorem Iunonis ob iram."""


@pytest.fixture
def word_dictionary() -> Dict[str, str]:
    """Sample word dictionary for Common to Classical conversion."""
    return {
        "cano": "canō",
        "troiae": "trōiae",
        "primus": "prīmus",
        "oris": "ōrīs",
        "italiam": "ītaliam",
        "fato": "fātō",
        "venit": "vēnit"
    }


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # Generate 1 second of silence at 16kHz
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio = np.zeros(samples, dtype=np.float32)

        # Add some noise
        audio += np.random.randn(samples).astype(np.float32) * 0.01

        sf.write(tmp.name, audio, sample_rate)
        yield tmp.name

    os.unlink(tmp.name)


@pytest.fixture
def sample_dataset_item() -> Dict[str, Any]:
    """Sample dataset item structure."""
    return {
        "audio": "/path/to/audio.wav",
        "sentence": "arma virumque canō",
        "original_text": "Arma virumque canō",
        "duration": 2.5,
        "orthography": "classical"
    }
