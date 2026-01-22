"""
Tests for the Latin orthography module.
Tests Classical (macrons) and Common (no macrons) conversions.
"""

import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from orthography import (
    to_common,
    to_classical,
    detect_orthography,
    has_macrons,
    count_macrons,
    normalize_text,
    get_vocabulary_for_orthography,
    build_ctc_vocabulary,
    MACRON_TO_PLAIN,
    MACRON_CHARS
)


class TestToCommon:
    """Tests for Classical to Common conversion."""

    def test_basic_conversion(self, sample_classical_text, sample_common_text):
        """Test basic macron removal."""
        result = to_common(sample_classical_text)
        assert "ō" not in result
        assert "ī" not in result

    def test_preserves_non_macron_text(self, sample_common_text):
        """Text without macrons should be unchanged."""
        result = to_common(sample_common_text)
        assert result == sample_common_text

    def test_all_macron_characters(self):
        """Test all macron conversions."""
        text = "ā ē ī ō ū ȳ Ā Ē Ī Ō Ū Ȳ"
        result = to_common(text)
        assert result == "a e i o u y A E I O U Y"

    def test_mixed_text(self):
        """Test text with macrons mixed with regular characters."""
        text = "canō and cano"
        result = to_common(text)
        assert result == "cano and cano"


class TestToClassical:
    """Tests for Common to Classical conversion (with dictionary)."""

    def test_without_dictionary(self):
        """Without dictionary, text should be unchanged."""
        text = "cano primus"
        result = to_classical(text)
        assert result == text

    def test_with_dictionary(self, word_dictionary):
        """With dictionary, known words should be converted."""
        text = "cano primus"
        result = to_classical(text, word_dictionary)
        assert "canō" in result
        assert "prīmus" in result

    def test_preserves_unknown_words(self, word_dictionary):
        """Unknown words should be preserved."""
        text = "cano unknown primus"
        result = to_classical(text, word_dictionary)
        assert "unknown" in result

    def test_preserves_punctuation(self, word_dictionary):
        """Punctuation should be preserved."""
        text = "cano, primus!"
        result = to_classical(text, word_dictionary)
        assert "," in result
        assert "!" in result


class TestDetectOrthography:
    """Tests for orthographic system detection."""

    def test_detects_classical(self, sample_classical_text):
        """Text with macrons should be detected as classical."""
        assert detect_orthography(sample_classical_text) == "classical"

    def test_detects_common(self, sample_common_text):
        """Text without macrons should be detected as common."""
        assert detect_orthography(sample_common_text) == "common"

    def test_empty_text(self):
        """Empty text should be common (no macrons)."""
        assert detect_orthography("") == "common"

    def test_single_macron(self):
        """Even a single macron should trigger classical."""
        assert detect_orthography("hello ā world") == "classical"


class TestHasMacrons:
    """Tests for has_macrons function."""

    def test_has_macrons_true(self):
        """Should return True for text with macrons."""
        assert has_macrons("canō") is True

    def test_has_macrons_false(self):
        """Should return False for text without macrons."""
        assert has_macrons("cano") is False


class TestCountMacrons:
    """Tests for count_macrons function."""

    def test_count_zero(self):
        """Text without macrons should return 0."""
        assert count_macrons("cano") == 0

    def test_count_multiple(self):
        """Should count all macron characters."""
        assert count_macrons("canō Trōiae") == 2  # ō in canō, ō in Trōiae

    def test_aeneid(self, aeneid_classical):
        """Should count macrons in longer text."""
        count = count_macrons(aeneid_classical)
        assert count > 10


class TestNormalizeText:
    """Tests for text normalization."""

    def test_lowercase(self):
        """Should convert to lowercase."""
        result = normalize_text("ARMA VIRUMQUE")
        assert result == "arma virumque"

    def test_remove_invalid_chars(self):
        """Should remove non-Latin characters."""
        result = normalize_text("arma! 123 virumque@#$")
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result

    def test_collapse_spaces(self):
        """Should collapse multiple spaces."""
        result = normalize_text("arma    virumque")
        assert "  " not in result

    def test_target_classical(self):
        """Should preserve macrons for classical target."""
        result = normalize_text("CANŌ", target_orthography="classical")
        assert "ō" in result

    def test_target_common(self):
        """Should remove macrons for common target."""
        result = normalize_text("canō", target_orthography="common")
        assert "ō" not in result
        assert "o" in result


class TestGetVocabularyForOrthography:
    """Tests for vocabulary character sets."""

    def test_classical_includes_macrons(self):
        """Classical vocab should include macron characters."""
        vocab = get_vocabulary_for_orthography("classical")
        assert "ā" in vocab
        assert "ē" in vocab
        assert "ī" in vocab
        assert "ō" in vocab
        assert "ū" in vocab

    def test_common_excludes_macrons(self):
        """Common vocab should not include macron characters."""
        vocab = get_vocabulary_for_orthography("common")
        assert "ā" not in vocab
        assert "ō" not in vocab

    def test_both_include_basic_chars(self):
        """Both systems should include basic Latin characters."""
        for system in ["classical", "common"]:
            vocab = get_vocabulary_for_orthography(system)
            assert "a" in vocab
            assert "z" in vocab
            assert " " in vocab


class TestBuildCTCVocabulary:
    """Tests for CTC vocabulary building."""

    def test_includes_special_tokens(self):
        """Should include CTC special tokens."""
        vocab = build_ctc_vocabulary(["arma"], orthography="classical")
        assert "<pad>" in vocab
        assert "<s>" in vocab
        assert "</s>" in vocab
        assert "<unk>" in vocab
        assert "|" in vocab

    def test_special_tokens_indices(self):
        """Special tokens should have correct indices."""
        vocab = build_ctc_vocabulary(["arma"], orthography="classical")
        assert vocab["<pad>"] == 0
        assert vocab["<s>"] == 1
        assert vocab["</s>"] == 2
        assert vocab["<unk>"] == 3
        assert vocab["|"] == 4

    def test_classical_vocab_size(self, aeneid_classical):
        """Classical vocab should be larger due to macrons."""
        vocab_c = build_ctc_vocabulary([aeneid_classical], orthography="classical")
        vocab_p = build_ctc_vocabulary([aeneid_classical], orthography="common")
        assert len(vocab_c) >= len(vocab_p)


class TestRoundTrip:
    """Tests for round-trip conversions."""

    def test_common_is_stable(self, sample_common_text):
        """Converting common→common should be stable."""
        result = to_common(sample_common_text)
        assert result == sample_common_text
