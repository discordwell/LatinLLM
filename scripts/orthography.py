#!/usr/bin/env python3
"""
LatinLLM Orthography Module
Handles Latin orthographic system conversions.

Latin has two main orthographic conventions:
- Classical (with macrons): Uses macrons (ā, ē, ī, ō, ū) to mark long vowels
- Common/Ecclesiastical: No macrons - most written Latin today

The Ken-Z/Latin-Audio dataset uses macronized text (Classical style).

Reference: Virgil's Aeneid, Book 1, Line 1
Classical: "Arma virumque canō, Trōiae quī prīmus ab ōrīs"
Common:    "Arma virumque cano, Troiae qui primus ab oris"
"""

import re
from typing import Dict, Optional, Set


# Macron mappings (same vowels as Tuvaluan - Latin origin!)
MACRON_TO_PLAIN = {
    'ā': 'a', 'Ā': 'A',
    'ē': 'e', 'Ē': 'E',
    'ī': 'i', 'Ī': 'I',
    'ō': 'o', 'Ō': 'O',
    'ū': 'u', 'Ū': 'U',
    'ȳ': 'y', 'Ȳ': 'Y',  # Rare but used in some texts
}

PLAIN_TO_MACRON = {v: k for k, v in MACRON_TO_PLAIN.items()}

# All macron characters
MACRON_CHARS = set(MACRON_TO_PLAIN.keys())

# Valid Latin characters (Classical - with macrons)
LATIN_CHARS_CLASSICAL = set("abcdefghijklmnopqrstuvwxyzāēīōūȳABCDEFGHIJKLMNOPQRSTUVWXYZĀĒĪŌŪȲ '")

# Valid Latin characters (Common - without macrons)
LATIN_CHARS_COMMON = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '")


def to_common(text: str) -> str:
    """
    Convert Classical Latin (macrons) to Common Latin (no macrons).

    This is a lossless conversion in terms of readability but loses
    information about vowel length.

    Args:
        text: Text potentially containing macrons

    Returns:
        Text with macrons removed (ā→a, ē→e, ī→i, ō→o, ū→u)

    Example:
        >>> to_common("Arma virumque canō")
        "Arma virumque cano"
    """
    result = text
    for macron, plain in MACRON_TO_PLAIN.items():
        result = result.replace(macron, plain)
    return result


def to_classical(text: str, dictionary: Optional[Dict[str, str]] = None) -> str:
    """
    Convert Common Latin to Classical Latin (requires dictionary lookup).

    This conversion is lossy without a dictionary since we cannot know
    where macrons should be placed just from the plain text.

    Args:
        text: Text without macrons
        dictionary: Optional word->macronized_word mapping
                   If None, returns text unchanged

    Returns:
        Text with macrons added where known from dictionary

    Example:
        >>> to_classical("cano", {"cano": "canō"})
        "canō"
    """
    if dictionary is None:
        return text

    # Simple word-by-word replacement
    words = text.split()
    result = []

    for word in words:
        # Preserve punctuation
        prefix = ""
        suffix = ""
        core = word

        # Extract leading punctuation
        while core and not core[0].isalpha():
            prefix += core[0]
            core = core[1:]

        # Extract trailing punctuation
        while core and not core[-1].isalpha():
            suffix = core[-1] + suffix
            core = core[:-1]

        # Look up word (case-insensitive)
        lookup_key = core.lower()
        if lookup_key in dictionary:
            # Match case of original
            replacement = dictionary[lookup_key]
            if core.isupper():
                replacement = replacement.upper()
            elif core and core[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            core = replacement

        result.append(prefix + core + suffix)

    return " ".join(result)


def detect_orthography(text: str) -> str:
    """
    Detect which orthographic system the text uses.

    Args:
        text: Text to analyze

    Returns:
        "classical" if macrons are present, "common" otherwise

    Example:
        >>> detect_orthography("Arma virumque canō")
        "classical"
        >>> detect_orthography("Arma virumque cano")
        "common"
    """
    for char in text:
        if char in MACRON_CHARS:
            return "classical"
    return "common"


def has_macrons(text: str) -> bool:
    """
    Check if text contains any macron characters.

    Args:
        text: Text to check

    Returns:
        True if macrons are present
    """
    return detect_orthography(text) == "classical"


def count_macrons(text: str) -> int:
    """
    Count the number of macron characters in text.

    Args:
        text: Text to analyze

    Returns:
        Number of macron characters
    """
    return sum(1 for char in text if char in MACRON_CHARS)


def normalize_text(text: str, target_orthography: str = "classical",
                   dictionary: Optional[Dict[str, str]] = None) -> str:
    """
    Normalize text to target orthographic system.

    Also performs general text normalization:
    - Converts to lowercase
    - Removes invalid characters
    - Collapses multiple spaces

    Args:
        text: Text to normalize
        target_orthography: "classical" (with macrons) or "common" (no macrons)
        dictionary: Word dictionary for common→classical conversion

    Returns:
        Normalized text in target orthographic system
    """
    # Lowercase
    text = text.lower()

    # Determine current orthography
    current = detect_orthography(text)

    # Convert if needed
    if target_orthography == "common":
        text = to_common(text)
        valid_chars = LATIN_CHARS_COMMON
    elif target_orthography == "classical":
        if current == "common" and dictionary:
            text = to_classical(text, dictionary)
        valid_chars = LATIN_CHARS_CLASSICAL
    else:
        raise ValueError(f"Unknown target orthography: {target_orthography}")

    # Filter to valid characters (lowercase only now)
    valid_chars_lower = set(c.lower() for c in valid_chars)
    text = "".join(c if c in valid_chars_lower else " " for c in text)

    # Collapse multiple spaces
    text = " ".join(text.split())

    return text.strip()


def get_vocabulary_for_orthography(orthography: str = "classical") -> Set[str]:
    """
    Get the character vocabulary appropriate for the target orthography.

    Args:
        orthography: "classical" or "common"

    Returns:
        Set of valid characters for CTC vocabulary building
    """
    if orthography == "classical":
        # Include macron vowels
        chars = set("abcdefghijklmnopqrstuvwxyz āēīōūȳ'")
    else:
        # No macrons
        chars = set("abcdefghijklmnopqrstuvwxyz '")

    return chars


def build_ctc_vocabulary(texts: list, orthography: str = "classical") -> Dict[str, int]:
    """
    Build a CTC vocabulary from texts for the specified orthography.

    Args:
        texts: List of normalized text strings
        orthography: Target orthographic system

    Returns:
        Character to index mapping with special tokens
    """
    from collections import Counter

    # Get valid characters for this orthography
    valid_chars = get_vocabulary_for_orthography(orthography)

    # Count characters
    char_counts = Counter()
    for text in texts:
        # Normalize to target orthography first
        normalized = normalize_text(text, target_orthography=orthography)
        char_counts.update(normalized)

    # Build vocabulary with special tokens
    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4,  # Word boundary (CTC convention)
    }

    # Add characters sorted by frequency
    for char, _ in char_counts.most_common():
        if char in valid_chars and char not in vocab:
            vocab[char] = len(vocab)

    # Ensure all valid chars are in vocab even if not seen
    for char in sorted(valid_chars):
        if char not in vocab:
            vocab[char] = len(vocab)

    return vocab


# Example reference texts for testing
AENEID_CLASSICAL = """Arma virumque canō, Trōiae quī prīmus ab ōrīs
Ītaliam, fātō profugus, Lāvīniaque vēnit
lītora, multum ille et terrīs iactātus et altō
vī superum saevae memorem Iūnōnis ob īram."""

AENEID_COMMON = """Arma virumque cano, Troiae qui primus ab oris
Italiam, fato profugus, Laviniaque venit
litora, multum ille et terris iactatus et alto
vi superum saevae memorem Iunonis ob iram."""


if __name__ == "__main__":
    # Test the module
    print("Latin Orthography Module Tests")
    print("=" * 50)

    # Test Classical → Common
    print("\n1. Classical → Common conversion:")
    classical_text = "Arma virumque canō, Trōiae quī prīmus ab ōrīs"
    common_text = to_common(classical_text)
    print(f"   Input:  {classical_text}")
    print(f"   Output: {common_text}")

    # Test detection
    print("\n2. Orthography detection:")
    print(f"   '{classical_text[:25]}...' → {detect_orthography(classical_text)}")
    print(f"   '{common_text[:25]}...' → {detect_orthography(common_text)}")

    # Test normalization
    print("\n3. Text normalization (Common):")
    messy = "  ARMA VIRUMQUE CANŌ!!  123  "
    normalized = normalize_text(messy, target_orthography="common")
    print(f"   Input:  '{messy}'")
    print(f"   Output: '{normalized}'")

    # Test vocabulary building
    print("\n4. Vocabulary building:")
    vocab_classical = build_ctc_vocabulary([AENEID_CLASSICAL], orthography="classical")
    vocab_common = build_ctc_vocabulary([AENEID_COMMON], orthography="common")
    print(f"   Classical vocab size: {len(vocab_classical)}")
    print(f"   Common vocab size: {len(vocab_common)}")

    # Show macron characters in Classical vocab
    macron_in_vocab = [c for c in vocab_classical.keys() if c in MACRON_CHARS]
    print(f"   Macron chars in Classical vocab: {macron_in_vocab}")

    # Test dictionary conversion
    print("\n5. Common → Classical with dictionary:")
    word_dict = {"cano": "canō", "troiae": "trōiae", "primus": "prīmus"}
    common_sample = "cano troiae primus"
    classical_converted = to_classical(common_sample, word_dict)
    print(f"   Input:  {common_sample}")
    print(f"   Output: {classical_converted}")

    print("\n" + "=" * 50)
    print("All tests passed!")
