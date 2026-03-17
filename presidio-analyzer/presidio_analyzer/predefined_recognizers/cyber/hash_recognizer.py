import math
from typing import Dict, List, Optional, Set

from presidio_analyzer import Pattern, PatternRecognizer

# Common dummy/placeholder hash patterns (hex substrings often used in examples)
KNOWN_DUMMY_PATTERNS: Set[str] = {
    "0" * 8,
    "f" * 8,
    "a" * 8,
    "deadbeef",
    "cafebabe",
    "badc0de",
    "c0ffee",
    "decaf",
    "face",
    "beef",
    "1234",
    "abcd",
    "ffff",
    "0000",
}


class HashRecognizer(PatternRecognizer):
    """
    Recognize MD5, SHA-1, SHA-256, SHA-384, SHA-512 hashes using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    PATTERNS = [
        Pattern("MD5", r"\b[a-fA-F0-9]{32}\b"),
        Pattern("SHA-1", r"\b[a-fA-F0-9]{40}\b"),
        Pattern("SHA-256", r"\b[a-fA-F0-9]{64}\b"),
        Pattern("SHA-384", r"\b[a-fA-F0-9]{96}\b"),
        Pattern("SHA-512", r"\b[a-fA-F0-9]{128}\b"),
    ]

    CONTEXT = ["hash", "md5", "sha1", "sha256", "sha384", "sha512"]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "HASH",
        name: Optional[str] = None,
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
            name=name,
        )

    def validate_result(self, pattern_text: str) -> bool:
        """Validate the hash string using heuristics to filter out dummy values.

        Returns False if the string matches known dummy/placeholder patterns,
        True otherwise. Real cryptographic hashes have high entropy and
        diverse character distribution; dummy values tend to be repetitive
        or use common placeholder hex patterns.

        :param pattern_text: The hash string to validate.
        :return: True if the hash appears to be a real value, False if it
        looks like a dummy/placeholder.
        """
        if not pattern_text or not pattern_text.strip():
            return False

        text = pattern_text.lower().strip()

        # If the hash is all the same character, it is not a real hash
        if len(set(text)) == 1:
            return False

        # If the hash has low entropy, it is not a real hash
        if self._has_low_entropy(text):
            return False

        # If the hash contains known dummy hex patterns, it is not a real hash
        if self._contains_known_dummy_patterns(text):
            return False

        # If the hash is highly repetitive, it is not a real hash
        if self._is_highly_repetitive(text):
            return False

        # If the hash has very few unique characters, it is not a real hash
        if len(set(text)) < 4:
            return False

        return True

    @staticmethod
    def _has_low_entropy(text: str, min_entropy: float = 2.5) -> bool:
        """Check if string has low Shannon entropy (indicator of dummy value)."""
        if not text:
            return True
        freq: Dict[str, int] = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        length = len(text)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        return entropy < min_entropy

    @staticmethod
    def _contains_known_dummy_patterns(text: str) -> bool:
        """Check if string is dominated by known dummy hex patterns."""
        text_lower = text.lower()
        # Count how many chars are part of known dummy patterns
        dummy_chars = 0
        for pattern in KNOWN_DUMMY_PATTERNS:
            count = text_lower.count(pattern)
            dummy_chars += count * len(pattern)
        # If more than 50% of the string is known dummy patterns, reject
        return dummy_chars > len(text) * 0.5

    @staticmethod
    def _is_highly_repetitive(text: str, min_unique_ratio: float = 0.3) -> bool:
        """Check if string is mostly a repeating short pattern."""
        if len(text) < 8:
            return False
        # Try common repetition lengths (2, 4, 8, 16)
        for block_len in [2, 4, 8, 16]:
            if len(text) % block_len != 0:
                continue
            blocks = [text[i : i + block_len] for i in range(0, len(text), block_len)]
            unique_blocks = len(set(blocks))
            if unique_blocks == 1:
                return True  # Entire string is one block repeated
            if unique_blocks / len(blocks) < min_unique_ratio:
                return True  # Very few unique blocks
        return False
