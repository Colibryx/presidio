from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class PortNumberRecognizer(PatternRecognizer):
    """
    Recognize port numbers using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    PATTERNS = [
        # Score 0.2: sotto soglia 0.35, richiede contesto ("port") per boost
        Pattern("PORT_NUMBER", r"\b\d{1,5}\b", 0.2),
    ]

    CONTEXT = ["port", "ports", "port number", "porta", "numero"]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "PORT_NUMBER",
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
