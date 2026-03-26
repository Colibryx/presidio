"""LangExtract helper utilities."""

import logging
from typing import Dict, List, Optional

from presidio_analyzer import AnalysisExplanation, RecognizerResult

logger = logging.getLogger("presidio-analyzer")


def _patch_openai_extra_body():
    """Patch LangExtract OpenAI provider to forward extra_body to the API.

    LangExtract's OpenAI provider only passes a whitelist of params and omits
    extra_body. This patch ensures extra_body (e.g. enable_thinking for Qwen)
    is merged into the API request.
    """
    try:
        from langextract.providers.openai import OpenAILanguageModel

        _original_process = OpenAILanguageModel._process_single_prompt

        def _patched_process_with_extra_body(self, prompt, config):
            extra_body = getattr(self, "_extra_kwargs", {}).get("extra_body")
            original_create = None
            if extra_body is not None:
                original_create = self._client.chat.completions.create

                def create_with_extra_body(**kwargs):
                    merged = dict(kwargs)
                    existing = merged.get("extra_body") or {}
                    merged["extra_body"] = {**existing, **extra_body}
                    return original_create(**merged)

                self._client.chat.completions.create = create_with_extra_body
            try:
                return _original_process(self, prompt, config)
            finally:
                if original_create is not None:
                    self._client.chat.completions.create = original_create

        OpenAILanguageModel._process_single_prompt = _patched_process_with_extra_body
    except Exception as e:
        logger.debug(
            "Could not patch LangExtract OpenAI provider for extra_body: %s", e
        )


try:
    import langextract as lx
    import langextract.factory as lx_factory

    # Make sure builtin providers are pre-loaded to work around a bug in LangExtract
    # that fails to load them automatically if the provider name is specified for a
    # built-in provider rather than inferred from the model_id.
    lx.providers.load_builtins_once()
    lx.providers.load_plugins_once()

    # Patch OpenAI provider to pass extra_body to the API (LangExtract omits it).
    # Required for provider-specific params like Qwen's enable_thinking.
    _patch_openai_extra_body()
except ImportError:
    lx = None
    lx_factory = None

__all__ = [
    "lx",
    "lx_factory",
    "check_langextract_available",
    "extract_lm_config",
    "get_supported_entities",
    "create_reverse_entity_mapping",
    "calculate_extraction_confidence",
    "convert_langextract_to_presidio_results",
    "convert_langextract_batch_to_presidio_results",
    "convert_langextract_batch_to_presidio_results_aligned",
    "LANGEXTRACT_PRESIDIO_BATCH_DOC_ID_PREFIX",
    "presidio_langextract_batch_document_id",
]


def check_langextract_available():
    """Check if langextract is available and raise error if not."""
    if not lx:
        raise ImportError(
            "LangExtract is not installed. "
            "Install it with: poetry install --extras langextract"
        )


# Default alignment score mappings for LangExtract extractions
DEFAULT_ALIGNMENT_SCORES = {
    "MATCH_EXACT": 0.95,
    "MATCH_FUZZY": 0.80,
    "MATCH_LESSER": 0.70,
    "NOT_ALIGNED": 0.60,
}

# Must match ``document_id`` passed to ``langextract.data.Document`` in batch mode
# so outputs can be re-mapped to Presidio input order.
LANGEXTRACT_PRESIDIO_BATCH_DOC_ID_PREFIX = "presidio_batch_"


def presidio_langextract_batch_document_id(index: int) -> str:
    """Stable ``document_id`` for the *i*-th string in a Presidio batch request."""
    return f"{LANGEXTRACT_PRESIDIO_BATCH_DOC_ID_PREFIX}{index}"


def extract_lm_config(config: Dict) -> Dict:
    """Extract LM recognizer configuration section with default values.

    :param config: Full configuration dictionary.
    :return: LM recognizer config with keys: supported_entities, min_score,
             labels_to_ignore, enable_generic_consolidation.
    """
    lm_config_section = config.get("lm_recognizer", {})

    return {
        "supported_entities": lm_config_section.get("supported_entities"),
        "min_score": lm_config_section.get("min_score", 0.5),
        "labels_to_ignore": lm_config_section.get("labels_to_ignore", []),
        "enable_generic_consolidation": lm_config_section.get(
            "enable_generic_consolidation", True
        ),
    }


def get_supported_entities(
    lm_config: Dict, langextract_config: Dict
) -> Optional[List[str]]:
    """Get supported entities list, checking LM config first then LangExtract config.

    :param lm_config: LM recognizer configuration dictionary.
    :param langextract_config: LangExtract configuration dictionary.
    :return: List of supported entity types, or None if not specified.
    """
    return lm_config.get("supported_entities") or langextract_config.get(
        "supported_entities"
    )


def create_reverse_entity_mapping(entity_mappings: Dict) -> Dict:
    """Create reverse mapping from values to keys.

    :param entity_mappings: Original entity mapping dictionary.
    :return: Reversed dictionary mapping values to keys.
    """
    return {v: k for k, v in entity_mappings.items()}


def calculate_extraction_confidence(
    extraction, alignment_scores: Optional[Dict[str, float]] = None
) -> float:
    """Calculate confidence score based on extraction alignment status.

    :param extraction: LangExtract extraction object with optional alignment_status.
    :param alignment_scores: Custom score mapping for alignment statuses (optional).
    :return: Confidence score between 0.0 and 1.0.
    """
    default_score = 0.85

    if alignment_scores is None:
        alignment_scores = DEFAULT_ALIGNMENT_SCORES

    if not hasattr(extraction, "alignment_status") or not (extraction.alignment_status):
        return default_score

    status = str(extraction.alignment_status).upper()
    for status_key, score in alignment_scores.items():
        if status_key in status:
            return score

    return default_score


def _document_results_from_langextract_batch(langextract_batch_result) -> List:
    """Normalize LangExtract batch output to a list of per-document result objects.

    LangExtract may return a list of annotated documents or a wrapper with a
    ``documents`` / ``annotated_documents`` / ``results`` attribute.
    """
    if langextract_batch_result is None:
        return []
    if isinstance(langextract_batch_result, (list, tuple)):
        return list(langextract_batch_result)
    for attr in ("documents", "annotated_documents", "results"):
        if hasattr(langextract_batch_result, attr):
            docs = getattr(langextract_batch_result, attr)
            if docs is not None:
                return list(docs)
    # Single-document shape: one result object for the whole batch call
    return [langextract_batch_result]


def convert_langextract_batch_to_presidio_results(
    langextract_batch_result,
    entity_mappings: Dict,
    supported_entities: List[str],
    enable_generic_consolidation: bool,
    recognizer_name: str,
    alignment_scores: Optional[Dict[str, float]] = None,
) -> List[List[RecognizerResult]]:
    """Convert LangExtract multi-document extraction output to Presidio results.

    Each item in the returned list corresponds to one input document, in order.

    :param langextract_batch_result: Output from ``lx.extract`` when multiple
        documents were passed via ``text_or_documents``.
    :return: List of ``RecognizerResult`` lists, one per document.
    """
    doc_results = _document_results_from_langextract_batch(langextract_batch_result)
    return [
        convert_langextract_to_presidio_results(
            doc,
            entity_mappings=entity_mappings,
            supported_entities=supported_entities,
            enable_generic_consolidation=enable_generic_consolidation,
            recognizer_name=recognizer_name,
            alignment_scores=alignment_scores,
        )
        for doc in doc_results
    ]


def convert_langextract_batch_to_presidio_results_aligned(
    langextract_batch_result,
    num_input_documents: int,
    entity_mappings: Dict,
    supported_entities: List[str],
    enable_generic_consolidation: bool,
    recognizer_name: str,
    alignment_scores: Optional[Dict[str, float]] = None,
    batch_document_id_prefix: str = LANGEXTRACT_PRESIDIO_BATCH_DOC_ID_PREFIX,
) -> List[List[RecognizerResult]]:
    """Convert LangExtract multi-document output to Presidio results per input text.

    When each returned annotated document has a unique ``document_id`` matching
    ``{batch_document_id_prefix}{i}`` (as produced by
    :func:`presidio_langextract_batch_document_id`), results are ordered by *i*
    even if LangExtract returns documents in a different order. Otherwise
    falls back to positional alignment of the normalized document list (same
    behavior as :func:`convert_langextract_batch_to_presidio_results` plus
    padding/truncation).

    :param num_input_documents: Number of input texts Presidio sent in the batch.
    :return: ``num_input_documents`` lists of :class:`~presidio_analyzer.RecognizerResult`.
    """
    expected_ids = [
        f"{batch_document_id_prefix}{i}" for i in range(num_input_documents)
    ]

    def _pad_to_n(rows: List[List[RecognizerResult]]) -> List[List[RecognizerResult]]:
        while len(rows) < num_input_documents:
            rows.append([])
        return rows[:num_input_documents]

    def _positional() -> List[List[RecognizerResult]]:
        return _pad_to_n(
            convert_langextract_batch_to_presidio_results(
                langextract_batch_result=langextract_batch_result,
                entity_mappings=entity_mappings,
                supported_entities=supported_entities,
                enable_generic_consolidation=enable_generic_consolidation,
                recognizer_name=recognizer_name,
                alignment_scores=alignment_scores,
            )
        )

    doc_results = _document_results_from_langextract_batch(langextract_batch_result)

    if not doc_results:
        return [[] for _ in range(num_input_documents)]

    by_id: Dict[str, List[RecognizerResult]] = {}
    for doc in doc_results:
        rid = getattr(doc, "document_id", None)
        if rid is None:
            logger.debug(
                "LangExtract batch output missing document_id on a document; "
                "using positional alignment to Presidio input order"
            )
            return _positional()
        if rid in by_id:
            logger.warning(
                "Duplicate document_id %r in LangExtract batch output; "
                "using positional alignment",
                rid,
            )
            return _positional()
        by_id[rid] = convert_langextract_to_presidio_results(
            langextract_result=doc,
            entity_mappings=entity_mappings,
            supported_entities=supported_entities,
            enable_generic_consolidation=enable_generic_consolidation,
            recognizer_name=recognizer_name,
            alignment_scores=alignment_scores,
        )

    unknown = set(by_id.keys()) - set(expected_ids)
    if unknown:
        logger.debug(
            "LangExtract returned document_id(s) not in Presidio batch ids: %s",
            unknown,
        )

    return [by_id.get(eid, []) for eid in expected_ids]


def convert_langextract_to_presidio_results(
    langextract_result,
    entity_mappings: Dict,
    supported_entities: List[str],
    enable_generic_consolidation: bool,
    recognizer_name: str,
    alignment_scores: Optional[Dict[str, float]] = None,
) -> List[RecognizerResult]:
    """Convert LangExtract extraction results to Presidio RecognizerResult objects.

    :param langextract_result: LangExtract result object with extractions.
    :param entity_mappings: Mapping of extraction classes to Presidio entity types.
    :param supported_entities: List of supported Presidio entity types.
    :param enable_generic_consolidation: Whether to consolidate unknown entities.
    :param recognizer_name: Name of recognizer for result metadata.
    :param alignment_scores: Custom alignment score mappings (optional).
    :return: List of Presidio RecognizerResult objects.
    """
    results = []
    if not langextract_result or not langextract_result.extractions:
        return results

    supported_entities_set = set(supported_entities)

    for extraction in langextract_result.extractions:
        extraction_class = extraction.extraction_class

        if extraction_class in supported_entities_set:
            entity_type = extraction_class
        else:
            extraction_class_lower = extraction_class.lower()
            entity_type = entity_mappings.get(extraction_class_lower)

        if not entity_type:
            if enable_generic_consolidation:
                entity_type = extraction_class.upper()
                logger.debug(
                    "Unknown extraction class '%s' will be consolidated to "
                    "GENERIC_PII_ENTITY",
                    extraction_class,
                )
            else:
                logger.warning(
                    "Unknown extraction class '%s' not found in entity "
                    "mappings, skipping",
                    extraction_class,
                )
                continue

        if not extraction.char_interval:
            logger.warning("Extraction missing char_interval, skipping")
            continue

        confidence = calculate_extraction_confidence(extraction, alignment_scores)

        metadata = {}
        if hasattr(extraction, "attributes") and extraction.attributes:
            metadata["attributes"] = extraction.attributes
        if hasattr(extraction, "alignment_status") and extraction.alignment_status:
            metadata["alignment"] = str(extraction.alignment_status)

        explanation = AnalysisExplanation(
            recognizer=recognizer_name,
            original_score=confidence,
            textual_explanation=(
                f"LangExtract extraction with {extraction.alignment_status} alignment"
                if hasattr(extraction, "alignment_status")
                and extraction.alignment_status
                else "LangExtract extraction"
            ),
        )

        result = RecognizerResult(
            entity_type=entity_type,
            start=extraction.char_interval.start_pos,
            end=extraction.char_interval.end_pos,
            score=confidence,
            analysis_explanation=explanation,
            recognition_metadata=metadata if metadata else None,
        )

        results.append(result)

    return results
