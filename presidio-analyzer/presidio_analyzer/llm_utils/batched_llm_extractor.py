"""Batched LLM extractor: one LLM call for N input texts.

This module bypasses LangExtract's per-document LLM call loop to perform
true batching: it issues a SINGLE OpenAI/Azure OpenAI Chat Completions
request that asks the model to analyze N texts at once and return a
structured JSON response. The returned per-text extractions are then
aligned to the source texts via LangExtract's standalone
``Resolver.align()`` so that character offsets are recovered correctly.

The result of :meth:`BatchedLLMExtractor.extract_batch` is a list of lists
of ``langextract.data.Extraction`` objects (one inner list per input text,
in the same order as the input). This is the same shape produced by the
existing per-document LangExtract conversion path, so callers can reuse
:func:`presidio_analyzer.llm_utils.convert_langextract_to_presidio_results`
without modification.

Errors that are recoverable through fallback (e.g. invalid JSON, count
mismatch, alignment failures) raise :class:`BatchedExtractionError` so the
caller can decide whether to fall back to the legacy LangExtract path.
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from .langextract_helper import check_langextract_available, lx
from .prompt_loader import render_jinja_template

logger = logging.getLogger("presidio_analyzer.batched_llm_extractor")

__all__ = [
    "BatchedExtractionError",
    "BatchedLLMExtractor",
    "BATCH_EXTRACTION_SCHEMA",
]


# JSON schema sent to the OpenAI Chat Completions API via
# ``response_format={"type": "json_schema", "json_schema": ...}``. The
# ``strict: true`` flag forces compliant models (gpt-4o family, Qwen3 served
# via vLLM/SGLang with structured output enabled, etc.) to produce output
# that matches this schema exactly.
BATCH_EXTRACTION_SCHEMA: Dict[str, Any] = {
    "name": "presidio_batch_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text_id", "extractions"],
                    "properties": {
                        "text_id": {"type": "integer"},
                        "extractions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["class", "text"],
                                "properties": {
                                    "class": {"type": "string"},
                                    "text": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


class BatchedExtractionError(Exception):
    """Recoverable error raised by :class:`BatchedLLMExtractor`.

    The ``stage`` attribute identifies which step failed so the caller can
    log it and decide whether to fall back to the legacy per-text path.
    Stages used by this module:

    - ``llm_call`` — the OpenAI SDK raised (timeout, 5xx, auth, rate-limit).
    - ``json_parse`` — the model output is not valid JSON.
    - ``schema_validation`` — JSON does not contain the expected fields.
    - ``count_mismatch`` — number of returned ``results`` ≠ number of inputs.
    - ``alignment`` — every extraction for a given text failed to align,
      meaning the model fabricated substrings that are not in the source.
    """

    def __init__(
        self,
        reason: str,
        *,
        stage: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.reason = reason
        self.stage = stage
        self.details = details or {}
        super().__init__(f"[{stage}] {reason}")


class BatchedLLMExtractor:
    """Single-call batched LLM extractor with offset recovery via LangExtract.

    The extractor is constructed once per recognizer and cached. Each call
    to :meth:`extract_batch` issues a single Chat Completions request for
    all input texts (splitting into chunks of ``max_batch_size`` if needed)
    and returns aligned ``Extraction`` lists ready for conversion into
    Presidio ``RecognizerResult`` objects.
    """

    def __init__(
        self,
        openai_client: Any,
        model_id: str,
        base_prompt_rendered: str,
        wrapper_template_str: str,
        examples: Sequence[Any],
        max_batch_size: int = 20,
        extra_llm_params: Optional[Dict[str, Any]] = None,
        request_logger: Optional[logging.Logger] = None,
    ):
        """Construct a batched extractor.

        :param openai_client: An ``openai.OpenAI`` or ``openai.AzureOpenAI``
            instance. The caller is responsible for any auth wiring; this
            class only invokes ``client.chat.completions.create``.
        :param model_id: Model or deployment identifier passed as ``model``
            to the Chat Completions API.
        :param base_prompt_rendered: The recognizer's base prompt already
            rendered with entity lists, cybersecurity guidance, etc.
            Embedded into the batch wrapper template as ``base_prompt``.
        :param wrapper_template_str: Raw Jinja2 template string for the
            batch wrapper (typically loaded from
            ``batch_extraction_wrapper.j2``).
        :param examples: Sequence of ``langextract.data.ExampleData`` (or
            equivalent objects with ``text`` and ``extractions`` attributes)
            used as few-shot examples in the rendered prompt.
        :param max_batch_size: Maximum number of texts to send in a single
            LLM call. Larger batches are split into chunks processed
            sequentially.
        :param extra_llm_params: Additional kwargs forwarded to
            ``client.chat.completions.create`` (e.g. ``temperature``,
            ``max_tokens``, ``extra_body``).
        :param request_logger: Optional override for the module logger.
        """
        check_langextract_available()
        self._client = openai_client
        self._model_id = model_id
        self._base_prompt_rendered = base_prompt_rendered
        self._wrapper_template_str = wrapper_template_str
        self._examples = list(examples) if examples else []
        self._max_batch_size = max(1, int(max_batch_size))
        self._extra_llm_params = dict(extra_llm_params or {})
        self._logger = request_logger or logger

        # The wrapper rendering does not depend on the input texts, so we
        # cache it after first use to avoid re-rendering on every call.
        self._cached_batch_prompt: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract_batch(
        self,
        texts: Sequence[str],
        requested_entities: Optional[Sequence[str]] = None,
    ) -> List[List[Any]]:
        """Run a batched extraction on ``texts``.

        :param texts: Input texts in the order Presidio passed them.
        :param requested_entities: Currently informational only; the
            entity list is already baked into ``base_prompt_rendered``.
            Accepted to keep the call site uniform.
        :return: A list of length ``len(texts)``. Each element is a list of
            ``lx.data.Extraction`` objects with ``char_interval`` populated
            (offsets relative to the corresponding input text).
        :raises BatchedExtractionError: On any recoverable failure. The
            caller is expected to log this and fall back.
        """
        del requested_entities  # informational, not enforced here
        if not texts:
            return []

        # Chunked path: split into batches of max_batch_size and concatenate
        # results in input order. Errors in any chunk propagate and abort
        # the whole batch (the caller falls back to the legacy path for the
        # full list, preserving simple semantics).
        if len(texts) > self._max_batch_size:
            aggregated: List[List[Any]] = []
            for chunk_start in range(0, len(texts), self._max_batch_size):
                chunk = texts[chunk_start : chunk_start + self._max_batch_size]
                aggregated.extend(self._extract_single_chunk(chunk))
            return aggregated

        return self._extract_single_chunk(texts)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_or_render_batch_prompt(self) -> str:
        """Render the batch wrapper prompt once and cache it."""
        if self._cached_batch_prompt is None:
            self._cached_batch_prompt = render_jinja_template(
                self._wrapper_template_str,
                base_prompt=self._base_prompt_rendered,
                examples=self._examples,
            )
        return self._cached_batch_prompt

    def _build_user_message(self, texts: Sequence[str]) -> str:
        """Build the user message JSON payload describing the batch input."""
        return json.dumps(
            {
                "texts": [
                    {"id": i, "content": t} for i, t in enumerate(texts)
                ]
            },
            ensure_ascii=False,
        )

    def _call_llm(self, texts: Sequence[str]) -> str:
        """Issue the Chat Completions request and return the raw content."""
        system_prompt = self._get_or_render_batch_prompt()
        user_message = self._build_user_message(texts)

        request_kwargs: Dict[str, Any] = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": BATCH_EXTRACTION_SCHEMA,
            },
        }
        request_kwargs.update(self._extra_llm_params)

        try:
            response = self._client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            raise BatchedExtractionError(
                f"OpenAI chat.completions.create failed: {exc}",
                stage="llm_call",
                details={
                    "model": self._model_id,
                    "batch_size": len(texts),
                    "exception_type": type(exc).__name__,
                },
            ) from exc

        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            raise BatchedExtractionError(
                f"Unexpected response shape from LLM: {exc}",
                stage="schema_validation",
                details={"model": self._model_id, "batch_size": len(texts)},
            ) from exc

    def _parse_and_validate(
        self, raw_content: str, expected_count: int
    ) -> List[Dict[str, Any]]:
        """Parse the JSON content and validate it has one entry per text."""
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise BatchedExtractionError(
                f"Model output is not valid JSON: {exc}",
                stage="json_parse",
                details={"raw_snippet": raw_content[:500]},
            ) from exc

        if not isinstance(payload, dict) or "results" not in payload:
            raise BatchedExtractionError(
                "Model output missing required 'results' key",
                stage="schema_validation",
                details={"raw_snippet": raw_content[:500]},
            )

        results = payload["results"]
        if not isinstance(results, list):
            raise BatchedExtractionError(
                "'results' must be a list",
                stage="schema_validation",
                details={"got_type": type(results).__name__},
            )

        if len(results) != expected_count:
            raise BatchedExtractionError(
                f"Expected {expected_count} results, got {len(results)}",
                stage="count_mismatch",
                details={
                    "expected": expected_count,
                    "got": len(results),
                    "text_ids": [
                        r.get("text_id") if isinstance(r, dict) else None
                        for r in results
                    ],
                },
            )

        # Reorder by text_id so callers always see results aligned to input
        # order, regardless of the order the model emitted them.
        ordered: List[Optional[Dict[str, Any]]] = [None] * expected_count
        seen_ids = set()
        for entry in results:
            if not isinstance(entry, dict):
                raise BatchedExtractionError(
                    "Each result must be an object",
                    stage="schema_validation",
                    details={"got_type": type(entry).__name__},
                )
            tid = entry.get("text_id")
            if not isinstance(tid, int) or tid < 0 or tid >= expected_count:
                raise BatchedExtractionError(
                    f"Invalid text_id: {tid!r}",
                    stage="schema_validation",
                    details={"text_id": tid, "expected_range": expected_count},
                )
            if tid in seen_ids:
                raise BatchedExtractionError(
                    f"Duplicate text_id: {tid}",
                    stage="count_mismatch",
                    details={"text_id": tid},
                )
            seen_ids.add(tid)
            ordered[tid] = entry

        if any(entry is None for entry in ordered):
            missing = [i for i, e in enumerate(ordered) if e is None]
            raise BatchedExtractionError(
                f"Missing text_ids in response: {missing}",
                stage="count_mismatch",
                details={"missing_ids": missing},
            )

        return [entry for entry in ordered if entry is not None]

    def _align_extractions_for_text(
        self,
        text_index: int,
        source_text: str,
        raw_extractions: Sequence[Dict[str, Any]],
    ) -> List[Any]:
        """Build ``lx.data.Extraction`` objects and run ``Resolver.align()``."""
        if not raw_extractions:
            return []

        extraction_objs: List[Any] = []
        for ext in raw_extractions:
            if not isinstance(ext, dict):
                self._logger.debug(
                    "Skipping non-dict extraction in text_id=%d: %r",
                    text_index,
                    ext,
                )
                continue
            cls = ext.get("class")
            txt = ext.get("text")
            if not isinstance(cls, str) or not isinstance(txt, str) or not txt:
                self._logger.debug(
                    "Skipping malformed extraction in text_id=%d: %r",
                    text_index,
                    ext,
                )
                continue
            extraction_objs.append(
                lx.data.Extraction(
                    extraction_class=cls,
                    extraction_text=txt,
                )
            )

        if not extraction_objs:
            return []

        # Import lazily so the module remains importable when langextract is
        # not installed (the constructor's check_langextract_available() will
        # have already raised in that case).
        from langextract.resolver import Resolver

        aligned = list(
            Resolver().align(
                extractions=extraction_objs,
                source_text=source_text,
                token_offset=0,
                char_offset=0,
            )
        )

        # Filter extractions that the aligner could not place anywhere in
        # the source text. ``alignment_status is None`` and ``char_interval
        # is None`` both indicate a hallucinated substring.
        kept: List[Any] = []
        dropped: List[str] = []
        for ext_obj in aligned:
            if (
                getattr(ext_obj, "char_interval", None) is None
                or getattr(ext_obj, "alignment_status", None) is None
            ):
                dropped.append(ext_obj.extraction_text)
                continue
            kept.append(ext_obj)

        if dropped:
            self._logger.debug(
                "text_id=%d: dropped %d hallucinated extractions: %s",
                text_index,
                len(dropped),
                dropped,
            )

        # If the model returned at least one extraction but ALL of them
        # failed to align, treat it as a hallucination signal and let the
        # caller fall back to the legacy path.
        if not kept and extraction_objs:
            raise BatchedExtractionError(
                f"All {len(extraction_objs)} extractions for text_id={text_index} "
                f"failed to align to source text",
                stage="alignment",
                details={
                    "text_index": text_index,
                    "extracted": [e.extraction_text for e in extraction_objs],
                },
            )

        return kept

    def _extract_single_chunk(
        self, texts: Sequence[str]
    ) -> List[List[Any]]:
        """Run one Chat Completions call for a single chunk of ``texts``."""
        start_time = time.monotonic()
        raw_content = self._call_llm(texts)
        results = self._parse_and_validate(raw_content, len(texts))

        aligned_per_text: List[List[Any]] = []
        total_extractions = 0
        for i, (text, entry) in enumerate(zip(texts, results)):
            aligned = self._align_extractions_for_text(
                text_index=i,
                source_text=text,
                raw_extractions=entry.get("extractions", []),
            )
            aligned_per_text.append(aligned)
            total_extractions += len(aligned)

        duration_ms = int((time.monotonic() - start_time) * 1000)
        self._logger.info(
            "Batched extraction OK: model=%s batch_size=%d "
            "total_extractions=%d duration_ms=%d",
            self._model_id,
            len(texts),
            total_extractions,
            duration_ms,
        )
        return aligned_per_text


def make_extraction_document(extractions: Sequence[Any]) -> Any:
    """Wrap a list of Extractions so they can be passed to the existing
    ``convert_langextract_to_presidio_results`` helper without changes.

    The helper only reads ``.extractions`` from its input, so a simple
    namespace object is enough.
    """
    return SimpleNamespace(extractions=list(extractions))
