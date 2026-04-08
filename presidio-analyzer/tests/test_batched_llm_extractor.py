"""Unit tests for BatchedLLMExtractor.

These tests mock the OpenAI client so no real network calls happen, and
exercise the full pipeline: prompt rendering, JSON parsing, schema and
count validation, text_id reordering, alignment via langextract's
``Resolver.align()``, hallucination filtering, and chunk splitting.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from presidio_analyzer.llm_utils.batched_llm_extractor import (
    BATCH_EXTRACTION_SCHEMA,
    BatchedExtractionError,
    BatchedLLMExtractor,
)


# Minimal Jinja wrapper template that lets us assert both the embedded
# base prompt and a recognizable batch marker appear in the rendered output.
WRAPPER_TEMPLATE = (
    "{{ base_prompt }}\n\n"
    "BATCH MODE EXAMPLES:\n"
    "{% for ex in examples %}- {{ ex.text }}\n{% endfor %}"
)

BASE_PROMPT = "You are a PII detector. Entities: PERSON, EMAIL_ADDRESS."


def _make_mock_chat_response(content: str) -> MagicMock:
    """Build a mock OpenAI Chat Completions response with given content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    return response


def _make_mock_client(content: str) -> MagicMock:
    """Build a mock OpenAI client whose chat.completions.create returns content."""
    client = MagicMock()
    client.chat.completions.create.return_value = _make_mock_chat_response(content)
    return client


def _make_extractor(client, max_batch_size: int = 20) -> BatchedLLMExtractor:
    return BatchedLLMExtractor(
        openai_client=client,
        model_id="test-model",
        base_prompt_rendered=BASE_PROMPT,
        wrapper_template_str=WRAPPER_TEMPLATE,
        examples=[],
        max_batch_size=max_batch_size,
    )


class TestExtractBatchHappyPath:
    """End-to-end success path: 3 texts, 3 results, correct alignment."""

    def test_three_texts_aligned_with_correct_offsets(self):
        texts = [
            "Contact John Doe at john@example.com",
            "Call 555-1234",
            "No PII here",
        ]
        llm_response = json.dumps(
            {
                "results": [
                    {
                        "text_id": 0,
                        "extractions": [
                            {"class": "person", "text": "John Doe"},
                            {"class": "email", "text": "john@example.com"},
                        ],
                    },
                    {"text_id": 1, "extractions": []},
                    {"text_id": 2, "extractions": []},
                ]
            }
        )
        client = _make_mock_client(llm_response)
        extractor = _make_extractor(client)

        result = extractor.extract_batch(texts)

        assert len(result) == 3
        # First text: 2 extractions, char intervals populated by Resolver
        assert len(result[0]) == 2
        person, email = result[0]
        assert person.extraction_class == "person"
        assert person.extraction_text == "John Doe"
        assert person.char_interval is not None
        assert person.char_interval.start_pos == texts[0].index("John Doe")
        assert person.char_interval.end_pos == person.char_interval.start_pos + len(
            "John Doe"
        )
        assert email.char_interval is not None
        assert email.char_interval.start_pos == texts[0].index("john@example.com")
        # Second and third texts: empty
        assert result[1] == []
        assert result[2] == []

    def test_chat_completions_called_once_with_json_schema(self):
        texts = ["Hello world"]
        llm_response = json.dumps(
            {"results": [{"text_id": 0, "extractions": []}]}
        )
        client = _make_mock_client(llm_response)
        extractor = _make_extractor(client)

        extractor.extract_batch(texts)

        assert client.chat.completions.create.call_count == 1
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert (
            call_kwargs["response_format"]["json_schema"]
            is BATCH_EXTRACTION_SCHEMA
        )
        # System message must include the base prompt and the wrapper marker
        system_msg = call_kwargs["messages"][0]["content"]
        assert "BATCH MODE EXAMPLES" in system_msg
        assert BASE_PROMPT in system_msg
        # User message must encode the input texts as JSON
        user_payload = json.loads(call_kwargs["messages"][1]["content"])
        assert user_payload == {"texts": [{"id": 0, "content": "Hello world"}]}


class TestReorderingAndCountValidation:
    """Reorder by text_id and validate count/missing/duplicate ids."""

    def test_reorders_results_by_text_id(self):
        texts = ["text zero", "text one", "text two"]
        # LLM returns out-of-order results
        llm_response = json.dumps(
            {
                "results": [
                    {"text_id": 2, "extractions": []},
                    {"text_id": 0, "extractions": []},
                    {"text_id": 1, "extractions": []},
                ]
            }
        )
        client = _make_mock_client(llm_response)
        extractor = _make_extractor(client)

        result = extractor.extract_batch(texts)

        assert len(result) == 3
        # Empty per text, but reordering must not raise
        assert all(r == [] for r in result)

    def test_count_mismatch_raises(self):
        texts = ["a", "b", "c"]
        llm_response = json.dumps(
            {
                "results": [
                    {"text_id": 0, "extractions": []},
                    {"text_id": 1, "extractions": []},
                    # Missing text_id 2
                ]
            }
        )
        client = _make_mock_client(llm_response)
        extractor = _make_extractor(client)

        with pytest.raises(BatchedExtractionError) as exc_info:
            extractor.extract_batch(texts)
        assert exc_info.value.stage == "count_mismatch"

    def test_duplicate_text_id_raises(self):
        texts = ["a", "b"]
        llm_response = json.dumps(
            {
                "results": [
                    {"text_id": 0, "extractions": []},
                    {"text_id": 0, "extractions": []},
                ]
            }
        )
        client = _make_mock_client(llm_response)
        extractor = _make_extractor(client)

        with pytest.raises(BatchedExtractionError) as exc_info:
            extractor.extract_batch(texts)
        assert exc_info.value.stage == "count_mismatch"

    def test_invalid_text_id_raises(self):
        texts = ["a", "b"]
        llm_response = json.dumps(
            {
                "results": [
                    {"text_id": 0, "extractions": []},
                    {"text_id": 99, "extractions": []},
                ]
            }
        )
        client = _make_mock_client(llm_response)
        extractor = _make_extractor(client)

        with pytest.raises(BatchedExtractionError) as exc_info:
            extractor.extract_batch(texts)
        assert exc_info.value.stage == "schema_validation"


class TestErrorStages:
    """Each recoverable failure raises BatchedExtractionError with the right stage."""

    def test_invalid_json_raises_json_parse(self):
        client = _make_mock_client("not valid json {")
        extractor = _make_extractor(client)

        with pytest.raises(BatchedExtractionError) as exc_info:
            extractor.extract_batch(["hello"])
        assert exc_info.value.stage == "json_parse"

    def test_missing_results_key_raises_schema_validation(self):
        client = _make_mock_client(json.dumps({"oops": []}))
        extractor = _make_extractor(client)

        with pytest.raises(BatchedExtractionError) as exc_info:
            extractor.extract_batch(["hello"])
        assert exc_info.value.stage == "schema_validation"

    def test_llm_call_exception_raises_llm_call(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("connection refused")
        extractor = _make_extractor(client)

        with pytest.raises(BatchedExtractionError) as exc_info:
            extractor.extract_batch(["hello"])
        assert exc_info.value.stage == "llm_call"
        assert "connection refused" in exc_info.value.reason

    def test_hallucinated_text_raises_alignment(self):
        # Model returns an extraction whose text is not in the source
        text = "Contact us at info@example.com"
        llm_response = json.dumps(
            {
                "results": [
                    {
                        "text_id": 0,
                        "extractions": [
                            {"class": "person", "text": "Hallucinated McGee"},
                        ],
                    }
                ]
            }
        )
        client = _make_mock_client(llm_response)
        extractor = _make_extractor(client)

        with pytest.raises(BatchedExtractionError) as exc_info:
            extractor.extract_batch([text])
        assert exc_info.value.stage == "alignment"


class TestPartialAlignmentFiltering:
    """Mix of valid and hallucinated extractions: keep valid, drop hallucinated."""

    def test_valid_extractions_kept_when_some_hallucinated(self):
        text = "Contact John Doe at john@example.com"
        llm_response = json.dumps(
            {
                "results": [
                    {
                        "text_id": 0,
                        "extractions": [
                            {"class": "person", "text": "John Doe"},
                            {"class": "person", "text": "Definitely Not Here"},
                            {"class": "email", "text": "john@example.com"},
                        ],
                    }
                ]
            }
        )
        client = _make_mock_client(llm_response)
        extractor = _make_extractor(client)

        result = extractor.extract_batch([text])

        # Hallucinated one is dropped, valid two are kept
        assert len(result) == 1
        kept_texts = sorted(e.extraction_text for e in result[0])
        assert kept_texts == ["John Doe", "john@example.com"]


class TestChunking:
    """Splitting into multiple chunks when len(texts) > max_batch_size."""

    def test_splits_into_chunks_when_exceeds_max_size(self):
        texts = [f"text {i}" for i in range(5)]
        # Each chunk gets its own JSON response. The mock client returns
        # the next response in order on each call.
        client = MagicMock()
        responses = [
            _make_mock_chat_response(
                json.dumps(
                    {"results": [{"text_id": i, "extractions": []} for i in range(2)]}
                )
            ),
            _make_mock_chat_response(
                json.dumps(
                    {"results": [{"text_id": i, "extractions": []} for i in range(2)]}
                )
            ),
            _make_mock_chat_response(
                json.dumps(
                    {"results": [{"text_id": 0, "extractions": []}]}
                )
            ),
        ]
        client.chat.completions.create.side_effect = responses

        extractor = _make_extractor(client, max_batch_size=2)
        result = extractor.extract_batch(texts)

        assert len(result) == 5
        assert client.chat.completions.create.call_count == 3


class TestEmptyAndEdgeCases:
    """Edge cases: empty input list, empty extractions, no examples."""

    def test_empty_input_returns_empty_list(self):
        client = MagicMock()
        extractor = _make_extractor(client)

        result = extractor.extract_batch([])

        assert result == []
        assert client.chat.completions.create.call_count == 0

    def test_text_with_no_extractions_returns_empty_inner_list(self):
        llm_response = json.dumps(
            {"results": [{"text_id": 0, "extractions": []}]}
        )
        client = _make_mock_client(llm_response)
        extractor = _make_extractor(client)

        result = extractor.extract_batch(["just a normal sentence"])

        assert result == [[]]


class TestPromptRendering:
    """The cached batch prompt embeds the base prompt and the BATCH MODE marker."""

    def test_prompt_includes_base_prompt_and_batch_section(self):
        client = _make_mock_client(
            json.dumps({"results": [{"text_id": 0, "extractions": []}]})
        )
        # Use a wrapper template with a recognizable batch marker
        custom_wrapper = (
            "{{ base_prompt }}\n=== BATCH MODE BANNER ===\n"
        )
        extractor = BatchedLLMExtractor(
            openai_client=client,
            model_id="test-model",
            base_prompt_rendered="MY UNIQUE BASE PROMPT",
            wrapper_template_str=custom_wrapper,
            examples=[],
        )
        rendered = extractor._get_or_render_batch_prompt()

        assert "MY UNIQUE BASE PROMPT" in rendered
        assert "=== BATCH MODE BANNER ===" in rendered

    def test_prompt_is_cached_across_calls(self):
        client = _make_mock_client(
            json.dumps({"results": [{"text_id": 0, "extractions": []}]})
        )
        extractor = _make_extractor(client)
        first = extractor._get_or_render_batch_prompt()
        second = extractor._get_or_render_batch_prompt()
        assert first is second
