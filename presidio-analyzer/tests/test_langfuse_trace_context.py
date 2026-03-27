"""Tests for Langfuse request trace context helpers."""

from unittest.mock import MagicMock, patch

import pytest

from presidio_analyzer.langfuse_trace_context import (
    langfuse_trace_scope,
    normalize_langfuse_trace_options,
)


class TestNormalizeLangfuseTraceOptions:
    def test_empty_object_returns_none(self):
        assert normalize_langfuse_trace_options({}) is None

    def test_session_user_tags(self):
        out = normalize_langfuse_trace_options(
            {
                "session_id": "sess-1",
                "user_id": "user-1",
                "tags": ["a", "b"],
            }
        )
        assert out == {
            "session_id": "sess-1",
            "user_id": "user-1",
            "tags": ["a", "b"],
        }

    def test_tags_single_string(self):
        out = normalize_langfuse_trace_options({"tags": "single"})
        assert out == {"tags": ["single"]}

    def test_skips_blank_session_id(self):
        out = normalize_langfuse_trace_options({"session_id": "", "user_id": "u"})
        assert out == {"user_id": "u"}

    def test_metadata_dict(self):
        out = normalize_langfuse_trace_options(
            {"session_id": "s", "metadata": {"k": "v"}}
        )
        assert out["metadata"] == {"k": "v"}

    def test_metadata_must_be_object(self):
        with pytest.raises(TypeError, match="langfuse.metadata"):
            normalize_langfuse_trace_options({"metadata": "not-a-dict"})

    def test_tags_invalid_type(self):
        with pytest.raises(TypeError, match="langfuse.tags"):
            normalize_langfuse_trace_options({"tags": 123})


class TestLangfuseTraceScope:
    def test_no_options_yields_without_propagate(self):
        with langfuse_trace_scope(None):
            pass

    def test_when_propagate_missing_skips(self):
        with patch(
            "presidio_analyzer.langfuse_trace_context._propagate_attributes",
            None,
        ):
            with langfuse_trace_scope({"session_id": "s"}):
                pass

    def test_calls_propagate_when_available(self):
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)

        def fake_propagate(**kwargs):
            mock_cm._kwargs = kwargs
            return mock_cm

        with patch(
            "presidio_analyzer.langfuse_trace_context._propagate_attributes",
            fake_propagate,
        ):
            with langfuse_trace_scope(
                {"session_id": "sid", "user_id": "uid", "tags": ["t"]}
            ):
                pass

        assert mock_cm._kwargs == {
            "session_id": "sid",
            "user_id": "uid",
            "tags": ["t"],
        }
        mock_cm.__enter__.assert_called_once()
        mock_cm.__exit__.assert_called_once()
