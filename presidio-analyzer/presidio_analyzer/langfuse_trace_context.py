"""Request-scoped Langfuse trace attributes via ``propagate_attributes``."""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("presidio-analyzer")

try:
    from langfuse import propagate_attributes as _propagate_attributes
except ImportError:  # optional extra / old SDK
    _propagate_attributes = None


def _normalize_tags(tags: Any) -> Optional[List[str]]:
    if tags is None:
        return None
    if isinstance(tags, str):
        return [tags]
    if isinstance(tags, (list, tuple)):
        return [str(t) for t in tags]
    raise TypeError("langfuse.tags must be a string or a list of strings")


def normalize_langfuse_trace_options(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate and normalize the ``langfuse`` object from an /analyze JSON body.

    Supported keys mirror :func:`langfuse.propagate_attributes` (SDK v3+).
    Unknown keys are ignored. Only non-empty values are kept.
    """
    out: Dict[str, Any] = {}
    for key in ("session_id", "user_id", "trace_name", "version"):
        val = raw.get(key)
        if val is not None and val != "":
            out[key] = val
    if "tags" in raw:
        tags = _normalize_tags(raw.get("tags"))
        if tags:
            out["tags"] = tags
    md = raw.get("metadata")
    if md is not None:
        if not isinstance(md, dict):
            raise TypeError("langfuse.metadata must be an object")
        out["metadata"] = md
    return out or None


def _kwargs_for_propagate(options: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys the installed ``propagate_attributes`` does not accept."""
    if _propagate_attributes is None:
        return {}
    try:
        sig = inspect.signature(_propagate_attributes)
    except (TypeError, ValueError):
        return {k: v for k, v in options.items() if v is not None}
    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return {k: v for k, v in options.items() if v is not None}
    names = {p.name for p in params}
    return {k: v for k, v in options.items() if k in names and v is not None}


@contextmanager
def langfuse_trace_scope(
    options: Optional[Dict[str, Any]],
) -> Iterator[None]:
    """
    Apply Langfuse correlating attributes to the current trace for this block.

    No-op when ``options`` is None/empty, when Langfuse is not installed, or when
    ``propagate_attributes`` is missing (requires Langfuse Python SDK v3+).
    """
    if not options:
        yield
        return
    if _propagate_attributes is None:
        logger.debug(
            "langfuse.propagate_attributes not available — "
            "install langfuse>=3 to enable request-scoped trace attributes"
        )
        yield
        return
    filtered = _kwargs_for_propagate(options)
    if not filtered:
        yield
        return
    with _propagate_attributes(**filtered):
        yield
