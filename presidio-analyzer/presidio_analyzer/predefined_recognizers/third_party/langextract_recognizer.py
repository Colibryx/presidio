import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from presidio_analyzer.llm_utils import (
    BatchedExtractionError,
    BatchedLLMExtractor,
    check_langextract_available,
    convert_langextract_batch_to_presidio_results_aligned,
    convert_langextract_to_presidio_results,
    convert_to_langextract_format,
    extract_lm_config,
    get_model_config,
    get_supported_entities,
    load_file_from_conf,
    load_prompt_file,
    load_yaml_examples,
    load_yaml_file,
    lx,
    make_extraction_document,
    presidio_langextract_batch_document_id,
    render_jinja_template,
    validate_config_fields,
)
from presidio_analyzer.lm_recognizer import LMRecognizer

logger = logging.getLogger("presidio-analyzer")


def _normalize_text_or_documents_for_langextract(
    text_or_documents: Any,
) -> Any:
    """Prepare input for :func:`langextract.extract`.

    A single ``str`` is passed through. An iterable of strings is converted to
    ``langextract.data.Document`` instances, which is what LangExtract requires
    for multi-document extraction (raw strings lack ``document_id``).
    """
    if isinstance(text_or_documents, str):
        return text_or_documents
    if isinstance(text_or_documents, (list, tuple)):
        normalized = []
        for i, item in enumerate(text_or_documents):
            if isinstance(item, str):
                normalized.append(
                    lx.data.Document(
                        text=item,
                        document_id=presidio_langextract_batch_document_id(i),
                    )
                )
            else:
                normalized.append(item)
        return normalized
    return text_or_documents


# Entity types that benefit from cybersecurity extraction guidance in the prompt
_CYBERSECURITY_ENTITIES = frozenset(
    {
        "INCIDENT_ID",
        "MALWARE",
        "RANSOMWARE",
        "THREAT_ACTOR",
        "THREAT_GROUP",
        "VULNERABILITY",
        "CVE",
        "CWE",
        "HASH_FILE",
        "INTERNAL_SERVER",
        "SOURCE_IP",
        "DESTINATION_IP",
        "PORT_NUMBER",
        "PROTOCOL",
        "USERNAME",
        "PASSWORD",
        "PROCESS_NAME",
        "COMMAND",
        "MITRE_TECHNIQUE",
        "IP_ADDRESS",
    }
)


class LangExtractRecognizer(LMRecognizer, ABC):
    """
    Base class for LangExtract-based PII recognizers.

    Subclasses implement _call_langextract() for specific LLM providers.
    """

    supports_multi_text_llm_extraction = True

    def __init__(
        self,
        config_path: str,
        name: str = "LangExtract LLM PII",
        supported_language: str = "en",
        extract_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize LangExtract recognizer.

        :param config_path: Path to configuration file.
        :param name: Name of the recognizer (provided by subclass).
        :param supported_language: Language this recognizer supports (default: "en").
        :param extract_params: Dict with 'extract' and/or 'language_model'
            keys containing param defaults.
        """
        check_langextract_available()

        full_config = load_yaml_file(config_path)

        lm_config = extract_lm_config(full_config)
        langextract_config = full_config.get("langextract", {})

        supported_entities = get_supported_entities(lm_config, langextract_config)

        if not supported_entities:
            raise ValueError("Configuration must contain 'supported_entities' in 'lm_recognizer' or 'langextract'")

        validate_config_fields(
            full_config,
            [
                ("langextract",),
                ("langextract", "model"),
                ("langextract", "model", "model_id"),
                ("langextract", "entity_mappings"),
                ("langextract", "prompt_file"),
                ("langextract", "examples_file"),
            ],
        )

        self.config = langextract_config
        model_config = get_model_config(full_config, provider_key="langextract")

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name,
            version="1.0.0",
            model_id=model_config["model_id"],
            temperature=model_config.get("temperature"),
            min_score=lm_config.get("min_score"),
            labels_to_ignore=lm_config.get("labels_to_ignore"),
            enable_generic_consolidation=lm_config.get("enable_generic_consolidation"),
        )

        examples_data = load_yaml_examples(langextract_config["examples_file"])
        self.examples = convert_to_langextract_format(examples_data)

        self._prompt_template = load_prompt_file(langextract_config["prompt_file"])

        self.entity_mappings = langextract_config["entity_mappings"]
        self.debug = langextract_config.get("debug", False)
        self._model_config = model_config

        # Process extract params with config override
        self._extract_params = {}
        self._language_model_params = {}

        if extract_params:
            if "extract" in extract_params:
                for param_name, default_value in extract_params["extract"].items():
                    self._extract_params[param_name] = self._model_config.get(param_name, default_value)

            if "language_model" in extract_params:
                for param_name, default_value in extract_params["language_model"].items():
                    self._language_model_params[param_name] = self._model_config.get(param_name, default_value)

        # Fallback: if model.provider exists, merge provider.extract_params
        # and provider.language_model_params (nested config structure)
        provider_config = self._model_config.get("provider", {})
        if provider_config:
            self._extract_params.update(provider_config.get("extract_params", {}))
            self._language_model_params.update(provider_config.get("language_model_params", {}))

        # Native batching configuration. When enabled, _call_llm_batch() uses
        # BatchedLLMExtractor to issue a single Chat Completions call for N
        # texts. When disabled (or absent), batch calls fall through to the
        # legacy LangExtract path (one LLM call per document).
        batch_config = langextract_config.get("batch", {}) or {}
        self._batch_enabled = bool(batch_config.get("enabled", False))
        self._batch_max_size = int(batch_config.get("max_size", 20))
        self._batch_wrapper_prompt_file = batch_config.get(
            "wrapper_prompt_file",
            "presidio_analyzer/conf/langextract_prompts/batch_extraction_wrapper.j2",
        )
        # Lazy: built on first batch call so subclass-specific client setup
        # is fully initialized before we touch it.
        self._batched_extractor: Optional[BatchedLLMExtractor] = None

    def _call_llm(self, text: str, entities: List[str], **kwargs):
        """Call LangExtract LLM."""
        # Render prompt with requested entities only (reduces tokens, improves focus)
        show_cybersecurity_guidance = bool(set(entities) & _CYBERSECURITY_ENTITIES)
        prompt_description = render_jinja_template(
            self._prompt_template,
            supported_entities=entities,
            enable_generic_consolidation=self.enable_generic_consolidation,
            labels_to_ignore=self.labels_to_ignore,
            show_cybersecurity_guidance=show_cybersecurity_guidance,
        )

        # Build extract params
        extract_params = {
            "text": text,
            "prompt": prompt_description,
            "examples": self.examples,
            "debug": self.debug,
        }

        # Add temperature if configured
        if self.temperature is not None:
            extract_params["temperature"] = self.temperature

        # Add any additional kwargs
        extract_params.update(kwargs)

        langextract_result = self._call_langextract(**extract_params)

        return convert_langextract_to_presidio_results(
            langextract_result=langextract_result,
            entity_mappings=self.entity_mappings,
            supported_entities=self.supported_entities,
            enable_generic_consolidation=self.enable_generic_consolidation,
            recognizer_name=self.__class__.__name__,
        )

    def _call_llm_batch(self, texts: List[str], entities: List[str], **kwargs):
        """Run a single LLM call for multiple texts when batch mode is enabled.

        With ``langextract.batch.enabled: true`` in the config, this dispatches
        to :class:`BatchedLLMExtractor` to issue ONE Chat Completions request
        for the entire ``texts`` list and recover per-text offsets via
        ``Resolver.align()``. On any recoverable error
        (:class:`BatchedExtractionError`) or unexpected exception we log and
        fall back to the legacy LangExtract path (one LLM call per document).

        With batch mode disabled, this is a thin pass-through to the legacy
        path so behavior matches the previous version exactly.
        """
        if not self._batch_enabled:
            return self._call_llm_batch_via_langextract(texts, entities, **kwargs)

        try:
            extractor = self._get_or_build_batched_extractor()
            aligned_per_text = extractor.extract_batch(texts, entities)
        except BatchedExtractionError as e:
            logger.warning(
                "Batched LLM extraction failed at stage=%s reason=%s; "
                "falling back to per-text langextract for %d texts (details=%s)",
                e.stage,
                e.reason,
                len(texts),
                e.details,
            )
            return self._call_llm_batch_via_langextract(texts, entities, **kwargs)
        except Exception:
            logger.exception(
                "Unexpected error in batched extraction; falling back to "
                "langextract for %d texts",
                len(texts),
            )
            return self._call_llm_batch_via_langextract(texts, entities, **kwargs)

        # Convert each per-text Extraction list into RecognizerResults using
        # the existing converter (which only reads .extractions from its
        # input — wrap with a SimpleNamespace for free).
        return [
            convert_langextract_to_presidio_results(
                make_extraction_document(aligned),
                entity_mappings=self.entity_mappings,
                supported_entities=self.supported_entities,
                enable_generic_consolidation=self.enable_generic_consolidation,
                recognizer_name=self.__class__.__name__,
            )
            for aligned in aligned_per_text
        ]

    def _call_llm_batch_via_langextract(
        self, texts: List[str], entities: List[str], **kwargs
    ):
        """Legacy batch path: one LangExtract call that loops per document.

        This is the previous implementation of ``_call_llm_batch``, kept
        intact as a fallback for the new native-batching path. It is also
        used directly when ``langextract.batch.enabled`` is false.
        """
        show_cybersecurity_guidance = bool(set(entities) & _CYBERSECURITY_ENTITIES)
        prompt_description = render_jinja_template(
            self._prompt_template,
            supported_entities=entities,
            enable_generic_consolidation=self.enable_generic_consolidation,
            labels_to_ignore=self.labels_to_ignore,
            show_cybersecurity_guidance=show_cybersecurity_guidance,
        )

        extract_params = {
            "text": texts,
            "prompt": prompt_description,
            "examples": self.examples,
            "debug": self.debug,
        }

        if self.temperature is not None:
            extract_params["temperature"] = self.temperature

        extract_params.update(kwargs)

        langextract_result = self._call_langextract(**extract_params)

        batch_results = convert_langextract_batch_to_presidio_results_aligned(
            langextract_batch_result=langextract_result,
            num_input_documents=len(texts),
            entity_mappings=self.entity_mappings,
            supported_entities=self.supported_entities,
            enable_generic_consolidation=self.enable_generic_consolidation,
            recognizer_name=self.__class__.__name__,
        )
        return batch_results

    def _get_or_build_batched_extractor(self) -> BatchedLLMExtractor:
        """Lazily build and cache the :class:`BatchedLLMExtractor`.

        The wrapper prompt is rendered with the recognizer's full entity
        list (the same one used by the legacy path); cybersecurity guidance
        is enabled iff at least one cybersecurity-relevant entity is in the
        recognizer's supported_entities. The Resolver downstream will only
        keep alignments for what the model actually emitted, so this
        defensive choice is safe.
        """
        if self._batched_extractor is not None:
            return self._batched_extractor

        # Render the base prompt with the recognizer's full entity list.
        # We use supported_entities (not the per-call ``entities``) so the
        # cached extractor stays valid across calls regardless of which
        # subset of entities a given AnalyzerEngine call requests.
        show_cybersecurity_guidance = bool(
            set(self.supported_entities) & _CYBERSECURITY_ENTITIES
        )
        base_prompt_rendered = render_jinja_template(
            self._prompt_template,
            supported_entities=self.supported_entities,
            enable_generic_consolidation=self.enable_generic_consolidation,
            labels_to_ignore=self.labels_to_ignore,
            show_cybersecurity_guidance=show_cybersecurity_guidance,
        )

        wrapper_template_str = load_file_from_conf(self._batch_wrapper_prompt_file)

        client, model_id_for_batch = self._get_openai_client_for_batch()

        # Forward only the LLM-call relevant params (temperature, extra_body,
        # max_tokens) — extract-time params like max_char_buffer are
        # langextract-specific and have no analog in raw chat.completions.
        extra_llm_params: Dict[str, Any] = {}
        if self.temperature is not None:
            extra_llm_params["temperature"] = self.temperature
        for key in ("max_tokens", "max_completion_tokens", "extra_body"):
            if key in self._language_model_params:
                extra_llm_params[key] = self._language_model_params[key]
        # ``max_output_tokens`` is the langextract spelling — translate to
        # the OpenAI ``max_tokens`` parameter when present.
        if (
            "max_tokens" not in extra_llm_params
            and "max_output_tokens" in self._language_model_params
        ):
            extra_llm_params["max_tokens"] = self._language_model_params[
                "max_output_tokens"
            ]

        self._batched_extractor = BatchedLLMExtractor(
            openai_client=client,
            model_id=model_id_for_batch,
            base_prompt_rendered=base_prompt_rendered,
            wrapper_template_str=wrapper_template_str,
            examples=self.examples,
            max_batch_size=self._batch_max_size,
            extra_llm_params=extra_llm_params,
        )
        return self._batched_extractor

    def _call_langextract(self, **kwargs):
        """Call LangExtract with configured parameters."""
        try:
            extract_params = {
                "text_or_documents": _normalize_text_or_documents_for_langextract(kwargs.pop("text")),
                "prompt_description": kwargs.pop("prompt"),
                "examples": kwargs.pop("examples"),
            }

            extract_params.update(self._get_provider_params())
            extract_params.update(self._extract_params)
            if self._language_model_params:
                extract_params["language_model_params"] = self._language_model_params
            extract_params.update(kwargs)

            return lx.extract(**extract_params)
        except Exception:
            logger.exception("LangExtract extraction failed (model '%s')", self.model_id)
            raise

    @abstractmethod
    def _get_provider_params(self) -> Dict[str, Any]:
        """Return provider-specific params.

        Examples: model_id, model_url, azure_endpoint, etc.
        """
        ...

    @abstractmethod
    def _get_openai_client_for_batch(self) -> tuple:
        """Return ``(openai_client, model_id_for_chat_completions)``.

        Used by :meth:`_call_llm_batch` (when ``batch.enabled: true``) to
        issue a raw OpenAI Chat Completions request bypassing LangExtract.
        Subclasses must return:

        - ``openai_client``: an ``openai.OpenAI`` or ``openai.AzureOpenAI``
          instance with the right base_url/auth already configured. Should
          be created via the langfuse-patched openai module so traces are
          captured automatically when Langfuse is enabled.
        - ``model_id``: the string passed as ``model`` to
          ``client.chat.completions.create`` (a deployment name for Azure,
          a model id like ``"qwen/qwen3.5-35b-a3b"`` for Basic).
        """
        ...
