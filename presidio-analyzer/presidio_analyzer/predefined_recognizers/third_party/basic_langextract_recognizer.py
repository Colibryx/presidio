import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from presidio_analyzer.llm_utils import lx_factory
from presidio_analyzer.predefined_recognizers.third_party.langextract_recognizer import (
    LangExtractRecognizer,
)

logger = logging.getLogger("presidio-analyzer")

load_dotenv()

DEFAULT_EXTRACT_PARAMS = {
    "max_char_buffer": 400,
    "use_schema_constraints": False,
    "fence_output": False,
}

DEFAULT_LANGUAGE_MODEL_PARAMS = {"timeout": 240, "num_ctx": 8192}


class BasicLangExtractRecognizer(LangExtractRecognizer):
    """Basic LangExtract recognizer using configurable backend."""

    DEFAULT_CONFIG_PATH = (
        Path(__file__).parent.parent.parent / "conf" / "langextract_config_basic.yaml"
    )

    def __init__(
        self,
        config_path: Optional[str] = None,
        supported_language: str = "en",
        context: Optional[list] = None,
        name="BasicLangExtractRecognizer",
        **kwargs,
    ):
        """Initialize Basic LangExtract recognizer.

        :param config_path: Path to configuration file (optional).
        :param supported_language: Language this recognizer supports
            (optional, default: "en").
        :param context: List of context words
            (optional, currently not used by LLM recognizers).
        """
        actual_config_path = (
            config_path if config_path else str(self.DEFAULT_CONFIG_PATH)
        )

        super().__init__(
            config_path=actual_config_path,
            name=name,
            supported_language=supported_language,
            extract_params={
                "extract": DEFAULT_EXTRACT_PARAMS,
                "language_model": DEFAULT_LANGUAGE_MODEL_PARAMS,
            },
        )

        model_config: Dict[str, Any] = self.config.get("model", {})
        provider_config = model_config.get("provider", {})

        self.model_id = model_config.get("model_id")
        self.provider = provider_config.get("name")
        self.provider_kwargs = provider_config.get("kwargs", {})

        if not self.provider:
            raise ValueError(
                "Configuration must contain 'langextract.model.provider.name'"
            )

        if self.provider_kwargs.get("base_url") is None:
            if env_value := os.environ.get("LANGEXTRACT_BASE_URL"):
                self.provider_kwargs["base_url"] = env_value
            else:
                self.provider_kwargs.pop("base_url", None)

        if self.provider_kwargs.get("api_key") is None:
            if env_value := os.environ.get("LANGEXTRACT_API_KEY"):
                self.provider_kwargs["api_key"] = env_value
            else:
                self.provider_kwargs.pop("api_key", None)

        # if (
        #     "api_key" not in self.provider_kwargs
        #     and "LANGEXTRACT_API_KEY" in os.environ
        # ):
        #     self.provider_kwargs["api_key"] = os.environ["LANGEXTRACT_API_KEY"]

        # Merge language_model_params into provider_kwargs so they reach
        # the OllamaLanguageModel constructor (and ultimately _ollama_query).
        # This is necessary because lx.extract() ignores language_model_params
        # when a config object is provided.
        provider_kwargs_with_lm_params = dict(self.provider_kwargs)
        provider_kwargs_with_lm_params.update(self._language_model_params)

        self.lx_model_config = lx_factory.ModelConfig(
            model_id=self.model_id,
            provider=self.provider,
            provider_kwargs=provider_kwargs_with_lm_params,
        )

    def _get_provider_params(self):
        """Return supplementary params."""
        return {
            "config": self.lx_model_config,
        }

    def _get_openai_client_for_batch(self):
        """Return an OpenAI-compatible client for the batched extractor.

        Reuses ``base_url``/``api_key`` from ``provider.kwargs`` (already
        resolved against environment variables in ``__init__``). The
        ``openai`` module is imported through the same indirection as the
        Azure provider, so when Langfuse is enabled the client created
        here is automatically traced.

        :return: ``(client, model_id)`` tuple suitable for
            ``BatchedLLMExtractor``.
        :raises ImportError: If neither ``openai`` nor the Langfuse-wrapped
            equivalent is importable.
        :raises ValueError: If the configured provider is not OpenAI-
            compatible (e.g. native Ollama).
        """
        if self.provider != "openai":
            raise ValueError(
                f"BasicLangExtractRecognizer batch mode currently supports "
                f"only the 'openai' provider (OpenAI-compatible APIs such as "
                f"vLLM, SGLang, Together, etc.). Configured provider: "
                f"{self.provider!r}."
            )

        # Reuse the same module reference the Azure provider uses so that
        # Langfuse tracing is consistent across both backends.
        from presidio_analyzer.predefined_recognizers.third_party import (
            azure_openai_provider,
        )

        openai_module = azure_openai_provider.openai
        if openai_module is None:
            raise ImportError(
                "openai SDK is not installed. Install it with: "
                "pip install openai"
            )

        client_kwargs = {}
        base_url = self.provider_kwargs.get("base_url")
        api_key = self.provider_kwargs.get("api_key")
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key

        client = openai_module.OpenAI(**client_kwargs)
        return client, self.model_id
