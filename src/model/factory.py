from typing import Literal

from src.config import Configuration
from src.model.oai import OpenAIClient

LLMProvider = Literal["openai", "anthropic", "google"]


def create_llm_client(provider: LLMProvider, config: Configuration) -> OpenAIClient:
    """Create appropriate LLM client based on provider.

    Args:
        provider: LLM provider type
        config: Configuration object containing LLM model name, API key, and base URL

    Returns:
        Initialized LLM client instance

    """
    if provider == "openai":
        return OpenAIClient(
            api_key=config.llm_api_key,
            model_name=config.llm_model_name,
        )
    if provider == "anthropic":
        raise NotImplementedError("Anthropic is not supported yet")
    if provider == "google":
        raise NotImplementedError("Google is not supported yet")
    raise ValueError(f"Unsupported LLM provider: {provider}")
