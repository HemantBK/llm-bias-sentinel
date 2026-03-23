"""Unified model loading for Ollama local models."""

from typing import Optional
from loguru import logger

from src.config import config


def load_model(
    provider: str = "ollama",
    model_id: str = "llama3",
    temperature: float = 0.0,
    name: Optional[str] = None,
    **kwargs,
):
    """Load a model through a unified interface.

    Currently supports Ollama for free local inference.
    Easily extensible to OpenAI or other providers.
    """
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        model = ChatOllama(
            model=model_id,
            temperature=temperature,
            base_url=config.ollama_base_url,
        )
        logger.info(f"Loaded Ollama model: {model_id}")
        return model
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Currently supported: ollama"
        )


def generate_response(
    model,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate a response from any loaded model."""
    messages = []
    if system_prompt:
        messages.append(("system", system_prompt))
    messages.append(("human", prompt))

    response = model.invoke(messages)
    return response.content
