"""Model metadata registry for tracking evaluated models."""

from dataclasses import dataclass

from src.config import config
from src.models.model_loader import load_model


@dataclass
class ModelInfo:
    """Metadata about a model under evaluation."""
    name: str
    provider: str
    model_id: str
    parameters: str | None = None
    description: str | None = None


# Known model metadata
MODEL_CATALOG = {
    "llama3-8b": ModelInfo(
        name="llama3-8b",
        provider="ollama",
        model_id="llama3",
        parameters="8B",
        description="Meta LLaMA 3 8B Instruct — general-purpose open model",
    ),
    "mistral-7b": ModelInfo(
        name="mistral-7b",
        provider="ollama",
        model_id="mistral",
        parameters="7B",
        description="Mistral 7B Instruct v0.3 — efficient open model",
    ),
}


class ModelRegistry:
    """Registry for managing models under evaluation."""

    def __init__(self):
        self._loaded_models: dict = {}

    def get_model(self, name: str):
        """Get or load a model by registry name."""
        if name in self._loaded_models:
            return self._loaded_models[name]

        if name in MODEL_CATALOG:
            info = MODEL_CATALOG[name]
            model = load_model(
                provider=info.provider,
                model_id=info.model_id,
                name=info.name,
            )
            self._loaded_models[name] = model
            return model

        raise KeyError(f"Model '{name}' not found in registry. Available: {list(MODEL_CATALOG.keys())}")

    def get_all_configured_models(self) -> dict:
        """Load and return all models from config."""
        models = {}
        for model_config in config.models:
            name = model_config["name"]
            models[name] = self.get_model(name)
        return models

    def list_available(self) -> list[str]:
        """List all available model names."""
        return list(MODEL_CATALOG.keys())

    def get_info(self, name: str) -> ModelInfo | None:
        """Get model metadata."""
        return MODEL_CATALOG.get(name)
