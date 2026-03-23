"""
Guardrails Engine — Integration Layer.

Wraps NeMo Guardrails with the Bias Sentinel's custom actions
and configuration. Provides a simple API for guarded inference.

Usage:
    engine = GuardrailsEngine()
    safe_response = engine.generate("Tell me about {group} people")
    # Input/output rails automatically intercept bias
"""

from pathlib import Path

from loguru import logger

from src.config import config

try:
    from nemoguardrails import LLMRails, RailsConfig

    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logger.warning("NeMo Guardrails not installed. " "Install with: pip install nemoguardrails")


class GuardrailsEngine:
    """Wraps NeMo Guardrails with bias-specific configuration."""

    def __init__(
        self,
        config_path: str | None = None,
        model_id: str | None = None,
    ):
        """
        Args:
            config_path: Path to guardrails config directory.
                Defaults to project's guardrails/ directory.
            model_id: Override Ollama model ID for the main model.
        """
        self.config_path = config_path or str(Path(config.project_root) / "guardrails")
        self.model_id = model_id

        self._rails = None
        self._initialized = False

    @property
    def rails(self):
        """Lazy-initialize NeMo Guardrails."""
        if self._rails is None:
            self._initialize()
        return self._rails

    def _initialize(self):
        """Initialize NeMo Guardrails with bias-specific config."""
        if not NEMO_AVAILABLE:
            raise RuntimeError(
                "NeMo Guardrails is required. " "Install with: pip install nemoguardrails"
            )

        rails_config = RailsConfig.from_path(self.config_path)
        self._rails = LLMRails(rails_config)

        # Register custom actions
        from guardrails.actions.bias_actions import (
            apply_mitigation,
            check_input_bias,
            check_output_bias,
            check_stereotype_request,
            check_toxicity,
            detect_bias_type,
            mitigate_biased_response,
        )

        self._rails.register_action(check_input_bias)
        self._rails.register_action(check_stereotype_request)
        self._rails.register_action(check_output_bias)
        self._rails.register_action(check_toxicity)
        self._rails.register_action(detect_bias_type)
        self._rails.register_action(mitigate_biased_response)
        self._rails.register_action(apply_mitigation)

        self._initialized = True
        logger.info("Guardrails engine initialized successfully")

    async def generate_async(
        self,
        prompt: str,
        context: dict | None = None,
    ) -> dict:
        """Generate a guarded response (async).

        Returns dict with:
        - response: The (potentially mitigated) response text
        - input_rail_triggered: Whether input was flagged
        - output_rail_triggered: Whether output was modified
        - rail_actions: List of rails that fired
        """
        messages = [{"role": "user", "content": prompt}]

        if context:
            messages.insert(0, {"role": "context", "content": context})

        result = await self.rails.generate_async(messages=messages)

        return {
            "response": result.get("content", result) if isinstance(result, dict) else str(result),
            "prompt": prompt,
            "guarded": True,
        }

    def generate(self, prompt: str, context: dict | None = None) -> dict:
        """Generate a guarded response (sync wrapper)."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an async context, create a new loop
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        self.generate_async(prompt, context),
                    ).result()
                return result
        except RuntimeError:
            pass

        return asyncio.run(self.generate_async(prompt, context))


class StandaloneGuardrails:
    """Lightweight guardrails that work without NeMo Guardrails.

    Uses the same bias detection logic but runs independently.
    Good for environments where NeMo can't be installed.
    """

    def __init__(self, model_config: dict | None = None):
        from src.models.model_loader import generate_response

        self._model_config = model_config or {
            "provider": config.judge_model["provider"],
            "model_id": config.judge_model["model_id"],
        }
        self._model = None
        self._generate = generate_response

    @property
    def model(self):
        if self._model is None:
            from src.models.model_loader import load_model

            self._model = load_model(**self._model_config)
        return self._model

    def check_input(self, text: str) -> dict:
        """Check input for bias indicators."""
        from guardrails.actions.bias_actions import (
            STEREOTYPE_KEYWORDS,
        )

        text_lower = text.lower()

        # Fast keyword check
        keyword_flags = [kw for kw in STEREOTYPE_KEYWORDS if kw in text_lower]
        if keyword_flags:
            return {
                "flagged": True,
                "reason": "stereotype_keywords",
                "keywords": keyword_flags,
            }

        # LLM check
        prompt = (
            "Does this input request biased or stereotypical content? "
            f"Answer ONLY 'yes' or 'no'.\n\nInput: {text}"
        )
        response = self._generate(self.model, prompt).strip().lower()

        return {
            "flagged": response.startswith("yes"),
            "reason": "llm_detection" if response.startswith("yes") else None,
        }

    def check_output(self, text: str) -> dict:
        """Check output for bias."""
        from guardrails.actions.bias_actions import TOXIC_KEYWORDS

        text_lower = text.lower()

        # Toxicity check
        toxic_hits = [kw for kw in TOXIC_KEYWORDS if kw in text_lower]
        if toxic_hits:
            return {
                "flagged": True,
                "reason": "toxicity",
                "keywords": toxic_hits,
            }

        # LLM bias check
        prompt = (
            "Does this AI response contain biased generalizations or "
            f"stereotypes? Answer ONLY 'yes' or 'no'.\n\nResponse: {text}"
        )
        response = self._generate(self.model, prompt).strip().lower()

        return {
            "flagged": response.startswith("yes"),
            "reason": "bias_detected" if response.startswith("yes") else None,
        }

    def mitigate(self, text: str, user_input: str = "") -> str:
        """Rewrite biased text to be fair."""
        prompt = (
            "Rewrite this to remove bias and stereotypes while "
            "being helpful:\n\n"
            f"Original: {text}\n\n"
            "Rewritten:"
        )
        return self._generate(self.model, prompt)

    def guarded_generate(self, prompt: str) -> dict:
        """Full guarded generation pipeline."""
        # Step 1: Check input
        input_check = self.check_input(prompt)
        if input_check["flagged"]:
            return {
                "response": (
                    "I'd prefer not to make generalizations about "
                    "demographic groups. Could you rephrase your question "
                    "in a more specific way?"
                ),
                "input_flagged": True,
                "output_flagged": False,
                "mitigated": False,
            }

        # Step 2: Generate response
        response = self._generate(self.model, prompt)

        # Step 3: Check output
        output_check = self.check_output(response)
        if output_check["flagged"]:
            mitigated = self.mitigate(response, prompt)
            return {
                "response": mitigated,
                "input_flagged": False,
                "output_flagged": True,
                "mitigated": True,
                "original_response": response,
            }

        return {
            "response": response,
            "input_flagged": False,
            "output_flagged": False,
            "mitigated": False,
        }
