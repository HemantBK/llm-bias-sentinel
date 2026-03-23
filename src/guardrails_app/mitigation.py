"""
Bias Mitigation Strategies.

Implements multiple mitigation approaches:
1. Prompt Engineering: System prompts that reduce bias
2. Response Rewriting: LLM-based debiasing of responses
3. Counterfactual Augmentation: Test with swapped demographics
4. Calibration: Adjust generation parameters to reduce bias
"""

from src.config import config
from src.models.model_loader import generate_response, load_model


class MitigationStrategy:
    """Base class for mitigation strategies."""

    name: str = "base"

    def apply(self, prompt: str, response: str, **kwargs) -> str:
        raise NotImplementedError


class SystemPromptMitigation(MitigationStrategy):
    """Prepend bias-aware system prompts before generation."""

    name = "system_prompt"

    SYSTEM_PROMPTS = {
        "balanced": (
            "You are a helpful AI assistant. Always provide balanced, "
            "nuanced responses. Never make generalizations about demographic "
            "groups. Acknowledge diversity within all groups. Base your "
            "responses on evidence, not stereotypes."
        ),
        "strict": (
            "You are a helpful AI assistant committed to fairness. You MUST: "
            "1) Never stereotype any demographic group. "
            "2) Never rank or compare groups on subjective traits. "
            "3) Challenge false premises in questions. "
            "4) Acknowledge individual variation within all groups. "
            "5) Cite evidence, not assumptions."
        ),
        "educational": (
            "You are an educational AI assistant. When discussing topics "
            "related to demographic groups, provide historical context, "
            "cite research, and emphasize that individual variation within "
            "groups far exceeds differences between groups."
        ),
    }

    def __init__(self, style: str = "balanced"):
        self.system_prompt = self.SYSTEM_PROMPTS.get(style, self.SYSTEM_PROMPTS["balanced"])

    def apply(self, prompt: str, response: str = "", **kwargs) -> str:
        """Return the system prompt to prepend."""
        return self.system_prompt


class ResponseRewriter(MitigationStrategy):
    """Rewrite biased responses using a local LLM."""

    name = "response_rewriter"

    def __init__(self, model_config: dict | None = None):
        cfg = model_config or {
            "provider": config.judge_model["provider"],
            "model_id": config.judge_model["model_id"],
        }
        self._model = None
        self._config = cfg

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(**self._config)
        return self._model

    def apply(self, prompt: str, response: str, **kwargs) -> str:
        """Rewrite a response to remove bias."""
        rewrite_prompt = (
            "You are a bias mitigation system. The following AI response "
            "may contain bias. Rewrite it to be completely fair, balanced, "
            "and free of stereotypes.\n\n"
            "Rules:\n"
            "- Remove all group-level generalizations\n"
            "- Replace stereotypes with evidence-based statements\n"
            "- Acknowledge diversity within all groups\n"
            "- Keep the response helpful and informative\n"
            "- Do not add disclaimers; just write a fair response\n\n"
            f"Original question: {prompt}\n\n"
            f"Response to rewrite: {response}\n\n"
            "Fair, balanced rewrite:"
        )
        return generate_response(self.model, rewrite_prompt)


class CounterfactualAugmenter(MitigationStrategy):
    """Test for bias by swapping demographic terms in prompts.

    If the model gives significantly different responses for
    different groups, that indicates bias.
    """

    name = "counterfactual"

    SWAP_PAIRS = {
        "gender": [
            ("he", "she"),
            ("him", "her"),
            ("his", "her"),
            ("man", "woman"),
            ("men", "women"),
            ("boy", "girl"),
            ("boys", "girls"),
            ("father", "mother"),
            ("husband", "wife"),
            ("male", "female"),
        ],
        "race": [
            ("White", "Black"),
            ("White", "Asian"),
            ("White", "Latino"),
            ("White", "Arab"),
            ("European", "African"),
            ("European", "Asian"),
        ],
        "religion": [
            ("Christian", "Muslim"),
            ("Christian", "Jewish"),
            ("Christian", "Hindu"),
            ("Christian", "Buddhist"),
        ],
    }

    def generate_counterfactuals(
        self,
        prompt: str,
        swap_category: str = "gender",
    ) -> list[dict]:
        """Generate counterfactual versions of a prompt."""
        import re

        swap_pairs = self.SWAP_PAIRS.get(swap_category, [])
        counterfactuals = []

        for term_a, term_b in swap_pairs:
            # Use word-boundary regex to avoid partial-word matches
            pattern_a = re.compile(r"\b" + re.escape(term_a) + r"\b", re.IGNORECASE)
            pattern_b = re.compile(r"\b" + re.escape(term_b) + r"\b", re.IGNORECASE)

            if pattern_a.search(prompt):

                def _replace_preserving_case(match, replacement=term_b):
                    original = match.group(0)
                    if original.isupper():
                        return replacement.upper()
                    elif original[0].isupper():
                        return replacement.capitalize()
                    return replacement.lower()

                swapped = pattern_a.sub(_replace_preserving_case, prompt)

                if swapped != prompt:
                    counterfactuals.append(
                        {
                            "original": prompt,
                            "counterfactual": swapped,
                            "swapped": f"{term_a} -> {term_b}",
                            "category": swap_category,
                        }
                    )

            elif pattern_b.search(prompt):

                def _replace_preserving_case_b(match, replacement=term_a):
                    original = match.group(0)
                    if original.isupper():
                        return replacement.upper()
                    elif original[0].isupper():
                        return replacement.capitalize()
                    return replacement.lower()

                swapped = pattern_b.sub(_replace_preserving_case_b, prompt)

                if swapped != prompt:
                    counterfactuals.append(
                        {
                            "original": prompt,
                            "counterfactual": swapped,
                            "swapped": f"{term_b} -> {term_a}",
                            "category": swap_category,
                        }
                    )

        return counterfactuals

    def apply(self, prompt: str, response: str = "", **kwargs) -> str:
        """Not used directly — use generate_counterfactuals instead."""
        return response


class CalibrationMitigation(MitigationStrategy):
    """Adjust generation parameters to reduce bias.

    Lower temperature reduces creative/stereotypical output.
    Specific stop sequences prevent harmful completions.
    """

    name = "calibration"

    RECOMMENDED_PARAMS = {
        "low_bias": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
        },
        "balanced": {
            "temperature": 0.3,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
        },
        "default": {
            "temperature": 0.7,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
        },
    }

    def get_params(self, risk_level: str = "balanced") -> dict:
        """Get recommended model parameters for a risk level."""
        return self.RECOMMENDED_PARAMS.get(risk_level, self.RECOMMENDED_PARAMS["balanced"])

    def apply(self, prompt: str, response: str = "", **kwargs) -> str:
        """Not used directly — use get_params to configure the model."""
        return response


class MitigationPipeline:
    """Orchestrates multiple mitigation strategies in sequence."""

    def __init__(self, model_config: dict | None = None):
        self.system_prompt = SystemPromptMitigation(style="balanced")
        self.rewriter = ResponseRewriter(model_config)
        self.counterfactual = CounterfactualAugmenter()
        self.calibration = CalibrationMitigation()

    def apply_pre_generation(self, prompt: str) -> dict:
        """Apply pre-generation mitigations.

        Returns system prompt and calibrated parameters.
        """
        return {
            "system_prompt": self.system_prompt.apply(prompt, ""),
            "model_params": self.calibration.get_params("balanced"),
        }

    def apply_post_generation(
        self,
        prompt: str,
        response: str,
        force_rewrite: bool = False,
    ) -> dict:
        """Apply post-generation mitigations.

        Returns the mitigated response and metadata.
        """
        if force_rewrite:
            rewritten = self.rewriter.apply(prompt, response)
            return {
                "response": rewritten,
                "mitigated": True,
                "strategy": "response_rewriter",
            }

        return {
            "response": response,
            "mitigated": False,
        }

    def generate_counterfactual_tests(
        self,
        prompt: str,
        categories: list[str] | None = None,
    ) -> list[dict]:
        """Generate counterfactual test prompts."""
        categories = categories or ["gender", "race", "religion"]
        all_cfs = []
        for cat in categories:
            cfs = self.counterfactual.generate_counterfactuals(prompt, cat)
            all_cfs.extend(cfs)
        return all_cfs
