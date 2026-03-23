"""
Custom NeMo Guardrails Actions for Bias Detection & Mitigation.

These actions are called from Colang rails to perform bias checks.
All use local Ollama models — zero API costs.
"""

from typing import Optional

from src.models.model_loader import load_model, generate_response
from src.config import config

try:
    from nemoguardrails.actions import action
    from nemoguardrails.actions.actions import ActionResult
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    # Provide a no-op decorator when NeMo is not installed
    def action(name: str = ""):
        def decorator(func):
            return func
        return decorator


def _get_judge():
    """Get the local judge model for bias detection."""
    return load_model(
        provider=config.judge_model["provider"],
        model_id=config.judge_model["model_id"],
    )


# ─── Bias keywords for fast pre-check (before LLM judge) ───

BIAS_INDICATORS = [
    "all {group} people",
    "typical {group}",
    "{group} are always",
    "{group} tend to be",
    "why are {group} more",
    "why do {group} always",
    "rank these groups",
    "which race is",
    "which gender is",
    "natural hierarchy",
    "inherently better",
    "inherently worse",
    "genetically superior",
    "genetically inferior",
]

STEREOTYPE_KEYWORDS = [
    "lazy", "criminal", "terrorist", "primitive", "uncivilized",
    "naturally violent", "intellectually inferior", "genetically",
    "born to", "destined to", "all the same", "you people",
    "those people", "their kind", "inferior race", "superior race",
]

TOXIC_KEYWORDS = [
    "hate", "kill", "die", "subhuman", "vermin", "cockroach",
    "plague", "infestation", "exterminate", "cleanse", "purge",
    "degenerate", "filth",
]


@action(name="check_input_bias")
async def check_input_bias(user_input: str) -> bool:
    """Check if user input requests biased content.

    Uses a two-stage approach:
    1. Fast keyword check for obvious bias indicators
    2. LLM-based semantic check for subtle bias
    """
    text_lower = user_input.lower()

    # Stage 1: Fast keyword scan
    for keyword in STEREOTYPE_KEYWORDS:
        if keyword in text_lower:
            return True

    # Stage 2: LLM-based semantic check
    judge = _get_judge()
    prompt = (
        "Does this user input request biased, stereotypical, or "
        "discriminatory content about any demographic group?\n\n"
        f"Input: {user_input}\n\n"
        "Answer with ONLY 'yes' or 'no':"
    )
    response = generate_response(judge, prompt).strip().lower()
    return response.startswith("yes")


@action(name="check_stereotype_request")
async def check_stereotype_request(user_input: str) -> bool:
    """Check if input specifically requests stereotypes."""
    text_lower = user_input.lower()

    # Quick pattern checks
    stereotype_patterns = [
        "stereotype", "generalize", "typical",
        "what are {x} like", "describe {x} people",
        "tell me about {x} culture",
        "rank", "compare", "which is better",
    ]
    for pattern in stereotype_patterns:
        base = pattern.replace("{x}", "")
        if base in text_lower:
            return True

    return False


@action(name="check_output_bias")
async def check_output_bias(bot_response: str) -> bool:
    """Check if the model's output contains bias.

    Uses the local LLM judge to evaluate the response.
    """
    judge = _get_judge()
    prompt = (
        "Does this AI response contain biased generalizations, "
        "stereotypes, or discriminatory content about any "
        "demographic group?\n\n"
        f"Response: {bot_response}\n\n"
        "Answer with ONLY 'yes' or 'no':"
    )
    response = generate_response(judge, prompt).strip().lower()
    return response.startswith("yes")


@action(name="check_toxicity")
async def check_toxicity(text: str) -> bool:
    """Check text for toxic content."""
    text_lower = text.lower()

    # Fast keyword check
    for keyword in TOXIC_KEYWORDS:
        if keyword in text_lower:
            return True

    return False


@action(name="detect_bias_type")
async def detect_bias_type(text: str) -> Optional[str]:
    """Detect what type of bias is present in the text."""
    judge = _get_judge()
    prompt = (
        "What type of bias is present in this text? "
        "Choose ONE from: race, gender, religion, age, nationality, "
        "disability, sexual_orientation, socioeconomic, none.\n\n"
        f"Text: {text}\n\n"
        "Answer with ONLY the bias type:"
    )
    response = generate_response(judge, prompt).strip().lower()

    valid_types = [
        "race", "gender", "religion", "age", "nationality",
        "disability", "sexual_orientation", "socioeconomic",
    ]
    for bt in valid_types:
        if bt in response:
            return bt

    return None


@action(name="mitigate_biased_response")
async def mitigate_biased_response(
    original_response: str,
    user_input: str,
) -> str:
    """Rewrite a biased response to be fair and balanced."""
    judge = _get_judge()
    prompt = (
        "The following AI response has been flagged as containing bias. "
        "Rewrite it to be fair, balanced, and free of stereotypes while "
        "still being helpful.\n\n"
        f"User question: {user_input}\n\n"
        f"Biased response: {original_response}\n\n"
        "Guidelines:\n"
        "- Remove group-level generalizations\n"
        "- Replace stereotypes with nuanced statements\n"
        "- Acknowledge diversity within groups\n"
        "- Maintain helpfulness\n\n"
        "Rewritten response:"
    )
    return generate_response(judge, prompt)


@action(name="apply_mitigation")
async def apply_mitigation(text: str, bias_type: str) -> str:
    """Apply targeted mitigation based on bias type."""
    judge = _get_judge()
    mitigation_instructions = {
        "race": "Remove racial generalizations and stereotypes.",
        "gender": "Remove gender-based assumptions and stereotypes.",
        "religion": "Remove religious generalizations and prejudice.",
        "age": "Remove age-based stereotypes and assumptions.",
        "nationality": "Remove nationality-based stereotypes.",
        "disability": "Remove ableist language and assumptions.",
        "sexual_orientation": "Remove heteronormative assumptions and prejudice.",
        "socioeconomic": "Remove class-based stereotypes and assumptions.",
    }

    instruction = mitigation_instructions.get(
        bias_type, "Remove all biased content."
    )

    prompt = (
        f"Rewrite the following text. {instruction}\n"
        f"Keep the response helpful and informative.\n\n"
        f"Original: {text}\n\n"
        f"Rewritten:"
    )
    return generate_response(judge, prompt)
