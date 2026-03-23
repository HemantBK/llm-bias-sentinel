"""Tests for prompt templates utility."""

import pytest
from src.utils.prompt_templates import (
    SYSTEM_PROMPTS,
    BIAS_PROBE_TEMPLATES,
    DEMOGRAPHIC_GROUPS,
    DIVERSE_NAMES,
    format_probe,
    generate_occupation_probes,
    generate_group_probes,
)


class TestPromptTemplates:
    def test_system_prompts(self):
        assert "neutral" in SYSTEM_PROMPTS
        assert "bias_aware" in SYSTEM_PROMPTS
        assert "strict" in SYSTEM_PROMPTS
        for name, prompt in SYSTEM_PROMPTS.items():
            assert len(prompt) > 20

    def test_bias_probe_templates(self):
        assert len(BIAS_PROBE_TEMPLATES) >= 5
        assert "occupation" in BIAS_PROBE_TEMPLATES
        assert "group_trait" in BIAS_PROBE_TEMPLATES

    def test_demographic_groups(self):
        assert "gender" in DEMOGRAPHIC_GROUPS
        assert "race" in DEMOGRAPHIC_GROUPS
        assert len(DEMOGRAPHIC_GROUPS["race"]) >= 4

    def test_diverse_names(self):
        assert "male" in DIVERSE_NAMES
        assert "female" in DIVERSE_NAMES
        assert len(DIVERSE_NAMES["male"]) >= 4

    def test_format_probe_occupation(self):
        result = format_probe("occupation", occupation="CEO")
        assert "CEO" in result
        assert "{" not in result

    def test_format_probe_comparison(self):
        result = format_probe(
            "comparison",
            skill="math",
            group_a="boys",
            group_b="girls",
        )
        assert "math" in result
        assert "boys" in result

    def test_format_probe_invalid_key(self):
        with pytest.raises(KeyError):
            format_probe("nonexistent_template")

    def test_generate_occupation_probes(self):
        probes = generate_occupation_probes()
        assert len(probes) >= 10
        assert all(isinstance(p, str) for p in probes)

    def test_generate_occupation_probes_custom(self):
        probes = generate_occupation_probes(["astronaut", "librarian"])
        assert len(probes) == 2
        assert "astronaut" in probes[0]

    def test_generate_group_probes(self):
        probes = generate_group_probes()
        assert len(probes) >= 15
        assert all("category" in p for p in probes)
        assert all("group" in p for p in probes)
        assert all("prompt" in p for p in probes)
        assert all("{" not in p["prompt"] for p in probes)
