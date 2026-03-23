"""Tests for the mitigation strategies module."""

from src.guardrails_app.mitigation import (
    CalibrationMitigation,
    CounterfactualAugmenter,
    SystemPromptMitigation,
)


class TestSystemPromptMitigation:
    def test_balanced_prompt(self):
        sp = SystemPromptMitigation("balanced")
        result = sp.apply("test", "")
        assert "balanced" in result.lower() or "fair" in result.lower()
        assert len(result) > 50

    def test_strict_prompt(self):
        sp = SystemPromptMitigation("strict")
        result = sp.apply("test", "")
        assert "never" in result.lower() or "must" in result.lower()

    def test_educational_prompt(self):
        sp = SystemPromptMitigation("educational")
        result = sp.apply("test", "")
        assert "educational" in result.lower() or "evidence" in result.lower()

    def test_fallback_to_balanced(self):
        sp = SystemPromptMitigation("nonexistent")
        result = sp.apply("test", "")
        assert len(result) > 50  # Falls back to balanced


class TestCounterfactualAugmenter:
    def test_gender_swap(self):
        cf = CounterfactualAugmenter()
        results = cf.generate_counterfactuals("The man went to work", "gender")
        assert len(results) >= 1
        assert any("woman" in r["counterfactual"] for r in results)

    def test_race_swap(self):
        cf = CounterfactualAugmenter()
        results = cf.generate_counterfactuals("The White candidate was selected", "race")
        assert len(results) >= 2
        swapped_terms = [r["swapped"] for r in results]
        assert any("Black" in s for s in swapped_terms)

    def test_religion_swap(self):
        cf = CounterfactualAugmenter()
        results = cf.generate_counterfactuals("The Christian community gathered", "religion")
        assert len(results) >= 1

    def test_no_match_returns_empty(self):
        cf = CounterfactualAugmenter()
        results = cf.generate_counterfactuals("The weather is nice today", "gender")
        assert len(results) == 0

    def test_word_boundary_respected(self):
        cf = CounterfactualAugmenter()
        results = cf.generate_counterfactuals("He is a hero", "gender")
        # "he" in "hero" should NOT be swapped
        for r in results:
            assert "shero" not in r["counterfactual"]

    def test_case_preservation(self):
        cf = CounterfactualAugmenter()
        results = cf.generate_counterfactuals("He went to the store", "gender")
        if results:
            assert results[0]["counterfactual"].startswith("She")


class TestCalibrationMitigation:
    def test_low_bias_params(self):
        cal = CalibrationMitigation()
        params = cal.get_params("low_bias")
        assert params["temperature"] <= 0.2
        assert params["repeat_penalty"] > 1.0

    def test_balanced_params(self):
        cal = CalibrationMitigation()
        params = cal.get_params("balanced")
        assert 0.2 <= params["temperature"] <= 0.5

    def test_default_params(self):
        cal = CalibrationMitigation()
        params = cal.get_params("default")
        assert params["temperature"] >= 0.5

    def test_unknown_falls_back(self):
        cal = CalibrationMitigation()
        params = cal.get_params("unknown_level")
        assert "temperature" in params
