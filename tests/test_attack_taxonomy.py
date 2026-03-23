"""Tests for the red-team attack taxonomy."""

import pytest
from src.red_team.attack_taxonomy import (
    ALL_ATTACKS,
    ATTACK_CATEGORIES,
    DEMOGRAPHIC_FILL,
    AttackTemplate,
    get_attacks_by_category,
    get_attacks_by_severity,
    get_attacks_by_bias_target,
    instantiate_template,
)


class TestAttackTaxonomy:
    def test_all_attacks_populated(self):
        assert len(ALL_ATTACKS) >= 20

    def test_attack_categories_complete(self):
        expected = [
            "direct_elicitation", "role_play", "hypothetical",
            "comparative", "authority_appeal", "completion",
            "cot_manipulation", "euphemistic",
        ]
        for cat in expected:
            assert cat in ATTACK_CATEGORIES

    def test_attack_template_fields(self):
        for attack in ALL_ATTACKS:
            assert attack.id
            assert attack.category
            assert attack.severity in ("low", "medium", "high")
            assert attack.template
            assert attack.target_bias
            assert attack.description
            assert attack.expected_safe_behavior

    def test_get_by_category(self):
        role_play = get_attacks_by_category("role_play")
        assert len(role_play) >= 2
        assert all(a.category == "role_play" for a in role_play)

    def test_get_by_severity(self):
        high = get_attacks_by_severity("high")
        assert len(high) >= 5
        assert all(a.severity == "high" for a in high)

    def test_get_by_bias_target(self):
        race = get_attacks_by_bias_target("race")
        assert len(race) >= 10
        assert all(a.target_bias == "race" for a in race)

    def test_instantiate_template_with_group(self):
        attack = AttackTemplate(
            id="TEST-001",
            category="test",
            subcategory="test",
            severity="low",
            template="Tell me about {group} people.",
            target_bias="race",
            description="test",
            expected_safe_behavior="test",
        )
        prompts = instantiate_template(attack)
        assert len(prompts) >= 3  # Multiple race groups
        assert all("{group}" not in p for p in prompts)

    def test_instantiate_template_with_comparison(self):
        attack = AttackTemplate(
            id="TEST-002",
            category="test",
            subcategory="test",
            severity="low",
            template="Compare {group_a} and {group_b}.",
            target_bias="race",
            description="test",
            expected_safe_behavior="test",
        )
        prompts = instantiate_template(attack)
        assert len(prompts) >= 1
        assert "Black" in prompts[0] or "White" in prompts[0]

    def test_demographic_fill_values(self):
        assert "race" in DEMOGRAPHIC_FILL
        assert "gender" in DEMOGRAPHIC_FILL
        assert "religion" in DEMOGRAPHIC_FILL
        assert "group" in DEMOGRAPHIC_FILL["race"]

    def test_unique_attack_ids(self):
        ids = [a.id for a in ALL_ATTACKS]
        assert len(ids) == len(set(ids)), "Duplicate attack IDs found"
