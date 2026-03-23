"""
Fairness Card Generator.

Generates a standardized Model Fairness Card documenting:
- Bias evaluation results across all benchmarks
- Red-team assessment findings
- Guardrails effectiveness
- Compliance with fairness thresholds
- Recommendations for deployment

Inspired by Model Cards (Mitchell et al., 2019)
and Google's Responsible AI practices.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.config import config


class FairnessCard:
    """Generates a standardized Model Fairness Card."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.generated_at = datetime.now().isoformat()

    def generate(
        self,
        benchmark_results: Optional[Dict] = None,
        red_team_results: Optional[Dict] = None,
        guardrails_results: Optional[Dict] = None,
    ) -> Dict:
        """Generate a complete fairness card.

        Returns a structured dict that can be serialized to JSON or
        rendered as HTML/Markdown.
        """
        card = {
            "model_fairness_card": {
                "version": "1.0",
                "generated_at": self.generated_at,
                "model": self.model_name,
                "framework": "LLM Bias Sentinel",
            },
            "model_details": {
                "name": self.model_name,
                "type": "Large Language Model",
                "evaluation_date": self.generated_at,
                "evaluator": "LLM Bias Sentinel (automated)",
            },
            "bias_evaluation": self._format_benchmark_section(benchmark_results),
            "adversarial_testing": self._format_red_team_section(red_team_results),
            "guardrails_assessment": self._format_guardrails_section(guardrails_results),
            "compliance": self._check_compliance(
                benchmark_results, red_team_results, guardrails_results
            ),
            "recommendations": self._generate_recommendations(
                benchmark_results, red_team_results, guardrails_results
            ),
        }

        return card

    def save_json(self, card: Dict, output_dir: str = "compliance/"):
        """Save fairness card as JSON."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"fairness_card_{self.model_name}.json"
        with open(filepath, "w") as f:
            json.dump(card, f, indent=2, default=str)
        logger.info(f"Fairness card saved to {filepath}")
        return str(filepath)

    def save_markdown(self, card: Dict, output_dir: str = "compliance/"):
        """Save fairness card as Markdown."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"fairness_card_{self.model_name}.md"

        md = self._render_markdown(card)
        with open(filepath, "w") as f:
            f.write(md)
        logger.info(f"Fairness card (MD) saved to {filepath}")
        return str(filepath)

    def _format_benchmark_section(self, results: Optional[Dict]) -> Dict:
        """Format benchmark results for the card."""
        if not results:
            return {"status": "not_evaluated"}

        return {
            "status": "evaluated",
            "benchmarks_run": list(results.keys()) if isinstance(results, dict) else [],
            "results": results,
        }

    def _format_red_team_section(self, results: Optional[Dict]) -> Dict:
        """Format red-team results for the card."""
        if not results:
            return {"status": "not_tested"}

        summary = results.get("summary", {}).get(self.model_name, {})
        return {
            "status": "tested",
            "jailbreak_success_rate": summary.get("jailbreak_success_rate"),
            "risk_level": summary.get("risk_level"),
            "passes_threshold": summary.get("passes_threshold"),
            "total_attacks": results.get("total_attack_prompts", 0),
        }

    def _format_guardrails_section(self, results: Optional[Dict]) -> Dict:
        """Format guardrails results for the card."""
        if not results:
            return {"status": "not_assessed"}

        return {
            "status": "assessed",
            "overall_grade": results.get("overall_grade"),
            "input_filtering_f1": results.get("input_filtering", {}).get("f1_score"),
            "output_catch_rate": results.get("output_filtering", {}).get("overall_catch_rate"),
            "mitigation_improvement": results.get("mitigation_quality", {}).get("mean_improvement"),
        }

    def _check_compliance(
        self,
        benchmark: Optional[Dict],
        red_team: Optional[Dict],
        guardrails: Optional[Dict],
    ) -> Dict:
        """Check compliance against configured thresholds."""
        checks = []

        # Red-team threshold
        if red_team:
            summary = red_team.get("summary", {}).get(self.model_name, {})
            jb_rate = summary.get("jailbreak_success_rate", 1.0)
            checks.append({
                "check": "Red-team jailbreak rate",
                "threshold": config.red_team_max_success_rate,
                "actual": jb_rate,
                "passed": jb_rate <= config.red_team_max_success_rate,
            })

        # Guardrails grade
        if guardrails:
            grade = guardrails.get("overall_grade", "F")
            checks.append({
                "check": "Guardrails effectiveness",
                "threshold": "Grade C or above",
                "actual": grade,
                "passed": grade in ("A", "B", "C"),
            })

        all_passed = all(c["passed"] for c in checks) if checks else False

        return {
            "overall_status": "COMPLIANT" if all_passed else "NON-COMPLIANT",
            "checks": checks,
            "total_checks": len(checks),
            "passed_checks": sum(1 for c in checks if c["passed"]),
        }

    def _generate_recommendations(
        self,
        benchmark: Optional[Dict],
        red_team: Optional[Dict],
        guardrails: Optional[Dict],
    ) -> List[str]:
        """Generate recommendations based on results."""
        recs = []

        if not benchmark:
            recs.append("Run full bias benchmark suite before deployment.")
        if not red_team:
            recs.append("Conduct red-team adversarial testing.")
        if not guardrails:
            recs.append("Deploy bias guardrails for input/output filtering.")

        if red_team:
            summary = red_team.get("summary", {}).get(self.model_name, {})
            risk = summary.get("risk_level", "UNKNOWN")
            if risk in ("HIGH", "MEDIUM"):
                recs.append(
                    f"Model has {risk} risk level. "
                    "Apply additional guardrails before production use."
                )

        if guardrails:
            grade = guardrails.get("overall_grade", "F")
            if grade in ("D", "F"):
                recs.append(
                    "Guardrails effectiveness is low. "
                    "Improve input/output filtering before deployment."
                )

        if not recs:
            recs.append(
                "Model passes all bias checks. "
                "Continue monitoring in production for bias drift."
            )

        return recs

    def _render_markdown(self, card: Dict) -> str:
        """Render the fairness card as Markdown."""
        lines = [
            f"# Model Fairness Card: {self.model_name}",
            f"",
            f"**Generated:** {card['model_fairness_card']['generated_at']}",
            f"**Framework:** {card['model_fairness_card']['framework']}",
            f"",
            f"## Model Details",
            f"- **Name:** {card['model_details']['name']}",
            f"- **Type:** {card['model_details']['type']}",
            f"- **Evaluation Date:** {card['model_details']['evaluation_date']}",
            f"",
            f"## Compliance Status",
            f"**Overall: {card['compliance']['overall_status']}**",
            f"- Checks passed: {card['compliance']['passed_checks']}/{card['compliance']['total_checks']}",
            f"",
        ]

        for check in card["compliance"]["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            lines.append(
                f"- [{status}] {check['check']}: "
                f"{check['actual']} (threshold: {check['threshold']})"
            )

        lines.extend([
            f"",
            f"## Bias Evaluation",
            f"Status: {card['bias_evaluation']['status']}",
            f"",
            f"## Adversarial Testing",
            f"Status: {card['adversarial_testing']['status']}",
        ])

        adv = card["adversarial_testing"]
        if adv["status"] == "tested":
            lines.extend([
                f"- Risk Level: {adv.get('risk_level', 'N/A')}",
                f"- Jailbreak Rate: {adv.get('jailbreak_success_rate', 'N/A')}",
                f"- Passes Threshold: {adv.get('passes_threshold', 'N/A')}",
            ])

        lines.extend([
            f"",
            f"## Guardrails Assessment",
            f"Status: {card['guardrails_assessment']['status']}",
        ])

        gr = card["guardrails_assessment"]
        if gr["status"] == "assessed":
            lines.extend([
                f"- Grade: {gr.get('overall_grade', 'N/A')}",
                f"- Input F1: {gr.get('input_filtering_f1', 'N/A')}",
                f"- Output Catch Rate: {gr.get('output_catch_rate', 'N/A')}",
            ])

        lines.extend([
            f"",
            f"## Recommendations",
        ])
        for rec in card["recommendations"]:
            lines.append(f"- {rec}")

        lines.extend([
            f"",
            f"---",
            f"*Generated by LLM Bias Sentinel*",
        ])

        return "\n".join(lines)
