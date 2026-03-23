"""
Evidently AI Integration for Bias Drift Detection.

Generates data quality and drift reports using Evidently AI.
Compares current bias metrics against baseline to detect
model degradation over time.
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger


class EvidentlyReporter:
    """Generates Evidently AI reports for bias monitoring."""

    def __init__(self):
        self._evidently_available = False
        try:
            from evidently.metric_preset import (  # noqa: F401
                DataDriftPreset,
                DataQualityPreset,
            )
            from evidently.report import Report  # noqa: F401
            self._evidently_available = True
        except ImportError:
            logger.warning(
                "Evidently not installed. "
                "Install with: pip install evidently"
            )

    def create_bias_drift_report(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: str = "reports/monitoring/bias_drift_report.html",
    ) -> str | None:
        """Generate a bias drift report comparing current vs baseline.

        Args:
            baseline_data: DataFrame with columns:
                [model, bias_score, toxicity_score, sentiment, bias_category]
            current_data: Same schema as baseline.
            output_path: Where to save the HTML report.

        Returns:
            Path to generated report, or None if Evidently unavailable.
        """
        if not self._evidently_available:
            logger.warning("Evidently not available, skipping drift report")
            return None

        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=baseline_data,
            current_data=current_data,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(output_path)
        logger.info(f"Drift report saved to {output_path}")
        return output_path

    def create_quality_report(
        self,
        data: pd.DataFrame,
        output_path: str = "reports/monitoring/data_quality_report.html",
    ) -> str | None:
        """Generate a data quality report for evaluation results."""
        if not self._evidently_available:
            return None

        from evidently.metric_preset import DataQualityPreset
        from evidently.report import Report

        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=None, current_data=data)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(output_path)
        logger.info(f"Quality report saved to {output_path}")
        return output_path

    def create_baseline_from_results(
        self,
        results_path: str = "reports/benchmark_results/all_results.json",
    ) -> pd.DataFrame:
        """Create a baseline DataFrame from benchmark results."""
        with open(results_path) as f:
            results = json.load(f)

        rows = []
        for r in results:
            if "error" in r:
                continue
            row = {
                "model": r.get("model", ""),
                "benchmark": r.get("benchmark", ""),
            }
            # Extract numeric metrics
            for key, val in r.items():
                if isinstance(val, (int, float)):
                    row[key] = val
            rows.append(row)

        return pd.DataFrame(rows)
