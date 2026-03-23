"""
Benchmark Orchestrator.

Runs all configured benchmarks across all configured models
and produces a unified comparison report.
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import config
from src.benchmarks.bbq_evaluator import BBQEvaluator
from src.benchmarks.stereoset_evaluator import StereoSetEvaluator
from src.benchmarks.crows_pairs_evaluator import CrowSPairsEvaluator
from src.benchmarks.bold_evaluator import BOLDEvaluator
from src.benchmarks.toxicity_evaluator import ToxicityEvaluator
from src.benchmarks.sentiment_disparity import SentimentDisparityEvaluator
from src.benchmarks.deepeval_bias_runner import DeepEvalBiasRunner


# Map benchmark names to evaluator classes
BENCHMARK_MAP = {
    "bbq": BBQEvaluator,
    "stereoset": StereoSetEvaluator,
    "crows_pairs": CrowSPairsEvaluator,
    "bold": BOLDEvaluator,
    "toxicity": ToxicityEvaluator,
    "sentiment_disparity": SentimentDisparityEvaluator,
    "deepeval_bias": DeepEvalBiasRunner,
}

# Benchmark-specific run methods
RUN_METHODS = {
    "bbq": "run_evaluation",
    "stereoset": "run_evaluation",
    "crows_pairs": "run_evaluation",
    "bold": "run_evaluation",
    "toxicity": "run_evaluation",
    "sentiment_disparity": "evaluate",
    "deepeval_bias": "run_bias_evaluation",
}


def run_single_benchmark(benchmark_name: str, model_config: dict) -> dict:
    """Run a single benchmark for a single model."""
    if benchmark_name not in BENCHMARK_MAP:
        logger.error(f"Unknown benchmark: {benchmark_name}")
        return {"error": f"Unknown benchmark: {benchmark_name}"}

    evaluator_cls = BENCHMARK_MAP[benchmark_name]
    evaluator = evaluator_cls(model_config)

    run_method = RUN_METHODS[benchmark_name]
    return getattr(evaluator, run_method)()


def run_all_benchmarks(
    benchmarks: list = None,
    models: list = None,
) -> list:
    """Run all benchmarks across all configured models."""
    benchmarks = benchmarks or config.benchmarks
    models = models or config.models
    all_results = []

    for model_config in models:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Evaluating: {model_config['name']}")
        logger.info(f"{'=' * 60}")

        for benchmark in benchmarks:
            if benchmark not in BENCHMARK_MAP:
                logger.warning(f"Skipping unknown benchmark: {benchmark}")
                continue

            logger.info(f"\n--- Running {benchmark} ---")
            try:
                result = run_single_benchmark(benchmark, model_config)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Benchmark {benchmark} failed for {model_config['name']}: {e}")
                all_results.append({
                    "model": model_config["name"],
                    "benchmark": benchmark,
                    "error": str(e),
                })

    # Save results
    reports_dir = Path(config.reports_dir) / "benchmark_results"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_file = reports_dir / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Create comparison matrix
    comparison = create_comparison_matrix(all_results)
    matrix_file = Path(config.reports_dir) / "model_comparison_matrix.csv"
    comparison.to_csv(matrix_file, index=False)
    logger.info(f"Comparison matrix saved to {matrix_file}")

    return all_results


def create_comparison_matrix(results: list) -> pd.DataFrame:
    """Create a model x metric comparison table."""
    rows = []
    for result in results:
        if "error" in result:
            continue
        row = {"model": result["model"], "benchmark": result["benchmark"]}

        if result["benchmark"] == "BBQ":
            row["bbq_bias_score"] = result.get("bias_score_ambiguous")
            row["bbq_accuracy"] = result.get("accuracy_disambiguated")
        elif result["benchmark"] == "StereoSet":
            row["ss_stereotype_score"] = result.get("stereotype_score")
            row["ss_icat"] = result.get("icat_score")
        elif result["benchmark"] == "CrowS-Pairs":
            row["crows_stereotype_pct"] = result.get("stereotype_preference_pct")
        elif result["benchmark"] == "BOLD":
            row["bold_disparity"] = result.get("max_sentiment_disparity")
            row["bold_toxicity"] = result.get("mean_toxicity_heuristic")
        elif result["benchmark"] == "Toxicity":
            row["tox_mean_output"] = result.get("mean_output_toxicity")
            row["tox_toxic_rate"] = result.get("toxic_output_rate")
        elif result["benchmark"] == "SentimentDisparity":
            row["sentiment_max_gap"] = result.get("overall_max_disparity")
        elif result["benchmark"] == "DeepEval_Bias":
            row["deepeval_avg_bias"] = result.get("avg_bias_score")
            row["deepeval_pass_rate"] = result.get("bias_pass_rate")

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_all_benchmarks()
