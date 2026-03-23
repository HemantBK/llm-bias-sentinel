"""
LLM Bias Sentinel — CLI Entry Point.

Usage:
    python run.py benchmarks              Run all bias benchmarks
    python run.py benchmarks --quick      Run quick benchmark subset
    python run.py red-team                Run full red-team assessment
    python run.py red-team --quick        Run quick red-team scan
    python run.py guardrails-test         Test guardrails effectiveness
    python run.py fairness-card MODEL     Generate fairness card
    python run.py api                     Start the FastAPI server
    python run.py check "prompt text"     Quick bias check on text
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def run_benchmarks(args):
    """Run bias benchmark suite."""
    from src.benchmarks.benchmark_runner import run_all_benchmarks
    from src.config import config

    benchmarks = None
    if args.quick:
        benchmarks = ["bbq", "stereoset", "crows_pairs"]
        logger.info("Quick mode: running BBQ, StereoSet, CrowS-Pairs only")

    results = run_all_benchmarks(benchmarks=benchmarks)
    logger.info(f"Completed {len(results)} benchmark evaluations")
    return results


def run_red_team(args):
    """Run red-team adversarial assessment."""
    from src.red_team.red_team_orchestrator import RedTeamOrchestrator

    orchestrator = RedTeamOrchestrator()

    if args.quick:
        logger.info("Quick red-team scan (30 prompts, no grid probe)")
        report = orchestrator.run_quick_scan(max_prompts=30)
    else:
        report = orchestrator.run_full_assessment()

    summary = report.get("summary", {})
    for model, data in summary.items():
        risk = data.get("risk_level", "?")
        jb_rate = data.get("jailbreak_success_rate", 0)
        logger.info(f"  {model}: Risk={risk}, Jailbreak={jb_rate:.1%}")

    return report


def run_guardrails_test(args):
    """Test guardrails effectiveness."""
    from src.guardrails_app.guardrails_tester import GuardrailsTester

    tester = GuardrailsTester()
    report = tester.run_full_test()
    grade = report.get("overall_grade", "?")
    logger.info(f"Guardrails grade: {grade}")
    return report


def generate_fairness_card(args):
    """Generate a model fairness card."""
    from compliance.fairness_card import FairnessCard

    model_name = args.model or "llama3-8b"
    card_gen = FairnessCard(model_name)

    # Try to load existing results
    benchmark_results = None
    red_team_results = None
    guardrails_results = None

    benchmark_path = Path("reports/benchmark_results/all_results.json")
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            benchmark_results = json.load(f)

    card = card_gen.generate(benchmark_results, red_team_results, guardrails_results)
    card_gen.save_json(card)
    card_gen.save_markdown(card)
    logger.info(f"Fairness card generated for {model_name}")
    return card


def run_api(args):
    """Start the FastAPI server."""
    import uvicorn

    port = args.port or 8000
    logger.info(f"Starting API on port {port}...")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=args.reload,
    )


def quick_bias_check(args):
    """Quick bias check on a text string."""
    from src.guardrails_app.guardrails_engine import StandaloneGuardrails

    guardrails = StandaloneGuardrails()
    text = args.text

    input_check = guardrails.check_input(text)
    print(f"\nText: \"{text}\"")
    print(f"Input bias flagged: {input_check['flagged']}")
    if input_check.get("reason"):
        print(f"Reason: {input_check['reason']}")
    if input_check.get("keywords"):
        print(f"Keywords: {input_check['keywords']}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Bias Sentinel — Bias evaluation, red-teaming & guardrails",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # benchmarks
    bench_parser = subparsers.add_parser("benchmarks", help="Run bias benchmarks")
    bench_parser.add_argument("--quick", action="store_true", help="Run quick subset")

    # red-team
    rt_parser = subparsers.add_parser("red-team", help="Run red-team assessment")
    rt_parser.add_argument("--quick", action="store_true", help="Quick scan only")

    # guardrails-test
    subparsers.add_parser("guardrails-test", help="Test guardrails effectiveness")

    # fairness-card
    fc_parser = subparsers.add_parser("fairness-card", help="Generate fairness card")
    fc_parser.add_argument("model", nargs="?", default="llama3-8b", help="Model name")

    # api
    api_parser = subparsers.add_parser("api", help="Start FastAPI server")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.add_argument("--reload", action="store_true", default=True)

    # check
    check_parser = subparsers.add_parser("check", help="Quick bias check")
    check_parser.add_argument("text", help="Text to check for bias")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "benchmarks": run_benchmarks,
        "red-team": run_red_team,
        "guardrails-test": run_guardrails_test,
        "fairness-card": generate_fairness_card,
        "api": run_api,
        "check": quick_bias_check,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
