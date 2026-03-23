"""
LLM Bias Sentinel — FastAPI Application.

Provides REST endpoints for:
- Running bias benchmarks
- Red-team adversarial testing
- Guarded text generation with bias mitigation
- Bias checking for arbitrary text
- Counterfactual fairness analysis
- Prometheus metrics export
- Health checks

All inference runs locally via Ollama — zero API costs.
"""

import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    BenchmarkReport,
    BiasCheckRequest,
    BiasCheckResponse,
    CounterfactualRequest,
    CounterfactualResponse,
    GuardedGenerateRequest,
    GuardedResponse,
    HealthResponse,
    JobResponse,
    JobStatus,
    MetricsResponse,
    RedTeamReport,
    RunBenchmarkRequest,
    RunRedTeamRequest,
)
from src.config import config

# ─── Prometheus Metrics ──────────────────────────

EVAL_COUNTER = Counter(
    "bias_evaluations_total",
    "Total bias evaluations run",
    ["benchmark", "model"],
)
RED_TEAM_COUNTER = Counter(
    "red_team_runs_total",
    "Total red-team assessment runs",
)
GUARDED_GEN_COUNTER = Counter(
    "guarded_generations_total",
    "Total guarded text generations",
)
BIAS_FLAGS_COUNTER = Counter(
    "bias_flags_total",
    "Total bias flags triggered",
    ["stage"],  # "input" or "output"
)
EVAL_DURATION = Histogram(
    "evaluation_duration_seconds",
    "Duration of evaluation runs",
    ["benchmark"],
)
ACTIVE_JOBS = Gauge("active_jobs", "Number of currently running jobs")

# ─── In-Memory Job Store ─────────────────────────

jobs: dict[str, dict] = {}
START_TIME = time.time()


# ─── Lifespan ────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("LLM Bias Sentinel API starting up...")
    logger.info(f"Ollama URL: {config.ollama_base_url}")
    yield
    logger.info("API shutting down...")


# ─── App ─────────────────────────────────────────

app = FastAPI(
    title="LLM Bias Sentinel",
    description=(
        "Comprehensive bias evaluation, red-teaming, and guardrails "
        "for LLMs. 100% local inference via Ollama — zero API costs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health & Metrics ────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and Ollama connectivity."""
    ollama_ok = False
    models = []
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{config.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                ollama_ok = True
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        ollama_connected=ollama_ok,
        models_available=models,
    )


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/stats", response_model=MetricsResponse, tags=["System"])
async def get_stats():
    """Get API usage statistics."""
    return MetricsResponse(
        total_evaluations=int(EVAL_COUNTER._value.sum() if hasattr(EVAL_COUNTER, "_value") else 0),
        total_red_team_runs=int(
            RED_TEAM_COUNTER._value.get() if hasattr(RED_TEAM_COUNTER, "_value") else 0
        ),
        total_guarded_generations=int(
            GUARDED_GEN_COUNTER._value.get() if hasattr(GUARDED_GEN_COUNTER, "_value") else 0
        ),
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


# ─── Benchmark Evaluation ───────────────────────


def _run_benchmarks_job(job_id: str, request: RunBenchmarkRequest):
    """Background job: run bias benchmarks."""
    from src.benchmarks.benchmark_runner import run_single_benchmark

    jobs[job_id]["status"] = JobStatus.RUNNING
    jobs[job_id]["started_at"] = datetime.now()
    ACTIVE_JOBS.inc()

    results = []
    try:
        for model_cfg in request.models:
            model_dict = model_cfg.model_dump()
            for benchmark in request.benchmarks:
                with EVAL_DURATION.labels(benchmark=benchmark.value).time():
                    result = run_single_benchmark(benchmark.value, model_dict)
                    results.append(result)
                    EVAL_COUNTER.labels(
                        benchmark=benchmark.value,
                        model=model_cfg.name,
                    ).inc()

        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["results"] = results
    except Exception as e:
        logger.error(f"Benchmark job {job_id} failed: {e}")
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
    finally:
        jobs[job_id]["completed_at"] = datetime.now()
        ACTIVE_JOBS.dec()


@app.post("/evaluate", response_model=JobResponse, tags=["Evaluation"])
async def run_benchmarks(
    request: RunBenchmarkRequest,
    background_tasks: BackgroundTasks,
):
    """Launch bias benchmark evaluation (runs in background)."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "type": "benchmark",
        "status": JobStatus.PENDING,
        "request": request.model_dump(),
        "results": [],
    }
    background_tasks.add_task(_run_benchmarks_job, job_id, request)
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Benchmark evaluation started. Poll /evaluate/{job_id} for results.",
    )


@app.get("/evaluate/{job_id}", response_model=BenchmarkReport, tags=["Evaluation"])
async def get_benchmark_results(job_id: str):
    """Get benchmark evaluation results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return BenchmarkReport(
        job_id=job_id,
        status=job["status"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        results=job.get("results", []),
    )


# ─── Red-Team Testing ───────────────────────────


def _run_red_team_job(job_id: str, request: RunRedTeamRequest):
    """Background job: run red-team assessment."""
    from src.red_team.red_team_orchestrator import RedTeamOrchestrator

    jobs[job_id]["status"] = JobStatus.RUNNING
    jobs[job_id]["started_at"] = datetime.now()
    ACTIVE_JOBS.inc()
    RED_TEAM_COUNTER.inc()

    try:
        models = [m.model_dump() for m in request.models]
        orchestrator = RedTeamOrchestrator(target_models=models)
        report = orchestrator.run_full_assessment(
            include_taxonomy=request.include_taxonomy,
            include_mutations=request.include_mutations,
            include_llm_generation=request.include_llm_generation,
            include_grid_probe=request.include_grid_probe,
            max_attack_prompts=request.max_attack_prompts,
        )
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["report"] = report
    except Exception as e:
        logger.error(f"Red-team job {job_id} failed: {e}")
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
    finally:
        jobs[job_id]["completed_at"] = datetime.now()
        ACTIVE_JOBS.dec()


@app.post("/red-team", response_model=JobResponse, tags=["Red Team"])
async def run_red_team(
    request: RunRedTeamRequest,
    background_tasks: BackgroundTasks,
):
    """Launch red-team adversarial assessment (runs in background)."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "type": "red_team",
        "status": JobStatus.PENDING,
        "request": request.model_dump(),
    }
    background_tasks.add_task(_run_red_team_job, job_id, request)
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Red-team assessment started. Poll /red-team/{job_id} for results.",
    )


@app.get("/red-team/{job_id}", response_model=RedTeamReport, tags=["Red Team"])
async def get_red_team_results(job_id: str):
    """Get red-team assessment results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    report = job.get("report", {})

    risk_levels = {}
    summary = report.get("summary", {})
    for model, data in summary.items():
        risk_levels[model] = data.get("risk_level", "UNKNOWN")

    return RedTeamReport(
        job_id=job_id,
        status=job["status"],
        models_tested=report.get("models_tested", []),
        total_attacks=report.get("total_attack_prompts", 0),
        summary=summary,
        risk_levels=risk_levels,
    )


# ─── Guarded Generation ─────────────────────────


@app.post("/generate", response_model=GuardedResponse, tags=["Guardrails"])
async def guarded_generate(request: GuardedGenerateRequest):
    """Generate text with bias guardrails applied."""
    GUARDED_GEN_COUNTER.inc()

    if request.use_guardrails:
        from src.guardrails_app.guardrails_engine import StandaloneGuardrails

        guardrails = StandaloneGuardrails(
            {
                "provider": "ollama",
                "model_id": request.model_id,
            }
        )
        result = guardrails.guarded_generate(request.prompt)

        if result.get("input_flagged"):
            BIAS_FLAGS_COUNTER.labels(stage="input").inc()
        if result.get("output_flagged"):
            BIAS_FLAGS_COUNTER.labels(stage="output").inc()

        return GuardedResponse(
            response=result["response"],
            input_flagged=result.get("input_flagged", False),
            output_flagged=result.get("output_flagged", False),
            mitigated=result.get("mitigated", False),
            original_response=result.get("original_response"),
        )
    else:
        from src.models.model_loader import generate_response, load_model

        model = load_model(provider="ollama", model_id=request.model_id)
        response = generate_response(model, request.prompt)
        return GuardedResponse(response=response)


# ─── Bias Check ──────────────────────────────────


@app.post("/check-bias", response_model=BiasCheckResponse, tags=["Guardrails"])
async def check_bias(request: BiasCheckRequest):
    """Check text for bias (input-side, output-side, or both)."""
    from src.guardrails_app.guardrails_engine import StandaloneGuardrails

    guardrails = StandaloneGuardrails()
    result = BiasCheckResponse(text=request.text)

    if request.check_type in ("input", "both"):
        input_check = guardrails.check_input(request.text)
        result.input_check = input_check
        if input_check["flagged"]:
            result.overall_flagged = True

    if request.check_type in ("output", "both"):
        output_check = guardrails.check_output(request.text)
        result.output_check = output_check
        if output_check["flagged"]:
            result.overall_flagged = True

    return result


# ─── Counterfactual Fairness ─────────────────────


@app.post("/counterfactual", response_model=CounterfactualResponse, tags=["Analysis"])
async def generate_counterfactuals(request: CounterfactualRequest):
    """Generate counterfactual versions of a prompt for fairness testing."""
    from src.guardrails_app.mitigation import MitigationPipeline

    pipeline = MitigationPipeline()
    cfs = pipeline.generate_counterfactual_tests(request.prompt, request.categories)

    return CounterfactualResponse(
        original=request.prompt,
        counterfactuals=cfs,
        total_generated=len(cfs),
    )


# ─── Job Management ─────────────────────────────


@app.get("/jobs", tags=["System"])
async def list_jobs():
    """List all evaluation jobs."""
    return {
        job_id: {
            "type": job.get("type"),
            "status": job["status"],
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }
        for job_id, job in jobs.items()
    }


@app.delete("/jobs/{job_id}", tags=["System"])
async def delete_job(job_id: str):
    """Delete a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete running job")
    del jobs[job_id]
    return {"message": f"Job {job_id} deleted"}
