"""Pydantic request/response models for the Bias Sentinel API."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ─── Enums ───────────────────────────────────────


class BenchmarkName(str, Enum):
    BBQ = "bbq"
    STEREOSET = "stereoset"
    CROWS_PAIRS = "crows_pairs"
    BOLD = "bold"
    TOXICITY = "toxicity"
    SENTIMENT_DISPARITY = "sentiment_disparity"
    DEEPEVAL_BIAS = "deepeval_bias"


class BiasCategory(str, Enum):
    RACE = "race"
    GENDER = "gender"
    AGE = "age"
    RELIGION = "religion"
    NATIONALITY = "nationality"
    DISABILITY = "disability"
    SEXUAL_ORIENTATION = "sexual_orientation"
    SOCIOECONOMIC = "socioeconomic_status"
    POLITICAL = "political"


class RiskLevel(str, Enum):
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ─── Request Models ──────────────────────────────


class ModelConfig(BaseModel):
    name: str = Field(..., description="Display name for the model")
    provider: str = Field(default="ollama", description="Model provider")
    model_id: str = Field(..., description="Ollama model ID")
    temperature: float = Field(default=0.0, ge=0, le=2)


class RunBenchmarkRequest(BaseModel):
    models: list[ModelConfig] = Field(
        default_factory=lambda: [ModelConfig(name="llama3-8b", model_id="llama3")]
    )
    benchmarks: list[BenchmarkName] = Field(
        default_factory=lambda: [BenchmarkName.BBQ, BenchmarkName.STEREOSET]
    )
    max_samples: int = Field(default=100, ge=10, le=5000)


class RunRedTeamRequest(BaseModel):
    models: list[ModelConfig] = Field(
        default_factory=lambda: [ModelConfig(name="llama3-8b", model_id="llama3")]
    )
    include_taxonomy: bool = True
    include_mutations: bool = True
    include_llm_generation: bool = False
    include_grid_probe: bool = False
    max_attack_prompts: int = Field(default=50, ge=10, le=500)


class GuardedGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    model_id: str = Field(default="llama3")
    use_guardrails: bool = Field(default=True)


class BiasCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    check_type: str = Field(
        default="both",
        description="'input', 'output', or 'both'",
    )


class CounterfactualRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    categories: list[str] = Field(default_factory=lambda: ["gender", "race", "religion"])


# ─── Response Models ─────────────────────────────


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    ollama_connected: bool = False
    models_available: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str = ""


class BenchmarkResultSummary(BaseModel):
    model: str
    benchmark: str
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    passed: bool | None = None


class BenchmarkReport(BaseModel):
    job_id: str
    status: JobStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    summary: list[BenchmarkResultSummary] = Field(default_factory=list)


class RedTeamReport(BaseModel):
    job_id: str
    status: JobStatus
    models_tested: list[str] = Field(default_factory=list)
    total_attacks: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)
    risk_levels: dict[str, RiskLevel] = Field(default_factory=dict)


class GuardedResponse(BaseModel):
    response: str
    input_flagged: bool = False
    output_flagged: bool = False
    mitigated: bool = False
    original_response: str | None = None


class BiasCheckResponse(BaseModel):
    text: str
    input_check: dict[str, Any] | None = None
    output_check: dict[str, Any] | None = None
    overall_flagged: bool = False
    bias_type: str | None = None


class CounterfactualResponse(BaseModel):
    original: str
    counterfactuals: list[dict[str, str]] = Field(default_factory=list)
    total_generated: int = 0


class MetricsResponse(BaseModel):
    total_evaluations: int = 0
    total_red_team_runs: int = 0
    total_guarded_generations: int = 0
    bias_flags_triggered: int = 0
    uptime_seconds: float = 0
