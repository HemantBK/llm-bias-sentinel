# Architecture

## System Overview

```
                    +-------------------+
                    |   CLI (run.py)    |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+     +-------------v-----------+
    |   FastAPI (api/)  |     |  Benchmark Runner       |
    |  10 REST endpoints|     |  7 evaluators           |
    +--------+----------+     +-------------+-----------+
             |                              |
    +--------v----------+     +-------------v-----------+
    |  Guardrails Engine|     |  Red-Team Orchestrator  |
    |  NeMo + Standalone|     |  22 attacks, 8 categories|
    +--------+----------+     +-------------+-----------+
             |                              |
    +--------v--------------------------+---v-----------+
    |           Model Loader (Ollama)                    |
    |           Local inference — zero API costs          |
    +---------------------------------------------------+
             |
    +--------v----------+
    |  Monitoring Stack  |
    |  Prometheus+Grafana|
    +-------------------+
```

## Data Flow

1. **Evaluation Flow:** CLI/API -> Benchmark Runner -> Model Loader -> Ollama -> Results JSON
2. **Red-Team Flow:** CLI/API -> Generator -> Jailbreak Tester -> Judge Model -> Report HTML
3. **Guarded Generation:** API -> Input Rail -> Model -> Output Rail -> Mitigation -> Response
4. **Monitoring Flow:** API Metrics -> Prometheus -> Grafana Dashboard

## Module Dependencies

```
src/config.py (central config, no dependencies)
    |
    +-> src/models/model_loader.py (Ollama interface)
    |       |
    |       +-> src/benchmarks/* (all 7 evaluators)
    |       +-> src/red_team/* (adversarial testing)
    |       +-> src/guardrails_app/* (input/output filtering)
    |       +-> src/image_bias/* (CLIP + Stable Diffusion)
    |
    +-> src/monitoring/* (Prometheus metrics, Evidently)
    +-> compliance/* (fairness cards, reports, EU AI Act)
    +-> api/* (FastAPI, wraps all modules)
```

## Key Design Decisions

1. **Ollama-only inference** — All models run locally, eliminating API costs and data privacy concerns
2. **Local LLM judge** — DeepEval and red-team judging use local Ollama models instead of GPT-4
3. **Standalone guardrails fallback** — Works without NeMo Guardrails installed (keyword + LLM check)
4. **Background job processing** — Long-running evaluations run as FastAPI background tasks
5. **Word-boundary counterfactuals** — Regex-based swapping prevents partial-word matches
6. **Configurable thresholds** — All fairness thresholds in YAML, enforced in CI/CD
