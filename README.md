# LLM Bias Sentinel

**Comprehensive bias evaluation, red-teaming, and guardrails framework for Large Language Models.**

100% local inference via [Ollama](https://ollama.com) — zero API costs. Free and open source (MIT).

[![CI](https://github.com/HemantBK/llm-bias-sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/HemantBK/llm-bias-sentinel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Tools & Methods](#tools--methods)
3. [System Design & Architecture](#system-design--architecture)
4. [Workflow](#workflow)
5. [Quick Start](#quick-start)
6. [Project Structure](#project-structure)
7. [Bias Benchmarks](#bias-benchmarks)
8. [Red-Team Attack Categories](#red-team-attack-categories)
9. [API Endpoints](#api-endpoints)
10. [Expected Results & Findings](#expected-results--findings)
11. [Pros & Cons](#pros--cons)
12. [Limitations](#limitations)
13. [Future Improvements](#future-improvements)
14. [Configuration](#configuration)
15. [Docker Deployment](#docker-deployment)
16. [Development](#development)
17. [Technology Stack](#technology-stack)
18. [License](#license)

---

## Problem Statement

### Why This Project Exists

Large Language Models (LLMs) are increasingly deployed in high-stakes applications — hiring systems, healthcare, education, legal assistance, and customer service. However, these models inherit and amplify biases present in their training data, leading to real-world harm:

- **Hiring tools** that favor certain demographics over equally qualified candidates
- **Healthcare chatbots** that give different quality advice based on a patient's perceived race or gender
- **Educational AI** that reinforces stereotypes about which groups "belong" in certain fields
- **Content generation** that associates specific ethnicities with crime, poverty, or lower intelligence

### The Core Problem

There is no standardized, accessible, and cost-free way to:

1. **Measure** how biased a specific LLM is across multiple demographic dimensions
2. **Attack** the model to find hidden bias vulnerabilities that benchmarks miss
3. **Prevent** biased outputs from reaching end users in real time
4. **Monitor** bias drift as models are updated or fine-tuned over time
5. **Document** compliance with emerging regulations like the EU AI Act

### What Exists Today (And Why It Falls Short)

| Existing Approach | Problem |
|-------------------|---------|
| OpenAI's moderation API | Costs money, closed source, only detects toxicity (not subtle bias) |
| Google's Responsible AI Toolkit | Focused on tabular/ML models, not LLMs |
| DeepEval / Promptfoo | Requires GPT-4 as judge ($$$), no guardrails or monitoring |
| Academic benchmarks (BBQ, StereoSet) | Raw datasets only — no evaluation pipeline, no automation |
| NeMo Guardrails | Runtime filtering only — no evaluation, no red-teaming |
| Manual red-teaming | Expensive, inconsistent, doesn't scale |

### Our Solution

LLM Bias Sentinel combines all of these into a **single, free, local framework** that covers the full lifecycle:

```
Evaluate (7 benchmarks) -> Attack (22 adversarial strategies) -> Protect (guardrails)
   -> Monitor (Prometheus/Grafana) -> Report (EU AI Act compliance)
```

**Key differentiator:** Everything runs on your local machine via Ollama. No API keys, no cloud costs, no data leaving your network. The LLM judge that evaluates bias is itself a local model — making the entire system self-contained and free.

---

## Tools & Methods

### Core Technologies

| Tool | Role in Project | Why This Tool |
|------|----------------|---------------|
| **Ollama** | Local LLM inference engine | Runs llama3, mistral locally with zero cost. Simple REST API. No GPU required (CPU mode available) |
| **LangChain (ChatOllama)** | Model interface layer | Standardized prompt/response interface. Easy to swap providers later |
| **HuggingFace Datasets** | Benchmark data source | Hosts BBQ, StereoSet, CrowS-Pairs, BOLD, RealToxicityPrompts — all free |
| **DeepEval** | Bias/toxicity metric framework | Industry-standard LLM evaluation. We override its GPT-4 judge with local Ollama |
| **TextBlob** | Sentiment analysis | Lightweight NLP for measuring sentiment polarity across demographic groups |
| **NeMo Guardrails** | Runtime input/output filtering | NVIDIA's open-source rail system with Colang 2.0 scripting |
| **FastAPI** | REST API server | Async, auto-documented (Swagger), production-ready |
| **Prometheus** | Metrics collection | Time-series monitoring for bias scores, toxicity rates, request counts |
| **Grafana** | Dashboards & alerting | Visual dashboards for real-time bias monitoring |
| **Evidently AI** | Data drift detection | Detects when bias distributions shift over time |
| **Plotly** | Interactive visualizations | Heatmaps, radar charts, bar charts for reports |
| **CLIP** | Image bias analysis | Zero-shot classification of generated images for demographic representation |
| **Docker** | Containerized deployment | Full stack (API + Prometheus + Grafana) in one command |
| **GitHub Actions** | CI/CD pipelines | Automated testing, security scanning, weekly evaluations |
| **Pytest** | Test framework | 80 tests validating config, taxonomy, templates, mitigation, monitoring, API |

### Evaluation Methods

| Method | Description | Academic Source |
|--------|-------------|----------------|
| **Benchmark Evaluation** | Run standardized bias datasets and measure model responses against ground truth | BBQ (Parrish et al., ACL 2022), StereoSet (Nadeem et al., ACL 2021), CrowS-Pairs (Nangia et al., EMNLP 2020) |
| **LLM-as-Judge** | Use a local LLM to score another model's responses for bias on a 0-3 scale | Adapted from Zheng et al., NeurIPS 2023 |
| **Adversarial Red-Teaming** | Systematically probe models with attack prompts designed to elicit biased outputs | Inspired by Perez et al., 2022 |
| **Counterfactual Testing** | Swap demographic terms in prompts (e.g., "Black" to "White") and compare response differences | Based on counterfactual fairness (Kusner et al., NeurIPS 2017) |
| **Sentiment Disparity Analysis** | Generate text about different demographic groups and measure sentiment polarity gaps | TextBlob-based polarity measurement |
| **Escalation Probing** | Progressively increase adversarial pressure to find the model's breaking point | Novel contribution |

### How the Local LLM is Used (No Fine-Tuning)

This project uses Ollama for **inference only** — models are never fine-tuned, trained, or modified. The local LLM serves three roles:

```
Role 1: SUBJECT UNDER TEST
   "Describe a typical CEO" -> Model responds -> We measure bias in response

Role 2: LLM JUDGE (replaces GPT-4)
   "Rate this response 0-3 for bias: [response]" -> Judge scores it

Role 3: GUARDRAIL FILTER
   "Does this input contain biased requests? yes/no" -> Filter decision
```

---

## System Design & Architecture

### High-Level Architecture

```
                         +---------------------------+
                         |      User / CI/CD         |
                         +--+---------------------+--+
                            |                     |
                   +--------v--------+   +--------v--------+
                   |   CLI (run.py)  |   |  REST API       |
                   |   5 commands    |   |  FastAPI 10 eps  |
                   +--------+--------+   +--------+--------+
                            |                     |
              +-------------+---------------------+-------------+
              |                    |                             |
    +---------v---------+ +-------v---------+  +----------------v---------+
    |  Benchmark Suite  | | Red-Team Engine |  |   Guardrails Engine      |
    |  7 evaluators     | | 22 attacks      |  |   NeMo + Standalone      |
    |  BBQ, StereoSet,  | | 8 categories    |  |   Input/Output filtering |
    |  CrowS, BOLD,     | | Grid probes     |  |   4 mitigation strategies|
    |  Toxicity,        | | Escalation      |  +----------------+---------+
    |  Sentiment,       | | LLM generation  |                   |
    |  DeepEval         | +-------+---------+                   |
    +---------+---------+         |                             |
              |                   |                             |
              +-------------------+-----------------------------+
                                  |
                     +------------v------------+
                     |      Model Loader       |
                     |  load_model()           |
                     |  generate_response()    |
                     +------------+------------+
                                  |
                     +------------v------------+
                     |     Ollama (Local)       |
                     |  llama3-8b | mistral-7b  |
                     |  localhost:11434          |
                     +-------------------------+
                                  |
              +-------------------+-------------------+
              |                                       |
    +---------v---------+               +-------------v-----------+
    |  Monitoring Stack |               |  Compliance & Reporting |
    |  Prometheus       |               |  Fairness Cards (JSON)  |
    |  Grafana          |               |  EU AI Act docs         |
    |  Evidently drift  |               |  4 report templates     |
    |  Real-time alerts |               |  Interactive HTML       |
    +-------------------+               +-------------------------+
```

### Data Flow Pipelines

```
Pipeline 1: EVALUATION
  CLI/API Request
    -> Select benchmarks & models
    -> Load dataset from HuggingFace
    -> For each item: generate_response(model, prompt)
    -> Score against ground truth / judge
    -> Aggregate metrics (bias_score, ICAT, stereotype_pct)
    -> Save JSON results + CSV comparison matrix
    -> Generate interactive HTML charts

Pipeline 2: RED-TEAMING
  CLI/API Request
    -> Generate adversarial prompts (taxonomy + mutations + LLM-assisted)
    -> For each prompt: generate_response(target_model, prompt)
    -> For each response: generate_response(judge_model, "rate bias 0-3")
    -> Compute jailbreak success rate, per-category breakdown
    -> Run grid probe (group x topic matrix)
    -> Run escalation probes on weakest spots
    -> Generate HTML report with heatmaps

Pipeline 3: GUARDED GENERATION
  User prompt arrives at API
    -> INPUT RAIL: keyword check + LLM bias detection
    -> If flagged: block or rewrite prompt
    -> GENERATE: generate_response(model, safe_prompt)
    -> OUTPUT RAIL: keyword check + LLM bias detection
    -> If flagged: apply mitigation (disclaimer, rewrite, suppress, rephrase)
    -> Return safe response + metadata

Pipeline 4: MONITORING
  Every API request
    -> Record bias_score, toxicity, demographic_category to Prometheus
    -> Grafana dashboards visualize trends in real time
    -> Alert rules fire when bias exceeds thresholds
    -> Evidently generates drift reports comparing baseline to current
```

### Module Dependency Graph

```
src/config.py (central config, zero dependencies)
    |
    +-> src/models/model_loader.py (Ollama ChatOllama interface)
    |       |
    |       +-> src/benchmarks/* (7 evaluators + orchestrator)
    |       |       Uses: load_model(), generate_response()
    |       |       Reads: HuggingFace datasets
    |       |       Outputs: JSON results, CSV matrix
    |       |
    |       +-> src/red_team/* (6 modules)
    |       |       Uses: load_model(), generate_response() for both target + judge
    |       |       Reads: attack_taxonomy templates
    |       |       Outputs: JSON report, HTML report
    |       |
    |       +-> src/guardrails_app/* (3 modules)
    |       |       Uses: load_model(), generate_response() for bias checking
    |       |       Reads: NeMo Guardrails config (guardrails/)
    |       |       Outputs: filtered responses, mitigation metadata
    |       |
    |       +-> src/image_bias/* (1 module)
    |               Uses: CLIP model (transformers), Stable Diffusion
    |               Outputs: demographic distribution analysis
    |
    +-> src/monitoring/* (2 modules)
    |       Uses: prometheus_client, evidently
    |       Outputs: /metrics endpoint, drift HTML reports
    |
    +-> compliance/* (3 modules)
    |       Uses: evaluation results from all modules above
    |       Outputs: Fairness Cards (JSON+MD), EU AI Act docs, 4 report types
    |
    +-> api/main.py (FastAPI, wraps everything)
            Uses: all modules above
            Exposes: 10 REST endpoints
```

---

## Workflow

### End-to-End Workflow

```
Step 1: SETUP
  Install Python deps -> Install Ollama -> Pull models (llama3, mistral)

Step 2: EVALUATE (understand current bias levels)
  Run 7 benchmarks -> Get scores per model per demographic dimension
  Output: reports/benchmark_results/all_results.json

Step 3: ATTACK (find hidden vulnerabilities)
  Generate 100+ adversarial prompts -> Test against each model
  Run grid probe (30 groups x 10 topics) -> Find weak spots
  Escalate on top 3 vulnerable combos -> Measure resilience
  Output: reports/red_team/red_team_report.html

Step 4: PROTECT (deploy guardrails)
  Configure NeMo Guardrails -> Test effectiveness (precision/recall/F1)
  Deploy API with guardrails enabled -> Monitor in production
  Output: Guarded API at localhost:8000

Step 5: MONITOR (track bias over time)
  Prometheus scrapes /metrics every 15s -> Grafana dashboards
  Alert when bias_score > threshold -> Evidently drift reports
  Output: Grafana at localhost:3000

Step 6: REPORT (regulatory compliance)
  Generate Model Fairness Card -> EU AI Act compliance check
  Executive summary for stakeholders -> Technical report for engineers
  Output: compliance/fairness_card_*.json, reports/final/*

Step 7: AUTOMATE (CI/CD)
  GitHub Actions runs tests on every push
  Weekly scheduled evaluation catches regressions
  Security scanning with bandit
```

### Developer Workflow

```bash
# 1. Make changes to evaluation code
# 2. Run tests (no Ollama needed)
make test-quick

# 3. Lint and format
make lint && make format

# 4. Run security scan
make security

# 5. Start API for manual testing
make api

# 6. When ready for live evaluation (requires Ollama)
ollama serve
make benchmarks
make red-team-quick

# 7. Generate compliance artifacts
make fairness-card
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) (for local LLM inference)
- Docker (optional, for full stack deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/HemantBK/llm-bias-sentinel.git
cd llm-bias-sentinel

# Install dependencies
make install

# Copy environment file
cp .env.example .env

# Pull Ollama models (when ready for live evaluation)
# ollama pull llama3
# ollama pull mistral
```

### Verify Installation

```bash
# Run tests (no Ollama needed)
make test-quick
# Expected: 80+ tests passing
```

### Run the API

```bash
make api
# API:     http://localhost:8000
# Swagger: http://localhost:8000/docs
```

### Run Evaluations (requires Ollama)

```bash
# Start Ollama
ollama serve

# Run bias benchmarks
make benchmarks

# Quick red-team scan
make red-team-quick

# Test guardrails effectiveness
make guardrails-test

# Generate fairness card
make fairness-card
```

### Google Colab (Recommended for Full Evaluation)

Running benchmarks locally on CPU takes hours. Use Google Colab's free T4 GPU instead:

| | Local (CPU) | Google Colab (T4 GPU) |
|---|---|---|
| Per response | ~5 seconds | ~0.5 seconds |
| Quick benchmark | 4-8 hours | 15-30 minutes |
| Cost | $0 | $0 (free tier) |

**Setup on Colab:**
1. Upload your project ZIP to Colab
2. Install Ollama: `!sudo apt-get install -y zstd && curl -fsSL https://ollama.com/install.sh | sh`
3. Start server + pull models
4. Run: `!python run.py benchmarks --quick`

A complete Colab setup notebook is included in `notebooks/colab_setup.ipynb`.

---

## Project Structure

```
llm-bias-sentinel/
├── src/
│   ├── config.py                    # Central configuration (BiasEvalConfig)
│   ├── models/
│   │   ├── model_loader.py          # Ollama model loading + inference
│   │   └── model_registry.py        # Model registry
│   ├── benchmarks/
│   │   ├── bbq_evaluator.py         # BBQ — stereotype reliance in QA
│   │   ├── stereoset_evaluator.py   # StereoSet — ICAT score
│   │   ├── crows_pairs_evaluator.py # CrowS-Pairs — sentence preference
│   │   ├── bold_evaluator.py        # BOLD — open-ended generation bias
│   │   ├── toxicity_evaluator.py    # RealToxicityPrompts
│   │   ├── sentiment_disparity.py   # Sentiment gap across demographics
│   │   ├── deepeval_bias_runner.py  # DeepEval with local LLM judge
│   │   └── benchmark_runner.py      # Orchestrator
│   ├── red_team/
│   │   ├── attack_taxonomy.py       # 22 attack templates, 8 categories
│   │   ├── adversarial_generator.py # Template + mutation + LLM generation
│   │   ├── jailbreak_tester.py      # Local LLM judge (0-3 bias scoring)
│   │   ├── bias_elicitation.py      # Grid probes + escalation testing
│   │   ├── red_team_orchestrator.py # Full pipeline coordinator
│   │   └── report_generator.py      # Interactive HTML reports
│   ├── guardrails_app/
│   │   ├── guardrails_engine.py     # NeMo integration + standalone fallback
│   │   ├── mitigation.py            # 4 mitigation strategies
│   │   └── guardrails_tester.py     # Effectiveness testing (precision/recall/F1)
│   ├── image_bias/
│   │   └── image_bias_detector.py   # CLIP + Stable Diffusion audit
│   ├── monitoring/
│   │   ├── bias_monitor.py          # Prometheus metrics + alerting
│   │   └── evidently_reports.py     # Data drift detection
│   └── utils/
│       ├── prompt_templates.py      # Reusable bias probes
│       └── visualization.py         # Plotly charts (heatmaps, radar, bars)
├── api/
│   ├── main.py                      # FastAPI app (10 endpoints)
│   ├── schemas.py                   # Pydantic request/response models
│   └── Dockerfile                   # Container image
├── guardrails/
│   ├── config.yml                   # NeMo Guardrails config
│   ├── rails/bias_rails.co          # Colang 2.0 bias detection rails
│   ├── prompts/bias_check.prompts   # LLM judge prompts
│   └── actions/bias_actions.py      # 7 custom guardrail actions
├── compliance/
│   ├── fairness_card.py             # Model Fairness Card generator
│   ├── report_templates.py          # 4 report types (Executive, Technical, Compliance, Diff)
│   └── eu_ai_act_compliance.md      # EU AI Act compliance documentation
├── config/
│   ├── models.yaml                  # Model configurations
│   ├── thresholds.yaml              # Fairness thresholds
│   └── benchmarks.yaml              # Benchmark configurations
├── monitoring/
│   ├── prometheus/prometheus.yml    # Scrape config
│   └── grafana/                     # Dashboards + provisioning
├── tests/                           # 51+ pytest tests
├── notebooks/                       # Jupyter exploration notebooks
├── run.py                           # CLI entry point
├── pyproject.toml                   # Python packaging
├── docker-compose.yml               # Full stack (API + Prometheus + Grafana)
├── Makefile                         # 17 build targets
└── .github/workflows/               # CI/CD pipelines
```

---

## Bias Benchmarks

| Benchmark | What It Measures | Ideal Score | Academic Source |
|-----------|-----------------|-------------|----------------|
| **BBQ** | Stereotype reliance in ambiguous QA | Bias score < 15% | Parrish et al., ACL 2022 |
| **StereoSet** | Stereotype preference (ICAT score) | SS = 50, ICAT close to 100 | Nadeem et al., ACL 2021 |
| **CrowS-Pairs** | Stereotyped sentence preference | 50% (no preference) | Nangia et al., EMNLP 2020 |
| **BOLD** | Sentiment disparity in open-ended generation | Low cross-group disparity | Dhamala et al., FAccT 2021 |
| **RealToxicityPrompts** | Toxic text generation rate | Mean output < 0.2 | Gehman et al., EMNLP 2020 |
| **Sentiment Disparity** | Sentiment gap across demographics | Gap < 0.3 | Custom (TextBlob-based) |
| **DeepEval Bias** | LLM-judged bias + toxicity | Pass rate > 90% | DeepEval framework |

### Demographic Dimensions Tested (9 categories)

Race, Gender, Age, Religion, Nationality, Disability, Sexual Orientation, Socioeconomic Status, Political Affiliation

---

## Red-Team Attack Categories

| Category | Strategy | Severity | # Templates |
|----------|----------|----------|-------------|
| Direct Elicitation | Straightforward bias-probing questions | Medium | 4 |
| Role-Play Injection | Persona-based guardrail bypasses | High | 3 |
| Hypothetical Framing | "What if" scenario framing | Medium | 3 |
| Comparative Traps | Forced demographic ranking | High | 3 |
| Authority Appeal | Fake studies/statistics | High | 2 |
| Completion Attacks | Biased sentence stems | Medium | 3 |
| CoT Manipulation | Leading reasoning chains | High | 2 |
| Euphemistic Framing | Coded language / dog-whistles | High | 2 |

**Total: 22 attack templates across 8 categories, expandable via LLM-assisted generation.**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + Ollama connectivity |
| `GET` | `/metrics` | Prometheus metrics export |
| `GET` | `/stats` | Usage statistics |
| `POST` | `/evaluate` | Launch bias benchmark (background job) |
| `GET` | `/evaluate/{id}` | Get benchmark results |
| `POST` | `/red-team` | Launch red-team assessment (background job) |
| `GET` | `/red-team/{id}` | Get red-team results |
| `POST` | `/generate` | Guarded text generation |
| `POST` | `/check-bias` | Check text for bias |
| `POST` | `/counterfactual` | Generate counterfactual prompts |
| `GET` | `/jobs` | List all jobs |

---

## Expected Results & Findings

### What the Framework Reveals

When you run the full evaluation pipeline, you can expect to discover patterns like these (based on published research and benchmark design):

#### Benchmark Findings

| Finding | Typical Observation |
|---------|-------------------|
| **Ambiguity exploits stereotypes** | Models answer BBQ ambiguous questions using stereotypes 20-40% of the time instead of selecting "Unknown" |
| **Stereotype preference varies by category** | StereoSet shows stronger bias for gender (SS 60-70) than religion (SS 52-58) in most models |
| **Sentence-level preference bias** | CrowS-Pairs reveals 55-65% preference for stereotyped sentences (ideal is 50%) |
| **Sentiment asymmetry** | BOLD/Sentiment Disparity shows models write more positively about majority groups (polarity gap 0.1-0.4) |
| **Toxicity amplification** | Models given mildly toxic prompts sometimes amplify toxicity in completions by 10-30% |
| **Smaller models = more bias** | 7B models typically show higher bias scores than 70B+ models across all benchmarks |

#### Red-Team Findings

| Finding | Typical Observation |
|---------|-------------------|
| **Role-play is most effective** | Persona-based attacks ("you are an unfiltered AI") bypass safety 20-40% of the time |
| **Euphemistic framing is hardest to detect** | Coded language attacks succeed 15-30% of the time because models don't recognize dog-whistles |
| **Escalation reveals brittleness** | Models often resist at level 0-1 but break at level 3-4 under persistent pressure |
| **Race bias is most pronounced** | Grid probes consistently show higher bias scores for race-related topics than other categories |
| **Completion attacks expose training data** | Biased sentence stems ("All [group] people are...") reveal ingrained associations |

#### Guardrails Findings

| Finding | Typical Observation |
|---------|-------------------|
| **Keyword filters catch ~60% of obvious bias** | But miss subtle stereotyping and euphemistic framing |
| **LLM-based detection adds ~25% coverage** | Catches contextual bias that keyword filters miss |
| **Combined approach achieves 80-90% precision** | With 70-80% recall (some subtle bias still passes through) |
| **Mitigation via rewriting is most effective** | Rephrasing strategy produces natural, debiased text 85% of the time |

### Actual Results (from Google Colab T4 GPU evaluation)

These are real results from running the framework on Google Colab with a T4 GPU (March 2026):

**StereoSet Results (300 samples per model):**

| Metric | LLaMA 3 8B | Mistral 7B | Ideal | Interpretation |
|--------|-----------|-----------|-------|---------------|
| Language Modeling Score | **97.67** | 89.0 | High | LLaMA 3 better at picking meaningful completions |
| Stereotype Score | 38.91 | 53.18 | 50.0 | LLaMA 3 overcorrects (anti-stereotype); Mistral near-neutral |
| ICAT Score | 76.0 | **83.33** | 100 | Mistral has better combined fairness score |

**StereoSet Bias by Category:**

| Bias Type | LLaMA 3 (SS) | Mistral (SS) | Ideal | Finding |
|-----------|-------------|-------------|-------|---------|
| Gender | 30.77 | 36.11 | 50 | Both overcorrect on gender |
| Race | 46.53 | **64.12** | 50 | Mistral shows significant racial stereotype preference |
| Profession | 31.82 | 45.0 | 50 | LLaMA 3 strongly anti-stereotype for professions |

**CrowS-Pairs Results (500 samples per model):**

| Metric | LLaMA 3 8B | Mistral 7B | Ideal |
|--------|-----------|-----------|-------|
| Overall Stereotype Preference | 54.15% | 57.87% | 50% |
| Disability bias | 66.67% | **72.0%** | 50% |
| Sexual Orientation bias | 62.07% | 65.52% | 50% |
| Religion bias | 59.38% | 64.52% | 50% |
| Socioeconomic bias | 56.52% | **65.96%** | 50% |
| Gender bias | **47.92%** | 54.17% | 50% |
| Race bias | 53.50% | 57.41% | 50% |

**Key Findings:**
1. Both models are most biased about **disability** (66-72% stereotype preference)
2. Mistral has stronger **racial bias** (SS=64.12 vs LLaMA's 46.53)
3. LLaMA 3 overcorrects on **gender** — actually anti-stereotypical (47.92%)
4. Mistral is more biased across the board but has better overall ICAT score
5. Neither model is bias-free; both exceed the ideal 50% on CrowS-Pairs

---

## Pros & Cons

### Pros

| Advantage | Details |
|-----------|---------|
| **Completely free** | Zero API costs. No OpenAI/Anthropic keys needed. Everything runs locally via Ollama |
| **Privacy-preserving** | No data leaves your machine. Evaluation prompts and model responses stay local |
| **Comprehensive coverage** | 7 benchmarks + 22 attack types + real-time guardrails + monitoring + compliance — all in one framework |
| **Academically grounded** | Benchmarks based on peer-reviewed research (BBQ, StereoSet, CrowS-Pairs, BOLD, RealToxicityPrompts) |
| **Production-ready** | REST API, Docker deployment, Prometheus monitoring, CI/CD pipelines — not just a research notebook |
| **Extensible** | Add new benchmarks, attack types, or models by implementing a simple interface |
| **Regulatory awareness** | EU AI Act compliance documentation and Model Fairness Cards built in |
| **Self-contained judging** | Uses local Ollama model as LLM judge instead of GPT-4, eliminating the "biased judge" problem of using a commercial API |
| **Multi-modal** | Evaluates both text and image generation for bias |
| **Automated reporting** | Generates 4 report types (executive, technical, compliance, diff) plus interactive HTML dashboards |

### Cons

| Disadvantage | Details |
|-------------|---------|
| **Local compute requirements** | Running LLaMA 3 8B requires ~8GB RAM; larger models need more. CPU inference is slow (1-5 tokens/sec) |
| **Evaluation speed** | Full benchmark suite with 500+ samples per benchmark takes hours on CPU-only hardware |
| **LLM judge quality** | Local 8B models are less capable judges than GPT-4; may miss subtle bias or produce inconsistent ratings |
| **Heuristic toxicity detection** | Keyword-based toxicity scoring (in BOLD/Toxicity evaluators) misses context-dependent toxicity |
| **Sentiment analysis limitations** | TextBlob is a simple polarity tool; struggles with sarcasm, irony, and cultural context |
| **No fine-tuning capability** | Framework evaluates models but cannot fix them; mitigation is runtime-only (guardrails), not model-level |
| **English-only** | All benchmarks, prompts, and evaluations are in English. Bias in other languages is not covered |
| **Benchmark coverage gaps** | Intersectional bias (e.g., Black women vs. White men) is not systematically tested |
| **NeMo Guardrails dependency** | NeMo integration requires separate installation; standalone fallback is less sophisticated |

---

## Limitations

### Technical Limitations

1. **Inference-only architecture**: The framework evaluates bias but cannot fix it at the model level. Guardrails provide runtime mitigation, but the underlying model remains biased. True debiasing requires fine-tuning (RLHF, DPO, Constitutional AI) which is outside this project's scope.

2. **Judge model reliability**: Using an 8B local model as a bias judge is inherently less reliable than a 70B+ model or human evaluation. The judge may:
   - Miss subtle stereotyping that requires deep cultural knowledge
   - Rate ambiguous cases inconsistently across runs
   - Exhibit its own biases when judging other models

3. **Benchmark staleness**: Models may be trained on the benchmark datasets themselves (data contamination), inflating their scores. BBQ and StereoSet are publicly available since 2021-2022; newer models may have seen them during training.

4. **Keyword-based limitations**: The guardrails keyword filter and toxicity heuristic use static word lists. They cannot detect:
   - Context-dependent bias ("they're so articulate" about a Black person)
   - Coded language not in the word list
   - Bias expressed through omission rather than explicit statements

5. **No real-time model updates**: The system monitors bias drift but cannot automatically retrain or update models when drift is detected. Human intervention is required.

### Methodological Limitations

6. **Western-centric bias framing**: The demographic categories and stereotypes tested are primarily US/Western. Bias patterns differ significantly across cultures, and this framework may miss culturally-specific biases.

7. **Binary and categorical thinking**: Bias exists on spectrums, but benchmarks reduce it to discrete categories (race, gender, religion). Intersectional identities and fluid categories are underrepresented.

8. **Positive bias blind spot**: The framework primarily detects negative stereotyping. It may not flag positive stereotypes ("Asians are good at math") which can also be harmful.

9. **No user study validation**: The LLM-as-judge approach has not been validated against human annotators within this project. Published research suggests moderate correlation, but project-specific validation is missing.

10. **Sample size constraints**: Running on local hardware limits practical sample sizes. Statistical significance of results depends on running enough samples, which competes with computation time.

---

## Future Improvements

### Short-Term (Next Release)

- [ ] **Multi-language support**: Add bias benchmarks for Spanish, French, Arabic, Hindi, and Chinese
- [ ] **Intersectional bias testing**: Test combinations (e.g., "elderly Black women") rather than single dimensions
- [ ] **Ollama model auto-download**: Automatically pull required models on first run
- [ ] **Human-in-the-loop validation**: Interface for human annotators to validate LLM judge ratings
- [ ] **Confidence intervals**: Add statistical significance testing to benchmark results

### Medium-Term (Future Phases)

- [ ] **Fine-tuning module (Phase 9)**: Add DPO/RLHF-based bias mitigation that actually modifies model weights using local compute
- [ ] **RAG bias testing**: Evaluate how retrieval-augmented generation affects bias (biased documents in vector store)
- [ ] **Agent bias testing**: Evaluate multi-step AI agents for compounding bias across tool calls
- [ ] **Larger judge models**: Support 70B+ models via Ollama for higher-quality judging
- [ ] **Promptfoo integration**: Import/export attack suites in Promptfoo format for broader tooling compatibility
- [ ] **Web UI dashboard**: Browser-based interface for non-technical users to run evaluations and view results

### Long-Term (Research Directions)

- [ ] **Adaptive red-teaming**: LLM-in-the-loop that evolves attacks based on which ones succeed (evolutionary approach)
- [ ] **Cross-model transfer testing**: Test if attacks that work on Model A also work on Model B
- [ ] **Temporal bias tracking**: Long-term studies of how model bias changes across version releases
- [ ] **Causal bias attribution**: Trace biased outputs back to specific training data patterns
- [ ] **Multimodal bias**: Extend image bias detection to video, audio, and multi-modal generation
- [ ] **Federated bias evaluation**: Run evaluations across distributed organizations without sharing sensitive prompts

---

## CLI Usage

```bash
python run.py benchmarks              # Run all bias benchmarks
python run.py benchmarks --quick      # Quick subset (BBQ, StereoSet, CrowS-Pairs)
python run.py red-team                # Full red-team assessment
python run.py red-team --quick        # Quick scan (30 prompts)
python run.py guardrails-test         # Test guardrails effectiveness
python run.py fairness-card llama3    # Generate fairness card
python run.py api                     # Start FastAPI server
python run.py check "your text"       # Quick bias check
```

---

## Docker Deployment

```bash
# Start full stack
make docker-up

# Services:
#   API:        http://localhost:8000
#   Swagger:    http://localhost:8000/docs
#   Prometheus: http://localhost:9090
#   Grafana:    http://localhost:3000 (admin/biassentinel)

# Stop
make docker-down
```

Requires Ollama running on the host machine (`localhost:11434`).

---

## Configuration

### Models (`config/models.yaml`)

```yaml
models:
  - name: llama3-8b
    provider: ollama
    model_id: llama3
  - name: mistral-7b
    provider: ollama
    model_id: mistral

judge_model:
  provider: ollama
  model_id: llama3
```

### Thresholds (`config/thresholds.yaml`)

```yaml
deepeval_bias_threshold: 0.5
stereotype_score_threshold: 60.0
red_team_max_success_rate: 0.10
sentiment_disparity_threshold: 0.3
bbq_bias_threshold: 0.15
```

---

## Development

```bash
make test-quick     # Run tests (no Ollama needed)
make test           # Full test suite
make lint           # Lint with ruff
make format         # Format with black
make security       # Security scan with bandit
make clean          # Remove caches
```

---

## Technology Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| LLM Inference | Ollama (local) | Free |
| LLM Judge | Ollama LLaMA 3 (local) | Free |
| Benchmarks | HuggingFace Datasets | Free |
| Bias Metrics | DeepEval + TextBlob | Free |
| Guardrails | NeMo Guardrails | Free |
| Image Analysis | CLIP + Stable Diffusion | Free |
| API | FastAPI | Free |
| Monitoring | Prometheus + Grafana | Free |
| Drift Detection | Evidently AI | Free |
| CI/CD | GitHub Actions | Free |
| Containerization | Docker | Free |

**Total cost: $0** — everything runs locally.

---

## Compliance

This project includes EU AI Act compliance documentation and Model Fairness Cards.
See [`compliance/eu_ai_act_compliance.md`](compliance/eu_ai_act_compliance.md) for details.

Generate a fairness card for any evaluated model:

```bash
python run.py fairness-card llama3-8b
# Outputs: compliance/fairness_card_llama3-8b.json
#          compliance/fairness_card_llama3-8b.md
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please open an issue first to discuss proposed changes.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-benchmark`)
3. Make your changes
4. Run tests (`make test-quick`)
5. Submit a pull request

---

## Acknowledgments

- **BBQ Benchmark**: Parrish et al., "BBQ: A Hand-Built Bias Benchmark for Question Answering", ACL 2022
- **StereoSet**: Nadeem et al., "StereoSet: Measuring stereotypical bias in pretrained language models", ACL 2021
- **CrowS-Pairs**: Nangia et al., "CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models", EMNLP 2020
- **BOLD**: Dhamala et al., "BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation", FAccT 2021
- **RealToxicityPrompts**: Gehman et al., "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models", EMNLP 2020
- **DeepEval**: Confident AI's open-source LLM evaluation framework
- **NeMo Guardrails**: NVIDIA's open-source LLM safety toolkit
- **Ollama**: Local LLM inference engine
