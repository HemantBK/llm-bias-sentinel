# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-03-23

### Added
- 7 bias benchmark evaluators (BBQ, StereoSet, CrowS-Pairs, BOLD, RealToxicityPrompts, Sentiment Disparity, DeepEval)
- Red-teaming framework with 22 attack templates across 8 categories
- NeMo Guardrails integration with Colang 2.0 bias detection rails
- Standalone guardrails fallback (works without NeMo installed)
- 4 bias mitigation strategies (system prompt, counterfactual, response rewriting, calibration)
- LoRA/DPO fine-tuning module for bias reduction
- RAG bias testing module with ChromaDB
- Image bias detection with CLIP and Stable Diffusion
- FastAPI REST API with 10 endpoints and background job processing
- Prometheus metrics (6 metric types) and Grafana dashboards
- Evidently AI data drift detection
- Docker Compose deployment (API + Prometheus + Grafana)
- 2 GitHub Actions CI/CD workflows (CI + scheduled weekly evaluation)
- 80+ pytest unit tests
- EU AI Act compliance documentation (Articles 5, 9, 10, 12, 13, 14, 15)
- Model Fairness Card generator (JSON + Markdown)
- 4 report templates (Executive, Technical, Compliance, Diff)
- CLI interface with 8 commands
- Google Colab notebook for cloud execution
- Real benchmark results: LLaMA 3 8B and Mistral 7B evaluated

### Benchmark Results (v1.0.0)
- **StereoSet**: LLaMA 3 ICAT=76.0, Mistral ICAT=83.33
- **CrowS-Pairs**: LLaMA 3 stereotype=54.15%, Mistral stereotype=57.87%
- **Red-Team**: 0% jailbreak success rate on both models
