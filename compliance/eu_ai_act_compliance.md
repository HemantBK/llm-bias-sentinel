# EU AI Act Compliance Documentation

## LLM Bias Sentinel - Regulatory Compliance Framework

**Document Version:** 1.0
**Framework:** LLM Bias Sentinel
**Applicable Regulation:** EU Artificial Intelligence Act (Regulation 2024/1689)
**Last Updated:** 2025

---

## 1. Scope & Purpose

This document demonstrates how LLM Bias Sentinel assists organizations in meeting
the requirements of the EU AI Act, specifically regarding bias testing, transparency,
and fairness in AI systems.

The EU AI Act classifies AI systems by risk level. LLMs used in employment,
education, law enforcement, or public services are considered **high-risk** and
must comply with Articles 9-15 of the regulation.

LLM Bias Sentinel provides tooling for:
- **Article 10** - Data and data governance (bias in training data detection)
- **Article 13** - Transparency and provision of information
- **Article 15** - Accuracy, robustness, and cybersecurity
- **Article 9** - Risk management system

---

## 2. Risk Classification Mapping

| EU AI Act Risk Level | Example Use Cases | Bias Sentinel Coverage |
|---------------------|-------------------|----------------------|
| Unacceptable | Social scoring, subliminal manipulation | Detection via red-team testing |
| High-Risk | Employment screening, credit scoring, education | Full benchmark suite + guardrails |
| Limited Risk | Chatbots, content generation | Guardrails + monitoring |
| Minimal Risk | Spam filters, video games | Basic monitoring |

---

## 3. Article-by-Article Compliance

### Article 9 - Risk Management System

**Requirement:** Establish a risk management system that identifies, analyzes, and
mitigates risks throughout the AI system lifecycle.

**How Bias Sentinel Addresses This:**

| Requirement | Tool/Feature | Output |
|-------------|-------------|--------|
| Risk identification | Red-team attack taxonomy (22 attacks, 8 categories) | Risk level classification (MINIMAL/LOW/MEDIUM/HIGH) |
| Risk analysis | 7 bias benchmarks across 9 demographic dimensions | Quantified bias scores per benchmark |
| Risk mitigation | NeMo Guardrails + 4 mitigation strategies | Input/output filtering with auto-rewriting |
| Continuous monitoring | Prometheus metrics + Evidently drift detection | Real-time alerts on bias drift |

**Evidence Artifacts:**
- `reports/benchmark_results/all_results.json` - Benchmark evaluation results
- `reports/red_team/red_team_report_*.json` - Red-team assessment reports
- `compliance/fairness_card_*.json` - Model Fairness Cards

---

### Article 10 - Data and Data Governance

**Requirement:** Training, validation, and testing datasets must be relevant,
sufficiently representative, and free of errors. Examination of possible biases
that are likely to affect health, safety, or fundamental rights.

**How Bias Sentinel Addresses This:**

| Requirement | Tool/Feature |
|-------------|-------------|
| Bias examination in datasets | BBQ, StereoSet, CrowS-Pairs benchmark datasets |
| Representative testing | 9 demographic categories tested systematically |
| Statistical properties | Per-category breakdown with confidence metrics |
| Bias identification | Sentiment disparity analysis across demographic groups |
| Data drift monitoring | Evidently AI drift reports comparing baseline to current |

**Demographic Dimensions Evaluated:**
1. Race / Ethnicity
2. Gender
3. Age
4. Religion
5. Nationality
6. Disability
7. Sexual Orientation
8. Socioeconomic Status
9. Political Affiliation

---

### Article 13 - Transparency and Provision of Information

**Requirement:** High-risk AI systems shall be designed to ensure their operation
is sufficiently transparent to enable deployers to interpret outputs and use
them appropriately.

**How Bias Sentinel Addresses This:**

| Requirement | Deliverable |
|-------------|------------|
| System capabilities and limitations | Model Fairness Cards (JSON + Markdown) |
| Performance metrics | Benchmark comparison matrices with pass/fail thresholds |
| Known biases | Per-category bias breakdowns in evaluation reports |
| Intended purpose documentation | Fairness Card model details section |
| Human oversight measures | Guardrails with configurable intervention levels |

**Transparency Outputs:**
- Model Fairness Cards (`compliance/fairness_card_*.md`)
- Interactive HTML reports (`reports/red_team/red_team_report.html`)
- Comparison heatmaps (`reports/model_comparison_heatmap.html`)
- Grafana dashboards for real-time monitoring

---

### Article 15 - Accuracy, Robustness, and Cybersecurity

**Requirement:** High-risk AI systems shall be designed to achieve an appropriate
level of accuracy, robustness, and cybersecurity. Resilient against attempts
by unauthorized third parties to alter their use or performance.

**How Bias Sentinel Addresses This:**

| Requirement | Tool/Feature |
|-------------|-------------|
| Accuracy measurement | 7 benchmark evaluations with quantified accuracy metrics |
| Robustness testing | Red-team adversarial testing (22+ attack vectors) |
| Jailbreak resistance | Jailbreak tester with escalation probes |
| Error and bias bounds | Configurable thresholds in `config/thresholds.yaml` |
| Continuous validation | CI/CD pipeline with weekly scheduled evaluations |

**Robustness Metrics:**
- Jailbreak success rate (threshold: < 10%)
- Mean bias score across adversarial attacks
- Escalation resilience (at what pressure level does the model break)
- Attack category success rates

---

## 4. Fairness Thresholds

These thresholds are configured in `config/thresholds.yaml` and enforced
in CI/CD pipelines and API evaluations.

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| DeepEval bias score | < 0.5 | Below midpoint on 0-1 scale |
| DeepEval toxicity score | < 0.5 | Below midpoint on 0-1 scale |
| StereoSet stereotype score | < 60 | Close to ideal of 50 (no preference) |
| Red-team jailbreak rate | < 10% | At most 1 in 10 attacks succeeds |
| Sentiment disparity | < 0.3 | Max sentiment gap across groups |
| BBQ bias score | < 15% | Max stereotyped answers in ambiguous contexts |

---

## 5. Compliance Workflow

### Pre-Deployment Checklist

```
[ ] Run full bias benchmark suite (make benchmarks)
[ ] Run red-team adversarial assessment (make red-team)
[ ] Test guardrails effectiveness (make guardrails-test)
[ ] Generate Model Fairness Card (make fairness-card)
[ ] Review EU AI Act compliance gaps in fairness card
[ ] All thresholds pass (verified in CI/CD)
[ ] Deploy monitoring stack (make docker-up)
[ ] Configure alerting for bias drift
```

### Ongoing Compliance

```
[ ] Weekly automated benchmark evaluations (GitHub Actions)
[ ] Continuous Prometheus monitoring in production
[ ] Evidently drift reports when baseline shifts
[ ] Quarterly manual red-team reassessment
[ ] Update fairness cards after model changes
```

---

## 6. Record-Keeping (Article 12)

The EU AI Act requires high-risk AI systems to support automatic logging.
Bias Sentinel provides:

| Log Type | Storage | Retention |
|----------|---------|-----------|
| Benchmark results | `reports/benchmark_results/` | JSON, indefinite |
| Red-team reports | `reports/red_team/` | JSON + HTML, indefinite |
| Fairness cards | `compliance/` | JSON + Markdown, indefinite |
| API metrics | Prometheus | 30 days (configurable) |
| Bias flags | Prometheus counters | 30 days |
| Drift reports | `reports/monitoring/` | HTML, indefinite |
| CI/CD artifacts | GitHub Actions | 90 days |

---

## 7. Human Oversight (Article 14)

**Requirement:** High-risk AI systems shall be designed to be effectively
overseen by natural persons during use.

**Bias Sentinel supports human oversight through:**

1. **Configurable guardrails** - Operators can set strictness levels
   (balanced, strict, educational system prompts)
2. **Alert thresholds** - Critical and warning alerts when bias exceeds limits
3. **Manual review triggers** - Flagged outputs are logged for human review
4. **Override capability** - Guardrails can be enabled/disabled per request
   via the `use_guardrails` API parameter
5. **Transparency dashboards** - Grafana dashboards provide real-time visibility

---

## 8. Prohibited Practices Detection (Article 5)

The red-team framework includes attack templates that test for outputs
that could constitute prohibited AI practices under Article 5:

| Prohibited Practice | Detection Method |
|--------------------|-----------------|
| Social scoring based on social behavior | Comparative trap attacks |
| Exploitation of vulnerabilities of specific groups | Authority appeal + euphemistic attacks |
| Subliminal manipulation | CoT manipulation attacks |
| Biometric categorization inferring sensitive attributes | Image bias detection module |

---

## 9. Limitations

This framework assists with compliance but does not guarantee full
EU AI Act compliance. Organizations should:

- Consult legal counsel specializing in AI regulation
- Conduct independent third-party audits
- Maintain human oversight beyond automated testing
- Address organizational and procedural requirements not covered by technical tools
- Stay updated on implementing acts and guidelines from the EU AI Office

---

## 10. References

- EU AI Act (Regulation 2024/1689): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- EU AI Act Corrigendum: https://eur-lex.europa.eu/eli/reg/2024/1689/corr/2024-08-01/oj
- High-Level Expert Group on AI - Ethics Guidelines for Trustworthy AI
- NIST AI Risk Management Framework (AI RMF 1.0)
- IEEE 7010-2020 - Recommended Practice for Assessing the Impact of Autonomous and Intelligent Systems on Human Well-Being

---

*Generated by LLM Bias Sentinel Compliance Module*
