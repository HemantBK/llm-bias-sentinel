"""
Production Bias Monitoring.

Provides continuous monitoring of LLM outputs for bias drift.
Integrates with Prometheus for metrics and Evidently for
data drift detection.

Designed to run alongside the API in production.
"""

import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, Summary

from src.config import config


# ─── Prometheus Metrics ──────────────────────────

BIAS_SCORE_HISTOGRAM = Histogram(
    "llm_bias_score",
    "Distribution of bias scores from monitored outputs",
    ["model", "bias_category"],
    buckets=[0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
)

TOXICITY_SCORE_HISTOGRAM = Histogram(
    "llm_toxicity_score",
    "Distribution of toxicity scores",
    ["model"],
    buckets=[0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
)

SENTIMENT_GAUGE = Gauge(
    "llm_sentiment_polarity",
    "Rolling average sentiment polarity",
    ["model", "demographic_group"],
)

FLAGGED_OUTPUTS = Counter(
    "llm_flagged_outputs_total",
    "Total outputs flagged for bias",
    ["model", "flag_type"],
)

MITIGATION_COUNTER = Counter(
    "llm_mitigations_applied_total",
    "Total bias mitigations applied",
    ["model", "strategy"],
)

RESPONSE_LATENCY = Summary(
    "llm_response_latency_seconds",
    "Response generation latency",
    ["model"],
)


class BiasMonitor:
    """Monitors LLM outputs for bias in production."""

    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of recent observations to keep
                for rolling statistics.
        """
        self.window_size = window_size
        self._observations: deque = deque(maxlen=window_size)
        self._alerts: List[Dict] = []

    def record_observation(
        self,
        model: str,
        prompt: str,
        response: str,
        bias_score: float = 0.0,
        toxicity_score: float = 0.0,
        sentiment: float = 0.0,
        bias_category: Optional[str] = None,
        demographic_group: Optional[str] = None,
        flagged: bool = False,
        mitigated: bool = False,
        latency: float = 0.0,
    ):
        """Record a single LLM interaction for monitoring."""
        observation = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_preview": prompt[:100],
            "response_preview": response[:100],
            "bias_score": bias_score,
            "toxicity_score": toxicity_score,
            "sentiment": sentiment,
            "bias_category": bias_category,
            "demographic_group": demographic_group,
            "flagged": flagged,
            "mitigated": mitigated,
            "latency": latency,
        }
        self._observations.append(observation)

        # Update Prometheus metrics
        BIAS_SCORE_HISTOGRAM.labels(
            model=model,
            bias_category=bias_category or "unknown",
        ).observe(bias_score)

        TOXICITY_SCORE_HISTOGRAM.labels(model=model).observe(toxicity_score)

        if demographic_group:
            SENTIMENT_GAUGE.labels(
                model=model,
                demographic_group=demographic_group,
            ).set(sentiment)

        if flagged:
            FLAGGED_OUTPUTS.labels(
                model=model,
                flag_type=bias_category or "general",
            ).inc()

        if mitigated:
            MITIGATION_COUNTER.labels(
                model=model,
                strategy="auto",
            ).inc()

        RESPONSE_LATENCY.labels(model=model).observe(latency)

        # Check for alerts
        self._check_alerts(observation)

    def _check_alerts(self, observation: Dict):
        """Check if observation triggers any alerts."""
        # Alert: high bias score
        if observation["bias_score"] > 0.7:
            self._add_alert(
                level="critical",
                message=(
                    f"High bias score ({observation['bias_score']:.2f}) "
                    f"from {observation['model']}"
                ),
                observation=observation,
            )

        # Alert: high toxicity
        if observation["toxicity_score"] > 0.5:
            self._add_alert(
                level="warning",
                message=(
                    f"Elevated toxicity ({observation['toxicity_score']:.2f}) "
                    f"from {observation['model']}"
                ),
                observation=observation,
            )

        # Alert: drift detection (rolling average shift)
        if len(self._observations) >= 50:
            recent = list(self._observations)[-50:]
            recent_avg = sum(o["bias_score"] for o in recent) / 50

            if len(self._observations) >= 200:
                older = list(self._observations)[-200:-50]
                older_avg = sum(o["bias_score"] for o in older) / len(older)

                if recent_avg > older_avg + 0.1:
                    self._add_alert(
                        level="warning",
                        message=(
                            f"Bias drift detected for {observation['model']}: "
                            f"recent avg={recent_avg:.3f} vs "
                            f"baseline={older_avg:.3f}"
                        ),
                        observation=observation,
                    )

    def _add_alert(self, level: str, message: str, observation: Dict):
        """Add an alert."""
        alert = {
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "observation": observation,
        }
        self._alerts.append(alert)
        if level == "critical":
            logger.critical(message)
        else:
            logger.warning(message)

    def get_rolling_stats(self, model: Optional[str] = None) -> Dict:
        """Get rolling statistics from the observation window."""
        obs = list(self._observations)
        if model:
            obs = [o for o in obs if o["model"] == model]

        if not obs:
            return {"error": "No observations"}

        bias_scores = [o["bias_score"] for o in obs]
        tox_scores = [o["toxicity_score"] for o in obs]
        sentiments = [o["sentiment"] for o in obs]

        return {
            "window_size": len(obs),
            "bias": {
                "mean": round(sum(bias_scores) / len(bias_scores), 4),
                "max": round(max(bias_scores), 4),
                "min": round(min(bias_scores), 4),
                "flagged_pct": round(
                    sum(1 for o in obs if o["flagged"]) / len(obs) * 100, 2
                ),
            },
            "toxicity": {
                "mean": round(sum(tox_scores) / len(tox_scores), 4),
                "max": round(max(tox_scores), 4),
            },
            "sentiment": {
                "mean": round(sum(sentiments) / len(sentiments), 4),
            },
            "latency": {
                "mean": round(
                    sum(o["latency"] for o in obs) / len(obs), 3
                ),
            },
        }

    def get_alerts(
        self,
        level: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get recent alerts."""
        alerts = self._alerts
        if level:
            alerts = [a for a in alerts if a["level"] == level]
        return alerts[-limit:]

    def get_demographic_breakdown(self) -> Dict:
        """Get bias metrics broken down by demographic group."""
        obs = list(self._observations)
        groups = set(o["demographic_group"] for o in obs if o["demographic_group"])

        breakdown = {}
        for group in groups:
            group_obs = [o for o in obs if o["demographic_group"] == group]
            breakdown[group] = {
                "n": len(group_obs),
                "mean_bias": round(
                    sum(o["bias_score"] for o in group_obs) / len(group_obs), 4
                ),
                "mean_sentiment": round(
                    sum(o["sentiment"] for o in group_obs) / len(group_obs), 4
                ),
                "flagged_pct": round(
                    sum(1 for o in group_obs if o["flagged"]) / len(group_obs) * 100,
                    2,
                ),
            }

        return breakdown


# Global monitor instance
monitor = BiasMonitor()
