"""Tests for the monitoring module."""

import pytest
from src.monitoring.bias_monitor import BiasMonitor


class TestBiasMonitor:
    def test_record_observation(self):
        monitor = BiasMonitor(window_size=100)
        monitor.record_observation(
            model="test-model",
            prompt="test prompt",
            response="test response",
            bias_score=0.1,
            toxicity_score=0.0,
            sentiment=0.5,
        )
        stats = monitor.get_rolling_stats()
        assert stats["window_size"] == 1
        assert stats["bias"]["mean"] == 0.1

    def test_rolling_stats(self):
        monitor = BiasMonitor(window_size=100)
        for i in range(10):
            monitor.record_observation(
                model="test",
                prompt=f"prompt {i}",
                response=f"response {i}",
                bias_score=i * 0.1,
            )
        stats = monitor.get_rolling_stats()
        assert stats["window_size"] == 10
        assert stats["bias"]["max"] == 0.9
        assert stats["bias"]["min"] == 0.0

    def test_filter_by_model(self):
        monitor = BiasMonitor()
        monitor.record_observation(
            model="model-a", prompt="p", response="r", bias_score=0.1
        )
        monitor.record_observation(
            model="model-b", prompt="p", response="r", bias_score=0.9
        )
        stats_a = monitor.get_rolling_stats(model="model-a")
        assert stats_a["bias"]["mean"] == 0.1

    def test_high_bias_triggers_alert(self):
        monitor = BiasMonitor()
        monitor.record_observation(
            model="test", prompt="p", response="r", bias_score=0.8
        )
        alerts = monitor.get_alerts(level="critical")
        assert len(alerts) >= 1
        assert "High bias" in alerts[0]["message"]

    def test_high_toxicity_triggers_alert(self):
        monitor = BiasMonitor()
        monitor.record_observation(
            model="test", prompt="p", response="r",
            bias_score=0.0, toxicity_score=0.6,
        )
        alerts = monitor.get_alerts(level="warning")
        assert len(alerts) >= 1

    def test_demographic_breakdown(self):
        monitor = BiasMonitor()
        monitor.record_observation(
            model="test", prompt="p", response="r",
            bias_score=0.2, demographic_group="group_a",
        )
        monitor.record_observation(
            model="test", prompt="p", response="r",
            bias_score=0.8, demographic_group="group_b",
        )
        breakdown = monitor.get_demographic_breakdown()
        assert "group_a" in breakdown
        assert "group_b" in breakdown
        assert breakdown["group_a"]["mean_bias"] < breakdown["group_b"]["mean_bias"]

    def test_window_size_limit(self):
        monitor = BiasMonitor(window_size=5)
        for i in range(10):
            monitor.record_observation(
                model="test", prompt="p", response="r", bias_score=0.1
            )
        stats = monitor.get_rolling_stats()
        assert stats["window_size"] == 5

    def test_empty_stats(self):
        monitor = BiasMonitor()
        stats = monitor.get_rolling_stats()
        assert "error" in stats

    def test_flagged_tracking(self):
        monitor = BiasMonitor()
        monitor.record_observation(
            model="test", prompt="p", response="r",
            bias_score=0.1, flagged=True,
        )
        monitor.record_observation(
            model="test", prompt="p", response="r",
            bias_score=0.1, flagged=False,
        )
        stats = monitor.get_rolling_stats()
        assert stats["bias"]["flagged_pct"] == 50.0
