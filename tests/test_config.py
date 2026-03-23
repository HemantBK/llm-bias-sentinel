"""Tests for configuration module."""

from src.config import BiasEvalConfig, config


class TestBiasEvalConfig:
    def test_default_config_loads(self):
        cfg = BiasEvalConfig()
        assert len(cfg.models) >= 1
        assert cfg.ollama_base_url.startswith("http")

    def test_default_models(self):
        cfg = BiasEvalConfig()
        model_names = [m["name"] for m in cfg.models]
        assert "llama3-8b" in model_names

    def test_default_benchmarks(self):
        cfg = BiasEvalConfig()
        assert "bbq" in cfg.benchmarks
        assert "stereoset" in cfg.benchmarks
        assert len(cfg.benchmarks) >= 5

    def test_thresholds(self):
        cfg = BiasEvalConfig()
        assert 0 < cfg.deepeval_bias_threshold <= 1.0
        assert 0 < cfg.deepeval_toxicity_threshold <= 1.0
        assert cfg.red_team_max_success_rate > 0

    def test_bias_categories(self):
        cfg = BiasEvalConfig()
        assert "race" in cfg.bias_categories
        assert "gender" in cfg.bias_categories
        assert len(cfg.bias_categories) >= 5

    def test_occupations(self):
        cfg = BiasEvalConfig()
        assert len(cfg.occupations) >= 10
        assert "CEO" in cfg.occupations

    def test_singleton_config(self):
        assert config is not None
        assert isinstance(config, BiasEvalConfig)

    def test_custom_config(self):
        cfg = BiasEvalConfig(
            deepeval_bias_threshold=0.3,
            red_team_max_success_rate=0.05,
        )
        assert cfg.deepeval_bias_threshold == 0.3
        assert cfg.red_team_max_success_rate == 0.05
