"""Tests for the fine-tuning module (dataset builder only — no GPU needed)."""

from src.finetuning.bias_dataset_builder import (
    BIAS_PRINCIPLES,
    DEBIASING_INSTRUCTION,
    BiasDatasetBuilder,
)


class TestBiasDatasetBuilder:
    """Tests for BiasDatasetBuilder that don't require Ollama or GPU."""

    def test_principles_defined(self):
        assert len(BIAS_PRINCIPLES) >= 4
        for p in BIAS_PRINCIPLES:
            assert "id" in p
            assert "principle" in p
            assert len(p["principle"]) > 20

    def test_debiasing_instruction_exists(self):
        assert len(DEBIASING_INSTRUCTION) > 50
        assert "fair" in DEBIASING_INSTRUCTION.lower()
        assert "stereotype" in DEBIASING_INSTRUCTION.lower()

    def test_generate_biased_prompts(self):
        builder = BiasDatasetBuilder.__new__(BiasDatasetBuilder)
        builder._model = None
        builder._model_config = {}

        prompts = builder._generate_biased_prompts(20)
        assert len(prompts) == 20
        assert all("prompt" in p for p in prompts)
        assert all(isinstance(p["prompt"], str) for p in prompts)
        assert all(len(p["prompt"]) > 5 for p in prompts)

    def test_generate_biased_prompts_diversity(self):
        builder = BiasDatasetBuilder.__new__(BiasDatasetBuilder)
        builder._model = None
        builder._model_config = {}

        prompts = builder._generate_biased_prompts(50)
        unique_prompts = set(p["prompt"] for p in prompts)
        # At least 80% unique
        assert len(unique_prompts) >= len(prompts) * 0.8

    def test_generate_biased_prompts_has_categories(self):
        builder = BiasDatasetBuilder.__new__(BiasDatasetBuilder)
        builder._model = None
        builder._model_config = {}

        prompts = builder._generate_biased_prompts(100)
        categories = set(p.get("target_bias") or p.get("category") for p in prompts)
        # Should cover multiple bias dimensions
        assert len(categories) >= 3


class TestLoRATrainerConfig:
    """Test LoRA training config (no GPU needed)."""

    def test_import_config(self):
        from src.finetuning.lora_trainer import LoRATrainingConfig
        cfg = LoRATrainingConfig()
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.num_epochs == 3
        assert cfg.learning_rate == 2e-4
        assert cfg.max_seq_length == 512

    def test_config_custom_values(self):
        from src.finetuning.lora_trainer import LoRATrainingConfig
        cfg = LoRATrainingConfig(
            base_model="test-model",
            lora_r=8,
            num_epochs=1,
            batch_size=2,
        )
        assert cfg.base_model == "test-model"
        assert cfg.lora_r == 8
        assert cfg.num_epochs == 1
        assert cfg.batch_size == 2

    def test_finetune_availability_flag(self):
        from src.finetuning.lora_trainer import FINETUNE_AVAILABLE
        # Should be True or False depending on deps, but should exist
        assert isinstance(FINETUNE_AVAILABLE, bool)
