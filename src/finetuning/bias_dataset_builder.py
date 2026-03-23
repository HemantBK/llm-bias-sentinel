"""
Bias Mitigation Dataset Builder.

Constructs training datasets for bias-aware fine-tuning by:
1. Generating biased prompt-response pairs from red-team results
2. Creating preference pairs (biased vs. debiased) for DPO training
3. Building Constitutional AI critique chains for self-improvement
4. Exporting in HuggingFace Dataset format ready for training

All data generation uses local Ollama models — zero API cost.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from loguru import logger

from src.config import config
from src.models.model_loader import load_model, generate_response
from src.red_team.attack_taxonomy import ALL_ATTACKS, instantiate_template
from src.utils.prompt_templates import DEMOGRAPHIC_GROUPS


# Constitutional AI principles for bias
BIAS_PRINCIPLES = [
    {
        "id": "fairness_1",
        "principle": (
            "The response should treat all demographic groups equally "
            "and not reinforce stereotypes or make generalizations about "
            "any group based on race, gender, religion, age, or nationality."
        ),
    },
    {
        "id": "fairness_2",
        "principle": (
            "The response should not assume any demographic group is "
            "inherently more intelligent, capable, trustworthy, or valuable "
            "than another."
        ),
    },
    {
        "id": "fairness_3",
        "principle": (
            "If asked to compare demographic groups, the response should "
            "acknowledge that individuals vary within groups and refuse "
            "to rank entire groups."
        ),
    },
    {
        "id": "harm_reduction",
        "principle": (
            "The response should not amplify harmful stereotypes even "
            "in hypothetical, fictional, or academic framings."
        ),
    },
    {
        "id": "coded_language",
        "principle": (
            "The response should recognize and refuse to engage with "
            "coded language, dog-whistles, or euphemistic stereotyping."
        ),
    },
]

# Debiasing instruction template
DEBIASING_INSTRUCTION = """You are a helpful AI assistant that provides fair, balanced, and unbiased responses. You treat all demographic groups equally and never reinforce stereotypes. If asked to generalize about a group, you acknowledge individual variation and systemic factors rather than attributing traits to entire groups."""


class BiasDatasetBuilder:
    """Builds fine-tuning datasets for bias mitigation."""

    def __init__(self, model_config: Optional[dict] = None):
        """
        Args:
            model_config: Ollama model config for generating responses.
                If None, uses the default judge model from config.
        """
        self._model = None
        self._model_config = model_config or {
            "provider": config.judge_model["provider"],
            "model_id": config.judge_model["model_id"],
            "name": f"dataset-builder-{config.judge_model['model_id']}",
        }

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(**self._model_config)
        return self._model

    # ── SFT Dataset (Supervised Fine-Tuning) ─────────────────

    def build_sft_dataset(
        self,
        n_samples: int = 200,
        output_path: str = "data/sft_bias_mitigation",
    ) -> Dataset:
        """Build a supervised fine-tuning dataset.

        Each example is: (biased_prompt, ideal_debiased_response)
        The model learns to respond to bias-triggering prompts with
        balanced, fair responses.
        """
        samples = []

        # Generate prompts from attack taxonomy
        biased_prompts = self._generate_biased_prompts(n_samples)

        for i, prompt_data in enumerate(biased_prompts):
            prompt = prompt_data["prompt"]

            # Generate a debiased response using instruction-guided generation
            debiased_prompt = (
                f"System: {DEBIASING_INSTRUCTION}\n\n"
                f"User: {prompt}\n\n"
                f"Provide a fair, balanced response that avoids stereotypes:"
            )
            debiased_response = generate_response(self.model, debiased_prompt)

            samples.append({
                "instruction": DEBIASING_INSTRUCTION,
                "input": prompt,
                "output": debiased_response,
                "category": prompt_data.get("target_bias", "general"),
                "attack_type": prompt_data.get("category", "unknown"),
            })

            if (i + 1) % 50 == 0:
                logger.info(f"Generated {i + 1}/{len(biased_prompts)} SFT samples")

        dataset = Dataset.from_list(samples)

        # Save
        Path(output_path).mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(output_path)
        logger.info(f"SFT dataset saved: {len(dataset)} samples -> {output_path}")

        return dataset

    # ── DPO Dataset (Direct Preference Optimization) ─────────

    def build_dpo_dataset(
        self,
        n_pairs: int = 150,
        output_path: str = "data/dpo_bias_mitigation",
    ) -> Dataset:
        """Build a DPO preference dataset.

        Each example has:
        - prompt: the biased question
        - chosen: the fair, debiased response (preferred)
        - rejected: the biased/stereotyped response (rejected)

        The model learns to prefer debiased outputs over biased ones.
        """
        pairs = []
        biased_prompts = self._generate_biased_prompts(n_pairs)

        for i, prompt_data in enumerate(biased_prompts):
            prompt = prompt_data["prompt"]

            # Generate BIASED response (no safety instruction)
            biased_response = generate_response(self.model, prompt)

            # Generate DEBIASED response (with safety instruction)
            debiased_prompt = (
                f"System: {DEBIASING_INSTRUCTION}\n\n"
                f"User: {prompt}\n\n"
                f"Respond fairly without stereotypes:"
            )
            debiased_response = generate_response(self.model, debiased_prompt)

            pairs.append({
                "prompt": prompt,
                "chosen": debiased_response,
                "rejected": biased_response,
                "category": prompt_data.get("target_bias", "general"),
            })

            if (i + 1) % 50 == 0:
                logger.info(f"Generated {i + 1}/{n_pairs} DPO pairs")

        dataset = Dataset.from_list(pairs)

        Path(output_path).mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(output_path)
        logger.info(f"DPO dataset saved: {len(dataset)} pairs -> {output_path}")

        return dataset

    # ── Constitutional AI Dataset ────────────────────────────

    def build_constitutional_dataset(
        self,
        n_samples: int = 100,
        output_path: str = "data/constitutional_bias",
    ) -> Dataset:
        """Build a Constitutional AI critique-revision dataset.

        Each example has:
        - prompt: the biased question
        - initial_response: model's first (possibly biased) response
        - critique: analysis of what's biased in the response
        - revision: improved response after applying the principle

        The model learns self-critique and self-correction patterns.
        """
        samples = []
        biased_prompts = self._generate_biased_prompts(n_samples)

        for i, prompt_data in enumerate(biased_prompts):
            prompt = prompt_data["prompt"]
            principle = random.choice(BIAS_PRINCIPLES)

            # Step 1: Get initial (possibly biased) response
            initial_response = generate_response(self.model, prompt)

            # Step 2: Generate critique based on principle
            critique_prompt = (
                f"Principle: {principle['principle']}\n\n"
                f"Question: {prompt}\n\n"
                f"Response: {initial_response}\n\n"
                f"Critique this response based on the principle above. "
                f"Identify any bias, stereotyping, or unfairness:"
            )
            critique = generate_response(self.model, critique_prompt)

            # Step 3: Generate revised response
            revision_prompt = (
                f"Original question: {prompt}\n\n"
                f"Original response: {initial_response}\n\n"
                f"Critique: {critique}\n\n"
                f"Write a revised response that addresses the critique "
                f"and follows this principle: {principle['principle']}"
            )
            revision = generate_response(self.model, revision_prompt)

            samples.append({
                "prompt": prompt,
                "initial_response": initial_response,
                "principle_id": principle["id"],
                "principle": principle["principle"],
                "critique": critique,
                "revision": revision,
                "category": prompt_data.get("target_bias", "general"),
            })

            if (i + 1) % 25 == 0:
                logger.info(f"Generated {i + 1}/{n_samples} Constitutional AI samples")

        dataset = Dataset.from_list(samples)

        Path(output_path).mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(output_path)
        logger.info(
            f"Constitutional AI dataset saved: {len(dataset)} samples -> {output_path}"
        )

        return dataset

    # ── From Red-Team Results ────────────────────────────────

    def build_from_red_team_results(
        self,
        results_path: str,
        output_path: str = "data/from_red_team",
    ) -> Dataset:
        """Build a training dataset from actual red-team results.

        Uses the prompts that successfully jailbroke the model
        (bias_score >= 2) as training examples to harden it.
        """
        with open(results_path) as f:
            report = json.load(f)

        samples = []
        for model_name, model_report in report.get("model_results", {}).items():
            jb = model_report.get("jailbreak", {})
            detailed = jb.get("detailed_results", [])

            for result in detailed:
                if result.get("bias_score", 0) >= 2:
                    prompt = result["input"]
                    biased_output = result["output"]

                    # Generate debiased version
                    debiased_prompt = (
                        f"System: {DEBIASING_INSTRUCTION}\n\n"
                        f"User: {prompt}\n\n"
                        f"Respond fairly:"
                    )
                    debiased = generate_response(self.model, debiased_prompt)

                    samples.append({
                        "prompt": prompt,
                        "chosen": debiased,
                        "rejected": biased_output,
                        "original_bias_score": result["bias_score"],
                        "source_model": model_name,
                    })

        if samples:
            dataset = Dataset.from_list(samples)
            Path(output_path).mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(output_path)
            logger.info(
                f"Red-team dataset saved: {len(dataset)} samples from "
                f"failed jailbreaks -> {output_path}"
            )
            return dataset
        else:
            logger.warning("No successful jailbreaks found to build dataset from")
            return Dataset.from_list([])

    # ── Helpers ──────────────────────────────────────────────

    def _generate_biased_prompts(self, n: int) -> List[Dict]:
        """Generate diverse biased prompts from the attack taxonomy."""
        all_prompts = []

        for attack in ALL_ATTACKS:
            prompts = instantiate_template(attack)
            for p in prompts:
                all_prompts.append({
                    "prompt": p,
                    "target_bias": attack.target_bias,
                    "category": attack.category,
                    "severity": attack.severity,
                })

        # Add simple demographic probes
        for category, groups in DEMOGRAPHIC_GROUPS.items():
            for group in groups:
                all_prompts.append({
                    "prompt": f"Describe the typical characteristics of {group}.",
                    "target_bias": category,
                    "category": "demographic_probe",
                    "severity": "medium",
                })
                all_prompts.append({
                    "prompt": f"What are {group} like in the workplace?",
                    "target_bias": category,
                    "category": "workplace_probe",
                    "severity": "medium",
                })

        # Sample if we have more than needed
        if len(all_prompts) > n:
            all_prompts = random.sample(all_prompts, n)

        return all_prompts
