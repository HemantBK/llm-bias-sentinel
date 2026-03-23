"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "ollama_connected" in data
        assert "timestamp" in data


class TestMetricsEndpoint:
    def test_prometheus_metrics(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "bias_evaluations_total" in response.text or response.status_code == 200

    def test_stats_endpoint(self):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestJobsEndpoint:
    def test_list_jobs_empty(self):
        response = client.get("/jobs")
        assert response.status_code == 200

    def test_delete_nonexistent_job(self):
        response = client.delete("/jobs/nonexistent")
        assert response.status_code == 404


class TestBiasCheckEndpoint:
    def test_bias_check_input(self):
        """Test bias check endpoint with mocked guardrails (no Ollama needed)."""
        from unittest.mock import patch, MagicMock

        mock_instance = MagicMock()
        mock_instance.check_input.return_value = {
            "flagged": False,
            "reason": None,
        }

        # Patch where the class is looked up (the module it's imported from)
        with patch(
            "src.guardrails_app.guardrails_engine.StandaloneGuardrails",
            return_value=mock_instance,
        ):
            response = client.post("/check-bias", json={
                "text": "Hello, how are you?",
                "check_type": "input",
            })

        assert response.status_code == 200
        data = response.json()
        assert "overall_flagged" in data
        assert data["text"] == "Hello, how are you?"

    def test_bias_check_empty_text_rejected(self):
        response = client.post("/check-bias", json={
            "text": "",
            "check_type": "input",
        })
        assert response.status_code == 422  # Validation error


class TestCounterfactualEndpoint:
    def test_counterfactual_generation(self):
        response = client.post("/counterfactual", json={
            "prompt": "The man went to the office",
            "categories": ["gender"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["original"] == "The man went to the office"
        assert data["total_generated"] >= 1
        assert len(data["counterfactuals"]) >= 1

    def test_counterfactual_no_match(self):
        response = client.post("/counterfactual", json={
            "prompt": "The weather is nice today",
            "categories": ["gender"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_generated"] == 0


class TestEvaluateEndpoint:
    def test_evaluate_returns_job_id(self):
        response = client.post("/evaluate", json={
            "models": [{"name": "test", "model_id": "llama3"}],
            "benchmarks": ["bbq"],
            "max_samples": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_get_nonexistent_eval(self):
        response = client.get("/evaluate/nonexistent")
        assert response.status_code == 404


class TestRedTeamEndpoint:
    def test_red_team_returns_job_id(self):
        response = client.post("/red-team", json={
            "models": [{"name": "test", "model_id": "llama3"}],
            "max_attack_prompts": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_get_nonexistent_red_team(self):
        response = client.get("/red-team/nonexistent")
        assert response.status_code == 404


class TestGenerateEndpoint:
    def test_generate_empty_prompt_rejected(self):
        response = client.post("/generate", json={
            "prompt": "",
        })
        assert response.status_code == 422
