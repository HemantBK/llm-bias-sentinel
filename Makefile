.PHONY: setup install ollama-pull test test-quick lint format security \
       benchmarks benchmarks-quick red-team red-team-quick guardrails-test \
       fairness-card api docker-up docker-down clean help

# ─── Help ───────────────────────────────────────────────
help:
	@echo "LLM Bias Sentinel - Available Commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup            Full setup (install + pull models)"
	@echo "    make install          Install Python dependencies"
	@echo "    make ollama-pull      Pull required Ollama models"
	@echo ""
	@echo "  Development:"
	@echo "    make test             Run full test suite (61 tests)"
	@echo "    make test-quick       Run tests that don't need Ollama"
	@echo "    make lint             Lint with ruff"
	@echo "    make format           Format with black"
	@echo "    make security         Run bandit security scan"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make benchmarks       Run all bias benchmarks"
	@echo "    make benchmarks-quick Run quick benchmark subset"
	@echo "    make red-team         Full red-team assessment"
	@echo "    make red-team-quick   Quick red-team scan"
	@echo "    make guardrails-test  Test guardrails effectiveness"
	@echo "    make fairness-card    Generate model fairness card"
	@echo ""
	@echo "  Server:"
	@echo "    make api              Start FastAPI server (port 8000)"
	@echo "    make docker-up        Start all services (API + Prometheus + Grafana)"
	@echo "    make docker-down      Stop all services"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean            Remove caches and build artifacts"

# ─── Setup ───────────────────────────────────────────────
setup: install ollama-pull
	@cp -n .env.example .env 2>/dev/null || true
	@echo ""
	@echo "Setup complete!"
	@echo "  1. Edit .env if needed"
	@echo "  2. Ensure Ollama is running: ollama serve"
	@echo "  3. Run: make test-quick"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

ollama-pull:
	ollama pull llama3
	ollama pull mistral

# ─── Development ─────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-quick:
	pytest tests/test_config.py tests/test_attack_taxonomy.py \
	       tests/test_prompt_templates.py tests/test_mitigation.py \
	       tests/test_monitoring.py -v --tb=short

lint:
	ruff check src/ api/ tests/ compliance/ || python -m py_compile src/config.py

format:
	black src/ api/ tests/ compliance/

security:
	bandit -r src/ api/ -ll --skip B101

# ─── Evaluation ──────────────────────────────────────────
benchmarks:
	python run.py benchmarks

benchmarks-quick:
	python run.py benchmarks --quick

red-team:
	python run.py red-team

red-team-quick:
	python run.py red-team --quick

guardrails-test:
	python run.py guardrails-test

fairness-card:
	python run.py fairness-card

# ─── API ─────────────────────────────────────────────────
api:
	python run.py api

# ─── Docker ──────────────────────────────────────────────
docker-up:
	docker-compose up -d
	@echo ""
	@echo "Services running:"
	@echo "  API:        http://localhost:8000"
	@echo "  Swagger:    http://localhost:8000/docs"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000 (admin/biassentinel)"

docker-down:
	docker-compose down

# ─── Cleanup ─────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ dist/ build/ *.egg-info/ .ruff_cache/
