# Contributing to LLM Bias Sentinel

Thank you for your interest in contributing! This project aims to make AI fairness testing accessible to everyone.

## How to Contribute

### Reporting Bugs
- Open a [GitHub Issue](https://github.com/HemantBK/llm-bias-sentinel/issues/new?template=bug_report.md)
- Include your Python version, OS, and Ollama version
- Provide steps to reproduce the issue

### Suggesting Features
- Open a [Feature Request](https://github.com/HemantBK/llm-bias-sentinel/issues/new?template=feature_request.md)
- Describe the use case and expected behavior

### Submitting Code

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/llm-bias-sentinel.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Install dev dependencies**: `pip install -e ".[dev]"`
5. **Make your changes**
6. **Run tests**: `make test-quick`
7. **Run linter**: `make lint`
8. **Commit** with a descriptive message
9. **Push** and open a Pull Request

### Code Style

- **Formatter**: Black (100 char line length)
- **Linter**: Ruff
- **Type hints**: Required for public functions
- **Docstrings**: Google style

### Areas We Need Help With

- Adding new bias benchmarks (WinoBias, TruthfulQA)
- Multilingual bias evaluation support
- Better BBQ answer parsing for diverse model output formats
- Improving the local LLM judge accuracy
- Adding more red-team attack templates
- Dashboard UI improvements
- Documentation translations

## Development Setup

```bash
git clone https://github.com/HemantBK/llm-bias-sentinel.git
cd llm-bias-sentinel
python -m venv venv
venv\Scripts\activate  # Windows
pip install -e ".[dev]"
make test-quick
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
