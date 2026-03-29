# Contributing to TurboQuant

## Development Setup

```bash
git clone https://github.com/BFinn/turboquant-vllm.git
cd turboquant-vllm
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest                    # all tests
pytest tests/test_quantizer.py  # specific module
pytest -k "needle"        # by keyword
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Pull Requests

1. Fork the repo and create a feature branch
2. Add tests for new functionality
3. Ensure all tests pass: `pytest`
4. Ensure code passes lint: `ruff check src/ tests/`
5. Submit a PR with a clear description of changes
