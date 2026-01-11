.PHONY: test test-cov test-verbose install install-dev clean lint format help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dependencies
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"

install-test: ## Install package with test dependencies
	pip install -e ".[test]"

test: ## Run tests
	pytest

test-verbose: ## Run tests with verbose output
	pytest -v

test-cov: ## Run tests with coverage report
	pytest --cov=trk_algorithms --cov-report=html --cov-report=term

test-fast: ## Run tests excluding slow tests
	pytest -m "not slow"

test-utils: ## Run only utils tests
	pytest tests/test_utils.py -v

lint: ## Run linter (pylint)
	pylint trk_algorithms/

format: ## Format code with autopep8
	autopep8 --in-place --recursive trk_algorithms/

clean: ## Clean up generated files
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist

watch: ## Run tests continuously (requires pytest-watch)
	pytest-watch

.DEFAULT_GOAL := help
