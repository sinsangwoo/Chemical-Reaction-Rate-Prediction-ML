# Makefile for chemical-reaction-ml project

.PHONY: help install test lint format clean

help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies using Poetry"
	@echo "  test       - Run tests with pytest"
	@echo "  lint       - Run code quality checks"
	@echo "  format     - Format code with black and ruff"
	@echo "  clean      - Remove build artifacts and cache"
	@echo "  run        - Run the main training pipeline"

install:
	poetry install --with dev,test
	poetry run pre-commit install

test:
	poetry run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	poetry run black --check src/ tests/
	poetry run ruff check src/ tests/
	poetry run mypy src/

format:
	poetry run black src/ tests/
	poetry run ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

run:
	poetry run python src/main.py
