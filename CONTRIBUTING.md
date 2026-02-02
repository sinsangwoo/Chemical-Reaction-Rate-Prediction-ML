# Contributing to Chemical Reaction Rate Prediction ML

Thank you for your interest in contributing! This document provides guidelines and workflows for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

Be respectful, inclusive, and constructive. We're building a tool for the scientific community.

---

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/Chemical-Reaction-Rate-Prediction-ML.git
cd Chemical-Reaction-Rate-Prediction-ML
```

### 2. Set Up Development Environment

```bash
# Install Poetry if you haven't
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --with dev,test

# Activate virtual environment
poetry shell

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

---

## Development Workflow

### Making Changes

1. **Write Code**: Make your changes in the appropriate module
2. **Add Tests**: Write tests for new functionality
3. **Run Tests**: Ensure all tests pass
4. **Format Code**: Use black and ruff
5. **Type Check**: Run mypy
6. **Commit**: Write clear commit messages

### Quick Commands

```bash
# Run tests
make test

# Format code
make format

# Run linters
make lint

# All quality checks
make format && make lint && make test
```

---

## Coding Standards

### Python Style

- **Formatter**: Black (line length: 100)
- **Linter**: Ruff
- **Type Checker**: MyPy
- **Docstrings**: Google style

### Example

```python
def calculate_reaction_rate(
    temperature: float,
    concentration: float,
    catalyst: bool = False
) -> float:
    """Calculate reaction rate using Arrhenius equation.

    Args:
        temperature: Temperature in Celsius
        concentration: Concentration in mol/L
        catalyst: Whether catalyst is used

    Returns:
        Reaction rate in mol/L·s

    Raises:
        ValueError: If temperature is below absolute zero
    """
    if temperature < -273.15:
        raise ValueError("Temperature cannot be below absolute zero")
    # implementation...
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add graph neural network model
fix: correct activation energy calculation
docs: update installation instructions
test: add tests for data loader
refactor: simplify model training loop
perf: optimize feature extraction
```

---

## Testing

### Writing Tests

- **Location**: `tests/` directory
- **Naming**: `test_*.py`
- **Framework**: pytest
- **Coverage**: Aim for >80%

### Test Structure

```python
import pytest
from src.models.your_model import YourModel

class TestYourModel:
    """Test suite for YourModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = YourModel()
        assert model is not None

    def test_training(self, sample_data):
        """Test model training."""
        model = YourModel()
        model.train(sample_data)
        assert model.is_trained
```

### Running Tests

```bash
# All tests
poetry run pytest

# Specific file
poetry run pytest tests/test_models.py

# With coverage
poetry run pytest --cov=src --cov-report=html

# Verbose
poetry run pytest -v
```

---

## Pull Request Process

### Before Submitting

1. ✅ All tests pass
2. ✅ Code is formatted (black, ruff)
3. ✅ Type checks pass (mypy)
4. ✅ No linter warnings
5. ✅ Documentation updated if needed
6. ✅ CHANGELOG.md updated (if applicable)

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe tests added/modified

## Checklist
- [ ] Tests pass locally
- [ ] Code formatted with black
- [ ] Linters pass (ruff, mypy)
- [ ] Documentation updated
- [ ] Self-review completed
```

### Review Process

1. **Automated Checks**: CI pipeline must pass
2. **Code Review**: At least one maintainer approval
3. **Discussion**: Address feedback promptly
4. **Merge**: Squash and merge (clean history)

---

## Project Structure

Understanding the architecture:

```
src/
├── data/           # Data generation and loading
│   ├── data_generator.py
│   └── data_loader.py
├── models/         # ML models
│   ├── base_model.py
│   └── traditional_models.py
├── evaluation/     # Metrics and evaluation
│   └── metrics.py
├── utils/          # Utilities
│   └── visualization.py
└── main.py         # Main pipeline
```

### Adding New Models

1. Create class inheriting from `BaseReactionModel`
2. Implement `train()` and `predict()` methods
3. Add to `src/models/`
4. Write tests in `tests/test_models.py`
5. Update documentation

### Adding New Features

1. Create module in appropriate directory
2. Follow existing code patterns
3. Add comprehensive tests
4. Update configuration if needed
5. Document in README

---

## Questions?

Feel free to:
- Open an issue for discussion
- Ask in pull request comments
- Check existing issues/PRs for similar questions

---

**Thank you for contributing! 🧪🔬**
