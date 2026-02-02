# Installation Guide

This guide covers different installation methods for the Chemical Reaction Rate Prediction ML framework.

---

## Prerequisites

- Python 3.9 or higher
- pip or Poetry
- Git (for cloning repository)

### Check Python Version

```bash
python --version
# or
python3 --version
```

If you don't have Python 3.9+, download it from [python.org](https://www.python.org/downloads/).

---

## Method 1: Poetry (Recommended for Development)

Poetry provides better dependency management and virtual environment handling.

### 1.1 Install Poetry

```bash
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Add Poetry to PATH:

```bash
# macOS/Linux
export PATH="$HOME/.local/bin:$PATH"

# Windows: Add %APPDATA%\Python\Scripts to PATH
```

### 1.2 Clone and Install

```bash
# Clone repository
git clone https://github.com/sinsangwoo/Chemical-Reaction-Rate-Prediction-ML.git
cd Chemical-Reaction-Rate-Prediction-ML

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Verify installation
python -c "import src; print('Installation successful!')"
```

### 1.3 Install Development Tools

```bash
# Install with dev dependencies
poetry install --with dev,test

# Set up pre-commit hooks
poetry run pre-commit install
```

---

## Method 2: pip (Simple Installation)

### 2.1 Clone Repository

```bash
git clone https://github.com/sinsangwoo/Chemical-Reaction-Rate-Prediction-ML.git
cd Chemical-Reaction-Rate-Prediction-ML
```

### 2.2 Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2.3 Install Dependencies

```bash
pip install -r requirements.txt

# Verify installation
python -c "import src; print('Installation successful!')"
```

---

## Method 3: Development Installation with Makefile

For quick setup on Unix-like systems.

```bash
# Clone repository
git clone https://github.com/sinsangwoo/Chemical-Reaction-Rate-Prediction-ML.git
cd Chemical-Reaction-Rate-Prediction-ML

# Install everything (Poetry + dependencies + pre-commit)
make install

# Activate virtual environment
poetry shell
```

---

## Verification

### Test Installation

```bash
# Run tests
pytest tests/ -v

# Or with coverage
pytest --cov=src
```

### Run Example

```bash
# Generate data and train models
python src/main.py --generate-data
```

Expected output:
```
Chemical Reaction Rate Prediction ML
Modern Machine Learning Framework

Generating synthetic data...
✓ Data saved to data/chemical_reaction_data.csv

Loading data...
Train set: 240 samples
Test set: 60 samples

...
```

---

## Troubleshooting

### Issue: Python version too old

**Error**: `Python 3.9 or higher is required`

**Solution**:
```bash
# Install Python 3.11 (recommended)
# Visit https://www.python.org/downloads/

# Or use pyenv
pyenv install 3.11
pyenv local 3.11
```

### Issue: Poetry not found

**Error**: `poetry: command not found`

**Solution**:
```bash
# Make sure Poetry is in PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Issue: ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're in the project root
pwd  # Should show .../Chemical-Reaction-Rate-Prediction-ML

# And virtual environment is activated
which python  # Should show path to venv or Poetry env
```

### Issue: Pre-commit hooks failing

**Error**: Pre-commit hooks fail on commit

**Solution**:
```bash
# Update pre-commit
pre-commit autoupdate

# Run manually to see errors
pre-commit run --all-files

# Fix issues (usually formatting)
make format
```

### Issue: Tests failing

**Error**: Tests fail after installation

**Solution**:
```bash
# Ensure all dependencies installed
poetry install --with test

# Clear cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
rm -rf .pytest_cache

# Run tests again
pytest -v
```

---

## Optional: Install Jupyter

For interactive exploration:

```bash
poetry install --with dev
poetry run jupyter notebook
```

---

## Next Steps

After installation:

1. 📖 Read the [Quick Start Guide](../README_NEW.md#-quick-start)
2. 🧪 Try the examples in `notebooks/`
3. 🔧 Check [CONTRIBUTING.md](../CONTRIBUTING.md) for development
4. 🚀 Start building your own models!

---

## Uninstallation

### Remove Virtual Environment

```bash
# If using Poetry
poetry env remove python

# If using venv
deactivate
rm -rf venv/
```

### Remove Repository

```bash
cd ..
rm -rf Chemical-Reaction-Rate-Prediction-ML/
```

---

For more help, please [open an issue](https://github.com/sinsangwoo/Chemical-Reaction-Rate-Prediction-ML/issues).
