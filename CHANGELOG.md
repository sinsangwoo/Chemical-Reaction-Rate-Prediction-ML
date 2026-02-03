# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 2 (Planned)
- Integration with real chemical datasets (USPTO, ORD)
- SMILES notation support
- RDKit molecular descriptors
- Enhanced data preprocessing pipeline

### Phase 3 (Planned)
- Graph Neural Network implementation
- Transformer-based models
- Multi-modal fusion architecture

---

## [0.1.0] - 2025-02-02

### Added - Phase 1: Foundation

#### Project Structure
- Modern modular architecture with `src/` layout
- Separation of concerns: data, models, evaluation, utils
- Configuration management with YAML
- Comprehensive `.gitignore` for Python projects

#### Dependency Management
- `pyproject.toml` with Poetry configuration
- `requirements.txt` for pip compatibility
- Locked dependencies for reproducibility
- Support for Python 3.9, 3.10, 3.11

#### Code Quality
- Black formatter (line length: 100)
- Ruff linter with custom rules
- MyPy type checking
- Pre-commit hooks for automated checks

#### Data Module
- `ArrheniusDataGenerator`: Physics-based data generation
- `ReactionDataLoader`: Data loading and preprocessing
- Configurable parameters (temperature, concentration ranges)
- Statistical analysis utilities

#### Models
- `BaseReactionModel`: Abstract base class for all models
- `LinearRegressionModel`: Simple baseline
- `PolynomialRegressionModel`: Configurable degree
- `SVRModel`: Support Vector Regression
- `RandomForestModel`: Ensemble method with feature importance
- Model saving/loading functionality

#### Evaluation
- `RegressionMetrics`: Comprehensive metrics (MAE, RMSE, R², MAPE)
- Model comparison utilities
- Pretty-printed results tables

#### Visualization
- Cross-validation results (boxplot)
- Prediction vs actual scatter plots
- Feature importance bar charts
- Residual analysis plots
- High-quality figure export (300 DPI)

#### Testing
- 20+ unit tests with pytest
- Test coverage >80%
- Fixtures for sample data
- Parametrized tests for multiple scenarios

#### CI/CD
- GitHub Actions workflow
- Multi-version Python testing (3.9, 3.10, 3.11)
- Automated linting and type checking
- Test coverage reporting
- Build artifact generation

#### CLI Interface
- Rich terminal interface with colors and progress
- Command-line arguments for customization
- Pretty-printed tables and results
- User-friendly error messages

#### Documentation
- Comprehensive README with badges and examples
- Contributing guidelines (CONTRIBUTING.md)
- Detailed docstrings (Google style)
- MIT License
- Project roadmap (6 phases)

#### Development Tools
- Makefile with common tasks
- Poetry scripts for automation
- VS Code configuration (recommended)

### Changed

- Refactored monolithic `coool.py` into modular components
- Improved code organization and separation of concerns
- Enhanced error handling and validation
- Better naming conventions (PEP 8 compliant)

### Deprecated

- `coool.py` (kept for reference, will be removed in v0.2.0)
- Old README.md (replaced with README_NEW.md)

### Fixed

- Korean font issues in matplotlib (cross-platform support)
- Hardcoded paths replaced with configurable settings
- Magic numbers extracted to configuration

### Security

- No known security issues
- Dependencies locked to specific versions
- Pre-commit hooks prevent common security issues

---

## [0.0.1] - 2024 (Legacy)

### Initial Release

- Basic chemical reaction rate prediction
- Single script implementation (`coool.py`)
- Arrhenius equation-based data generation
- Random Forest model training
- Basic visualization
- High school chemistry project

---

## Comparison: v0.0.1 vs v0.1.0

| Aspect | v0.0.1 (Legacy) | v0.1.0 (Phase 1) |
|--------|-----------------|------------------|
| Lines of Code | ~200 (1 file) | ~1500+ (20+ files) |
| Test Coverage | 0% | >80% |
| Documentation | Basic README | Comprehensive docs |
| Code Quality | No standards | Black+Ruff+MyPy |
| CI/CD | None | GitHub Actions |
| Modularity | Monolithic | Highly modular |
| Extensibility | Limited | Designed for growth |
| Production-Ready | No | Yes |

---

[Unreleased]: https://github.com/sinsangwoo/Chemical-Reaction-Rate-Prediction-ML/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sinsangwoo/Chemical-Reaction-Rate-Prediction-ML/releases/tag/v0.1.0
[0.0.1]: https://github.com/sinsangwoo/Chemical-Reaction-Rate-Prediction-ML/releases/tag/v0.0.1
