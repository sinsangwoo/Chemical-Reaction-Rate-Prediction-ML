# Chemical Reaction Rate Prediction Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine learning platform for predicting chemical reaction kinetics. Combines Graph Neural Networks with physics-informed learning to model reaction rates, activation energies, and time-resolved concentration profiles.

---

## What it does

**v1 — Rate prediction**
Given reactants, products, and conditions, predict the reaction rate constant k with calibrated uncertainty.

**v2 — Multi-step kinetic simulation** *(new)*
Build an elementary reaction network (Reaction Path Graph), identify the rate-determining step, and run a Kinetic Monte Carlo simulation to compute time-resolved concentration profiles with Bayesian confidence intervals.

---

## Installation

```bash
git clone https://github.com/sinsangwoo/Chemical-Reaction-Rate-Prediction-ML.git
cd Chemical-Reaction-Rate-Prediction-ML
pip install -r requirements.txt
```

Optional — initialize the database:
```bash
python -c "from api.database import init_db; init_db()"
```

Frontend:
```bash
cd frontend && npm install && npm run dev
```

Requirements: Python 3.10+, PyTorch 2.0+, RDKit, Node.js 18+

---

## Quick start

**Rate prediction (v1)**

```python
from src.models.gnn import GINModel
import torch

model = GINModel(node_features=37, hidden_dim=128)
x = torch.randn(1, 37)
k = model(x)
print(f"k = {k.item():.4f} mol/L·s")
```

**KMC simulation (v2)**

```python
from src.models.rpg import ReactionPathGraph
from src.simulation import KMCSolver

rpg = ReactionPathGraph(temperature=500.0)
rpg.build_from_smiles(
    reactants=["CCO"],
    products=["CC=O"],
    intermediates=["CC[OH2+]"],
    ea_values=[80.0, 55.0],   # kJ/mol
    a_values=[1e13, 1e12],
)
rpg.mark_rate_determining_step()

solver = KMCSolver(rpg, max_time=1e-6, n_trajectories=50)
result = solver.run({"n0": 1.0})
print(f"Max yield {result.max_yield:.1%} at t = {result.max_yield_time:.2e} s")
```

**REST API**

```bash
uvicorn api.main:app --reload

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"reaction": {"reactants": ["CCO"], "products": ["CC=O"], "conditions": {"temperature": 500}}, "model_type": "gin"}'

# KMC simulation
curl -X POST http://localhost:8000/simulate/kmc \
  -H "Content-Type: application/json" \
  -d '{"reactants": ["CCO"], "products": ["CC=O"], "intermediates": ["CC[OH2+]"], "temperature": 500, "max_time": 1e-6, "n_trajectories": 50}'
```

---

## Architecture

```
React Frontend
  SimulationPanel
    ├── ReactionPathGraph      SVG node-link network, RDS highlighted
    ├── EnergyProfileChart     Gibbs ΔG profile, CI band, RDS marker
    └── ConcentrationDashboard KMC time-series, 95 % CI shadow areas
  [legacy] PredictionTab | AnalyticsTab | ModelsTab
        │
        │ REST API
        ▼
FastAPI Backend
  POST /predict          v1 rate prediction
  POST /simulate/rpg     build Reaction Path Graph
  POST /simulate/kmc     run KMC simulation
        │
        ├── src/models/rpg/        ReactionPathGraph, ElementaryReaction
        ├── src/simulation/        KMCSolver (Gillespie), arrhenius_utils
        └── src/models/novel/      MultiStepHybridGNN, MultiTaskKineticLoss
```

---

## Models

| Model | R² | MAE | Inference |
|---|---|---|---|
| GIN | 0.985 | 0.050 | 52 ms |
| MPNN | 0.940 | 0.080 | 65 ms |
| GAT | 0.930 | 0.090 | 58 ms |
| GCN | 0.910 | 0.110 | 48 ms |
| XGBoost | 0.850 | 0.130 | 28 ms |
| RandomForest | 0.820 | 0.150 | 22 ms |
| Bayesian Ensemble | 0.985 | 0.052 | 245 ms |

Benchmarked on the USPTO dataset.

---

## Key modules

### Reaction Path Graph (`src/models/rpg`)

Builds a directed acyclic graph of elementary steps from SMILES input. Each edge carries GNN-predicted Ea and A values; `mark_rate_determining_step()` identifies the highest-barrier step.

### KMC Solver (`src/simulation`)

Gillespie-algorithm simulator. Runs N independent trajectories with Ea sampled from the Bayesian uncertainty distribution, then aggregates to produce mean concentrations and 95 % confidence intervals. Returns `max_yield_time` — the optimal reaction time at the given temperature.

### MultiStepHybridGNN (`src/models/novel/hybrid_model_v2.py`)

Extends the v1 HybridGNN with two output heads:
- **Ea head** — activation energy per elementary step (kJ/mol), lower-bounded via Softplus
- **ln(A) head** — log pre-exponential factor, exponentiated before use

`predict_with_uncertainty(x, T)` runs MC Dropout over 50 samples and returns `(k_mean, Ea_mean, A_mean, Ea_std, A_std)`. Output is directly compatible with `ReactionPathGraph.build_from_smiles()`.

### MultiTaskKineticLoss (`src/models/novel/multi_task_loss.py`)

Kendall et al. (2018) homoscedastic uncertainty weighting over three objectives: Ea MSE, ln(A) MSE, and log-k MSE. Loss weights are learned automatically during training.

---

## Project structure

```
.
├── api/
│   ├── main.py
│   ├── routes/
│   │   └── simulation.py       POST /simulate/rpg, POST /simulate/kmc
│   ├── models.py
│   ├── database.py
│   └── auth.py
├── src/
│   ├── models/
│   │   ├── rpg/                ReactionPathGraph, ElementaryReaction
│   │   ├── gnn/                GCN, GAT, GIN, MPNN
│   │   ├── uncertainty/        MC Dropout, Bayesian, Conformal
│   │   └── novel/
│   │       ├── hybrid_model.py        v1 HybridGNN
│   │       ├── hybrid_model_v2.py     v2 MultiStepHybridGNN
│   │       ├── multi_task_loss.py
│   │       ├── few_shot_learning.py
│   │       ├── interpretable_gnn.py
│   │       └── industry_finetuning.py
│   ├── simulation/
│   │   ├── kmc_solver.py
│   │   └── arrhenius_utils.py
│   └── data/
│       └── reaction_network_dataset.py
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── SimulationPanel.tsx
│       │   ├── ReactionPathGraph.tsx
│       │   ├── EnergyProfileChart.tsx
│       │   └── ConcentrationDashboard.tsx
│       └── lib/
│           ├── api.ts
│           └── simulation-api.ts
├── tests/
│   ├── simulation/
│   │   ├── test_kmc_solver.py
│   │   └── test_rpg.py
│   ├── models/
│   │   └── test_hybrid_model_v2.py
│   ├── data/
│   │   └── test_reaction_network_dataset.py
│   └── integration/
│       └── test_kmc_pipeline.py
├── experiments/
│   ├── benchmark.py
│   ├── statistical_analysis.py
│   └── ablation_study.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── DEPLOYMENT.md
└── configs/
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Single reaction rate prediction |
| POST | `/predict/batch` | Batch prediction (up to 100) |
| GET | `/models` | List available models |
| POST | `/simulate/rpg` | Build Reaction Path Graph |
| POST | `/simulate/kmc` | Run KMC simulation |
| GET | `/health` | Health check |

Interactive docs at `http://localhost:8000/docs` after starting the server.

---

## Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/models/ tests/simulation/ tests/data/ -v

# Integration
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Deployment

**Docker**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

**Railway**
```bash
npm i -g @railway/cli && railway login && railway up
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for AWS and Vercel guides.

---

## Stack

| Layer | Technology |
|---|---|
| ML | PyTorch 2.0+, PyTorch Geometric |
| Chemistry | RDKit |
| Backend | FastAPI, SQLAlchemy, PostgreSQL |
| Auth | JWT (python-jose) |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Recharts |
| DevOps | Docker, GitHub Actions, Railway |

---

## Citation

```bibtex
@software{chemical_reaction_ml_platform,
  title  = {Chemical Reaction Rate Prediction Platform},
  author = {Sin Sang Woo},
  year   = {2026},
  url    = {https://github.com/sinsangwoo/Chemical-Reaction-Rate-Prediction-ML}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
