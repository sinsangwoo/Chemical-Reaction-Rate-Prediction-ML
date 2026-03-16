# Architecture Guide — v2.0

This document describes the system architecture introduced in **v2.0** of the
Chemical Reaction Rate Prediction Platform.  It covers the new multi-step
kinetic engine (Phase 1 + 2) and the Insight-Driven UI layer (Phase 3).

---

## Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                   React Frontend (TypeScript)                        │
│  SimulationPanel                                                     │
│    ├─ ReactionPathGraph    (SVG node-link, RDS highlighted)           │
│    ├─ EnergyProfileChart   (Recharts, Gibbs ΔG, CI band)             │
│    └─ ConcentrationDashboard (KMC time-series, 95% CI shadow)        │
│  [legacy] PredictionTab | AnalyticsTab | ModelsTab                   │
└─────────────────────────────────┬──────────────────────────────┘
                                 │ REST API (JSON)
┌─────────────────────────────────┴──────────────────────────────┐
│                   FastAPI Backend                                     │
│  /predict         [v1, legacy GNN prediction]                        │
│  /simulate/rpg    [v2, build Reaction Path Graph]                    │
│  /simulate/kmc    [v2, run KMC + return CI profiles]                 │
└─────────────────────────────────┬──────────────────────────────┘
                                 │
           ┌────────────────┤
           │                   │
┌──────────┴───────┐ ┌───┴──────────┐
│  src/models/rpg   │ │ src/simulation  │
│  ReactionPathGraph│ │  KMCSolver      │
│  ElementaryRxn   │ │  ArrheniusUtils │
└─────────┬──────┘ └───────────────┘
           │ GNN prediction (optional)
┌─────────┴───────────┐
│ src/models/novel  │
│  MultiStepHybridGNN │
│  MultiTaskKineticLoss│
└───────────────────┘
```

---

## Module Reference

### `src/models/rpg` — Reaction Path Graph

| Class | Purpose |
|---|---|
| `NodeType` | Enum: `reactant`, `intermediate`, `transition_state`, `product` |
| `ReactionNode` | Species node: `smiles`, `gibbs_free_energy`, `uncertainty` |
| `ElementaryReaction` | Edge: `activation_energy` (kJ/mol), `frequency_factor` (s⁻¹), `is_rate_determining` |
| `ReactionPathGraph` | Builder + serialiser. `build_from_smiles()` → `mark_rate_determining_step()` → `to_dict()` |

**Key method: `build_from_smiles`**

```python
rpg = ReactionPathGraph(temperature=500.0)
rpg.build_from_smiles(
    reactants=["CCO"],
    products=["CC=O"],
    intermediates=["CC[OH2+]"],
    ea_values=[80.0, 55.0],        # kJ/mol per step
    a_values=[1e13, 1e12],         # s^-1 per step
    ea_uncertainties=[4.0, 2.75],  # Bayesian 1-sigma
)
rpg.mark_rate_determining_step()
print(rpg.summary())
```

---

### `src/simulation` — Kinetic Monte Carlo

| Function / Class | Purpose |
|---|---|
| `arrhenius_rate(A, Ea, T)` | k(T) = A · exp(−Ea/RT) |
| `activation_energy_from_rates(k1, T1, k2, T2)` | Back-calculate Ea from two temperature points |
| `KMCSolver` | Gillespie-algorithm simulator; returns `KMCResult` |
| `KMCResult` | `times`, `concentrations`, `lower_ci`, `upper_ci`, `max_yield_time`, `max_yield` |

**Uncertainty propagation:**  `KMCSolver` draws `Ea ~ N(Ea, σ²)` for each trajectory,
so the 95 % CI in `KMCResult` reflects Bayesian uncertainty from the GNN.

```python
solver = KMCSolver(rpg, max_time=1e-6, n_snapshots=200, n_trajectories=50)
result = solver.run({"n0": 1.0})     # n0 = first-node id
print(result.max_yield_time)         # optimal reaction time
```

---

### `src/models/novel/hybrid_model_v2.py` — MultiStepHybridGNN

Extends `HybridGNN` with two kinetic heads.

| Head | Output | Constraint |
|---|---|---|
| `ea_head` | Ea (kJ/mol) | Softplus + min_val ≥ 1 |
| `log_a_head` | ln(A) | Softplus; A = exp(ln(A)) ≥ 1 |
| `rate_head` | k (legacy) | Softplus |

```python
model = MultiStepHybridGNN(node_features=37, hidden_dim=128)
k_mean, ea_mean, a_mean, ea_std, a_std = model.predict_with_uncertainty(
    x_batch, temperature=500.0
)
# Pass directly to RPG:
rpg.build_from_smiles(..., **model.predict_kinetic_params_for_rpg(x, T=500))
```

---

### `api/routes/simulation.py` — Simulation Endpoints

#### `POST /simulate/rpg`

```json
{
  "reactants": ["CCO"],
  "products": ["CC=O"],
  "intermediates": ["CC[OH2+]"],
  "temperature": 500
}
```
Returns `RPGResponse` with `nodes`, `edges`, `rate_determining_step`.

#### `POST /simulate/kmc`

```json
{
  "reactants": ["CCO"],
  "products": ["CC=O"],
  "intermediates": ["CC[OH2+]"],
  "temperature": 500,
  "max_time": 1e-6,
  "n_trajectories": 50
}
```
Returns `KMCResponse` with `times`, `concentrations`, `lower_ci`, `upper_ci`,
`max_yield_time`, `max_yield`, plus the full `graph` (RPG data).

---

### Frontend Components

| Component | File | Description |
|---|---|---|
| `SimulationPanel` | `SimulationPanel.tsx` | Top-level panel; owns API state |
| `ReactionPathGraph` | `ReactionPathGraph.tsx` | SVG node-link network; click for details |
| `EnergyProfileChart` | `EnergyProfileChart.tsx` | Gibbs ΔG profile; RDS annotation |
| `ConcentrationDashboard` | `ConcentrationDashboard.tsx` | KMC time-series; CI shadow areas |
| `simulationApi` | `simulation-api.ts` | Typed fetch wrappers for both endpoints |

---

## Data Flow

```
User inputs SMILES + T + max_time
        ↓
SimulationPanel.handleRun()
        ↓
POST /simulate/kmc
        ↓
FastAPI: builds RPG → runs KMCSolver (50 Gillespie trajectories)
        ↓
KMCResponse { times, concentrations, lower_ci, upper_ci,
              max_yield_time, max_yield, graph: RPGResponse }
        ↓
React state update
        ↓
├─ ReactionPathGraph    renders nodes/edges
├─ EnergyProfileChart   renders ΔG + RDS marker
└─ ConcentrationDashboard renders time-series + CI
```

---

## Running the v2 Stack

```bash
# Backend
uvicorn api.main:app --reload
# Frontend
cd frontend && npm install && npm run dev
```

Example KMC request:

```bash
curl -X POST http://localhost:8000/simulate/kmc \
  -H 'Content-Type: application/json' \
  -d '{
    "reactants": ["CCO"],
    "products": ["CC=O"],
    "intermediates": ["CC[OH2+]"],
    "temperature": 500,
    "max_time": 1e-6,
    "n_trajectories": 50
  }'
```
