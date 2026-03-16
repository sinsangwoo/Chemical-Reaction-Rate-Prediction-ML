# Changelog

All notable changes to this project will be documented in this file.
This project follows [Semantic Versioning](https://semver.org/).

---

## [2.0.0] — 2026-03-16

### Added — Phase 1: Multi-step Kinetic Engine

- `src/models/rpg/` — **Reaction Path Graph (RPG)** module
  - `ReactionPathGraph`: builds an elementary reaction network (DAG) from SMILES lists.
    Nodes represent reactants, intermediates, transition states, and products with
    Gibbs free energies; edges carry `activation_energy` (Ea) and `frequency_factor` (A).
  - `ElementaryReaction`: per-step data class with `rate_constant(T)` via Arrhenius.
  - `ReactionNode` / `NodeType` enum.
  - `mark_rate_determining_step()`: identifies the highest-Ea step.
  - `to_dict()`: JSON-serialisable output for API responses.
- `src/simulation/` — **Kinetic Monte Carlo (KMC) Solver**
  - `KMCSolver`: Gillespie-algorithm stochastic simulation over all network edges.
  - `KMCResult`: time-resolved concentration profiles (`times`, `concentrations`,
    `lower_ci`, `upper_ci`, `max_yield_time`, `max_yield`, `n_trajectories`).
  - `arrhenius_rate(A, Ea, T)` and `activation_energy_from_rates()` utilities.
  - Bayesian uncertainty propagation: Ea sampled from `N(Ea, σ²)` per trajectory;
    95 % CI computed over 50 independent Gillespie runs.
- `api/routes/simulation.py` — new FastAPI router
  - `POST /simulate/rpg`: build and return an RPG.
  - `POST /simulate/kmc`: run KMC and return time-series profiles + RPG graph.
- `tests/simulation/test_kmc_solver.py`, `test_rpg.py`: unit test suites.

### Added — Phase 2: GNN Model Upgrade

- `src/models/novel/hybrid_model_v2.py` — **MultiStepHybridGNN**
  - `KineticOutputHead`: Softplus-bounded MLP head for Ea and ln(A) prediction.
  - `predict_with_uncertainty(x, T)`: MC Dropout over 50 samples; returns
    `(k_mean, ea_mean, a_mean, ea_std, a_std)`.
  - `predict_kinetic_params_for_rpg(x, T)`: dict output compatible with
    `ReactionPathGraph.build_from_smiles()` keyword arguments.
  - Backward-compatible with v1 `HybridGNN` API.
- `src/models/novel/multi_task_loss.py` — **MultiTaskKineticLoss**
  - Kendall et al. (2018) homoscedastic uncertainty weighting.
  - Three objectives: Ea MSE, ln(A) MSE, log-k MSE.
  - Learnable `log_sigma` parameters for automatic loss balancing.
- `src/data/reaction_network_dataset.py` — **ReactionNetworkDataset**
  - `ReactionNetworkSample` dataclass with `ea_values`, `a_values` lists.
  - JSON loader, synthetic generator, and `split()` helper.
- `tests/models/test_hybrid_model_v2.py`, `tests/data/test_reaction_network_dataset.py`.

### Added — Phase 3: Insight-Driven UI

- `frontend/src/lib/simulation-api.ts`: typed fetch wrappers for `/simulate/rpg`
  and `/simulate/kmc`.
- `frontend/src/components/EnergyProfileChart.tsx`:
  - Recharts `AreaChart` Gibbs energy profile.
  - Rate-determining step red-dashed `ReferenceLine`.
  - Bayesian CI as a shaded gradient band.
  - Node-type colour coding (reactant/intermediate/TS/product).
- `frontend/src/components/ConcentrationDashboard.tsx`:
  - Time-resolved concentration chart from KMC output.
  - Per-species 95 % CI shadow areas.
  - `max_yield_time` green marker annotation.
- `frontend/src/components/ReactionPathGraph.tsx`:
  - SVG node-link network diagram (no D3 dependency).
  - Click-to-inspect node detail panel.
  - RDS edge highlighted with red dashed arrow.
- `frontend/src/components/SimulationPanel.tsx`:
  - Integrating panel with input form + all three visualisation components.
  - Single `/simulate/kmc` call returns both KMC data and RPG graph.

### Added — Phase 4: Integration & Documentation

- `tests/integration/test_kmc_pipeline.py`: E2E tests (ethanol dehydrogenation,
  single-step reaction, temperature-effect regression).
- `docs/ARCHITECTURE.md`: full system architecture guide for v2.
- `CHANGELOG.md`: this file.

---

## [1.0.0] — 2025-07-20

### Added
- 8 ML models: RandomForest, XGBoost, GCN, GAT, GIN, MPNN, Bayesian GNN, Deep Ensemble.
- Bayesian uncertainty quantification (MC Dropout, BBB, Conformal Prediction).
- Hybrid Physics-GNN (Arrhenius-GNN architecture).
- Few-shot learning (MAML) for rare reaction types.
- Interpretable AI (attention mechanism + integrated gradients).
- Industry-specific transfer learning (pharma, agrochem, polymer, specialty).
- FastAPI REST backend with JWT authentication.
- React + TypeScript frontend with Recharts visualisations.
- Docker / Railway / Vercel deployment configurations.
- CI/CD via GitHub Actions.
