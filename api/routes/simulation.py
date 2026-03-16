"""FastAPI routes for the KMC simulation engine.

Endpoints
---------
POST /simulate/rpg
    Build a Reaction Path Graph and return its node/edge structure.
POST /simulate/kmc
    Run a Kinetic Monte Carlo simulation on a reaction network and
    return time-resolved concentration profiles with confidence intervals.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.models.rpg import ReactionPathGraph
from src.simulation import KMCSolver

router = APIRouter(prefix="/simulate", tags=["simulation"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class RPGRequest(BaseModel):
    """Input for building a Reaction Path Graph."""

    reactants: List[str] = Field(
        ..., description="SMILES strings for reactant species", min_items=1
    )
    products: List[str] = Field(
        ..., description="SMILES strings for product species", min_items=1
    )
    intermediates: Optional[List[str]] = Field(
        default=None, description="Optional SMILES for intermediate species"
    )
    transition_states: Optional[List[str]] = Field(
        default=None, description="Optional labels for transition states"
    )
    activation_energies: Optional[List[float]] = Field(
        default=None,
        description="Ea values (kJ/mol) for each elementary step",
    )
    frequency_factors: Optional[List[float]] = Field(
        default=None, description="Pre-exponential factors A (s^-1) per step"
    )
    temperature: float = Field(
        default=298.15, description="Temperature in Kelvin", gt=0
    )


class RPGResponse(BaseModel):
    """Serialised Reaction Path Graph."""

    nodes: list
    edges: list
    temperature: float
    n_nodes: int
    n_edges: int
    rate_determining_step: Optional[str] = None


class KMCRequest(BaseModel):
    """Input for a KMC simulation."""

    reactants: List[str] = Field(..., min_items=1)
    products: List[str] = Field(..., min_items=1)
    intermediates: Optional[List[str]] = None
    activation_energies: Optional[List[float]] = None
    frequency_factors: Optional[List[float]] = None
    ea_uncertainties: Optional[List[float]] = Field(
        default=None,
        description="1-sigma Bayesian uncertainty on each Ea (kJ/mol)",
    )
    temperature: float = Field(default=298.15, gt=0)
    initial_concentrations: Optional[Dict[str, float]] = Field(
        default=None,
        description="node_id → initial concentration (mol/L); missing IDs default to 0",
    )
    max_time: float = Field(
        default=1.0, description="Simulation duration (s)", gt=0
    )
    n_snapshots: int = Field(default=200, ge=10, le=1000)
    n_trajectories: int = Field(
        default=50, ge=1, le=500,
        description="Number of Gillespie trajectories for CI estimation"
    )


class KMCResponse(BaseModel):
    """Time-resolved concentration profiles from the KMC simulation."""

    times: List[float]
    species_ids: List[str]
    concentrations: Dict[str, List[float]]
    lower_ci: Dict[str, List[float]]
    upper_ci: Dict[str, List[float]]
    max_yield_time: float
    max_yield: float
    n_trajectories: int
    rate_determining_step: Optional[str] = None
    graph: Optional[dict] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_rpg(req: RPGRequest) -> ReactionPathGraph:
    rpg = ReactionPathGraph(temperature=req.temperature)
    rpg.build_from_smiles(
        reactants=req.reactants,
        products=req.products,
        intermediates=req.intermediates,
        transition_states=req.transition_states,
        ea_values=req.activation_energies,
        a_values=req.frequency_factors,
    )
    rpg.mark_rate_determining_step()
    return rpg


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/rpg", response_model=RPGResponse, summary="Build Reaction Path Graph")
def build_rpg(req: RPGRequest) -> RPGResponse:
    """Construct an elementary reaction network and return the graph.

    The graph includes node Gibbs energies and edge Ea/A values so the
    frontend can render the Interactive Energy Profile.
    """
    try:
        rpg = _build_rpg(req)
        data = rpg.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    rds = next((e["id"] for e in data["edges"] if e.get("is_rate_determining")), None)
    return RPGResponse(
        nodes=data["nodes"],
        edges=data["edges"],
        temperature=data["temperature"],
        n_nodes=data["n_nodes"],
        n_edges=data["n_edges"],
        rate_determining_step=rds,
    )


@router.post("/kmc", response_model=KMCResponse, summary="Run KMC simulation")
def run_kmc(req: KMCRequest) -> KMCResponse:
    """Run a Kinetic Monte Carlo simulation on the reaction network.

    Returns time-resolved concentration profiles for all species plus
    95 % Bayesian confidence intervals derived from perturbed Ea samples.

    The ``max_yield_time`` field indicates the optimal reaction time for
    maximum product yield at the given temperature.
    """
    try:
        rpg_req = RPGRequest(
            reactants=req.reactants,
            products=req.products,
            intermediates=req.intermediates,
            activation_energies=req.activation_energies,
            frequency_factors=req.frequency_factors,
            temperature=req.temperature,
        )
        rpg = _build_rpg(rpg_req)

        if req.ea_uncertainties:
            for i, edge in enumerate(rpg.edges):
                if i < len(req.ea_uncertainties):
                    edge.ea_uncertainty = req.ea_uncertainties[i]

        solver = KMCSolver(
            rpg,
            max_time=req.max_time,
            n_snapshots=req.n_snapshots,
            n_trajectories=req.n_trajectories,
        )

        initial = req.initial_concentrations or {}
        # Default: put all concentration in the first reactant node
        if not initial:
            first_id = list(rpg.nodes.keys())[0]
            initial = {first_id: 1.0}

        result = solver.run(initial)
        graph_data = rpg.to_dict()

    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    rds = next(
        (e["id"] for e in graph_data["edges"] if e.get("is_rate_determining")), None
    )

    return KMCResponse(
        times=result.times,
        species_ids=result.species_ids,
        concentrations=result.concentrations,
        lower_ci=result.lower_ci,
        upper_ci=result.upper_ci,
        max_yield_time=result.max_yield_time,
        max_yield=result.max_yield,
        n_trajectories=result.n_trajectories,
        rate_determining_step=rds,
        graph=graph_data,
    )
