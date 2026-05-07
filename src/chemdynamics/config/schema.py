from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SimulationConfig(BaseModel):
    seed: int = Field(42, description="Global deterministic seed")
    temperature_K: float = Field(298.15, description="Simulation temperature in Kelvin")
    pressure_atm: float = Field(1.0, description="Simulation pressure in atm")
    num_steps: int = Field(1000, description="Number of simulation steps")
    
    # Numerical Stability Safeguards
    # Design Intent: Explicitly parameterize integration properties to avoid hidden constants
    # and provide configurable bounds for trajectory anomaly detection.
    dt: float = Field(0.01, description="Integration timestep size")
    max_concentration: float = Field(100.0, description="Maximum allowable concentration (exploding trajectory threshold)")
    allow_negative_concentration: bool = Field(False, description="If false, negative concentrations are clamped to zero")
    
    # Stochastic Process Control
    # Design Intent: Foundation for probabilistic trajectories.
    # When enabled, deterministic Euler steps transition to Euler-Maruyama SDE steps.
    stochastic: bool = Field(False, description="Enable stochastic (SDE) trajectory execution")
    noise_scale: float = Field(0.05, description="Scale of Gaussian noise in stochastic steps")


class GraphEngineConfig(BaseModel):
    model_type: str = Field("gin", description="GNN architecture type")
    hidden_dim: int = Field(128, description="Hidden dimensions")
    dropout: float = Field(0.1, description="Dropout rate")

class KineticsConfig(BaseModel):
    use_physics_informed: bool = Field(True, description="Enable hybrid Arrhenius modeling")

class DataProvenanceConfig(BaseModel):
    """
    Data Lifecycle & Research Reproducibility Foundation.
    
    Design Intent:
    Ensures that every simulation or training run can be traced back to a specific
    dataset version and outputs are structured reproducibly. Prevents 'floating experiments'.
    """
    dataset_version: str = Field("v1.0", description="Version identifier of the input dataset")
    raw_data_path: str = Field("data/raw", description="Path to raw input datasets")
    processed_data_path: str = Field("data/processed", description="Path to explicitly preprocessed artifacts")
    experiment_id: Optional[str] = Field(None, description="Unique identifier for the current experiment/run")
    output_dir: str = Field("results/experiments", description="Root directory for saving deterministic artifacts")

class ChemDynamicsConfig(BaseModel):
    simulation: SimulationConfig = SimulationConfig()
    graph: GraphEngineConfig = GraphEngineConfig()
    kinetics: KineticsConfig = KineticsConfig()
    provenance: DataProvenanceConfig = DataProvenanceConfig()
