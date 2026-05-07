from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SimulationConfig(BaseModel):
    seed: int = Field(42, description="Global deterministic seed")
    temperature_K: float = Field(298.15, description="Simulation temperature in Kelvin")
    pressure_atm: float = Field(1.0, description="Simulation pressure in atm")
    num_steps: int = Field(1000, description="Number of simulation steps")

class GraphEngineConfig(BaseModel):
    model_type: str = Field("gin", description="GNN architecture type")
    hidden_dim: int = Field(128, description="Hidden dimensions")
    dropout: float = Field(0.1, description="Dropout rate")

class KineticsConfig(BaseModel):
    use_physics_informed: bool = Field(True, description="Enable hybrid Arrhenius modeling")

class ChemDynamicsConfig(BaseModel):
    simulation: SimulationConfig = SimulationConfig()
    graph: GraphEngineConfig = GraphEngineConfig()
    kinetics: KineticsConfig = KineticsConfig()
