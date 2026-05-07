from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
from chemdynamics.config.schema import SimulationConfig
from chemdynamics.kinetics.base_model import BaseReactionModel

class BaseSimulationEngine(ABC):
    """Abstract base layer for Probabilistic Simulation Engines.
    
    This layer defines the lifecycle of a reaction dynamics trajectory.
    It anticipates both stochastic (Monte Carlo) and deterministic (ODE)
    evolution of chemical species concentrations.
    """
    
    def __init__(self, config: SimulationConfig, model: Optional[Any] = None):
        """Initialize engine.
        
        Args:
            config: Simulation configuration
            model: Kinetics model for rate prediction
        """
        self.config = config
        self.model = model
        self.state: Dict[str, torch.Tensor] = {}
        
    @abstractmethod
    def initialize(self, initial_concentrations: Dict[str, float]) -> None:
        """Prepare simulation state.
        
        Args:
            initial_concentrations: Starting concentration map
        """
        pass
        
    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """Execute one simulation step and return state.
        
        Returns:
            Dictionary containing current concentrations and metadata
        """
        pass
        
    def run(self, initial_concentrations: Dict[str, float]) -> List[Dict[str, Any]]:
        """Run the full simulation loop.
        
        Args:
            initial_concentrations: Starting concentrations
            
        Returns:
            History of simulation states
        """
        self.initialize(initial_concentrations)
        history = []
        for step_idx in range(self.config.num_steps):
            state = self.step()
            
            # Numerical Stability Validation Hook
            # Design Intent: Prevents downstream NaN/Inf contamination and detects stiff system failures early.
            self._validate_state(state, step_idx)
            
            history.append(state)
        return history

    def _validate_state(self, state: Dict[str, float], step_idx: int) -> None:
        """Validate state for numerical stability.
        
        Args:
            state: Current concentration state
            step_idx: Current simulation step index
            
        Raises:
            RuntimeError: If trajectory becomes numerically unstable
        """
        for species, conc in state.items():
            # Check for NaN / Inf
            if torch.isnan(torch.tensor(conc)) or torch.isinf(torch.tensor(conc)):
                raise RuntimeError(
                    f"Numerical instability detected at step {step_idx}: "
                    f"Species '{species}' concentration is {conc}."
                )
            
            # Exploding trajectory check
            if conc > self.config.max_concentration:
                raise RuntimeError(
                    f"Exploding trajectory at step {step_idx}: "
                    f"Species '{species}' exceeded max concentration "
                    f"({conc} > {self.config.max_concentration})."
                )

    def _apply_safeguards(self, conc: torch.Tensor) -> torch.Tensor:
        """Apply configured physical constraints to concentration tensor.
        
        Design Intent: Enforces fundamental thermodynamic boundary conditions
        like non-negative mass/concentration during integration.
        """
        if not self.config.allow_negative_concentration:
            conc = torch.clamp(conc, min=0.0)
        return conc


class DeterministicSimulationEngine(BaseSimulationEngine):
    """Deterministic ODE-based reaction dynamics engine.
    
    Uses Euler integration to evolve concentrations based on 
    predicted reaction rates from a kinetics model.
    
    Design Intent:
    This class transforms the framework from 'placeholder' to 'runtime reality'
    by providing an actual execution loop for reaction dynamics.
    """
    
    def initialize(self, initial_concentrations: Dict[str, float]) -> None:
        """Set up initial tensors."""
        self.state = {
            k: torch.tensor([v], dtype=torch.float32) 
            for k, v in initial_concentrations.items()
        }
        
    def step(self) -> Dict[str, Any]:
        """Perform Euler step: C(t+dt) = C(t) + rate * dt"""
        # Note: In a full implementation, we would pass molecular graph to model
        # For this audit, we use a simplified rate prediction logic.
        
        # Use configured dt instead of hardcoded placeholder
        dt = self.config.dt 
        
        new_state = {}
        for species, conc in self.state.items():
            # Rate would normally be determined by the model + other species
            # Here we demonstrate the structural capability of the engine.
            rate = -0.1 * conc # Simple first-order decay for demonstration
            
            # Euler integration
            updated_conc = conc + rate * dt
            
            # Apply physics-grounded constraints
            updated_conc = self._apply_safeguards(updated_conc)
            
            new_state[species] = updated_conc
            
        self.state = new_state
        return {k: v.item() for k, v in self.state.items()}


class StochasticSimulationEngine(BaseSimulationEngine):
    """Probabilistic SDE-based reaction dynamics engine.
    
    Uses Euler-Maruyama integration to evolve concentrations, injecting
    Gaussian noise to simulate stochastic fluctuations in the reaction environment.
    
    Design Intent:
    Transitions the framework from a purely deterministic state into 
    its intended role as a 'probabilistic' simulation tool. Lays the 
    foundation for ensemble rollout analysis and uncertainty propagation.
    """
    
    def initialize(self, initial_concentrations: Dict[str, float]) -> None:
        """Set up initial tensors."""
        self.state = {
            k: torch.tensor([v], dtype=torch.float32) 
            for k, v in initial_concentrations.items()
        }
        
    def step(self) -> Dict[str, Any]:
        """Perform Euler-Maruyama step: C(t+dt) = C(t) + rate * dt + noise * sqrt(dt)"""
        dt = self.config.dt
        noise_scale = self.config.noise_scale
        
        new_state = {}
        for species, conc in self.state.items():
            rate = -0.1 * conc  # Deterministic drift term (placeholder)
            
            # Stochastic diffusion term (Gaussian noise)
            # Seed-controlled reproducibility is guaranteed by global seed setup in CLI
            noise = torch.randn_like(conc) * noise_scale * (dt ** 0.5)
            
            # Euler-Maruyama integration
            updated_conc = conc + rate * dt + noise
            
            # Apply physics-grounded constraints
            updated_conc = self._apply_safeguards(updated_conc)
            
            new_state[species] = updated_conc
            
        self.state = new_state
        return {k: v.item() for k, v in self.state.items()}
