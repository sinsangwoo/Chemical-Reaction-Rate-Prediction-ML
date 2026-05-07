import typer
from rich.console import Console
from pathlib import Path
from typing import Optional, Any
from omegaconf import OmegaConf

# Internal framework imports
from chemdynamics.config.schema import ChemDynamicsConfig
from chemdynamics.utils.seed import set_deterministic_seed
from chemdynamics.utils.logger import setup_logging, get_logger
from chemdynamics.simulation.engine import DeterministicSimulationEngine

app = typer.Typer(
    help="ChemDynamics: Probabilistic Industrial Reaction Dynamics Simulation Framework",
    add_completion=False,
)
console = Console()

@app.command()
def simulate(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to simulation config YAML"),
    num_steps: Optional[int] = typer.Option(None, "--steps", "-s", help="Override number of simulation steps")
):
    """Run a deterministic reaction dynamics simulation.
    
    This command forms the primary entry point for the framework,
    connecting the static typed configuration to the dynamic simulation runtime.
    """
    setup_logging()
    logger = get_logger("chemdynamics.cli")
    
    # 1. Config Loading: Transition from 'placeholder' to 'runtime infrastructure'
    # Uses OmegaConf to merge CLI overrides with YAML configuration and Pydantic validation.
    if config_path and Path(config_path).exists():
        conf_obj = OmegaConf.load(config_path)
        # Validate against Pydantic schema
        config = ChemDynamicsConfig(**OmegaConf.to_container(conf_obj))
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = ChemDynamicsConfig()
        logger.info("Using default configuration")
        
    # Manual overrides from CLI
    if num_steps:
        config.simulation.num_steps = num_steps
    
    # 2. Global Determinism: Enforcing scientific reproducibility
    set_deterministic_seed(config.simulation.seed)
    logger.info(f"Initialized simulation with seed: {config.simulation.seed}")
    
    # 3. Execution: Instantiating the real engine (Deterministic ODE Integration)
    console.print(f"\n[bold magenta]ChemDynamics Simulation Engine v0.1[/bold magenta]")
    console.print(f"[dim]Mode: Deterministic | Steps: {config.simulation.num_steps}[/dim]\n")
    
    # Placeholder initial concentrations
    initial_state = {"Reactant_A": 1.0, "Product_B": 0.0}
    
    engine = DeterministicSimulationEngine(config.simulation)
    
    console.print(f"[bold green]Starting Probabilistic Simulation...[/bold green]")
    history = engine.run(initial_state)
    
    # Display final state
    final_state = history[-1]
    console.print(f"\n[bold cyan]Final Simulation State:[/bold cyan]")
    for species, conc in final_state.items():
        console.print(f"  {species}: {conc:.6f} mol/L")
        
    console.print(f"\n[bold green]Simulation Complete.[/bold green]")

if __name__ == "__main__":
    app()
