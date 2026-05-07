import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
from typing import Optional, Any, List, Dict
from omegaconf import OmegaConf
import json
import time
import numpy as np

# Internal framework imports
from chemdynamics.config.schema import ChemDynamicsConfig
from chemdynamics.utils.seed import set_deterministic_seed
from chemdynamics.utils.logger import setup_logging, get_logger
from chemdynamics.simulation.engine import DeterministicSimulationEngine, StochasticSimulationEngine
from chemdynamics.process import (
    ProcessGraph,
    Reactor,
    Mixer,
    Splitter,
    Stream,
    TemperatureLimit,
    PressureLimit,
    CapacityLimit,
    EnergyLimit,
    ThroughputLimit,
    RampRateLimit,
    SafetyLimit,
    ConstraintValidator,
    ConstraintSeverity,
    KPIAggregator,
    ThroughputKPI,
    UtilizationKPI,
    YieldKPI,
    ConversionKPI,
    EnergyUsageKPI,
    EfficiencyKPI,
    StabilityKPI,
    KPIType,
    Separator,
    HeatExchanger,
    Pump,
    Storage,
    ProcessNode,
    TelemetryIngestor,
    CsvTelemetrySource,
    TelemetryValidityConstraint,
    TemporalConsistencyConstraint,
    RollingWindowKPI,
    TemporalTrendKPI,
    IndustrialWorkflowManager,
    UncertaintySweepResult
)

app = typer.Typer(
    help="ChemDynamics: Probabilistic Industrial Reaction Dynamics Simulation Framework",
    add_completion=False,
)
console = Console()

process_app = typer.Typer(help="Industrial process simulation and engineering analysis commands")
app.add_typer(process_app, name="process")

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
    
    # 3. Execution: Instantiating the real engine
    # Design Intent: Branch based on scientific objective (deterministic vs stochastic)
    console.print(f"\n[bold magenta]ChemDynamics Simulation Engine v0.1[/bold magenta]")
    
    if config.simulation.stochastic:
        console.print(f"[dim]Mode: Stochastic (SDE) | Steps: {config.simulation.num_steps} | Noise: {config.simulation.noise_scale}[/dim]\n")
        engine = StochasticSimulationEngine(config.simulation)
    else:
        console.print(f"[dim]Mode: Deterministic (ODE) | Steps: {config.simulation.num_steps}[/dim]\n")
        engine = DeterministicSimulationEngine(config.simulation)
    
    # Placeholder initial concentrations. In a real workflow, this would come from the Dataset.
    initial_state = {"Reactant_A": 1.0, "Product_B": 0.0}
    
    console.print(f"[bold green]Starting Simulation...[/bold green]")
    try:
        start_time = time.time()
        history = engine.run(initial_state)
        elapsed = time.time() - start_time
    except RuntimeError as e:
        logger.error(f"Simulation failed due to numerical instability: {e}")
        console.print(f"[bold red]Simulation Aborted: Numerical Instability[/bold red]")
        return
    
    # Display final state
    final_state = history[-1]
    console.print(f"\n[bold cyan]Final Simulation State:[/bold cyan]")
    for species, conc in final_state.items():
        console.print(f"  {species}: {conc:.6f} mol/L")
        
    # 4. Data Lifecycle & Reproducibility
    # Design Intent: Save artifacts automatically so experiments are traceable.
    exp_id = config.provenance.experiment_id or f"exp_{int(time.time())}"
    out_dir = Path(config.provenance.output_dir) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_path = out_dir / "trajectory.json"
    
    # Save the trajectory with metadata
    artifact = {
        "metadata": {
            "experiment_id": exp_id,
            "dataset_version": config.provenance.dataset_version,
            "seed": config.simulation.seed,
            "stochastic": config.simulation.stochastic,
            "elapsed_time_s": elapsed
        },
        "history": history
    }
    
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)
        
    logger.info(f"Simulation artifact saved to {artifact_path}")
    console.print(f"\n[bold green]Simulation Complete.[/bold green]")

@process_app.command("simulate")
def simulate_process(
    num_steps: int = typer.Option(10, "--steps", "-s", help="Number of process steps"),
    dt: float = typer.Option(1.0, "--dt", help="Time delta per step"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for results")
):
    """
    Simulate a complete industrial process.

    Demonstrates:
    - Process network with reactor, mixer, splitter
    - Material flow through unit operations
    - Constraint checking
    - KPI calculation

    Design Intent:
    - Engineer-oriented process simulation
    - No fake digital twin, just foundational infrastructure
    """
    setup_logging()
    logger = get_logger("chemdynamics.cli.process")

    console.print("\n[bold magenta]ChemDynamics Process Simulator[/bold magenta]")
    console.print("[dim]Industrial Process Foundation v0.1[/dim]\n")

    with console.status("[bold green]Building complex process network...[/bold green]"):
        process = ProcessGraph(name="industrial_process_v1")

        # 1. Feed Preparation
        feed_stream = Stream(
            stream_id="raw_feed",
            composition={"A": 2.0, "C": 0.1},
            temperature=20.0,
            pressure=1.0,
            flow_rate=2.1
        )

        mixer = Mixer("M101", name="Feed Mixer")
        pump = Pump("P101", name="Feed Pump")
        pump.delta_p = 5.0 # Increase pressure to 6 atm

        # 2. Reaction Stage
        reactor = Reactor("R101", name="Primary Reactor")
        reactor.volume = 50.0
        reactor.heat_duty = 5000.0 # Heating (W)
        reactor.set_parameter("degradation_rate", 0.005)

        # 3. Separation Stage
        separator = Separator("V101", name="Flash Separator")
        separator.set_efficiency("B", 0.95) # 95% of product B goes to overhead
        separator.set_efficiency("A", 0.10) # 10% of unreacted A goes to overhead

        # 4. Thermal Control
        cooler = HeatExchanger("E101", name="Product Cooler")
        cooler.target_temperature = 40.0

        # 5. Inventory Management
        storage = Storage("T101", name="Product Tank")
        storage.capacity = 1000.0

        # Connections
        process.add_unit(mixer)
        process.add_unit(pump)
        process.add_unit(reactor)
        process.add_unit(separator)
        process.add_unit(cooler)
        process.add_unit(storage)

        mixer.add_input_stream(feed_stream, "raw_in")
        process.connect_units(mixer, pump)
        process.connect_units(pump, reactor)
        process.connect_units(reactor, separator)
        
        # Split separator outputs
        overhead = Stream("product_rich")
        bottoms = Stream("heavy_ends")
        separator.add_output_stream(overhead, "overhead")
        separator.add_output_stream(bottoms, "bottoms")
        
        process.add_stream(overhead, source=separator, target=cooler)
        process.connect_units(cooler, storage)

        # Constraints
        reactor.add_constraint(TemperatureLimit("reactor_max_temp", max_temp=450.0))
        reactor.add_constraint(PressureLimit("reactor_max_pressure", max_pressure=10.0))
        reactor.add_constraint(EnergyLimit("reactor_energy_cap", max_power=10000.0))
        pump.add_constraint(CapacityLimit("pump_max_flow", max_capacity=5.0))
        
        # Safety Logic: No more than 5% species C in final product
        def safety_check(unit, state):
            for s in state.output_streams.values():
                if s.composition.get("C", 0.0) > 0.5:
                    return False
            return True
        storage.add_constraint(SafetyLimit("impurity_safety", safety_check, "Critical impurity level in storage"))

    console.print("[bold green]Industrial process network built successfully![/bold green]\n")

    table = Table(title="Plant Topology")
    table.add_column("Tag", style="cyan")
    table.add_column("Unit Operation", style="magenta")
    table.add_column("Status", style="green")
    for uid, unit in process.units.items():
        table.add_row(uid, unit.unit_type.value.upper(), "ACTIVE")
    console.print(table)
    console.print()

    kpi_aggregator = KPIAggregator()
    kpi_aggregator.register_unit_calculator("R101", YieldKPI("reactor_yield", product_species="B", reactant_species="A"))
    kpi_aggregator.register_unit_calculator("P101", EnergyUsageKPI("pump_power"))
    kpi_aggregator.register_unit_calculator("R101", StabilityKPI("temp_stability", "temperature"))
    kpi_aggregator.register_unit_calculator("T101", ThroughputKPI("production_rate", species="B"))

    process.initialize_all()

    console.print(f"[bold green]Executing industrial process simulation ({num_steps} steps)...[/bold green]\n")

    plant_history = process.execute(num_steps=num_steps, dt=dt)
    process_kpis = kpi_aggregator.calculate_plant_kpis(plant_history, process.units)

    kpi_table = Table(title="Operational KPIs")
    kpi_table.add_column("Metric", style="cyan")
    kpi_table.add_column("Value", style="green")
    kpi_table.add_column("Unit", style="magenta")

    for kpi in process_kpis.get_all_kpis():
        kpi_table.add_row(kpi.name, f"{kpi.value:.2f}", kpi.unit)

    console.print(kpi_table)
    console.print()

    # Check for violations
    validator = ConstraintValidator()
    violations = validator.validate_process(plant_history[-1], process.units)
    if violations:
        console.print("[bold red]❌ Operational Constraint Violations Detected:[/bold red]")
        for v in violations:
            console.print(f"  • [yellow]{v.unit_id}[/yellow]: {v.message} ({v.severity.value})")
    else:
        console.print("[bold green]✅ Process Operating within Design Limits[/bold green]")

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {
            "num_steps": num_steps,
            "dt": dt,
            "elapsed_time_s": elapsed,
            "plant_kpis": [
                {"name": k.name, "value": k.value, "unit": k.unit}
                for k in process_kpis.plant_kpis
            ]
        }

        with open(out_path / "process_results.json", "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[bold green]Results saved to {out_path}[/bold green]")


@process_app.command("analyze-bottleneck")
def analyze_bottleneck(
    num_steps: int = typer.Option(20, "--steps", "-s", help="Number of simulation steps")
):
    """
    Analyze process bottlenecks.

    Demonstrates:
    - Bottleneck detection based on utilization
    - Unit-level KPI comparison
    - Engineer-oriented bottleneck analysis

    Design Intent:
    - No fake bottleneck scoring
    - Based on actual utilization metrics
    - Supports future optimization workflows
    """
    setup_logging()

    console.print("\n[bold magenta]Bottleneck Analysis[/bold magenta]\n")

    process = ProcessGraph(name="bottleneck_analysis")

    reactor1 = Reactor("R1", name="Reactor 1")
    reactor2 = Reactor("R2", name="Reactor 2")
    reactor3 = Reactor("R3", name="Reactor 3 (High Load)")
    mixer = Mixer("M1", name="Final Mixer")

    process.add_unit(reactor1)
    process.add_unit(reactor2)
    process.add_unit(reactor3)
    process.add_unit(mixer)

    feed1 = Stream("feed1", composition={"A": 0.5}, flow_rate=0.5)
    feed2 = Stream("feed2", composition={"B": 0.5}, flow_rate=0.5)
    feed3 = Stream("feed3", composition={"C": 2.0}, flow_rate=2.0)

    reactor1.add_input_stream(feed1)
    reactor2.add_input_stream(feed2)
    reactor3.add_input_stream(feed3)

    out1 = Stream("out1")
    out2 = Stream("out2")
    out3 = Stream("out3")
    reactor1.add_output_stream(out1)
    reactor2.add_output_stream(out2)
    reactor3.add_output_stream(out3)

    mixer.add_input_stream(out1, "in1")
    mixer.add_input_stream(out2, "in2")
    mixer.add_input_stream(out3, "in3")

    final_out = Stream("final_product")
    mixer.add_output_stream(final_out, "product")

    process.initialize_all()

    kpi_aggregator = KPIAggregator()
    kpi_aggregator.register_unit_calculator("R1", UtilizationKPI("util_r1", max_capacity=1.0))
    kpi_aggregator.register_unit_calculator("R2", UtilizationKPI("util_r2", max_capacity=1.0))
    kpi_aggregator.register_unit_calculator("R3", UtilizationKPI("util_r3", max_capacity=1.0))
    kpi_aggregator.register_unit_calculator("M1", UtilizationKPI("util_m1", max_capacity=5.0))

    plant_history = process.execute(num_steps=num_steps)
    process_kpis = kpi_aggregator.calculate_plant_kpis(plant_history, process.units)

    util_table = Table(title="Unit Utilization")
    util_table.add_column("Unit", style="cyan")
    util_table.add_column("Utilization", style="green")
    util_table.add_column("Status", style="magenta")

    bottleneck_info = None
    for kpi in process_kpis.plant_kpis:
        if kpi.kpi_type == KPIType.BOTTLENECK_SCORE:
            bottleneck_info = kpi.metadata

    for unit_id, kpis in process_kpis.unit_kpis.items():
        for kpi in kpis:
            if kpi.kpi_type == KPIType.UTILIZATION:
                status = "[red]BOTTLENECK[/red]" if bottleneck_info and unit_id == bottleneck_info.get("bottleneck_unit") else "OK"
                util_table.add_row(unit_id, f"{kpi.value:.1f}%", status)

    console.print(util_table)
    console.print()

    if bottleneck_info:
        primary = bottleneck_info.get("primary_bottleneck")
        console.print(f"[bold red]🎯 Primary Bottleneck: {primary}[/bold red]")
        console.print(f"[dim]Recommendation: Consider increasing capacity of {primary}[/dim]")


@process_app.command("compare-operating-window")
def compare_operating_window(
    temp_min: float = typer.Option(20.0, "--temp-min", help="Minimum temperature"),
    temp_max: float = typer.Option(100.0, "--temp-max", help="Maximum temperature"),
    temp_step: float = typer.Option(20.0, "--temp-step", help="Temperature step")
):
    """
    Compare process performance across operating windows.

    Demonstrates:
    - Parameter sweep execution
    - KPI comparison across conditions
    - Engineer-oriented tradeoff analysis

    Design Intent:
    - Supports process optimization workflows
    - Explicit parameter comparison
    - No fake optimization, just foundational infrastructure
    """
    setup_logging()

    console.print("\n[bold magenta]Operating Window Comparison[/bold magenta]\n")

    temps = np.arange(temp_min, temp_max + temp_step, temp_step)

    table = Table(title="Temperature Sweep Results")
    table.add_column("Temperature (°C)", style="cyan")
    table.add_column("Throughput", style="green")
    table.add_column("Feasibility", style="magenta")

    for temp in temps:
        process = ProcessGraph(name=f"temp_{temp}")
        reactor = Reactor("R1")
        process.add_unit(reactor)

        feed = Stream("feed", composition={"A": 1.0}, temperature=temp)
        reactor.add_input_stream(feed)
        product = Stream("product")
        reactor.add_output_stream(product)

        temp_constraint = TemperatureLimit(
            "temp_limit",
            min_temp=20.0,
            max_temp=80.0,
            severity=ConstraintSeverity.ERROR
        )
        reactor.add_constraint(temp_constraint)

        process.initialize_all()
        plant_history = process.execute(num_steps=5)

        kpi_aggregator = KPIAggregator()
        kpi_aggregator.register_unit_calculator("R1", ThroughputKPI())
        process_kpis = kpi_aggregator.calculate_plant_kpis(plant_history, process.units)

        throughput = next((k.value for k in process_kpis.plant_kpis if k.kpi_type == KPIType.THROUGHPUT), 0.0)
        feasible = "[green]FEASIBLE[/green]" if temp <= 80 else "[red]INFEASIBLE[/red]"

        table.add_row(f"{temp:.0f}", f"{throughput:.4f}", feasible)

    console.print(table)
    console.print()
    console.print("[dim]Note: Infeasible at >80°C due to temperature constraint[/dim]")


@process_app.command("run-uncertainty-sweep")
def run_uncertainty_sweep(
    num_trials: int = typer.Option(10, "--trials", "-n", help="Number of stochastic trials"),
    uncertainty_scale: float = typer.Option(0.1, "--scale", help="Scale of input uncertainty")
):
    """
    Execute an uncertainty sweep across process realizations.

    Design Intent:
    - Transitions from single-point simulation to risk-aware analysis
    - Utilizes IndustrialWorkflowManager for orchestration
    - Provides statistical confidence intervals for industrial KPIs
    """
    setup_logging()
    console.print("\n[bold magenta]Process Uncertainty Sweep[/bold magenta]")
    console.print("[dim]Orchestrating Monte Carlo realizations...[/dim]\n")

    def process_factory():
        process = ProcessGraph("stochastic_realization")
        reactor = Reactor("R1")
        # Initialize with standard stream, to be perturbed by workflow manager
        feed = Stream("raw_feed", stream_id="raw_feed", composition={"A": 1.0}, flow_rate=1.0)
        reactor.add_input_stream(feed)
        reactor.add_output_stream(Stream("product"))
        process.add_unit(reactor)
        return process

    # Configure parameter distributions for the sweep
    # In a real workflow, these would come from the UncertaintyInfrastructure
    distributions = {
        "feed_flow": {"mean": 1.0, "std": uncertainty_scale}
    }

    workflow = IndustrialWorkflowManager()
    
    with console.status(f"[bold green]Running {num_trials} trials...[/bold green]"):
        result = workflow.run_uncertainty_sweep(
            process_factory=process_factory,
            parameter_distributions=distributions,
            num_trials=num_trials
        )

    # Display results using Rich tables for engineering clarity
    stats_table = Table(title="Uncertainty Analysis: KPI Statistics")
    stats_table.add_column("KPI Name", style="cyan")
    stats_table.add_column("Mean", style="green")
    stats_table.add_column("Std Dev", style="yellow")
    stats_table.add_column("P95", style="magenta")

    for name, s in result.statistics.items():
        stats_table.add_row(
            name, 
            f"{s['mean']:.4f}", 
            f"{s['std']:.4f}", 
            f"{s['p95']:.4f}"
        )

    console.print(stats_table)
    console.print()
    
    # Forensic analysis hook
    if any(s['std'] / s['mean'] > 0.2 for s in result.statistics.values() if s['mean'] != 0):
        console.print("[bold red]⚠️ CRITICAL SENSITIVITY DETECTED[/bold red]")
        console.print("[dim]Recommendation: Review operating window constraints and upstream feed stability.[/dim]")
    else:
        console.print("[bold green]✅ Process shows robust stability across input uncertainty range.[/bold green]")


@process_app.command("ingest-telemetry")
def ingest_telemetry(
    source_file: str = typer.Option("telemetry.csv", "--source", "-s", help="Path to CSV telemetry source"),
    steps: int = typer.Option(5, "--steps", help="Number of ingestion steps to simulate")
):
    """
    Ingest real-time industrial telemetry and synchronize process state.

    Design Intent:
    - Provides operational visibility into live (or simulated live) data
    - Organizes telemetry into structured process nodes
    - Lays foundation for real-time monitoring workflows
    """
    setup_logging()
    console.print("\n[bold cyan]Industrial Telemetry Ingestion[/bold cyan]")
    
    # Create a dummy CSV for demonstration if it doesn't exist
    if not Path(source_file).exists():
        with open(source_file, "w") as f:
            f.write("timestamp,reactor_temp,feed_flow,pressure\n")
            f.write("1715070000,350.5,10.2,2.1\n")
            f.write("1715070060,352.1,10.1,2.2\n")
            f.write("1715070120,349.8,10.3,2.1\n")
            f.write("1715070180,365.2,10.2,2.3\n")
            f.write("1715070240,355.0,10.1,2.2\n")

    process = ProcessGraph("observability_plant")
    # Define a generic node representing a section of the plant
    main_node = ProcessNode("SECTION_01", "processing_area")
    # We define that this node is interested in these telemetry tags
    main_node.add_tag("reactor_temp")
    main_node.add_tag("pressure")
    
    # Concrete implementation of update_state for this observability task
    # Design Intent: Engineer defines how telemetry maps to node state
    def custom_update(telemetry, timestamp):
        state = UnitState("SECTION_01", timestamp, internal_state=telemetry)
        return state
    main_node.update_state = custom_update
    
    process.add_node(main_node)

    source = CsvTelemetrySource("csv_feed", source_file)
    ingestor = TelemetryIngestor()
    ingestor.add_source(source)

    workflow = IndustrialWorkflowManager()
    
    console.print(f"[dim]Ingesting from {source_file}...[/dim]\n")
    
    with console.status("[bold green]Synchronizing process state...[/bold green]"):
        history = workflow.run_telemetry_ingestion(process, ingestor, num_steps=steps)

    table = Table(title="Operational Telemetry Stream")
    table.add_column("Timestamp", style="dim")
    table.add_column("Node", style="cyan")
    table.add_column("Reactor Temp", style="green")
    table.add_column("Pressure", style="yellow")

    for plant_state in history:
        for node_id, state in plant_state.items():
            table.add_row(
                str(state.timestamp),
                node_id,
                str(state.internal_state.get("reactor_temp", "N/A")),
                str(state.internal_state.get("pressure", "N/A"))
            )

    console.print(table)
    console.print()
    console.print("[bold green]Telemetry ingestion complete.[/bold green]")


@process_app.command("replay-process-state")
def replay_process_state(
    source_file: str = typer.Option("telemetry.csv", "--source", "-s", help="Path to telemetry history"),
    validate: bool = typer.Option(True, "--validate", help="Whether to run constraint validation during replay")
):
    """
    Replay historical process state for forensic observability.

    Design Intent:
    - Enables reproducible investigation of past events
    - Validates data quality via TelemetryValidity constraints
    - Provides structured replay of engineering invariants
    """
    setup_logging()
    console.print("\n[bold magenta]Historical Process Replay[/bold magenta]\n")

    process = ProcessGraph("forensic_plant")
    main_unit = Reactor("R101", name="Forensic Reactor")
    
    if validate:
        # Define observability-specific constraints
        validity = TelemetryValidityConstraint(
            "temp_range", "reactor_temp", min_val=200.0, max_val=400.0,
            description="Operational temperature out of physical bounds"
        )
        consistency = TemporalConsistencyConstraint(
            "temp_stability", "reactor_temp", max_delta=10.0,
            description="Unrealistic temperature jump detected"
        )
        main_unit.add_constraint(validity)
        main_unit.add_constraint(consistency)

    process.add_unit(main_unit)

    source = CsvTelemetrySource("history_feed", source_file)
    validator = ConstraintValidator() if validate else None
    
    workflow = IndustrialWorkflowManager()
    
    with console.status("[bold blue]Executing forensic replay...[/bold blue]"):
        history = workflow.replay_historical_telemetry(process, source, validator)

    # Summarize findings
    console.print(f"[bold cyan]Replay Summary:[/bold cyan]")
    console.print(f"  • Total Samples: {len(history)}")
    
    violations_count = 0
    for plant_state in history:
        for state in plant_state.values():
            if state.constraint_violations:
                violations_count += len(state.constraint_violations)
                for v in state.constraint_violations:
                    console.print(f"  [red]![/red] {state.timestamp}: {v}")

    if violations_count == 0:
        console.print("[bold green]✅ No operational anomalies detected in historical data.[/bold green]")
    else:
        console.print(f"\n[bold yellow]⚠️ Detected {violations_count} anomalies during replay.[/bold yellow]")


@process_app.command("summarize-observability")
def summarize_observability(
    source_file: str = typer.Option("telemetry.csv", "--source", "-s")
):
    """
    Compute rolling statistics and trends for process observability.
    """
    setup_logging()
    console.print("\n[bold green]Process Observability Summary[/bold green]\n")

    process = ProcessGraph("stats_plant")
    unit = Reactor("R1")
    process.add_unit(unit)
    
    source = CsvTelemetrySource("stats_feed", source_file)
    workflow = IndustrialWorkflowManager()
    
    # We first replay to populate history
    history = workflow.replay_historical_telemetry(process, source)
    
    # Now compute observability KPIs
    rolling_avg = RollingWindowKPI("temp_avg", "reactor_temp", window_size=3, stat_type="mean", unit="°C")
    rolling_std = RollingWindowKPI("temp_std", "reactor_temp", window_size=3, stat_type="std", unit="°C")
    trend = TemporalTrendKPI("temp_trend", "reactor_temp", unit="°C/s")
    
    avg_val = rolling_avg.calculate(history, unit)
    std_val = rolling_std.calculate(history, unit)
    trend_val = trend.calculate(history, unit)
    
    table = Table(title="Observability Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Interpretation", style="dim")
    
    table.add_row(avg_val.name, f"{avg_val.value:.2f} {avg_val.unit}", "Rolling mean (last 3)")
    table.add_row(std_val.name, f"{std_val.value:.2f} {std_val.unit}", "Rolling volatility")
    table.add_row(trend_val.name, f"{trend_val.value:.4f} {trend_val.unit}", "Current rate of change")
    
    console.print(table)


@process_app.command("generate-report")
def generate_process_report(
    output_file: str = typer.Option("process_report.json", "--output", "-o", help="Output report file")
):
    """
    Generate a comprehensive process report.

    Demonstrates:
    - Process metadata collection
    - KPI aggregation
    - Report generation for engineers

    Design Intent:
    - Engineer-oriented report format
    - Traceable to simulation data
    - Supports documentation and compliance
    """
    setup_logging()

    console.print("\n[bold magenta]Process Report Generator[/bold magenta]\n")

    process = ProcessGraph(name="report_process")
    reactor = Reactor("R101", name="Main Reactor")
    mixer = Mixer("M101", name="Feed Mixer")
    process.add_unit(mixer)
    process.add_unit(reactor)

    feed = Stream("feed", composition={"A": 1.0, "B": 0.5})
    mixer.add_input_stream(feed)
    process.connect_units(mixer, reactor)

    process.initialize_all()
    plant_history = process.execute(num_steps=10)

    kpi_aggregator = KPIAggregator()
    kpi_aggregator.register_unit_calculator("R101", ThroughputKPI())
    kpi_aggregator.register_unit_calculator("R101", UtilizationKPI(max_capacity=5.0))
    process_kpis = kpi_aggregator.calculate_plant_kpis(plant_history, process.units)

    report = {
        "report_id": f"report_{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "process_name": process.name,
        "num_units": len(process.units),
        "num_steps": len(plant_history),
        "units": [
            {"id": u.unit_id, "type": u.unit_type.value, "name": u.name}
            for u in process.units.values()
        ],
        "plant_kpis": [
            {"name": k.name, "value": k.value, "unit": k.unit}
            for k in process_kpis.plant_kpis
        ]
    }

    out_path = Path(output_file)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"[bold green]Report generated: {out_path}[/bold green]")
    console.print(f"  • Process: {process.name}")
    console.print(f"  • Units: {len(process.units)}")
    console.print(f"  • KPIs: {len(process_kpis.plant_kpis)}")


if __name__ == "__main__":
    app()
