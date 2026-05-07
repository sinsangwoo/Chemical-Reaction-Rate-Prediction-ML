"""
Engineering Workflow Orchestration Foundation.

This module provides:
- Process simulation orchestration
- Uncertainty sweep execution
- Operating condition comparison
- Bottleneck analysis workflows
- Experiment orchestration semantics

Design Intent:
- Move logic out of CLI into reusable workflow components
- Support complex engineering analysis (Monte Carlo, sensitivity)
- Maintain full traceability of workflow results
- Extensible for future optimization and search algorithms

Architectural Principles:
- Workflows are higher-level orchestrators
- Workflows use ProcessGraph, KPIAggregator, and ConstraintValidator
- Workflows return structured results for reporting/analysis
"""

from typing import Dict, List, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass, field
import time

from .network import ProcessGraph, UnitOperation, UnitState, Stream, ProcessNode
from .kpis import KPIAggregator, ProcessKPIs, KPIValue
from .constraints import ConstraintValidator, ConstraintViolation
from .telemetry import TelemetryIngestor, TelemetrySource, TelemetryPacket


@dataclass
class WorkflowResult:
    """Base class for workflow execution results."""
    workflow_id: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class UncertaintySweepResult(WorkflowResult):
    """Result of an uncertainty sweep execution."""
    num_trials: int
    kpi_distributions: Dict[str, List[float]] = field(default_factory=dict)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class IndustrialWorkflowManager:
    """
    Orchestrator for complex industrial engineering workflows.
    
    Design Intent:
    - Provides a unified interface for executing standard engineering analyses
    - Manages lifecycle of process realizations
    - Aggregates results across multiple simulation runs
    """

    def __init__(self, kpi_aggregator: Optional[KPIAggregator] = None, 
                 validator: Optional[ConstraintValidator] = None):
        self.kpi_aggregator = kpi_aggregator or KPIAggregator()
        self.validator = validator or ConstraintValidator()

    def run_uncertainty_sweep(
        self,
        process_factory: Callable[[], ProcessGraph],
        parameter_distributions: Dict[str, Any],
        num_trials: int = 10,
        num_steps: int = 10
    ) -> UncertaintySweepResult:
        """
        Execute an uncertainty sweep (Monte Carlo simulation).
        
        Design Intent:
        - Quantifies process sensitivity to input fluctuations
        - Provides statistical confidence in KPI predictions
        - Identifies high-risk operating regimes
        """
        start_time = time.time()
        kpi_distributions: Dict[str, List[float]] = {}
        
        for trial_idx in range(num_trials):
            # 1. Instantiate a fresh process realization
            process = process_factory()
            
            # 2. Apply stochastic perturbations
            # Design Intent: Assume parameter_distributions contains mean/std for key variables
            for param_path, dist in parameter_distributions.items():
                if "mean" in dist and "std" in dist:
                    noise = np.random.normal(dist["mean"], dist["std"])
                    # Simple parameter mapping logic
                    # In a real system, this would use a more robust path-based setter
                    if param_path == "feed_flow":
                        for unit in process.units.values():
                            for s in unit.input_streams.values():
                                if s.stream_id == "raw_feed":
                                    s.flow_rate = noise
            
            # 3. Execute simulation
            process.initialize_all()
            history = process.execute(num_steps=num_steps)
            
            # 4. Aggregate KPIs
            kpis = self.kpi_aggregator.calculate_plant_kpis(history, process.units)
            for kpi in kpis.get_all_kpis():
                if kpi.name not in kpi_distributions:
                    kpi_distributions[kpi.name] = []
                kpi_distributions[kpi.name].append(kpi.value)

        # 5. Calculate statistics
        stats: Dict[str, Dict[str, float]] = {}
        for name, values in kpi_distributions.items():
            arr = np.array(values)
            stats[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95))
            }

        return UncertaintySweepResult(
            workflow_id=f"sweep_{int(start_time)}",
            timestamp=start_time,
            num_trials=num_trials,
            kpi_distributions=kpi_distributions,
            statistics=stats,
            metadata={"elapsed_time": time.time() - start_time}
        )

    def compare_operating_regimes(
        self,
        process_factory: Callable[[float], ProcessGraph],
        variable_range: List[float],
        num_steps: int = 10
    ) -> Dict[float, ProcessKPIs]:
        """
        Compare process performance across a range of operating conditions.
        
        Design Intent:
        - Supports "What-if" analysis for engineers
        - Maps the operating window for feasibility and performance
        """
        results: Dict[float, ProcessKPIs] = {}
        
        for val in variable_range:
            process = process_factory(val)
            process.initialize_all()
            history = process.execute(num_steps=num_steps)
            
            kpis = self.kpi_aggregator.calculate_plant_kpis(history, process.units)
            results[val] = kpis
            
        return results

    def run_telemetry_ingestion(
        self,
        process: ProcessGraph,
        ingestor: TelemetryIngestor,
        num_steps: int = 10,
        dt: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Orchestrate real-time telemetry ingestion and state synchronization.

        Design Intent:
        - Syncs process-graph state with external telemetry stream
        - Provides live observability into operational variables
        - Maintains full history for trend analysis
        """
        history = []
        ingestor.start_ingestion()
        
        try:
            for step in range(num_steps):
                timestamp = step * dt
                # 1. Poll new telemetry
                packets = ingestor.ingest_step()
                
                # 2. Update process nodes
                plant_state: Dict[str, Any] = {}
                global_telemetry = ingestor.get_latest_state()
                
                for node_id, node in process.nodes.items():
                    # In a real system, we'd map specific tags to specific nodes
                    # For now, we pass the global telemetry state
                    node_state = node.update_state(global_telemetry, timestamp)
                    plant_state[node_id] = node_state
                
                process._history.append(plant_state)
                history.append(plant_state)
                
                # 3. Optional: Trigger constraints or KPIs
                # (Observability focus: we only record the state evolution)
                
                time.sleep(min(0.1, dt)) # Simulate real-time pacing
                
        finally:
            ingestor.stop_ingestion()
            
        return history

    def replay_historical_telemetry(
        self,
        process: ProcessGraph,
        telemetry_source: TelemetrySource,
        validator: Optional[ConstraintValidator] = None
    ) -> List[Dict[str, Any]]:
        """
        Replay historical telemetry through the process graph.

        Design Intent:
        - Enables forensic analysis of past process events
        - Validates historical data against engineering constraints
        - Ensures reproducible operational observability
        """
        history = []
        telemetry_source.connect()
        
        try:
            while True:
                packets = telemetry_source.poll()
                if not packets:
                    break
                
                for packet in packets:
                    plant_state: Dict[str, Any] = {}
                    for node_id, node in process.nodes.items():
                        node_state = node.update_state(packet.data, packet.timestamp)
                        plant_state[node_id] = node_state
                        
                        if validator:
                            # Verify constraints during replay
                            violations = validator.validate_unit(node, node_state)
                            node_state.constraint_violations = [v.message for v in violations]
                    
                    history.append(plant_state)
                    process._history.append(plant_state)
                    
        finally:
            telemetry_source.disconnect()
            
        return history
