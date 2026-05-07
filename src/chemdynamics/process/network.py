"""
Process Network Modeling Foundation.

This module provides:
- Unit operation abstractions (reactors, separators, mixers, etc.)
- Process topology representation
- Stream connections and material flow
- Process graph semantics
- Inter-unit dependency propagation
- Plant-level state management

Design Intent:
- Process-centric, not reaction-centric architecture
- Modular unit operations with clear responsibility boundaries
- Extensible for future unit types
- Explicit material and energy flow semantics
- Plant-level state ownership and propagation

IMPORTANT:
This is a FOUNDATIONAL layer, not a full industrial simulator.
The goal is to establish architectural boundaries and interfaces.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
from datetime import datetime

from chemdynamics.data.reaction_dataset import ChemicalReaction, ReactionConditions


class UnitType(Enum):
    """Types of industrial unit operations."""
    REACTOR = "reactor"
    SEPARATOR = "separator"
    HEAT_EXCHANGER = "heat_exchanger"
    MIXER = "mixer"
    SPLITTER = "splitter"
    STORAGE = "storage"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    DISTILLATION = "distillation"
    EXTRACTOR = "extractor"
    VALVE = "valve"
    FILTER = "filter"


@dataclass
class Stream:
    """
    Material or energy stream between unit operations.

    Design Intent:
    - Explicit material/energy flow representation
    - Traceable stream properties
    - Compatible with mass/energy balance calculations
    """
    stream_id: str
    source_unit: Optional[str] = None
    target_unit: Optional[str] = None
    composition: Dict[str, float] = field(default_factory=dict)
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    flow_rate: Optional[float] = None
    phase: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def total_flow(self) -> float:
        return sum(self.composition.values())

    def copy(self) -> "Stream":
        return Stream(
            stream_id=f"{self.stream_id}_copy",
            source_unit=self.source_unit,
            target_unit=self.target_unit,
            composition=self.composition.copy(),
            temperature=self.temperature,
            pressure=self.pressure,
            flow_rate=self.flow_rate,
            phase=self.phase,
            properties=self.properties.copy()
        )


@dataclass
class UnitState:
    """
    State of a unit operation at a point in time.

    Design Intent:
    - Encapsulates all internal state of a unit
    - Supports serialization and lineage tracking
    - Compatible with process trajectory storage
    """
    unit_id: str
    timestamp: float
    internal_state: Dict[str, Any] = field(default_factory=dict)
    input_streams: Dict[str, Stream] = field(default_factory=dict)
    output_streams: Dict[str, Stream] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    constraint_violations: List[str] = field(default_factory=list)


class ProcessNode(ABC):
    """
    Generic abstraction for any entity in an industrial process.

    Design Intent:
    - Fundamental building block for generalized process graphs
    - Decoupled from specific industrial semantics (e.g. chemistry)
    - Supports arbitrary telemetry and state organization
    - Interpretive meaning is defined by the engineer via metadata

    Architectural Rules:
    - Nodes manage their own state and telemetry
    - Nodes define their own operational boundaries
    - Nodes communicate via explicit connections (edges/streams)
    """

    def __init__(self, node_id: str, node_type: str, name: Optional[str] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.name = name or node_id
        self.metadata: Dict[str, Any] = {}
        self.tags: Set[str] = set()
        self._history: List[Any] = []

    def set_metadata(self, key: str, value: Any) -> None:
        """Set engineering metadata for this node."""
        self.metadata[key] = value

    def add_tag(self, tag: str) -> None:
        """Add an operational tag for telemetry mapping."""
        self.tags.add(tag)

    @abstractmethod
    def update_state(self, telemetry: Dict[str, Any], timestamp: float) -> Any:
        """
        Update the node state based on incoming telemetry.

        Design Intent:
        - Engineers define how telemetry maps to state
        - Supports real-time state synchronization
        """
        pass


class UnitOperation(ProcessNode):
    """
    Specialized ProcessNode for industrial operations with stream semantics.

    Design Intent:
    - Maintains the established I/O contract for material/energy flow
    - Extends ProcessNode with constraint and parameter management
    - Remains a generic base for specific unit implementations
    """

    def __init__(self, unit_id: str, unit_type: Union[UnitType, str], name: Optional[str] = None):
        type_str = unit_type.value if isinstance(unit_type, UnitType) else str(unit_type)
        super().__init__(unit_id, type_str, name)
        self.unit_id = unit_id # For backward compatibility
        self.unit_type = unit_type # For backward compatibility
        self.input_streams: Dict[str, Stream] = {}
        self.output_streams: Dict[str, Stream] = {}
        self.internal_parameters: Dict[str, Any] = {}
        self.constraints: List["ProcessConstraint"] = []

    def update_state(self, telemetry: Dict[str, Any], timestamp: float) -> UnitState:
        """
        Default implementation mapping telemetry to internal state.
        """
        # In a generalized observability framework, we don't assume the mapping.
        # But for continuity, we provide a mechanism to ingest telemetry into internal_state.
        state = UnitState(
            unit_id=self.unit_id,
            timestamp=timestamp,
            internal_state=telemetry.copy(),
            input_streams={k: v.copy() for k, v in self.input_streams.items()},
            output_streams={k: v.copy() for k, v in self.output_streams.items()}
        )
        return state

    def add_input_stream(self, stream: Stream, input_name: Optional[str] = None) -> None:
        """Connect an input stream to this unit."""
        name = input_name or f"in_{len(self.input_streams)}"
        stream.target_unit = self.unit_id
        self.input_streams[name] = stream

    def add_output_stream(self, stream: Stream, output_name: Optional[str] = None) -> None:
        """Connect an output stream to this unit."""
        name = output_name or f"out_{len(self.output_streams)}"
        stream.source_unit = self.unit_id
        self.output_streams[name] = stream

    def set_parameter(self, key: str, value: Any) -> None:
        """Set an internal parameter for this unit."""
        self.internal_parameters[key] = value

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get an internal parameter."""
        return self.internal_parameters.get(key, default)

    def add_constraint(self, constraint: "ProcessConstraint") -> None:
        """Add a process constraint to this unit."""
        self.constraints.append(constraint)

    @abstractmethod
    def initialize(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the unit operation state.

        Design Intent:
        - Must be called before first execution
        - Sets up initial internal state
        - Validates input/output connections
        """
        pass

    @abstractmethod
    def execute(self, timestamp: float) -> UnitState:
        """
        Execute one step of the unit operation.

        Design Intent:
        - Reads from input streams
        - Computes internal state transitions
        - Writes to output streams
        - Checks constraints
        - Returns complete unit state

        Returns:
            UnitState containing current state of the unit
        """
        pass

    def check_constraints(self, state: UnitState) -> List[str]:
        """Check all constraints for this unit."""
        violations = []
        for constraint in self.constraints:
            if not constraint.check(self, state):
                violations.append(constraint.description)
        return violations

    def get_history(self) -> List[UnitState]:
        """Get the execution history of this unit."""
        return self._history.copy()


class Reactor(UnitOperation):
    """
    Chemical reactor unit operation.

    Design Intent:
    - Encapsulates reaction kinetics
    - Supports multiple reaction types
    - Manages reactor-specific state (temperature, pressure, catalyst)
    - Extensible for different reactor models (CSTR, PFR, batch)
    """

    def __init__(self, unit_id: str, name: Optional[str] = None):
        super().__init__(unit_id, UnitType.REACTOR, name)
        self.reactions: List[ChemicalReaction] = []
        self.reactor_type: str = "CSTR"
        self.volume: float = 1.0
        self.residence_time: float = 10.0
        self.heat_duty: float = 0.0  # Heat added/removed (W)
        self.catalyst_activity: float = 1.0  # 0.0 to 1.0

    def add_reaction(self, reaction: ChemicalReaction) -> None:
        self.reactions.append(reaction)

    def initialize(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize reactor parameters.
        
        Design Intent:
        - Sets physical dimensions and initial catalyst state
        - Prepares kinetics models for execution
        """
        if initial_conditions:
            self.volume = initial_conditions.get("volume", self.volume)
            self.residence_time = initial_conditions.get("residence_time", self.residence_time)
            self.reactor_type = initial_conditions.get("reactor_type", self.reactor_type)
            self.catalyst_activity = initial_conditions.get("catalyst_activity", 1.0)

    def execute(self, timestamp: float) -> UnitState:
        """
        Execute reactor step with mass and energy balance.
        
        Design Intent:
        - Models chemical transformation of species
        - Accounts for residence time effects (CSTR approximation)
        - Simulates catalyst degradation
        - Calculates outlet temperature based on heat duty
        
        Physics:
        - Accumulation = In - Out + Generation
        - dC/dt = (sum(F_in * C_in) - F_out * C_out) / V + r
        """
        state = UnitState(
            unit_id=self.unit_id,
            timestamp=timestamp,
            input_streams={k: v.copy() for k, v in self.input_streams.items()},
            output_streams={}
        )

        # Catalyst degradation: linear decay
        degradation_rate = self.get_parameter("degradation_rate", 0.001)
        self.catalyst_activity = max(0.0, self.catalyst_activity - degradation_rate)
        state.internal_state["catalyst_activity"] = self.catalyst_activity

        # Determine total input flow and composition
        total_in_flow = 0.0
        in_composition: Dict[str, float] = {}
        in_temp_weighted = 0.0

        for in_stream in self.input_streams.values():
            flow = in_stream.flow_rate or in_stream.total_flow()
            total_in_flow += flow
            if in_stream.temperature is not None:
                in_temp_weighted += in_stream.temperature * flow
            
            for species, amount in in_stream.composition.items():
                in_composition[species] = in_composition.get(species, 0.0) + amount

        avg_in_temp = in_temp_weighted / total_in_flow if total_in_flow > 0 else 25.0

        for out_name, out_stream in self.output_streams.items():
            new_stream = out_stream.copy()
            new_stream.flow_rate = total_in_flow # Steady state flow assumption
            
            # Simplified CSTR-like kinetics
            # Design Intent: establish the interface for future GNN-backed kinetics
            # Reaction: A -> B
            conversion_base = self.get_parameter("base_conversion", 0.5)
            # Conversion scales with residence time and catalyst activity
            actual_conversion = conversion_base * (1.0 - np.exp(-self.residence_time / 10.0)) * self.catalyst_activity
            
            for species, amount in in_composition.items():
                if species == "A":
                    converted = amount * actual_conversion
                    new_stream.composition["A"] = amount - converted
                    new_stream.composition["B"] = new_stream.composition.get("B", 0.0) + converted
                else:
                    new_stream.composition[species] = new_stream.composition.get(species, 0.0) + amount

            # Energy balance: T_out = T_in + (HeatDuty / (Flow * Cp))
            cp = self.get_parameter("heat_capacity", 4184.0) # J/kg*K
            delta_t = self.heat_duty / (total_in_flow * cp) if total_in_flow > 0 else 0
            new_stream.temperature = avg_in_temp + delta_t
            
            state.output_streams[out_name] = new_stream

        state.constraint_violations = self.check_constraints(state)
        state.is_valid = len(state.constraint_violations) == 0

        self._history.append(state)
        return state


class Separator(UnitOperation):
    """
    Unit operation for material separation (e.g., flash drum, filter).
    
    Design Intent:
    - Splits species based on separation efficiency
    - Models phase separation or purity targets
    - Essential for recycle loop closure
    """

    def __init__(self, unit_id: str, name: Optional[str] = None):
        super().__init__(unit_id, UnitType.SEPARATOR, name)
        self.efficiencies: Dict[str, float] = {}  # Species -> Fraction to primary output

    def set_efficiency(self, species: str, fraction: float) -> None:
        self.efficiencies[species] = fraction

    def initialize(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        if initial_conditions and "efficiencies" in initial_conditions:
            self.efficiencies.update(initial_conditions["efficiencies"])

    def execute(self, timestamp: float) -> UnitState:
        state = UnitState(
            unit_id=self.unit_id,
            timestamp=timestamp,
            input_streams={k: v.copy() for k, v in self.input_streams.items()},
            output_streams={}
        )

        if not self.input_streams or not self.output_streams:
            return state

        # Assume two outputs: 'overhead' and 'bottoms'
        # Overhead is the primary output for efficiency calculation
        primary_out_name = "overhead" if "overhead" in self.output_streams else next(iter(self.output_streams.keys()))
        
        for out_name, out_stream in self.output_streams.items():
            new_stream = out_stream.copy()
            new_stream.composition = {}
            
            for in_stream in self.input_streams.values():
                for species, conc in in_stream.composition.items():
                    eff = self.efficiencies.get(species, 0.5)
                    if out_name == primary_out_name:
                        new_stream.composition[species] = new_stream.composition.get(species, 0.0) + conc * eff
                    else:
                        new_stream.composition[species] = new_stream.composition.get(species, 0.0) + conc * (1 - eff)
                
                new_stream.temperature = in_stream.temperature
                new_stream.pressure = in_stream.pressure
            
            state.output_streams[out_name] = new_stream

        state.constraint_violations = self.check_constraints(state)
        state.is_valid = len(state.constraint_violations) == 0
        self._history.append(state)
        return state


class HeatExchanger(UnitOperation):
    """
    Unit operation for heat transfer.
    
    Design Intent:
    - Models temperature change without chemical reaction
    - Tracks energy transfer (heat duty)
    - Supports cooling and heating operations
    """

    def __init__(self, unit_id: str, name: Optional[str] = None):
        super().__init__(unit_id, UnitType.HEAT_EXCHANGER, name)
        self.target_temperature: Optional[float] = None
        self.ua: float = 1000.0  # Heat transfer coefficient * area (W/K)

    def initialize(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        if initial_conditions:
            self.target_temperature = initial_conditions.get("target_temperature")
            self.ua = initial_conditions.get("ua", self.ua)

    def execute(self, timestamp: float) -> UnitState:
        state = UnitState(
            unit_id=self.unit_id,
            timestamp=timestamp,
            input_streams={k: v.copy() for k, v in self.input_streams.items()},
            output_streams={}
        )

        for out_name, out_stream in self.output_streams.items():
            new_stream = out_stream.copy()
            
            for in_stream in self.input_streams.values():
                new_stream.composition = in_stream.composition.copy()
                new_stream.flow_rate = in_stream.flow_rate
                new_stream.pressure = in_stream.pressure
                
                if self.target_temperature is not None:
                    # Ideal temperature control
                    new_stream.temperature = self.target_temperature
                    # Energy calculation (simplified)
                    flow = in_stream.flow_rate or in_stream.total_flow()
                    cp = self.get_parameter("heat_capacity", 4184.0)
                    duty = flow * cp * (new_stream.temperature - in_stream.temperature)
                    state.internal_state["heat_duty"] = duty
                else:
                    new_stream.temperature = in_stream.temperature

            state.output_streams[out_name] = new_stream

        state.constraint_violations = self.check_constraints(state)
        state.is_valid = len(state.constraint_violations) == 0
        self._history.append(state)
        return state


class Pump(UnitOperation):
    """
    Unit operation for increasing fluid pressure.
    
    Design Intent:
    - Models pressure increase and power consumption
    - Supports pump curves and efficiency assumptions
    """

    def __init__(self, unit_id: str, name: Optional[str] = None):
        super().__init__(unit_id, UnitType.PUMP, name)
        self.delta_p: float = 1.0  # Pressure increase (atm)
        self.efficiency: float = 0.75

    def initialize(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        if initial_conditions:
            self.delta_p = initial_conditions.get("delta_p", self.delta_p)
            self.efficiency = initial_conditions.get("efficiency", self.efficiency)

    def execute(self, timestamp: float) -> UnitState:
        state = UnitState(
            unit_id=self.unit_id,
            timestamp=timestamp,
            input_streams={k: v.copy() for k, v in self.input_streams.items()},
            output_streams={}
        )

        for out_name, out_stream in self.output_streams.items():
            new_stream = out_stream.copy()
            for in_stream in self.input_streams.values():
                new_stream.composition = in_stream.composition.copy()
                new_stream.flow_rate = in_stream.flow_rate
                new_stream.temperature = in_stream.temperature
                if in_stream.pressure is not None:
                    new_stream.pressure = in_stream.pressure + self.delta_p
                
                # Power calculation: P = (Q * delta_P) / efficiency
                flow = in_stream.flow_rate or in_stream.total_flow()
                power = (flow * self.delta_p * 101325) / self.efficiency # approx W
                state.internal_state["power_consumption"] = power

            state.output_streams[out_name] = new_stream

        state.constraint_violations = self.check_constraints(state)
        state.is_valid = len(state.constraint_violations) == 0
        self._history.append(state)
        return state


class Storage(UnitOperation):
    """
    Unit operation for material storage (tanks, vessels).
    
    Design Intent:
    - Models material hold-up and inventory
    - Buffers flow fluctuations
    - Tracks tank level and residence time
    """

    def __init__(self, unit_id: str, name: Optional[str] = None):
        super().__init__(unit_id, UnitType.STORAGE, name)
        self.capacity: float = 100.0
        self.current_inventory: Dict[str, float] = {}

    def initialize(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        if initial_conditions and "inventory" in initial_conditions:
            self.current_inventory = initial_conditions["inventory"].copy()

    def execute(self, timestamp: float) -> UnitState:
        state = UnitState(
            unit_id=self.unit_id,
            timestamp=timestamp,
            input_streams={k: v.copy() for k, v in self.input_streams.items()},
            output_streams={}
        )

        # Update inventory based on inputs
        for in_stream in self.input_streams.values():
            for species, conc in in_stream.composition.items():
                self.current_inventory[species] = self.current_inventory.get(species, 0.0) + conc

        total_inv = sum(self.current_inventory.values())
        state.internal_state["total_inventory"] = total_inv
        state.internal_state["level_percent"] = (total_inv / self.capacity) * 100 if self.capacity > 0 else 0

        # Set outputs based on target flow rate or discharge logic
        discharge_fraction = self.get_parameter("discharge_fraction", 0.1)
        for out_name, out_stream in self.output_streams.items():
            new_stream = out_stream.copy()
            new_stream.composition = {}
            for species, inv in self.current_inventory.items():
                amount = inv * discharge_fraction
                new_stream.composition[species] = amount
                self.current_inventory[species] -= amount
            
            state.output_streams[out_name] = new_stream

        state.constraint_violations = self.check_constraints(state)
        state.is_valid = len(state.constraint_violations) == 0
        self._history.append(state)
        return state


class Mixer(UnitOperation):
    """
    Mixer unit operation - combines multiple input streams.

    Design Intent:
    - Simple material mixing logic
    - Energy balance-aware (temperature averaging)
    - Mass conservation enforced
    """

    def __init__(self, unit_id: str, name: Optional[str] = None):
        super().__init__(unit_id, UnitType.MIXER, name)

    def initialize(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        pass

    def execute(self, timestamp: float) -> UnitState:
        state = UnitState(
            unit_id=self.unit_id,
            timestamp=timestamp,
            input_streams={k: v.copy() for k, v in self.input_streams.items()},
            output_streams={}
        )

        if not self.input_streams:
            return state

        total_mass = 0.0
        temp_sum = 0.0
        mixed_composition: Dict[str, float] = {}

        for in_stream in self.input_streams.values():
            stream_mass = in_stream.total_flow()
            total_mass += stream_mass

            if in_stream.temperature is not None:
                temp_sum += in_stream.temperature * stream_mass

            for species, conc in in_stream.composition.items():
                if species in mixed_composition:
                    mixed_composition[species] += conc
                else:
                    mixed_composition[species] = conc

        for out_name, out_stream in self.output_streams.items():
            new_stream = out_stream.copy()
            new_stream.composition = mixed_composition.copy()
            new_stream.flow_rate = total_mass
            if total_mass > 0 and temp_sum > 0:
                new_stream.temperature = temp_sum / total_mass
            state.output_streams[out_name] = new_stream

        state.constraint_violations = self.check_constraints(state)
        state.is_valid = len(state.constraint_violations) == 0

        self._history.append(state)
        return state


class Splitter(UnitOperation):
    """
    Splitter unit operation - splits input stream into multiple outputs.

    Design Intent:
    - Mass-conserving stream splitting
    - Supports split ratio configuration
    - Maintains stream properties
    """

    def __init__(self, unit_id: str, name: Optional[str] = None):
        super().__init__(unit_id, UnitType.SPLITTER, name)
        self.split_ratios: Dict[str, float] = {}

    def set_split_ratio(self, output_name: str, ratio: float) -> None:
        self.split_ratios[output_name] = ratio

    def initialize(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        if not self.split_ratios and self.output_streams:
            num_outputs = len(self.output_streams)
            ratio = 1.0 / num_outputs
            for out_name in self.output_streams.keys():
                self.split_ratios[out_name] = ratio

    def execute(self, timestamp: float) -> UnitState:
        state = UnitState(
            unit_id=self.unit_id,
            timestamp=timestamp,
            input_streams={k: v.copy() for k, v in self.input_streams.items()},
            output_streams={}
        )

        if not self.input_streams:
            return state

        primary_input = next(iter(self.input_streams.values()))

        for out_name, out_stream in self.output_streams.items():
            ratio = self.split_ratios.get(out_name, 1.0 / len(self.output_streams))
            new_stream = out_stream.copy()

            new_stream.composition = {
                species: conc * ratio
                for species, conc in primary_input.composition.items()
            }
            new_stream.temperature = primary_input.temperature
            new_stream.pressure = primary_input.pressure
            if primary_input.flow_rate:
                new_stream.flow_rate = primary_input.flow_rate * ratio

            state.output_streams[out_name] = new_stream

        state.constraint_violations = self.check_constraints(state)
        state.is_valid = len(state.constraint_violations) == 0

        self._history.append(state)
        return state


class ProcessGraph:
    """
    Process graph representing the entire plant topology.

    Design Intent:
    - Manages the complete process network
    - Handles topological sorting and execution order
    - Tracks stream connections
    - Supports plant-level state aggregation
    - Extensible for complex process layouts

    Architectural Principles:
    - Nodes = UnitOperations
    - Edges = Streams
    - No cycles without explicit recycle handling
    - Single responsibility: topology management only
    """

    def __init__(self, name: str = "process"):
        self.name = name
        self.nodes: Dict[str, ProcessNode] = {}
        self.units: Dict[str, UnitOperation] = {} # For backward compatibility
        self.streams: Dict[str, Stream] = {}
        self._graph = nx.DiGraph()
        self._history: List[Dict[str, Any]] = []

    def add_node(self, node: ProcessNode) -> None:
        """Add a generic process node to the graph."""
        self.nodes[node.node_id] = node
        if isinstance(node, UnitOperation):
            self.units[node.unit_id] = node
        self._graph.add_node(node.node_id, node_type=node.node_type)

    def add_unit(self, unit: UnitOperation) -> None:
        """Add a unit operation (legacy support)."""
        self.add_node(unit)

    def add_stream(self, stream: Stream, source: Optional[UnitOperation] = None,
                   target: Optional[UnitOperation] = None) -> None:
        """Add a stream and connect units."""
        self.streams[stream.stream_id] = stream

        if source:
            stream.source_unit = source.unit_id
            source.add_output_stream(stream)

        if target:
            stream.target_unit = target.unit_id
            target.add_input_stream(stream)

        if source and target:
            self._graph.add_edge(source.unit_id, target.unit_id, stream_id=stream.stream_id)

    def connect_units(self, source: UnitOperation, target: UnitOperation,
                      stream: Optional[Stream] = None) -> Stream:
        """Connect two units with a stream."""
        if stream is None:
            stream = Stream(
                stream_id=f"stream_{source.unit_id}_{target.unit_id}"
            )

        self.add_stream(stream, source, target)
        return stream

    def get_execution_order(self) -> List[str]:
        """
        Get topological execution order of units.

        Design Intent:
        - Ensures correct dependency resolution
        - Handles DAG structure only
        - Recycles require special handling

        Returns:
            Ordered list of unit IDs
        """
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Process graph contains cycles - recycle loops require special handling")

    def initialize_all(self, initial_conditions: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Initialize all units in the process."""
        conditions = initial_conditions or {}
        for unit in self.units.values():
            unit.initialize(conditions.get(unit.unit_id))

    def execute(self, num_steps: int = 1, dt: float = 1.0) -> List[Dict[str, UnitState]]:
        """
        Execute the complete process for multiple steps.

        Design Intent:
        - Orchestrates plant-level execution
        - Maintains correct execution order
        - Aggregates plant state
        - Tracks process trajectory

        Args:
            num_steps: Number of execution steps
            dt: Time delta per step

        Returns:
            List of plant states (each state is dict of UnitState)
        """
        execution_order = self.get_execution_order()

        for step_idx in range(num_steps):
            timestamp = step_idx * dt
            plant_state: Dict[str, UnitState] = {}

            # Handle recycle loops via fixed-point iteration if cycles detected
            has_cycles = not nx.is_directed_acyclic_graph(self._graph)
            
            if not has_cycles:
                # Normal DAG execution
                for unit_id in execution_order:
                    unit = self.units[unit_id]
                    unit_state = unit.execute(timestamp)
                    plant_state[unit_id] = unit_state
            else:
                # Iterative execution for recycles
                # Design Intent: Converge on mass balance across recycles
                max_iter = self.get_parameter("max_recycle_iterations", 5)
                convergence_threshold = self.get_parameter("convergence_threshold", 1e-4)
                
                for i in range(max_iter):
                    prev_plant_state = plant_state.copy()
                    current_plant_state: Dict[str, UnitState] = {}
                    
                    # Execution order is non-trivial for cycles, use arbitrary but consistent order
                    for unit_id in self.units.keys():
                        unit = self.units[unit_id]
                        unit_state = unit.execute(timestamp)
                        current_plant_state[unit_id] = unit_state
                    
                    plant_state = current_plant_state
                    
                    # Simple convergence check on total flow
                    if prev_plant_state:
                        diff = 0.0
                        for uid in plant_state:
                            f1 = sum(plant_state[uid].output_streams[s].total_flow() for s in plant_state[uid].output_streams)
                            f2 = sum(prev_plant_state[uid].output_streams[s].total_flow() for s in prev_plant_state[uid].output_streams)
                            diff += abs(f1 - f2)
                        
                        if diff < convergence_threshold:
                            break

            self._history.append(plant_state)

        return self._history.copy()

    def set_parameter(self, key: str, value: Any) -> None:
        """Set a plant-level parameter."""
        if not hasattr(self, "_parameters"):
            self._parameters = {}
        self._parameters[key] = value

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a plant-level parameter."""
        if not hasattr(self, "_parameters"):
            return default
        return self._parameters.get(key, default)

    def get_plant_history(self) -> List[Dict[str, UnitState]]:
        """Get the complete execution history of the plant."""
        return self._history.copy()

    def get_unit(self, unit_id: str) -> Optional[UnitOperation]:
        """Get a unit by ID."""
        return self.units.get(unit_id)
