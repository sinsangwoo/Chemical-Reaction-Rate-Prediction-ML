"""
Industrial Process Constraint Foundation.

This module provides:
- Engineering constraint abstractions
- Temperature/pressure operating limits
- Equipment capacity constraints
- Throughput limits
- Energy consumption boundaries
- Constraint violation propagation
- Process feasibility validation

Design Intent:
- Extensible constraint system, not hardcoded rules
- Clear constraint ownership semantics
- Explicit violation handling
- Compatible with future optimization workflows
- Traceable constraint provenance

Architectural Location:
- Constraints live at the UnitOperation and ProcessGraph levels
- Constraints are checked during execution
- Violations propagate up to plant level
- No silent constraint failures
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from .network import UnitOperation, UnitState, Stream


class ConstraintType(Enum):
    """Types of industrial process constraints."""
    TEMPERATURE_LIMIT = "temperature_limit"
    PRESSURE_LIMIT = "pressure_limit"
    CAPACITY_LIMIT = "capacity_limit"
    THROUGHPUT_LIMIT = "throughput_limit"
    COMPOSITION_LIMIT = "composition_limit"
    ENERGY_LIMIT = "energy_limit"
    RAMP_RATE_LIMIT = "ramp_rate_limit"
    SAFETY_LIMIT = "safety_limit"
    CATALYST_CONSTRAINT = "catalyst_constraint"
    TELEMETRY_VALIDITY = "telemetry_validity"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    INVARIANT_CHECK = "invariant_check"


class ConstraintSeverity(Enum):
    """Severity level for constraint violations."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SAFETY_STOP = "safety_stop"


@dataclass
class ConstraintViolation:
    """
    Record of a constraint violation.

    Design Intent:
    - Captures complete violation context
    - Supports forensic analysis
    - Traceable to specific constraint and unit
    """
    constraint_name: str
    constraint_type: ConstraintType
    severity: ConstraintSeverity
    unit_id: Optional[str]
    timestamp: Optional[float]
    message: str
    current_value: Optional[Any] = None
    limit_value: Optional[Any] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ProcessConstraint(ABC):
    """
    Abstract base class for process constraints.

    Design Intent:
    - Defines the constraint interface
    - Supports both hard and soft constraints
    - Encapsulates constraint logic
    - Extensible for custom constraints

    Architectural Principles:
    - A constraint knows what it checks
    - A constraint knows how to check it
    - A constraint knows what to do when violated
    - Constraints are stateless (except configuration)
    """

    def __init__(
        self,
        name: str,
        constraint_type: ConstraintType,
        severity: ConstraintSeverity = ConstraintSeverity.ERROR,
        description: str = ""
    ):
        self.name = name
        self.constraint_type = constraint_type
        self.severity = severity
        self.description = description

    @abstractmethod
    def check(self, unit: UnitOperation, state: UnitState) -> bool:
        """
        Check if the constraint is satisfied.

        Args:
            unit: The unit operation being checked
            state: Current state of the unit

        Returns:
            True if constraint is satisfied, False if violated
        """
        pass

    @abstractmethod
    def get_violation(self, unit: UnitOperation, state: UnitState) -> Optional[ConstraintViolation]:
        """
        Get detailed violation information if constraint is violated.

        Args:
            unit: The unit operation being checked
            state: Current state of the unit

        Returns:
            ConstraintViolation if violated, None otherwise
        """
        pass


class TemperatureLimit(ProcessConstraint):
    """
    Temperature operating limit constraint.

    Design Intent:
    - Enforces minimum and maximum temperature bounds
    - Supports both inlet and outlet temperature checks
    - Critical for equipment safety and reaction kinetics
    """

    def __init__(
        self,
        name: str,
        min_temp: Optional[float] = None,
        max_temp: Optional[float] = None,
        check_input: bool = True,
        check_output: bool = True,
        severity: ConstraintSeverity = ConstraintSeverity.ERROR,
        description: str = ""
    ):
        super().__init__(name, ConstraintType.TEMPERATURE_LIMIT, severity, description)
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.check_input = check_input
        self.check_output = check_output

    def check(self, unit: UnitOperation, state: UnitState) -> bool:
        streams = []
        if self.check_input:
            streams.extend(state.input_streams.values())
        if self.check_output:
            streams.extend(state.output_streams.values())

        for stream in streams:
            if stream.temperature is not None:
                if self.min_temp is not None and stream.temperature < self.min_temp:
                    return False
                if self.max_temp is not None and stream.temperature > self.max_temp:
                    return False
        return True

    def get_violation(self, unit: UnitOperation, state: UnitState) -> Optional[ConstraintViolation]:
        if self.check(unit, state):
            return None

        streams = []
        if self.check_input:
            streams.extend(state.input_streams.values())
        if self.check_output:
            streams.extend(state.output_streams.values())

        for stream in streams:
            if stream.temperature is not None:
                if self.min_temp is not None and stream.temperature < self.min_temp:
                    return ConstraintViolation(
                        constraint_name=self.name,
                        constraint_type=self.constraint_type,
                        severity=self.severity,
                        unit_id=unit.unit_id,
                        timestamp=state.timestamp,
                        message=f"Temperature {stream.temperature:.2f}°C below minimum {self.min_temp:.2f}°C",
                        current_value=stream.temperature,
                        limit_value=self.min_temp
                    )
                if self.max_temp is not None and stream.temperature > self.max_temp:
                    return ConstraintViolation(
                        constraint_name=self.name,
                        constraint_type=self.constraint_type,
                        severity=self.severity,
                        unit_id=unit.unit_id,
                        timestamp=state.timestamp,
                        message=f"Temperature {stream.temperature:.2f}°C above maximum {self.max_temp:.2f}°C",
                        current_value=stream.temperature,
                        limit_value=self.max_temp
                    )
        return None


class PressureLimit(ProcessConstraint):
    """
    Pressure operating limit constraint.

    Design Intent:
    - Enforces pressure bounds for equipment safety
    - Critical for pressure vessel integrity
    - Supports both absolute and gauge pressure
    """

    def __init__(
        self,
        name: str,
        min_pressure: Optional[float] = None,
        max_pressure: Optional[float] = None,
        severity: ConstraintSeverity = ConstraintSeverity.CRITICAL,
        description: str = ""
    ):
        super().__init__(name, ConstraintType.PRESSURE_LIMIT, severity, description)
        self.min_pressure = min_pressure
        self.max_pressure = max_pressure

    def check(self, unit: UnitOperation, state: UnitState) -> bool:
        streams = list(state.input_streams.values()) + list(state.output_streams.values())
        for stream in streams:
            if stream.pressure is not None:
                if self.min_pressure is not None and stream.pressure < self.min_pressure:
                    return False
                if self.max_pressure is not None and stream.pressure > self.max_pressure:
                    return False
        return True

    def get_violation(self, unit: UnitOperation, state: UnitState) -> Optional[ConstraintViolation]:
        if self.check(unit, state):
            return None

        streams = list(state.input_streams.values()) + list(state.output_streams.values())
        for stream in streams:
            if stream.pressure is not None:
                if self.min_pressure is not None and stream.pressure < self.min_pressure:
                    return ConstraintViolation(
                        constraint_name=self.name,
                        constraint_type=self.constraint_type,
                        severity=self.severity,
                        unit_id=unit.unit_id,
                        timestamp=state.timestamp,
                        message=f"Pressure {stream.pressure:.2f} atm below minimum {self.min_pressure:.2f} atm",
                        current_value=stream.pressure,
                        limit_value=self.min_pressure
                    )
                if self.max_pressure is not None and stream.pressure > self.max_pressure:
                    return ConstraintViolation(
                        constraint_name=self.name,
                        constraint_type=self.constraint_type,
                        severity=self.severity,
                        unit_id=unit.unit_id,
                        timestamp=state.timestamp,
                        message=f"Pressure {stream.pressure:.2f} atm above maximum {self.max_pressure:.2f} atm",
                        current_value=stream.pressure,
                        limit_value=self.max_pressure
                    )
        return None


class CapacityLimit(ProcessConstraint):
    """
    Equipment capacity constraint.

    Design Intent:
    - Enforces maximum throughput/capacity of equipment
    - Prevents overloading of unit operations
    - Critical for realistic process simulation
    """

    def __init__(
        self,
        name: str,
        max_capacity: float,
        capacity_type: str = "flow_rate",
        severity: ConstraintSeverity = ConstraintSeverity.WARNING,
        description: str = ""
    ):
        super().__init__(name, ConstraintType.CAPACITY_LIMIT, severity, description)
        self.max_capacity = max_capacity
        self.capacity_type = capacity_type

    def check(self, unit: UnitOperation, state: UnitState) -> bool:
        total_flow = 0.0
        for stream in state.output_streams.values():
            if stream.flow_rate is not None:
                total_flow += stream.flow_rate
            else:
                total_flow += stream.total_flow()
        return total_flow <= self.max_capacity

    def get_violation(self, unit: UnitOperation, state: UnitState) -> Optional[ConstraintViolation]:
        if self.check(unit, state):
            return None

        total_flow = 0.0
        for stream in state.output_streams.values():
            if stream.flow_rate is not None:
                total_flow += stream.flow_rate
            else:
                total_flow += stream.total_flow()

        return ConstraintViolation(
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            severity=self.severity,
            unit_id=unit.unit_id,
            timestamp=state.timestamp,
            message=f"Capacity exceeded: {total_flow:.2f} > {self.max_capacity:.2f}",
            current_value=total_flow,
            limit_value=self.max_capacity
        )


class CompositionLimit(ProcessConstraint):
    """
    Chemical composition constraint.

    Design Intent:
    - Enforces limits on specific species concentrations
    - Useful for product purity requirements
    - Supports both minimum and maximum bounds
    """

    def __init__(
        self,
        name: str,
        species: str,
        min_concentration: Optional[float] = None,
        max_concentration: Optional[float] = None,
        severity: ConstraintSeverity = ConstraintSeverity.WARNING,
        description: str = ""
    ):
        super().__init__(name, ConstraintType.COMPOSITION_LIMIT, severity, description)
        self.species = species
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration

    def check(self, unit: UnitOperation, state: UnitState) -> bool:
        for stream in state.output_streams.values():
            conc = stream.composition.get(self.species, 0.0)
            if self.min_concentration is not None and conc < self.min_concentration:
                return False
            if self.max_concentration is not None and conc > self.max_concentration:
                return False
        return True

    def get_violation(self, unit: UnitOperation, state: UnitState) -> Optional[ConstraintViolation]:
        if self.check(unit, state):
            return None

        return None


class EnergyLimit(ProcessConstraint):
    """
    Energy consumption limit constraint.
    
    Design Intent:
    - Enforces maximum power or heat duty bounds
    - Critical for operational cost and utility capacity
    - Supports both unit-level and global energy caps
    """

    def __init__(
        self,
        name: str,
        max_power: float,
        severity: ConstraintSeverity = ConstraintSeverity.ERROR,
        description: str = ""
    ):
        super().__init__(name, ConstraintType.ENERGY_LIMIT, severity, description)
        self.max_power = max_power

    def check(self, unit: UnitOperation, state: UnitState) -> bool:
        power = state.internal_state.get("power_consumption", 0.0)
        duty = abs(state.internal_state.get("heat_duty", 0.0))
        total_energy = power + duty
        return total_energy <= self.max_power

    def get_violation(self, unit: UnitOperation, state: UnitState) -> Optional[ConstraintViolation]:
        power = state.internal_state.get("power_consumption", 0.0)
        duty = abs(state.internal_state.get("heat_duty", 0.0))
        total_energy = power + duty
        
        if total_energy > self.max_power:
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                severity=self.severity,
                unit_id=unit.unit_id,
                timestamp=state.timestamp,
                message=f"Energy limit exceeded: {total_energy:.2f} W > {self.max_power:.2f} W",
                current_value=total_energy,
                limit_value=self.max_power
            )
        return None


class RampRateLimit(ProcessConstraint):
    """
    Operational ramp rate constraint.
    
    Design Intent:
    - Enforces limits on how fast process variables can change
    - Critical for equipment longevity and stability
    - Prevents "thermal shock" or sudden pressure surges
    """

    def __init__(
        self,
        name: str,
        max_temp_change: float,  # K/s
        severity: ConstraintSeverity = ConstraintSeverity.WARNING,
        description: str = ""
    ):
        super().__init__(name, ConstraintType.RAMP_RATE_LIMIT, severity, description)
        self.max_temp_change = max_temp_change

    def check(self, unit: UnitOperation, state: UnitState) -> bool:
        history = unit.get_history()
        if not history:
            return True
        
        prev_state = history[-1]
        dt = state.timestamp - prev_state.timestamp
        if dt <= 0:
            return True
        
        for s_name, stream in state.output_streams.items():
            if s_name in prev_state.output_streams:
                prev_stream = prev_state.output_streams[s_name]
                if stream.temperature is not None and prev_stream.temperature is not None:
                    rate = abs(stream.temperature - prev_stream.temperature) / dt
                    if rate > self.max_temp_change:
                        return False
        return True

    def get_violation(self, unit: UnitOperation, state: UnitState) -> Optional[ConstraintViolation]:
        history = unit.get_history()
        if not history:
            return None
        
        prev_state = history[-1]
        dt = state.timestamp - prev_state.timestamp
        if dt <= 0:
            return None
            
        for s_name, stream in state.output_streams.items():
            if s_name in prev_state.output_streams:
                prev_stream = prev_state.output_streams[s_name]
                if stream.temperature is not None and prev_stream.temperature is not None:
                    rate = abs(stream.temperature - prev_stream.temperature) / dt
                    if rate > self.max_temp_change:
                        return ConstraintViolation(
                            constraint_name=self.name,
                            constraint_type=self.constraint_type,
                            severity=self.severity,
                            unit_id=unit.unit_id,
                            timestamp=state.timestamp,
                            message=f"Ramp rate exceeded: {rate:.2f} K/s > {self.max_temp_change:.2f} K/s",
                            current_value=rate,
                            limit_value=self.max_temp_change
                        )
        return None


class ThroughputLimit(ProcessConstraint):
    """
    Operational throughput limit.
    
    Design Intent:
    - Enforces bounds on specific species production rate
    - Critical for meeting production targets without overloading
    """

    def __init__(
        self,
        name: str,
        species: str,
        max_throughput: float,
        severity: ConstraintSeverity = ConstraintSeverity.ERROR,
        description: str = ""
    ):
        super().__init__(name, ConstraintType.THROUGHPUT_LIMIT, severity, description)
        self.species = species
        self.max_throughput = max_throughput

    def check(self, unit: UnitOperation, state: UnitState) -> bool:
        throughput = 0.0
        for stream in state.output_streams.values():
            throughput += stream.composition.get(self.species, 0.0)
        return throughput <= self.max_throughput

    def get_violation(self, unit: UnitOperation, state: UnitState) -> Optional[ConstraintViolation]:
        throughput = 0.0
        for stream in state.output_streams.values():
            throughput += stream.composition.get(self.species, 0.0)
            
        if throughput > self.max_throughput:
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                severity=self.severity,
                unit_id=unit.unit_id,
                timestamp=state.timestamp,
                message=f"Throughput limit for {self.species} exceeded: {throughput:.2f} > {self.max_throughput:.2f}",
                current_value=throughput,
                limit_value=self.max_throughput
            )
        return None


class SafetyLimit(ProcessConstraint):
    """
    Operational safety constraint.
    
    Design Intent:
    - Enforces critical safety bounds (e.g., explosive concentration limits)
    - Always high severity
    - Hardcoded logic for critical safety checks
    """

    def __init__(
        self,
        name: str,
        safety_logic: Callable[[UnitOperation, UnitState], bool],
        description: str = "Safety violation detected"
    ):
        super().__init__(name, ConstraintType.SAFETY_LIMIT, ConstraintSeverity.SAFETY_STOP, description)
        self.safety_logic = safety_logic

    def check(self, unit: UnitOperation, state: UnitState) -> bool:
        return self.safety_logic(unit, state)

    def get_violation(self, unit: UnitOperation, state: UnitState) -> Optional[ConstraintViolation]:
        if not self.check(unit, state):
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                severity=self.severity,
                unit_id=unit.unit_id,
                timestamp=state.timestamp,
                message=self.description
            )
        return None


class ConstraintValidator:
    """
    Centralized constraint validation system.

    Design Intent:
    - Manages all constraints at plant level
    - Aggregates violations across units
    - Supports process feasibility checks
    - Provides violation reporting

    Architectural Location:
    - Sits at the ProcessGraph level
    - Coordinates unit-level constraint checks
    - Owns the violation collection and reporting
    """

    def __init__(self):
        self.global_constraints: List[ProcessConstraint] = []
        self.violations: List[ConstraintViolation] = []

    def add_global_constraint(self, constraint: ProcessConstraint) -> None:
        """Add a constraint that applies to all units."""
        self.global_constraints.append(constraint)

    def validate_unit(self, unit: UnitOperation, state: UnitState) -> List[ConstraintViolation]:
        """Validate a single unit against all applicable constraints."""
        violations: List[ConstraintViolation] = []

        for constraint in unit.constraints:
            if not constraint.check(unit, state):
                violation = constraint.get_violation(unit, state)
                if violation:
                    violations.append(violation)
                    self.violations.append(violation)

        for constraint in self.global_constraints:
            if not constraint.check(unit, state):
                violation = constraint.get_violation(unit, state)
                if violation:
                    violations.append(violation)
                    self.violations.append(violation)

        return violations

    def validate_process(self, plant_state: Dict[str, UnitState],
                        units: Dict[str, UnitOperation]) -> List[ConstraintViolation]:
        """Validate the entire process plant."""
        all_violations: List[ConstraintViolation] = []

        for unit_id, state in plant_state.items():
            unit = units.get(unit_id)
            if unit:
                unit_violations = self.validate_unit(unit, state)
                all_violations.extend(unit_violations)

        return all_violations

    def is_feasible(self) -> bool:
        """Check if process is feasible (no critical errors)."""
        return not any(
            v.severity in (ConstraintSeverity.CRITICAL, ConstraintSeverity.SAFETY_STOP)
            for v in self.violations
        )

    def get_violations_by_severity(self) -> Dict[ConstraintSeverity, List[ConstraintViolation]]:
        """Get violations grouped by severity."""
        grouped: Dict[ConstraintSeverity, List[ConstraintViolation]] = {}
        for severity in ConstraintSeverity:
            grouped[severity] = []
        for v in self.violations:
            grouped[v.severity].append(v)
        return grouped

    def clear_violations(self) -> None:
        """Clear accumulated violations."""
        self.violations = []


class TelemetryValidityConstraint(ProcessConstraint):
    """
    Constraint for validating telemetry data quality.

    Design Intent:
    - Verifies data is within expected physical ranges
    - Detects sensor malfunctions or communication errors
    - Essential for trustworthy observability
    """

    def __init__(
        self,
        name: str,
        tag: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        severity: ConstraintSeverity = ConstraintSeverity.ERROR,
        description: str = "Telemetry value out of valid range"
    ):
        super().__init__(name, ConstraintType.TELEMETRY_VALIDITY, severity, description)
        self.tag = tag
        self.min_val = min_val
        self.max_val = max_val

    def check(self, unit: "UnitOperation", state: "UnitState") -> bool:
        val = state.internal_state.get(self.tag)
        if val is None:
            return True # Assume missing data is handled elsewhere
        
        try:
            f_val = float(val)
            if self.min_val is not None and f_val < self.min_val:
                return False
            if self.max_val is not None and f_val > self.max_val:
                return False
        except (ValueError, TypeError):
            return False
            
        return True

    def get_violation(self, unit: "UnitOperation", state: "UnitState") -> Optional[ConstraintViolation]:
        if not self.check(unit, state):
            val = state.internal_state.get(self.tag)
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                severity=self.severity,
                unit_id=unit.unit_id,
                timestamp=state.timestamp,
                message=f"{self.description}: {self.tag}={val}",
                current_value=val,
                details={"tag": self.tag, "min": self.min_val, "max": self.max_val}
            )
        return None


class TemporalConsistencyConstraint(ProcessConstraint):
    """
    Constraint for validating temporal consistency between states.

    Design Intent:
    - Detects unrealistic jumps in telemetry values
    - Validates process inertia assumptions
    - Critical for detecting data glitches during replay
    """

    def __init__(
        self,
        name: str,
        tag: str,
        max_delta: float,
        severity: ConstraintSeverity = ConstraintSeverity.WARNING,
        description: str = "Temporal jump detected in telemetry"
    ):
        super().__init__(name, ConstraintType.TEMPORAL_CONSISTENCY, severity, description)
        self.tag = tag
        self.max_delta = max_delta

    def check(self, unit: "UnitOperation", state: "UnitState") -> bool:
        history = unit.get_history()
        if not history:
            return True
        
        prev_state = history[-1]
        curr_val = state.internal_state.get(self.tag)
        prev_val = prev_state.internal_state.get(self.tag)
        
        if curr_val is None or prev_val is None:
            return True
            
        try:
            delta = abs(float(curr_val) - float(prev_val))
            return delta <= self.max_delta
        except (ValueError, TypeError):
            return False

    def get_violation(self, unit: "UnitOperation", state: "UnitState") -> Optional[ConstraintViolation]:
        if not self.check(unit, state):
            curr_val = state.internal_state.get(self.tag)
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                severity=self.severity,
                unit_id=unit.unit_id,
                timestamp=state.timestamp,
                message=f"{self.description}: {self.tag} delta > {self.max_delta}",
                current_value=curr_val,
                details={"tag": self.tag, "max_delta": self.max_delta}
            )
        return None
