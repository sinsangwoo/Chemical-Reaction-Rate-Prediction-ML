"""
Industrial KPI Foundation.

This module provides:
- Throughput calculation
- Yield and conversion metrics
- Bottleneck identification
- Process efficiency calculation
- Energy usage estimation
- Operational stability metrics
- Uncertainty-aware KPI reporting
- Future optimization compatibility

Design Intent:
- KPIs correspond to meaningful process semantics
- Traceable to simulation outputs
- Support future optimization workflows
- Engineer-oriented, not research-oriented
- No fake dashboard metrics

Architectural Location:
- KPIs live at multiple levels:
  - Unit-level KPIs (per-unit metrics)
  - Stream-level KPIs (material/energy flow)
  - Plant-level KPIs (aggregated process metrics)
- KPI ownership propagates up the hierarchy
- Aggregation is explicit and traceable
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from .network import UnitOperation, UnitState, Stream, ProcessGraph


class KPILevel(Enum):
    """Level at which a KPI is calculated."""
    STREAM = "stream"
    UNIT = "unit"
    PLANT = "plant"


class KPIType(Enum):
    """Type of industrial KPI."""
    THROUGHPUT = "throughput"
    YIELD = "yield"
    CONVERSION = "conversion"
    SELECTIVITY = "selectivity"
    EFFICIENCY = "efficiency"
    ENERGY_USAGE = "energy_usage"
    BOTTLENECK_SCORE = "bottleneck_score"
    UTILIZATION = "utilization"
    STABILITY = "stability"
    ROLLING_STATISTIC = "rolling_statistic"
    TEMPORAL_TREND = "temporal_trend"
    UNCERTAINTY_ENVELOPE = "uncertainty_envelope"


@dataclass
class KPIValue:
    """
    Single KPI measurement with uncertainty.

    Design Intent:
    - Encapsulates KPI value and uncertainty
    - Supports statistical interpretation
    - Traceable to source data
    """
    name: str
    kpi_type: KPIType
    level: KPILevel
    value: float
    unit: str
    uncertainty: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    timestamp: Optional[float] = None
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessKPIs:
    """
    Collection of KPIs for a process at a point in time.

    Design Intent:
    - Aggregates KPIs at all levels
    - Supports snapshot comparison
    - Enables KPI trend analysis
    """
    timestamp: float
    stream_kpis: Dict[str, List[KPIValue]] = field(default_factory=dict)
    unit_kpis: Dict[str, List[KPIValue]] = field(default_factory=dict)
    plant_kpis: List[KPIValue] = field(default_factory=dict)

    def get_all_kpis(self) -> List[KPIValue]:
        all_kpis = []
        for kpis in self.stream_kpis.values():
            all_kpis.extend(kpis)
        for kpis in self.unit_kpis.values():
            all_kpis.extend(kpis)
        all_kpis.extend(self.plant_kpis)
        return all_kpis


class BaseKPI(ABC):
    """
    Abstract base class for KPI calculators.

    Design Intent:
    - Defines KPI calculation interface
    - Supports both deterministic and stochastic KPIs
    - Extensible for custom KPI types
    """

    def __init__(self, name: str, kpi_type: KPIType, level: KPILevel):
        self.name = name
        self.kpi_type = kpi_type
        self.level = level

    @abstractmethod
    def calculate(self, *args, **kwargs) -> KPIValue:
        """Calculate the KPI value."""
        pass


class ThroughputKPI(BaseKPI):
    """
    Throughput KPI - material flow rate through a unit or plant.

    Design Intent:
    - Measures production rate
    - Critical for capacity planning
    - Supports bottleneck identification
    """

    def __init__(self, name: str = "throughput", species: Optional[str] = None):
        super().__init__(name, KPIType.THROUGHPUT, KPILevel.UNIT)
        self.species = species

    def calculate(self, unit: UnitOperation, state: UnitState) -> KPIValue:
        total_throughput = 0.0

        for stream in state.output_streams.values():
            if self.species:
                total_throughput += stream.composition.get(self.species, 0.0)
            else:
                total_throughput += stream.total_flow()

        return KPIValue(
            name=self.name,
            kpi_type=self.kpi_type,
            level=self.level,
            value=total_throughput,
            unit="mol/s" if self.species is None else "mol/s",
            timestamp=state.timestamp,
            source_id=unit.unit_id
        )


class YieldKPI(BaseKPI):
    """
    Yield KPI - product produced vs theoretical maximum.

    Design Intent:
    - Measures reaction efficiency
    - (Actual product) / (Theoretical maximum product)
    - Critical for process optimization
    """

    def __init__(self, name: str = "yield", product_species: str = "",
                 reactant_species: str = "", stoichiometric_ratio: float = 1.0):
        super().__init__(name, KPIType.YIELD, KPILevel.UNIT)
        self.product_species = product_species
        self.reactant_species = reactant_species
        self.stoichiometric_ratio = stoichiometric_ratio

    def calculate(self, unit: UnitOperation, state: UnitState) -> KPIValue:
        reactant_in = 0.0
        product_out = 0.0

        for stream in state.input_streams.values():
            reactant_in += stream.composition.get(self.reactant_species, 0.0)

        for stream in state.output_streams.values():
            product_out += stream.composition.get(self.product_species, 0.0)

        theoretical_max = reactant_in * self.stoichiometric_ratio
        yield_value = product_out / theoretical_max if theoretical_max > 0 else 0.0
        yield_value = min(max(yield_value, 0.0), 1.0)

        return KPIValue(
            name=self.name,
            kpi_type=self.kpi_type,
            level=self.level,
            value=yield_value * 100,
            unit="%",
            timestamp=state.timestamp,
            source_id=unit.unit_id,
            metadata={
                "reactant_consumed": reactant_in,
                "product_produced": product_out,
                "theoretical_max": theoretical_max
            }
        )


class ConversionKPI(BaseKPI):
    """
    Conversion KPI - fraction of reactant converted.

    Design Intent:
    - Measures reactant consumption
    - (Reactant in - Reactant out) / Reactant in
    - Different from yield (focuses on reactant, not product)
    """

    def __init__(self, name: str = "conversion", reactant_species: str = ""):
        super().__init__(name, KPIType.CONVERSION, KPILevel.UNIT)
        self.reactant_species = reactant_species

    def calculate(self, unit: UnitOperation, state: UnitState) -> KPIValue:
        reactant_in = 0.0
        reactant_out = 0.0

        for stream in state.input_streams.values():
            reactant_in += stream.composition.get(self.reactant_species, 0.0)

        for stream in state.output_streams.values():
            reactant_out += stream.composition.get(self.reactant_species, 0.0)

        conversion = (reactant_in - reactant_out) / reactant_in if reactant_in > 0 else 0.0
        conversion = min(max(conversion, 0.0), 1.0)

        return KPIValue(
            name=self.name,
            kpi_type=self.kpi_type,
            level=self.level,
            value=conversion * 100,
            unit="%",
            timestamp=state.timestamp,
            source_id=unit.unit_id,
            metadata={
                "reactant_in": reactant_in,
                "reactant_out": reactant_out
            }
        )


class UtilizationKPI(BaseKPI):
    """
    Utilization KPI - fraction of equipment capacity used.

    Design Intent:
    - Measures equipment efficiency
    - (Actual throughput) / (Maximum capacity)
    - Critical for bottleneck identification
    """

    def __init__(self, name: str = "utilization", max_capacity: float = 1.0):
        super().__init__(name, KPIType.UTILIZATION, KPILevel.UNIT)
        self.max_capacity = max_capacity

    def calculate(self, unit: UnitOperation, state: UnitState) -> KPIValue:
        total_flow = 0.0
        for stream in state.output_streams.values():
            total_flow += stream.total_flow()

        utilization = total_flow / self.max_capacity if self.max_capacity > 0 else 0.0
        utilization = min(max(utilization, 0.0), 1.0)

        return KPIValue(
            name=self.name,
            kpi_type=self.kpi_type,
            level=self.level,
            value=utilization * 100,
            unit="%",
            timestamp=state.timestamp,
            source_id=unit.unit_id,
            metadata={
                "actual_throughput": total_flow,
                "max_capacity": self.max_capacity
            }
        )


class EnergyUsageKPI(BaseKPI):
    """
    Energy Usage KPI - power or heat consumption.
    
    Design Intent:
    - Measures operational energy intensity
    - Critical for environmental and economic assessment
    """

    def __init__(self, name: str = "energy_usage"):
        super().__init__(name, KPIType.ENERGY_USAGE, KPILevel.UNIT)

    def calculate(self, unit: UnitOperation, state: UnitState) -> KPIValue:
        power = state.internal_state.get("power_consumption", 0.0)
        duty = abs(state.internal_state.get("heat_duty", 0.0))
        total_energy = power + duty

        return KPIValue(
            name=self.name,
            kpi_type=self.kpi_type,
            level=self.level,
            value=total_energy,
            unit="W",
            timestamp=state.timestamp,
            source_id=unit.unit_id,
            metadata={"power": power, "heat_duty": duty}
        )


class EfficiencyKPI(BaseKPI):
    """
    Efficiency KPI - actual vs ideal performance.
    
    Design Intent:
    - Measures how close a unit is to its design intent
    - (Actual KPI) / (Design Target KPI)
    """

    def __init__(self, name: str = "efficiency", target_kpi: str = "yield", target_value: float = 100.0):
        super().__init__(name, KPIType.EFFICIENCY, KPILevel.UNIT)
        self.target_kpi = target_kpi
        self.target_value = target_value

    def calculate(self, unit: UnitOperation, state: UnitState, current_kpis: List[KPIValue]) -> KPIValue:
        actual_val = 0.0
        for kpi in current_kpis:
            if kpi.name == self.target_kpi:
                actual_val = kpi.value
                break
        
        efficiency = (actual_val / self.target_value) * 100 if self.target_value > 0 else 0.0
        
        return KPIValue(
            name=self.name,
            kpi_type=self.kpi_type,
            level=self.level,
            value=efficiency,
            unit="%",
            timestamp=state.timestamp,
            source_id=unit.unit_id,
            metadata={"target_kpi": self.target_kpi, "target_value": self.target_value}
        )


class StabilityKPI(BaseKPI):
    """
    Stability KPI - variance of process variables over time.
    
    Design Intent:
    - Measures how steady the process is
    - Low variance = High stability
    - Critical for quality control and safety
    """

    def __init__(self, name: str = "stability", variable_path: str = "temperature"):
        super().__init__(name, KPIType.STABILITY, KPILevel.UNIT)
        self.variable_path = variable_path

    def calculate(self, unit: UnitOperation, state: UnitState) -> KPIValue:
        history = unit.get_history()
        if len(history) < 2:
            return KPIValue(self.name, self.kpi_type, self.level, 100.0, "%", timestamp=state.timestamp)

        values = []
        for h_state in history:
            # Simple check for temperature in output streams
            for s in h_state.output_streams.values():
                if self.variable_path == "temperature" and s.temperature is not None:
                    values.append(s.temperature)
                elif self.variable_path == "pressure" and s.pressure is not None:
                    values.append(s.pressure)
        
        if not values:
            return KPIValue(self.name, self.kpi_type, self.level, 0.0, "%", timestamp=state.timestamp)
            
        std = np.std(values)
        mean = np.mean(values)
        cv = (std / mean) if mean != 0 else 0
        stability = max(0, 100 * (1.0 - cv)) # Simplified stability metric
        
        return KPIValue(
            name=self.name,
            kpi_type=self.kpi_type,
            level=self.level,
            value=stability,
            unit="%",
            timestamp=state.timestamp,
            source_id=unit.unit_id,
            metadata={"std": std, "mean": mean}
        )


class BottleneckDetector:
    """
    Bottleneck detection for process plants.

    Design Intent:
    - Identifies limiting units in the process
    - Based on utilization and throughput
    - Supports process optimization
    - No fake bottleneck scoring

    Methodology:
    - Unit with highest utilization is primary bottleneck
    - Units with >80% utilization are potential bottlenecks
    - Consider both current state and trends
    """

    def __init__(self, utilization_threshold: float = 0.8):
        self.utilization_threshold = utilization_threshold

    def identify_bottlenecks(self, unit_kpis: Dict[str, List[KPIValue]]) -> Dict[str, Any]:
        """
        Identify bottleneck units from KPI data.

        Returns:
            Dictionary with:
            - primary_bottleneck: Unit with highest utilization
            - potential_bottlenecks: Units above threshold
            - utilization_scores: All units' utilization
        """
        utilization_scores: Dict[str, float] = {}

        for unit_id, kpis in unit_kpis.items():
            for kpi in kpis:
                if kpi.kpi_type == KPIType.UTILIZATION:
                    utilization_scores[unit_id] = kpi.value / 100

        if not utilization_scores:
            return {
                "primary_bottleneck": None,
                "potential_bottlenecks": [],
                "utilization_scores": {}
            }

        sorted_units = sorted(
            utilization_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary_bottleneck = sorted_units[0][0] if sorted_units else None

        potential_bottlenecks = [
            unit_id for unit_id, score in utilization_scores.items()
            if score >= self.utilization_threshold
        ]

        return {
            "primary_bottleneck": primary_bottleneck,
            "potential_bottlenecks": potential_bottlenecks,
            "utilization_scores": utilization_scores
        }


class KPIAggregator:
    """
    KPI aggregator for plant-level metrics.

    Design Intent:
    - Aggregates unit-level KPIs to plant level
    - Maintains traceability
    - Supports multiple aggregation strategies
    - No silent aggregation assumptions
    """

    def __init__(self):
        self.unit_calculators: Dict[str, List[BaseKPI]] = {}
        self.bottleneck_detector = BottleneckDetector()

    def register_unit_calculator(self, unit_id: str, calculator: BaseKPI) -> None:
        """Register a KPI calculator for a specific unit."""
        if unit_id not in self.unit_calculators:
            self.unit_calculators[unit_id] = []
        self.unit_calculators[unit_id].append(calculator)

    def calculate_unit_kpis(self, unit: UnitOperation, state: UnitState) -> List[KPIValue]:
        """Calculate all KPIs for a single unit."""
        kpis = []

        if unit.unit_id in self.unit_calculators:
            for calculator in self.unit_calculators[unit.unit_id]:
                kpi = calculator.calculate(unit, state)
                kpis.append(kpi)

        return kpis

    def calculate_plant_kpis(self, plant_history: List[Dict[str, UnitState]],
                            units: Dict[str, UnitOperation]) -> ProcessKPIs:
        """
        Calculate KPIs for the entire plant.

        Design Intent:
        - Calculates unit-level KPIs first
        - Aggregates to plant level
        - Identifies bottlenecks
        - Maintains full traceability
        """
        if not plant_history:
            raise ValueError("No plant history available")

        latest_state = plant_history[-1]
        latest_timestamp = list(latest_state.values())[0].timestamp if latest_state else 0.0

        process_kpis = ProcessKPIs(timestamp=latest_timestamp)

        for unit_id, state in latest_state.items():
            unit = units.get(unit_id)
            if unit:
                unit_kpis = self.calculate_unit_kpis(unit, state)
                process_kpis.unit_kpis[unit_id] = unit_kpis

        bottleneck_info = self.bottleneck_detector.identify_bottlenecks(process_kpis.unit_kpis)

        total_throughput = 0.0
        for unit_kpis in process_kpis.unit_kpis.values():
            for kpi in unit_kpis:
                if kpi.kpi_type == KPIType.THROUGHPUT:
                    total_throughput += kpi.value

        plant_throughput = KPIValue(
            name="plant_throughput",
            kpi_type=KPIType.THROUGHPUT,
            level=KPILevel.PLANT,
            value=total_throughput,
            unit="mol/s",
            timestamp=latest_timestamp
        )
        process_kpis.plant_kpis.append(plant_throughput)

        if bottleneck_info["primary_bottleneck"]:
            bottleneck_kpi = KPIValue(
                name="primary_bottleneck",
                kpi_type=KPIType.BOTTLENECK_SCORE,
                level=KPILevel.PLANT,
                value=1.0,
                unit="",
                timestamp=latest_timestamp,
                metadata={
                    "bottleneck_unit": bottleneck_info["primary_bottleneck"],
                    "potential_bottlenecks": bottleneck_info["potential_bottlenecks"],
                    "utilization_scores": bottleneck_info["utilization_scores"]
                }
            )
            process_kpis.plant_kpis.append(bottleneck_kpi)

        return process_kpis


class RollingWindowKPI(BaseKPI):
    """
    KPI for computing rolling statistics on telemetry data.

    Design Intent:
    - Provides temporal observability into process variables
    - Supports rolling averages, variance, and min/max
    - Essential for detecting process drift and instability
    """

    def __init__(
        self,
        name: str,
        tag: str,
        window_size: int = 10,
        stat_type: str = "mean",
        unit: str = ""
    ):
        super().__init__(name, KPIType.ROLLING_STATISTIC, unit)
        self.tag = tag
        self.window_size = window_size
        self.stat_type = stat_type

    def calculate(self, history: List[Any], unit: "UnitOperation") -> KPIValue:
        # history is List[Dict[str, UnitState]]
        values = []
        for plant_state in history[-self.window_size:]:
            state = plant_state.get(unit.unit_id)
            if state and self.tag in state.internal_state:
                try:
                    values.append(float(state.internal_state[self.tag]))
                except (ValueError, TypeError):
                    continue

        if not values:
            return KPIValue(self.name, 0.0, self.unit, self.kpi_type)

        if self.stat_type == "mean":
            val = np.mean(values)
        elif self.stat_type == "std":
            val = np.std(values)
        elif self.stat_type == "min":
            val = np.min(values)
        elif self.stat_type == "max":
            val = np.max(values)
        else:
            val = values[-1]

        return KPIValue(
            self.name,
            float(val),
            self.unit,
            self.kpi_type,
            metadata={"window_size": self.window_size, "stat_type": self.stat_type, "tag": self.tag}
        )


class TemporalTrendKPI(BaseKPI):
    """
    KPI for computing the rate of change (trend) of a process variable.

    Design Intent:
    - Quantifies how fast a variable is evolving
    - Helps engineers anticipate future state crossings
    """

    def __init__(self, name: str, tag: str, unit: str = "/s"):
        super().__init__(name, KPIType.TEMPORAL_TREND, unit)
        self.tag = tag

    def calculate(self, history: List[Any], unit: "UnitOperation") -> KPIValue:
        if len(history) < 2:
            return KPIValue(self.name, 0.0, self.unit, self.kpi_type)

        s1 = history[-2].get(unit.unit_id)
        s2 = history[-1].get(unit.unit_id)

        if not s1 or not s2:
            return KPIValue(self.name, 0.0, self.unit, self.kpi_type)

        try:
            v1 = float(s1.internal_state.get(self.tag, 0))
            v2 = float(s2.internal_state.get(self.tag, 0))
            dt = s2.timestamp - s1.timestamp
            
            trend = (v2 - v1) / dt if dt > 0 else 0.0
            return KPIValue(self.name, trend, self.unit, self.kpi_type)
        except (ValueError, TypeError):
            return KPIValue(self.name, 0.0, self.unit, self.kpi_type)
