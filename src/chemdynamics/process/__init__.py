"""
Industrial Process Foundation Module.

This module provides:
- Process network modeling (unit operations, process graph)
- Industrial constraint system
- KPI calculation and aggregation
- Engineering workflow infrastructure

Design Intent:
- Process-centric, not reaction-centric architecture
- Extensible foundations for industrial simulation
- Engineer-oriented workflows and abstractions
"""

from .network import (
    ProcessGraph,
    UnitOperation,
    UnitType,
    UnitState,
    Stream,
    Reactor,
    Mixer,
    Splitter,
    Separator,
    HeatExchanger,
    Pump,
    Storage,
    ProcessNode
)
from .telemetry import (
    TelemetryIngestor,
    TelemetrySource,
    CsvTelemetrySource,
    TelemetryPacket
)
from .constraints import (
    ProcessConstraint,
    ConstraintType,
    ConstraintSeverity,
    ConstraintViolation,
    TemperatureLimit,
    PressureLimit,
    CapacityLimit,
    CompositionLimit,
    EnergyLimit,
    ThroughputLimit,
    RampRateLimit,
    SafetyLimit,
    TelemetryValidityConstraint,
    TemporalConsistencyConstraint,
    ConstraintValidator
)
from .kpis import (
    KPIAggregator,
    KPIValue,
    ProcessKPIs,
    KPILevel,
    KPIType,
    BaseKPI,
    ThroughputKPI,
    YieldKPI,
    ConversionKPI,
    UtilizationKPI,
    EnergyUsageKPI,
    EfficiencyKPI,
    StabilityKPI,
    RollingWindowKPI,
    TemporalTrendKPI,
    BottleneckDetector
)
from .workflows import (
    IndustrialWorkflowManager,
    WorkflowResult,
    UncertaintySweepResult
)

__all__ = [
    "ProcessGraph",
    "UnitOperation",
    "UnitType",
    "UnitState",
    "Stream",
    "Reactor",
    "Mixer",
    "Splitter",
    "ProcessConstraint",
    "ConstraintType",
    "ConstraintSeverity",
    "ConstraintViolation",
    "TemperatureLimit",
    "PressureLimit",
    "CapacityLimit",
    "CompositionLimit",
    "ConstraintValidator",
    "KPIAggregator",
    "KPIValue",
    "ProcessKPIs",
    "KPILevel",
    "KPIType",
    "BaseKPI",
    "ThroughputKPI",
    "YieldKPI",
    "ConversionKPI",
    "UtilizationKPI",
    "EnergyUsageKPI",
    "EfficiencyKPI",
    "StabilityKPI",
    "BottleneckDetector",
    "Separator",
    "HeatExchanger",
    "Pump",
    "Storage",
    "EnergyLimit",
    "ThroughputLimit",
    "RampRateLimit",
    "SafetyLimit",
    "TelemetryValidityConstraint",
    "TemporalConsistencyConstraint",
    "IndustrialWorkflowManager",
    "WorkflowResult",
    "UncertaintySweepResult",
    "TelemetryIngestor",
    "TelemetrySource",
    "CsvTelemetrySource",
    "TelemetryPacket",
    "ProcessNode",
    "RollingWindowKPI",
    "TemporalTrendKPI"
]
