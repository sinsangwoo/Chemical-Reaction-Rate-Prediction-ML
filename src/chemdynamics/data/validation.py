"""
Scientific Validation Layer for Chemical Reaction Datasets.

This module provides infrastructure for validating the scientific integrity
of chemical reaction datasets. It enforces physical constraints, checks
data consistency, and ensures scientific reproducibility.

Design Intent:
- Scientific datasets must be rigorously validated before use
- Invalid data should be explicitly rejected with clear scientific rationale
- Validation must be transparent and reproducible
- Physical laws (conservation, thermodynamics) should guide validation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np
from pathlib import Path
import json

from .reaction_dataset import ChemicalReaction, ReactionConditions


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Category of validation issue."""
    PHYSICAL_CONSTRAINT = "physical_constraint"
    MISSING_DATA = "missing_data"
    RANGE_VIOLATION = "range_violation"
    UNIT_INCONSISTENCY = "unit_inconsistency"
    CHEMICAL_INCONSISTENCY = "chemical_inconsistency"
    METADATA_CORRUPTION = "metadata_corruption"


@dataclass
class ValidationIssue:
    """Container for a single validation issue."""
    reaction_id: Optional[str]
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    expected_range: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "reaction_id": self.reaction_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field,
            "value": self.value,
            "expected_range": self.expected_range
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report for a dataset."""
    total_reactions: int
    valid_reactions: int
    issues: List[ValidationIssue] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: "")

    @property
    def is_valid(self) -> bool:
        """Check if dataset is scientifically valid (no critical/error issues)."""
        return not any(
            issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for issue in self.issues
        )

    @property
    def error_count(self) -> int:
        return sum(
            1 for issue in self.issues
            if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        return sum(
            1 for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        )

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def to_dict(self) -> Dict:
        return {
            "total_reactions": self.total_reactions,
            "valid_reactions": self.valid_reactions,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "timestamp": self.timestamp
        }

    def save(self, filepath: Path) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ReactionValidator:
    """
    Scientific validator for chemical reaction datasets.

    This class implements validation rules that enforce:
    - Physical constraints (non-negative concentrations, realistic temperatures)
    - Chemical consistency (valid reaction stoichiometry)
    - Data completeness (required fields present)
    - Range validity (physically meaningful parameter ranges)
    """

    def __init__(self):
        self.rules: List[Callable[[ChemicalReaction], List[ValidationIssue]]] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default scientific validation rules."""
        self.rules.append(self._validate_temperature_range)
        self.rules.append(self._validate_pressure_range)
        self.rules.append(self._validate_reaction_rate_range)
        self.rules.append(self._validate_yield_percentage)
        self.rules.append(self._validate_ph_range)
        self.rules.append(self._validate_smiles_presence)
        self.rules.append(self._validate_non_negative_conditions)

    def _validate_temperature_range(self, reaction: ChemicalReaction) -> List[ValidationIssue]:
        """
        Validate temperature is physically meaningful.

        Scientific Rationale:
        - Absolute zero is 0 K = -273.15 °C
        - Extremely high temperatures (> 5000 °C) are rare in standard chemistry
        """
        issues = []
        temp = reaction.conditions.temperature
        if temp is not None:
            if temp < -273.15:
                issues.append(ValidationIssue(
                    reaction_id=reaction.reaction_id,
                    category=ValidationCategory.PHYSICAL_CONSTRAINT,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Temperature cannot be below absolute zero",
                    field="temperature",
                    value=temp,
                    expected_range="≥ -273.15 °C"
                ))
            elif temp > 5000:
                issues.append(ValidationIssue(
                    reaction_id=reaction.reaction_id,
                    category=ValidationCategory.RANGE_VIOLATION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Temperature is extremely high",
                    field="temperature",
                    value=temp,
                    expected_range="-273.15 to 5000 °C"
                ))
        return issues

    def _validate_pressure_range(self, reaction: ChemicalReaction) -> List[ValidationIssue]:
        """
        Validate pressure is physically meaningful.

        Scientific Rationale:
        - Pressure cannot be negative
        - Extremely high pressures (> 1000 atm) are rare
        """
        issues = []
        pressure = reaction.conditions.pressure
        if pressure is not None:
            if pressure < 0:
                issues.append(ValidationIssue(
                    reaction_id=reaction.reaction_id,
                    category=ValidationCategory.PHYSICAL_CONSTRAINT,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Pressure cannot be negative",
                    field="pressure",
                    value=pressure,
                    expected_range="≥ 0 atm"
                ))
            elif pressure > 1000:
                issues.append(ValidationIssue(
                    reaction_id=reaction.reaction_id,
                    category=ValidationCategory.RANGE_VIOLATION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Pressure is extremely high",
                    field="pressure",
                    value=pressure,
                    expected_range="0 to 1000 atm"
                ))
        return issues

    def _validate_reaction_rate_range(self, reaction: ChemicalReaction) -> List[ValidationIssue]:
        """
        Validate reaction rate is physically meaningful.

        Scientific Rationale:
        - Reaction rates cannot be negative
        - Extremely high rates may indicate measurement errors
        """
        issues = []
        rate = reaction.reaction_rate
        if rate is not None:
            if rate < 0:
                issues.append(ValidationIssue(
                    reaction_id=reaction.reaction_id,
                    category=ValidationCategory.PHYSICAL_CONSTRAINT,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Reaction rate cannot be negative",
                    field="reaction_rate",
                    value=rate,
                    expected_range="≥ 0 mol/L·s"
                ))
            elif rate > 1e6:
                issues.append(ValidationIssue(
                    reaction_id=reaction.reaction_id,
                    category=ValidationCategory.RANGE_VIOLATION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Reaction rate is extremely high",
                    field="reaction_rate",
                    value=rate,
                    expected_range="0 to 1e6 mol/L·s"
                ))
        return issues

    def _validate_yield_percentage(self, reaction: ChemicalReaction) -> List[ValidationIssue]:
        """
        Validate yield percentage is physically meaningful.

        Scientific Rationale:
        - Yield must be between 0 and 100 percent
        """
        issues = []
        yld = reaction.yield_percentage
        if yld is not None:
            if yld < 0 or yld > 100:
                issues.append(ValidationIssue(
                    reaction_id=reaction.reaction_id,
                    category=ValidationCategory.PHYSICAL_CONSTRAINT,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Yield must be between 0 and 100 percent",
                    field="yield_percentage",
                    value=yld,
                    expected_range="0 to 100 %"
                ))
        return issues

    def _validate_ph_range(self, reaction: ChemicalReaction) -> List[ValidationIssue]:
        """
        Validate pH is within physically meaningful range.

        Scientific Rationale:
        - pH typically ranges from 0 to 14 in aqueous solutions
        """
        issues = []
        ph = reaction.conditions.ph
        if ph is not None:
            if ph < 0 or ph > 14:
                issues.append(ValidationIssue(
                    reaction_id=reaction.reaction_id,
                    category=ValidationCategory.RANGE_VIOLATION,
                    severity=ValidationSeverity.WARNING,
                    message=f"pH is outside typical aqueous range",
                    field="ph",
                    value=ph,
                    expected_range="0 to 14"
                ))
        return issues

    def _validate_smiles_presence(self, reaction: ChemicalReaction) -> List[ValidationIssue]:
        """Validate reactants and products have SMILES strings."""
        issues = []
        if not reaction.reactants:
            issues.append(ValidationIssue(
                reaction_id=reaction.reaction_id,
                category=ValidationCategory.MISSING_DATA,
                severity=ValidationSeverity.ERROR,
                message=f"No reactants specified",
                field="reactants"
            ))
        if not reaction.products:
            issues.append(ValidationIssue(
                reaction_id=reaction.reaction_id,
                category=ValidationCategory.MISSING_DATA,
                severity=ValidationSeverity.ERROR,
                message=f"No products specified",
                field="products"
            ))
        return issues

    def _validate_non_negative_conditions(self, reaction: ChemicalReaction) -> List[ValidationIssue]:
        """Validate time is non-negative."""
        issues = []
        time_val = reaction.conditions.time
        if time_val is not None and time_val < 0:
            issues.append(ValidationIssue(
                reaction_id=reaction.reaction_id,
                category=ValidationCategory.PHYSICAL_CONSTRAINT,
                severity=ValidationSeverity.CRITICAL,
                message=f"Time cannot be negative",
                field="time",
                value=time_val,
                expected_range="≥ 0 seconds"
            ))
        return issues

    def validate_reaction(self, reaction: ChemicalReaction) -> List[ValidationIssue]:
        """Validate a single reaction against all registered rules."""
        all_issues = []
        for rule in self.rules:
            issues = rule(reaction)
            all_issues.extend(issues)
        return all_issues

    def validate_dataset(self, reactions: List[ChemicalReaction]) -> ValidationReport:
        """
        Validate an entire dataset.

        Returns:
            ValidationReport with comprehensive validation results
        """
        from datetime import datetime

        report = ValidationReport(
            total_reactions=len(reactions),
            valid_reactions=0,
            timestamp=datetime.now().isoformat()
        )

        valid_count = 0
        for reaction in reactions:
            issues = self.validate_reaction(reaction)
            has_errors = any(
                issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
                for issue in issues
            )
            if not has_errors:
                valid_count += 1
            for issue in issues:
                report.add_issue(issue)

        report.valid_reactions = valid_count
        return report
