"""
Explicit Preprocessing DAG Semantics.

This module provides:
- Explicit preprocessing DAG (Directed Acyclic Graph)
- Reproducible transformation ordering
- Preprocessing artifact traceability
- Cache-safe preprocessing stages
- Deterministic preprocessing execution

Design Intent:
- Preprocessing must be explicit and reproducible
- Transformation order matters for scientific validity
- Artifacts must be traceable to their source transformations
- Caching must be safe and version-aware
- DAG structure enables dependency-aware execution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from pathlib import Path
import hashlib
import json
from datetime import datetime
import inspect

from .reaction_dataset import ChemicalReaction, ReactionDataset


class PreprocessingStageType(Enum):
    """Type of preprocessing stage."""
    FILTER = "filter"
    TRANSFORM = "transform"
    FEATURE_ENGINEERING = "feature_engineering"
    VALIDATION = "validation"
    NORMALIZATION = "normalization"


@dataclass
class PreprocessingArtifact:
    """Artifact produced by a preprocessing stage."""
    name: str
    stage_name: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class StageExecutionLog:
    """Log of a preprocessing stage execution."""
    stage_name: str
    stage_type: PreprocessingStageType
    start_time: str
    end_time: str
    duration_seconds: float
    input_hash: str
    output_hash: str
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreprocessingStage:
    """
    A single stage in the preprocessing DAG.

    Design Intent:
    - Each stage is a self-contained transformation
    - Stages declare their inputs and outputs
    - Deterministic execution with caching
    - Versioned for reproducibility
    """

    def __init__(
        self,
        name: str,
        stage_type: PreprocessingStageType,
        transform: Callable[[List[ChemicalReaction]], List[ChemicalReaction]],
        version: str = "1.0.0",
        description: str = "",
        dependencies: Optional[List[str]] = None
    ):
        self.name = name
        self.stage_type = stage_type
        self.transform = transform
        self.version = version
        self.description = description
        self.dependencies = dependencies or []
        self.artifacts: List[PreprocessingArtifact] = []

    def __call__(self, reactions: List[ChemicalReaction]) -> List[ChemicalReaction]:
        """Execute the preprocessing stage."""
        return self.transform(reactions)

    def get_hash(self) -> str:
        """Get a hash representing this stage's identity."""
        source = inspect.getsource(self.transform) if hasattr(self.transform, "__code__") else str(self.transform)
        content = f"{self.name}:{self.version}:{source}"
        return hashlib.sha256(content.encode()).hexdigest()


class PreprocessingDAG:
    """
    Directed Acyclic Graph for preprocessing pipelines.

    Design Intent:
    - Explicit dependency declaration
    - Topological sorting for execution order
    - Caching of intermediate results
    - Full provenance tracking
    - Reproducible execution
    """

    def __init__(self, name: str = "preprocessing_pipeline"):
        self.name = name
        self.stages: Dict[str, PreprocessingStage] = {}
        self.execution_logs: List[StageExecutionLog] = []
        self._cache: Dict[str, Any] = {}

    def add_stage(self, stage: PreprocessingStage) -> None:
        """
        Add a stage to the DAG.

        Args:
            stage: Preprocessing stage to add
        """
        self.stages[stage.name] = stage

    def _topological_sort(self) -> List[str]:
        """
        Perform topological sort to determine execution order.

        Returns:
            Ordered list of stage names
        """
        visited: Set[str] = set()
        temp: Set[str] = set()
        order: List[str] = []

        def visit(name: str) -> None:
            if name in temp:
                raise ValueError(f"Circular dependency detected involving stage '{name}'")
            if name not in visited:
                temp.add(name)
                stage = self.stages[name]
                for dep in stage.dependencies:
                    if dep in self.stages:
                        visit(dep)
                temp.remove(name)
                visited.add(name)
                order.insert(0, name)

        for stage_name in self.stages:
            if stage_name not in visited:
                visit(stage_name)

        return order

    def _hash_reactions(self, reactions: List[ChemicalReaction]) -> str:
        """Generate a hash for a list of reactions."""
        content = json.dumps([r.to_dict() for r in reactions], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def execute(
        self,
        reactions: List[ChemicalReaction],
        use_cache: bool = True
    ) -> List[ChemicalReaction]:
        """
        Execute the preprocessing DAG.

        Args:
            reactions: Input reactions to process
            use_cache: Whether to use cached results

        Returns:
            Processed reactions
        """
        self.execution_logs = []
        order = self._topological_sort()

        current_reactions = reactions.copy()
        stage_outputs: Dict[str, List[ChemicalReaction]] = {}

        for stage_name in order:
            stage = self.stages[stage_name]

            input_hash = self._hash_reactions(current_reactions)
            cache_key = f"{stage.name}:{stage.version}:{input_hash}"

            start_time = datetime.now()
            success = False
            error_message = None
            output_reactions = current_reactions

            try:
                if use_cache and cache_key in self._cache:
                    output_reactions = self._cache[cache_key]
                else:
                    if stage.dependencies:
                        for dep in stage.dependencies:
                            if dep in stage_outputs:
                                pass

                    output_reactions = stage(current_reactions)
                    self._cache[cache_key] = output_reactions

                success = True
            except Exception as e:
                error_message = str(e)
                raise

            finally:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                output_hash = self._hash_reactions(output_reactions) if success else ""

                log = StageExecutionLog(
                    stage_name=stage.name,
                    stage_type=stage.stage_type,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_seconds=duration,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    success=success,
                    error_message=error_message
                )
                self.execution_logs.append(log)

            stage_outputs[stage.name] = output_reactions
            current_reactions = output_reactions

        return current_reactions

    def get_provenance_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution provenance."""
        return {
            "pipeline_name": self.name,
            "num_stages": len(self.stages),
            "execution_logs": [
                {
                    "stage_name": log.stage_name,
                    "stage_type": log.stage_type.value,
                    "duration_seconds": log.duration_seconds,
                    "success": log.success
                }
                for log in self.execution_logs
            ]
        }

    def clear_cache(self) -> None:
        """Clear the execution cache."""
        self._cache.clear()


def create_basic_preprocessing_pipeline() -> PreprocessingDAG:
    """
    Create a basic preprocessing pipeline with common stages.

    Returns:
        PreprocessingDAG with basic stages
    """
    dag = PreprocessingDAG("basic_chemistry_pipeline")

    def filter_invalid_temperatures(reactions: List[ChemicalReaction]) -> List[ChemicalReaction]:
        return [
            r for r in reactions
            if r.conditions.temperature is None or r.conditions.temperature >= -273.15
        ]

    def filter_missing_rates(reactions: List[ChemicalReaction]) -> List[ChemicalReaction]:
        return [r for r in reactions if r.reaction_rate is not None]

    stage1 = PreprocessingStage(
        name="filter_invalid_temperatures",
        stage_type=PreprocessingStageType.FILTER,
        transform=filter_invalid_temperatures,
        version="1.0.0",
        description="Filter reactions with temperatures below absolute zero"
    )

    stage2 = PreprocessingStage(
        name="filter_missing_rates",
        stage_type=PreprocessingStageType.FILTER,
        transform=filter_missing_rates,
        version="1.0.0",
        description="Filter reactions without reaction rate data",
        dependencies=["filter_invalid_temperatures"]
    )

    dag.add_stage(stage1)
    dag.add_stage(stage2)

    return dag
