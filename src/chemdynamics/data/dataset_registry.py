"""
Benchmark Dataset Infrastructure & Schema Evolution Support.

This module provides:
- Benchmark dataset registry
- Versioned dataset loading
- Dataset metadata tracking
- Schema evolution with backward compatibility
- Experiment-to-dataset traceability

Design Intent:
- Scientific datasets must be versioned and traceable
- Schema changes must be handled safely with backward compatibility
- Dataset provenance must be preserved for reproducibility
- Benchmark comparisons require standardized dataset interfaces
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path
import json
from datetime import datetime
import semver


class DatasetType(Enum):
    """Type of chemical reaction dataset."""
    BENCHMARK = "benchmark"
    EXPERIMENTAL = "experimental"
    SYNTHETIC = "synthetic"
    CURATED = "curated"


@dataclass
class DatasetMetadata:
    """Comprehensive metadata for a dataset."""
    name: str
    version: str
    dataset_type: DatasetType
    description: str
    schema_version: str
    num_reactions: int
    created_at: str
    source: Optional[str] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    preprocessing_steps: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "dataset_type": self.dataset_type.value,
            "description": self.description,
            "schema_version": self.schema_version,
            "num_reactions": self.num_reactions,
            "created_at": self.created_at,
            "source": self.source,
            "citation": self.citation,
            "license": self.license,
            "tags": self.tags,
            "preprocessing_steps": self.preprocessing_steps,
            "statistics": self.statistics
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetMetadata":
        return cls(
            name=data["name"],
            version=data["version"],
            dataset_type=DatasetType(data["dataset_type"]),
            description=data["description"],
            schema_version=data["schema_version"],
            num_reactions=data["num_reactions"],
            created_at=data["created_at"],
            source=data.get("source"),
            citation=data.get("citation"),
            license=data.get("license"),
            tags=data.get("tags", []),
            preprocessing_steps=data.get("preprocessing_steps", []),
            statistics=data.get("statistics", {})
        )


@dataclass
class SchemaMigration:
    """Represents a schema migration between versions."""
    from_version: str
    to_version: str
    migration_function: Callable[[Dict], Dict]
    description: str


class SchemaEvolutionManager:
    """
    Manages dataset schema evolution with backward compatibility.

    Design Intent:
    - Scientific datasets evolve over time
    - Schema changes must not break existing code
    - Migrations must be explicit and reversible
    - Versioned schemas enable reproducibility
    """

    def __init__(self):
        self.migrations: List[SchemaMigration] = []
        self.current_schema_version = "1.0.0"
        self._register_default_migrations()

    def _register_default_migrations(self) -> None:
        """Register default schema migrations."""
        pass

    def register_migration(self, migration: SchemaMigration) -> None:
        """Register a new schema migration."""
        self.migrations.append(migration)

    def migrate_data(
        self,
        data: Dict,
        from_version: str,
        to_version: Optional[str] = None
    ) -> Dict:
        """
        Migrate data from one schema version to another.

        Args:
            data: Data to migrate
            from_version: Current schema version of data
            to_version: Target schema version (defaults to current)

        Returns:
            Migrated data
        """
        target_version = to_version if to_version else self.current_schema_version

        if from_version == target_version:
            return data.copy()

        if semver.compare(from_version, target_version) > 0:
            raise ValueError(f"Cannot migrate from newer version {from_version} to older {target_version}")

        result = data.copy()

        for migration in self.migrations:
            if (semver.compare(migration.from_version, from_version) >= 0 and
                semver.compare(migration.to_version, target_version) <= 0):
                result = migration.migration_function(result)

        return result


class DatasetRegistry:
    """
    Registry for managing versioned benchmark datasets.

    Design Intent:
    - Centralized dataset management
    - Versioned access to datasets
    - Reproducible experiment-to-dataset linking
    - Metadata-driven dataset discovery
    """

    def __init__(self, registry_dir: Optional[Path] = None):
        self.registry_dir = registry_dir or Path("data/registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, Dict[str, DatasetMetadata]] = {}
        self.schema_manager = SchemaEvolutionManager()
        self._load_registry()

    def _get_registry_path(self) -> Path:
        return self.registry_dir / "registry.json"

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_path = self._get_registry_path()
        if registry_path.exists():
            with open(registry_path, "r") as f:
                data = json.load(f)
                for name, versions in data.items():
                    self.datasets[name] = {
                        v: DatasetMetadata.from_dict(m)
                        for v, m in versions.items()
                    }

    def _save_registry(self) -> None:
        """Save registry to disk."""
        data = {}
        for name, versions in self.datasets.items():
            data[name] = {v: m.to_dict() for v, m in versions.items()}
        with open(self._get_registry_path(), "w") as f:
            json.dump(data, f, indent=2)

    def register_dataset(
        self,
        metadata: DatasetMetadata,
        data_path: Optional[Path] = None
    ) -> None:
        """
        Register a new dataset version.

        Args:
            metadata: Dataset metadata
            data_path: Optional path to dataset file (will be copied)
        """
        if metadata.name not in self.datasets:
            self.datasets[metadata.name] = {}

        if metadata.version in self.datasets[metadata.name]:
            raise ValueError(f"Dataset {metadata.name} version {metadata.version} already exists")

        self.datasets[metadata.name][metadata.version] = metadata

        if data_path and data_path.exists():
            dataset_dir = self.registry_dir / metadata.name / metadata.version
            dataset_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(data_path, dataset_dir / data_path.name)

        self._save_registry()

    def get_dataset(
        self,
        name: str,
        version: Optional[str] = None
    ) -> DatasetMetadata:
        """
        Get dataset metadata.

        Args:
            name: Dataset name
            version: Specific version (latest if None)

        Returns:
            DatasetMetadata
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found in registry")

        versions = self.datasets[name]

        if version is None:
            version = sorted(versions.keys(), key=semver.Version.parse, reverse=True)[0]

        if version not in versions:
            raise ValueError(f"Version {version} not found for dataset {name}")

        return versions[version]

    def list_datasets(self) -> List[str]:
        """List all registered dataset names."""
        return list(self.datasets.keys())

    def list_versions(self, name: str) -> List[str]:
        """List all versions for a dataset."""
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found in registry")
        return sorted(self.datasets[name].keys(), key=semver.Version.parse, reverse=True)

    def get_dataset_path(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[Path]:
        """Get path to dataset file if available."""
        metadata = self.get_dataset(name, version)
        dataset_dir = self.registry_dir / name / metadata.version
        if dataset_dir.exists():
            files = list(dataset_dir.glob("*"))
            if files:
                return files[0]
        return None
