"""
Industrial Telemetry Ingestion Foundation.

This module provides:
- Generic telemetry ingestion abstractions
- Real-time data stream handling
- Timestamp normalization and synchronization
- Reproducible state snapshots
- Telemetry lineage tracking

Design Intent:
- Decouple framework from specific industrial protocols (MQTT, OPC-UA, etc.)
- Establish clear contracts for process-state ingestion
- Enable deterministic replay of historical telemetry
- Support scientifically reproducible operational observability
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Iterator
from enum import Enum
from abc import ABC, abstractmethod
import time
import json
from datetime import datetime

@dataclass
class TelemetryPacket:
    """
    Standardized telemetry data packet.

    Design Intent:
    - Normalizes data from diverse sources
    - Tracks provenance and timestamps
    - Supports arbitrary tag-value pairs
    """
    source_id: str
    timestamp: float  # Unix timestamp
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    sequence_id: Optional[int] = None

class TelemetrySource(ABC):
    """
    Abstract base class for telemetry data sources.

    Design Intent:
    - Defines the ingestion contract
    - Supports both streaming and batch sources
    - Encapsulates source-specific connection logic
    """

    def __init__(self, source_id: str):
        self.source_id = source_id
        self._is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the data source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass

    @abstractmethod
    def poll(self) -> List[TelemetryPacket]:
        """
        Poll for new telemetry packets.

        Returns:
            List of new TelemetryPacket instances
        """
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected

class CsvTelemetrySource(TelemetrySource):
    """
    CSV-based telemetry source for historical replay and batch ingestion.
    """

    def __init__(self, source_id: str, file_path: str, 
                 timestamp_col: str = "timestamp",
                 tag_cols: Optional[List[str]] = None):
        super().__init__(source_id)
        self.file_path = file_path
        self.timestamp_col = timestamp_col
        self.tag_cols = tag_cols
        self._data: List[Dict[str, Any]] = []
        self._cursor = 0

    def connect(self) -> bool:
        try:
            import csv
            with open(self.file_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self._data = [row for row in reader]
            self._is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to CSV source: {e}")
            return False

    def disconnect(self) -> None:
        self._data = []
        self._cursor = 0
        self._is_connected = False

    def poll(self) -> List[TelemetryPacket]:
        if not self._is_connected or self._cursor >= len(self._data):
            return []

        # For demonstration, we poll one row at a time
        row = self._data[self._cursor]
        self._cursor += 1

        try:
            ts = float(row[self.timestamp_col])
        except (ValueError, KeyError):
            ts = time.time()

        data = {k: v for k, v in row.items() if k != self.timestamp_col}
        if self.tag_cols:
            data = {k: v for k, v in data.items() if k in self.tag_cols}

        return [TelemetryPacket(
            source_id=self.source_id,
            timestamp=ts,
            data=data,
            sequence_id=self._cursor
        )]

class TelemetryIngestor:
    """
    Centralized telemetry ingestion manager.

    Design Intent:
    - Coordinates multiple telemetry sources
    - Manages timestamp normalization
    - Provides a unified stream of process state updates
    - Supports buffering and replay
    """

    def __init__(self):
        self.sources: Dict[str, TelemetrySource] = {}
        self._buffer: List[TelemetryPacket] = []
        self._max_buffer_size = 1000

    def add_source(self, source: TelemetrySource) -> None:
        """Register a new telemetry source."""
        self.sources[source.source_id] = source

    def start_ingestion(self) -> None:
        """Connect all registered sources."""
        for source in self.sources.values():
            source.connect()

    def stop_ingestion(self) -> None:
        """Disconnect all registered sources."""
        for source in self.sources.values():
            source.disconnect()

    def ingest_step(self) -> List[TelemetryPacket]:
        """
        Poll all sources and aggregate telemetry.
        """
        new_packets = []
        for source in self.sources.values():
            packets = source.poll()
            new_packets.extend(packets)

        self._buffer.extend(new_packets)
        if len(self._buffer) > self._max_buffer_size:
            self._buffer = self._buffer[-self._max_buffer_size:]

        return new_packets

    def get_latest_state(self) -> Dict[str, Any]:
        """
        Aggregate the latest values for all known tags.
        """
        state = {}
        # Sort by timestamp to ensure latest values win
        sorted_buffer = sorted(self._buffer, key=lambda x: x.timestamp)
        for packet in sorted_buffer:
            state.update(packet.data)
        return state

    def clear_buffer(self) -> None:
        self._buffer = []
