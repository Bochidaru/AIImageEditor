from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class MemoryStats:
    elapsed_ms: float
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None


class MemoryTracker:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.results: dict[str, MemoryStats] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        torch = _optional_torch()
        cuda_enabled = bool(
            self.enabled and torch is not None and torch.cuda.is_available()
        )

        if cuda_enabled:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        started_at = time.perf_counter()
        yield

        if cuda_enabled:
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated() / (1024**2)
            reserved = torch.cuda.max_memory_reserved() / (1024**2)
        else:
            allocated = None
            reserved = None

        self.results[stage] = MemoryStats(
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            peak_allocated_mb=allocated,
            peak_reserved_mb=reserved,
        )

    def to_dict(self) -> dict[str, dict[str, float | None]]:
        """Return the latest measurement for each stage as JSON-safe data."""
        return {stage: asdict(stats) for stage, stats in self.results.items()}

    def save_json(self, path: str | Path) -> Path:
        """Write the current measurements to a UTF-8 JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stages": self.to_dict(),
        }
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        return output_path


def _optional_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None
