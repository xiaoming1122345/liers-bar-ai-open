from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import SolverConfig
from .types import TrainingBatchReport


@dataclass(frozen=True)
class SolverArtifact:
    artifact_id: str
    solver_config: SolverConfig
    created_at: str
    metrics: TrainingBatchReport | None = None
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        artifact_id: str,
        solver_config: SolverConfig,
        metrics: TrainingBatchReport | None = None,
        metadata: dict[str, str | int | float] | None = None,
    ) -> "SolverArtifact":
        return cls(
            artifact_id=artifact_id,
            solver_config=solver_config,
            created_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "solver_config": asdict(self.solver_config),
            "created_at": self.created_at,
            "metrics": asdict(self.metrics) if self.metrics is not None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SolverArtifact":
        metrics_payload = payload.get("metrics")
        seat_mean_terminal_utility = {}
        seat_sample_count = {}
        if metrics_payload is not None:
            seat_mean_terminal_utility = {
                int(seat): float(value)
                for seat, value in metrics_payload.get("seat_mean_terminal_utility", {}).items()
            }
            seat_sample_count = {
                int(seat): int(value)
                for seat, value in metrics_payload.get("seat_sample_count", {}).items()
            }

        return cls(
            artifact_id=payload["artifact_id"],
            solver_config=SolverConfig(**payload["solver_config"]),
            created_at=payload["created_at"],
            metrics=(
                TrainingBatchReport(
                    sample_count=int(metrics_payload["sample_count"]),
                    mean_modeling_degree=float(metrics_payload["mean_modeling_degree"]),
                    mean_calibration_score=float(metrics_payload["mean_calibration_score"]),
                    mean_confidence_gap=float(metrics_payload["mean_confidence_gap"]),
                    mean_ghost_location_probability=float(metrics_payload["mean_ghost_location_probability"]),
                    mean_ghost_rank_probability=float(metrics_payload["mean_ghost_rank_probability"]),
                    mean_support_alignment=float(metrics_payload["mean_support_alignment"]),
                    mean_support_error=float(metrics_payload["mean_support_error"]),
                    mean_terminal_utility=float(metrics_payload["mean_terminal_utility"]),
                    seat_mean_terminal_utility=seat_mean_terminal_utility,
                    seat_sample_count=seat_sample_count,
                )
                if metrics_payload is not None
                else None
            ),
            metadata=payload.get("metadata", {}),
        )


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, artifact_id: str) -> Path:
        safe_name = artifact_id.replace(":", "_")
        path = self.root / safe_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_metadata(self, artifact: SolverArtifact) -> Path:
        path = self.run_dir(artifact.artifact_id) / "artifact.json"
        path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")
        return path

    def load_metadata(self, artifact_id: str) -> SolverArtifact:
        path = self.run_dir(artifact_id) / "artifact.json"
        return SolverArtifact.from_dict(json.loads(path.read_text(encoding="utf-8")))
