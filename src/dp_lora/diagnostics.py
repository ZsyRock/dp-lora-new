"""Access-controlled, explicitly non-DP gradient diagnostics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from dp_lora.config import GradientObservationConfig

logger = logging.getLogger(__name__)

_NON_DP_LABEL = "NON_DP_PRIVATE_DIAGNOSTIC"


def _tensor_summary(
    values: torch.Tensor, quantile_values: tuple[float, ...]
) -> dict[str, Any]:
    """Return exact descriptive statistics for one one-dimensional tensor."""
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "quantiles": {},
        }
    levels = torch.tensor(quantile_values, dtype=torch.float64)
    quantiles = torch.quantile(values, levels)
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "quantiles": {
            str(float(level)): float(value)
            for level, value in zip(levels.tolist(), quantiles.tolist())
        },
    }


class TrustedGradientObserver:
    """Accumulates exact norms and writes one JSONL record per logical step.

    This class is deliberately local-file-only. It has no W&B or remote logger
    integration, so exact statistics cannot be uploaded accidentally by this
    package. The containing output directory is ignored by Git.
    """

    def __init__(self, config: GradientObservationConfig):
        if not config.enabled or not config.acknowledge_non_dp:
            raise ValueError("TrustedGradientObserver requires explicit non-DP consent")
        self.config = config
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.output_dir / "gradient_observations.jsonl"
        self.training_steps_path = self.output_dir / "training_steps.jsonl"
        self.epoch_summaries_path = self.output_dir / "epoch_summaries.jsonl"
        self.warning_path = self.output_dir / "NON_DP_PRIVATE_DATA.txt"
        self.warning_path.write_text(
            "These files contain exact, data-dependent training diagnostics.\n"
            "They are NOT covered by the model's (epsilon, delta)-DP guarantee.\n"
            "Keep them access-controlled; do not upload, commit, or publish them.\n",
            encoding="utf-8",
        )
        self._norm_chunks: list[torch.Tensor] = []
        self._parameter_norm_chunks: dict[str, list[torch.Tensor]] = {}
        logger.warning(
            "Exact gradient observation is ENABLED. Output %s is not DP.",
            self.output_dir,
        )

    def observe_microbatch(
        self,
        per_sample_norms: torch.Tensor,
        per_parameter_norms: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self._norm_chunks.append(
            per_sample_norms.detach().to(device="cpu", dtype=torch.float64)
        )
        if self.config.parameter_statistics and per_parameter_norms:
            for name, norms in per_parameter_norms.items():
                self._parameter_norm_chunks.setdefault(name, []).append(
                    norms.detach().to(device="cpu", dtype=torch.float64)
                )

    def reset(self) -> None:
        self._norm_chunks.clear()
        self._parameter_norm_chunks.clear()

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def record_training_step(self, record: dict[str, Any]) -> None:
        """Write exact loss/runtime context aligned to one logical DP step."""
        payload = {
            "privacy_status": _NON_DP_LABEL,
            "record_type": "logical_training_step",
            **record,
        }
        self._append_jsonl(self.training_steps_path, payload)

    def record_epoch_summary(self, record: dict[str, Any]) -> None:
        """Write exact private training summaries separately from DP outputs."""
        payload = {
            "privacy_status": _NON_DP_LABEL,
            "record_type": "epoch_summary",
            **record,
        }
        self._append_jsonl(self.epoch_summaries_path, payload)

    def finalize_step(
        self,
        *,
        logical_step: int,
        clipping_mode: str,
        clip_before: float,
        clip_after: float,
        private_aux: dict[str, Any] | None = None,
        optimizer_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._norm_chunks:
            norms = torch.cat(self._norm_chunks)
        else:
            norms = torch.empty(0, dtype=torch.float64)

        bins = self.config.histogram_bins
        edges = torch.linspace(0.0, float(clip_before), bins + 1, dtype=torch.float64)
        if norms.numel():
            counts = torch.histogram(
                norms.clamp(max=float(clip_before)), bins=edges
            ).hist
            overflow = int((norms > float(clip_before)).sum().item())
            # The final in-range bin currently contains overflow after clamping.
            counts[-1] -= overflow
            mean_norm = float(norms.mean().item())
            clipped_fraction = float(overflow / norms.numel())
        else:
            counts = torch.zeros(bins, dtype=torch.float64)
            overflow = 0
            mean_norm = 0.0
            clipped_fraction = 0.0

        norm_summary = _tensor_summary(
            norms, (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        )
        parameter_summaries = {
            name: _tensor_summary(torch.cat(chunks), (0.5, 0.9, 0.99))
            for name, chunks in sorted(self._parameter_norm_chunks.items())
            if chunks
        }

        record: dict[str, Any] = {
            "privacy_status": _NON_DP_LABEL,
            "logical_step": int(logical_step),
            "clipping_mode": clipping_mode,
            "sample_count": int(norms.numel()),
            "clip_before": float(clip_before),
            "clip_after": float(clip_after),
            "mean_norm": mean_norm,
            "norm_statistics": norm_summary,
            "clipped_count": overflow,
            "clipped_fraction": clipped_fraction,
            "histogram": {
                "definition": "exact norm bins on [0, clip_before] plus overflow",
                "edges": edges.tolist(),
                "counts": [int(v) for v in counts.tolist()],
                "overflow_count": overflow,
            },
            "quantiles": norm_summary["quantiles"],
        }
        if parameter_summaries:
            record["parameter_norm_statistics"] = parameter_summaries
        if private_aux:
            # These are already-DP auxiliary values for SlaClip, but the record
            # remains labelled non-DP because it also contains exact norms.
            record["private_optimizer_aux"] = private_aux
        if optimizer_stats:
            record["optimizer_statistics"] = optimizer_stats
        if self.config.store_per_sample_norms:
            record["per_sample_norms"] = norms.tolist()

        self._append_jsonl(self.records_path, record)
        self.reset()
        return record
