from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GradientObservationConfig:
    """Configuration for trusted, non-DP gradient diagnostics.

    Enabling this observer does not change the private optimizer update. The
    resulting diagnostic file is nevertheless an exact, data-dependent output
    and therefore is *not* covered by the training run's privacy budget.
    """

    enabled: bool = False
    output_dir: str = "private_diagnostics"
    histogram_bins: int = 20
    store_per_sample_norms: bool = False
    acknowledge_non_dp: bool = False

    def __post_init__(self) -> None:
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be a positive integer")
        if self.enabled and not self.acknowledge_non_dp:
            raise ValueError(
                "Exact gradient observation is not a DP output. Set "
                "acknowledge_non_dp=True only for access-controlled research runs."
            )


@dataclass
class SlaClipConfig:
    """Parameters for the main SlaClip threshold controller.

    ``num_slots=None`` selects K from the paper's batch-size/noise rule. This
    project intentionally does not implement SlaClip-Q.
    """

    num_slots: Optional[int] = None
    eta: float = 0.2
    beta: float = 0.5
    c_min: float = 0.1
    c_max: float = 50.0

    def __post_init__(self) -> None:
        if self.num_slots is not None and self.num_slots <= 0:
            raise ValueError("num_slots must be positive when specified")
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        if self.c_min <= 0 or self.c_max < self.c_min:
            raise ValueError("Require 0 < c_min <= c_max")


@dataclass
class DPLoRAConfig:
    """Configuration for differentially private LoRA fine-tuning."""

    # Privacy parameters
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: Optional[float] = None  # Auto-computed from epsilon if None

    # Clipping experiment: fixed-C baseline or the main SlaClip controller.
    clipping_mode: str = "fixed"  # "fixed" or "slaclip"
    slaclip: SlaClipConfig = field(default_factory=SlaClipConfig)

    # LoRA method
    method: str = "ffa"  # "vanilla" or "ffa"
    adapter_name: str = "default"

    # Data
    poisson_sampling: bool = True
    loss_reduction: str = "mean"

    # Memory optimization
    ghost_clipping: bool = False
    max_physical_batch_size: Optional[int] = None

    # Training
    epochs: int = 1  # Needed for noise multiplier calibration

    # Exact, access-controlled analysis output. Disabled by default.
    gradient_observation: GradientObservationConfig = field(
        default_factory=GradientObservationConfig
    )

    def __post_init__(self):
        if self.method not in ("vanilla", "ffa"):
            raise ValueError(f"method must be 'vanilla' or 'ffa', got '{self.method}'")
        if self.target_epsilon <= 0:
            raise ValueError(
                f"target_epsilon must be positive, got {self.target_epsilon}"
            )
        if self.clipping_mode not in ("fixed", "slaclip"):
            raise ValueError(
                "clipping_mode must be 'fixed' or 'slaclip'; "
                f"got '{self.clipping_mode}'"
            )
        if self.loss_reduction not in ("mean", "sum"):
            raise ValueError("loss_reduction must be 'mean' or 'sum'")
        if self.ghost_clipping:
            raise ValueError(
                "ghost_clipping is not implemented end-to-end and is disabled "
                "to prevent non-private parameter updates"
            )
        if self.target_delta <= 0 or self.target_delta >= 1:
            raise ValueError(f"target_delta must be in (0, 1), got {self.target_delta}")
        if self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be positive, got {self.max_grad_norm}"
            )
