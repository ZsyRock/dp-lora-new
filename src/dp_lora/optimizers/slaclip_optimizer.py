"""Main SlaClip controller on top of the same flat DP-LoRA mechanism."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch.optim import Optimizer

from dp_lora.diagnostics import TrustedGradientObserver
from dp_lora.optimizers.dp_optimizer import DPOptimizer, _generate_noise

_Z_0995 = 2.576
_PAPER_K_AT_SIGMA_ONE = {128: 8, 256: 10, 512: 20, 1024: 30, 2048: 50}


def recommended_num_slots(expected_batch_size: int, noise_multiplier: float) -> int:
    """Select K using the SlaClip paper rule (with its tabulated sigma=1 values)."""
    batch_size = int(expected_batch_size)
    sigma = float(noise_multiplier)
    if batch_size <= 0 or sigma <= 0:
        raise ValueError("Automatic SlaClip K selection requires batch_size,sigma > 0")
    if math.isclose(sigma, 1.0, rel_tol=0.0, abs_tol=1e-12):
        if batch_size in _PAPER_K_AT_SIGMA_ONE:
            return _PAPER_K_AT_SIGMA_ONE[batch_size]
    k_max = (batch_size / (2.0 * _Z_0995 * sigma)) ** (2.0 / 3.0)
    return max(1, int(math.floor(k_max)))


class SlaClipOptimizer(DPOptimizer):
    """DPOptimizer whose clipping threshold follows the main SlaClip update.

    The K-dimensional slack vector and the clipped gradient are one joint
    sensitivity-C release with iid Gaussian noise of standard deviation
    ``noise_multiplier * C`` on every coordinate. Threshold adaptation uses
    only that noised slack release, so it adds no separate accountant event.
    """

    clipping_mode = "slaclip"

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: int,
        num_slots: Optional[int] = None,
        eta: float = 0.2,
        beta: float = 0.5,
        c_min: float = 0.1,
        c_max: float = 50.0,
        generator: Optional[torch.Generator] = None,
        auxiliary_generator: Optional[torch.Generator] = None,
        secure_mode: bool = False,
        observer: Optional[TrustedGradientObserver] = None,
        parameter_names: Optional[dict[int, str]] = None,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            generator=generator,
            secure_mode=secure_mode,
            observer=observer,
            parameter_names=parameter_names,
        )
        self.num_slots = int(
            num_slots
            if num_slots is not None
            else recommended_num_slots(expected_batch_size, noise_multiplier)
        )
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if eta <= 0 or not 0 <= beta <= 1:
            raise ValueError("Require eta > 0 and beta in [0, 1]")
        if c_min <= 0 or c_max < c_min:
            raise ValueError("Require 0 < c_min <= c_max")
        if not c_min <= max_grad_norm <= c_max:
            raise ValueError("Initial max_grad_norm must lie in [c_min, c_max]")
        self.eta = float(eta)
        self.beta = float(beta)
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.current_clip = float(max_grad_norm)
        self.auxiliary_generator = auxiliary_generator
        self._slack_sum: Optional[torch.Tensor] = None
        self._slack_scale = 0.0

    def _clip_threshold(self) -> float:
        return self.current_clip

    def _build_slack_vector(
        self, encoded_slack: torch.Tensor, coordinate_bound: float
    ) -> torch.Tensor:
        batch_size = int(encoded_slack.shape[0])
        result = torch.zeros(
            batch_size,
            self.num_slots,
            device=encoded_slack.device,
            dtype=torch.float32,
        )
        full = torch.floor(encoded_slack / coordinate_bound).to(torch.int64)
        full = torch.clamp(full, max=self.num_slots)
        remainder = encoded_slack - full.to(encoded_slack.dtype) * coordinate_bound
        remainder = torch.where(full >= self.num_slots, 0.0, remainder)
        positions = torch.arange(self.num_slots, device=encoded_slack.device)
        result = (positions.view(1, -1) < full.view(-1, 1)).to(result.dtype)
        result.mul_(coordinate_bound)
        has_remainder = full < self.num_slots
        if has_remainder.any():
            row = torch.arange(batch_size, device=encoded_slack.device)[has_remainder]
            result[row, full[has_remainder]] = remainder[has_remainder]
        return result

    def _after_clip(
        self,
        per_sample_norms: torch.Tensor,
        clip_factors: torch.Tensor,
        threshold: float,
        per_parameter_norms: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        super()._after_clip(
            per_sample_norms,
            clip_factors,
            threshold,
            per_parameter_norms=per_parameter_norms,
        )
        coordinate_bound = threshold / math.sqrt(self.num_slots)
        if self._slack_scale and not math.isclose(
            self._slack_scale, coordinate_bound, rel_tol=1e-12
        ):
            raise RuntimeError("SlaClip threshold changed inside one logical batch")
        self._slack_scale = coordinate_bound
        slack = (threshold - per_sample_norms).clamp(min=0.0)
        encoded = slack * math.sqrt(self.num_slots)
        batch_sum = self._build_slack_vector(encoded, coordinate_bound).sum(dim=0)
        if self._slack_sum is None:
            self._slack_sum = batch_sum
        else:
            self._slack_sum.add_(batch_sum.to(self._slack_sum.device))

    def _private_auxiliary_release(self, threshold: float) -> dict[str, Any]:
        if self._slack_sum is None or self._slack_scale <= 0:
            raise RuntimeError("Missing SlaClip slack state")
        noise = _generate_noise(
            std=self.noise_multiplier * threshold,
            reference=self._slack_sum,
            generator=self.auxiliary_generator,
            secure_mode=self.secure_mode,
        )
        slack_indicator = (self._slack_sum + noise) / (
            self._slack_scale * self.expected_batch_size
        )
        q_hat = float(slack_indicator[0].item())
        r_hat = float(slack_indicator[-1].item())
        z_t = r_hat / (threshold + 1e-6)
        gamma_t = max(0.0, min(1.0, 1.0 - self.beta * (1.0 - z_t)))

        log_next = math.log(threshold) + self.eta * (gamma_t - q_hat)
        log_next = max(math.log(self.c_min), min(math.log(self.c_max), log_next))
        self.current_clip = math.exp(log_next)
        self.max_grad_norm = self.current_clip
        return {
            "slack_indicator": [float(v) for v in slack_indicator.tolist()],
            "q_hat": q_hat,
            "r_hat": r_hat,
            "gamma_t": gamma_t,
            "num_slots": self.num_slots,
        }

    def _reset_pending(self) -> None:
        super()._reset_pending()
        self._slack_sum = None
        self._slack_scale = 0.0

    def _extra_state_dict(self) -> dict[str, Any]:
        return {
            "num_slots": self.num_slots,
            "eta": self.eta,
            "beta": self.beta,
            "c_min": self.c_min,
            "c_max": self.c_max,
            "current_clip": self.current_clip,
            "auxiliary_generator_state": (
                self.auxiliary_generator.get_state()
                if self.auxiliary_generator is not None
                else None
            ),
        }

    def _load_extra_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("num_slots", "eta", "beta", "c_min", "c_max"):
            if state[key] != getattr(self, key):
                raise ValueError(
                    f"Checkpoint SlaClip {key} does not match configuration"
                )
        self.current_clip = float(state["current_clip"])
        self.max_grad_norm = self.current_clip
        auxiliary_state = state.get("auxiliary_generator_state")
        if auxiliary_state is not None:
            if self.auxiliary_generator is None:
                raise ValueError(
                    "Checkpoint has SlaClip auxiliary RNG state but no generator"
                )
            self.auxiliary_generator.set_state(auxiliary_state)
