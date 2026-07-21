"""Fail-closed DP optimizer: flat per-sample clipping plus Gaussian noise."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from dp_lora.diagnostics import TrustedGradientObserver


def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    secure_mode: bool = False,
) -> torch.Tensor:
    """Generate zero-mean Gaussian noise matching ``reference``."""
    zeros = torch.zeros_like(reference)
    if std == 0:
        return zeros
    if secure_mode:
        # Discard one sample and average four independent draws, following the
        # floating-point hardened construction used by Opacus.
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )
        total = zeros
        for _ in range(4):
            total += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                dtype=reference.dtype,
                generator=generator,
            )
        return total / 2
    return torch.normal(
        mean=0,
        std=std,
        size=reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


class DPOptimizer(Optimizer):
    """Wrap a PyTorch optimizer with record-level flat DP-SGD.

    Every trainable optimizer parameter must appear exactly once in the supplied
    per-sample gradients. This invariant prevents ordinary (unclipped, unnoised)
    gradients from silently updating classification heads or other trainable
    modules.
    """

    clipping_mode = "fixed"

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: int,
        generator: Optional[torch.Generator] = None,
        secure_mode: bool = False,
        observer: Optional[TrustedGradientObserver] = None,
    ):
        if noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if expected_batch_size <= 0:
            raise ValueError("expected_batch_size must be positive")
        self.optimizer = optimizer
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)
        self.expected_batch_size = int(expected_batch_size)
        self.generator = generator
        self.secure_mode = bool(secure_mode)
        self.observer = observer
        self._step_hooks: list[Callable[[], None]] = []

        self._step_skip_queue: list[bool] = []
        self._is_last_step_skipped = False
        self._summed_grads: dict[int, torch.Tensor] = {}
        self._logical_steps = 0
        self._pending_sample_count = 0
        self._pending_num_clipped = 0
        self._pending_clip_factor_sum = 0.0
        self._last_step_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Parameter and virtual-batch invariants
    # ------------------------------------------------------------------

    @property
    def _trainable_params(self) -> list[nn.Parameter]:
        return [
            param
            for group in self.optimizer.param_groups
            for param in group["params"]
            if param.requires_grad
        ]

    def _validate_per_sample_grads(
        self, per_sample_grads: list[tuple[nn.Parameter, torch.Tensor]]
    ) -> int:
        expected = {id(param): param for param in self._trainable_params}
        supplied: dict[int, nn.Parameter] = {}
        batch_size: Optional[int] = None

        for param, grad in per_sample_grads:
            pid = id(param)
            if pid in supplied:
                raise ValueError(
                    "A parameter appeared more than once in per_sample_grads"
                )
            if pid not in expected:
                raise ValueError(
                    "per_sample_grads contains a parameter not in optimizer"
                )
            if grad.ndim != param.ndim + 1 or tuple(grad.shape[1:]) != tuple(
                param.shape
            ):
                raise ValueError(
                    f"Invalid per-sample gradient shape {tuple(grad.shape)} for "
                    f"parameter shape {tuple(param.shape)}"
                )
            if batch_size is None:
                batch_size = int(grad.shape[0])
            elif int(grad.shape[0]) != batch_size:
                raise ValueError("Inconsistent batch dimension across parameters")
            supplied[pid] = param

        missing = set(expected) - set(supplied)
        if missing:
            raise RuntimeError(
                f"Refusing a non-private update: {len(missing)} trainable optimizer "
                "parameter(s) have no per-sample gradient"
            )
        return int(batch_size or 0)

    def signal_skip_step(self, do_skip: bool = True) -> None:
        self._step_skip_queue.append(bool(do_skip))

    def _check_skip_next_step(self) -> bool:
        if self._step_skip_queue:
            return self._step_skip_queue.pop(0)
        return False

    # ------------------------------------------------------------------
    # Core DP operations
    # ------------------------------------------------------------------

    def _clip_threshold(self) -> float:
        return float(self.max_grad_norm)

    def _after_clip(
        self,
        per_sample_norms: torch.Tensor,
        clip_factors: torch.Tensor,
        threshold: float,
    ) -> None:
        if self.observer is not None:
            self.observer.observe_microbatch(per_sample_norms)

    def clip_and_accumulate(
        self, per_sample_grads: list[tuple[nn.Parameter, torch.Tensor]]
    ) -> dict[str, Any]:
        """Jointly norm, clip and sum a physical batch of per-sample gradients."""
        batch_size = self._validate_per_sample_grads(per_sample_grads)
        threshold = self._clip_threshold()

        if batch_size == 0:
            # Poisson sampling can legitimately produce an empty batch.
            device = self._trainable_params[0].device
            per_sample_norms = torch.empty(0, device=device, dtype=torch.float32)
            clip_factors = torch.empty(0, device=device, dtype=torch.float32)
        else:
            device = per_sample_grads[0][1].device
            norm_sq = torch.zeros(batch_size, device=device, dtype=torch.float32)
            for _, grad in per_sample_grads:
                flat = (
                    grad.detach()
                    .to(device=device, dtype=torch.float32)
                    .reshape(batch_size, -1)
                )
                norm_sq.add_((flat * flat).sum(dim=1))
            per_sample_norms = norm_sq.sqrt()
            clip_factors = (threshold / (per_sample_norms + 1e-12)).clamp(max=1.0)

        for param, grad in per_sample_grads:
            factor = clip_factors.to(device=grad.device, dtype=grad.dtype).reshape(
                -1, *([1] * (grad.ndim - 1))
            )
            clipped_sum = (grad * factor).sum(dim=0).to(dtype=param.dtype)
            pid = id(param)
            if pid in self._summed_grads:
                self._summed_grads[pid].add_(clipped_sum)
            else:
                self._summed_grads[pid] = clipped_sum

        num_clipped = int((per_sample_norms > threshold).sum().item())
        clip_factor_sum = float(clip_factors.sum().item())
        self._pending_sample_count += batch_size
        self._pending_num_clipped += num_clipped
        self._pending_clip_factor_sum += clip_factor_sum
        self._after_clip(per_sample_norms, clip_factors, threshold)

        return {
            "privacy_status": "EXACT_INTERNAL_DIAGNOSTIC",
            "sample_count": batch_size,
            "num_clipped": num_clipped,
            "mean_clip_factor": (clip_factor_sum / batch_size if batch_size else 1.0),
            "max_grad_norm": threshold,
        }

    def _private_auxiliary_release(self, threshold: float) -> dict[str, Any]:
        return {}

    def add_noise_and_finalize(self) -> dict[str, Any]:
        """Noise the clipped sum and replace every ordinary parameter gradient."""
        threshold = self._clip_threshold()
        for param in self._trainable_params:
            summed = self._summed_grads.get(id(param))
            if summed is None:
                raise RuntimeError(
                    "Refusing a non-private update: missing clipped gradient sum"
                )
            noise = _generate_noise(
                std=self.noise_multiplier * threshold,
                reference=summed,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            param.grad = (summed + noise).view_as(param) / self.expected_batch_size
        return self._private_auxiliary_release(threshold)

    def _reset_pending(self) -> None:
        self._summed_grads.clear()
        self._pending_sample_count = 0
        self._pending_num_clipped = 0
        self._pending_clip_factor_sum = 0.0

    def step(
        self,
        per_sample_grads: Optional[list[tuple[nn.Parameter, torch.Tensor]]] = None,
    ) -> dict[str, Any]:
        if per_sample_grads is not None:
            self.clip_and_accumulate(per_sample_grads)
        if not self._summed_grads:
            raise RuntimeError(
                "DPOptimizer.step() requires validated per-sample gradients before update"
            )

        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return {
                "skipped": True,
                "sample_count": self._pending_sample_count,
                "max_grad_norm": self._clip_threshold(),
            }

        clip_before = self._clip_threshold()
        private_aux = self.add_noise_and_finalize()
        self.optimizer.step()
        for fn in self._step_hooks:
            fn()

        self._logical_steps += 1
        clip_after = self._clip_threshold()
        sample_count = self._pending_sample_count
        stats: dict[str, Any] = {
            "skipped": False,
            "logical_step": self._logical_steps,
            "sample_count": sample_count,
            "num_clipped": self._pending_num_clipped,
            "mean_clip_factor": (
                self._pending_clip_factor_sum / sample_count if sample_count else 1.0
            ),
            "max_grad_norm_before": clip_before,
            "max_grad_norm_after": clip_after,
        }
        stats.update(private_aux)
        if self.observer is not None:
            self.observer.finalize_step(
                logical_step=self._logical_steps,
                clipping_mode=self.clipping_mode,
                clip_before=clip_before,
                clip_after=clip_after,
                private_aux=private_aux,
            )

        self._last_step_stats = stats
        self._is_last_step_skipped = False
        self._reset_pending()
        return stats

    def zero_grad(self, set_to_none: bool = True) -> None:
        if not self._is_last_step_skipped:
            self._reset_pending()
            if self.observer is not None:
                self.observer.reset()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------
    # Hooks, compatibility and checkpoint state
    # ------------------------------------------------------------------

    def attach_step_hook(self, fn: Callable[[], None]) -> None:
        self._step_hooks.append(fn)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, value):
        self.optimizer.state = value

    @property
    def defaults(self):
        return self.optimizer.defaults

    @defaults.setter
    def defaults(self, value):
        self.optimizer.defaults = value

    def _extra_state_dict(self) -> dict[str, Any]:
        return {}

    def _load_extra_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError(f"Unexpected DP optimizer state keys: {sorted(state)}")

    def state_dict(self) -> dict[str, Any]:
        if self._summed_grads:
            raise RuntimeError("Checkpoint only at logical-step boundaries")
        generator_state = None
        if self.generator is not None:
            generator_state = self.generator.get_state()
        return {
            "optimizer": self.optimizer.state_dict(),
            "dp": {
                "noise_multiplier": self.noise_multiplier,
                "max_grad_norm": self.max_grad_norm,
                "expected_batch_size": self.expected_batch_size,
                "secure_mode": self.secure_mode,
                "logical_steps": self._logical_steps,
                "generator_state": generator_state,
                "extra": self._extra_state_dict(),
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "optimizer" not in state_dict or "dp" not in state_dict:
            raise ValueError("Not a complete DPOptimizer checkpoint")
        dp_state = state_dict["dp"]
        for key, expected in (
            ("noise_multiplier", self.noise_multiplier),
            ("expected_batch_size", self.expected_batch_size),
            ("secure_mode", self.secure_mode),
        ):
            if dp_state[key] != expected:
                raise ValueError(
                    f"Checkpoint {key}={dp_state[key]!r} does not match {expected!r}"
                )
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.max_grad_norm = float(dp_state["max_grad_norm"])
        self._logical_steps = int(dp_state["logical_steps"])
        if dp_state.get("generator_state") is not None:
            if self.generator is None:
                raise ValueError(
                    "Checkpoint has DP RNG state but optimizer has no generator"
                )
            self.generator.set_state(dp_state["generator_state"])
        self._load_extra_state_dict(dp_state.get("extra", {}))
        self._reset_pending()
