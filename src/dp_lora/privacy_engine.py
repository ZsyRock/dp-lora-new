"""Orchestrate fail-closed DP-LoRA training and privacy accounting."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dp_lora.accounting.accountant import PrivacyAccountant, get_noise_multiplier
from dp_lora.config import GradientObservationConfig, SlaClipConfig
from dp_lora.data.poisson_loader import create_poisson_dataloader
from dp_lora.diagnostics import TrustedGradientObserver
from dp_lora.grad_sample.grad_sample_module import GradSampleModule
from dp_lora.optimizers.dp_optimizer import DPOptimizer
from dp_lora.optimizers.slaclip_optimizer import SlaClipOptimizer

logger = logging.getLogger(__name__)


class DPLoRAEngine:
    """Prepare a PEFT model for record-level flat DP-SGD.

    Formal epsilon accounting is provided only for Poisson sampling. A regular
    DataLoader is available solely for deterministic tests when callers
    explicitly set ``accounting_mode="disabled"``; such runs must not claim an
    epsilon value.
    """

    def __init__(self):
        self.accountant: Optional[PrivacyAccountant] = None
        self.grad_sample_module: Optional[GradSampleModule] = None
        self._dp_optimizer: Optional[DPOptimizer] = None
        self.observer: Optional[TrustedGradientObserver] = None
        self.sample_rate: Optional[float] = None

    def make_private(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: float,
        method: str = "ffa",
        adapter_name: str = "default",
        clipping_mode: str = "fixed",
        slaclip: Optional[SlaClipConfig] = None,
        poisson_sampling: bool = True,
        accounting_mode: str = "rdp_poisson",
        target_delta: float = 1e-5,
        loss_reduction: str = "mean",
        ghost_clipping: bool = False,
        secure_mode: bool = False,
        generator: Optional[torch.Generator] = None,
        gradient_observation: Optional[GradientObservationConfig] = None,
    ) -> tuple[nn.Module, DPOptimizer, DataLoader]:
        """Make the optimizer private using fixed-C or the main SlaClip method."""
        if ghost_clipping:
            raise NotImplementedError(
                "Ghost clipping is disabled until its two-pass update protects "
                "every trainable parameter end-to-end"
            )
        if clipping_mode not in ("fixed", "slaclip"):
            raise ValueError("clipping_mode must be 'fixed' or 'slaclip'")
        if accounting_mode not in ("rdp_poisson", "disabled"):
            raise ValueError("accounting_mode must be 'rdp_poisson' or 'disabled'")
        if not poisson_sampling and accounting_mode != "disabled":
            raise ValueError(
                "This accountant supports Poisson sampling only. For deterministic "
                "unit tests use accounting_mode='disabled' and do not report epsilon."
            )
        if target_delta <= 0 or target_delta >= 1:
            raise ValueError("target_delta must lie in (0, 1)")

        self.grad_sample_module = GradSampleModule(
            model,
            method=method,
            adapter_name=adapter_name,
            loss_reduction=loss_reduction,
        )
        logger.info(
            "DP-LoRA protects %d trainable parameters (LoRA method=%s)",
            self.grad_sample_module.num_trainable_params,
            method,
        )

        dataset_size = len(data_loader.dataset)
        if poisson_sampling:
            dp_data_loader = create_poisson_dataloader(data_loader)
            sample_rate = float(dp_data_loader.sample_rate)
        else:
            if data_loader.batch_size is None:
                raise ValueError("Non-Poisson DataLoader must define batch_size")
            dp_data_loader = data_loader
            sample_rate = float(data_loader.batch_size) / dataset_size
        expected_batch_size = int(round(dataset_size * sample_rate))
        if expected_batch_size <= 0:
            raise ValueError("Expected logical batch size must be positive")
        self.sample_rate = sample_rate

        observation_config = gradient_observation or GradientObservationConfig()
        self.observer = (
            TrustedGradientObserver(observation_config)
            if observation_config.enabled
            else None
        )

        common: dict[str, Any] = {
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": max_grad_norm,
            "expected_batch_size": expected_batch_size,
            "generator": generator,
            "secure_mode": secure_mode,
            "observer": self.observer,
        }
        if clipping_mode == "fixed":
            self._dp_optimizer = DPOptimizer(optimizer, **common)
        else:
            controller = slaclip or SlaClipConfig()
            self._dp_optimizer = SlaClipOptimizer(
                optimizer,
                **common,
                num_slots=controller.num_slots,
                eta=controller.eta,
                beta=controller.beta,
                c_min=controller.c_min,
                c_max=controller.c_max,
            )

        if accounting_mode == "rdp_poisson":
            self.accountant = PrivacyAccountant(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                delta=target_delta,
            )
            self._dp_optimizer.attach_step_hook(self.accountant.step)
        else:
            self.accountant = None

        logger.info(
            "DP-LoRA ready: clipping=%s sigma=%.4f C0=%.4f q=%.6f "
            "expected_batch=%d accountant=%s",
            clipping_mode,
            noise_multiplier,
            max_grad_norm,
            sample_rate,
            expected_batch_size,
            accounting_mode,
        )
        return model, self._dp_optimizer, dp_data_loader

    def make_private_with_epsilon(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: float,
        method: str = "ffa",
        adapter_name: str = "default",
        clipping_mode: str = "fixed",
        slaclip: Optional[SlaClipConfig] = None,
        poisson_sampling: bool = True,
        loss_reduction: str = "mean",
        ghost_clipping: bool = False,
        secure_mode: bool = False,
        generator: Optional[torch.Generator] = None,
        gradient_observation: Optional[GradientObservationConfig] = None,
    ) -> tuple[nn.Module, DPOptimizer, DataLoader]:
        """Calibrate sigma for the exact logical-step schedule, then make private."""
        if not poisson_sampling:
            raise ValueError("Epsilon calibration requires poisson_sampling=True")
        # DPDataLoader.from_data_loader uses exactly one Poisson draw per
        # original loader iteration, hence q = 1 / len(loader). Calibration,
        # runtime sampling and accounting must use this identical value.
        sample_rate = 1.0 / len(data_loader)
        steps_per_epoch = len(data_loader)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
        logger.info(
            "Calibrated sigma=%.6f for eps=%.4f delta=%.3e q=%.6f steps=%d",
            noise_multiplier,
            target_epsilon,
            target_delta,
            sample_rate,
            steps_per_epoch * epochs,
        )
        return self.make_private(
            model=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            method=method,
            adapter_name=adapter_name,
            clipping_mode=clipping_mode,
            slaclip=slaclip,
            poisson_sampling=True,
            accounting_mode="rdp_poisson",
            target_delta=target_delta,
            loss_reduction=loss_reduction,
            ghost_clipping=ghost_clipping,
            secure_mode=secure_mode,
            generator=generator,
            gradient_observation=gradient_observation,
        )

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        if self.accountant is None:
            raise RuntimeError(
                "Privacy accounting is disabled; this run must not report epsilon"
            )
        return self.accountant.get_epsilon(delta)

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint optimizer, DP RNG/controller and privacy accountant state."""
        if self._dp_optimizer is None:
            raise RuntimeError("Engine has not been initialized")
        return {
            "optimizer": self._dp_optimizer.state_dict(),
            "accountant": (
                self.accountant.state_dict() if self.accountant is not None else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._dp_optimizer is None:
            raise RuntimeError("Call make_private() before loading engine state")
        self._dp_optimizer.load_state_dict(state["optimizer"])
        accountant_state = state.get("accountant")
        if accountant_state is None:
            if self.accountant is not None:
                raise ValueError("Checkpoint has no accountant state")
        else:
            if self.accountant is None:
                raise ValueError("Checkpoint requires an enabled accountant")
            self.accountant.load_state_dict(accountant_state)
