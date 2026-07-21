"""Wrap a PEFT model and fail closed on per-sample gradient coverage."""

from __future__ import annotations

import logging
import torch
from torch import nn

from dp_lora.grad_sample.hooks import (
    clear_per_sample_grads,
    linear_backward_hook,
    linear_forward_hook,
)

logger = logging.getLogger(__name__)


class GradSampleModule:
    """Attach per-sample hooks to every supported trainable parameter.

    FFA mode freezes LoRA A first. The wrapper then hooks every trainable
    ``nn.Linear`` parameter, including PEFT ``modules_to_save`` classification
    heads. Any remaining trainable parameter type is rejected: silently
    updating an uncovered parameter with its ordinary gradient would invalidate
    the DP guarantee.

    After a forward+backward pass, per-sample gradients are available via
    ``get_per_sample_grads()``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        method: str = "ffa",
        adapter_name: str = "default",
        loss_reduction: str = "mean",
    ):
        self.model = model
        self.method = method
        self.adapter_name = adapter_name
        if loss_reduction not in ("mean", "sum"):
            raise ValueError("loss_reduction must be 'mean' or 'sum'")
        self.loss_reduction = loss_reduction
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._hooked_modules: list[tuple[str, nn.Linear]] = []

        self._configure_lora_trainability()
        self._attach_hooks()

    def _configure_lora_trainability(self) -> None:
        """Apply the requested LoRA A/B trainability before coverage checks."""
        lora_layer_count = 0
        for _, module in self.model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            if (
                self.adapter_name not in module.lora_A
                or self.adapter_name not in module.lora_B
            ):
                continue
            lora_layer_count += 1
            if self.method == "ffa":
                for param in module.lora_A[self.adapter_name].parameters():
                    param.requires_grad = False

        if lora_layer_count == 0:
            raise ValueError(
                f"No LoRA layers found for adapter '{self.adapter_name}'. "
                "Wrap the model with PEFT get_peft_model() first."
            )

    def _attach_hooks(self) -> None:
        """Register hooks on all trainable Linear modules, then check coverage."""
        seen_param_ids: set[int] = set()
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            direct_trainable = [
                param
                for param in module.parameters(recurse=False)
                if param.requires_grad
            ]
            if not direct_trainable:
                continue
            duplicate = [
                param for param in direct_trainable if id(param) in seen_param_ids
            ]
            if duplicate:
                raise ValueError(
                    f"Trainable parameters are shared by multiple modules: {name}"
                )
            self._register_hooks(module, name)
            seen_param_ids.update(id(param) for param in direct_trainable)

        logger.info(
            "Attached per-sample gradient hooks to %d trainable Linear module(s) "
            "(method=%s, adapter=%s, loss_reduction=%s)",
            len(self._hooked_modules),
            self.method,
            self.adapter_name,
            self.loss_reduction,
        )

        if not self._hooked_modules:
            raise ValueError(
                "No supported trainable nn.Linear parameters were found after "
                "configuring LoRA trainability"
            )

        uncovered = [
            name
            for name, param in self.model.named_parameters()
            if param.requires_grad and id(param) not in seen_param_ids
        ]
        if uncovered:
            self.remove_hooks()
            preview = ", ".join(uncovered[:8])
            suffix = " ..." if len(uncovered) > 8 else ""
            raise ValueError(
                "DP protection is fail-closed: these trainable parameters do not "
                f"have supported per-sample gradients: {preview}{suffix}. Freeze "
                "them or add a validated grad sampler before training."
            )

    def _register_hooks(self, linear_module: nn.Linear, name: str) -> None:
        linear_module._dp_loss_reduction = self.loss_reduction
        h_fwd = linear_module.register_forward_hook(linear_forward_hook)
        h_bwd = linear_module.register_full_backward_hook(linear_backward_hook)
        self._hooks.extend([h_fwd, h_bwd])
        self._hooked_modules.append((name, linear_module))

    def get_per_sample_grads(self) -> list[tuple[nn.Parameter, torch.Tensor]]:
        """Collect per-sample gradients from all hooked modules.

        Returns:
            List of (parameter, per_sample_grad) tuples.
            per_sample_grad has shape [batch_size, *param.shape].
        """
        grads = []
        missing = []
        batch_size = None
        for name, module in self._hooked_modules:
            if module.weight.requires_grad and hasattr(
                module, "_dp_per_sample_grad_weight"
            ):
                grads.append((module.weight, module._dp_per_sample_grad_weight))
            elif module.weight.requires_grad:
                missing.append(f"{name}.weight")
            if (
                module.bias is not None
                and module.bias.requires_grad
                and hasattr(module, "_dp_per_sample_grad_bias")
            ):
                grads.append((module.bias, module._dp_per_sample_grad_bias))
            elif module.bias is not None and module.bias.requires_grad:
                missing.append(f"{name}.bias")

        if missing:
            raise RuntimeError(
                "Per-sample gradients were not produced for: " + ", ".join(missing)
            )
        for _, grad in grads:
            if batch_size is None:
                batch_size = grad.shape[0]
            elif grad.shape[0] != batch_size:
                raise RuntimeError("Inconsistent per-sample gradient batch dimensions")
        return grads

    def clear_per_sample_grads(self) -> None:
        """Free stored per-sample gradient tensors."""
        for _, module in self._hooked_modules:
            clear_per_sample_grads(module)

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return the list of LoRA parameters that are trainable (have hooks)."""
        params = []
        for _, module in self._hooked_modules:
            params.append(module.weight)
            if module.bias is not None and module.bias.requires_grad:
                params.append(module.bias)
        return params

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for _, module in self._hooked_modules:
            if hasattr(module, "_dp_loss_reduction"):
                del module._dp_loss_reduction
        self._hooked_modules.clear()
        logger.info("Removed all per-sample gradient hooks.")

    def __del__(self):
        self.remove_hooks()
