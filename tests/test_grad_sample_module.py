"""Coverage tests for LoRA adapters and trainable heads."""

import pytest
import torch
from torch import nn

from dp_lora.grad_sample.grad_sample_module import GradSampleModule


class TinyLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 3, bias=False)})
        self.classifier = nn.Linear(3, 2)

    def forward(self, x):
        return self.classifier(x + self.lora_B["default"](self.lora_A["default"](x)))


def test_trainable_classification_head_is_dp_covered():
    model = TinyLoRAModel()
    wrapper = GradSampleModule(model, method="vanilla", loss_reduction="mean")
    inputs = torch.randn(4, 3)
    targets = torch.randint(0, 2, (4,))
    nn.functional.cross_entropy(model(inputs), targets).backward()
    protected_ids = {id(param) for param, _ in wrapper.get_per_sample_grads()}
    trainable_ids = {id(param) for param in model.parameters() if param.requires_grad}
    assert protected_ids == trainable_ids
    wrapper.remove_hooks()


def test_unsupported_trainable_parameter_fails_closed():
    model = TinyLoRAModel()
    model.unsupported_scale = nn.Parameter(torch.ones(()))
    with pytest.raises(ValueError, match="fail-closed"):
        GradSampleModule(model, method="vanilla")


def test_ffa_freezes_a_but_protects_head():
    model = TinyLoRAModel()
    wrapper = GradSampleModule(model, method="ffa")
    assert not model.lora_A["default"].weight.requires_grad
    assert model.lora_B["default"].weight.requires_grad
    assert model.classifier.weight.requires_grad
    wrapper.remove_hooks()
