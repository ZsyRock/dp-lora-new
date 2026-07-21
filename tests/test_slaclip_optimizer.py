"""Main SlaClip tests; SlaClip-Q is intentionally absent."""

import math

import torch

from dp_lora.optimizers.slaclip_optimizer import (
    SlaClipOptimizer,
    recommended_num_slots,
)


def _optimizer(**kwargs):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = SlaClipOptimizer(
        torch.optim.SGD([param], lr=0.1),
        noise_multiplier=kwargs.pop("noise_multiplier", 0.0),
        max_grad_norm=kwargs.pop("max_grad_norm", 2.0),
        expected_batch_size=kwargs.pop("expected_batch_size", 1),
        num_slots=kwargs.pop("num_slots", 2),
        **kwargs,
    )
    return param, optimizer


def test_paper_recommended_k_table():
    assert recommended_num_slots(128, 1.0) == 8
    assert recommended_num_slots(256, 1.0) == 10
    assert recommended_num_slots(512, 1.0) == 20


def test_extended_gradient_and_slack_norm_is_bounded():
    _, optimizer = _optimizer(num_slots=4)
    norms = torch.tensor([0.0, 0.5, 1.25, 2.0, 3.0])
    clip = 2.0
    scale = clip / math.sqrt(optimizer.num_slots)
    slack = optimizer._build_slack_vector(
        (clip - norms).clamp(min=0) * math.sqrt(optimizer.num_slots), scale
    )
    extended = torch.sqrt(norms.clamp(max=clip).square() + slack.square().sum(1))
    assert torch.all(extended <= clip + 1e-6)


def test_main_update_uses_noised_slack_release():
    param, optimizer = _optimizer(
        max_grad_norm=2.0,
        expected_batch_size=2,
        num_slots=2,
        eta=0.5,
        beta=0.5,
    )
    stats = optimizer.step([(param, torch.tensor([[0.0], [2.0]]))])
    assert stats["num_slots"] == 2
    assert len(stats["slack_indicator"]) == 2
    assert 0.1 <= optimizer.current_clip <= 50.0


def test_checkpoint_restores_controller_and_rng_state():
    generator = torch.Generator().manual_seed(123)
    auxiliary_generator = torch.Generator().manual_seed(456)
    p1 = torch.nn.Parameter(torch.zeros(1))
    first = SlaClipOptimizer(
        torch.optim.SGD([p1], lr=0.1),
        noise_multiplier=1.0,
        max_grad_norm=2.0,
        expected_batch_size=1,
        num_slots=2,
        generator=generator,
        auxiliary_generator=auxiliary_generator,
    )
    first.step([(p1, torch.tensor([[0.5]]))])
    state = first.state_dict()

    p2 = torch.nn.Parameter(torch.zeros(1))
    second = SlaClipOptimizer(
        torch.optim.SGD([p2], lr=0.1),
        noise_multiplier=1.0,
        max_grad_norm=2.0,
        expected_batch_size=1,
        num_slots=2,
        generator=torch.Generator(),
        auxiliary_generator=torch.Generator(),
    )
    second.load_state_dict(state)
    assert second.current_clip == first.current_clip
    assert second._logical_steps == first._logical_steps
    torch.testing.assert_close(
        second.auxiliary_generator.get_state(),
        first.auxiliary_generator.get_state(),
    )
