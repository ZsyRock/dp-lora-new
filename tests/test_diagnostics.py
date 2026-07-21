"""Trusted gradient observer is explicit, exact, local and labelled non-DP."""

import json

import pytest
import torch

from dp_lora.config import GradientObservationConfig
from dp_lora.diagnostics import TrustedGradientObserver
from dp_lora.optimizers.dp_optimizer import DPOptimizer


def test_observer_requires_explicit_non_dp_acknowledgement():
    with pytest.raises(ValueError, match="not a DP output"):
        GradientObservationConfig(enabled=True)


def test_observer_writes_exact_logical_step_record(tmp_path):
    config = GradientObservationConfig(
        enabled=True,
        acknowledge_non_dp=True,
        output_dir=str(tmp_path / "private_diagnostics"),
        histogram_bins=2,
        store_per_sample_norms=True,
    )
    observer = TrustedGradientObserver(config)
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = DPOptimizer(
        torch.optim.SGD([param], lr=0.1),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=2,
        observer=observer,
        parameter_names={id(param): "classifier.weight"},
    )
    optimizer.step([(param, torch.tensor([[0.5], [2.0]]))])

    lines = observer.records_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert record["privacy_status"] == "NON_DP_PRIVATE_DIAGNOSTIC"
    assert record["per_sample_norms"] == [0.5, 2.0]
    assert record["clipped_count"] == 1
    assert record["clipped_fraction"] == 0.5
    assert record["norm_statistics"]["std"] == 0.75
    assert "classifier.weight" in record["parameter_norm_statistics"]
    assert record["optimizer_statistics"]["private_gradient_norm"] > 0
    assert observer.warning_path.exists()

    observer.record_training_step(
        {
            "logical_step": 1,
            "epoch": 1,
            "mean_training_loss": 0.75,
            "epsilon": 1.2,
        }
    )
    training_record = json.loads(
        observer.training_steps_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert training_record["privacy_status"] == "NON_DP_PRIVATE_DIAGNOSTIC"
    assert training_record["mean_training_loss"] == 0.75


def test_observer_does_not_change_private_update(tmp_path):
    observed_param = torch.nn.Parameter(torch.zeros(3))
    control_param = torch.nn.Parameter(torch.zeros(3))
    observer = TrustedGradientObserver(
        GradientObservationConfig(
            enabled=True,
            acknowledge_non_dp=True,
            output_dir=str(tmp_path / "private_diagnostics"),
        )
    )
    observed = DPOptimizer(
        torch.optim.SGD([observed_param], lr=0.1),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        expected_batch_size=2,
        generator=torch.Generator().manual_seed(123),
        observer=observer,
    )
    control = DPOptimizer(
        torch.optim.SGD([control_param], lr=0.1),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        expected_batch_size=2,
        generator=torch.Generator().manual_seed(123),
    )
    per_sample = torch.tensor([[0.2, 0.1, 0.0], [2.0, 0.0, 0.0]])
    observed.step([(observed_param, per_sample.clone())])
    control.step([(control_param, per_sample.clone())])
    torch.testing.assert_close(observed_param, control_param)
