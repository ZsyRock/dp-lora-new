from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from paper_repro.train_federated import (
    accumulate_state,
    clip_noise_and_step,
    equal_record_loss,
    empty_state_like,
    parameter_groups,
    partition_indices,
    restore_trainable_state,
)


class TinyLoRA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(2, 1, bias=False)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(1, 2, bias=False)}
        )


class PaperReproTests(unittest.TestCase):
    def test_equal_record_loss_does_not_token_weight_records(self) -> None:
        class FixedLogits(nn.Module):
            def forward(self, **_: torch.Tensor):  # type: ignore[no-untyped-def]
                # Record 0 has one supervised token with p(class 0)=0.5;
                # record 1 has three with p(class 0) close to one.
                logits = torch.tensor(
                    [
                        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]],
                    ]
                )
                return type("Output", (), {"logits": logits})()

        batch = {
            "input_ids": torch.zeros((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "labels": torch.tensor([[0, -100, -100], [0, 0, 0]]),
        }
        loss = equal_record_loss(FixedLogits(), batch, "bert")
        expected = (
            torch.log(torch.tensor(2.0))
            + torch.log1p(torch.exp(torch.tensor(-4.0)))
        ) / 2
        self.assertTrue(torch.allclose(loss, expected, atol=1e-7))

    def test_partitions_are_disjoint_and_complete(self) -> None:
        parts = partition_indices(103, 5, 42)
        flattened = np.concatenate(parts)
        self.assertEqual(len(flattened), 103)
        self.assertEqual(len(set(flattened.tolist())), 103)
        self.assertEqual(sorted(len(part) for part in parts), [20, 20, 21, 21, 21])

    def test_separate_group_clipping_and_manual_step(self) -> None:
        model = TinyLoRA()
        groups = parameter_groups(model)
        with torch.no_grad():
            for _, parameter in groups["A"]:
                parameter.fill_(1.0)
                parameter.grad = torch.tensor([[3.0, 4.0]])
            for _, parameter in groups["B"]:
                parameter.fill_(1.0)
                parameter.grad = torch.tensor([[0.0], [2.0]])
        stats = clip_noise_and_step(
            groups,
            clip_norm=2.0,
            noise_multiplier=0.0,
            learning_rate=0.5,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertAlmostEqual(float(stats["A"]["raw_norm"]), 5.0, places=6)
        self.assertAlmostEqual(float(stats["A"]["clip_factor"]), 0.4, places=6)
        self.assertTrue(bool(stats["A"]["clipped"]))
        self.assertAlmostEqual(float(stats["B"]["raw_norm"]), 2.0, places=6)
        self.assertFalse(bool(stats["B"]["clipped"]))
        self.assertTrue(
            torch.allclose(groups["A"][0][1], torch.tensor([[0.4, 0.2]]))
        )
        self.assertTrue(
            torch.allclose(groups["B"][0][1], torch.tensor([[1.0], [0.0]]))
        )

    def test_equal_weight_state_aggregation(self) -> None:
        model = TinyLoRA()
        groups = parameter_groups(model)
        reference = {
            name: torch.zeros_like(parameter, device="cpu")
            for entries in groups.values()
            for name, parameter in entries
        }
        aggregate = empty_state_like(reference)
        with torch.no_grad():
            for entries in groups.values():
                for _, parameter in entries:
                    parameter.fill_(2.0)
        accumulate_state(aggregate, groups, 0.25)
        with torch.no_grad():
            for entries in groups.values():
                for _, parameter in entries:
                    parameter.fill_(6.0)
        accumulate_state(aggregate, groups, 0.75)
        restore_trainable_state(groups, aggregate)
        for entries in groups.values():
            for _, parameter in entries:
                self.assertTrue(torch.allclose(parameter, torch.full_like(parameter, 5.0)))


if __name__ == "__main__":
    unittest.main()
