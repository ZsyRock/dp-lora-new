from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from paper_repro.train_federated import (
    accumulate_state,
    acquire_run_lock,
    canonical_adapter_state_sha256,
    clip_noise_and_step,
    equal_record_loss,
    EffectiveConfig,
    empty_mechanism_components,
    empty_state_like,
    illustrative_accounting_diagnostic,
    make_effective_config,
    parameter_groups,
    parse_args,
    partition_indices,
    load_data_protocol,
    round_update_statistics,
    clone_trainable_state,
    restore_trainable_state,
    resolve_slaclip_base_targets,
    seed_model_stochasticity,
    validate_adapter_artifact,
    validate_checkpoint_trainer_state,
    validate_input_manifest,
    validate_private_directory,
    read_round_shards,
    save_final_adapter_atomically,
    SLACLIP_PAIRED_SLACK_NOISE_SCOPE,
    slaclip_slack_noise_method_scope,
    slaclip_round_controller_summary,
    validate_completed_root_summary,
)
from paper_repro.reproducibility import METHOD_SPECS, canonical_json_fingerprint


class TinyLoRA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(2, 1, bias=False)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(1, 2, bias=False)}
        )


def paper_config(*, rounds: int = 3, eval_every: int = 1) -> EffectiveConfig:
    return EffectiveConfig(
        method="paper_dp_lora",
        num_clients=2,
        rounds=rounds,
        batch_size=1,
        noise_multiplier=2.0,
        learning_rate=5e-4,
        clip_norm=10.0,
        rank=8,
        max_seq_length=16,
        seed=42,
        data_split_seed=1729,
        evaluation_seed=2718,
        max_validation_records=1,
        eval_every=eval_every,
        checkpoint_every=1,
        data_protocol="paper_union_minus_fixed_holdout",
        delta=1e-5,
        pair_noise_across_methods=False,
        smoke=True,
    )


def valid_checkpoint_state(config: EffectiveConfig, completed_round: int) -> dict:
    evaluation_rounds = [0] + [
        round_index
        for round_index in range(1, completed_round + 1)
        if round_index % config.eval_every == 0 or round_index == config.rounds
    ]
    return {
        "model": "bert",
        "method": config.method,
        "run_config_fingerprint": "run-fingerprint",
        "private_key_commitment": "key-commitment",
        "rng_domain": "rng-domain",
        "total_client_steps": completed_round * config.num_clients,
        "clipped_counts": {"A": 0, "B": 0, "any": 0},
        "would_clip_counts": {"A": 0, "B": 0, "any": 0},
        "evaluations": [
            {"round": round_index, "loss": 1.0}
            for round_index in evaluation_rounds
        ],
        "sampled_unique_indices": [[0], [1]],
        "active_elapsed_seconds": 1.0,
        "last_round_summary": {
            "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
            "method": config.method,
            "model": "bert",
            "round": completed_round,
            "mean_training_loss": 1.0,
        },
        "round_shard_prefix_sha256": "a" * 64,
    }


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_inventory_metadata(manifest: dict) -> None:
    manifest["inventory_files"] = len(manifest["inventory"])
    manifest["inventory_bytes"] = sum(
        entry["bytes"] for entry in manifest["inventory"]
    )
    manifest["inventory_sha256"] = canonical_json_fingerprint(
        manifest["inventory"]
    )


def minimal_valid_input_manifest(root: Path) -> dict:
    inventory: list[dict] = []
    combined_splits: dict[str, dict] = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.parquet"
        path.write_bytes(split.encode("ascii"))
        inventory.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_bytes(path),
                "role": "formal_dataset",
            }
        )
        combined_splits[split] = {"files": [str(path)], "rows": 1}

    model_specs = {
        "bert-base-uncased": (
            "google-bert/bert-base-uncased",
            "86b5e0934494bd15c9632b12f734a8a67f723594",
        ),
        "gpt2": (
            "openai-community/gpt2",
            "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        ),
    }
    models: dict[str, dict] = {}
    for model_key, (repo_id, revision) in model_specs.items():
        snapshot = root / model_key
        snapshot.mkdir()
        model_files = []
        for filename in ("config.json", "model.safetensors"):
            path = snapshot / filename
            path.write_bytes(f"{model_key}:{filename}".encode("ascii"))
            metadata = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_bytes(path),
            }
            model_files.append(metadata)
            inventory.append(
                {**metadata, "role": "model", "model": model_key}
            )
        models[model_key] = {
            "repo_id": repo_id,
            "revision": revision,
            "snapshot_path": str(snapshot),
            "files": model_files,
        }

    manifest = {
        "status": "STAGING_COMPLETE_VERIFIED",
        "formal_dataset": {
            "repo_id": "lighteval/med_dialog",
            "revision": "ce8a234c92ea9a37743ad8154253ba897a4a70a5",
            "combined_splits": combined_splits,
        },
        "models": models,
        "inventory": inventory,
    }
    refresh_inventory_metadata(manifest)
    return manifest


class PaperReproTests(unittest.TestCase):
    def test_slaclip_accounting_diagnostic_reports_the_explicit_k(self) -> None:
        diagnostic = illustrative_accounting_diagnostic(
            client_partition_sizes=[10] * 5,
            batch_size=8,
            rounds=50,
            noise_multiplier=2.0,
            delta=1e-5,
            contains_slaclip=True,
            slaclip_num_slots=5,
        )
        self.assertTrue(any("K=5" in reason for reason in diagnostic["reasons"]))
        self.assertFalse(any("K=15" in reason for reason in diagnostic["reasons"]))
        self.assertEqual(diagnostic["controller_input"], "noisy_endpoints")
        oracle = illustrative_accounting_diagnostic(
            client_partition_sizes=[10] * 5,
            batch_size=8,
            rounds=50,
            noise_multiplier=2.0,
            delta=1e-5,
            contains_slaclip=True,
            slaclip_num_slots=5,
            controller_input="exact_endpoints",
        )
        self.assertEqual(oracle["controller_input"], "exact_endpoints")
        self.assertTrue(any("explicitly NON-DP" in reason for reason in oracle["reasons"]))
        self.assertFalse(
            any("are not controller inputs" in reason for reason in oracle["reasons"])
        )
        self.assertEqual(
            SLACLIP_PAIRED_SLACK_NOISE_SCOPE,
            "slaclip_dp_lora",
        )
        self.assertEqual(
            slaclip_slack_noise_method_scope(
                "slaclip_dp_lora", pair_noise_across_methods=True
            ),
            "slaclip_dp_lora",
        )
        self.assertEqual(
            slaclip_slack_noise_method_scope(
                "slaclip_dp_lora", pair_noise_across_methods=False
            ),
            "slaclip_dp_lora",
        )
        self.assertEqual(
            slaclip_slack_noise_method_scope(
                "oracle_slaclip_control", pair_noise_across_methods=True
            ),
            "slaclip_dp_lora",
        )
        self.assertEqual(
            slaclip_slack_noise_method_scope(
                "oracle_slaclip_control", pair_noise_across_methods=False
            ),
            "oracle_slaclip_control",
        )
        with self.assertRaisesRegex(ValueError, "no SlaClip slack-noise domain"):
            slaclip_slack_noise_method_scope(
                "paper_dp_lora", pair_noise_across_methods=True
            )
        with self.assertRaisesRegex(ValueError, "requires at least two slots"):
            illustrative_accounting_diagnostic(
                client_partition_sizes=[10] * 5,
                batch_size=8,
                rounds=50,
                noise_multiplier=2.0,
                delta=1e-5,
                contains_slaclip=True,
            )

    def test_full_slaclip_base_target_cli_and_beta_alias_resolve_fail_closed(
        self,
    ) -> None:
        default = parse_args(["--input-manifest", "inputs.json"])
        self.assertEqual(default.slaclip_base_target_clipped_fraction, 0.5)
        self.assertEqual(default.slaclip_beta, 0.5)
        self.assertIsNone(
            default.slaclip_base_target_clipped_fraction_by_group
        )
        self.assertEqual(
            default.slaclip_base_target_clipped_fraction_source,
            "default",
        )

        canonical = parse_args(
            [
                "--input-manifest",
                "inputs.json",
                "--slaclip-base-target-clipped-fraction",
                "0.125",
            ]
        )
        self.assertEqual(canonical.slaclip_base_target_clipped_fraction, 0.125)
        self.assertEqual(canonical.slaclip_beta, 0.125)
        self.assertEqual(
            canonical.slaclip_base_target_clipped_fraction_source,
            "canonical_cli",
        )

        alias = parse_args(
            [
                "--input-manifest",
                "inputs.json",
                "--slaclip-beta",
                "0.25",
            ]
        )
        self.assertEqual(alias.slaclip_base_target_clipped_fraction, 0.25)
        self.assertEqual(alias.slaclip_beta, 0.25)
        self.assertEqual(
            alias.slaclip_base_target_clipped_fraction_source,
            "compatibility_alias",
        )

        compatible = parse_args(
            [
                "--input-manifest",
                "inputs.json",
                "--slaclip-base-target-clipped-fraction",
                "0.75",
                "--slaclip-beta",
                "0.75",
            ]
        )
        self.assertEqual(
            compatible.slaclip_base_target_clipped_fraction_source,
            "canonical_and_compatible_alias",
        )
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--input-manifest",
                    "inputs.json",
                    "--slaclip-base-target-clipped-fraction",
                    "0.75",
                    "--slaclip-beta",
                    "0.5",
                ]
            )

    def test_groupwise_full_slaclip_base_target_cli_is_canonical_and_exclusive(
        self,
    ) -> None:
        groupwise = parse_args(
            [
                "--input-manifest",
                "inputs.json",
                "--slaclip-base-target-clipped-fraction-a",
                "0.3333333333333333",
                "--slaclip-base-target-clipped-fraction-b",
                "0.8",
            ]
        )
        self.assertIsNone(groupwise.slaclip_base_target_clipped_fraction)
        self.assertIsNone(groupwise.slaclip_beta)
        self.assertEqual(
            groupwise.slaclip_base_target_clipped_fraction_by_group,
            {"A": 1.0 / 3.0, "B": 0.8},
        )
        self.assertEqual(
            groupwise.slaclip_base_target_clipped_fraction_source,
            "groupwise_canonical_cli",
        )

        invalid_argv = (
            [
                "--slaclip-base-target-clipped-fraction-a",
                "0.4",
            ],
            [
                "--slaclip-base-target-clipped-fraction-b",
                "0.4",
            ],
            [
                "--slaclip-base-target-clipped-fraction-a",
                "0.4",
                "--slaclip-base-target-clipped-fraction-b",
                "0.6",
                "--slaclip-base-target-clipped-fraction",
                "0.5",
            ],
            [
                "--slaclip-base-target-clipped-fraction-a",
                "0.4",
                "--slaclip-base-target-clipped-fraction-b",
                "0.6",
                "--slaclip-beta",
                "0.5",
            ],
            [
                "--slaclip-base-target-clipped-fraction-a",
                "-0.1",
                "--slaclip-base-target-clipped-fraction-b",
                "0.6",
            ],
        )
        for options in invalid_argv:
            with self.subTest(options=options), self.assertRaises(SystemExit):
                parse_args(["--input-manifest", "inputs.json", *options])

        with self.assertRaisesRegex(ValueError, "exactly A and B"):
            resolve_slaclip_base_targets(
                base_target_clipped_fraction_by_group={"A": 0.5}
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_slaclip_base_targets(
                beta=0.5,
                base_target_clipped_fraction_by_group={"A": 0.5, "B": 0.5},
            )

    def test_groupwise_clip_norm_cli_is_paired_positive_and_effective(self) -> None:
        shared_args = parse_args(["--input-manifest", "inputs.json"])
        shared = make_effective_config(shared_args)
        self.assertEqual(shared.clip_norm_by_group, {"A": 10.0, "B": 10.0})

        groupwise_args = parse_args(
            [
                "--input-manifest",
                "inputs.json",
                "--clip-norm",
                "3.0",
                "--clip-norm-a",
                "0.75",
                "--clip-norm-b",
                "3.0",
            ]
        )
        groupwise = make_effective_config(groupwise_args)
        self.assertEqual(groupwise.clip_norm, 3.0)
        self.assertEqual(groupwise.clip_norm_by_group, {"A": 0.75, "B": 3.0})

        for options in (
            ["--clip-norm-a", "0.75"],
            ["--clip-norm-b", "3.0"],
            ["--clip-norm-a", "0", "--clip-norm-b", "3.0"],
            ["--clip-norm-a", "nan", "--clip-norm-b", "3.0"],
        ):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                parse_args(["--input-manifest", "inputs.json", *options])

    def test_non_dp_baseline_calibration_provenance_requires_a_lock_and_ack(self) -> None:
        digest = "a" * 64
        parsed = parse_args(
            [
                "--input-manifest",
                "inputs.json",
                "--slaclip-baseline-calibration-lock-sha256",
                digest,
                "--acknowledge-slaclip-baseline-calibration-is-non-dp",
            ]
        )
        self.assertEqual(
            parsed.slaclip_baseline_calibration_lock_sha256,
            digest,
        )
        self.assertTrue(
            parsed.acknowledge_slaclip_baseline_calibration_is_non_dp
        )
        invalid_argv = (
            ["--slaclip-baseline-calibration-lock-sha256", digest],
            ["--acknowledge-slaclip-baseline-calibration-is-non-dp"],
            [
                "--slaclip-baseline-calibration-lock-sha256",
                "A" * 64,
                "--acknowledge-slaclip-baseline-calibration-is-non-dp",
            ],
        )
        for options in invalid_argv:
            with self.subTest(options=options), self.assertRaises(SystemExit):
                parse_args(["--input-manifest", "inputs.json", *options])

    def test_completed_root_summary_validation_rejects_derived_mismatch(self) -> None:
        config = paper_config()
        results = {
            "bert": {
                "behavior_summary": {"sample_schedule_sha256": "a" * 64}
            }
        }
        summary = {
            "schema_version": 2,
            "status": "COMPLETED",
            "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
            "contains_slaclip": False,
            "method": config.method,
            "method_spec": asdict(METHOD_SPECS[config.method]),
            "run_config_fingerprint": "run-fingerprint",
            "reproduction_claim_level": 1,
            "paper_result_reproduced": False,
            "paper_benchmarks_evaluated": False,
            "privacy_claim": False,
            "models": results,
            "sample_schedule_sha256_by_model": {"bert": "a" * 64},
            "elapsed_seconds": 12.5,
            "completed_at_utc": "2026-07-23T00:00:00+00:00",
        }
        validate_completed_root_summary(
            summary,
            config=config,
            run_config_fingerprint="run-fingerprint",
            results=results,
        )
        corrupted = copy.deepcopy(summary)
        corrupted["sample_schedule_sha256_by_model"]["bert"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "sample_schedule"):
            validate_completed_root_summary(
                corrupted,
                config=config,
                run_config_fingerprint="run-fingerprint",
                results=results,
            )

    def test_failed_adapter_write_is_quarantined_outside_publish_namespace(self) -> None:
        class FailingModel:
            @staticmethod
            def save_pretrained(directory: Path, *, safe_serialization: bool) -> None:
                if safe_serialization is not True:
                    raise AssertionError("safe serialization was not requested")
                partial = directory / "partial.bin"
                partial.write_bytes(b"incomplete")
                raise RuntimeError("injected adapter save failure")

        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root) / "model"
            output_dir.mkdir(mode=0o700)
            with self.assertRaisesRegex(RuntimeError, "injected adapter save failure"):
                save_final_adapter_atomically(
                    FailingModel(), output_dir, resume=False
                )
            self.assertFalse((output_dir / "final_adapter").exists())
            self.assertEqual(list(output_dir.glob(".final_adapter.*.tmp")), [])
            quarantined = list(
                (output_dir / "failed_final_adapter_writes").glob("failed-*")
            )
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(
                (quarantined[0] / "partial.bin").read_bytes(), b"incomplete"
            )

    def test_resume_never_replaces_an_existing_final_adapter(self) -> None:
        class MustNotSaveModel:
            @staticmethod
            def save_pretrained(directory: Path, *, safe_serialization: bool) -> None:
                raise AssertionError("resume attempted to replace final adapter")

        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root) / "model"
            output_dir.mkdir(mode=0o700)
            adapter_dir = output_dir / "final_adapter"
            adapter_dir.mkdir(mode=0o700)
            sentinel = adapter_dir / "sentinel"
            sentinel.write_bytes(b"authoritative")
            sentinel.chmod(0o600)
            result = save_final_adapter_atomically(
                MustNotSaveModel(), output_dir, resume=True
            )
            self.assertEqual(result, adapter_dir)
            self.assertEqual(sentinel.read_bytes(), b"authoritative")
            self.assertFalse((output_dir / "superseded_final_adapters").exists())

    def test_canonical_adapter_state_hash_normalizes_peft_default_namespace(self) -> None:
        saved = {
            "layer.lora_A.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "layer.lora_B.weight": torch.ones(3, 2),
        }
        live = {
            name.replace(".weight", ".default.weight"): tensor.clone()
            for name, tensor in saved.items()
        }
        self.assertEqual(
            canonical_adapter_state_sha256(saved),
            canonical_adapter_state_sha256(live),
        )

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

    def test_groupwise_fixed_thresholds_control_clipping_and_noise_per_group(self) -> None:
        model = TinyLoRA()
        groups = parameter_groups(model)
        with torch.no_grad():
            groups["A"][0][1].grad = torch.tensor([[3.0, 4.0]])
            groups["B"][0][1].grad = torch.tensor([[0.0], [2.0]])
        stats = clip_noise_and_step(
            groups,
            clip_norm_by_group={"A": 1.0, "B": 4.0},
            noise_multiplier=0.5,
            learning_rate=0.0,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertAlmostEqual(stats["A"]["clip_threshold"], 1.0)
        self.assertAlmostEqual(stats["B"]["clip_threshold"], 4.0)
        self.assertAlmostEqual(stats["A"]["clip_factor"], 0.2, places=6)
        self.assertEqual(stats["B"]["clip_factor"], 1.0)
        self.assertAlmostEqual(stats["A"]["noise_std_per_coordinate"], 0.5)
        self.assertAlmostEqual(stats["B"]["noise_std_per_coordinate"], 2.0)

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

    def test_no_dp_control_records_counterfactual_without_clipping(self) -> None:
        model = TinyLoRA()
        groups = parameter_groups(model)
        with torch.no_grad():
            for _, parameter in groups["A"]:
                parameter.zero_()
                parameter.grad = torch.tensor([[3.0, 4.0]])
            for _, parameter in groups["B"]:
                parameter.zero_()
                parameter.grad = torch.tensor([[0.0], [2.0]])
        stats = clip_noise_and_step(
            groups,
            clip_norm=2.0,
            noise_multiplier=0.0,
            learning_rate=0.5,
            generator=torch.Generator().manual_seed(7),
            apply_clipping=False,
        )
        self.assertFalse(stats["A"]["clipped"])
        self.assertTrue(stats["A"]["would_clip"])
        self.assertEqual(stats["A"]["clip_factor"], 1.0)
        self.assertAlmostEqual(stats["A"]["counterfactual_clip_factor"], 0.4)
        self.assertTrue(
            torch.allclose(groups["A"][0][1], torch.tensor([[-1.5, -2.0]]))
        )

    def test_slaclip_joint_release_preserves_baseline_gradient_noise_direction(self) -> None:
        baseline = TinyLoRA()
        adaptive = TinyLoRA()
        adaptive.load_state_dict(baseline.state_dict())
        baseline_groups = parameter_groups(baseline)
        adaptive_groups = parameter_groups(adaptive)
        for groups in (baseline_groups, adaptive_groups):
            with torch.no_grad():
                groups["A"][0][1].grad = torch.tensor([[3.0, 4.0]])
                groups["B"][0][1].grad = torch.tensor([[1.0], [2.0]])
        baseline_stats = clip_noise_and_step(
            baseline_groups,
            clip_norm=2.0,
            noise_multiplier=0.5,
            learning_rate=0.1,
            generator=torch.Generator().manual_seed(123),
        )
        adaptive_stats = clip_noise_and_step(
            adaptive_groups,
            clip_norm_by_group={"A": 2.0, "B": 2.0},
            noise_multiplier=0.5,
            learning_rate=0.1,
            generator=torch.Generator().manual_seed(123),
            slaclip_num_slots=1,
            slack_noise_generators={
                "A": torch.Generator().manual_seed(400),
                "B": torch.Generator().manual_seed(500),
            },
        )
        for group in ("A", "B"):
            for (_, baseline_parameter), (_, adaptive_parameter) in zip(
                baseline_groups[group], adaptive_groups[group]
            ):
                self.assertTrue(torch.equal(baseline_parameter, adaptive_parameter))
            baseline_core = dict(baseline_stats[group])
            adaptive_core = dict(adaptive_stats[group])
            adaptive_core.pop("slaclip")
            self.assertEqual(baseline_core, adaptive_core)
            self.assertTrue(
                adaptive_stats[group]["slaclip"][
                    "joint_sensitivity_bound_passed"
                ]
            )

    def test_slaclip_round_uses_both_noisy_cdf_endpoints_for_A_and_B(self) -> None:
        records = []
        for _ in range(2):
            records.append(
                {
                    "gradient_groups": {
                        "A": {
                            "clip_threshold": 2.0,
                            "clipped": True,
                            "slaclip": {
                                "num_slots": 2,
                                "noisy_slack": [1.0, 0.2],
                                "slack_signal": [0.9, 0.1],
                                "joint_sensitivity_bound_passed": True,
                                "slack_noise_std_per_coordinate": 1.0,
                            },
                        },
                        "B": {
                            "clip_threshold": 1.0,
                            "clipped": False,
                            "slaclip": {
                                "num_slots": 2,
                                "noisy_slack": [0.4, 0.2],
                                "slack_signal": [0.3, 0.1],
                                "joint_sensitivity_bound_passed": True,
                                "slack_noise_std_per_coordinate": 0.5,
                            },
                        },
                    }
                }
            )
        summary = slaclip_round_controller_summary(
            records,
            clip_thresholds={"A": 2.0, "B": 1.0},
            num_slots=2,
            eta=0.2,
            beta=0.5,
            c_min=0.1,
            c_max=50.0,
        )
        self.assertEqual(summary["variant"], "full_slaclip_cdf_endpoints")
        self.assertEqual(summary["base_target_clipped_fraction"], 0.5)
        self.assertEqual(summary["beta"], 0.5)
        self.assertNotIn("base_target_clipped_fraction_by_group", summary)
        self.assertNotIn("beta_by_group", summary)
        self.assertNotIn("target_parameterization", summary)
        self.assertEqual(summary["near_threshold_index"], 0)
        self.assertEqual(summary["near_zero_index"], 1)
        self.assertEqual(summary["A"]["clip_threshold_used"], 2.0)
        self.assertEqual(summary["B"]["clip_threshold_used"], 1.0)
        self.assertEqual(summary["A"]["actual_clip_fraction"], 1.0)
        self.assertEqual(summary["B"]["actual_clip_fraction"], 0.0)
        for group, threshold in (("A", 2.0), ("B", 1.0)):
            controller = summary[group]
            noisy_cdf = controller["noisy_cdf_proxy_by_slot"]
            self.assertAlmostEqual(
                controller["near_threshold_proxy"], noisy_cdf[0]
            )
            self.assertAlmostEqual(controller["near_zero_proxy"], noisy_cdf[-1])
            expected_remaining = (
                1.0
                - noisy_cdf[-1] / (threshold + controller["epsilon"])
            )
            self.assertEqual(
                controller["base_target_clipped_fraction"],
                0.5,
            )
            self.assertAlmostEqual(
                controller["remaining_non_small_gradient_fraction"],
                expected_remaining,
            )
            self.assertAlmostEqual(
                controller["raw_dynamic_target_clipped"],
                0.5 * expected_remaining,
            )
            self.assertAlmostEqual(
                controller["clamped_dynamic_target_clipped"],
                controller["dynamic_target_clipped"],
            )
            expected_gamma = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - 0.5
                    * (
                        1.0
                        - noisy_cdf[-1]
                        / (threshold + controller["epsilon"])
                    ),
                ),
            )
            self.assertAlmostEqual(
                controller["dynamic_target_unclipped"], expected_gamma
            )
            expected_next = threshold * np.exp(
                0.2 * (expected_gamma - noisy_cdf[0])
            )
            self.assertAlmostEqual(
                controller["next_clip_threshold"], expected_next
            )

        oracle = slaclip_round_controller_summary(
            records,
            clip_thresholds={"A": 2.0, "B": 1.0},
            num_slots=2,
            eta=0.2,
            controller_input="exact_endpoints",
            beta=0.5,
            c_min=0.1,
            c_max=50.0,
        )
        self.assertEqual(oracle["controller_input"], "exact_endpoints")
        self.assertTrue(oracle["controller_input_is_non_dp_exact"])
        for group, threshold in (("A", 2.0), ("B", 1.0)):
            controller = oracle[group]
            exact_cdf = controller["exact_cdf_proxy_by_slot"]
            self.assertEqual(controller["controller_input"], "exact_endpoints")
            self.assertTrue(controller["controller_input_is_non_dp_exact"])
            self.assertAlmostEqual(
                controller["near_threshold_proxy"], exact_cdf[0]
            )
            self.assertAlmostEqual(controller["near_zero_proxy"], exact_cdf[-1])
            expected_gamma = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - 0.5
                    * (1.0 - exact_cdf[-1] / (threshold + controller["epsilon"])),
                ),
            )
            expected_next = threshold * np.exp(
                0.2 * (expected_gamma - exact_cdf[0])
            )
            self.assertAlmostEqual(
                controller["next_clip_threshold"], expected_next
            )
            self.assertAlmostEqual(
                controller["next_clip_threshold"],
                controller["oracle_next_clip_threshold"],
            )
            self.assertNotAlmostEqual(
                controller["next_clip_threshold"],
                controller["noisy_next_clip_threshold"],
            )

        with self.assertRaisesRegex(ValueError, "unsupported controller input"):
            slaclip_round_controller_summary(
                records,
                clip_thresholds={"A": 2.0, "B": 1.0},
                num_slots=2,
                eta=0.2,
                controller_input="unlabelled",
                beta=0.5,
                c_min=0.1,
                c_max=50.0,
            )

    def test_groupwise_full_slaclip_betas_back_substitute_stationary_targets(
        self,
    ) -> None:
        epsilon = 1e-6
        settings = {
            "A": {"threshold": 2.0, "q": 0.7, "z": 0.1},
            "B": {"threshold": 1.0, "q": 0.4, "z": 0.25},
        }
        targets = {
            group: (1.0 - values["q"]) / (1.0 - values["z"])
            for group, values in settings.items()
        }
        records = []
        for _ in range(2):
            gradient_groups = {}
            for group, values in settings.items():
                threshold = values["threshold"]
                normalized_endpoint = values["z"] * (threshold + epsilon)
                per_client_scale = threshold / np.sqrt(2.0)
                slack = [
                    values["q"] * per_client_scale,
                    normalized_endpoint * per_client_scale,
                ]
                gradient_groups[group] = {
                    "clip_threshold": threshold,
                    "clipped": False,
                    "slaclip": {
                        "num_slots": 2,
                        "noisy_slack": slack,
                        "slack_signal": slack,
                        "joint_sensitivity_bound_passed": True,
                        "slack_noise_std_per_coordinate": per_client_scale,
                    },
                }
            records.append({"gradient_groups": gradient_groups})

        summary = slaclip_round_controller_summary(
            records,
            clip_thresholds={
                group: values["threshold"]
                for group, values in settings.items()
            },
            num_slots=2,
            eta=0.2,
            base_target_clipped_fraction_by_group=targets,
            c_min=0.1,
            c_max=50.0,
            epsilon=epsilon,
        )
        self.assertEqual(
            summary["variant"],
            "groupwise_generalized_full_slaclip_beta",
        )
        self.assertEqual(summary["target_parameterization"], "per_gradient_group")
        self.assertTrue(summary["generalized_full_slaclip_beta"])
        self.assertEqual(
            summary["base_target_clipped_fraction_by_group"], targets
        )
        self.assertEqual(summary["beta_by_group"], targets)
        self.assertNotIn("base_target_clipped_fraction", summary)
        self.assertNotIn("beta", summary)
        for group, values in settings.items():
            controller = summary[group]
            self.assertAlmostEqual(
                controller["base_target_clipped_fraction"], targets[group]
            )
            self.assertAlmostEqual(controller["beta"], targets[group])
            self.assertAlmostEqual(
                controller["remaining_non_small_gradient_fraction"],
                1.0 - values["z"],
            )
            self.assertAlmostEqual(
                controller["dynamic_target_clipped"],
                1.0 - values["q"],
            )
            self.assertAlmostEqual(
                controller["dynamic_target_unclipped"], values["q"]
            )
            self.assertAlmostEqual(controller["raw_log_step"], 0.0, places=12)
            self.assertAlmostEqual(
                controller["next_clip_threshold"],
                values["threshold"],
                places=12,
            )

    def test_recorded_signal_and_noise_reconstruct_fedavg_update(self) -> None:
        model = TinyLoRA()
        groups = parameter_groups(model)
        before = clone_trainable_state(groups)
        aggregate = empty_state_like(before)
        components = empty_mechanism_components(groups)
        for client_id in range(2):
            restore_trainable_state(groups, before)
            with torch.no_grad():
                for _, parameter in groups["A"]:
                    parameter.grad = torch.tensor(
                        [[3.0 + client_id, 4.0 - client_id]]
                    )
                for _, parameter in groups["B"]:
                    parameter.grad = torch.tensor([[1.0], [2.0 + client_id]])
            clip_noise_and_step(
                groups,
                clip_norm=2.0,
                noise_multiplier=0.25,
                learning_rate=0.1,
                generator=torch.Generator().manual_seed(100 + client_id),
                component_accumulators=components,
                component_weight=0.5,
            )
            accumulate_state(aggregate, groups, 0.5)
        update = round_update_statistics(
            before, aggregate, components, learning_rate=0.1
        )
        for group in ("A", "B"):
            self.assertLess(update[group]["fedavg_reconstruction_residual_l2"], 1e-6)
            self.assertLess(update[group]["fedavg_relative_residual"], 1e-5)

    def test_stateless_model_rng_makes_dropout_resume_stable(self) -> None:
        dropout = nn.Dropout(p=0.5).train()
        values = torch.ones(1024)
        seed_model_stochasticity(12345, torch.device("cpu"))
        uninterrupted = dropout(values)
        # Simulate arbitrary RNG consumption and then a resumed client step.
        torch.rand(5000)
        seed_model_stochasticity(12345, torch.device("cpu"))
        resumed = dropout(values)
        self.assertTrue(torch.equal(uninterrupted, resumed))

    def test_run_lock_rejects_a_concurrent_writer(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            output.mkdir(mode=0o700)
            os.chmod(output, 0o700)
            first = acquire_run_lock(output)
            try:
                with self.assertRaisesRegex(RuntimeError, "another writer"):
                    acquire_run_lock(output)
            finally:
                first.close()

    def test_checkpoint_rejects_non_object_evaluation_entry(self) -> None:
        config = paper_config()
        state = valid_checkpoint_state(config, completed_round=2)
        state["evaluations"] = [None]
        with self.assertRaisesRegex(RuntimeError, "entries must be objects"):
            validate_checkpoint_trainer_state(
                state,
                completed_round=2,
                model_kind="bert",
                config=config,
                run_config_fingerprint="run-fingerprint",
                private_key_commitment="key-commitment",
                rng_domain="rng-domain",
                clients=[np.array([0]), np.array([1])],
            )

    def test_checkpoint_binds_full_slaclip_contract_and_next_thresholds(self) -> None:
        config = replace(paper_config(rounds=2), method="slaclip_dp_lora")
        state = valid_checkpoint_state(config, completed_round=2)
        state["last_round_summary"]["slaclip_controller"] = {
            "A": {"next_clip_threshold": 11.0},
            "B": {"next_clip_threshold": 12.0},
        }
        contract = {
            "schema_version": "full_slaclip_contract_v1",
            "variant": "full_slaclip_cdf_endpoints",
            "controller_input": "noisy_endpoints",
            "controller": {
                "eta": 0.2,
                "beta": 0.5,
                "num_slots": 15,
                "c_min": 0.1,
                "c_max": 50.0,
            },
        }
        state["slaclip_controller_state"] = {
            "controller_contract_sha256": canonical_json_fingerprint(contract),
            "controller_input": "noisy_endpoints",
            "updates_completed": 2,
            "next_clip_threshold_by_group": {"A": 11.0, "B": 12.0},
        }
        validate_checkpoint_trainer_state(
            state,
            completed_round=2,
            model_kind="bert",
            config=config,
            run_config_fingerprint="run-fingerprint",
            private_key_commitment="key-commitment",
            rng_domain="rng-domain",
            clients=[np.array([0]), np.array([1])],
            slaclip_contract=contract,
        )
        corrupted = copy.deepcopy(state)
        corrupted["slaclip_controller_state"][
            "controller_contract_sha256"
        ] = "d" * 64
        with self.assertRaisesRegex(RuntimeError, "controller_contract_sha256"):
            validate_checkpoint_trainer_state(
                corrupted,
                completed_round=2,
                model_kind="bert",
                config=config,
                run_config_fingerprint="run-fingerprint",
                private_key_commitment="key-commitment",
                rng_domain="rng-domain",
                clients=[np.array([0]), np.array([1])],
                slaclip_contract=contract,
            )
        wrong_input = copy.deepcopy(state)
        wrong_input["slaclip_controller_state"]["controller_input"] = (
            "exact_endpoints"
        )
        with self.assertRaisesRegex(RuntimeError, "controller_input"):
            validate_checkpoint_trainer_state(
                wrong_input,
                completed_round=2,
                model_kind="bert",
                config=config,
                run_config_fingerprint="run-fingerprint",
                private_key_commitment="key-commitment",
                rng_domain="rng-domain",
                clients=[np.array([0]), np.array([1])],
                slaclip_contract=contract,
            )

    def test_checkpoint_accepts_explicit_exact_endpoint_oracle_control(self) -> None:
        config = replace(paper_config(rounds=2), method="oracle_slaclip_control")
        state = valid_checkpoint_state(config, completed_round=2)
        state["last_round_summary"]["slaclip_controller"] = {
            "A": {"next_clip_threshold": 4.0},
            "B": {"next_clip_threshold": 5.0},
        }
        contract = {
            "schema_version": "full_slaclip_contract_v1",
            "variant": "full_slaclip_cdf_endpoints",
            "controller_input": "exact_endpoints",
            "non_private_oracle_control": True,
            "controller": {
                "controller_input": "exact_endpoints",
                "eta": 0.05,
                "beta": 0.76,
                "num_slots": 5,
                "c_min": 0.1,
                "c_max": 50.0,
            },
        }
        state["slaclip_controller_state"] = {
            "controller_contract_sha256": canonical_json_fingerprint(contract),
            "controller_input": "exact_endpoints",
            "updates_completed": 2,
            "next_clip_threshold_by_group": {"A": 4.0, "B": 5.0},
        }
        validate_checkpoint_trainer_state(
            state,
            completed_round=2,
            model_kind="bert",
            config=config,
            run_config_fingerprint="run-fingerprint",
            private_key_commitment="key-commitment",
            rng_domain="rng-domain",
            clients=[np.array([0]), np.array([1])],
            slaclip_contract=contract,
        )

    def test_checkpoint_binds_groupwise_full_slaclip_base_targets(self) -> None:
        config = replace(paper_config(rounds=2), method="slaclip_dp_lora")
        state = valid_checkpoint_state(config, completed_round=2)
        targets = {"A": 1.0 / 3.0, "B": 0.8}
        state["last_round_summary"]["slaclip_controller"] = {
            "A": {
                "next_clip_threshold": 3.0,
                "base_target_clipped_fraction": targets["A"],
            },
            "B": {
                "next_clip_threshold": 1.0,
                "base_target_clipped_fraction": targets["B"],
            },
        }
        contract = {
            "schema_version": (
                "groupwise_generalized_full_slaclip_beta_contract_v1"
            ),
            "variant": "groupwise_generalized_full_slaclip_beta",
            "controller_input": "noisy_endpoints",
            "controller": {
                "controller_input": "noisy_endpoints",
                "eta": 0.05,
                "base_target_clipped_fraction_by_group": targets,
                "beta_by_group": targets,
                "num_slots": 5,
                "c_min": 0.1,
                "c_max": 50.0,
            },
        }
        state["slaclip_controller_state"] = {
            "controller_contract_sha256": canonical_json_fingerprint(contract),
            "controller_input": "noisy_endpoints",
            "updates_completed": 2,
            "next_clip_threshold_by_group": {"A": 3.0, "B": 1.0},
            "base_target_clipped_fraction_by_group": targets,
        }
        validate_checkpoint_trainer_state(
            state,
            completed_round=2,
            model_kind="bert",
            config=config,
            run_config_fingerprint="run-fingerprint",
            private_key_commitment="key-commitment",
            rng_domain="rng-domain",
            clients=[np.array([0]), np.array([1])],
            slaclip_contract=contract,
        )

        corrupted = copy.deepcopy(state)
        corrupted["slaclip_controller_state"][
            "base_target_clipped_fraction_by_group"
        ]["A"] = 0.4
        with self.assertRaisesRegex(RuntimeError, "base_target.*by_group"):
            validate_checkpoint_trainer_state(
                corrupted,
                completed_round=2,
                model_kind="bert",
                config=config,
                run_config_fingerprint="run-fingerprint",
                private_key_commitment="key-commitment",
                rng_domain="rng-domain",
                clients=[np.array([0]), np.array([1])],
                slaclip_contract=contract,
            )

    def test_checkpoint_rejects_wrong_evaluation_round_sequence(self) -> None:
        config = paper_config()
        state = valid_checkpoint_state(config, completed_round=2)
        state["evaluations"] = [
            {"round": 0, "loss": 1.0},
            {"round": 2, "loss": 1.0},
        ]
        with self.assertRaisesRegex(RuntimeError, "round sequence is invalid"):
            validate_checkpoint_trainer_state(
                state,
                completed_round=2,
                model_kind="bert",
                config=config,
                run_config_fingerprint="run-fingerprint",
                private_key_commitment="key-commitment",
                rng_domain="rng-domain",
                clients=[np.array([0]), np.array([1])],
            )

    def test_checkpoint_rejects_non_finite_evaluation_loss(self) -> None:
        config = paper_config()
        for invalid_loss in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(loss=invalid_loss):
                state = valid_checkpoint_state(config, completed_round=2)
                state["evaluations"][1]["loss"] = invalid_loss
                with self.assertRaisesRegex(RuntimeError, "loss is invalid"):
                    validate_checkpoint_trainer_state(
                        state,
                        completed_round=2,
                        model_kind="bert",
                        config=config,
                        run_config_fingerprint="run-fingerprint",
                        private_key_commitment="key-commitment",
                        rng_domain="rng-domain",
                        clients=[np.array([0]), np.array([1])],
                    )

    def test_round_shards_reject_unsupported_schema(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            rounds = Path(root) / "rounds"
            rounds.mkdir(mode=0o700)
            shard = rounds / "round-00001.json"
            shard.write_text('{"schema_version": 1}\n', encoding="utf-8")
            os.chmod(shard, 0o600)
            with self.assertRaisesRegex(RuntimeError, "unsupported round shard schema"):
                read_round_shards(
                    rounds,
                    expected_rounds=1,
                    expected_model="bert",
                    expected_method="paper_dp_lora",
                    expected_clients=2,
                    expected_batch_size=1,
                )

    def test_round_shards_reject_symlink_and_wide_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            for case in ("symlink", "wide"):
                with self.subTest(case=case):
                    rounds = root_path / case
                    rounds.mkdir(mode=0o700)
                    shard = rounds / "round-00001.json"
                    if case == "symlink":
                        target = root_path / "target.json"
                        target.write_text("{}\n", encoding="utf-8")
                        os.chmod(target, 0o600)
                        shard.symlink_to(target)
                        expected_error = "not a real regular file"
                    else:
                        shard.write_text("{}\n", encoding="utf-8")
                        os.chmod(shard, 0o644)
                        expected_error = "must have mode 0600"
                    with self.assertRaisesRegex(RuntimeError, expected_error):
                        read_round_shards(
                            rounds,
                            expected_rounds=1,
                            expected_model="bert",
                            expected_method="paper_dp_lora",
                            expected_clients=2,
                            expected_batch_size=1,
                        )

    def test_private_directory_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            target = root_path / "real"
            target.mkdir(mode=0o700)
            link = root_path / "linked"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "not a real directory"):
                validate_private_directory(link, "test private directory")

    def test_input_manifest_requires_inventory_reference_closure(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            manifest = minimal_valid_input_manifest(root_path)

            # Establish that the fixture itself satisfies all pinned-input gates.
            manifest_path = root_path / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validate_input_manifest(manifest_path), manifest)

            unreferenced = root_path / "unreferenced.parquet"
            unreferenced.write_bytes(b"unreferenced")
            invalid = copy.deepcopy(manifest)
            invalid["inventory"].append(
                {
                    "path": str(unreferenced),
                    "bytes": unreferenced.stat().st_size,
                    "sha256": sha256_bytes(unreferenced),
                    "role": "formal_dataset",
                }
            )
            refresh_inventory_metadata(invalid)
            manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "dataset references do not exactly match"
            ):
                validate_input_manifest(manifest_path)

            unhashed = root_path / "unhashed.parquet"
            unhashed.write_bytes(b"unhashed")
            invalid = copy.deepcopy(manifest)
            invalid["formal_dataset"]["combined_splits"]["train"]["files"] = [
                str(unhashed)
            ]
            manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "dataset references do not exactly match"
            ):
                validate_input_manifest(manifest_path)

    def test_holdout_excludes_all_normalized_content_duplicates(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)

            def write_split(name: str, rows: list[tuple[str, str]]) -> Path:
                path = root_path / f"{name}.parquet"
                pq.write_table(
                    pa.table(
                        {
                            "id": list(range(len(rows))),
                            "src": [row[0] for row in rows],
                            "tgt": [row[1] for row in rows],
                        }
                    ),
                    path,
                )
                return path

            train_rows = [("hold one", "answer"), ("train only", "answer")]
            validation_rows = [
                (" hold one ", "answer"),
                ("hold two", "answer"),
            ]
            test_rows = [("hold two", "answer"), ("test only", "answer")]
            paths = {
                "train": write_split("train", train_rows),
                "validation": write_split("validation", validation_rows),
                "test": write_split("test", test_rows),
            }
            manifest = {
                "formal_dataset": {
                    "combined_splits": {
                        split: {"files": [str(paths[split])], "rows": 2}
                        for split in ("train", "validation", "test")
                    }
                }
            }
            config = EffectiveConfig(
                method="paper_dp_lora",
                num_clients=1,
                rounds=1,
                batch_size=1,
                noise_multiplier=2.0,
                learning_rate=5e-4,
                clip_norm=10.0,
                rank=8,
                max_seq_length=16,
                seed=42,
                data_split_seed=1729,
                evaluation_seed=2718,
                max_validation_records=1,
                eval_every=1,
                checkpoint_every=1,
                data_protocol="paper_union_minus_fixed_holdout",
                delta=1e-5,
                pair_noise_across_methods=False,
                smoke=True,
            )
            loaded = load_data_protocol(manifest, config)
            held_keys = set(
                loaded.validation.content_keys(loaded.validation_indices)
            )
            training_keys = set(loaded.training.content_keys(loaded.training_pool))
            self.assertTrue(held_keys.isdisjoint(training_keys))
            self.assertGreaterEqual(
                loaded.protocol["content_excluded_training_rows"], 2
            )
            self.assertFalse(
                loaded.protocol["holdout_content_overlaps_training"]
            )

    def test_adapter_tensor_and_configuration_are_both_validated(self) -> None:
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as root:
            adapter_dir = Path(root) / "adapter"
            adapter_dir.mkdir()
            tensor_path = adapter_dir / "adapter_model.safetensors"
            save_file(
                {
                    "layer.lora_A.weight": torch.ones(2, 3),
                    "layer.lora_B.weight": torch.ones(3, 2),
                },
                tensor_path,
            )
            os.chmod(tensor_path, 0o600)
            snapshot = Path(root) / "snapshot"
            config_path = adapter_dir / "adapter_config.json"
            config_path.write_text(
                "{\n"
                f'  "base_model_name_or_path": "{snapshot}",\n'
                '  "bias": "none",\n'
                '  "lora_alpha": 2,\n'
                '  "lora_dropout": 0.0,\n'
                '  "peft_type": "LORA",\n'
                '  "r": 2,\n'
                '  "target_modules": ["query"]\n'
                "}\n",
                encoding="utf-8",
            )
            os.chmod(config_path, 0o600)
            integrity = validate_adapter_artifact(
                tensor_path,
                expected_parameter_elements=12,
                expected_parameter_tensors=2,
                expected_rank=2,
                expected_target_modules=["query"],
                expected_base_model_path=snapshot,
            )
            self.assertTrue(integrity["config_semantics_verified"])
            payload = config_path.read_text(encoding="utf-8").replace(
                '"r": 2', '"r": 3'
            )
            config_path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch: r"):
                validate_adapter_artifact(
                    tensor_path,
                    expected_parameter_elements=12,
                    expected_parameter_tensors=2,
                    expected_rank=2,
                    expected_target_modules=["query"],
                    expected_base_model_path=snapshot,
                )


if __name__ == "__main__":
    unittest.main()
