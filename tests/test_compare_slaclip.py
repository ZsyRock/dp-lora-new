from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from paper_repro.calibrate_slaclip import (
    atomic_create_calibration,
    build_calibration,
    round_shard_prefix_sha256,
)
from paper_repro.compare_slaclip import (
    ADAPTIVE_METHOD,
    BASELINE_METHOD,
    COMPARISON_STATUS,
    build_comparison,
    main,
)
from paper_repro.reproducibility import METHOD_SPECS, canonical_json_fingerprint


MODELS = ("bert", "gpt2")
GROUPS = ("A", "B")
ROUNDS = 3
CLIENTS = 2
COUNTS = {
    "bert": {"A": [0, 1, 2], "B": [0, 0, 1]},
    "gpt2": {"A": [1, 1, 2], "B": [0, 2, 2]},
}


def private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    os.chmod(path, 0o700)
    return path


def private_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def digest(value: bytes | str) -> str:
    encoded = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def effective_config(method: str) -> dict:
    return {
        "method": method,
        "num_clients": CLIENTS,
        "rounds": ROUNDS,
        "batch_size": 8,
        "noise_multiplier": 2.0,
        "learning_rate": 5e-4,
        "clip_norm": 10.0,
        "rank": 512,
        "max_seq_length": 128,
        "seed": 42,
        "max_validation_records": 8,
        "eval_every": 1,
        "checkpoint_every": 1,
        "data_protocol": "paper_union_minus_fixed_holdout",
        "delta": 1e-5,
        "pair_noise_across_methods": False,
        "smoke": False,
    }


def scientific_contract(method: str, slaclip_contract: dict | None) -> dict:
    return {
        "schema_version": 2,
        "repository_sha": "a" * 40,
        "input_manifest_sha256": digest("manifest"),
        "input_inventory_sha256": digest("inventory"),
        "dataset": {"repo_id": "dataset", "revision": "b" * 40},
        "data_protocol": "paper_union_minus_fixed_holdout",
        "method": asdict(METHOD_SPECS[method]),
        "effective_config": effective_config(method),
        "models": list(MODELS),
        "model_revisions": {"bert": "pinned", "gpt2": "pinned"},
        "private_key_commitment": digest("private-key"),
        "rng_domain": "matched-pair",
        "dependency_versions": {"torch": "2.7.1"},
        "execution_backend": {"deterministic_algorithms": True},
        "algorithm_contract": {
            "federated_clients_simulated_sequentially": True,
            "client_weighting": "equal_one_over_K",
            "gradient_grouping": "aggregate_batch_gradient_separate_A_and_B",
            "local_optimizer": "one_manual_sgd_step",
            "record_weighting": "equal_records_after_within_record_token_mean",
            "dropout_rng": "stateless_private_hmac_per_model_round_client",
            "contains_slaclip": slaclip_contract is not None,
            "slaclip_q": slaclip_contract,
        },
    }


def write_adapter(model_dir: Path, method: str, model: str) -> tuple[str, str]:
    adapter_dir = private_directory(model_dir / "final_adapter")
    adapter = f"adapter:{method}:{model}".encode()
    config = b'{"peft_type":"LORA"}\n'
    adapter_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    adapter_path.write_bytes(adapter)
    config_path.write_bytes(config)
    os.chmod(adapter_path, 0o600)
    os.chmod(config_path, 0o600)
    return digest(adapter), digest(config)


def run_config(method: str, contract: dict, *, adaptive: bool) -> dict:
    fingerprint = canonical_json_fingerprint(contract)
    return {
        "schema_version": 2,
        "method": method,
        "method_spec": asdict(METHOD_SPECS[method]),
        "contains_slaclip": adaptive,
        "run_config_fingerprint": fingerprint,
        "scientific_contract": contract,
        "effective_config": contract["effective_config"],
        "models": list(MODELS),
        "reproduction_claim": {
            "level": 1,
            "name": "algorithm_execution_reconstruction",
            "paper_result_reproduced": False,
            "paper_benchmarks_evaluated": False,
        },
        "privacy_claim": {
            "end_to_end_dp_certified": False,
            "epsilon": None,
            "sigma_is_not_epsilon": True,
            "diagnostics_are_private_non_dp_data": True,
            "baseline_derived_calibration_is_non_dp": adaptive,
        },
    }


def base_model_summary(
    *,
    method: str,
    model: str,
    fingerprint: str,
    prefix: str,
    adapter_sha: str,
    adapter_config_sha: str,
    behavior: dict,
) -> dict:
    total = ROUNDS * CLIENTS
    clipping = {
        group: {
            "count": sum(COUNTS[model][group]),
            "fraction": sum(COUNTS[model][group]) / total,
        }
        for group in GROUPS
    }
    clipping["any_group"] = {"count": 0, "fraction": 0.0}
    final_eval = {"round": ROUNDS, "loss": 3.5}
    return {
        "schema_version": 2,
        "status": "COMPLETED",
        "method": method,
        "model": model,
        "privacy_claim": False,
        "run_config_fingerprint": fingerprint,
        "client_steps": total,
        "client_partition_sha256": [
            digest(f"{model}:partition:{client}") for client in range(CLIENTS)
        ],
        "clipping": clipping,
        "behavior_summary": behavior,
        "round_shard_prefix_sha256": prefix,
        "evaluations": [{"round": 0, "loss": 4.0}, final_eval],
        "final_evaluation": final_eval,
        "adapter_sha256": adapter_sha,
        "adapter_config_sha256": adapter_config_sha,
        "adapter_state_sha256": digest(f"{method}:{model}:state"),
        "adapter_integrity": {"all_finite": True},
        "privacy_accounting": {
            "status": "NOT_CERTIFIED",
            "epsilon": None,
            "sigma_is_not_epsilon": True,
        },
    }


def write_baseline(parent: Path) -> Path:
    baseline = private_directory(parent / "baseline")
    contract = scientific_contract(BASELINE_METHOD, None)
    config = run_config(BASELINE_METHOD, contract, adaptive=False)
    private_json(baseline / "run_config.json", config)
    fingerprint = config["run_config_fingerprint"]
    root_models: dict[str, dict] = {}
    schedules: dict[str, str] = {}
    for model in MODELS:
        model_dir = private_directory(baseline / model)
        diagnostics = private_directory(model_dir / "private_diagnostics")
        rounds_dir = private_directory(diagnostics / "rounds")
        for round_index in range(1, ROUNDS + 1):
            counts = {
                group: COUNTS[model][group][round_index - 1] for group in GROUPS
            }
            records = [
                {
                    "method": BASELINE_METHOD,
                    "model": model,
                    "round": round_index,
                    "client": client,
                    "gradient_groups": {
                        group: {"clipped": client < counts[group]}
                        for group in GROUPS
                    },
                }
                for client in range(CLIENTS)
            ]
            private_json(
                rounds_dir / f"round-{round_index:05d}.json",
                {
                    "schema_version": 2,
                    "method": BASELINE_METHOD,
                    "model": model,
                    "round": round_index,
                    "client_records": records,
                    "round_summary": {
                        "round": round_index,
                        "clients": CLIENTS,
                        **{
                            group: {
                                "clipped_count": counts[group],
                                "clipped_fraction": counts[group] / CLIENTS,
                            }
                            for group in GROUPS
                        },
                    },
                },
            )
        prefix = round_shard_prefix_sha256(rounds_dir, ROUNDS)
        adapter_sha, adapter_config_sha = write_adapter(
            model_dir, BASELINE_METHOD, model
        )
        schedule = digest(f"{model}:shared-samples")
        schedules[model] = schedule
        behavior = {
            "sample_schedule_sha256": schedule,
            "supervision_schedule_sha256": digest(f"{model}:shared-masks"),
            "groups": {
                group: {
                    "actual_clipped_count": sum(COUNTS[model][group]),
                    "actual_clipped_fraction": sum(COUNTS[model][group])
                    / (ROUNDS * CLIENTS),
                }
                for group in GROUPS
            },
        }
        summary = base_model_summary(
            method=BASELINE_METHOD,
            model=model,
            fingerprint=fingerprint,
            prefix=prefix,
            adapter_sha=adapter_sha,
            adapter_config_sha=adapter_config_sha,
            behavior=behavior,
        )
        private_json(model_dir / "final_summary.json", summary)
        root_models[model] = summary
    private_json(
        baseline / "final_summary.json",
        {
            "schema_version": 2,
            "status": "COMPLETED",
            "method": BASELINE_METHOD,
            "method_spec": asdict(METHOD_SPECS[BASELINE_METHOD]),
            "contains_slaclip": False,
            "run_config_fingerprint": fingerprint,
            "reproduction_claim_level": 1,
            "paper_result_reproduced": False,
            "paper_benchmarks_evaluated": False,
            "privacy_claim": False,
            "models": root_models,
            "sample_schedule_sha256_by_model": schedules,
        },
    )
    return baseline


def calibration_contract(calibration: dict, calibration_path: Path) -> dict:
    targets = {
        model: {
            group: calibration["models"][model]["groups"][group][
                "target_clip_fraction"
            ]
            for group in GROUPS
        }
        for model in MODELS
    }
    return {
        "variant": "SlaClip-Q_fixed_target",
        "federated_adaptation": (
            "per_client_joint_gradient_slack_release_then_equal_fedavg_"
            "and_one_controller_update_per_round"
        ),
        "reference": {
            "paper": "https://openreview.net/pdf?id=48suUeYKdb",
            "repository": "https://example.invalid/slaclip",
            "revision": "c" * 40,
        },
        "controller": {
            "eta": 0.2,
            "num_slots": 1,
            "num_slots_selection": "automatic_monotonicity_rule",
            "expected_release_records": CLIENTS,
            "c_min": 0.1,
            "c_max": 50.0,
            "initial_clip_threshold": 10.0,
            "target_semantics": "clipped_fraction_complemented_to_unclipped_proxy",
        },
        "calibration": {
            "privacy_class": calibration["privacy_class"],
            "calibration_fingerprint": calibration["calibration_fingerprint"],
            "file_sha256": digest(calibration_path.read_bytes()),
            "source": calibration["source"],
            "reducer": calibration["reducer"],
            "targets": targets,
        },
        "independently_privacy_certified": False,
    }


def controller_round(targets: dict[str, float]) -> dict:
    value: dict = {
        "variant": "SlaClip-Q",
        "update_timing": "once_after_all_clients_for_use_in_next_round",
        "clients": CLIENTS,
        "num_slots": 1,
        "eta": 0.2,
        "c_min": 0.1,
        "c_max": 50.0,
    }
    for group in GROUPS:
        target = targets[group]
        value[group] = {
            "clip_threshold_used": 10.0,
            "target_clip_fraction": target,
            "target_clipped_fraction": target,
            "target_unclipped_proxy": 1.0 - target,
            "noisy_unclipped_proxy_by_slot": [1.0 - target],
            "exact_unclipped_proxy_by_slot": [1.0 - target],
            "controller_error": 0.0,
            "actual_clip_fraction": target,
            "actual_minus_target_clip_fraction": 0.0,
            "noisy_unclipped_proxy": 1.0 - target,
            "eta": 0.2,
            "raw_log_update": 0.0,
            "bounded_log_update": 0.0,
            "log_update_was_clamped": False,
            "unbounded_next_clip_threshold": 10.0,
            "next_clip_threshold": 10.0,
            "c_min": 0.1,
            "c_max": 50.0,
            "hit_lower_bound": False,
            "hit_upper_bound": False,
        }
    return value


def write_adaptive(
    parent: Path, baseline: Path, calibration_path: Path, calibration: dict
) -> Path:
    del baseline  # Source binding is carried by the immutable calibration contract.
    adaptive = private_directory(parent / "adaptive")
    slaclip = calibration_contract(calibration, calibration_path)
    contract = scientific_contract(ADAPTIVE_METHOD, slaclip)
    config = run_config(ADAPTIVE_METHOD, contract, adaptive=True)
    private_json(adaptive / "run_config.json", config)
    fingerprint = config["run_config_fingerprint"]
    root_models: dict[str, dict] = {}
    schedules: dict[str, str] = {}
    for model in MODELS:
        model_dir = private_directory(adaptive / model)
        diagnostics = private_directory(model_dir / "private_diagnostics")
        rounds_dir = private_directory(diagnostics / "rounds")
        targets = slaclip["calibration"]["targets"][model]
        trajectory = []
        for round_index in range(1, ROUNDS + 1):
            controller = controller_round(targets)
            trajectory.append(controller)
            private_json(
                rounds_dir / f"round-{round_index:05d}.json",
                {
                    "schema_version": 2,
                    "method": ADAPTIVE_METHOD,
                    "model": model,
                    "round": round_index,
                    "round_summary": {
                        "round": round_index,
                        "slaclip_controller": controller,
                    },
                },
            )
        prefix = round_shard_prefix_sha256(rounds_dir, ROUNDS)
        adapter_sha, adapter_config_sha = write_adapter(
            model_dir, ADAPTIVE_METHOD, model
        )
        schedule = digest(f"{model}:shared-samples")
        schedules[model] = schedule
        controller_summary = {
            "variant": "SlaClip-Q",
            "rounds": ROUNDS,
            "trajectory_sha256": canonical_json_fingerprint(trajectory),
            "groups": {
                group: {
                    "target_clip_fraction": targets[group],
                    "final_next_clip_threshold": 10.0,
                }
                for group in GROUPS
            },
        }
        behavior = {
            "sample_schedule_sha256": schedule,
            "supervision_schedule_sha256": digest(f"{model}:shared-masks"),
            "groups": {},
            "slaclip_controller": controller_summary,
        }
        summary = base_model_summary(
            method=ADAPTIVE_METHOD,
            model=model,
            fingerprint=fingerprint,
            prefix=prefix,
            adapter_sha=adapter_sha,
            adapter_config_sha=adapter_config_sha,
            behavior=behavior,
        )
        summary["slaclip_q"] = {
            "contract": slaclip,
            "target_clip_fraction_by_group": targets,
            "final_next_clip_threshold_by_group": {group: 10.0 for group in GROUPS},
            "controller_summary": controller_summary,
        }
        private_json(model_dir / "final_summary.json", summary)
        root_models[model] = summary
    private_json(
        adaptive / "final_summary.json",
        {
            "schema_version": 2,
            "status": "COMPLETED",
            "method": ADAPTIVE_METHOD,
            "method_spec": asdict(METHOD_SPECS[ADAPTIVE_METHOD]),
            "contains_slaclip": True,
            "run_config_fingerprint": fingerprint,
            "reproduction_claim_level": 1,
            "paper_result_reproduced": False,
            "paper_benchmarks_evaluated": False,
            "privacy_claim": False,
            "models": root_models,
            "sample_schedule_sha256_by_model": schedules,
        },
    )
    return adaptive


def build_fixture(parent: Path) -> tuple[Path, Path, Path]:
    baseline = write_baseline(parent)
    calibration_dir = private_directory(parent / "calibration")
    calibration_path = calibration_dir / "median.json"
    calibration = build_calibration(baseline)
    atomic_create_calibration(calibration_path, calibration)
    adaptive = write_adaptive(parent, baseline, calibration_path, calibration)
    return baseline, adaptive, calibration_path


def rewrite_adaptive_model_and_root(adaptive: Path, model: str, summary: dict) -> None:
    private_json(adaptive / model / "final_summary.json", summary)
    root = load_json(adaptive / "final_summary.json")
    root["models"][model] = summary
    root["sample_schedule_sha256_by_model"][model] = summary["behavior_summary"][
        "sample_schedule_sha256"
    ]
    private_json(adaptive / "final_summary.json", root)


class CompareSlaClipTests(unittest.TestCase):
    def test_valid_pair_is_matched_level_one_and_explicitly_non_dp(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            baseline, adaptive, calibration = build_fixture(root)
            result = build_comparison(baseline, adaptive, calibration)
            self.assertEqual(result["status"], COMPARISON_STATUS)
            self.assertEqual(result["claim_level"], 1)
            self.assertTrue(result["contains_slaclip"])
            self.assertFalse(result["paper_result_reproduced"])
            self.assertIsNone(result["privacy_notice"]["epsilon"])
            self.assertFalse(
                result["privacy_notice"]["end_to_end_dp_certified"]
            )
            self.assertTrue(
                result["privacy_notice"][
                    "data_dependent_baseline_calibration_is_non_dp"
                ]
            )
            self.assertEqual(
                result["models"]["bert"]["controller"]["rounds"], ROUNDS
            )
            self.assertEqual(
                result["models"]["bert"]["controller"][
                    "initial_clip_threshold"
                ],
                10.0,
            )

    def test_cli_create_and_verify_existing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            baseline, adaptive, calibration = build_fixture(root)
            output = root / "comparison.json"
            arguments = [
                "--baseline-dir",
                str(baseline),
                "--adaptive-dir",
                str(adaptive),
                "--calibration",
                str(calibration),
                "--output",
                str(output),
            ]
            main(arguments)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            main([*arguments, "--verify-existing"])
            with self.assertRaises(SystemExit):
                main(arguments)

    def test_schedule_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            baseline, adaptive, calibration = build_fixture(root)
            path = adaptive / "bert" / "final_summary.json"
            summary = load_json(path)
            summary["behavior_summary"]["sample_schedule_sha256"] = digest(
                "tampered schedule"
            )
            rewrite_adaptive_model_and_root(adaptive, "bert", summary)
            with self.assertRaisesRegex(RuntimeError, "sample schedules do not match"):
                build_comparison(baseline, adaptive, calibration)

    def test_calibration_file_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            baseline, adaptive, calibration = build_fixture(root)
            value = load_json(calibration)
            value["models"]["bert"]["groups"]["A"][
                "target_clip_fraction"
            ] = 0.9
            private_json(calibration, value)
            with self.assertRaisesRegex(RuntimeError, "median mismatch"):
                build_comparison(baseline, adaptive, calibration)

    def test_adaptive_target_contract_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            baseline, adaptive, calibration = build_fixture(root)
            config_path = adaptive / "run_config.json"
            config = load_json(config_path)
            contract = config["scientific_contract"]
            slaclip = contract["algorithm_contract"]["slaclip_q"]
            slaclip["calibration"]["targets"]["bert"]["A"] = 0.9
            fingerprint = canonical_json_fingerprint(contract)
            config["run_config_fingerprint"] = fingerprint
            private_json(config_path, config)

            root_summary = load_json(adaptive / "final_summary.json")
            root_summary["run_config_fingerprint"] = fingerprint
            for model in MODELS:
                summary = load_json(adaptive / model / "final_summary.json")
                summary["run_config_fingerprint"] = fingerprint
                summary["slaclip_q"]["contract"] = copy.deepcopy(slaclip)
                private_json(adaptive / model / "final_summary.json", summary)
                root_summary["models"][model] = summary
            private_json(adaptive / "final_summary.json", root_summary)
            with self.assertRaisesRegex(RuntimeError, "calibration contract"):
                build_comparison(baseline, adaptive, calibration)

    def test_controller_target_or_bounds_tampering_fails_closed(self) -> None:
        mutations = (
            ("target_clip_fraction", 0.99, "controller target mismatch"),
            ("next_clip_threshold", 100.0, "SlaClip-Q formula mismatch"),
            ("raw_log_update", 0.5, "SlaClip-Q formula mismatch"),
        )
        for field, value, expected in mutations:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as raw_root:
                    root = Path(raw_root)
                    baseline, adaptive, calibration = build_fixture(root)
                    shard_path = (
                        adaptive
                        / "bert"
                        / "private_diagnostics"
                        / "rounds"
                        / "round-00001.json"
                    )
                    shard = load_json(shard_path)
                    shard["round_summary"]["slaclip_controller"]["A"][field] = value
                    private_json(shard_path, shard)
                    with self.assertRaisesRegex(RuntimeError, expected):
                        build_comparison(baseline, adaptive, calibration)

    def test_adaptive_adapter_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            baseline, adaptive, calibration = build_fixture(root)
            adapter = (
                adaptive / "gpt2" / "final_adapter" / "adapter_model.safetensors"
            )
            adapter.write_bytes(b"tampered")
            os.chmod(adapter, 0o600)
            with self.assertRaisesRegex(RuntimeError, "adapter checksum mismatch"):
                build_comparison(baseline, adaptive, calibration)


if __name__ == "__main__":
    unittest.main()
