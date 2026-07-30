from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from paper_repro.compare_slaclip import (
    ADAPTIVE_METHOD,
    BASELINE_METHOD,
    COMPARISON_STATUS,
    SLACLIP_CONTRACT_SCHEMA,
    SLACLIP_VARIANT,
    _independent_full_slaclip_update,
    _independent_normalize_noisy_slack,
    _round_shard_prefix_sha256,
    _validate_exact_cdf_range,
    build_comparison,
    main,
)
from paper_repro.reproducibility import (
    METHOD_SPECS,
    canonical_json_fingerprint,
    safe_quantiles,
)
from paper_repro.slaclip import (
    automatic_num_slots,
    full_slaclip_update,
    normalize_noisy_slack,
)


MODELS = ("bert", "gpt2")
GROUPS = ("A", "B")
ROUNDS = 3
CLIENTS = 2
NUM_SLOTS = 3
NOISE_MULTIPLIER = 2.0
ETA = 0.2
BETA = 0.5
EPSILON = 1e-6
C_MIN = 0.1
C_MAX = 50.0
INITIAL_C = 10.0
NOISY_CDF = {
    "A": [0.72, 0.51, 0.28],
    "B": [0.81, 0.62, 0.19],
}
EXACT_CDF = {
    "A": [0.75, 0.50, 0.25],
    "B": [1.00, 0.75, 0.25],
}
ACTUAL_CLIP = {"A": 0.5, "B": 0.5}


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
        "noise_multiplier": NOISE_MULTIPLIER,
        "learning_rate": 5e-4,
        "clip_norm": INITIAL_C,
        "rank": 512,
        "max_seq_length": 128,
        "seed": 42,
        "max_validation_records": 8,
        "eval_every": 1,
        "checkpoint_every": 1,
        "data_protocol": "paper_union_minus_fixed_holdout",
        "delta": 1e-5,
        "pair_noise_across_methods": True,
        "smoke": False,
    }


def slaclip_contract() -> dict:
    automatic_slots = automatic_num_slots(CLIENTS, NOISE_MULTIPLIER)
    return {
        "schema_version": SLACLIP_CONTRACT_SCHEMA,
        "variant": SLACLIP_VARIANT,
        "federated_adaptation": (
            "per_client_joint_gradient_slack_release_then_equal_fedavg_"
            "and_one_controller_update_per_round"
        ),
        "reference": {
            "paper": "https://openreview.net/pdf?id=48suUeYKdb",
            "repository": "https://github.com/ZsyRock/SlaClip",
            "revision": "c" * 40,
        },
        "controller": {
            "eta": ETA,
            "beta": BETA,
            "epsilon": EPSILON,
            "num_slots": NUM_SLOTS,
            "num_slots_selection": "explicit",
            "local_batch_size": 8,
            "num_clients": CLIENTS,
            "expected_release_records": CLIENTS,
            "automatic_release_num_slots": automatic_slots,
            "explicit_num_slots_exceeds_automatic_release_bound": (
                NUM_SLOTS > automatic_slots
            ),
            "normalized_proxy_noise_std_per_slot_theoretical": (
                NOISE_MULTIPLIER * math.sqrt(NUM_SLOTS / CLIENTS)
            ),
            "normalized_proxy_noise_std_formula": (
                "noise_multiplier*sqrt(num_slots/num_clients)"
            ),
            "near_threshold_index": 0,
            "near_zero_index": NUM_SLOTS - 1,
            "numerical_log_step_bounds": [-50.0, 50.0],
            "numerical_safeguard": "recorded test safeguard",
            "c_min": C_MIN,
            "c_max": C_MAX,
            "initial_clip_threshold": INITIAL_C,
            "controller_inputs": "noisy_joint_release_endpoints_only",
        },
        "exact_cdf_and_clipping_diagnostics": "NON_DP_PRIVATE_DIAGNOSTICS",
        "independently_privacy_certified": False,
    }


def scientific_contract(method: str, adaptive_contract: dict | None) -> dict:
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
            "contains_slaclip": adaptive_contract is not None,
            "slaclip": adaptive_contract,
        },
    }


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
            "paper_result_reproduced": False,
            "paper_benchmarks_evaluated": False,
        },
        "privacy_claim": {
            "end_to_end_dp_certified": False,
            "epsilon": None,
            "sigma_is_not_epsilon": True,
            "diagnostics_are_private_non_dp_data": True,
            "baseline_derived_calibration_is_non_dp": False,
            "exact_cdf_diagnostics_are_non_dp": adaptive,
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
    final_eval = {"round": ROUNDS, "loss": 3.5}
    return {
        "schema_version": 2,
        "status": "COMPLETED",
        "method": method,
        "model": model,
        "privacy_claim": False,
        "run_config_fingerprint": fingerprint,
        "client_steps": ROUNDS * CLIENTS,
        "client_partition_sha256": [
            digest(f"{model}:partition:{client}") for client in range(CLIENTS)
        ],
        "clipping": {
            "A": {"count": 1, "fraction": 1 / (ROUNDS * CLIENTS)},
            "B": {"count": 2, "fraction": 2 / (ROUNDS * CLIENTS)},
            "any_group": {"count": 2, "fraction": 2 / (ROUNDS * CLIENTS)},
        },
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


def write_root(
    directory: Path,
    method: str,
    fingerprint: str,
    models: dict[str, dict],
    schedules: dict[str, str],
    *,
    adaptive: bool,
) -> None:
    private_json(
        directory / "final_summary.json",
        {
            "schema_version": 2,
            "status": "COMPLETED",
            "method": method,
            "method_spec": asdict(METHOD_SPECS[method]),
            "contains_slaclip": adaptive,
            "run_config_fingerprint": fingerprint,
            "reproduction_claim_level": 1,
            "paper_result_reproduced": False,
            "paper_benchmarks_evaluated": False,
            "privacy_claim": False,
            "models": models,
            "sample_schedule_sha256_by_model": schedules,
        },
    )


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
            private_json(
                rounds_dir / f"round-{round_index:05d}.json",
                {
                    "schema_version": 2,
                    "method": BASELINE_METHOD,
                    "model": model,
                    "round": round_index,
                    "round_summary": {"round": round_index},
                },
            )
        prefix = _round_shard_prefix_sha256(rounds_dir, ROUNDS)
        adapter_sha, adapter_config_sha = write_adapter(
            model_dir, BASELINE_METHOD, model
        )
        schedule = digest(f"{model}:shared-samples")
        schedules[model] = schedule
        behavior = {
            "sample_schedule_sha256": schedule,
            "supervision_schedule_sha256": digest(f"{model}:shared-masks"),
            "groups": {},
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
    write_root(
        baseline,
        BASELINE_METHOD,
        fingerprint,
        root_models,
        schedules,
        adaptive=False,
    )
    return baseline


def client_release_records(thresholds: dict[str, float]) -> list[dict]:
    records = []
    for client in range(CLIENTS):
        groups = {}
        for group in GROUPS:
            threshold = thresholds[group]
            slack_lambda = threshold / math.sqrt(NUM_SLOTS)
            groups[group] = {
                "clip_threshold": threshold,
                "clipped": client < round(ACTUAL_CLIP[group] * CLIENTS),
                "slaclip": {
                    "variant": SLACLIP_VARIANT,
                    "num_slots": NUM_SLOTS,
                    "slack_signal": [
                        value * slack_lambda for value in EXACT_CDF[group]
                    ],
                    "noisy_slack": [
                        value * slack_lambda for value in NOISY_CDF[group]
                    ],
                    "joint_sensitivity_bound_passed": True,
                    "slack_noise_std_per_coordinate": (
                        NOISE_MULTIPLIER * threshold
                    ),
                },
            }
        records.append({"client": client, "gradient_groups": groups})
    return records


def controller_round(
    thresholds: dict[str, float], client_records: list[dict]
) -> dict:
    value: dict = {
        "variant": SLACLIP_VARIANT,
        "update_timing": "once_after_all_clients_for_use_in_next_round",
        "clients": CLIENTS,
        "num_slots": NUM_SLOTS,
        "eta": ETA,
        "beta": BETA,
        "epsilon": EPSILON,
        "near_threshold_index": 0,
        "near_zero_index": NUM_SLOTS - 1,
        "c_min": C_MIN,
        "c_max": C_MAX,
    }
    for group in GROUPS:
        noisy_sum = [
            math.fsum(
                record["gradient_groups"][group]["slaclip"]["noisy_slack"][slot]
                for record in client_records
            )
            for slot in range(NUM_SLOTS)
        ]
        exact_sum = [
            math.fsum(
                record["gradient_groups"][group]["slaclip"]["slack_signal"][slot]
                for record in client_records
            )
            for slot in range(NUM_SLOTS)
        ]
        noisy = list(
            normalize_noisy_slack(
                noisy_sum, thresholds[group], NUM_SLOTS, CLIENTS
            )
        )
        exact = list(
            normalize_noisy_slack(
                exact_sum, thresholds[group], NUM_SLOTS, CLIENTS
            )
        )
        update = full_slaclip_update(
            thresholds[group],
            noisy[0],
            noisy[-1],
            beta=BETA,
            eta=ETA,
            min_clip_norm=C_MIN,
            max_clip_norm=C_MAX,
            epsilon=EPSILON,
        )
        update_fields = dict(update)
        update_fields["unbounded_next_clip_threshold"] = update_fields.pop(
            "unbounded_next_clip_norm"
        )
        update_fields["next_clip_threshold"] = update_fields.pop("next_clip_norm")
        update_fields["c_min"] = update_fields.pop("min_clip_norm")
        update_fields["c_max"] = update_fields.pop("max_clip_norm")
        update_fields.pop("current_clip_norm")
        oracle_update = full_slaclip_update(
            thresholds[group],
            exact[0],
            exact[-1],
            beta=BETA,
            eta=ETA,
            min_clip_norm=C_MIN,
            max_clip_norm=C_MAX,
            epsilon=EPSILON,
        )
        cdf_errors = [
            noisy_value - exact_value
            for noisy_value, exact_value in zip(noisy, exact)
        ]
        cdf_error_mae = math.fsum(abs(value) for value in cdf_errors) / NUM_SLOTS
        cdf_error_rmse = math.sqrt(
            math.fsum(value * value for value in cdf_errors) / NUM_SLOTS
        )
        normalized_noise = NOISE_MULTIPLIER * math.sqrt(NUM_SLOTS / CLIENTS)
        noisy_log_step = float(update_fields["raw_log_step"])
        oracle_log_step = float(oracle_update["raw_log_step"])

        def direction(value: float) -> int:
            return -1 if value < 0.0 else (1 if value > 0.0 else 0)

        actual = ACTUAL_CLIP[group]
        value[group] = {
            "clip_threshold_used": thresholds[group],
            "noisy_cdf_proxy_by_slot": noisy,
            "exact_cdf_proxy_by_slot": exact,
            "normalized_proxy_noise_std_per_slot": normalized_noise,
            "cdf_error_mae": cdf_error_mae,
            "cdf_error_rmse": cdf_error_rmse,
            "cdf_error_max_abs": max(abs(value) for value in cdf_errors),
            "cdf_error_z_rmse": cdf_error_rmse / normalized_noise,
            "noisy_cdf_out_of_range_count": sum(
                not 0.0 <= value <= 1.0 for value in noisy
            ),
            "noisy_cdf_out_of_range_fraction": (
                sum(not 0.0 <= value <= 1.0 for value in noisy) / NUM_SLOTS
            ),
            "noisy_near_threshold_minus_exact": noisy[0] - exact[0],
            "noisy_near_zero_minus_exact": noisy[-1] - exact[-1],
            "noisy_adjacent_monotonicity_violations": sum(
                noisy[index + 1] > noisy[index]
                for index in range(NUM_SLOTS - 1)
            ),
            "exact_adjacent_monotonicity_violations": sum(
                exact[index + 1] > exact[index]
                for index in range(NUM_SLOTS - 1)
            ),
            "actual_clip_fraction": actual,
            "actual_minus_dynamic_target_clipped": (
                actual - update_fields["dynamic_target_clipped"]
            ),
            "actual_target_absolute_error": abs(
                actual - update_fields["dynamic_target_clipped"]
            ),
            "oracle_dynamic_target_clipped": oracle_update[
                "dynamic_target_clipped"
            ],
            "oracle_raw_log_step": oracle_log_step,
            "oracle_next_clip_threshold": oracle_update["next_clip_norm"],
            "noisy_minus_oracle_raw_log_step": (
                noisy_log_step - oracle_log_step
            ),
            "noisy_oracle_log_threshold_error": math.log(
                update_fields["next_clip_threshold"]
                / oracle_update["next_clip_norm"]
            ),
            "update_direction_agrees": (
                direction(noisy_log_step) == direction(oracle_log_step)
            ),
            **update_fields,
        }
    return value


def controller_group_summary(
    trajectory: list[dict], group: str, final_threshold: float
) -> dict:
    rounds = [value[group] for value in trajectory]
    quantile_fields = (
        "actual_target_absolute_error",
        "cdf_error_mae",
        "cdf_error_rmse",
        "cdf_error_max_abs",
        "cdf_error_z_rmse",
        "oracle_dynamic_target_clipped",
        "oracle_raw_log_step",
        "oracle_next_clip_threshold",
        "noisy_minus_oracle_raw_log_step",
        "noisy_oracle_log_threshold_error",
    )
    raw_steps = [float(value["raw_log_step"]) for value in rounds]
    nonzero_directions = [
        -1 if value < 0.0 else 1 for value in raw_steps if value != 0.0
    ]
    agreement_count = sum(
        bool(value["update_direction_agrees"]) for value in rounds
    )
    out_of_range_count = sum(
        int(value["noisy_cdf_out_of_range_count"]) for value in rounds
    )
    return {
        **{
            key: safe_quantiles(value[key] for value in rounds)
            for key in quantile_fields
        },
        "log_threshold_total_variation": math.fsum(
            abs(
                math.log(
                    value["next_clip_threshold"]
                    / value["clip_threshold_used"]
                )
            )
            for value in rounds
        ),
        "log_step_direction_flip_count": sum(
            left != right
            for left, right in zip(nonzero_directions, nonzero_directions[1:])
        ),
        "oracle_direction_agreement_count": agreement_count,
        "oracle_direction_agreement_fraction": agreement_count / len(rounds),
        "noisy_cdf_out_of_range_count": out_of_range_count,
        "noisy_cdf_out_of_range_fraction": (
            out_of_range_count / (len(rounds) * NUM_SLOTS)
        ),
        "final_next_clip_threshold": final_threshold,
    }


def write_adaptive(parent: Path) -> Path:
    adaptive = private_directory(parent / "adaptive")
    slaclip = slaclip_contract()
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
        thresholds = {group: INITIAL_C for group in GROUPS}
        trajectory = []
        for round_index in range(1, ROUNDS + 1):
            client_records = client_release_records(thresholds)
            controller = controller_round(thresholds, client_records)
            trajectory.append(controller)
            private_json(
                rounds_dir / f"round-{round_index:05d}.json",
                {
                    "schema_version": 2,
                    "method": ADAPTIVE_METHOD,
                    "model": model,
                    "round": round_index,
                    "client_records": client_records,
                    "round_summary": {
                        "round": round_index,
                        "slaclip_controller": controller,
                    },
                },
            )
            thresholds = {
                group: controller[group]["next_clip_threshold"] for group in GROUPS
            }
        prefix = _round_shard_prefix_sha256(rounds_dir, ROUNDS)
        adapter_sha, adapter_config_sha = write_adapter(
            model_dir, ADAPTIVE_METHOD, model
        )
        schedule = digest(f"{model}:shared-samples")
        schedules[model] = schedule
        controller_summary = {
            "variant": SLACLIP_VARIANT,
            "rounds": ROUNDS,
            "trajectory_sha256": canonical_json_fingerprint(trajectory),
            "groups": {
                group: controller_group_summary(
                    trajectory, group, thresholds[group]
                )
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
        summary["slaclip"] = {
            "contract": slaclip,
            "final_next_clip_threshold_by_group": thresholds,
            "controller_summary": controller_summary,
        }
        private_json(model_dir / "final_summary.json", summary)
        root_models[model] = summary
    write_root(
        adaptive,
        ADAPTIVE_METHOD,
        fingerprint,
        root_models,
        schedules,
        adaptive=True,
    )
    return adaptive


def build_fixture(parent: Path) -> tuple[Path, Path]:
    return write_baseline(parent), write_adaptive(parent)


def rewrite_adaptive_model_and_root(adaptive: Path, model: str, summary: dict) -> None:
    private_json(adaptive / model / "final_summary.json", summary)
    root = load_json(adaptive / "final_summary.json")
    root["models"][model] = summary
    root["sample_schedule_sha256_by_model"][model] = summary["behavior_summary"][
        "sample_schedule_sha256"
    ]
    private_json(adaptive / "final_summary.json", root)


def rewrite_adaptive_contract(adaptive: Path, contract: dict) -> None:
    config_path = adaptive / "run_config.json"
    config = load_json(config_path)
    scientific = config["scientific_contract"]
    scientific["algorithm_contract"]["slaclip"] = contract
    fingerprint = canonical_json_fingerprint(scientific)
    config["run_config_fingerprint"] = fingerprint
    private_json(config_path, config)
    root = load_json(adaptive / "final_summary.json")
    root["run_config_fingerprint"] = fingerprint
    for model in MODELS:
        summary = load_json(adaptive / model / "final_summary.json")
        summary["run_config_fingerprint"] = fingerprint
        summary["slaclip"]["contract"] = contract
        private_json(adaptive / model / "final_summary.json", summary)
        root["models"][model] = summary
    private_json(adaptive / "final_summary.json", root)


class CompareSlaClipTests(unittest.TestCase):
    def test_independent_controller_math_matches_golden_vector(self) -> None:
        self.assertEqual(
            _independent_normalize_noisy_slack(
                [4.0, 8.0, 12.0, 16.0], 4.0, 4, 2
            ),
            (1.0, 2.0, 3.0, 4.0),
        )
        update = _independent_full_slaclip_update(
            10.0,
            0.72,
            0.28,
            beta=0.5,
            eta=0.2,
            min_clip_norm=0.1,
            max_clip_norm=50.0,
            epsilon=1e-6,
        )
        self.assertAlmostEqual(update["near_zero_adjusted"], 0.027999997200000286)
        self.assertAlmostEqual(
            update["dynamic_target_clipped"], 0.4860000013999999
        )
        self.assertAlmostEqual(update["raw_log_step"], -0.041200000279999975)
        self.assertAlmostEqual(update["next_clip_norm"], 9.596371830484138)

    def test_exact_cdf_allows_only_float32_roundoff(self) -> None:
        _validate_exact_cdf_range([0.0, 1.0 + 5e-8], "exact CDF")
        with self.assertRaisesRegex(RuntimeError, "float32 tolerance"):
            _validate_exact_cdf_range([1.0 + 2e-6], "exact CDF")

    def test_validation_does_not_call_production_controller_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            baseline, adaptive = build_fixture(Path(raw_root))
            with (
                patch(
                    "paper_repro.slaclip.full_slaclip_update",
                    side_effect=AssertionError("production update was called"),
                ),
                patch(
                    "paper_repro.slaclip.normalize_noisy_slack",
                    side_effect=AssertionError("production normalization was called"),
                ),
            ):
                result = build_comparison(baseline, adaptive)
            self.assertEqual(result["status"], COMPARISON_STATUS)

    def test_valid_pair_uses_both_noisy_endpoints_without_fixed_target(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            baseline, adaptive = build_fixture(Path(raw_root))
            result = build_comparison(baseline, adaptive)
            self.assertEqual(result["status"], COMPARISON_STATUS)
            self.assertEqual(result["claim_level"], 1)
            self.assertFalse(result["paper_result_reproduced"])
            self.assertFalse(result["uses_baseline_calibration"])
            self.assertFalse(result["uses_fixed_target_clip_fraction"])
            self.assertEqual(result["slaclip_variant"], SLACLIP_VARIANT)
            self.assertTrue(
                result["privacy_notice"][
                    "controller_consumes_noisy_cdf_endpoints"
                ]
            )
            controller = result["models"]["bert"]["controller"]
            self.assertEqual(controller["near_threshold_index"], 0)
            self.assertEqual(controller["near_zero_index"], NUM_SLOTS - 1)
            self.assertEqual(controller["beta"], BETA)
            self.assertNotEqual(
                controller["final_clip_norm_by_group"]["A"], INITIAL_C
            )
            self.assertEqual(
                result["models"]["bert"]["paired_internal_holdout"][
                    "adaptive_minus_baseline_final"
                ],
                0.0,
            )
            self.assertNotIn("calibration_evidence", result)
            self.assertNotIn("active_target_spec", result)

    def test_cli_create_and_verify_existing_has_no_calibration_argument(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            baseline, adaptive = build_fixture(root)
            output = root / "comparison.json"
            arguments = [
                "--baseline-dir",
                str(baseline),
                "--adaptive-dir",
                str(adaptive),
                "--output",
                str(output),
            ]
            main(arguments)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            main([*arguments, "--verify-existing"])
            with self.assertRaises(SystemExit):
                main(arguments)
            with self.assertRaises(SystemExit):
                main([*arguments, "--calibration", "unused.json"])

    def test_schedule_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            baseline, adaptive = build_fixture(Path(raw_root))
            summary = load_json(adaptive / "bert" / "final_summary.json")
            summary["behavior_summary"]["sample_schedule_sha256"] = digest(
                "tampered schedule"
            )
            rewrite_adaptive_model_and_root(adaptive, "bert", summary)
            with self.assertRaisesRegex(RuntimeError, "sample schedules do not match"):
                build_comparison(baseline, adaptive)

    def test_near_zero_endpoint_tampering_fails_formula_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            baseline, adaptive = build_fixture(Path(raw_root))
            shard_path = (
                adaptive
                / "bert"
                / "private_diagnostics"
                / "rounds"
                / "round-00001.json"
            )
            shard = load_json(shard_path)
            shard["round_summary"]["slaclip_controller"]["A"][
                "noisy_cdf_proxy_by_slot"
            ][-1] += 0.2
            private_json(shard_path, shard)
            with self.assertRaisesRegex(RuntimeError, "noisy CDF does not reconcile"):
                build_comparison(baseline, adaptive)

    def test_exact_endpoint_telemetry_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            baseline, adaptive = build_fixture(Path(raw_root))
            shard_path = (
                adaptive
                / "gpt2"
                / "private_diagnostics"
                / "rounds"
                / "round-00001.json"
            )
            shard = load_json(shard_path)
            shard["round_summary"]["slaclip_controller"]["B"][
                "exact_cdf_proxy_by_slot"
            ][0] -= 0.1
            private_json(shard_path, shard)
            with self.assertRaisesRegex(RuntimeError, "exact CDF does not reconcile"):
                build_comparison(baseline, adaptive)

    def test_cdf_error_telemetry_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            baseline, adaptive = build_fixture(Path(raw_root))
            shard_path = (
                adaptive
                / "bert"
                / "private_diagnostics"
                / "rounds"
                / "round-00001.json"
            )
            shard = load_json(shard_path)
            shard["round_summary"]["slaclip_controller"]["A"][
                "cdf_error_rmse"
            ] += 0.1
            private_json(shard_path, shard)
            with self.assertRaisesRegex(
                RuntimeError, "endpoint telemetry mismatch: cdf_error_rmse"
            ):
                build_comparison(baseline, adaptive)

    def test_oracle_telemetry_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            baseline, adaptive = build_fixture(Path(raw_root))
            shard_path = (
                adaptive
                / "gpt2"
                / "private_diagnostics"
                / "rounds"
                / "round-00002.json"
            )
            shard = load_json(shard_path)
            shard["round_summary"]["slaclip_controller"]["B"][
                "oracle_next_clip_threshold"
            ] += 0.1
            private_json(shard_path, shard)
            with self.assertRaisesRegex(
                RuntimeError,
                "endpoint telemetry mismatch: oracle_next_clip_threshold",
            ):
                build_comparison(baseline, adaptive)

    def test_controller_summary_telemetry_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            baseline, adaptive = build_fixture(Path(raw_root))
            summary = load_json(adaptive / "bert" / "final_summary.json")
            controller = summary["behavior_summary"]["slaclip_controller"]
            controller["groups"]["A"]["cdf_error_mae"]["quantiles"]["0.5"] += 0.1
            summary["slaclip"]["controller_summary"] = controller
            rewrite_adaptive_model_and_root(adaptive, "bert", summary)
            with self.assertRaisesRegex(
                RuntimeError,
                "controller summary telemetry mismatch: cdf_error_mae",
            ):
                build_comparison(baseline, adaptive)

    def test_client_slack_release_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            baseline, adaptive = build_fixture(Path(raw_root))
            shard_path = (
                adaptive
                / "bert"
                / "private_diagnostics"
                / "rounds"
                / "round-00001.json"
            )
            shard = load_json(shard_path)
            shard["client_records"][0]["gradient_groups"]["A"]["slaclip"][
                "noisy_slack"
            ][-1] += 0.2
            private_json(shard_path, shard)
            with self.assertRaisesRegex(RuntimeError, "noisy CDF does not reconcile"):
                build_comparison(baseline, adaptive)

    def test_fixed_target_or_calibration_contract_is_rejected(self) -> None:
        for field in ("target_spec", "calibration"):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as raw_root:
                    baseline, adaptive = build_fixture(Path(raw_root))
                    contract = load_json(adaptive / "run_config.json")[
                        "scientific_contract"
                    ]["algorithm_contract"]["slaclip"]
                    contract[field] = {"legacy": True}
                    rewrite_adaptive_contract(adaptive, contract)
                    with self.assertRaisesRegex(
                        RuntimeError, "must not contain fixed-target field"
                    ):
                        build_comparison(baseline, adaptive)

    def test_endpoint_index_contract_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            baseline, adaptive = build_fixture(Path(raw_root))
            contract = load_json(adaptive / "run_config.json")[
                "scientific_contract"
            ]["algorithm_contract"]["slaclip"]
            contract["controller"]["near_zero_index"] = 0
            rewrite_adaptive_contract(adaptive, contract)
            with self.assertRaisesRegex(RuntimeError, "near-zero endpoint index"):
                build_comparison(baseline, adaptive)

    def test_adaptive_adapter_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            baseline, adaptive = build_fixture(Path(raw_root))
            adapter = (
                adaptive / "gpt2" / "final_adapter" / "adapter_model.safetensors"
            )
            adapter.write_bytes(b"tampered")
            os.chmod(adapter, 0o600)
            with self.assertRaisesRegex(RuntimeError, "adapter checksum mismatch"):
                build_comparison(baseline, adaptive)


if __name__ == "__main__":
    unittest.main()
