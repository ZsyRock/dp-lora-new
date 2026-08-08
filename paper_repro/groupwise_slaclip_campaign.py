#!/usr/bin/env python3
"""Run one hash-locked, groupwise generalized full-SlaClip-beta campaign.

This coordinator deliberately delegates every model update to
``full_slaclip_campaign.run_arm``.  It only materializes three sequential
stages inside one Slurm allocation, freezes development selections before
fresh-seed confirmation, and records the exact-endpoint oracle as NON_DP.
SlaClip-Q is neither represented in the schema nor accepted by this module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from paper_repro import full_slaclip_campaign as full
    from paper_repro import staged_slaclip_campaign as staged
    from paper_repro.slaclip import (
        build_slack_vector, normalize_noisy_slack,
        stationary_beta_from_exact_endpoints,
    )
except ModuleNotFoundError:  # direct-script execution
    import full_slaclip_campaign as full  # type: ignore[no-redef]
    import staged_slaclip_campaign as staged  # type: ignore[no-redef]
    from slaclip import (  # type: ignore[no-redef]
        build_slack_vector, normalize_noisy_slack,
        stationary_beta_from_exact_endpoints,
    )


SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1
MASTER_RUNTIME_NAME = "runtime-manifest.json"
PREFLIGHT_RUNTIME_NAME = "preflight-runtime-manifest.json"
STAGE2_RUNTIME_NAME = "stage2-runtime-manifest.json"
STAGE3_RUNTIME_NAME = "stage3-runtime-manifest.json"
FIXED_LOCK_NAME = "fixed-selection.lock.json"
SLACLIP_LOCK_NAME = "slaclip-selection.lock.json"
FIXED_STAGE = "fixed_development"
SLACLIP_STAGE = "slaclip_development"
CONFIRMATION_STAGE = "confirmation"
MODELS = ("bert", "gpt2")
GROUPS = ("A", "B")
FIXED_METHOD = full.FIXED_DP_METHOD
NOISY_METHOD = full.FULL_SLACLIP_METHOD
ORACLE_METHOD = full.ORACLE_SLACLIP_METHOD


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def load_spec(path: Path) -> dict[str, Any]:
    spec = full.load_object(path, "groupwise campaign specification")
    _exact_keys(
        spec,
        {
            "schema_version", "campaign_name", "description",
            "expected_stage_arm_counts", "common", "fixed_development",
            "slaclip_development", "confirmation", "scientific_boundary",
        },
        "specification",
    )
    if spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported groupwise campaign schema")
    common = spec["common"]
    _exact_keys(
        common,
        {
            "models", "num_clients", "rounds", "batch_size",
            "noise_multiplier", "learning_rate", "rank", "max_seq_length",
            "max_validation_records", "eval_every", "checkpoint_every",
            "data_split_seed", "evaluation_seed", "delta",
            "slaclip_num_slots", "slaclip_c_min", "slaclip_c_max",
            "slaclip_endpoint_epsilon",
        },
        "common",
    )
    if tuple(common["models"]) != MODELS:
        raise ValueError("common.models must be bert then gpt2")
    for key in (
        "num_clients", "rounds", "batch_size", "rank", "max_seq_length",
        "max_validation_records", "eval_every", "checkpoint_every",
        "data_split_seed", "evaluation_seed", "slaclip_num_slots",
    ):
        _integer(common[key], f"common.{key}")
    for key in (
        "noise_multiplier", "learning_rate", "delta", "slaclip_c_min",
        "slaclip_c_max", "slaclip_endpoint_epsilon",
    ):
        _number(common[key], f"common.{key}", positive=True)
    if common["rounds"] != 50 or common["batch_size"] != 8:
        raise ValueError("campaign requires T=50 and batch size 8")
    if common["slaclip_num_slots"] != 5 or common["num_clients"] != 5:
        raise ValueError("mechanism campaign requires K=5 and N=5")
    if float(common["noise_multiplier"]) != 2.0 or float(common["learning_rate"]) != 5e-4:
        raise ValueError("campaign fixes sigma=2 and learning rate=5e-4")
    if common["checkpoint_every"] != 25 or common["eval_every"] != 10:
        raise ValueError("checkpoint/evaluation cadence differs")

    fixed = spec["fixed_development"]
    _exact_keys(fixed, {"method", "clip_norm_by_model", "tuned_C_source", "seeds", "calibration"}, "fixed")
    if fixed["method"] != FIXED_METHOD or fixed["clip_norm_by_model"] != {"bert": 3.0, "gpt2": 1.0}:
        raise ValueError("fixed tuned-C anchors differ")
    if fixed["seeds"] != [300, 301, 302]:
        raise ValueError("development seeds differ")
    if fixed["tuned_C_source"] != {
        "campaign_run_id": "staged-slaclip-tuned-fixed-8bfa84c-20260805T073600Z",
        "repository_sha": "8bfa84cd149b145d2873c8b8153f85ea51cd9e0d",
        "fixed_lock_sha256": "6b1b9333afd15c316ae361f41e459b7115b36f5d8098218c70c01274175edefc",
        "fixed_lock_file_sha256": "1a782ed0262afd14f026d4e008fe11eb2c7fdebc12502401dacae3111e48ae00",
        "slurm_job_id": "1361071",
        "selected_clip_norm_by_model": {"bert": 3.0, "gpt2": 1.0},
    }:
        raise ValueError("tuned fixed-C source provenance differs")
    calibration = fixed["calibration"]
    if calibration.get("stationary_beta_formula") != "beta=(1-q)/(1-z)" or calibration.get("groups") != ["A", "B"]:
        raise ValueError("groupwise stationary-beta calibration differs")

    adaptive = spec["slaclip_development"]
    _exact_keys(adaptive, {"method", "seeds", "initial_clip_norms_by_model", "etas", "selection_rule"}, "adaptive")
    if adaptive["method"] != NOISY_METHOD or adaptive["seeds"] != fixed["seeds"]:
        raise ValueError("adaptive method/seeds differ")
    if adaptive["initial_clip_norms_by_model"] != {"bert": [2.5, 3.0], "gpt2": [0.75, 1.0]}:
        raise ValueError("initial-C candidates differ")
    if adaptive["etas"] != [0.0025, 0.005, 0.01]:
        raise ValueError("eta candidates differ")
    expected_rule = [
        "lowest_mean_paired_final_loss_delta_vs_tuned_fixed",
        "lowest_mean_paired_normalized_loss_auc_delta",
        "fewest_controller_instability_events", "smaller_eta",
        "smaller_beta_B", "smaller_initial_C",
    ]
    if adaptive["selection_rule"] != expected_rule:
        raise ValueError("selection rule differs")

    confirmation = spec["confirmation"]
    _exact_keys(confirmation, {"methods", "seeds", "primary_metric", "oracle_privacy_label"}, "confirmation")
    if confirmation["methods"] != [FIXED_METHOD, NOISY_METHOD, ORACLE_METHOD]:
        raise ValueError("confirmation methods differ")
    if confirmation["seeds"] != list(range(400, 420)):
        raise ValueError("confirmation seeds differ")
    if confirmation["oracle_privacy_label"] != "NON_DP_PRIVATE_DIAGNOSTIC":
        raise ValueError("oracle must be explicitly NON_DP")
    if set(confirmation["seeds"]) & set(fixed["seeds"]):
        raise ValueError("development and confirmation seeds overlap")
    if spec["scientific_boundary"].get("excluded_method_family") != "SlaClip-Q":
        raise ValueError("SlaClip-Q exclusion is missing")
    counts = spec["expected_stage_arm_counts"]
    expected = {FIXED_STAGE: 6, SLACLIP_STAGE: 180, CONFIRMATION_STAGE: 120, "total": 306}
    if counts != expected:
        raise ValueError(f"arm counts differ: {counts} != {expected}")
    return spec


def _token(value: float) -> str:
    return full.number_token(float(value))


def _base_arm(
    spec: Mapping[str, Any], *, arm_id: str, stage: str, method: str,
    model: str, seed: int, clip_norm: float, reference_arm_id: str | None,
    eta: float | None = None, betas: Mapping[str, float] | None = None,
    calibration_lock_sha256: str | None = None,
    calibration_provenance: str | None = None,
    require_calibration: bool = True,
) -> dict[str, Any]:
    adaptive = method in {NOISY_METHOD, ORACLE_METHOD}
    if adaptive != (eta is not None and betas is not None):
        raise ValueError("adaptive method and groupwise controller arguments disagree")
    if adaptive and set(betas or {}) != set(GROUPS):
        raise ValueError("adaptive targets must contain exactly A and B")
    if adaptive and require_calibration and stage in {SLACLIP_STAGE, CONFIRMATION_STAGE} and (
        not isinstance(calibration_lock_sha256, str)
        or len(calibration_lock_sha256) != 64
        or calibration_provenance != "exact_NON_DP_fixed_trajectory_diagnostics_frozen_before_fresh_seeds"
    ):
        raise ValueError("scientific adaptive arms require frozen calibration provenance")
    targets = {group: float(betas[group]) for group in GROUPS} if betas else None
    if targets and any(not 0 <= value <= 1 for value in targets.values()):
        raise ValueError("groupwise beta lies outside [0,1]")
    common = spec["common"]
    roles = {
        FIXED_STAGE: "fresh_tuned_fixed_trajectory_and_groupwise_beta_calibration",
        SLACLIP_STAGE: "noisy_groupwise_full_slaclip_development_selection",
        CONFIRMATION_STAGE: "fresh_seed_fixed_noisy_exact_oracle_confirmation",
    }
    return {
        "arm_id": arm_id, "stage": stage, "family": stage,
        "analysis_role": roles[stage], "method": method, "seed": seed,
        "initial_clip_norm": float(clip_norm), "slaclip_eta": eta,
        "slaclip_base_target_clipped_fraction": None,
        "slaclip_beta": None,
        "slaclip_base_target_clipped_fraction_by_group": targets,
        "slaclip_beta_by_group": dict(targets) if targets else None,
        "slaclip_baseline_calibration_lock_sha256": calibration_lock_sha256,
        "acknowledge_slaclip_baseline_calibration_is_non_dp": bool(calibration_lock_sha256),
        "slaclip_calibration_provenance": calibration_provenance,
        "controller_input": full.CONTROLLER_INPUT_BY_METHOD.get(method),
        "reference_arm_id": reference_arm_id,
        "rng_domain": f"groupwise-full-slaclip:s{seed}",
        "models": [model], "num_clients": common["num_clients"],
        "rounds": common["rounds"], "batch_size": common["batch_size"],
        "noise_multiplier": common["noise_multiplier"],
        "learning_rate": common["learning_rate"], "rank": common["rank"],
        "max_seq_length": common["max_seq_length"],
        "max_validation_records": common["max_validation_records"],
        "eval_every": common["eval_every"],
        "checkpoint_every": common["checkpoint_every"],
        "data_split_seed": common["data_split_seed"],
        "evaluation_seed": common["evaluation_seed"], "delta": common["delta"],
        "slaclip_num_slots": common["slaclip_num_slots"] if adaptive else None,
        "slaclip_c_min": common["slaclip_c_min"] if adaptive else None,
        "slaclip_c_max": common["slaclip_c_max"] if adaptive else None,
    }


def fixed_stage_arms(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    arms = [
        _base_arm(
            spec, arm_id=f"dev-fixed-{model}-c{_token(spec['fixed_development']['clip_norm_by_model'][model])}-s{seed}",
            stage=FIXED_STAGE, method=FIXED_METHOD, model=model, seed=int(seed),
            clip_norm=float(spec["fixed_development"]["clip_norm_by_model"][model]),
            reference_arm_id=None,
        )
        for model in MODELS for seed in spec["fixed_development"]["seeds"]
    ]
    return staged._indexed(arms)


def _runtime(
    spec: Mapping[str, Any], spec_path: Path, repository_sha: str,
    input_manifest: Path, created_at_utc: str, arms: list[dict[str, Any]],
    stage: str, parent_lock_sha256: str | None,
) -> dict[str, Any]:
    value = {
        "schema_version": full.SCHEMA_VERSION,
        "campaign_name": spec["campaign_name"], "stage": stage,
        "created_at_utc": created_at_utc, "repository_sha": repository_sha,
        "spec_sha256": full.sha256_file(spec_path),
        "input_manifest_path": str(input_manifest.resolve()),
        "input_manifest_sha256": full.sha256_file(input_manifest),
        "parent_selection_lock_sha256": parent_lock_sha256,
        "expected_arm_count": len(arms),
        "scientific_boundary": spec["scientific_boundary"], "arms": arms,
    }
    value["manifest_sha256"] = full.sha256_bytes(full.canonical_bytes(value))
    full.validate_runtime_manifest(value)
    return value


def _preflight(master: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    fixed = _base_arm(spec, arm_id="preflight-template-fixed", stage=FIXED_STAGE,
        method=FIXED_METHOD, model="bert", seed=300, clip_norm=10.0,
        reference_arm_id=None)
    adaptive = _base_arm(spec, arm_id="preflight-template-slaclip", stage=SLACLIP_STAGE,
        method=NOISY_METHOD, model="bert", seed=300, clip_norm=10.0,
        reference_arm_id=fixed["arm_id"], eta=0.005, betas={"A": 0.5, "B": 0.5},
        require_calibration=False)
    for arm in (fixed, adaptive):
        arm["family"] = "primary"
        arm["models"] = list(MODELS)
    arms = staged._indexed([fixed, adaptive])
    value = {
        "schema_version": full.SCHEMA_VERSION,
        "campaign_name": f"{master['campaign_name']}-preflight",
        "created_at_utc": master["created_at_utc"],
        "repository_sha": master["repository_sha"],
        "spec_sha256": master["spec_sha256"],
        "input_manifest_path": master["input_manifest_path"],
        "input_manifest_sha256": master["input_manifest_sha256"],
        "expected_arm_count": 2,
        "scientific_boundary": {"analysis_role": "real_model_smoke_only"},
        "arms": arms,
    }
    value["manifest_sha256"] = full.sha256_bytes(full.canonical_bytes(value))
    full.validate_runtime_manifest(value)
    return value


def _master(spec: Mapping[str, Any], spec_path: Path, sha: str, inputs: Path, created: str) -> dict[str, Any]:
    return _runtime(spec, spec_path, sha, inputs, created, fixed_stage_arms(spec), FIXED_STAGE, None)


def _identity(
    campaign_root: Path, spec_path: Path, repository: Path,
    expected_sha: str, input_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = load_spec(spec_path)
    if full.repository_sha(repository) != expected_sha or full.repository_dirty(repository):
        raise RuntimeError("repository snapshot SHA/cleanliness gate failed")
    manifest = full.load_runtime(campaign_root / MASTER_RUNTIME_NAME)
    candidate = _master(
        spec, spec_path, expected_sha, input_manifest,
        str(manifest.get("created_at_utc")),
    )
    if manifest != candidate:
        raise RuntimeError("master runtime manifest differs from immutable inputs")
    return spec, manifest


def _load_lock(path: Path, label: str) -> dict[str, Any]:
    value = full.load_object(path, label)
    staged._validate_lock(value, label)
    return value


def _write_or_verify(path: Path, value: Mapping[str, Any], label: str) -> None:
    staged._write_or_verify(path, value, label)


def prepare_campaign(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    repository = args.repository.resolve()
    inputs = args.input_manifest.resolve()
    spec = load_spec(spec_path)
    if full.repository_sha(repository) != args.expected_code_sha or full.repository_dirty(repository):
        raise RuntimeError("repository snapshot SHA/cleanliness gate failed")
    path = root / MASTER_RUNTIME_NAME
    if args.resume:
        if not root.is_dir() or not path.is_file():
            raise RuntimeError("resume requires an existing groupwise campaign")
        master = full.load_runtime(path)
        candidate = _master(spec, spec_path, args.expected_code_sha, inputs, str(master.get("created_at_utc")))
        if master != candidate:
            raise RuntimeError("resume inputs differ from immutable master manifest")
    else:
        if root.exists():
            raise RuntimeError(f"refusing to overwrite campaign root: {root}")
        root.mkdir(parents=True, mode=0o700)
        master = _master(spec, spec_path, args.expected_code_sha, inputs, full.utc_now())
        full.atomic_json(path, master)
    for name in ("arms", "arm-status", "arm-logs", "control", "tmp", "preflight", "selection"):
        (root / name).mkdir(mode=0o700, exist_ok=True)
    stop = root / "control" / "stop.request"
    if stop.exists():
        stop.unlink()
    full.validate_or_create_key(full.absolute_path(args.private_key), create=not args.resume)
    _write_or_verify(root / PREFLIGHT_RUNTIME_NAME, _preflight(master, spec), "preflight manifest")
    if (root / FIXED_LOCK_NAME).is_file():
        fixed = _load_lock(root / FIXED_LOCK_NAME, "fixed selection lock")
        _validate_fixed_lock(fixed, master, spec)
        _ensure_stage2(root, spec_path, spec, master, fixed)
    if (root / SLACLIP_LOCK_NAME).is_file():
        fixed = _load_lock(root / FIXED_LOCK_NAME, "fixed selection lock")
        stage2 = full.load_runtime(root / STAGE2_RUNTIME_NAME)
        selected = _load_lock(root / SLACLIP_LOCK_NAME, "SlaClip selection lock")
        _validate_slaclip_lock(selected, master, fixed, stage2, spec)
        _ensure_stage3(root, spec_path, spec, master, fixed, selected)
    print(f"runtime_manifest={path}")
    print(f"preflight_runtime_manifest={root / PREFLIGHT_RUNTIME_NAME}")


def _round_path(root: Path, arm: Mapping[str, Any], round_index: int) -> Path:
    return (
        root / "arms" / str(arm["arm_id"]) / str(arm["models"][0]) /
        "private_diagnostics" / "rounds" / f"round-{round_index:05d}.json"
    )


def stationary_beta(q: float, r: float, clip_norm: float, epsilon: float) -> tuple[float, float]:
    """Return (beta,z) from the camera-ready generalized Full-SlaClip rule."""
    try:
        result = stationary_beta_from_exact_endpoints(
            clip_norm, q, r, epsilon=epsilon
        )
    except (TypeError, ValueError, FloatingPointError) as error:
        raise RuntimeError("fixed trajectory cannot define a stationary beta") from error
    return float(result["stationary_beta"]), float(result["near_zero_adjusted"])


def _calibrate(
    root: Path, arms: Sequence[Mapping[str, Any]], *, clip_norm: float,
    num_slots: int, epsilon: float, rounds: int,
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    values = {group: [] for group in GROUPS}
    rows: list[dict[str, Any]] = []
    for arm in arms:
        model = str(arm["models"][0])
        for round_index in range(1, rounds + 1):
            shard = full.load_object(_round_path(root, arm, round_index), "fixed trajectory shard")
            records = shard.get("client_records")
            summary = shard.get("round_summary")
            if (
                shard.get("round") != round_index
                or shard.get("model") != model
                or shard.get("method") != FIXED_METHOD
                or not isinstance(records, list) or not records
                or not isinstance(summary, dict)
            ):
                raise RuntimeError(f"fixed trajectory shard is invalid: {arm['arm_id']}/{round_index}")
            for group in GROUPS:
                norms = [float(record["gradient_groups"][group]["raw_norm"]) for record in records]
                signals = [build_slack_vector(norm, clip_norm, num_slots) for norm in norms]
                signal_sum = [math.fsum(vector[slot] for vector in signals) for slot in range(num_slots)]
                exact = normalize_noisy_slack(signal_sum, clip_norm, num_slots, len(records))
                q_value = float(exact[0])
                r_value = float(exact[-1])
                beta, z_value = stationary_beta(q_value, r_value, clip_norm, epsilon)
                actual = float(summary[group]["clipped_fraction"])
                dynamic_target = beta * (1.0 - z_value)
                values[group].append(beta)
                rows.append({
                    "arm_id": arm["arm_id"], "model": model, "seed": arm["seed"],
                    "round": round_index, "group": group, "fixed_C": clip_norm,
                    "exact_q_endpoint_1": q_value, "exact_r_endpoint_K": r_value,
                    "z_r_over_C_plus_epsilon": z_value,
                    "stationary_beta": beta,
                    "actual_clipped_fraction": actual,
                    "stationary_dynamic_target_clipped": dynamic_target,
                    "tracking_bias_actual_minus_target": actual - dynamic_target,
                    "identity_error_actual_minus_one_minus_q": actual - (1.0 - q_value),
                })
    return values, rows


def _beta_grid(values: Sequence[float]) -> tuple[list[float], float, float]:
    return staged.derive_beta_grid(values, lower_quantile=0.1, upper_quantile=0.9, points=5)


def _validate_fixed_lock(lock: Mapping[str, Any], master: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    if (
        lock.get("status") != "GROUPWISE_FIXED_CALIBRATION_LOCKED"
        or lock.get("master_runtime_manifest_sha256") != master["manifest_sha256"]
        or lock.get("spec_sha256") != master["spec_sha256"]
        or lock.get("calibration_privacy_label") != "NON_DP_PRIVATE_DIAGNOSTIC"
        or lock.get("calibration_provenance") != "exact_NON_DP_fixed_trajectory_diagnostics_frozen_before_fresh_seeds"
        or lock.get("calibration_data_consumed_at_controller_runtime") is not False
    ):
        raise RuntimeError("fixed calibration lock identity differs")
    models = lock.get("models")
    if not isinstance(models, dict) or tuple(models) != MODELS:
        raise RuntimeError("fixed calibration lock model set differs")
    for model in MODELS:
        record = models[model]
        if (
            not isinstance(record, dict)
            or not 0 <= float(record.get("beta_A", math.nan)) <= 1
            or not isinstance(record.get("beta_B_grid"), list)
            or len(record["beta_B_grid"]) != 5
            or len({full.canonical_bytes(value) for value in record["beta_B_grid"]}) != 5
        ):
            raise RuntimeError("fixed calibration candidates are invalid")


def stage2_arms(spec: Mapping[str, Any], fixed_lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for model in MODELS:
        record = fixed_lock["models"][model]
        fixed_c = float(spec["fixed_development"]["clip_norm_by_model"][model])
        for c0 in spec["slaclip_development"]["initial_clip_norms_by_model"][model]:
            for beta_b in record["beta_B_grid"]:
                for eta in spec["slaclip_development"]["etas"]:
                    for seed in spec["slaclip_development"]["seeds"]:
                        arms.append(_base_arm(
                            spec,
                            arm_id=(f"dev-groupwise-{model}-c0{_token(c0)}-ba{_token(record['beta_A'])}-"
                                    f"bb{_token(beta_b)}-e{_token(eta)}-s{seed}"),
                            stage=SLACLIP_STAGE, method=NOISY_METHOD, model=model,
                            seed=int(seed), clip_norm=float(c0), eta=float(eta),
                            betas={"A": float(record["beta_A"]), "B": float(beta_b)},
                            calibration_lock_sha256=str(fixed_lock["lock_sha256"]),
                            calibration_provenance="exact_NON_DP_fixed_trajectory_diagnostics_frozen_before_fresh_seeds",
                            reference_arm_id=f"dev-fixed-{model}-c{_token(fixed_c)}-s{seed}",
                        ))
    return staged._indexed(arms)


def _ensure_stage2(
    root: Path, spec_path: Path, spec: Mapping[str, Any], master: Mapping[str, Any],
    fixed_lock: Mapping[str, Any],
) -> Path:
    _validate_fixed_lock(fixed_lock, master, spec)
    staged._verify_locked_evidence(root, master, fixed_lock.get("source_evidence"))
    calibration_path = root / "fixed_groupwise_beta_calibration.csv"
    if (
        not calibration_path.is_file()
        or full.sha256_file(calibration_path) != fixed_lock.get("calibration_csv_sha256")
    ):
        raise RuntimeError("locked groupwise calibration CSV differs")
    path = root / STAGE2_RUNTIME_NAME
    candidate = _runtime(
        spec, spec_path, str(master["repository_sha"]), Path(str(master["input_manifest_path"])),
        str(master["created_at_utc"]), stage2_arms(spec, fixed_lock), SLACLIP_STAGE,
        str(fixed_lock["lock_sha256"]),
    )
    if len(candidate["arms"]) != 180:
        raise RuntimeError("Stage 2 must contain 180 model-specific arms")
    _write_or_verify(path, candidate, "Stage 2 runtime manifest")
    return path


CALIBRATION_COLUMNS = (
    "arm_id", "model", "seed", "round", "group", "fixed_C",
    "exact_q_endpoint_1", "exact_r_endpoint_K", "z_r_over_C_plus_epsilon",
    "stationary_beta", "actual_clipped_fraction",
    "stationary_dynamic_target_clipped", "tracking_bias_actual_minus_target",
    "identity_error_actual_minus_one_minus_q",
)


def lock_fixed_selection(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, master = _identity(
        root, spec_path, args.repository.resolve(), args.expected_code_sha,
        args.input_manifest.resolve(),
    )
    lock_path = root / FIXED_LOCK_NAME
    if lock_path.is_file():
        lock = _load_lock(lock_path, "fixed selection lock")
        _validate_fixed_lock(lock, master, spec)
        stage2 = _ensure_stage2(root, spec_path, spec, master, lock)
        print(f"fixed_selection_reused={lock_path}")
        print(f"stage2_runtime_manifest={stage2}")
        return
    evidence = staged._arm_evidence(root, master)
    model_records: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for model in MODELS:
        arms = [arm for arm in master["arms"] if arm["models"] == [model]]
        if len(arms) != 3:
            raise RuntimeError(f"fixed trajectory arm set is incomplete: {model}")
        clip_norm = float(spec["fixed_development"]["clip_norm_by_model"][model])
        values, rows = _calibrate(
            root, arms, clip_norm=clip_norm,
            num_slots=int(spec["common"]["slaclip_num_slots"]),
            epsilon=float(spec["common"]["slaclip_endpoint_epsilon"]),
            rounds=int(spec["common"]["rounds"]),
        )
        beta_a = statistics.median(values["A"])
        beta_b_grid, q10, q90 = _beta_grid(values["B"])
        all_rows.extend(rows)
        model_rows = [row for row in rows if row["model"] == model]
        model_records[model] = {
            "selected_fixed_C": clip_norm,
            "beta_A": beta_a,
            "beta_A_policy": "stationary_full_trajectory_median_allow_zero",
            "beta_B_grid": beta_b_grid,
            "beta_B_q10": q10,
            "beta_B_median": statistics.median(values["B"]),
            "beta_B_q90": q90,
            "calibration_round_records_per_group": len(values["A"]),
            "mean_actual_clipped_fraction_by_group": {
                group: statistics.fmean(
                    float(row["actual_clipped_fraction"])
                    for row in model_rows if row["group"] == group
                ) for group in GROUPS
            },
            "mean_tracking_bias_by_group": {
                group: statistics.fmean(
                    float(row["tracking_bias_actual_minus_target"])
                    for row in model_rows if row["group"] == group
                ) for group in GROUPS
            },
            "formula": "beta_g=(1-q_g)/(1-r_g/(C+epsilon))",
        }
    calibration_path = root / "fixed_groupwise_beta_calibration.csv"
    full.atomic_csv(calibration_path, all_rows, CALIBRATION_COLUMNS)
    lock = staged._lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "GROUPWISE_FIXED_CALIBRATION_LOCKED",
        "campaign_name": master["campaign_name"],
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "spec_sha256": master["spec_sha256"],
        "development_seeds": spec["fixed_development"]["seeds"],
        "confirmation_data_accessed": False,
        "calibration_privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
        "calibration_provenance": "exact_NON_DP_fixed_trajectory_diagnostics_frozen_before_fresh_seeds",
        "calibration_data_consumed_at_controller_runtime": False,
        "models": model_records,
        "source_evidence": evidence,
        "calibration_csv_sha256": full.sha256_file(calibration_path),
        "created_at_utc": full.utc_now(),
    })
    full.atomic_json(lock_path, lock)
    staged._validate_lock(_load_lock(lock_path, "fixed selection lock"), "fixed selection lock")
    stage2 = _ensure_stage2(root, spec_path, spec, master, lock)
    _write_trajectories(root, [(master, master["arms"])])
    print(f"fixed_selection_lock={lock_path}")
    print(f"stage2_runtime_manifest={stage2}")


def _metric_row(root: Path, manifest: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    row = staged._metric_row(root, manifest, arm)
    _summary, model_summary, final_sha = staged._completed_summary(root, manifest, arm)
    partitions = model_summary.get("client_partition_sha256")
    if (
        not isinstance(partitions, list)
        or len(partitions) != int(arm["num_clients"])
        or any(not isinstance(value, str) or len(value) != 64 for value in partitions)
    ):
        raise RuntimeError(f"client-partition evidence is invalid: {arm['arm_id']}")
    row["client_partition_commitment_sha256"] = full.sha256_bytes(
        full.canonical_bytes(partitions)
    )
    row["final_summary_sha256"] = final_sha
    row["round_shard_prefix_sha256"] = model_summary.get("round_shard_prefix_sha256")
    run_config_path = root / "arms" / str(arm["arm_id"]) / "run_config.json"
    row["run_config_sha256"] = full.sha256_file(run_config_path)
    targets = arm.get("slaclip_base_target_clipped_fraction_by_group")
    row["slaclip_beta_A"] = targets.get("A") if isinstance(targets, dict) else None
    row["slaclip_beta_B"] = targets.get("B") if isinstance(targets, dict) else None
    row["privacy_label"] = (
        "NON_DP_PRIVATE_DIAGNOSTIC" if arm["method"] == ORACLE_METHOD
        else "DP_MECHANISM_WITH_NON_DP_PRIVATE_DIAGNOSTICS"
    )
    row["slaclip_baseline_calibration_lock_sha256"] = arm.get(
        "slaclip_baseline_calibration_lock_sha256"
    )
    row["slaclip_calibration_provenance"] = arm.get("slaclip_calibration_provenance")
    return row


def _validate_slaclip_lock(
    lock: Mapping[str, Any], master: Mapping[str, Any], fixed_lock: Mapping[str, Any],
    stage2: Mapping[str, Any], spec: Mapping[str, Any],
) -> None:
    if (
        lock.get("status") != "GROUPWISE_SLACLIP_DEVELOPMENT_SELECTION_LOCKED"
        or lock.get("master_runtime_manifest_sha256") != master["manifest_sha256"]
        or lock.get("fixed_selection_lock_sha256") != fixed_lock["lock_sha256"]
        or lock.get("stage2_runtime_manifest_sha256") != stage2["manifest_sha256"]
        or lock.get("selection_rule") != spec["slaclip_development"]["selection_rule"]
    ):
        raise RuntimeError("SlaClip selection lock identity differs")
    if not isinstance(lock.get("models"), dict) or tuple(lock["models"]) != MODELS:
        raise RuntimeError("SlaClip selection lock model set differs")
    for model in MODELS:
        selected = lock["models"][model]
        fixed = fixed_lock["models"][model]
        if (
            float(selected.get("selected_beta_A", math.nan)) != float(fixed["beta_A"])
            or float(selected.get("selected_beta_B", math.nan)) not in [float(v) for v in fixed["beta_B_grid"]]
            or float(selected.get("selected_eta", math.nan)) not in [float(v) for v in spec["slaclip_development"]["etas"]]
            or float(selected.get("selected_initial_C", math.nan)) not in [float(v) for v in spec["slaclip_development"]["initial_clip_norms_by_model"][model]]
        ):
            raise RuntimeError("selected groupwise controller is outside preregistered candidates")


def _fixed_reference_rows(
    root: Path, master: Mapping[str, Any]
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(arm["models"][0]), int(arm["seed"])): _metric_row(root, master, arm)
        for arm in master["arms"]
    }


def _selection_sort_key(row: Mapping[str, Any]) -> tuple[float, float, int, float, float, float]:
    """Implement the preregistered lexicographic development selection."""
    return (
        float(row["mean_paired_final_loss_delta"]),
        float(row["mean_paired_normalized_loss_auc_delta"]),
        int(row["controller_instability_event_count"]),
        float(row["eta"]), float(row["beta_B"]), float(row["initial_C"]),
    )


def lock_slaclip_selection(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, master = _identity(root, spec_path, args.repository.resolve(), args.expected_code_sha, args.input_manifest.resolve())
    fixed_lock = _load_lock(root / FIXED_LOCK_NAME, "fixed selection lock")
    _validate_fixed_lock(fixed_lock, master, spec)
    stage2_path = _ensure_stage2(root, spec_path, spec, master, fixed_lock)
    stage2 = full.load_runtime(stage2_path)
    lock_path = root / SLACLIP_LOCK_NAME
    if lock_path.is_file():
        lock = _load_lock(lock_path, "SlaClip selection lock")
        _validate_slaclip_lock(lock, master, fixed_lock, stage2, spec)
        stage3 = _ensure_stage3(root, spec_path, spec, master, fixed_lock, lock)
        print(f"slaclip_selection_reused={lock_path}")
        print(f"stage3_runtime_manifest={stage3}")
        return
    evidence = staged._arm_evidence(root, stage2)
    references = _fixed_reference_rows(root, master)
    rows = [_metric_row(root, stage2, arm) for arm in stage2["arms"]]
    model_records: dict[str, Any] = {}
    for model in MODELS:
        candidates: list[dict[str, Any]] = []
        beta_a = float(fixed_lock["models"][model]["beta_A"])
        for c0 in spec["slaclip_development"]["initial_clip_norms_by_model"][model]:
            for beta_b in fixed_lock["models"][model]["beta_B_grid"]:
                for eta in spec["slaclip_development"]["etas"]:
                    subset = [
                        row for row in rows
                        if row["model"] == model
                        and float(row["initial_clip_norm"]) == float(c0)
                        and float(row["slaclip_beta_B"]) == float(beta_b)
                        and float(row["slaclip_eta"]) == float(eta)
                    ]
                    if {int(row["seed"]) for row in subset} != set(spec["slaclip_development"]["seeds"]):
                        raise RuntimeError("adaptive candidate is incomplete")
                    final_deltas: list[float] = []
                    auc_deltas: list[float] = []
                    instability = 0
                    for row in subset:
                        reference = references[(model, int(row["seed"]))]
                        if row.get("initial_loss") != reference.get("initial_loss"):
                            raise RuntimeError(f"paired initial loss differs: {model}/{row['seed']}")
                        for digest in (
                            "client_partition_commitment_sha256", "sample_schedule_sha256",
                            "supervision_schedule_sha256", "private_key_commitment", "rng_domain",
                        ):
                            if row.get(digest) != reference.get(digest):
                                raise RuntimeError(f"paired development evidence differs: {model}/{row['seed']}/{digest}")
                        final_deltas.append(float(row["final_loss"]) - float(reference["final_loss"]))
                        auc_deltas.append(float(row["normalized_loss_auc"]) - float(reference["normalized_loss_auc"]))
                        instability += staged._controller_instability_events(row)
                    candidates.append({
                        "initial_C": float(c0), "beta_A": beta_a, "beta_B": float(beta_b),
                        "eta": float(eta), "seed_count": len(subset),
                        "mean_paired_final_loss_delta": statistics.fmean(final_deltas),
                        "mean_paired_normalized_loss_auc_delta": statistics.fmean(auc_deltas),
                        "controller_instability_event_count": instability,
                        "paired_final_loss_deltas": final_deltas,
                        "paired_normalized_loss_auc_deltas": auc_deltas,
                    })
        candidates.sort(key=_selection_sort_key)
        selected = candidates[0]
        model_records[model] = {
            "selected_fixed_C": float(fixed_lock["models"][model]["selected_fixed_C"]),
            "selected_initial_C": selected["initial_C"], "selected_beta_A": beta_a,
            "selected_beta_B": selected["beta_B"], "selected_eta": selected["eta"],
            "selected_num_slots": 5, "ordered_candidates": candidates,
        }
    lock = staged._lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "GROUPWISE_SLACLIP_DEVELOPMENT_SELECTION_LOCKED",
        "campaign_name": master["campaign_name"],
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "fixed_selection_lock_sha256": fixed_lock["lock_sha256"],
        "stage2_runtime_manifest_sha256": stage2["manifest_sha256"],
        "selection_rule": spec["slaclip_development"]["selection_rule"],
        "development_seeds": spec["slaclip_development"]["seeds"],
        "confirmation_data_accessed": False, "models": model_records,
        "source_evidence": evidence, "created_at_utc": full.utc_now(),
    })
    full.atomic_json(lock_path, lock)
    stage3 = _ensure_stage3(root, spec_path, spec, master, fixed_lock, lock)
    _write_trajectories(root, [(master, master["arms"]), (stage2, stage2["arms"])])
    print(f"slaclip_selection_lock={lock_path}")
    print(f"stage3_runtime_manifest={stage3}")


def stage3_arms(
    spec: Mapping[str, Any], fixed_lock: Mapping[str, Any],
    slaclip_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    provenance = "exact_NON_DP_fixed_trajectory_diagnostics_frozen_before_fresh_seeds"
    for model in MODELS:
        fixed_c = float(fixed_lock["models"][model]["selected_fixed_C"])
        selected = slaclip_lock["models"][model]
        targets = {"A": float(selected["selected_beta_A"]), "B": float(selected["selected_beta_B"])}
        for seed in spec["confirmation"]["seeds"]:
            fixed_id = f"confirm-fixed-{model}-c{_token(fixed_c)}-s{seed}"
            arms.append(_base_arm(
                spec, arm_id=fixed_id, stage=CONFIRMATION_STAGE,
                method=FIXED_METHOD, model=model, seed=int(seed), clip_norm=fixed_c,
                reference_arm_id=None,
            ))
            common = {
                "spec": spec, "stage": CONFIRMATION_STAGE, "model": model,
                "seed": int(seed), "clip_norm": float(selected["selected_initial_C"]),
                "reference_arm_id": fixed_id, "eta": float(selected["selected_eta"]),
                "betas": targets,
                "calibration_lock_sha256": str(fixed_lock["lock_sha256"]),
                "calibration_provenance": provenance,
            }
            arms.append(_base_arm(
                arm_id=(f"confirm-noisy-{model}-c0{_token(selected['selected_initial_C'])}-"
                        f"ba{_token(targets['A'])}-bb{_token(targets['B'])}-"
                        f"e{_token(selected['selected_eta'])}-s{seed}"),
                method=NOISY_METHOD, **common,
            ))
            arms.append(_base_arm(
                arm_id=(f"confirm-oracle-NONDP-{model}-c0{_token(selected['selected_initial_C'])}-"
                        f"ba{_token(targets['A'])}-bb{_token(targets['B'])}-"
                        f"e{_token(selected['selected_eta'])}-s{seed}"),
                method=ORACLE_METHOD, **common,
            ))
    return staged._indexed(arms)


def _ensure_stage3(
    root: Path, spec_path: Path, spec: Mapping[str, Any], master: Mapping[str, Any],
    fixed_lock: Mapping[str, Any], slaclip_lock: Mapping[str, Any],
) -> Path:
    _validate_slaclip_lock(slaclip_lock, master, fixed_lock,
        full.load_runtime(root / STAGE2_RUNTIME_NAME), spec)
    staged._verify_locked_evidence(root, master, fixed_lock.get("source_evidence"))
    staged._verify_locked_evidence(
        root, full.load_runtime(root / STAGE2_RUNTIME_NAME),
        slaclip_lock.get("source_evidence"),
    )
    path = root / STAGE3_RUNTIME_NAME
    candidate = _runtime(
        spec, spec_path, str(master["repository_sha"]), Path(str(master["input_manifest_path"])),
        str(master["created_at_utc"]), stage3_arms(spec, fixed_lock, slaclip_lock),
        CONFIRMATION_STAGE, str(slaclip_lock["lock_sha256"]),
    )
    if len(candidate["arms"]) != 120:
        raise RuntimeError("Stage 3 must contain 120 model-specific arms")
    if len(full.resolve_oracle_noisy_arm_pairs(candidate)) != 40:
        raise RuntimeError("Stage 3 oracle/noisy groupwise pairing is incomplete")
    _write_or_verify(path, candidate, "Stage 3 runtime manifest")
    return path


def _write_trajectories(
    root: Path,
    stages: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
) -> None:
    fixed_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    for manifest, arms in stages:
        for arm in arms:
            if not staged._status_completed(root, manifest, arm):
                continue
            rows = staged._round_trajectory_rows(root, arm)
            targets = arm.get("slaclip_base_target_clipped_fraction_by_group")
            for row in rows:
                group = row.get("group")
                row["groupwise_target_beta"] = (
                    targets.get(group) if isinstance(targets, dict) and group in GROUPS else None
                )
                row["slaclip_beta_A"] = targets.get("A") if isinstance(targets, dict) else None
                row["slaclip_beta_B"] = targets.get("B") if isinstance(targets, dict) else None
                row["slaclip_baseline_calibration_lock_sha256"] = arm.get("slaclip_baseline_calibration_lock_sha256")
                row["slaclip_calibration_provenance"] = arm.get("slaclip_calibration_provenance")
                row["oracle_privacy_label"] = (
                    "NON_DP_PRIVATE_DIAGNOSTIC" if arm["method"] == ORACLE_METHOD else None
                )
            (fixed_rows if arm["method"] == FIXED_METHOD else adaptive_rows).extend(rows)
    columns = tuple(staged.TRAJECTORY_COLUMNS) + (
        "groupwise_target_beta", "slaclip_beta_A", "slaclip_beta_B",
        "slaclip_baseline_calibration_lock_sha256", "slaclip_calibration_provenance",
        "oracle_privacy_label",
    )
    full.atomic_csv(root / "fixed_trajectory.csv", fixed_rows, columns)
    full.atomic_csv(root / "groupwise_slaclip_trajectory.csv", adaptive_rows, columns)


def _validate_adaptive_contract(
    root: Path, manifest: Mapping[str, Any], arm: Mapping[str, Any],
) -> None:
    if arm["method"] == FIXED_METHOD:
        for key in (
            "slaclip_base_target_clipped_fraction", "slaclip_beta",
            "slaclip_base_target_clipped_fraction_by_group", "slaclip_beta_by_group",
            "slaclip_baseline_calibration_lock_sha256", "slaclip_calibration_provenance",
        ):
            if arm.get(key) is not None:
                raise RuntimeError(f"fixed arm contains adaptive field: {arm['arm_id']}/{key}")
        if arm.get("acknowledge_slaclip_baseline_calibration_is_non_dp") is not False:
            raise RuntimeError("fixed arm contains calibration acknowledgement")
        return
    _summary, model_summary, _sha = staged._completed_summary(root, manifest, arm)
    try:
        from paper_repro.train_federated import behavior_summary, read_round_shards
    except ModuleNotFoundError:
        from train_federated import behavior_summary, read_round_shards  # type: ignore[no-redef]
    arm_root = root / "arms" / str(arm["arm_id"])
    run_config = full.load_object(arm_root / "run_config.json", "arm run config")
    try:
        algorithm_contract = run_config["scientific_contract"]["algorithm_contract"]
        persisted_contract = algorithm_contract["slaclip"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"persisted SlaClip contract is missing: {arm['arm_id']}") from error
    extension = model_summary.get("slaclip")
    contract = extension.get("contract") if isinstance(extension, dict) else None
    controller = contract.get("controller") if isinstance(contract, dict) else None
    provenance = contract.get("hyperparameter_provenance") if isinstance(contract, dict) else None
    expected_targets = arm["slaclip_base_target_clipped_fraction_by_group"]
    configured_initial = arm.get("initial_clip_norm_by_group")
    expected_initial = (
        {group: float(configured_initial[group]) for group in GROUPS}
        if isinstance(configured_initial, dict) and set(configured_initial) == set(GROUPS)
        else {group: float(arm["initial_clip_norm"]) for group in GROUPS}
    )
    persisted_controller_initial = (
        controller.get("initial_clip_threshold_by_group")
        if isinstance(controller, dict)
        else None
    )
    if persisted_controller_initial is None and isinstance(controller, dict):
        persisted_controller_initial = {
            group: controller.get("initial_clip_threshold") for group in GROUPS
        }
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != "groupwise_generalized_full_slaclip_beta_contract_v1"
        or contract.get("variant") != "groupwise_generalized_full_slaclip_beta"
        or contract.get("controller_input") != arm["controller_input"]
        or not isinstance(controller, dict)
        or controller.get("base_target_clipped_fraction_by_group") != expected_targets
        or controller.get("beta_by_group") != expected_targets
        or not isinstance(provenance, dict)
        or provenance.get("calibration_lock_sha256") != arm["slaclip_baseline_calibration_lock_sha256"]
        or provenance.get("baseline_derived_calibration_is_non_dp") is not True
        or provenance.get("calibration_data_consumed_at_controller_runtime") is not False
        or persisted_controller_initial != expected_initial
        or algorithm_contract.get("initial_clip_threshold_by_group") != expected_initial
    ):
        raise RuntimeError(f"groupwise runtime contract differs: {arm['arm_id']}")
    if persisted_contract != contract:
        raise RuntimeError(f"run-config/final contract differs: {arm['arm_id']}")
    if arm["method"] == ORACLE_METHOD and contract.get("non_private_oracle_control") is not True:
        raise RuntimeError("oracle arm lost its NON_DP exact-controller contract")
    computed_prefix = staged._computed_round_shard_prefix(root, arm)
    if model_summary.get("round_shard_prefix_sha256") != computed_prefix:
        raise RuntimeError(f"round telemetry prefix differs: {arm['arm_id']}")
    model = str(arm["models"][0])
    shards = read_round_shards(
        arm_root / model / "private_diagnostics" / "rounds",
        expected_rounds=int(arm["rounds"]), expected_model=model,
        expected_method=str(arm["method"]), expected_clients=int(arm["num_clients"]),
        expected_batch_size=int(arm["batch_size"]), slaclip_contract=contract,
    )
    if behavior_summary(shards) != model_summary.get("behavior_summary"):
        raise RuntimeError(f"deep round behavior revalidation differs: {arm['arm_id']}")


def _paired_rows(
    rows: Sequence[Mapping[str, Any]], stage3: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_arm = {str(row["arm_id"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for arm in stage3["arms"]:
        if arm["method"] not in {NOISY_METHOD, ORACLE_METHOD}:
            continue
        candidate = by_arm.get(str(arm["arm_id"]))
        reference = by_arm.get(str(arm["reference_arm_id"]))
        if candidate is None or reference is None:
            continue
        if candidate.get("initial_loss") != reference.get("initial_loss"):
            raise RuntimeError(f"confirmation initial loss differs: {arm['arm_id']}")
        for digest in (
            "client_partition_commitment_sha256", "sample_schedule_sha256",
            "supervision_schedule_sha256", "private_key_commitment", "rng_domain",
        ):
            if candidate.get(digest) != reference.get(digest):
                raise RuntimeError(f"confirmation pair differs: {arm['arm_id']}/{digest}")
        output.append({
            "model": candidate["model"], "seed": candidate["seed"],
            "candidate_method": candidate["method"],
            "privacy_label": candidate["privacy_label"],
            "fixed_arm_id": reference["arm_id"], "candidate_arm_id": candidate["arm_id"],
            "selected_fixed_C": reference["initial_clip_norm"],
            "selected_initial_C": candidate["initial_clip_norm"],
            "selected_eta": candidate["slaclip_eta"],
            "selected_beta_A": candidate["slaclip_beta_A"],
            "selected_beta_B": candidate["slaclip_beta_B"],
            "final_loss_fixed": reference["final_loss"],
            "final_loss_candidate": candidate["final_loss"],
            "final_loss_delta_candidate_minus_fixed": float(candidate["final_loss"]) - float(reference["final_loss"]),
            "normalized_loss_auc_fixed": reference["normalized_loss_auc"],
            "normalized_loss_auc_candidate": candidate["normalized_loss_auc"],
            "normalized_loss_auc_delta_candidate_minus_fixed": float(candidate["normalized_loss_auc"]) - float(reference["normalized_loss_auc"]),
            "actual_clipped_fraction_fixed": reference["actual_clipped_fraction"],
            "actual_clipped_fraction_candidate": candidate["actual_clipped_fraction"],
            "slaclip_baseline_calibration_lock_sha256": arm["slaclip_baseline_calibration_lock_sha256"],
            "slaclip_calibration_provenance": arm["slaclip_calibration_provenance"],
            "sample_schedule_sha256": candidate["sample_schedule_sha256"],
            "supervision_schedule_sha256": candidate["supervision_schedule_sha256"],
            "private_key_commitment": candidate["private_key_commitment"],
            "rng_domain": candidate["rng_domain"],
            "client_partition_commitment_sha256": candidate["client_partition_commitment_sha256"],
            "candidate_final_summary_sha256": candidate["final_summary_sha256"],
            "fixed_final_summary_sha256": reference["final_summary_sha256"],
            "candidate_run_config_sha256": candidate["run_config_sha256"],
            "fixed_run_config_sha256": reference["run_config_sha256"],
            "candidate_round_shard_prefix_sha256": candidate["round_shard_prefix_sha256"],
            "fixed_round_shard_prefix_sha256": reference["round_shard_prefix_sha256"],
        })
    return output


def _confirmation_aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODELS:
        for method in (NOISY_METHOD, ORACLE_METHOD):
            subset = [row for row in rows if row["model"] == model and row["candidate_method"] == method]
            record: dict[str, Any] = {"model": model, "candidate_method": method, "seed_count": len(subset)}
            for metric in (
                "final_loss_delta_candidate_minus_fixed",
                "normalized_loss_auc_delta_candidate_minus_fixed",
            ):
                for key, value in full.paired_inference([float(row[metric]) for row in subset]).items():
                    record[f"{metric}_{key}"] = value
            output.append(record)
    return output


def _oracle_vs_noisy_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(row["model"]), int(row["seed"]), str(row["method"])): row
        for row in rows if row["method"] in {NOISY_METHOD, ORACLE_METHOD}
        and int(row["seed"]) >= 400
    }
    metric_names = (
        "final_loss", "normalized_loss_auc", "actual_clipped_fraction_A",
        "actual_clipped_fraction_B", "final_threshold_A", "final_threshold_B",
        "cdf_error_rmse_median_A", "cdf_error_rmse_median_B",
        "oracle_direction_agreement_fraction_A", "oracle_direction_agreement_fraction_B",
        "log_threshold_total_variation_A", "log_threshold_total_variation_B",
    )
    output: list[dict[str, Any]] = []
    for model in MODELS:
        for seed in range(400, 420):
            noisy = by_key.get((model, seed, NOISY_METHOD))
            oracle = by_key.get((model, seed, ORACLE_METHOD))
            if noisy is None or oracle is None:
                continue
            for key in (
                "initial_loss", "client_partition_commitment_sha256",
                "sample_schedule_sha256", "supervision_schedule_sha256",
                "private_key_commitment", "rng_domain", "slaclip_beta_A",
                "slaclip_beta_B", "slaclip_eta", "initial_clip_norm",
                "slaclip_baseline_calibration_lock_sha256",
                "slaclip_calibration_provenance",
            ):
                if noisy.get(key) != oracle.get(key):
                    raise RuntimeError(f"oracle/noisy pairing differs: {model}/{seed}/{key}")
            record: dict[str, Any] = {
                "model": model, "seed": seed,
                "noisy_arm_id": noisy["arm_id"], "oracle_arm_id": oracle["arm_id"],
                "beta_A": noisy["slaclip_beta_A"], "beta_B": noisy["slaclip_beta_B"],
                "eta": noisy["slaclip_eta"], "initial_C": noisy["initial_clip_norm"],
                "calibration_lock_sha256": noisy["slaclip_baseline_calibration_lock_sha256"],
                "calibration_provenance": noisy["slaclip_calibration_provenance"],
                "noisy_final_summary_sha256": noisy["final_summary_sha256"],
                "oracle_final_summary_sha256": oracle["final_summary_sha256"],
                "noisy_run_config_sha256": noisy["run_config_sha256"],
                "oracle_run_config_sha256": oracle["run_config_sha256"],
                "noisy_round_shard_prefix_sha256": noisy["round_shard_prefix_sha256"],
                "oracle_round_shard_prefix_sha256": oracle["round_shard_prefix_sha256"],
            }
            for metric in metric_names:
                noisy_value = noisy.get(metric)
                oracle_value = oracle.get(metric)
                record[f"noisy_{metric}"] = noisy_value
                record[f"oracle_{metric}"] = oracle_value
                record[f"delta_oracle_minus_noisy_{metric}"] = (
                    float(oracle_value) - float(noisy_value)
                    if isinstance(noisy_value, (int, float))
                    and not isinstance(noisy_value, bool)
                    and isinstance(oracle_value, (int, float))
                    and not isinstance(oracle_value, bool)
                    else None
                )
            output.append(record)
    return output


def _oracle_vs_noisy_aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    delta_names = sorted({key for row in rows for key in row if key.startswith("delta_oracle_minus_noisy_")})
    output: list[dict[str, Any]] = []
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        record: dict[str, Any] = {"model": model, "seed_count": len(subset)}
        for name in delta_names:
            values = [float(row[name]) for row in subset if isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool)]
            for key, value in full.paired_inference(values).items():
                record[f"{name}_{key}"] = value
        output.append(record)
    return output


def aggregate_campaign(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec = load_spec(spec_path)
    master = full.load_runtime(root / MASTER_RUNTIME_NAME)
    stages: list[tuple[dict[str, Any], Sequence[Mapping[str, Any]]]] = [(master, master["arms"])]
    fixed_lock = selected_lock = stage2 = stage3 = None
    if (root / FIXED_LOCK_NAME).is_file():
        fixed_lock = _load_lock(root / FIXED_LOCK_NAME, "fixed selection lock")
        _validate_fixed_lock(fixed_lock, master, spec)
        stage2 = full.load_runtime(_ensure_stage2(root, spec_path, spec, master, fixed_lock))
        stages.append((stage2, stage2["arms"]))
    if (root / SLACLIP_LOCK_NAME).is_file():
        if fixed_lock is None or stage2 is None:
            raise RuntimeError("adaptive lock exists without fixed lock")
        selected_lock = _load_lock(root / SLACLIP_LOCK_NAME, "SlaClip selection lock")
        _validate_slaclip_lock(selected_lock, master, fixed_lock, stage2, spec)
        stage3 = full.load_runtime(_ensure_stage3(root, spec_path, spec, master, fixed_lock, selected_lock))
        stages.append((stage3, stage3["arms"]))
    status_counts = {"COMPLETED": 0, "FAILED": 0, "CHECKPOINTED_STOP": 0, "NOT_STARTED": 0, "OTHER": 0}
    rows: list[dict[str, Any]] = []
    for manifest, arms in stages:
        for arm in arms:
            path = root / "arm-status" / f"{arm['arm_id']}.json"
            status = "NOT_STARTED"
            if path.is_file():
                status = str(full.load_object(path, "arm status").get("status", "OTHER"))
            status_counts[status if status in status_counts else "OTHER"] += 1
            if status == "COMPLETED":
                _validate_adaptive_contract(root, manifest, arm)
                rows.append(_metric_row(root, manifest, arm))
    paired = _paired_rows(rows, stage3) if stage3 is not None else []
    aggregate = _confirmation_aggregate(paired)
    oracle_noisy = _oracle_vs_noisy_rows(rows) if stage3 is not None else []
    oracle_noisy_aggregate = _oracle_vs_noisy_aggregate(oracle_noisy)
    if args.require_complete:
        if fixed_lock is None or selected_lock is None or stage2 is None or stage3 is None:
            raise RuntimeError("strict aggregate requires both immutable selection locks")
        if len(rows) != 306 or status_counts["COMPLETED"] != 306:
            raise RuntimeError(f"strict aggregate has {len(rows)}/306 completed arms")
        if (
            len(paired) != 80
            or len(oracle_noisy) != 40
            or any(record["seed_count"] != 20 for record in aggregate)
            or any(record["seed_count"] != 20 for record in oracle_noisy_aggregate)
        ):
            raise RuntimeError("fresh-seed confirmation pairing is incomplete")
        staged._verify_locked_evidence(root, master, fixed_lock["source_evidence"])
        staged._verify_locked_evidence(root, stage2, selected_lock["source_evidence"])
    full.atomic_csv(root / "campaign_metrics.csv", rows, staged._columns(rows, (
        "stage", "arm_id", "method", "privacy_label", "model", "seed",
        "initial_clip_norm", "slaclip_eta", "slaclip_beta_A", "slaclip_beta_B",
        "final_loss", "best_loss", "normalized_loss_auc", "actual_clipped_fraction",
    )))
    full.atomic_csv(root / "confirmation_paired_metrics.csv", paired, staged._columns(paired, (
        "model", "seed", "candidate_method", "privacy_label", "selected_fixed_C",
        "selected_initial_C", "selected_eta", "selected_beta_A", "selected_beta_B",
        "final_loss_delta_candidate_minus_fixed", "normalized_loss_auc_delta_candidate_minus_fixed",
    )))
    full.atomic_csv(root / "confirmation_aggregate_metrics.csv", aggregate, staged._columns(aggregate, (
        "model", "candidate_method", "seed_count",
        "final_loss_delta_candidate_minus_fixed_mean",
        "final_loss_delta_candidate_minus_fixed_ci95_low",
        "final_loss_delta_candidate_minus_fixed_ci95_high",
    )))
    full.atomic_csv(root / "oracle_vs_noisy_paired_metrics.csv", oracle_noisy, staged._columns(oracle_noisy, (
        "model", "seed", "noisy_arm_id", "oracle_arm_id", "beta_A", "beta_B",
        "eta", "initial_C", "delta_oracle_minus_noisy_final_loss",
        "delta_oracle_minus_noisy_normalized_loss_auc",
        "delta_oracle_minus_noisy_final_threshold_A",
        "delta_oracle_minus_noisy_final_threshold_B",
    )))
    full.atomic_csv(root / "oracle_vs_noisy_aggregate_metrics.csv", oracle_noisy_aggregate, staged._columns(oracle_noisy_aggregate, (
        "model", "seed_count", "delta_oracle_minus_noisy_final_loss_mean",
        "delta_oracle_minus_noisy_final_loss_ci95_low",
        "delta_oracle_minus_noisy_final_loss_ci95_high",
    )))
    _write_trajectories(root, stages)
    complete = len(rows) == 306 and len(paired) == 80
    summary = {
        "schema_version": 1, "status": "COMPLETED" if complete else "IN_PROGRESS",
        "campaign_name": master["campaign_name"],
        "expected_total_model_arms": 306, "completed_model_arms": len(rows),
        "status_counts_for_materialized_arms": status_counts,
        "fixed_calibration_lock_sha256": fixed_lock.get("lock_sha256") if fixed_lock else None,
        "fixed_calibration_provenance": (
            "exact_NON_DP_fixed_trajectory_diagnostics_frozen_before_fresh_seeds"
            if fixed_lock else None
        ),
        "calibration_data_consumed_at_controller_runtime": False,
        "slaclip_selection_lock_sha256": selected_lock.get("lock_sha256") if selected_lock else None,
        "selected_models": selected_lock.get("models") if selected_lock else None,
        "confirmation_paired_rows": len(paired), "confirmation": aggregate,
        "oracle_vs_noisy_paired_rows": len(oracle_noisy),
        "oracle_vs_noisy": oracle_noisy_aggregate,
        "scientific_boundary": spec["scientific_boundary"],
        "warning": "Exact endpoint calibration and the matched oracle use private diagnostics and are NON_DP; K5/N5/sigma2 is mechanism development, not journal-level efficacy evidence.",
        "updated_at_utc": full.utc_now(),
    }
    full.atomic_json(root / "campaign_summary.json", summary)
    if args.require_complete and not complete:
        raise RuntimeError("strict final aggregate did not reach COMPLETED")
    print(f"campaign_summary={root / 'campaign_summary.json'}")


def validate_spec_command(args: argparse.Namespace) -> None:
    spec = load_spec(args.spec.resolve())
    print(json.dumps({
        "status": "VALID", "spec_sha256": full.sha256_file(args.spec.resolve()),
        "stage1_model_arm_count": len(fixed_stage_arms(spec)),
        "stage2_model_arm_count": 180, "stage3_model_arm_count": 120,
        "total_model_arm_count": 306, "models": list(MODELS), "K": 5,
        "learning_rate": 5e-4, "excluded_method": "SlaClip-Q",
    }, indent=2, sort_keys=True))


def print_waves(args: argparse.Namespace) -> None:
    runtime = full.load_runtime(args.manifest.resolve())
    for index in range(0, len(runtime["arms"]), 2):
        print(f"{index}\t{index + 1}")


def _identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-spec")
    validate.add_argument("--spec", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    _identity_args(prepare)
    prepare.add_argument("--private-key", type=Path, required=True)
    prepare.add_argument("--resume", action="store_true")
    fixed = commands.add_parser("lock-fixed")
    _identity_args(fixed)
    adaptive = commands.add_parser("lock-slaclip")
    _identity_args(adaptive)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--campaign-root", type=Path, required=True)
    aggregate.add_argument("--spec", type=Path, required=True)
    aggregate.add_argument("--require-complete", action="store_true")
    waves = commands.add_parser("waves")
    waves.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    args = parse_args(argv)
    if args.command == "validate-spec":
        validate_spec_command(args)
    elif args.command == "prepare":
        prepare_campaign(args)
    elif args.command == "lock-fixed":
        lock_fixed_selection(args)
    elif args.command == "lock-slaclip":
        lock_slaclip_selection(args)
    elif args.command == "aggregate":
        aggregate_campaign(args)
    elif args.command == "waves":
        print_waves(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
