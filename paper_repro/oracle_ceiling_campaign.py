#!/usr/bin/env python3
"""Strong groupwise-fixed versus exact Full-SlaClip oracle ceiling campaign.

The campaign is deliberately a NON-DP mechanism gate.  It first selects a
strong fixed pair ``(C_A, C_B)`` on development-only seeds, freezes A/B
stationary-beta grids from exact fixed trajectories, selects an exact-endpoint
Full-SlaClip controller, and finally compares the frozen controller with the
strong groupwise-fixed baseline on fresh seeds.  No noisy-controller efficacy
claim and no SlaClip-Q method are represented by this module.
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
    from paper_repro import groupwise_slaclip_campaign as groupwise
    from paper_repro import staged_slaclip_campaign as staged
    from paper_repro.slaclip import build_slack_vector, normalize_noisy_slack
except ModuleNotFoundError:  # direct-script execution
    import full_slaclip_campaign as full  # type: ignore[no-redef]
    import groupwise_slaclip_campaign as groupwise  # type: ignore[no-redef]
    import staged_slaclip_campaign as staged  # type: ignore[no-redef]
    from slaclip import build_slack_vector, normalize_noisy_slack  # type: ignore[no-redef]


SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1
MODELS = ("bert", "gpt2")
GROUPS = ("A", "B")
FIXED_METHOD = full.FIXED_DP_METHOD
ORACLE_METHOD = full.ORACLE_SLACLIP_METHOD
NOISY_METHOD = full.FULL_SLACLIP_METHOD
FIXED_COARSE_STAGE = "fixed_coarse"
FIXED_REFINEMENT_STAGE = "fixed_refinement"
ORACLE_DEVELOPMENT_STAGE = "oracle_development"
CONFIRMATION_STAGE = "confirmation"
MASTER_RUNTIME_NAME = "runtime-manifest.json"
PREFLIGHT_RUNTIME_NAME = "preflight-runtime-manifest.json"
STAGE2_RUNTIME_NAME = "stage2-fixed-refinement-runtime-manifest.json"
STAGE3_RUNTIME_NAME = "stage3-oracle-development-runtime-manifest.json"
STAGE4_RUNTIME_NAME = "stage4-confirmation-runtime-manifest.json"
COARSE_LOCK_NAME = "groupwise-fixed-coarse-selection.lock.json"
FIXED_LOCK_NAME = "strong-groupwise-fixed-selection.lock.json"
ORACLE_LOCK_NAME = "oracle-ceiling-selection.lock.json"
GATE_LOCK_NAME = "oracle-ceiling-gate.lock.json"
CALIBRATION_NAME = "strong_groupwise_fixed_beta_calibration.csv"
CALIBRATION_PROVENANCE = (
    "exact_NON_DP_strong_groupwise_fixed_trajectory_diagnostics_"
    "frozen_before_oracle_development"
)
EXPECTED_COUNTS = {
    FIXED_COARSE_STAGE: 60,
    FIXED_REFINEMENT_STAGE: 18,
    ORACLE_DEVELOPMENT_STAGE: 168,
    CONFIRMATION_STAGE: 80,
    "total": 326,
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{label} is outside its domain")
    return value


def load_spec(path: Path) -> dict[str, Any]:
    spec = full.load_object(path, "oracle-ceiling campaign specification")
    _exact_keys(
        spec,
        {
            "schema_version", "campaign_name", "description",
            "expected_stage_arm_counts", "common", "fixed_coarse",
            "fixed_refinement", "oracle_development", "confirmation",
            "scientific_boundary",
        },
        "specification",
    )
    if spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported oracle-ceiling campaign schema")
    if spec["expected_stage_arm_counts"] != EXPECTED_COUNTS:
        raise ValueError("oracle-ceiling arm counts differ")
    common = spec["common"]
    if tuple(common.get("models", ())) != MODELS:
        raise ValueError("common.models must be bert then gpt2")
    for name in (
        "num_clients", "rounds", "batch_size", "rank", "max_seq_length",
        "max_validation_records", "eval_every", "checkpoint_every",
        "data_split_seed", "evaluation_seed", "slaclip_num_slots",
    ):
        _integer(common.get(name), f"common.{name}", positive=True)
    for name in (
        "noise_multiplier", "learning_rate", "delta", "slaclip_c_min",
        "slaclip_c_max", "slaclip_endpoint_epsilon",
    ):
        _finite_number(common.get(name), f"common.{name}", positive=True)
    if (
        common["num_clients"] != 5
        or common["rounds"] != 50
        or common["batch_size"] != 8
        or common["noise_multiplier"] != 2.0
        or common["learning_rate"] != 5e-4
        or common["slaclip_num_slots"] != 5
    ):
        raise ValueError("paper/mechanism constants differ")

    coarse = spec["fixed_coarse"]
    if (
        coarse.get("method") != FIXED_METHOD
        or coarse.get("seeds") != [480, 481]
        or coarse.get("top_k_per_model") != 3
    ):
        raise ValueError("fixed-coarse policy differs")
    grids = coarse.get("clip_norm_grid_by_model")
    if not isinstance(grids, dict) or tuple(grids) != MODELS:
        raise ValueError("fixed groupwise grids differ")
    for model in MODELS:
        if not isinstance(grids[model], dict) or set(grids[model]) != set(GROUPS):
            raise ValueError("fixed grid must contain A and B")
        for group in GROUPS:
            values = grids[model][group]
            if (
                not isinstance(values, list)
                or len(values) != (5 if group == "A" else 3)
                or len(set(values)) != len(values)
            ):
                raise ValueError("fixed grid cardinality differs")
            for value in values:
                _finite_number(value, f"{model}.{group} fixed C", positive=True)
    refinement = spec["fixed_refinement"]
    if refinement.get("seeds") != [490, 491, 492]:
        raise ValueError("fixed-refinement seeds differ")
    calibration = refinement.get("beta_calibration")
    if (
        not isinstance(calibration, dict)
        or calibration.get("groups") != ["A", "B"]
        or calibration.get("quantiles") != [0.1, 0.5, 0.9]
    ):
        raise ValueError("beta-calibration policy differs")
    oracle = spec["oracle_development"]
    if (
        oracle.get("method") != ORACLE_METHOD
        or oracle.get("seeds") != [500, 501, 502]
        or oracle.get("etas") != [0.01, 0.05, 0.2]
    ):
        raise ValueError("oracle-development policy differs")
    confirmation = spec["confirmation"]
    if (
        confirmation.get("methods") != [FIXED_METHOD, ORACLE_METHOD]
        or confirmation.get("seeds") != list(range(600, 620))
        or confirmation.get("oracle_privacy_label")
        != "NON_DP_PRIVATE_DIAGNOSTIC"
        or confirmation.get("per_model_alpha_bonferroni_for_two_models")
        != 0.025
        or confirmation.get("gate_requires") != [
            "paired_final_loss_mean_below_negative_MPRD",
            "paired_final_loss_ci95_high_below_zero",
            "exact_sign_flip_p_below_bonferroni_0.025",
            "paired_normalized_loss_auc_mean_not_positive",
            "controller_instability_events_per_seed_at_most_10",
        ]
    ):
        raise ValueError("confirmation policy differs")
    all_seed_sets = [
        set(coarse["seeds"]), set(refinement["seeds"]),
        set(oracle["seeds"]), set(confirmation["seeds"]),
    ]
    if any(left & right for i, left in enumerate(all_seed_sets) for right in all_seed_sets[i + 1 :]):
        raise ValueError("campaign stage seeds overlap")
    boundary = spec["scientific_boundary"]
    if (
        boundary.get("excluded_method_family") != "SlaClip-Q"
        or boundary.get("noisy_slaclip_efficacy_tested") is not False
        or boundary.get("single_allocation") is not True
        or boundary.get("nested_sbatch_or_array") is not False
    ):
        raise ValueError("scientific boundary differs")
    return spec


def _token(value: float) -> str:
    return full.number_token(float(value))


def _fixed_id(prefix: str, model: str, c_a: float, c_b: float, seed: int) -> str:
    return f"{prefix}-fixed-{model}-ca{_token(c_a)}-cb{_token(c_b)}-s{seed}"


def _base_arm(
    spec: Mapping[str, Any], *, arm_id: str, stage: str, method: str,
    model: str, seed: int, c_a: float, c_b: float,
    reference_arm_id: str | None, eta: float | None = None,
    betas: Mapping[str, float] | None = None,
    calibration_lock_sha256: str | None = None,
    calibration_provenance: str | None = None,
    require_calibration: bool = True,
) -> dict[str, Any]:
    adaptive = method in {NOISY_METHOD, ORACLE_METHOD}
    if adaptive != (eta is not None and betas is not None):
        raise ValueError("adaptive method and controller arguments disagree")
    if adaptive and set(betas or {}) != set(GROUPS):
        raise ValueError("adaptive targets must contain A and B")
    if adaptive and require_calibration and (
        not isinstance(calibration_lock_sha256, str)
        or len(calibration_lock_sha256) != 64
        or calibration_provenance != CALIBRATION_PROVENANCE
    ):
        raise ValueError("oracle arm lacks frozen calibration provenance")
    common = spec["common"]
    roles = {
        FIXED_COARSE_STAGE: "strong_groupwise_fixed_coarse_development",
        FIXED_REFINEMENT_STAGE: "strong_groupwise_fixed_fresh_seed_refinement",
        ORACLE_DEVELOPMENT_STAGE: "exact_endpoint_oracle_development",
        CONFIRMATION_STAGE: "fresh_seed_strong_fixed_vs_exact_oracle_confirmation",
    }
    if stage not in roles:
        raise ValueError(f"unsupported campaign stage: {stage}")
    targets = {group: float(betas[group]) for group in GROUPS} if betas else None
    if targets and any(not 0.0 <= value <= 1.0 for value in targets.values()):
        raise ValueError("groupwise beta lies outside [0,1]")
    return {
        "arm_id": arm_id,
        "stage": stage,
        "family": stage,
        "analysis_role": roles[stage],
        "method": method,
        "seed": int(seed),
        # The legacy scalar is retained for older analysis tables; the A/B map
        # is authoritative for training, comparison, and fingerprints.
        "initial_clip_norm": float(c_b),
        "initial_clip_norm_by_group": {"A": float(c_a), "B": float(c_b)},
        "slaclip_eta": float(eta) if eta is not None else None,
        "slaclip_base_target_clipped_fraction": None,
        "slaclip_beta": None,
        "slaclip_base_target_clipped_fraction_by_group": targets,
        "slaclip_beta_by_group": dict(targets) if targets else None,
        "slaclip_baseline_calibration_lock_sha256": calibration_lock_sha256,
        "acknowledge_slaclip_baseline_calibration_is_non_dp": bool(calibration_lock_sha256),
        "slaclip_calibration_provenance": calibration_provenance,
        "controller_input": full.CONTROLLER_INPUT_BY_METHOD.get(method),
        "reference_arm_id": reference_arm_id,
        "rng_domain": f"oracle-ceiling:s{seed}",
        "models": [model],
        "num_clients": common["num_clients"],
        "rounds": common["rounds"],
        "batch_size": common["batch_size"],
        "noise_multiplier": common["noise_multiplier"],
        "learning_rate": common["learning_rate"],
        "rank": common["rank"],
        "max_seq_length": common["max_seq_length"],
        "max_validation_records": common["max_validation_records"],
        "eval_every": common["eval_every"],
        "checkpoint_every": common["checkpoint_every"],
        "data_split_seed": common["data_split_seed"],
        "evaluation_seed": common["evaluation_seed"],
        "delta": common["delta"],
        "slaclip_num_slots": common["slaclip_num_slots"] if adaptive else None,
        "slaclip_c_min": common["slaclip_c_min"] if adaptive else None,
        "slaclip_c_max": common["slaclip_c_max"] if adaptive else None,
    }


def fixed_coarse_arms(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    grids = spec["fixed_coarse"]["clip_norm_grid_by_model"]
    for model in MODELS:
        for c_a in grids[model]["A"]:
            for c_b in grids[model]["B"]:
                for seed in spec["fixed_coarse"]["seeds"]:
                    arms.append(_base_arm(
                        spec,
                        arm_id=_fixed_id("coarse", model, c_a, c_b, seed),
                        stage=FIXED_COARSE_STAGE, method=FIXED_METHOD,
                        model=model, seed=seed, c_a=c_a, c_b=c_b,
                        reference_arm_id=None,
                    ))
    return staged._indexed(arms)


def _runtime(
    spec: Mapping[str, Any], spec_path: Path, repository_sha: str,
    input_manifest: Path, created_at_utc: str,
    arms: list[dict[str, Any]], stage: str,
    parent_lock_sha256: str | None,
) -> dict[str, Any]:
    value = {
        "schema_version": full.SCHEMA_VERSION,
        "campaign_name": spec["campaign_name"],
        "stage": stage,
        "created_at_utc": created_at_utc,
        "repository_sha": repository_sha,
        "spec_sha256": full.sha256_file(spec_path),
        "input_manifest_path": str(input_manifest.resolve()),
        "input_manifest_sha256": full.sha256_file(input_manifest),
        "parent_selection_lock_sha256": parent_lock_sha256,
        "expected_arm_count": len(arms),
        "scientific_boundary": spec["scientific_boundary"],
        "arms": arms,
    }
    value["manifest_sha256"] = full.sha256_bytes(full.canonical_bytes(value))
    full.validate_runtime_manifest(value)
    return value


def _master(
    spec: Mapping[str, Any], spec_path: Path, sha: str,
    inputs: Path, created: str,
) -> dict[str, Any]:
    return _runtime(
        spec, spec_path, sha, inputs, created,
        fixed_coarse_arms(spec), FIXED_COARSE_STAGE, None,
    )


def _preflight(master: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    fixed = _base_arm(
        spec, arm_id="preflight-template-fixed", stage=FIXED_COARSE_STAGE,
        method=FIXED_METHOD, model="bert", seed=480,
        # Keep legacy B/scalar C=10 so the common real-model smoke resolver
        # recognizes the template, while A=1 exercises the unequal groupwise
        # threshold path before any formal arm is launched.
        c_a=1.0, c_b=10.0, reference_arm_id=None,
    )
    adaptive = _base_arm(
        # run_preflight_smoke deliberately derives the exact-endpoint oracle
        # smoke from this noisy Full-SlaClip template after validating the
        # common paper-style C=10 preflight contract.
        spec, arm_id="preflight-template-slaclip", stage=ORACLE_DEVELOPMENT_STAGE,
        method=NOISY_METHOD, model="bert", seed=500,
        c_a=1.0, c_b=10.0, reference_arm_id=fixed["arm_id"],
        eta=0.05, betas={"A": 0.5, "B": 0.5},
        require_calibration=False,
    )
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


def _write_or_verify(path: Path, value: Mapping[str, Any], label: str) -> None:
    staged._write_or_verify(path, value, label)


def _load_lock(path: Path, label: str) -> dict[str, Any]:
    value = full.load_object(path, label)
    staged._validate_lock(value, label)
    return value


def _identity(
    root: Path, spec_path: Path, repository: Path,
    expected_sha: str, inputs: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = load_spec(spec_path)
    if full.repository_sha(repository) != expected_sha or full.repository_dirty(repository):
        raise RuntimeError("repository snapshot SHA/cleanliness gate failed")
    manifest = full.load_runtime(root / MASTER_RUNTIME_NAME)
    candidate = _master(
        spec, spec_path, expected_sha, inputs,
        str(manifest.get("created_at_utc")),
    )
    if manifest != candidate:
        raise RuntimeError("master runtime manifest differs from immutable inputs")
    return spec, manifest


def _metric_row(
    root: Path, manifest: Mapping[str, Any], arm: Mapping[str, Any],
) -> dict[str, Any]:
    if arm["method"] == FIXED_METHOD:
        _validate_fixed_groupwise_threshold_evidence(root, arm)
    else:
        groupwise._validate_adaptive_contract(root, manifest, arm)
    row = groupwise._metric_row(root, manifest, arm)
    initial = arm["initial_clip_norm_by_group"]
    row["initial_clip_norm_A"] = float(initial["A"])
    row["initial_clip_norm_B"] = float(initial["B"])
    return row


def _validate_fixed_groupwise_threshold_evidence(
    root: Path, arm: Mapping[str, Any],
) -> None:
    expected = {
        group: float(arm["initial_clip_norm_by_group"][group])
        for group in GROUPS
    }
    arm_root = root / "arms" / str(arm["arm_id"])
    run_config = full.load_object(arm_root / "run_config.json", "fixed run config")
    try:
        effective = run_config["effective_config"]
        algorithm = run_config["scientific_contract"]["algorithm_contract"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"fixed groupwise run contract is missing: {arm['arm_id']}"
        ) from error
    if (
        not isinstance(effective, dict)
        or effective.get("clip_norm_A") != expected["A"]
        or effective.get("clip_norm_B") != expected["B"]
        or not isinstance(algorithm, dict)
        or algorithm.get("initial_clip_threshold_by_group") != expected
        or algorithm.get("groupwise_fixed_thresholds")
        != (expected["A"] != expected["B"])
    ):
        raise RuntimeError(
            f"fixed groupwise persisted contract differs: {arm['arm_id']}"
        )
    model = str(arm["models"][0])
    expected_noise = {
        group: float(arm["noise_multiplier"]) * expected[group]
        for group in GROUPS
    }
    for round_index in range(1, int(arm["rounds"]) + 1):
        shard = full.load_object(
            _round_path(root, arm, round_index), "fixed groupwise round shard"
        )
        records = shard.get("client_records")
        if (
            shard.get("round") != round_index
            or shard.get("model") != model
            or shard.get("method") != FIXED_METHOD
            or not isinstance(records, list)
            or len(records) != int(arm["num_clients"])
        ):
            raise RuntimeError(
                f"fixed groupwise round identity differs: "
                f"{arm['arm_id']}/{round_index}"
            )
        for record in records:
            groups = record.get("gradient_groups")
            if not isinstance(groups, dict) or set(groups) != set(GROUPS):
                raise RuntimeError("fixed groupwise client telemetry differs")
            for group in GROUPS:
                telemetry = groups[group]
                if (
                    not isinstance(telemetry, dict)
                    or float(telemetry.get("clip_threshold", math.nan))
                    != expected[group]
                    or float(telemetry.get("noise_std_per_coordinate", math.nan))
                    != expected_noise[group]
                ):
                    raise RuntimeError(
                        f"fixed groupwise threshold/noise evidence differs: "
                        f"{arm['arm_id']}/{round_index}/{group}"
                    )


def _candidate_rankings(
    rows: Sequence[Mapping[str, Any]], *, seeds: Sequence[int],
    candidates: Sequence[tuple[float, float]], noise_multiplier: float,
) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for c_a, c_b in candidates:
        subset = [
            row for row in rows
            if float(row["initial_clip_norm_A"]) == float(c_a)
            and float(row["initial_clip_norm_B"]) == float(c_b)
        ]
        if {int(row["seed"]) for row in subset} != set(seeds):
            raise RuntimeError(f"fixed groupwise candidate is incomplete: {c_a}/{c_b}")
        final = [float(row["final_loss"]) for row in subset]
        auc = [float(row["normalized_loss_auc"]) for row in subset]
        rankings.append({
            "C_A": float(c_a), "C_B": float(c_b),
            "seed_count": len(subset),
            "mean_final_loss": statistics.fmean(final),
            "mean_normalized_loss_auc": statistics.fmean(auc),
            "final_loss_sample_std": statistics.stdev(final),
            "groupwise_noise_scale_l2": float(
                noise_multiplier * math.sqrt(c_a * c_a + c_b * c_b)
            ),
            "mean_actual_clipped_fraction": statistics.fmean(
                float(row["actual_clipped_fraction"]) for row in subset
            ),
        })
    rankings.sort(key=lambda row: (
        row["mean_final_loss"], row["mean_normalized_loss_auc"],
        row["final_loss_sample_std"], row["groupwise_noise_scale_l2"],
        row["C_A"], row["C_B"],
    ))
    return rankings


def _validate_coarse_lock(
    lock: Mapping[str, Any], master: Mapping[str, Any], spec: Mapping[str, Any],
) -> None:
    if (
        lock.get("status") != "GROUPWISE_FIXED_COARSE_SELECTION_LOCKED"
        or lock.get("master_runtime_manifest_sha256") != master["manifest_sha256"]
        or lock.get("spec_sha256") != master["spec_sha256"]
        or lock.get("selection_rule") != spec["fixed_coarse"]["selection_rule"]
        or lock.get("development_seeds") != spec["fixed_coarse"]["seeds"]
        or lock.get("later_stage_data_accessed") is not False
    ):
        raise RuntimeError("coarse fixed lock identity differs")
    models = lock.get("models")
    if not isinstance(models, dict) or tuple(models) != MODELS:
        raise RuntimeError("coarse fixed lock model set differs")
    for model in MODELS:
        if not isinstance(models[model], dict):
            raise RuntimeError("coarse fixed model record differs")
        top = models[model].get("top_candidates")
        ordered = models[model].get("ordered_candidates")
        grids = spec["fixed_coarse"]["clip_norm_grid_by_model"][model]
        expected_pairs = {
            (float(c_a), float(c_b))
            for c_a in grids["A"] for c_b in grids["B"]
        }
        if (
            not isinstance(top, list)
            or len(top) != spec["fixed_coarse"]["top_k_per_model"]
            or not isinstance(ordered, list)
            or len(ordered) != len(expected_pairs)
            or top != ordered[: len(top)]
            or {
                (float(item["C_A"]), float(item["C_B"]))
                for item in ordered
            } != expected_pairs
            or ordered != sorted(
                ordered,
                key=lambda row: (
                    row["mean_final_loss"], row["mean_normalized_loss_auc"],
                    row["final_loss_sample_std"], row["groupwise_noise_scale_l2"],
                    row["C_A"], row["C_B"],
                ),
            )
        ):
            raise RuntimeError("coarse fixed top-three set differs")


def fixed_refinement_arms(
    spec: Mapping[str, Any], coarse_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for model in MODELS:
        for candidate in coarse_lock["models"][model]["top_candidates"]:
            c_a, c_b = float(candidate["C_A"]), float(candidate["C_B"])
            for seed in spec["fixed_refinement"]["seeds"]:
                arms.append(_base_arm(
                    spec,
                    arm_id=_fixed_id("refine", model, c_a, c_b, seed),
                    stage=FIXED_REFINEMENT_STAGE, method=FIXED_METHOD,
                    model=model, seed=seed, c_a=c_a, c_b=c_b,
                    reference_arm_id=None,
                ))
    return staged._indexed(arms)


def _ensure_stage2(
    root: Path, spec_path: Path, spec: Mapping[str, Any],
    master: Mapping[str, Any], coarse_lock: Mapping[str, Any],
) -> Path:
    _validate_coarse_lock(coarse_lock, master, spec)
    staged._verify_locked_evidence(root, master, coarse_lock.get("source_evidence"))
    candidate = _runtime(
        spec, spec_path, str(master["repository_sha"]),
        Path(str(master["input_manifest_path"])), str(master["created_at_utc"]),
        fixed_refinement_arms(spec, coarse_lock), FIXED_REFINEMENT_STAGE,
        str(coarse_lock["lock_sha256"]),
    )
    if len(candidate["arms"]) != EXPECTED_COUNTS[FIXED_REFINEMENT_STAGE]:
        raise RuntimeError("fixed-refinement arm count differs")
    path = root / STAGE2_RUNTIME_NAME
    _write_or_verify(path, candidate, "fixed-refinement manifest")
    return path


def lock_coarse(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, master = _identity(
        root, spec_path, args.repository.resolve(), args.expected_code_sha,
        args.input_manifest.resolve(),
    )
    path = root / COARSE_LOCK_NAME
    if path.is_file():
        lock = _load_lock(path, "coarse fixed lock")
        _validate_coarse_lock(lock, master, spec)
        stage2 = _ensure_stage2(root, spec_path, spec, master, lock)
        print(f"coarse_selection_reused={path}")
        print(f"stage2_runtime_manifest={stage2}")
        return
    evidence = staged._arm_evidence(root, master)
    rows = [_metric_row(root, master, arm) for arm in master["arms"]]
    models: dict[str, Any] = {}
    grids = spec["fixed_coarse"]["clip_norm_grid_by_model"]
    for model in MODELS:
        candidates = [
            (float(c_a), float(c_b))
            for c_a in grids[model]["A"] for c_b in grids[model]["B"]
        ]
        rankings = _candidate_rankings(
            [row for row in rows if row["model"] == model],
            seeds=spec["fixed_coarse"]["seeds"], candidates=candidates,
            noise_multiplier=float(spec["common"]["noise_multiplier"]),
        )
        models[model] = {
            "top_candidates": rankings[:3],
            "ordered_candidates": rankings,
        }
    lock = staged._lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "GROUPWISE_FIXED_COARSE_SELECTION_LOCKED",
        "campaign_name": master["campaign_name"],
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "spec_sha256": master["spec_sha256"],
        "selection_rule": spec["fixed_coarse"]["selection_rule"],
        "development_seeds": spec["fixed_coarse"]["seeds"],
        "later_stage_data_accessed": False,
        "models": models,
        "source_evidence": evidence,
        "created_at_utc": full.utc_now(),
    })
    full.atomic_json(path, lock)
    lock = _load_lock(path, "coarse fixed lock")
    _validate_coarse_lock(lock, master, spec)
    stage2 = _ensure_stage2(root, spec_path, spec, master, lock)
    print(f"coarse_selection_lock={path}")
    print(f"stage2_runtime_manifest={stage2}")


def _round_path(root: Path, arm: Mapping[str, Any], round_index: int) -> Path:
    return (
        root / "arms" / str(arm["arm_id"]) / str(arm["models"][0]) /
        "private_diagnostics" / "rounds" / f"round-{round_index:05d}.json"
    )


def _calibrate_groupwise(
    root: Path, arms: Sequence[Mapping[str, Any]], *,
    thresholds: Mapping[str, float], num_slots: int,
    epsilon: float, rounds: int,
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
                or not isinstance(records, list)
                or not records
                or not isinstance(summary, dict)
            ):
                raise RuntimeError(
                    f"fixed trajectory shard is invalid: {arm['arm_id']}/{round_index}"
                )
            for group in GROUPS:
                threshold = float(thresholds[group])
                group_records = [record["gradient_groups"][group] for record in records]
                if any(
                    float(record.get("clip_threshold", math.nan)) != threshold
                    for record in group_records
                ):
                    raise RuntimeError(
                        f"fixed groupwise threshold evidence differs: "
                        f"{arm['arm_id']}/{round_index}/{group}"
                    )
                norms = [float(record["raw_norm"]) for record in group_records]
                signals = [build_slack_vector(norm, threshold, num_slots) for norm in norms]
                signal_sum = [
                    math.fsum(vector[slot] for vector in signals)
                    for slot in range(num_slots)
                ]
                exact = normalize_noisy_slack(
                    signal_sum, threshold, num_slots, len(records)
                )
                beta, z_value = groupwise.stationary_beta(
                    float(exact[0]), float(exact[-1]), threshold, epsilon
                )
                values[group].append(beta)
                actual = float(summary[group]["clipped_fraction"])
                target = beta * (1.0 - z_value)
                # The first K-slot endpoint is a piecewise-linear near-C
                # proxy, not the empirical unclipped fraction.  Preserve the
                # difference as a controller-approximation diagnostic; it is
                # intentionally not treated as an identity or a failure gate.
                proxy_bias = actual - (1.0 - float(exact[0]))
                rows.append({
                    "arm_id": arm["arm_id"], "model": model,
                    "seed": arm["seed"], "round": round_index,
                    "group": group, "fixed_C_group": threshold,
                    "exact_q_endpoint_1": float(exact[0]),
                    "exact_r_endpoint_K": float(exact[-1]),
                    "z_r_over_C_plus_epsilon": z_value,
                    "stationary_beta": beta,
                    "actual_clipped_fraction": actual,
                    "stationary_dynamic_target_clipped": target,
                    "tracking_bias_actual_minus_target": actual - target,
                    "actual_minus_one_minus_near_threshold_proxy": proxy_bias,
                })
    return values, rows


def _three_point_beta_grid(values: Sequence[float]) -> tuple[list[float], bool]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite or any(not 0.0 <= value <= 1.0 for value in finite):
        raise RuntimeError("stationary-beta calibration values are invalid")
    selected: list[float] = []
    for probability in (0.1, 0.5, 0.9):
        value = float(staged._linear_quantile(finite, probability))
        if value not in selected:
            selected.append(value)
    fallback = len(selected) < 3
    median = float(staged._linear_quantile(finite, 0.5))
    candidates = [
        min(1.0, max(0.0, median - 0.25)),
        min(1.0, max(0.0, median + 0.25)),
        0.0, 0.5, 1.0,
    ]
    while len(selected) < 3:
        available = [value for value in candidates if value not in selected]
        if not available:
            raise RuntimeError("could not construct three unique beta candidates")
        value = max(
            available,
            key=lambda candidate: (
                min(abs(candidate - present) for present in selected)
                if selected else 1.0,
                -candidate,
            ),
        )
        selected.append(value)
    return sorted(selected), fallback


def _validate_fixed_lock(
    lock: Mapping[str, Any], master: Mapping[str, Any],
    stage2: Mapping[str, Any], spec: Mapping[str, Any],
) -> None:
    if (
        lock.get("status") != "STRONG_GROUPWISE_FIXED_SELECTION_LOCKED"
        or lock.get("master_runtime_manifest_sha256") != master["manifest_sha256"]
        or lock.get("stage2_runtime_manifest_sha256") != stage2["manifest_sha256"]
        or lock.get("selection_rule") != spec["fixed_refinement"]["selection_rule"]
        or lock.get("calibration_provenance") != CALIBRATION_PROVENANCE
        or lock.get("spec_sha256") != master["spec_sha256"]
        or lock.get("development_seeds") != {
            FIXED_COARSE_STAGE: spec["fixed_coarse"]["seeds"],
            FIXED_REFINEMENT_STAGE: spec["fixed_refinement"]["seeds"],
        }
        or lock.get("oracle_or_confirmation_data_accessed") is not False
        or lock.get("calibration_privacy_label")
        != "NON_DP_PRIVATE_DIAGNOSTIC"
        or lock.get("calibration_data_consumed_at_controller_runtime") is not False
    ):
        raise RuntimeError("strong groupwise-fixed lock identity differs")
    models = lock.get("models")
    if not isinstance(models, dict) or tuple(models) != MODELS:
        raise RuntimeError("strong fixed lock model set differs")
    for model in MODELS:
        record = models[model]
        if not isinstance(record, dict):
            raise RuntimeError("strong fixed model record differs")
        selected = record.get("selected_fixed_C_by_group", {})
        if not isinstance(selected, dict):
            raise RuntimeError("strong fixed threshold record differs")
        selected_pair = (
            float(selected.get("A", math.nan)),
            float(selected.get("B", math.nan)),
        )
        allowed_pairs = {
            (
                float(arm["initial_clip_norm_by_group"]["A"]),
                float(arm["initial_clip_norm_by_group"]["B"]),
            )
            for arm in stage2["arms"] if arm["models"] == [model]
        }
        rankings = record.get("ordered_refinement_candidates")
        beta_grids = [record.get(f"beta_{group}_grid") for group in GROUPS]
        if (
            set(selected) != set(GROUPS)
            or selected_pair not in allowed_pairs
            or not isinstance(rankings, list)
            or len(rankings) != 3
            or (float(rankings[0]["C_A"]), float(rankings[0]["C_B"]))
            != selected_pair
            or any(
                not isinstance(grid, list)
                or len(grid) != 3
                or grid != sorted(grid)
                or len(set(grid)) != 3
                or any(not 0.0 <= float(value) <= 1.0 for value in grid)
                for grid in beta_grids
            )
        ):
            raise RuntimeError("strong fixed lock model record differs")


def oracle_development_arms(
    spec: Mapping[str, Any], fixed_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for model in MODELS:
        record = fixed_lock["models"][model]
        thresholds = record["selected_fixed_C_by_group"]
        c_a, c_b = float(thresholds["A"]), float(thresholds["B"])
        for seed in spec["oracle_development"]["seeds"]:
            fixed_id = _fixed_id("oracledev", model, c_a, c_b, seed)
            arms.append(_base_arm(
                spec, arm_id=fixed_id, stage=ORACLE_DEVELOPMENT_STAGE,
                method=FIXED_METHOD, model=model, seed=seed,
                c_a=c_a, c_b=c_b, reference_arm_id=None,
            ))
            for beta_a in record["beta_A_grid"]:
                for beta_b in record["beta_B_grid"]:
                    for eta in spec["oracle_development"]["etas"]:
                        arms.append(_base_arm(
                            spec,
                            arm_id=(
                                f"oracledev-oracle-NONDP-{model}-ca{_token(c_a)}-"
                                f"cb{_token(c_b)}-ba{_token(beta_a)}-"
                                f"bb{_token(beta_b)}-e{_token(eta)}-s{seed}"
                            ),
                            stage=ORACLE_DEVELOPMENT_STAGE,
                            method=ORACLE_METHOD, model=model, seed=seed,
                            c_a=c_a, c_b=c_b, reference_arm_id=fixed_id,
                            eta=eta, betas={"A": beta_a, "B": beta_b},
                            calibration_lock_sha256=str(fixed_lock["lock_sha256"]),
                            calibration_provenance=CALIBRATION_PROVENANCE,
                        ))
    return staged._indexed(arms)


def _ensure_stage3(
    root: Path, spec_path: Path, spec: Mapping[str, Any],
    master: Mapping[str, Any], stage2: Mapping[str, Any],
    fixed_lock: Mapping[str, Any],
) -> Path:
    _validate_fixed_lock(fixed_lock, master, stage2, spec)
    staged._verify_locked_evidence(root, master, fixed_lock["source_evidence"][FIXED_COARSE_STAGE])
    staged._verify_locked_evidence(root, stage2, fixed_lock["source_evidence"][FIXED_REFINEMENT_STAGE])
    calibration = root / CALIBRATION_NAME
    if not calibration.is_file() or full.sha256_file(calibration) != fixed_lock.get("calibration_csv_sha256"):
        raise RuntimeError("strong fixed calibration CSV differs")
    candidate = _runtime(
        spec, spec_path, str(master["repository_sha"]),
        Path(str(master["input_manifest_path"])), str(master["created_at_utc"]),
        oracle_development_arms(spec, fixed_lock), ORACLE_DEVELOPMENT_STAGE,
        str(fixed_lock["lock_sha256"]),
    )
    if len(candidate["arms"]) != EXPECTED_COUNTS[ORACLE_DEVELOPMENT_STAGE]:
        raise RuntimeError("oracle-development arm count differs")
    path = root / STAGE3_RUNTIME_NAME
    _write_or_verify(path, candidate, "oracle-development manifest")
    return path


def lock_fixed(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, master = _identity(
        root, spec_path, args.repository.resolve(), args.expected_code_sha,
        args.input_manifest.resolve(),
    )
    coarse = _load_lock(root / COARSE_LOCK_NAME, "coarse fixed lock")
    _validate_coarse_lock(coarse, master, spec)
    stage2 = full.load_runtime(_ensure_stage2(root, spec_path, spec, master, coarse))
    path = root / FIXED_LOCK_NAME
    if path.is_file():
        lock = _load_lock(path, "strong fixed lock")
        _validate_fixed_lock(lock, master, stage2, spec)
        stage3 = _ensure_stage3(root, spec_path, spec, master, stage2, lock)
        print(f"strong_fixed_selection_reused={path}")
        print(f"stage3_runtime_manifest={stage3}")
        return
    rows = [_metric_row(root, stage2, arm) for arm in stage2["arms"]]
    model_records: dict[str, Any] = {}
    calibration_rows: list[dict[str, Any]] = []
    selected_arms_by_model: dict[str, list[Mapping[str, Any]]] = {}
    for model in MODELS:
        candidates = [
            (float(item["C_A"]), float(item["C_B"]))
            for item in coarse["models"][model]["top_candidates"]
        ]
        rankings = _candidate_rankings(
            [row for row in rows if row["model"] == model],
            seeds=spec["fixed_refinement"]["seeds"], candidates=candidates,
            noise_multiplier=float(spec["common"]["noise_multiplier"]),
        )
        selected = rankings[0]
        c_a, c_b = float(selected["C_A"]), float(selected["C_B"])
        selected_arms = [
            arm for arm in master["arms"]
            if arm["models"] == [model]
            and arm["initial_clip_norm_by_group"] == {"A": c_a, "B": c_b}
        ] + [
            arm for arm in stage2["arms"]
            if arm["models"] == [model]
            and arm["initial_clip_norm_by_group"] == {"A": c_a, "B": c_b}
        ]
        if len(selected_arms) != 5:
            raise RuntimeError("selected strong fixed trajectory set must contain five seeds")
        values, calibration = _calibrate_groupwise(
            root, selected_arms, thresholds={"A": c_a, "B": c_b},
            num_slots=int(spec["common"]["slaclip_num_slots"]),
            epsilon=float(spec["common"]["slaclip_endpoint_epsilon"]),
            rounds=int(spec["common"]["rounds"]),
        )
        grid_a, fallback_a = _three_point_beta_grid(values["A"])
        grid_b, fallback_b = _three_point_beta_grid(values["B"])
        calibration_rows.extend(calibration)
        selected_arms_by_model[model] = selected_arms
        model_records[model] = {
            "selected_fixed_C_by_group": {"A": c_a, "B": c_b},
            "beta_A_grid": grid_a, "beta_B_grid": grid_b,
            "beta_A_duplicate_quantile_fallback_used": fallback_a,
            "beta_B_duplicate_quantile_fallback_used": fallback_b,
            "beta_A_stationary_median": statistics.median(values["A"]),
            "beta_B_stationary_median": statistics.median(values["B"]),
            "calibration_seed_count": len(selected_arms),
            "ordered_refinement_candidates": rankings,
        }
    calibration_path = root / CALIBRATION_NAME
    full.atomic_csv(calibration_path, calibration_rows, (
        "arm_id", "model", "seed", "round", "group", "fixed_C_group",
        "exact_q_endpoint_1", "exact_r_endpoint_K",
        "z_r_over_C_plus_epsilon", "stationary_beta",
        "actual_clipped_fraction", "stationary_dynamic_target_clipped",
        "tracking_bias_actual_minus_target",
        "actual_minus_one_minus_near_threshold_proxy",
    ))
    lock = staged._lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "STRONG_GROUPWISE_FIXED_SELECTION_LOCKED",
        "campaign_name": master["campaign_name"],
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "stage2_runtime_manifest_sha256": stage2["manifest_sha256"],
        "spec_sha256": master["spec_sha256"],
        "selection_rule": spec["fixed_refinement"]["selection_rule"],
        "development_seeds": {
            FIXED_COARSE_STAGE: spec["fixed_coarse"]["seeds"],
            FIXED_REFINEMENT_STAGE: spec["fixed_refinement"]["seeds"],
        },
        "oracle_or_confirmation_data_accessed": False,
        "calibration_privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
        "calibration_provenance": CALIBRATION_PROVENANCE,
        "calibration_data_consumed_at_controller_runtime": False,
        "models": model_records,
        "source_evidence": {
            FIXED_COARSE_STAGE: staged._arm_evidence(root, master),
            FIXED_REFINEMENT_STAGE: staged._arm_evidence(root, stage2),
        },
        "calibration_csv_sha256": full.sha256_file(calibration_path),
        "created_at_utc": full.utc_now(),
    })
    full.atomic_json(path, lock)
    lock = _load_lock(path, "strong fixed lock")
    _validate_fixed_lock(lock, master, stage2, spec)
    stage3 = _ensure_stage3(root, spec_path, spec, master, stage2, lock)
    groupwise._write_trajectories(root, [(master, master["arms"]), (stage2, stage2["arms"])])
    print(f"strong_fixed_selection_lock={path}")
    print(f"stage3_runtime_manifest={stage3}")


def _paired_identity(candidate: Mapping[str, Any], reference: Mapping[str, Any], label: str) -> None:
    if candidate.get("initial_loss") != reference.get("initial_loss"):
        raise RuntimeError(f"paired initial loss differs: {label}")
    for digest in (
        "client_partition_commitment_sha256", "sample_schedule_sha256",
        "supervision_schedule_sha256", "private_key_commitment", "rng_domain",
    ):
        if candidate.get(digest) != reference.get(digest):
            raise RuntimeError(f"paired evidence differs: {label}/{digest}")


def _oracle_sort_key(row: Mapping[str, Any]) -> tuple[float, float, int, float, float, float]:
    return (
        float(row["mean_paired_final_loss_delta"]),
        float(row["mean_paired_normalized_loss_auc_delta"]),
        int(row["controller_instability_event_count"]),
        float(row["eta"]), float(row["beta_A"]), float(row["beta_B"]),
    )


def _validate_oracle_lock(
    lock: Mapping[str, Any], master: Mapping[str, Any],
    fixed_lock: Mapping[str, Any], stage3: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    if (
        lock.get("status") != "EXACT_ORACLE_DEVELOPMENT_SELECTION_LOCKED"
        or lock.get("master_runtime_manifest_sha256") != master["manifest_sha256"]
        or lock.get("fixed_selection_lock_sha256") != fixed_lock["lock_sha256"]
        or lock.get("stage3_runtime_manifest_sha256") != stage3["manifest_sha256"]
        or lock.get("selection_rule") != spec["oracle_development"]["selection_rule"]
        or lock.get("development_seeds") != spec["oracle_development"]["seeds"]
        or lock.get("confirmation_data_accessed") is not False
        or lock.get("oracle_privacy_label") != "NON_DP_PRIVATE_DIAGNOSTIC"
    ):
        raise RuntimeError("oracle selection lock identity differs")
    if not isinstance(lock.get("models"), dict) or tuple(lock["models"]) != MODELS:
        raise RuntimeError("oracle selection lock model set differs")
    etas = {float(value) for value in spec["oracle_development"]["etas"]}
    for model in MODELS:
        record = lock["models"][model]
        if not isinstance(record, dict):
            raise RuntimeError("oracle selected model record differs")
        fixed_record = fixed_lock["models"][model]
        candidates = record.get("ordered_candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 27
            or record.get("selected_initial_C_by_group")
            != fixed_record["selected_fixed_C_by_group"]
            or record.get("selected_num_slots") != 5
        ):
            raise RuntimeError("oracle selected model record differs")
        expected_grid = {
            (float(beta_a), float(beta_b), float(eta))
            for beta_a in fixed_record["beta_A_grid"]
            for beta_b in fixed_record["beta_B_grid"]
            for eta in etas
        }
        observed_grid = {
            (float(item["beta_A"]), float(item["beta_B"]), float(item["eta"]))
            for item in candidates
        }
        ordered = sorted(candidates, key=_oracle_sort_key)
        selected = ordered[0]
        if (
            observed_grid != expected_grid
            or ordered != candidates
            or float(record.get("selected_beta_A", math.nan))
            != float(selected["beta_A"])
            or float(record.get("selected_beta_B", math.nan))
            != float(selected["beta_B"])
            or float(record.get("selected_eta", math.nan))
            != float(selected["eta"])
        ):
            raise RuntimeError("oracle frozen candidate selection differs")


def confirmation_arms(
    spec: Mapping[str, Any], fixed_lock: Mapping[str, Any],
    oracle_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for model in MODELS:
        fixed = fixed_lock["models"][model]["selected_fixed_C_by_group"]
        selected = oracle_lock["models"][model]
        c_a, c_b = float(fixed["A"]), float(fixed["B"])
        betas = {
            "A": float(selected["selected_beta_A"]),
            "B": float(selected["selected_beta_B"]),
        }
        for seed in spec["confirmation"]["seeds"]:
            fixed_id = _fixed_id("confirm", model, c_a, c_b, seed)
            arms.append(_base_arm(
                spec, arm_id=fixed_id, stage=CONFIRMATION_STAGE,
                method=FIXED_METHOD, model=model, seed=seed,
                c_a=c_a, c_b=c_b, reference_arm_id=None,
            ))
            arms.append(_base_arm(
                spec,
                arm_id=(
                    f"confirm-oracle-NONDP-{model}-ca{_token(c_a)}-"
                    f"cb{_token(c_b)}-ba{_token(betas['A'])}-"
                    f"bb{_token(betas['B'])}-e{_token(selected['selected_eta'])}-s{seed}"
                ),
                stage=CONFIRMATION_STAGE, method=ORACLE_METHOD,
                model=model, seed=seed, c_a=c_a, c_b=c_b,
                reference_arm_id=fixed_id, eta=float(selected["selected_eta"]),
                betas=betas,
                calibration_lock_sha256=str(fixed_lock["lock_sha256"]),
                calibration_provenance=CALIBRATION_PROVENANCE,
            ))
    return staged._indexed(arms)


def _ensure_stage4(
    root: Path, spec_path: Path, spec: Mapping[str, Any],
    master: Mapping[str, Any], fixed_lock: Mapping[str, Any],
    stage3: Mapping[str, Any], oracle_lock: Mapping[str, Any],
) -> Path:
    _validate_oracle_lock(oracle_lock, master, fixed_lock, stage3, spec)
    staged._verify_locked_evidence(root, stage3, oracle_lock["source_evidence"])
    candidate = _runtime(
        spec, spec_path, str(master["repository_sha"]),
        Path(str(master["input_manifest_path"])), str(master["created_at_utc"]),
        confirmation_arms(spec, fixed_lock, oracle_lock), CONFIRMATION_STAGE,
        str(oracle_lock["lock_sha256"]),
    )
    if len(candidate["arms"]) != EXPECTED_COUNTS[CONFIRMATION_STAGE]:
        raise RuntimeError("confirmation arm count differs")
    path = root / STAGE4_RUNTIME_NAME
    _write_or_verify(path, candidate, "confirmation manifest")
    return path


def lock_oracle(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, master = _identity(
        root, spec_path, args.repository.resolve(), args.expected_code_sha,
        args.input_manifest.resolve(),
    )
    coarse = _load_lock(root / COARSE_LOCK_NAME, "coarse fixed lock")
    stage2 = full.load_runtime(_ensure_stage2(root, spec_path, spec, master, coarse))
    fixed_lock = _load_lock(root / FIXED_LOCK_NAME, "strong fixed lock")
    stage3 = full.load_runtime(_ensure_stage3(root, spec_path, spec, master, stage2, fixed_lock))
    path = root / ORACLE_LOCK_NAME
    if path.is_file():
        lock = _load_lock(path, "oracle selection lock")
        _validate_oracle_lock(lock, master, fixed_lock, stage3, spec)
        stage4 = _ensure_stage4(root, spec_path, spec, master, fixed_lock, stage3, lock)
        print(f"oracle_selection_reused={path}")
        print(f"stage4_runtime_manifest={stage4}")
        return
    rows = [_metric_row(root, stage3, arm) for arm in stage3["arms"]]
    references = {
        (str(row["model"]), int(row["seed"])): row
        for row in rows if row["method"] == FIXED_METHOD
    }
    models: dict[str, Any] = {}
    for model in MODELS:
        fixed_record = fixed_lock["models"][model]
        candidates: list[dict[str, Any]] = []
        for beta_a in fixed_record["beta_A_grid"]:
            for beta_b in fixed_record["beta_B_grid"]:
                for eta in spec["oracle_development"]["etas"]:
                    subset = [
                        row for row in rows
                        if row["model"] == model
                        and row["method"] == ORACLE_METHOD
                        and float(row["slaclip_beta_A"]) == float(beta_a)
                        and float(row["slaclip_beta_B"]) == float(beta_b)
                        and float(row["slaclip_eta"]) == float(eta)
                    ]
                    if {int(row["seed"]) for row in subset} != set(spec["oracle_development"]["seeds"]):
                        raise RuntimeError("oracle candidate is incomplete")
                    final_deltas: list[float] = []
                    auc_deltas: list[float] = []
                    instability = 0
                    for row in subset:
                        reference = references[(model, int(row["seed"]))]
                        _paired_identity(row, reference, f"{model}/{row['seed']}")
                        final_deltas.append(float(row["final_loss"]) - float(reference["final_loss"]))
                        auc_deltas.append(float(row["normalized_loss_auc"]) - float(reference["normalized_loss_auc"]))
                        instability += staged._controller_instability_events(row)
                    candidates.append({
                        "beta_A": float(beta_a), "beta_B": float(beta_b),
                        "eta": float(eta), "seed_count": len(subset),
                        "mean_paired_final_loss_delta": statistics.fmean(final_deltas),
                        "mean_paired_normalized_loss_auc_delta": statistics.fmean(auc_deltas),
                        "controller_instability_event_count": instability,
                        "paired_final_loss_deltas": final_deltas,
                        "paired_normalized_loss_auc_deltas": auc_deltas,
                    })
        candidates.sort(key=_oracle_sort_key)
        selected = candidates[0]
        models[model] = {
            "selected_beta_A": selected["beta_A"],
            "selected_beta_B": selected["beta_B"],
            "selected_eta": selected["eta"],
            "selected_initial_C_by_group": fixed_record["selected_fixed_C_by_group"],
            "selected_num_slots": 5,
            "ordered_candidates": candidates,
        }
    lock = staged._lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "EXACT_ORACLE_DEVELOPMENT_SELECTION_LOCKED",
        "campaign_name": master["campaign_name"],
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "fixed_selection_lock_sha256": fixed_lock["lock_sha256"],
        "stage3_runtime_manifest_sha256": stage3["manifest_sha256"],
        "selection_rule": spec["oracle_development"]["selection_rule"],
        "development_seeds": spec["oracle_development"]["seeds"],
        "confirmation_data_accessed": False,
        "oracle_privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
        "models": models,
        "source_evidence": staged._arm_evidence(root, stage3),
        "created_at_utc": full.utc_now(),
    })
    full.atomic_json(path, lock)
    lock = _load_lock(path, "oracle selection lock")
    _validate_oracle_lock(lock, master, fixed_lock, stage3, spec)
    stage4 = _ensure_stage4(root, spec_path, spec, master, fixed_lock, stage3, lock)
    groupwise._write_trajectories(root, [(master, master["arms"]), (stage2, stage2["arms"]), (stage3, stage3["arms"])])
    print(f"oracle_selection_lock={path}")
    print(f"stage4_runtime_manifest={stage4}")


def _confirmation_rows(
    root: Path, stage4: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [_metric_row(root, stage4, arm) for arm in stage4["arms"]]
    by_arm = {str(row["arm_id"]): row for row in rows}
    paired: list[dict[str, Any]] = []
    for arm in stage4["arms"]:
        if arm["method"] != ORACLE_METHOD:
            continue
        candidate = by_arm[str(arm["arm_id"])]
        reference = by_arm[str(arm["reference_arm_id"])]
        _paired_identity(candidate, reference, str(arm["arm_id"]))
        paired.append({
            "model": candidate["model"], "seed": candidate["seed"],
            "fixed_arm_id": reference["arm_id"],
            "oracle_arm_id": candidate["arm_id"],
            "C_A": candidate["initial_clip_norm_A"],
            "C_B": candidate["initial_clip_norm_B"],
            "beta_A": candidate["slaclip_beta_A"],
            "beta_B": candidate["slaclip_beta_B"],
            "eta": candidate["slaclip_eta"],
            "final_loss_fixed": reference["final_loss"],
            "final_loss_oracle": candidate["final_loss"],
            "final_loss_delta_oracle_minus_fixed": (
                float(candidate["final_loss"]) - float(reference["final_loss"])
            ),
            "normalized_loss_auc_fixed": reference["normalized_loss_auc"],
            "normalized_loss_auc_oracle": candidate["normalized_loss_auc"],
            "normalized_loss_auc_delta_oracle_minus_fixed": (
                float(candidate["normalized_loss_auc"])
                - float(reference["normalized_loss_auc"])
            ),
            "controller_instability_events": staged._controller_instability_events(candidate),
            "sample_schedule_sha256": candidate["sample_schedule_sha256"],
            "supervision_schedule_sha256": candidate["supervision_schedule_sha256"],
            "private_key_commitment": candidate["private_key_commitment"],
            "rng_domain": candidate["rng_domain"],
            "oracle_final_summary_sha256": candidate["final_summary_sha256"],
            "fixed_final_summary_sha256": reference["final_summary_sha256"],
        })
    return rows, paired


def _gate_records(
    spec: Mapping[str, Any], paired: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model in MODELS:
        subset = [row for row in paired if row["model"] == model]
        if len(subset) != 20:
            raise RuntimeError("confirmation pairing is incomplete")
        final = full.paired_inference([
            float(row["final_loss_delta_oracle_minus_fixed"]) for row in subset
        ])
        auc = full.paired_inference([
            float(row["normalized_loss_auc_delta_oracle_minus_fixed"])
            for row in subset
        ])
        instability = [int(row["controller_instability_events"]) for row in subset]
        mprd = float(
            spec["confirmation"]["minimum_practically_relevant_improvement"][model]
        )
        criteria = {
            "mean_below_negative_MPRD": float(final["mean"]) <= -mprd,
            "ci95_high_below_zero": float(final["ci95_high"]) < 0.0,
            "exact_sign_flip_p_below_bonferroni_0p025": (
                float(final["exact_sign_flip_p"]) < 0.025
            ),
            "normalized_loss_auc_mean_not_positive": float(auc["mean"]) <= 0.0,
            "max_controller_instability_events_at_most_10": max(instability) <= 10,
        }
        records.append({
            "model": model, "seed_count": len(subset),
            "minimum_practically_relevant_improvement": mprd,
            **{f"final_loss_delta_{key}": value for key, value in final.items()},
            **{f"normalized_loss_auc_delta_{key}": value for key, value in auc.items()},
            "controller_instability_events_mean": statistics.fmean(instability),
            "controller_instability_events_max": max(instability),
            "criteria": criteria,
            "oracle_ceiling_gate_passed": all(criteria.values()),
        })
    return records


def _validate_gate_lock(
    root: Path, lock: Mapping[str, Any], master: Mapping[str, Any],
    fixed_lock: Mapping[str, Any], oracle_lock: Mapping[str, Any],
    stage4: Mapping[str, Any], spec: Mapping[str, Any],
    expected_records: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if (
        lock.get("status")
        not in {"ORACLE_CEILING_GATE_PASSED", "ORACLE_CEILING_GATE_NO_GO"}
        or lock.get("master_runtime_manifest_sha256") != master["manifest_sha256"]
        or lock.get("fixed_selection_lock_sha256") != fixed_lock["lock_sha256"]
        or lock.get("oracle_selection_lock_sha256") != oracle_lock["lock_sha256"]
        or lock.get("stage4_runtime_manifest_sha256") != stage4["manifest_sha256"]
        or lock.get("confirmation_seeds") != spec["confirmation"]["seeds"]
        or lock.get("privacy_label") != "NON_DP_PRIVATE_DIAGNOSTIC"
    ):
        raise RuntimeError("oracle gate lock identity differs")
    staged._verify_locked_evidence(root, stage4, lock.get("source_evidence"))
    if expected_records is None:
        _rows, paired = _confirmation_rows(root, stage4)
        expected_records = _gate_records(spec, paired)
    expected_records = list(expected_records)
    expected_passed = [
        record["model"]
        for record in expected_records
        if record["oracle_ceiling_gate_passed"]
    ]
    expected_status = (
        "ORACLE_CEILING_GATE_PASSED"
        if expected_passed
        else "ORACLE_CEILING_GATE_NO_GO"
    )
    if (
        lock.get("models") != expected_records
        or lock.get("passed_models") != expected_passed
        or lock.get("status") != expected_status
    ):
        raise RuntimeError("oracle gate lock result differs from frozen evidence")


def lock_gate(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, master = _identity(
        root, spec_path, args.repository.resolve(), args.expected_code_sha,
        args.input_manifest.resolve(),
    )
    coarse = _load_lock(root / COARSE_LOCK_NAME, "coarse fixed lock")
    stage2 = full.load_runtime(_ensure_stage2(root, spec_path, spec, master, coarse))
    fixed_lock = _load_lock(root / FIXED_LOCK_NAME, "strong fixed lock")
    stage3 = full.load_runtime(_ensure_stage3(root, spec_path, spec, master, stage2, fixed_lock))
    oracle_lock = _load_lock(root / ORACLE_LOCK_NAME, "oracle selection lock")
    stage4 = full.load_runtime(_ensure_stage4(root, spec_path, spec, master, fixed_lock, stage3, oracle_lock))
    path = root / GATE_LOCK_NAME
    if path.is_file():
        lock = _load_lock(path, "oracle gate lock")
        _validate_gate_lock(
            root, lock, master, fixed_lock, oracle_lock, stage4, spec
        )
        print(f"oracle_gate_reused={path}")
        return
    _rows, paired = _confirmation_rows(root, stage4)
    records = _gate_records(spec, paired)
    passed_models = [record["model"] for record in records if record["oracle_ceiling_gate_passed"]]
    lock = staged._lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": (
            "ORACLE_CEILING_GATE_PASSED" if passed_models
            else "ORACLE_CEILING_GATE_NO_GO"
        ),
        "campaign_name": master["campaign_name"],
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "fixed_selection_lock_sha256": fixed_lock["lock_sha256"],
        "oracle_selection_lock_sha256": oracle_lock["lock_sha256"],
        "stage4_runtime_manifest_sha256": stage4["manifest_sha256"],
        "confirmation_seeds": spec["confirmation"]["seeds"],
        "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
        "passed_models": passed_models,
        "models": records,
        "source_evidence": staged._arm_evidence(root, stage4),
        "created_at_utc": full.utc_now(),
    })
    full.atomic_json(path, lock)
    lock = _load_lock(path, "oracle gate lock")
    _validate_gate_lock(
        root, lock, master, fixed_lock, oracle_lock, stage4, spec,
        expected_records=records,
    )
    print(f"oracle_gate_lock={path}")
    print(f"oracle_gate_status={lock['status']}")
    print(f"oracle_gate_passed_models={','.join(passed_models) if passed_models else 'none'}")


def prepare_campaign(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    repository = args.repository.resolve()
    inputs = args.input_manifest.resolve()
    spec = load_spec(spec_path)
    if full.repository_sha(repository) != args.expected_code_sha or full.repository_dirty(repository):
        raise RuntimeError("repository snapshot SHA/cleanliness gate failed")
    master_path = root / MASTER_RUNTIME_NAME
    if args.resume:
        if not root.is_dir() or not master_path.is_file():
            raise RuntimeError("resume requires an existing oracle-ceiling campaign")
        master = full.load_runtime(master_path)
        candidate = _master(spec, spec_path, args.expected_code_sha, inputs, str(master.get("created_at_utc")))
        if master != candidate:
            raise RuntimeError("resume inputs differ from immutable master manifest")
    else:
        if root.exists():
            raise RuntimeError(f"refusing to overwrite campaign root: {root}")
        root.mkdir(parents=True, mode=0o700)
        master = _master(spec, spec_path, args.expected_code_sha, inputs, full.utc_now())
        full.atomic_json(master_path, master)
    for name in ("arms", "arm-status", "arm-logs", "control", "tmp", "preflight", "selection"):
        (root / name).mkdir(mode=0o700, exist_ok=True)
    stop = root / "control" / "stop.request"
    if stop.exists():
        stop.unlink()
    full.validate_or_create_key(full.absolute_path(args.private_key), create=not args.resume)
    _write_or_verify(root / PREFLIGHT_RUNTIME_NAME, _preflight(master, spec), "preflight manifest")
    if (root / COARSE_LOCK_NAME).is_file():
        coarse = _load_lock(root / COARSE_LOCK_NAME, "coarse fixed lock")
        _ensure_stage2(root, spec_path, spec, master, coarse)
    if (root / FIXED_LOCK_NAME).is_file():
        coarse = _load_lock(root / COARSE_LOCK_NAME, "coarse fixed lock")
        stage2 = full.load_runtime(_ensure_stage2(root, spec_path, spec, master, coarse))
        fixed_lock = _load_lock(root / FIXED_LOCK_NAME, "strong fixed lock")
        _ensure_stage3(root, spec_path, spec, master, stage2, fixed_lock)
    if (root / ORACLE_LOCK_NAME).is_file():
        coarse = _load_lock(root / COARSE_LOCK_NAME, "coarse fixed lock")
        stage2 = full.load_runtime(_ensure_stage2(root, spec_path, spec, master, coarse))
        fixed_lock = _load_lock(root / FIXED_LOCK_NAME, "strong fixed lock")
        stage3 = full.load_runtime(_ensure_stage3(root, spec_path, spec, master, stage2, fixed_lock))
        oracle_lock = _load_lock(root / ORACLE_LOCK_NAME, "oracle selection lock")
        _ensure_stage4(root, spec_path, spec, master, fixed_lock, stage3, oracle_lock)
    print(f"runtime_manifest={master_path}")
    print(f"preflight_runtime_manifest={root / PREFLIGHT_RUNTIME_NAME}")


def _materialized_stages(
    root: Path, spec: Mapping[str, Any], spec_path: Path,
    master: Mapping[str, Any],
) -> list[tuple[dict[str, Any], Sequence[Mapping[str, Any]]]]:
    stages: list[tuple[dict[str, Any], Sequence[Mapping[str, Any]]]] = [(dict(master), master["arms"])]
    if not (root / COARSE_LOCK_NAME).is_file():
        return stages
    coarse = _load_lock(root / COARSE_LOCK_NAME, "coarse fixed lock")
    stage2 = full.load_runtime(_ensure_stage2(root, spec_path, spec, master, coarse))
    stages.append((stage2, stage2["arms"]))
    if not (root / FIXED_LOCK_NAME).is_file():
        return stages
    fixed_lock = _load_lock(root / FIXED_LOCK_NAME, "strong fixed lock")
    stage3 = full.load_runtime(_ensure_stage3(root, spec_path, spec, master, stage2, fixed_lock))
    stages.append((stage3, stage3["arms"]))
    if not (root / ORACLE_LOCK_NAME).is_file():
        return stages
    oracle_lock = _load_lock(root / ORACLE_LOCK_NAME, "oracle selection lock")
    stage4 = full.load_runtime(_ensure_stage4(root, spec_path, spec, master, fixed_lock, stage3, oracle_lock))
    stages.append((stage4, stage4["arms"]))
    return stages


def aggregate_campaign(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec = load_spec(spec_path)
    master = full.load_runtime(root / MASTER_RUNTIME_NAME)
    stages = _materialized_stages(root, spec, spec_path, master)
    status_counts = {"COMPLETED": 0, "FAILED": 0, "CHECKPOINTED_STOP": 0, "NOT_STARTED": 0, "OTHER": 0}
    rows: list[dict[str, Any]] = []
    for manifest, arms in stages:
        for arm in arms:
            status_path = root / "arm-status" / f"{arm['arm_id']}.json"
            status = "NOT_STARTED"
            if status_path.is_file():
                status = str(full.load_object(status_path, "arm status").get("status", "OTHER"))
            status_counts[status if status in status_counts else "OTHER"] += 1
            if status == "COMPLETED":
                rows.append(_metric_row(root, manifest, arm))
    paired: list[dict[str, Any]] = []
    gate_records: list[dict[str, Any]] = []
    if len(stages) == 4 and all(
        staged._status_completed(root, stages[-1][0], arm)
        for arm in stages[-1][1]
    ):
        _stage4_rows, paired = _confirmation_rows(root, stages[-1][0])
        gate_records = _gate_records(spec, paired)
    full.atomic_csv(root / "campaign_metrics.csv", rows, staged._columns(rows, (
        "stage", "arm_id", "method", "privacy_label", "model", "seed",
        "initial_clip_norm_A", "initial_clip_norm_B", "slaclip_eta",
        "slaclip_beta_A", "slaclip_beta_B", "final_loss",
        "normalized_loss_auc", "actual_clipped_fraction",
    )))
    full.atomic_csv(root / "confirmation_paired_metrics.csv", paired, staged._columns(paired, (
        "model", "seed", "C_A", "C_B", "beta_A", "beta_B", "eta",
        "final_loss_delta_oracle_minus_fixed",
        "normalized_loss_auc_delta_oracle_minus_fixed",
        "controller_instability_events",
    )))
    full.atomic_csv(root / "confirmation_aggregate_metrics.csv", gate_records, staged._columns(gate_records, (
        "model", "seed_count", "minimum_practically_relevant_improvement",
        "final_loss_delta_mean", "final_loss_delta_ci95_low",
        "final_loss_delta_ci95_high", "oracle_ceiling_gate_passed",
    )))
    groupwise._write_trajectories(root, stages)
    gate_lock = _load_lock(root / GATE_LOCK_NAME, "oracle gate lock") if (root / GATE_LOCK_NAME).is_file() else None
    if gate_lock is not None:
        if len(stages) != 4:
            raise RuntimeError("oracle gate lock exists before all stages materialized")
        fixed_lock = _load_lock(root / FIXED_LOCK_NAME, "strong fixed lock")
        oracle_lock = _load_lock(root / ORACLE_LOCK_NAME, "oracle selection lock")
        _validate_gate_lock(
            root, gate_lock, master, fixed_lock, oracle_lock,
            stages[-1][0], spec, expected_records=gate_records,
        )
    complete = (
        len(rows) == EXPECTED_COUNTS["total"]
        and status_counts["COMPLETED"] == EXPECTED_COUNTS["total"]
        and len(paired) == 40
        and gate_lock is not None
    )
    summary = {
        "schema_version": 1,
        "status": "COMPLETED" if complete else "IN_PROGRESS",
        "campaign_name": master["campaign_name"],
        "expected_total_model_arms": EXPECTED_COUNTS["total"],
        "completed_model_arms": len(rows),
        "status_counts_for_materialized_arms": status_counts,
        "selected_strong_groupwise_fixed": (
            _load_lock(root / FIXED_LOCK_NAME, "strong fixed lock").get("models")
            if (root / FIXED_LOCK_NAME).is_file() else None
        ),
        "selected_exact_oracle": (
            _load_lock(root / ORACLE_LOCK_NAME, "oracle selection lock").get("models")
            if (root / ORACLE_LOCK_NAME).is_file() else None
        ),
        "confirmation_paired_rows": len(paired),
        "confirmation": gate_records,
        "oracle_gate_status": gate_lock.get("status") if gate_lock else None,
        "oracle_gate_passed_models": gate_lock.get("passed_models") if gate_lock else None,
        "scientific_boundary": spec["scientific_boundary"],
        "warning": (
            "Exact endpoint calibration and oracle control consume private NON-DP diagnostics. "
            "This campaign is an oracle utility-ceiling gate, not a noisy-SlaClip efficacy result, "
            "external-test result, paper reproduction, or privacy certificate. A NO_GO applies "
            "only to the preregistered fixed-derived A/B beta grids, eta grid, and C0 equal to "
            "the selected strong groupwise-fixed comparator; it is not a universal impossibility claim."
        ),
        "updated_at_utc": full.utc_now(),
    }
    full.atomic_json(root / "campaign_summary.json", summary)
    if args.require_complete and not complete:
        raise RuntimeError(
            f"strict aggregate is incomplete: {len(rows)}/{EXPECTED_COUNTS['total']}"
        )
    print(f"campaign_summary={root / 'campaign_summary.json'}")


def validate_spec_command(args: argparse.Namespace) -> None:
    spec = load_spec(args.spec.resolve())
    print(json.dumps({
        "status": "VALID",
        "spec_sha256": full.sha256_file(args.spec.resolve()),
        "stage1_model_arm_count": len(fixed_coarse_arms(spec)),
        "stage2_model_arm_count": EXPECTED_COUNTS[FIXED_REFINEMENT_STAGE],
        "stage3_model_arm_count": EXPECTED_COUNTS[ORACLE_DEVELOPMENT_STAGE],
        "stage4_model_arm_count": EXPECTED_COUNTS[CONFIRMATION_STAGE],
        "total_model_arm_count": EXPECTED_COUNTS["total"],
        "models": list(MODELS), "K": 5,
        "excluded_method": "SlaClip-Q",
        "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
    }, indent=2, sort_keys=True))


def print_waves(args: argparse.Namespace) -> None:
    runtime = full.load_runtime(args.manifest.resolve())
    if len(runtime["arms"]) % 2:
        raise RuntimeError("sequential campaign stages must have even arm counts")
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
    for name in ("lock-coarse", "lock-fixed", "lock-oracle", "lock-gate"):
        command = commands.add_parser(name)
        _identity_args(command)
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
    elif args.command == "lock-coarse":
        lock_coarse(args)
    elif args.command == "lock-fixed":
        lock_fixed(args)
    elif args.command == "lock-oracle":
        lock_oracle(args)
    elif args.command == "lock-gate":
        lock_gate(args)
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
