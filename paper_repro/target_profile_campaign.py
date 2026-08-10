#!/usr/bin/env python3
"""Confirm five fixed-trajectory actual-clipping profiles with Full SlaClip.

This coordinator is downstream of the immutable oracle-ceiling campaign.  It
does not repeat the strong groupwise-fixed search: instead it validates and
hash-locks the upstream fixed selection and its exact private trajectory
diagnostics and derives five desired hard-clipping profiles.  Each hard-rate
label is first mapped through bracketing hard-rate strata to the stationary
endpoint surrogate ``1-q``; beta is then fit against that surrogate.  The
hard rate, surrogate dynamic target, and beta are never treated as aliases.

All model arms are intended to run sequentially in one Slurm allocation.
Only noisy Full SlaClip is tested; SlaClip-Q is outside this schema.
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
    from paper_repro import oracle_ceiling_campaign as oracle
    from paper_repro import staged_slaclip_campaign as staged
    from paper_repro.slaclip import build_slack_vector, normalize_noisy_slack
except ModuleNotFoundError:  # direct-script execution
    import full_slaclip_campaign as full  # type: ignore[no-redef]
    import groupwise_slaclip_campaign as groupwise  # type: ignore[no-redef]
    import oracle_ceiling_campaign as oracle  # type: ignore[no-redef]
    import staged_slaclip_campaign as staged  # type: ignore[no-redef]
    from slaclip import build_slack_vector, normalize_noisy_slack  # type: ignore[no-redef]


SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1
MODELS = ("bert", "gpt2")
GROUPS = ("A", "B")
FIXED_METHOD = full.FIXED_DP_METHOD
NOISY_METHOD = full.FULL_SLACLIP_METHOD
DEVELOPMENT_STAGE = "development"
CONFIRMATION_STAGE = "confirmation"
MASTER_RUNTIME_NAME = "runtime-manifest.json"
PREFLIGHT_RUNTIME_NAME = "preflight-runtime-manifest.json"
CONFIRMATION_RUNTIME_NAME = "stage2-confirmation-runtime-manifest.json"
CALIBRATION_LOCK_NAME = "target-profile-calibration.lock.json"
DEVELOPMENT_LOCK_NAME = "target-profile-development-selection.lock.json"
GATE_LOCK_NAME = "target-profile-confirmation-gate.lock.json"
CALIBRATION_CSV_NAME = "target_profile_calibration.csv"
CALIBRATION_TRAJECTORY_CSV_NAME = "target_profile_source_trajectory.csv"
CALIBRATION_PROVENANCE = (
    "exact_NON_DP_strong_groupwise_fixed_hard_rate_strata_to_stationary_"
    "endpoint_surrogate_profiles_frozen_before_target_profile_development"
)
EXPECTED_COUNTS = {DEVELOPMENT_STAGE: 156, CONFIRMATION_STAGE: 240, "total": 396}


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
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{label} is outside its domain")
    return value


def load_spec(path: Path) -> dict[str, Any]:
    spec = full.load_object(path, "target-profile campaign specification")
    _exact_keys(
        spec,
        {
            "schema_version", "campaign_name", "description",
            "expected_stage_arm_counts", "common", "upstream_oracle_ceiling",
            "target_profile_calibration", "development", "confirmation",
            "scientific_boundary",
        },
        "specification",
    )
    if spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported target-profile schema")
    if spec["expected_stage_arm_counts"] != EXPECTED_COUNTS:
        raise ValueError("target-profile arm counts differ")
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
        _integer(common[key], f"common.{key}", positive=True)
    for key in (
        "noise_multiplier", "learning_rate", "delta", "slaclip_c_min",
        "slaclip_c_max", "slaclip_endpoint_epsilon",
    ):
        _number(common[key], f"common.{key}", positive=True)
    constants = (
        common["num_clients"], common["rounds"], common["batch_size"],
        float(common["noise_multiplier"]), float(common["learning_rate"]),
        common["rank"], common["slaclip_num_slots"], common["eval_every"],
        common["checkpoint_every"],
    )
    if constants != (5, 50, 8, 2.0, 5e-4, 512, 5, 5, 25):
        raise ValueError("paper/mechanism constants differ")

    upstream = spec["upstream_oracle_ceiling"]
    _exact_keys(
        upstream,
        {
            "campaign_name", "repository_sha", "spec_sha256",
            "input_manifest_sha256", "required_fixed_lock_status",
            "selected_fixed_trajectory_seed_count_per_model",
            "selected_fixed_trajectory_source_stages",
        },
        "upstream_oracle_ceiling",
    )
    repository_sha = upstream["repository_sha"]
    if (
        not isinstance(repository_sha, str) or len(repository_sha) != 40
        or any(character not in "0123456789abcdef" for character in repository_sha)
    ):
        raise ValueError("upstream_oracle_ceiling.repository_sha is not a Git SHA")
    for key in ("spec_sha256", "input_manifest_sha256"):
        value = upstream[key]
        if (
            not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"upstream_oracle_ceiling.{key} is not a SHA-256")
    if (
        upstream["campaign_name"] != "groupwise_fixed_oracle_ceiling_v1"
        or upstream["required_fixed_lock_status"]
        != "STRONG_GROUPWISE_FIXED_SELECTION_LOCKED"
        or upstream["selected_fixed_trajectory_seed_count_per_model"] != 5
        or upstream["selected_fixed_trajectory_source_stages"]
        != [oracle.FIXED_COARSE_STAGE, oracle.FIXED_REFINEMENT_STAGE]
    ):
        raise ValueError("upstream strong-fixed provenance policy differs")

    calibration = spec["target_profile_calibration"]
    _exact_keys(
        calibration,
        {
            "groups", "range_quantiles", "profile_interpolation_positions",
            "desired_target_definition", "hard_rate_to_surrogate_mapping",
            "beta_semantics", "beta_fit", "beta_bounds",
            "private_diagnostic_label", "consumed_at_controller_runtime",
        },
        "target_profile_calibration",
    )
    if (
        calibration["groups"] != ["A", "B"]
        or calibration["range_quantiles"] != [0.1, 0.9]
        or calibration["profile_interpolation_positions"]
        != [0.0, 0.25, 0.5, 0.75, 1.0]
        or calibration["beta_bounds"] != [0.0, 1.0]
        or calibration["private_diagnostic_label"] != "NON_DP_PRIVATE_DIAGNOSTIC"
        or calibration["consumed_at_controller_runtime"] is not False
        or not str(calibration["beta_semantics"]).startswith(
            "desired_hard_clip_rate_is_an_outcome_label"
        )
        or not str(calibration["hard_rate_to_surrogate_mapping"]).startswith(
            "bracket_each_desired_hard_rate"
        )
        or not str(calibration["beta_fit"]).startswith(
            "box_constrained_weighted_least_squares"
        )
    ):
        raise ValueError("target-profile calibration policy differs")

    development = spec["development"]
    _exact_keys(
        development,
        {
            "reference_method", "method", "seeds", "etas",
            "initial_C_policy", "selection_scope",
            "hard_target_absolute_tolerance",
            "controller_tracking_absolute_error_tolerance", "selection_rule",
        },
        "development",
    )
    if (
        development["reference_method"] != FIXED_METHOD
        or development["method"] != NOISY_METHOD
        or development["seeds"] != [700, 701, 702]
        or development["etas"] != [0.0005, 0.001, 0.0025, 0.005, 0.01]
        or development["initial_C_policy"]
        != "selected_strong_groupwise_fixed_C_A_and_C_B"
        or development["selection_scope"]
        != "select_eta_separately_for_each_model_and_target_profile"
        or development["hard_target_absolute_tolerance"] != 0.15
        or development["controller_tracking_absolute_error_tolerance"] != 0.25
        or development["selection_rule"] != [
            "feasible_mean_hard_target_error_for_both_groups_at_most_0.15_and_mean_controller_tracking_error_for_both_groups_at_most_0.25",
            "lowest_mean_paired_final_validation_loss_delta_vs_strong_fixed",
            "lowest_mean_paired_normalized_loss_auc_delta_vs_strong_fixed",
            "fewest_controller_instability_events", "smaller_eta",
        ]
    ):
        raise ValueError("development policy differs")

    confirmation = spec["confirmation"]
    _exact_keys(
        confirmation,
        {
            "methods", "seeds", "primary_metric", "secondary_metrics",
            "hypothesis_count", "familywise_alpha",
            "bonferroni_alpha_per_hypothesis",
            "minimum_practically_relevant_improvement",
            "target_achievement_absolute_tolerance", "target_achievement_rule",
            "controller_tracking_absolute_error_tolerance",
            "controller_tracking_rule",
            "full_clipped_round_fraction_cap",
            "full_clipped_round_fraction_rule",
            "utility_stability_definition", "primary_gate_requires",
            "joint_claim_requires",
        },
        "confirmation",
    )
    if (
        confirmation["methods"] != [FIXED_METHOD, NOISY_METHOD]
        or confirmation["seeds"] != list(range(800, 820))
        or confirmation["primary_metric"] != "final_internal_validation_loss"
        or confirmation["hypothesis_count"] != 10
        or confirmation["familywise_alpha"] != 0.05
        or confirmation["bonferroni_alpha_per_hypothesis"] != 0.005
        or confirmation["target_achievement_absolute_tolerance"] != 0.075
        or confirmation["controller_tracking_absolute_error_tolerance"] != 0.2
        or confirmation["full_clipped_round_fraction_cap"] != 0.2
        or confirmation["secondary_metrics"] != [
            "normalized_internal_validation_loss_auc",
            "final_internal_supervised_token_accuracy",
            "cross_seed_final_loss_sample_std",
            "mean_loss_excess_total_variation",
            "mean_final_minus_best",
        ]
    ):
        raise ValueError("confirmation policy differs")
    mprd = confirmation["minimum_practically_relevant_improvement"]
    if mprd != {"bert": 0.0005, "gpt2": 0.00006}:
        raise ValueError("confirmation MPRD differs")
    if (
        confirmation["target_achievement_rule"]
        != "mean_actual_clipped_fraction_for_each_of_A_and_B_within_absolute_tolerance_of_its_desired_hard_clip_rate"
        or confirmation["controller_tracking_rule"]
        != "confirmation_mean_of_per_seed_actual_target_absolute_error_median_for_each_of_A_and_B_at_most_tolerance"
        or confirmation["full_clipped_round_fraction_rule"]
        != "confirmation_mean_fully_clipped_round_fraction_for_each_group_at_most_cap_when_desired_hard_rate_below_0.95"
        or confirmation["utility_stability_definition"] != [
            "paired_loss_excess_total_variation_delta_mean_not_positive_and_ci95_high_below_zero",
            "paired_final_minus_best_delta_mean_not_positive_and_ci95_high_below_zero",
            "cross_seed_final_loss_sample_std_is_descriptive_only",
        ]
        or confirmation["primary_gate_requires"] != [
            "paired_final_loss_mean_below_negative_model_MPRD",
            "paired_final_loss_ci95_high_below_zero",
            "exact_sign_flip_p_below_bonferroni_0.005",
            "paired_normalized_loss_auc_mean_not_positive",
            "hard_target_achieved_for_both_groups",
            "controller_surrogate_tracking_achieved_for_both_groups",
            "full_clipped_round_fraction_cap_achieved_for_both_groups",
        ]
        or confirmation["joint_claim_requires"] != [
            "primary_gate_passed", "utility_stability_gate_passed",
            "paired_final_internal_supervised_token_accuracy_delta_ci95_low_above_zero",
            "paired_final_internal_supervised_token_accuracy_exact_sign_flip_p_below_bonferroni_0.005",
        ]
    ):
        raise ValueError("confirmation gates differ")
    if set(development["seeds"]) & set(confirmation["seeds"]):
        raise ValueError("development and confirmation seeds overlap")
    boundary = spec["scientific_boundary"]
    if (
        boundary.get("excluded_method_family") != "SlaClip-Q"
        or boundary.get("full_slaclip_only") is not True
        or boundary.get(
            "desired_hard_clip_rate_is_distinct_from_beta_and_dynamic_target"
        ) is not True
        or boundary.get("utility_stability_not_threshold_stability") is not True
        or boundary.get("single_allocation") is not True
        or boundary.get("nested_sbatch_or_array") is not False
    ):
        raise ValueError("scientific boundary differs")
    return spec


def _token(value: float) -> str:
    return full.number_token(float(value))


def _base_arm(
    spec: Mapping[str, Any], *, arm_id: str, stage: str, method: str,
    model: str, seed: int, thresholds: Mapping[str, float],
    reference_arm_id: str | None, profile: Mapping[str, Any] | None = None,
    eta: float | None = None, calibration_lock_sha256: str | None = None,
) -> dict[str, Any]:
    adaptive = method == NOISY_METHOD
    if adaptive != (profile is not None and eta is not None):
        raise ValueError("adaptive arm/profile arguments disagree")
    if stage not in {DEVELOPMENT_STAGE, CONFIRMATION_STAGE}:
        raise ValueError("unsupported target-profile stage")
    c_a, c_b = float(thresholds["A"]), float(thresholds["B"])
    betas = (
        {group: float(profile["groups"][group]["beta"]) for group in GROUPS}
        if profile is not None else None
    )
    desired = (
        {
            group: float(profile["groups"][group]["desired_hard_clipped_fraction"])
            for group in GROUPS
        }
        if profile is not None else None
    )
    if adaptive and (
        not isinstance(calibration_lock_sha256, str)
        or len(calibration_lock_sha256) != 64
        or any(not 0.0 <= value <= 1.0 for value in (betas or {}).values())
    ):
        raise ValueError("adaptive arm calibration is invalid")
    common = spec["common"]
    roles = {
        DEVELOPMENT_STAGE: (
            "fresh_seed_fixed_reference_or_noisy_full_slaclip_eta_development"
        ),
        CONFIRMATION_STAGE: (
            "fresh_seed_fixed_reference_or_frozen_target_profile_confirmation"
        ),
    }
    return {
        "arm_id": arm_id,
        "stage": stage,
        "family": stage,
        "analysis_role": roles[stage],
        "method": method,
        "seed": int(seed),
        "initial_clip_norm": c_b,
        "initial_clip_norm_by_group": {"A": c_a, "B": c_b},
        "target_profile_index": int(profile["profile_index"]) if profile else None,
        "target_profile_position": (
            float(profile["interpolation_position"]) if profile else None
        ),
        "desired_hard_clipped_fraction_by_group": desired,
        "slaclip_eta": float(eta) if eta is not None else None,
        "slaclip_base_target_clipped_fraction": None,
        "slaclip_beta": None,
        "slaclip_base_target_clipped_fraction_by_group": betas,
        "slaclip_beta_by_group": dict(betas) if betas else None,
        "slaclip_baseline_calibration_lock_sha256": (
            calibration_lock_sha256 if adaptive else None
        ),
        "acknowledge_slaclip_baseline_calibration_is_non_dp": adaptive,
        "slaclip_calibration_provenance": (
            CALIBRATION_PROVENANCE if adaptive else None
        ),
        "controller_input": full.CONTROLLER_INPUT_BY_METHOD.get(method),
        "reference_arm_id": reference_arm_id,
        "rng_domain": f"target-profile-full-slaclip:s{seed}",
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


def calibrate_beta_for_hard_rate(
    source_rows: Sequence[Mapping[str, Any]], desired_hard_rate: float,
) -> dict[str, float | bool | int]:
    """Map a hard-clip label to the controller surrogate before fitting beta.

    Hard clipping is discrete for N=5 and is not equal to the exact endpoint
    surrogate ``1-q``.  We bracket ``tau`` by observed hard-rate strata, mix
    those strata so their weighted hard rate is exactly ``tau``, then solve
    the weighted least-squares stationary-controller problem
    ``beta*(1-z) ~= (1-q)``.  Thus ``tau`` remains an outcome label and never
    becomes either beta or the controller's dynamic target.
    """

    tau = float(desired_hard_rate)
    if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("desired hard clipping rate is invalid")
    rows = [dict(row) for row in source_rows]
    if not rows:
        raise ValueError("hard-rate calibration requires source rows")
    levels = sorted({float(row["actual_clipped_fraction"]) for row in rows})
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in levels):
        raise ValueError("observed hard clipping levels are invalid")
    tolerance = 1e-12
    exact = [value for value in levels if abs(value - tau) <= tolerance]
    if exact:
        lower = upper = min(exact, key=lambda value: abs(value - tau))
        lower_weight, upper_weight = 1.0, 0.0
    else:
        below = [value for value in levels if value < tau]
        above = [value for value in levels if value > tau]
        if not below or not above:
            raise RuntimeError("desired hard rate is outside observed bracketing levels")
        lower, upper = max(below), min(above)
        upper_weight = (tau - lower) / (upper - lower)
        lower_weight = 1.0 - upper_weight
    strata = [(lower, lower_weight)]
    if upper != lower and upper_weight > 0.0:
        strata.append((upper, upper_weight))
    weighted_points: list[tuple[float, float, float]] = []
    stratum_counts: dict[float, int] = {}
    for level, stratum_weight in strata:
        subset = [
            row for row in rows
            if abs(float(row["actual_clipped_fraction"]) - level) <= tolerance
        ]
        if not subset:
            raise RuntimeError("hard-rate calibration stratum is empty")
        stratum_counts[level] = len(subset)
        point_weight = stratum_weight / len(subset)
        for row in subset:
            remaining = float(row["remaining_non_small_gradient_fraction"])
            surrogate = float(row["stationary_surrogate_target_clipped"])
            if (
                not math.isfinite(remaining) or not 0.0 <= remaining <= 1.0
                or not math.isfinite(surrogate) or not 0.0 <= surrogate <= 1.0
            ):
                raise ValueError("stationary surrogate calibration values are invalid")
            weighted_points.append((remaining, surrogate, point_weight))
    total_weight = math.fsum(weight for _x, _y, weight in weighted_points)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("hard-rate stratum weights do not sum to one")
    denominator = math.fsum(weight * x * x for x, _y, weight in weighted_points)
    numerator = math.fsum(weight * x * y for x, y, weight in weighted_points)
    all_surrogate_zero = all(y == 0.0 for _x, y, _weight in weighted_points)
    unconstrained = numerator / denominator if denominator > 0.0 else 0.0
    beta = min(1.0, max(0.0, unconstrained))
    errors = [(beta * x - y, weight) for x, y, weight in weighted_points]
    weighted_hard = lower_weight * lower + upper_weight * upper
    weighted_surrogate = math.fsum(
        weight * y for _x, y, weight in weighted_points
    )
    predicted_mean = math.fsum(
        weight * beta * x for x, _y, weight in weighted_points
    )
    return {
        "desired_hard_clipped_fraction": tau,
        "bracketing_hard_rate_lower": lower,
        "bracketing_hard_rate_upper": upper,
        "bracketing_lower_weight": lower_weight,
        "bracketing_upper_weight": upper_weight,
        "bracketing_lower_row_count": stratum_counts[lower],
        "bracketing_upper_row_count": (
            stratum_counts[upper] if upper != lower else stratum_counts[lower]
        ),
        "weighted_hard_clipped_fraction": weighted_hard,
        "hard_target_reconstruction_error": weighted_hard - tau,
        "weighted_stationary_surrogate_target_mean": weighted_surrogate,
        "beta": beta,
        "unconstrained_beta": unconstrained,
        "beta_identifiable": denominator > 0.0,
        "box_constraint_feasible": (
            (denominator > 0.0 and 0.0 <= unconstrained <= 1.0)
            or (denominator == 0.0 and all_surrogate_zero)
        ),
        "beta_hit_lower_bound": beta == 0.0,
        "beta_hit_upper_bound": beta == 1.0,
        "surrogate_target_pointwise_feasible_weight": math.fsum(
            weight for x, y, weight in weighted_points if y <= x + tolerance
        ),
        "source_point_count": len(weighted_points),
        "predicted_dynamic_target_mean": predicted_mean,
        "surrogate_fit_bias": predicted_mean - weighted_surrogate,
        "surrogate_fit_mae": math.fsum(
            weight * abs(error) for error, weight in errors
        ),
        "surrogate_fit_rmse": math.sqrt(math.fsum(
            weight * error * error for error, weight in errors
        )),
        "surrogate_fit_max_abs_error": max(abs(error) for error, _weight in errors),
    }


def _load_lock(path: Path, label: str) -> dict[str, Any]:
    value = full.load_object(path, label)
    staged._validate_lock(value, label)
    return value


def _write_or_verify(path: Path, value: Mapping[str, Any], label: str) -> None:
    staged._write_or_verify(path, value, label)


def _validate_upstream(
    spec: Mapping[str, Any], *, campaign_root: Path, repository: Path,
    expected_sha: str, spec_path: Path, input_manifest: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Deeply validate the immutable upstream strong-fixed selection."""

    expected = spec["upstream_oracle_ceiling"]
    if (
        expected_sha != expected["repository_sha"]
        or full.sha256_file(spec_path) != expected["spec_sha256"]
        or full.sha256_file(input_manifest) != expected["input_manifest_sha256"]
    ):
        raise RuntimeError("upstream pinned SHA/spec/input identity differs")
    upstream_spec, master = oracle._identity(
        campaign_root.resolve(), spec_path.resolve(), repository.resolve(),
        expected_sha, input_manifest.resolve(),
    )
    if (
        upstream_spec["campaign_name"] != expected["campaign_name"]
        or master["campaign_name"] != expected["campaign_name"]
    ):
        raise RuntimeError("upstream oracle-ceiling campaign identity differs")
    coarse = _load_lock(
        campaign_root / oracle.COARSE_LOCK_NAME, "upstream coarse fixed lock"
    )
    oracle._validate_coarse_lock(coarse, master, upstream_spec)
    stage2 = full.load_runtime(
        oracle._ensure_stage2(
            campaign_root, spec_path, upstream_spec, master, coarse
        )
    )
    fixed_lock = _load_lock(
        campaign_root / oracle.FIXED_LOCK_NAME, "upstream strong fixed lock"
    )
    oracle._validate_fixed_lock(fixed_lock, master, stage2, upstream_spec)
    if fixed_lock.get("status") != expected["required_fixed_lock_status"]:
        raise RuntimeError("upstream strong fixed lock status differs")
    # _ensure_stage3 revalidates the locked coarse/refinement final summaries
    # and every round-shard prefix before allowing the downstream derivation.
    oracle._ensure_stage3(
        campaign_root, spec_path, upstream_spec, master, stage2, fixed_lock
    )
    provenance = {
        "campaign_root": str(campaign_root.resolve()),
        "campaign_name": master["campaign_name"],
        "repository_path": str(repository.resolve()),
        "repository_sha": expected_sha,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": full.sha256_file(spec_path),
        "input_manifest_path": str(input_manifest.resolve()),
        "input_manifest_sha256": full.sha256_file(input_manifest),
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "fixed_refinement_runtime_manifest_sha256": stage2["manifest_sha256"],
        "strong_fixed_selection_lock_sha256": fixed_lock["lock_sha256"],
        "strong_fixed_source_evidence_sha256": full.sha256_bytes(
            full.canonical_bytes(fixed_lock["source_evidence"])
        ),
    }
    return upstream_spec, master, stage2, fixed_lock, provenance


def _selected_fixed_arms(
    master: Mapping[str, Any], stage2: Mapping[str, Any],
    fixed_lock: Mapping[str, Any], spec: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    selected: dict[str, list[Mapping[str, Any]]] = {}
    expected_count = int(
        spec["upstream_oracle_ceiling"][
            "selected_fixed_trajectory_seed_count_per_model"
        ]
    )
    for model in MODELS:
        thresholds = fixed_lock["models"][model]["selected_fixed_C_by_group"]
        arms = [
            arm for manifest in (master, stage2) for arm in manifest["arms"]
            if arm["method"] == FIXED_METHOD
            and arm["models"] == [model]
            and arm.get("initial_clip_norm_by_group") == thresholds
        ]
        expected_seeds = {
            *[int(value) for value in (480, 481)],
            *[int(value) for value in (490, 491, 492)],
        }
        if (
            len(arms) != expected_count
            or {int(arm["seed"]) for arm in arms} != expected_seeds
        ):
            raise RuntimeError(
                f"selected upstream fixed trajectory set differs: {model}"
            )
        selected[model] = sorted(arms, key=lambda arm: int(arm["seed"]))
    return selected


def _source_trajectory(
    upstream_root: Path, arms_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    fixed_lock: Mapping[str, Any], spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slots = int(spec["common"]["slaclip_num_slots"])
    epsilon = float(spec["common"]["slaclip_endpoint_epsilon"])
    rounds = int(spec["common"]["rounds"])
    clients = int(spec["common"]["num_clients"])
    for model in MODELS:
        thresholds = fixed_lock["models"][model]["selected_fixed_C_by_group"]
        for arm in arms_by_model[model]:
            oracle._validate_fixed_groupwise_threshold_evidence(upstream_root, arm)
            for round_index in range(1, rounds + 1):
                shard = full.load_object(
                    oracle._round_path(upstream_root, arm, round_index),
                    "upstream selected fixed trajectory shard",
                )
                records = shard.get("client_records")
                summary = shard.get("round_summary")
                if (
                    shard.get("round") != round_index
                    or shard.get("model") != model
                    or shard.get("method") != FIXED_METHOD
                    or not isinstance(records, list) or len(records) != clients
                    or not isinstance(summary, dict)
                ):
                    raise RuntimeError(
                        f"upstream fixed trajectory identity differs: "
                        f"{arm['arm_id']}/{round_index}"
                    )
                for group in GROUPS:
                    threshold = float(thresholds[group])
                    norms = [
                        float(record["gradient_groups"][group]["raw_norm"])
                        for record in records
                    ]
                    vectors = [
                        build_slack_vector(norm, threshold, slots) for norm in norms
                    ]
                    exact = normalize_noisy_slack(
                        [
                            math.fsum(vector[index] for vector in vectors)
                            for index in range(slots)
                        ],
                        threshold, slots, len(records),
                    )
                    raw_z_value = float(exact[-1]) / (threshold + epsilon)
                    if not -1e-12 <= raw_z_value <= 1.0 + 1e-12:
                        raise RuntimeError(
                            "exact upstream z=r/(C+epsilon) lies outside its domain"
                        )
                    # Exact slack should already be in-domain. Normalize only a
                    # possible machine-epsilon excursion and preserve the raw
                    # value and the fact of normalization in the audit row.
                    z_value = min(1.0, max(0.0, raw_z_value))
                    actual = float(summary[group]["clipped_fraction"])
                    if not 0.0 <= actual <= 1.0:
                        raise RuntimeError("upstream actual clipped fraction is invalid")
                    surrogate_target = 1.0 - float(exact[0])
                    if not -1e-12 <= surrogate_target <= 1.0 + 1e-12:
                        raise RuntimeError("exact stationary surrogate target is invalid")
                    surrogate_target = min(1.0, max(0.0, surrogate_target))
                    rows.append({
                        "arm_id": arm["arm_id"], "model": model,
                        "seed": arm["seed"], "round": round_index,
                        "group": group, "fixed_C_group": threshold,
                        "actual_clipped_fraction": actual,
                        "exact_q_endpoint_1": float(exact[0]),
                        "exact_r_endpoint_K": float(exact[-1]),
                        "raw_z_r_over_C_plus_epsilon": raw_z_value,
                        "z_r_over_C_plus_epsilon": z_value,
                        "z_machine_epsilon_normalized": z_value != raw_z_value,
                        "remaining_non_small_gradient_fraction": 1.0 - z_value,
                        "stationary_surrogate_target_clipped": surrogate_target,
                        "hard_minus_stationary_surrogate_target": (
                            actual - surrogate_target
                        ),
                    })
    expected_rows = len(MODELS) * 5 * int(spec["common"]["rounds"]) * len(GROUPS)
    if len(rows) != expected_rows:
        raise RuntimeError("upstream calibration trajectory row count differs")
    return rows


def derive_target_profiles(
    trajectory_rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Derive five equally spaced actual-rate profiles over each P10--P90 range."""

    calibration = spec["target_profile_calibration"]
    lower_probability, upper_probability = [
        float(value) for value in calibration["range_quantiles"]
    ]
    positions = [
        float(value) for value in calibration["profile_interpolation_positions"]
    ]
    output: dict[str, list[dict[str, Any]]] = {}
    for model in MODELS:
        group_values: dict[str, dict[str, Any]] = {}
        for group in GROUPS:
            subset = [
                row for row in trajectory_rows
                if row["model"] == model and row["group"] == group
            ]
            actual = [float(row["actual_clipped_fraction"]) for row in subset]
            if len(actual) != 250:
                raise RuntimeError(f"target source rows are incomplete: {model}/{group}")
            lower = staged._linear_quantile(actual, lower_probability)
            upper = staged._linear_quantile(actual, upper_probability)
            group_values[group] = {
                "lower": float(lower), "upper": float(upper),
                "rows": subset,
            }
        profiles: list[dict[str, Any]] = []
        for index, position in enumerate(positions, start=1):
            groups: dict[str, Any] = {}
            for group in GROUPS:
                source = group_values[group]
                desired = float(
                    source["lower"]
                    + position * (source["upper"] - source["lower"])
                )
                fit = calibrate_beta_for_hard_rate(source["rows"], desired)
                groups[group] = {
                    "source_actual_clipped_fraction_P10": source["lower"],
                    "source_actual_clipped_fraction_P90": source["upper"],
                    **fit,
                }
            profiles.append({
                "profile_index": index,
                "interpolation_position": position,
                "groups": groups,
            })
        if len(profiles) != 5:
            raise RuntimeError("target-profile count differs")
        output[model] = profiles
    return output


def _calibration_lock_payload(
    spec: Mapping[str, Any], profiles: Mapping[str, Any],
    trajectory_rows: Sequence[Mapping[str, Any]], upstream: Mapping[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    return staged._lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "TARGET_PROFILE_CALIBRATION_LOCKED",
        "campaign_name": spec["campaign_name"],
        "calibration_provenance": CALIBRATION_PROVENANCE,
        "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
        "desired_hard_clip_rate_is_beta": False,
        "desired_hard_clip_rate_is_dynamic_target": False,
        "dynamic_target_formula": "beta_g*(1-z_g)",
        "stationary_surrogate_target_formula": "1-q_g",
        "target_profile_definition": spec["target_profile_calibration"][
            "desired_target_definition"
        ],
        "upstream": dict(upstream),
        "models": dict(profiles),
        "source_trajectory_rows": len(trajectory_rows),
        "source_trajectory_canonical_sha256": full.sha256_bytes(
            full.canonical_bytes(list(trajectory_rows))
        ),
        "created_at_utc": created_at_utc,
    })


def _validate_calibration_lock(
    lock: Mapping[str, Any], spec: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> None:
    staged._validate_lock(dict(lock), "target-profile calibration lock")
    if (
        lock.get("status") != "TARGET_PROFILE_CALIBRATION_LOCKED"
        or lock.get("campaign_name") != spec["campaign_name"]
        or lock.get("calibration_provenance") != CALIBRATION_PROVENANCE
        or lock.get("privacy_label") != "NON_DP_PRIVATE_DIAGNOSTIC"
        or lock.get("desired_hard_clip_rate_is_beta") is not False
        or lock.get("desired_hard_clip_rate_is_dynamic_target") is not False
        or lock.get("upstream") != upstream
        or not isinstance(lock.get("models"), dict)
        or tuple(lock["models"]) != MODELS
    ):
        raise RuntimeError("target-profile calibration lock identity differs")
    for model in MODELS:
        profiles = lock["models"][model]
        if not isinstance(profiles, list) or len(profiles) != 5:
            raise RuntimeError("calibration lock profile count differs")
        for expected_index, profile in enumerate(profiles, start=1):
            if (
                profile.get("profile_index") != expected_index
                or set(profile.get("groups", {})) != set(GROUPS)
            ):
                raise RuntimeError("calibration lock profile identity differs")
            for group in GROUPS:
                record = profile["groups"][group]
                beta = float(record.get("beta", math.nan))
                desired = float(
                    record.get("desired_hard_clipped_fraction", math.nan)
                )
                lower = float(record.get("bracketing_hard_rate_lower", math.nan))
                upper = float(record.get("bracketing_hard_rate_upper", math.nan))
                lower_weight = float(record.get("bracketing_lower_weight", math.nan))
                upper_weight = float(record.get("bracketing_upper_weight", math.nan))
                reconstructed = float(
                    record.get("weighted_hard_clipped_fraction", math.nan)
                )
                surrogate = float(
                    record.get("weighted_stationary_surrogate_target_mean", math.nan)
                )
                if (
                    not 0.0 <= beta <= 1.0
                    or not 0.0 <= lower <= desired <= upper <= 1.0
                    or not 0.0 <= lower_weight <= 1.0
                    or not 0.0 <= upper_weight <= 1.0
                    or not math.isclose(
                        lower_weight + upper_weight, 1.0,
                        rel_tol=0.0, abs_tol=1e-12,
                    )
                    or not math.isclose(
                        reconstructed, desired, rel_tol=0.0, abs_tol=1e-12
                    )
                    or not 0.0 <= surrogate <= 1.0
                ):
                    raise RuntimeError("calibration lock target/beta domain differs")


def _fixed_arm_id(stage_prefix: str, model: str, thresholds: Mapping[str, float], seed: int) -> str:
    return (
        f"{stage_prefix}-fixed-{model}-ca{_token(float(thresholds['A']))}-"
        f"cb{_token(float(thresholds['B']))}-s{seed}"
    )


def development_arms(
    spec: Mapping[str, Any], calibration_lock: Mapping[str, Any],
    upstream_fixed_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    calibration_sha = str(calibration_lock["lock_sha256"])
    for model in MODELS:
        thresholds = upstream_fixed_lock["models"][model][
            "selected_fixed_C_by_group"
        ]
        profiles = calibration_lock["models"][model]
        for seed in spec["development"]["seeds"]:
            fixed_id = _fixed_arm_id("dev", model, thresholds, int(seed))
            arms.append(_base_arm(
                spec, arm_id=fixed_id, stage=DEVELOPMENT_STAGE,
                method=FIXED_METHOD, model=model, seed=int(seed),
                thresholds=thresholds, reference_arm_id=None,
            ))
            for profile in profiles:
                profile_index = int(profile["profile_index"])
                for eta in spec["development"]["etas"]:
                    arms.append(_base_arm(
                        spec,
                        arm_id=(
                            f"dev-slaclip-{model}-p{profile_index}-"
                            f"e{_token(float(eta))}-s{seed}"
                        ),
                        stage=DEVELOPMENT_STAGE, method=NOISY_METHOD,
                        model=model, seed=int(seed), thresholds=thresholds,
                        reference_arm_id=fixed_id, profile=profile,
                        eta=float(eta),
                        calibration_lock_sha256=calibration_sha,
                    ))
    arms = staged._indexed(arms)
    if len(arms) != EXPECTED_COUNTS[DEVELOPMENT_STAGE]:
        raise RuntimeError("development arm count differs")
    return arms


def _runtime(
    spec: Mapping[str, Any], spec_path: Path, repository_sha: str,
    input_manifest: Path, created_at_utc: str, arms: list[dict[str, Any]],
    stage: str, calibration_lock: Mapping[str, Any],
    upstream: Mapping[str, Any], parent_lock_sha256: str | None,
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
        "target_profile_calibration_lock_sha256": calibration_lock["lock_sha256"],
        "upstream_strong_fixed_provenance": dict(upstream),
        "parent_selection_lock_sha256": parent_lock_sha256,
        "expected_arm_count": len(arms),
        "scientific_boundary": spec["scientific_boundary"],
        "arms": arms,
    }
    value["manifest_sha256"] = full.sha256_bytes(full.canonical_bytes(value))
    full.validate_runtime_manifest(value)
    return value


def _master(
    spec: Mapping[str, Any], spec_path: Path, repository_sha: str,
    input_manifest: Path, created_at_utc: str,
    calibration_lock: Mapping[str, Any], upstream_fixed_lock: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> dict[str, Any]:
    return _runtime(
        spec, spec_path, repository_sha, input_manifest, created_at_utc,
        development_arms(spec, calibration_lock, upstream_fixed_lock),
        DEVELOPMENT_STAGE, calibration_lock, upstream, None,
    )


def _preflight(
    master: Mapping[str, Any], spec: Mapping[str, Any],
    calibration_lock: Mapping[str, Any],
) -> dict[str, Any]:
    fixed = _base_arm(
        spec, arm_id="preflight-template-fixed", stage=DEVELOPMENT_STAGE,
        method=FIXED_METHOD, model="bert", seed=700,
        thresholds={"A": 1.0, "B": 10.0}, reference_arm_id=None,
    )
    profile = {
        "profile_index": 0, "interpolation_position": 0.5,
        "groups": {"A": {"beta": 0.5, "desired_hard_clipped_fraction": 0.5},
                   "B": {"beta": 0.5, "desired_hard_clipped_fraction": 0.5}},
    }
    adaptive = _base_arm(
        spec, arm_id="preflight-template-slaclip", stage=DEVELOPMENT_STAGE,
        method=NOISY_METHOD, model="bert", seed=700,
        thresholds={"A": 1.0, "B": 10.0}, reference_arm_id=fixed["arm_id"],
        profile=profile, eta=0.001,
        calibration_lock_sha256=str(calibration_lock["lock_sha256"]),
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


def _upstream_args(args: argparse.Namespace, spec: Mapping[str, Any]) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    return _validate_upstream(
        spec,
        campaign_root=args.upstream_campaign_root.resolve(),
        repository=args.upstream_repository.resolve(),
        expected_sha=args.upstream_expected_code_sha,
        spec_path=args.upstream_spec.resolve(),
        input_manifest=args.upstream_input_manifest.resolve(),
    )


def _identity(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    repository = args.repository.resolve()
    inputs = args.input_manifest.resolve()
    spec = load_spec(spec_path)
    if full.repository_sha(repository) != args.expected_code_sha or full.repository_dirty(repository):
        raise RuntimeError("repository snapshot SHA/cleanliness gate failed")
    _upstream_spec, upstream_master, upstream_stage2, upstream_fixed, upstream = (
        _upstream_args(args, spec)
    )
    calibration_lock = _load_lock(
        root / CALIBRATION_LOCK_NAME, "target-profile calibration lock"
    )
    _validate_calibration_lock(calibration_lock, spec, upstream)
    master = full.load_runtime(root / MASTER_RUNTIME_NAME)
    candidate = _master(
        spec, spec_path, args.expected_code_sha, inputs,
        str(master.get("created_at_utc")), calibration_lock, upstream_fixed,
        upstream,
    )
    if master != candidate:
        raise RuntimeError("target-profile master manifest differs from immutable inputs")
    return spec, master, calibration_lock, upstream_fixed, upstream


def prepare_campaign(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    repository = args.repository.resolve()
    inputs = args.input_manifest.resolve()
    spec = load_spec(spec_path)
    if full.repository_sha(repository) != args.expected_code_sha or full.repository_dirty(repository):
        raise RuntimeError("repository snapshot SHA/cleanliness gate failed")
    if full.sha256_file(inputs) != spec["upstream_oracle_ceiling"]["input_manifest_sha256"]:
        raise RuntimeError("current input manifest differs from the frozen upstream input")
    _uspec, upstream_master, upstream_stage2, upstream_fixed, upstream = (
        _upstream_args(args, spec)
    )
    selected = _selected_fixed_arms(
        upstream_master, upstream_stage2, upstream_fixed, spec
    )
    trajectory = _source_trajectory(
        args.upstream_campaign_root.resolve(), selected, upstream_fixed, spec
    )
    profiles = derive_target_profiles(trajectory, spec)
    master_path = root / MASTER_RUNTIME_NAME
    if args.resume:
        if not root.is_dir() or not master_path.is_file():
            raise RuntimeError("resume requires an existing target-profile campaign")
        existing_lock = _load_lock(
            root / CALIBRATION_LOCK_NAME, "target-profile calibration lock"
        )
        calibration_lock = _calibration_lock_payload(
            spec, profiles, trajectory, upstream,
            str(existing_lock.get("created_at_utc")),
        )
        if existing_lock != calibration_lock:
            raise RuntimeError("resume target-profile calibration differs")
        master = full.load_runtime(master_path)
        candidate = _master(
            spec, spec_path, args.expected_code_sha, inputs,
            str(master.get("created_at_utc")), calibration_lock,
            upstream_fixed, upstream,
        )
        if master != candidate:
            raise RuntimeError("resume inputs differ from immutable master manifest")
    else:
        if root.exists():
            raise RuntimeError(f"refusing to overwrite campaign root: {root}")
        root.mkdir(parents=True, mode=0o700)
        created = full.utc_now()
        calibration_lock = _calibration_lock_payload(
            spec, profiles, trajectory, upstream, created
        )
        full.atomic_json(root / CALIBRATION_LOCK_NAME, calibration_lock)
        master = _master(
            spec, spec_path, args.expected_code_sha, inputs, created,
            calibration_lock, upstream_fixed, upstream,
        )
        full.atomic_json(master_path, master)
        full.atomic_csv(
            root / CALIBRATION_TRAJECTORY_CSV_NAME, trajectory,
            (
                "arm_id", "model", "seed", "round", "group",
                "fixed_C_group", "actual_clipped_fraction",
                "exact_q_endpoint_1", "exact_r_endpoint_K",
                "raw_z_r_over_C_plus_epsilon", "z_r_over_C_plus_epsilon",
                "z_machine_epsilon_normalized",
                "remaining_non_small_gradient_fraction",
                "stationary_surrogate_target_clipped",
                "hard_minus_stationary_surrogate_target",
            ),
        )
        calibration_rows: list[dict[str, Any]] = []
        for model in MODELS:
            for profile in profiles[model]:
                for group in GROUPS:
                    calibration_rows.append({
                        "model": model,
                        "profile_index": profile["profile_index"],
                        "interpolation_position": profile["interpolation_position"],
                        "group": group,
                        **profile["groups"][group],
                    })
        full.atomic_csv(
            root / CALIBRATION_CSV_NAME, calibration_rows,
            staged._columns(calibration_rows, (
                "model", "profile_index", "interpolation_position", "group",
                "source_actual_clipped_fraction_P10",
                "source_actual_clipped_fraction_P90",
                "desired_hard_clipped_fraction",
                "bracketing_hard_rate_lower", "bracketing_hard_rate_upper",
                "bracketing_lower_weight", "bracketing_upper_weight",
                "weighted_hard_clipped_fraction",
                "hard_target_reconstruction_error",
                "weighted_stationary_surrogate_target_mean", "beta",
                "unconstrained_beta", "box_constraint_feasible",
                "surrogate_target_pointwise_feasible_weight",
                "predicted_dynamic_target_mean", "surrogate_fit_bias",
                "surrogate_fit_mae", "surrogate_fit_rmse",
                "surrogate_fit_max_abs_error",
            )),
        )
    _validate_calibration_lock(calibration_lock, spec, upstream)
    for name in (
        "arms", "arm-status", "arm-logs", "control", "tmp", "preflight",
        "selection",
    ):
        (root / name).mkdir(mode=0o700, exist_ok=True)
    stop = root / "control" / "stop.request"
    if stop.exists():
        stop.unlink()
    full.validate_or_create_key(full.absolute_path(args.private_key), create=not args.resume)
    _write_or_verify(
        root / PREFLIGHT_RUNTIME_NAME,
        _preflight(master, spec, calibration_lock), "preflight manifest",
    )
    if (root / DEVELOPMENT_LOCK_NAME).is_file():
        lock = _load_lock(root / DEVELOPMENT_LOCK_NAME, "development selection lock")
        _ensure_confirmation(
            root, spec_path, spec, master, calibration_lock,
            upstream_fixed, upstream, lock,
        )
    print(f"runtime_manifest={master_path}")
    print(f"preflight_runtime_manifest={root / PREFLIGHT_RUNTIME_NAME}")
    print(f"target_profile_calibration_lock={root / CALIBRATION_LOCK_NAME}")


def _metric_row(
    root: Path, manifest: Mapping[str, Any], arm: Mapping[str, Any],
) -> dict[str, Any]:
    row = oracle._metric_row(root, manifest, arm)
    row["target_profile_index"] = arm.get("target_profile_index")
    row["target_profile_position"] = arm.get("target_profile_position")
    desired = arm.get("desired_hard_clipped_fraction_by_group")
    row["desired_hard_clipped_fraction_A"] = (
        desired.get("A") if isinstance(desired, dict) else None
    )
    row["desired_hard_clipped_fraction_B"] = (
        desired.get("B") if isinstance(desired, dict) else None
    )
    return row


def _candidate_sort_key(
    row: Mapping[str, Any],
) -> tuple[int, float, float, int, float]:
    return (
        0 if row["target_feasible"] else 1,
        float(row["mean_paired_final_loss_delta"]),
        float(row["mean_paired_normalized_loss_auc_delta"]),
        int(row["controller_instability_event_count"]),
        float(row["eta"]),
    )


def _validate_development_lock(
    lock: Mapping[str, Any], master: Mapping[str, Any],
    calibration_lock: Mapping[str, Any], spec: Mapping[str, Any],
) -> None:
    staged._validate_lock(dict(lock), "target-profile development lock")
    if (
        lock.get("status") != "TARGET_PROFILE_DEVELOPMENT_SELECTION_LOCKED"
        or lock.get("master_runtime_manifest_sha256") != master["manifest_sha256"]
        or lock.get("calibration_lock_sha256") != calibration_lock["lock_sha256"]
        or lock.get("selection_rule") != spec["development"]["selection_rule"]
        or lock.get("development_seeds") != spec["development"]["seeds"]
        or lock.get("confirmation_data_accessed") is not False
        or not isinstance(lock.get("models"), dict)
        or tuple(lock["models"]) != MODELS
    ):
        raise RuntimeError("target-profile development lock identity differs")
    for model in MODELS:
        records = lock["models"][model]
        profiles = calibration_lock["models"][model]
        if not isinstance(records, list) or len(records) != 5:
            raise RuntimeError("development lock profile count differs")
        for expected, record in zip(profiles, records):
            ordered = record.get("ordered_eta_candidates")
            if not isinstance(ordered, list) or len(ordered) != len(
                spec["development"]["etas"]
            ):
                raise RuntimeError("development lock eta candidate count differs")
            try:
                candidate_etas = [float(candidate["eta"]) for candidate in ordered]
                configured_etas = [
                    float(value) for value in spec["development"]["etas"]
                ]
                recomputed_order = sorted(ordered, key=_candidate_sort_key)
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "development lock eta candidate evidence is invalid"
                ) from error
            if (
                record.get("profile_index") != expected["profile_index"]
                or float(record.get("selected_eta", math.nan))
                != float(ordered[0]["eta"])
                or sorted(candidate_etas) != sorted(configured_etas)
                or ordered != recomputed_order
                or record.get("desired_hard_clipped_fraction_by_group")
                != {
                    group: expected["groups"][group][
                        "desired_hard_clipped_fraction"
                    ]
                    for group in GROUPS
                }
                or record.get("beta_by_group")
                != {group: expected["groups"][group]["beta"] for group in GROUPS}
                or record.get("selected_candidate_feasible")
                != ordered[0].get("target_feasible")
                or record.get("no_feasible_candidate")
                != (not any(
                    bool(candidate.get("target_feasible"))
                    for candidate in ordered
                ))
            ):
                raise RuntimeError("development lock selected profile differs")


def confirmation_arms(
    spec: Mapping[str, Any], calibration_lock: Mapping[str, Any],
    upstream_fixed_lock: Mapping[str, Any], development_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    calibration_sha = str(calibration_lock["lock_sha256"])
    for model in MODELS:
        thresholds = upstream_fixed_lock["models"][model][
            "selected_fixed_C_by_group"
        ]
        profiles = {
            int(profile["profile_index"]): profile
            for profile in calibration_lock["models"][model]
        }
        selections = {
            int(record["profile_index"]): record
            for record in development_lock["models"][model]
        }
        for seed in spec["confirmation"]["seeds"]:
            fixed_id = _fixed_arm_id("confirm", model, thresholds, int(seed))
            arms.append(_base_arm(
                spec, arm_id=fixed_id, stage=CONFIRMATION_STAGE,
                method=FIXED_METHOD, model=model, seed=int(seed),
                thresholds=thresholds, reference_arm_id=None,
            ))
            for profile_index in range(1, 6):
                profile = profiles[profile_index]
                eta = float(selections[profile_index]["selected_eta"])
                arms.append(_base_arm(
                    spec,
                    arm_id=(
                        f"confirm-slaclip-{model}-p{profile_index}-"
                        f"e{_token(eta)}-s{seed}"
                    ),
                    stage=CONFIRMATION_STAGE, method=NOISY_METHOD,
                    model=model, seed=int(seed), thresholds=thresholds,
                    reference_arm_id=fixed_id, profile=profile, eta=eta,
                    calibration_lock_sha256=calibration_sha,
                ))
    arms = staged._indexed(arms)
    if len(arms) != EXPECTED_COUNTS[CONFIRMATION_STAGE]:
        raise RuntimeError("confirmation arm count differs")
    return arms


def _ensure_confirmation(
    root: Path, spec_path: Path, spec: Mapping[str, Any],
    master: Mapping[str, Any], calibration_lock: Mapping[str, Any],
    upstream_fixed_lock: Mapping[str, Any], upstream: Mapping[str, Any],
    development_lock: Mapping[str, Any],
) -> Path:
    _validate_development_lock(development_lock, master, calibration_lock, spec)
    staged._verify_locked_evidence(
        root, master, development_lock["source_evidence"]
    )
    candidate = _runtime(
        spec, spec_path, str(master["repository_sha"]),
        Path(str(master["input_manifest_path"])),
        str(master["created_at_utc"]),
        confirmation_arms(
            spec, calibration_lock, upstream_fixed_lock, development_lock
        ),
        CONFIRMATION_STAGE, calibration_lock, upstream,
        str(development_lock["lock_sha256"]),
    )
    path = root / CONFIRMATION_RUNTIME_NAME
    _write_or_verify(path, candidate, "target-profile confirmation manifest")
    return path


def lock_development(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, master, calibration_lock, upstream_fixed, upstream = _identity(args)
    path = root / DEVELOPMENT_LOCK_NAME
    if path.is_file():
        lock = _load_lock(path, "target-profile development lock")
        _validate_development_lock(lock, master, calibration_lock, spec)
        confirmation = _ensure_confirmation(
            root, spec_path, spec, master, calibration_lock,
            upstream_fixed, upstream, lock,
        )
        print(f"development_selection_reused={path}")
        print(f"confirmation_runtime_manifest={confirmation}")
        return
    rows = [_metric_row(root, master, arm) for arm in master["arms"]]
    references = {
        (str(row["model"]), int(row["seed"])): row
        for row in rows if row["method"] == FIXED_METHOD
    }
    models: dict[str, Any] = {}
    for model in MODELS:
        selected_profiles: list[dict[str, Any]] = []
        for profile in calibration_lock["models"][model]:
            profile_index = int(profile["profile_index"])
            candidates: list[dict[str, Any]] = []
            for eta in spec["development"]["etas"]:
                subset = [
                    row for row in rows
                    if row["model"] == model
                    and row["method"] == NOISY_METHOD
                    and int(row["target_profile_index"]) == profile_index
                    and float(row["slaclip_eta"]) == float(eta)
                ]
                if {int(row["seed"]) for row in subset} != set(
                    spec["development"]["seeds"]
                ):
                    raise RuntimeError(
                        f"development eta candidate is incomplete: "
                        f"{model}/p{profile_index}/{eta}"
                    )
                final_deltas: list[float] = []
                auc_deltas: list[float] = []
                instability = 0
                for row in subset:
                    reference = references[(model, int(row["seed"]))]
                    oracle._paired_identity(
                        row, reference,
                        f"development/{model}/p{profile_index}/{eta}/{row['seed']}",
                    )
                    final_deltas.append(
                        float(row["final_loss"]) - float(reference["final_loss"])
                    )
                    auc_deltas.append(
                        float(row["normalized_loss_auc"])
                        - float(reference["normalized_loss_auc"])
                    )
                    instability += staged._controller_instability_events(row)
                desired_by_group = {
                    group: float(profile["groups"][group][
                        "desired_hard_clipped_fraction"
                    ])
                    for group in GROUPS
                }
                mean_hard_by_group = {
                    group: statistics.fmean(
                        float(row[f"actual_clipped_fraction_{group}"])
                        for row in subset
                    )
                    for group in GROUPS
                }
                mean_controller_error_by_group = {
                    group: statistics.fmean(
                        float(row[f"actual_target_absolute_error_median_{group}"])
                        for row in subset
                    )
                    for group in GROUPS
                }
                hard_target_achieved_by_group = {
                    group: abs(
                        mean_hard_by_group[group] - desired_by_group[group]
                    ) <= float(spec["development"]["hard_target_absolute_tolerance"])
                    for group in GROUPS
                }
                controller_tracking_achieved_by_group = {
                    group: mean_controller_error_by_group[group] <= float(
                        spec["development"][
                            "controller_tracking_absolute_error_tolerance"
                        ]
                    )
                    for group in GROUPS
                }
                calibration_feasible = all(
                    bool(profile["groups"][group]["box_constraint_feasible"])
                    for group in GROUPS
                )
                target_feasible = (
                    calibration_feasible
                    and all(hard_target_achieved_by_group.values())
                    and all(controller_tracking_achieved_by_group.values())
                )
                candidates.append({
                    "eta": float(eta), "seed_count": len(subset),
                    "calibration_box_constraint_feasible_both_groups": (
                        calibration_feasible
                    ),
                    "development_mean_hard_clipped_fraction_by_group": (
                        mean_hard_by_group
                    ),
                    "development_hard_target_achieved_by_group": (
                        hard_target_achieved_by_group
                    ),
                    "development_mean_controller_tracking_error_by_group": (
                        mean_controller_error_by_group
                    ),
                    "development_controller_tracking_achieved_by_group": (
                        controller_tracking_achieved_by_group
                    ),
                    "target_feasible": target_feasible,
                    "mean_paired_final_loss_delta": statistics.fmean(final_deltas),
                    "mean_paired_normalized_loss_auc_delta": statistics.fmean(
                        auc_deltas
                    ),
                    "controller_instability_event_count": instability,
                    "paired_final_loss_deltas": final_deltas,
                    "paired_normalized_loss_auc_deltas": auc_deltas,
                })
            candidates.sort(key=_candidate_sort_key)
            selected = candidates[0]
            selected_profiles.append({
                "profile_index": profile_index,
                "interpolation_position": profile["interpolation_position"],
                "desired_hard_clipped_fraction_by_group": {
                    group: profile["groups"][group][
                        "desired_hard_clipped_fraction"
                    ]
                    for group in GROUPS
                },
                "beta_by_group": {
                    group: profile["groups"][group]["beta"] for group in GROUPS
                },
                "calibration_feasibility_by_group": {
                    group: {
                        key: profile["groups"][group][key]
                        for key in (
                            "box_constraint_feasible",
                            "bracketing_hard_rate_lower",
                            "bracketing_hard_rate_upper",
                            "bracketing_lower_weight",
                            "bracketing_upper_weight",
                            "weighted_hard_clipped_fraction",
                            "weighted_stationary_surrogate_target_mean",
                            "surrogate_target_pointwise_feasible_weight",
                            "predicted_dynamic_target_mean",
                            "surrogate_fit_mae", "surrogate_fit_rmse",
                        )
                    }
                    for group in GROUPS
                },
                "selected_eta": selected["eta"],
                "selected_candidate_feasible": selected["target_feasible"],
                "no_feasible_candidate": not any(
                    bool(candidate["target_feasible"]) for candidate in candidates
                ),
                "ordered_eta_candidates": candidates,
            })
        models[model] = selected_profiles
    lock = staged._lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "TARGET_PROFILE_DEVELOPMENT_SELECTION_LOCKED",
        "campaign_name": master["campaign_name"],
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "calibration_lock_sha256": calibration_lock["lock_sha256"],
        "selection_rule": spec["development"]["selection_rule"],
        "development_seeds": spec["development"]["seeds"],
        "confirmation_data_accessed": False,
        "models": models,
        "source_evidence": staged._arm_evidence(root, master),
        "created_at_utc": full.utc_now(),
    })
    full.atomic_json(path, lock)
    lock = _load_lock(path, "target-profile development lock")
    _validate_development_lock(lock, master, calibration_lock, spec)
    confirmation = _ensure_confirmation(
        root, spec_path, spec, master, calibration_lock,
        upstream_fixed, upstream, lock,
    )
    groupwise._write_trajectories(root, [(master, master["arms"])])
    print(f"development_selection_lock={path}")
    print(f"confirmation_runtime_manifest={confirmation}")


def _confirmation_paired_rows(
    root: Path, manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [_metric_row(root, manifest, arm) for arm in manifest["arms"]]
    by_arm = {str(row["arm_id"]): row for row in rows}
    paired: list[dict[str, Any]] = []
    for arm in manifest["arms"]:
        if arm["method"] != NOISY_METHOD:
            continue
        candidate = by_arm[str(arm["arm_id"])]
        reference = by_arm[str(arm["reference_arm_id"])]
        oracle._paired_identity(candidate, reference, str(arm["arm_id"]))
        candidate_metrics = (
            "final_loss", "normalized_loss_auc", "final_minus_best",
            "loss_total_variation", "loss_excess_total_variation",
            "final_token_accuracy", "actual_clipped_fraction_A",
            "actual_clipped_fraction_B",
            "actual_target_absolute_error_median_A",
            "actual_target_absolute_error_median_B",
            "fully_clipped_round_fraction_A",
            "fully_clipped_round_fraction_B",
        )
        reference_metrics = (
            "final_loss", "normalized_loss_auc", "final_minus_best",
            "loss_total_variation", "loss_excess_total_variation",
            "final_token_accuracy",
        )
        for metric in candidate_metrics:
            _require_confirmation_metric(
                candidate, metric, f"{arm['arm_id']}/candidate"
            )
        for metric in reference_metrics:
            _require_confirmation_metric(
                reference, metric, f"{arm['reference_arm_id']}/fixed"
            )
        for row, label in (
            (candidate, f"{arm['arm_id']}/candidate"),
            (reference, f"{arm['reference_arm_id']}/fixed"),
        ):
            _require_confirmation_metric(
                row, "final_token_accuracy", label, lower=0.0, upper=1.0
            )
            if (
                row.get("token_accuracy_definition")
                != "supervised_token_top1_micro_accuracy"
            ):
                raise RuntimeError(
                    f"confirmation token-accuracy definition differs: {label}"
                )
        for metric in (
            "actual_clipped_fraction_A", "actual_clipped_fraction_B",
            "actual_target_absolute_error_median_A",
            "actual_target_absolute_error_median_B",
            "fully_clipped_round_fraction_A",
            "fully_clipped_round_fraction_B",
        ):
            _require_confirmation_metric(
                candidate, metric, f"{arm['arm_id']}/candidate",
                lower=0.0, upper=1.0,
            )
        desired = arm["desired_hard_clipped_fraction_by_group"]
        paired.append({
            "model": candidate["model"], "seed": candidate["seed"],
            "profile_index": arm["target_profile_index"],
            "target_profile_position": arm["target_profile_position"],
            "fixed_arm_id": reference["arm_id"],
            "slaclip_arm_id": candidate["arm_id"],
            "C_A": candidate["initial_clip_norm_A"],
            "C_B": candidate["initial_clip_norm_B"],
            "beta_A": candidate["slaclip_beta_A"],
            "beta_B": candidate["slaclip_beta_B"],
            "desired_hard_clipped_fraction_A": desired["A"],
            "desired_hard_clipped_fraction_B": desired["B"],
            "eta": candidate["slaclip_eta"],
            "final_loss_fixed": reference["final_loss"],
            "final_loss_slaclip": candidate["final_loss"],
            "final_loss_delta_slaclip_minus_fixed": (
                float(candidate["final_loss"]) - float(reference["final_loss"])
            ),
            "normalized_loss_auc_fixed": reference["normalized_loss_auc"],
            "normalized_loss_auc_slaclip": candidate["normalized_loss_auc"],
            "normalized_loss_auc_delta_slaclip_minus_fixed": (
                float(candidate["normalized_loss_auc"])
                - float(reference["normalized_loss_auc"])
            ),
            "final_token_accuracy_fixed": reference["final_token_accuracy"],
            "final_token_accuracy_slaclip": candidate["final_token_accuracy"],
            "final_token_accuracy_delta_slaclip_minus_fixed": (
                float(candidate["final_token_accuracy"])
                - float(reference["final_token_accuracy"])
            ),
            "loss_total_variation_fixed": reference["loss_total_variation"],
            "loss_total_variation_slaclip": candidate["loss_total_variation"],
            "loss_excess_total_variation_fixed": reference[
                "loss_excess_total_variation"
            ],
            "loss_excess_total_variation_slaclip": candidate[
                "loss_excess_total_variation"
            ],
            "final_minus_best_fixed": reference["final_minus_best"],
            "final_minus_best_slaclip": candidate["final_minus_best"],
            "mean_actual_clipped_fraction_A_observation": candidate[
                "actual_clipped_fraction_A"
            ],
            "mean_actual_clipped_fraction_B_observation": candidate[
                "actual_clipped_fraction_B"
            ],
            "actual_target_absolute_error_median_A": candidate[
                "actual_target_absolute_error_median_A"
            ],
            "actual_target_absolute_error_median_B": candidate[
                "actual_target_absolute_error_median_B"
            ],
            "fully_clipped_round_fraction_A": candidate[
                "fully_clipped_round_fraction_A"
            ],
            "fully_clipped_round_fraction_B": candidate[
                "fully_clipped_round_fraction_B"
            ],
            "controller_instability_events": staged._controller_instability_events(
                candidate
            ),
            "sample_schedule_sha256": candidate["sample_schedule_sha256"],
            "supervision_schedule_sha256": candidate[
                "supervision_schedule_sha256"
            ],
            "private_key_commitment": candidate["private_key_commitment"],
            "rng_domain": candidate["rng_domain"],
            "slaclip_final_summary_sha256": candidate["final_summary_sha256"],
            "fixed_final_summary_sha256": reference["final_summary_sha256"],
        })
    return rows, paired


def _require_confirmation_metric(
    row: Mapping[str, Any], name: str, label: str, *,
    lower: float | None = None, upper: float | None = None,
) -> float:
    value = row.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(
            f"confirmation metric is missing/non-finite: {label}/{name}"
        )
    numeric = float(value)
    if lower is not None and numeric < lower:
        raise RuntimeError(
            f"confirmation metric is outside its domain: {label}/{name}"
        )
    if upper is not None and numeric > upper:
        raise RuntimeError(
            f"confirmation metric is outside its domain: {label}/{name}"
        )
    return numeric


def _gate_records(
    spec: Mapping[str, Any], paired: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    tolerance = float(
        spec["confirmation"]["target_achievement_absolute_tolerance"]
    )
    controller_tolerance = float(
        spec["confirmation"]["controller_tracking_absolute_error_tolerance"]
    )
    full_clipped_cap = float(
        spec["confirmation"]["full_clipped_round_fraction_cap"]
    )
    alpha = float(spec["confirmation"]["bonferroni_alpha_per_hypothesis"])
    for model in MODELS:
        for profile_index in range(1, 6):
            subset = [
                row for row in paired
                if row["model"] == model
                and int(row["profile_index"]) == profile_index
            ]
            if len(subset) != 20 or {int(row["seed"]) for row in subset} != set(
                spec["confirmation"]["seeds"]
            ):
                raise RuntimeError(
                    f"confirmation profile pairing is incomplete: "
                    f"{model}/p{profile_index}"
                )
            final = full.paired_inference([
                float(row["final_loss_delta_slaclip_minus_fixed"])
                for row in subset
            ])
            auc = full.paired_inference([
                float(row["normalized_loss_auc_delta_slaclip_minus_fixed"])
                for row in subset
            ])
            accuracy = full.paired_inference([
                float(row["final_token_accuracy_delta_slaclip_minus_fixed"])
                for row in subset
            ])
            excess_variation = full.paired_inference([
                float(row["loss_excess_total_variation_slaclip"])
                - float(row["loss_excess_total_variation_fixed"])
                for row in subset
            ])
            final_minus_best = full.paired_inference([
                float(row["final_minus_best_slaclip"])
                - float(row["final_minus_best_fixed"])
                for row in subset
            ])
            for label, inference in (
                ("final_loss", final), ("normalized_loss_auc", auc),
                ("final_token_accuracy", accuracy),
                ("loss_excess_total_variation", excess_variation),
                ("final_minus_best", final_minus_best),
            ):
                if int(inference.get("n", -1)) != len(subset):
                    raise RuntimeError(
                        f"confirmation inference dropped paired seeds: "
                        f"{model}/p{profile_index}/{label}"
                    )
                for field in ("mean", "ci95_low", "ci95_high", "exact_sign_flip_p"):
                    value = inference.get(field)
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                    ):
                        raise RuntimeError(
                            f"confirmation inference is incomplete: "
                            f"{model}/p{profile_index}/{label}/{field}"
                        )
            desired = {
                group: float(subset[0][f"desired_hard_clipped_fraction_{group}"])
                for group in GROUPS
            }
            actual = {
                group: statistics.fmean(
                    float(row[f"mean_actual_clipped_fraction_{group}_observation"])
                    for row in subset
                )
                for group in GROUPS
            }
            target_error = {
                group: actual[group] - desired[group] for group in GROUPS
            }
            target_achieved = {
                group: abs(target_error[group]) <= tolerance for group in GROUPS
            }
            controller_tracking_error = {
                group: statistics.fmean(
                    float(row[f"actual_target_absolute_error_median_{group}"])
                    for row in subset
                )
                for group in GROUPS
            }
            controller_tracking_achieved = {
                group: controller_tracking_error[group] <= controller_tolerance
                for group in GROUPS
            }
            mean_fully_clipped_round_fraction = {
                group: statistics.fmean(
                    float(row[f"fully_clipped_round_fraction_{group}"])
                    for row in subset
                )
                for group in GROUPS
            }
            full_clipped_cap_achieved = {
                group: (
                    desired[group] >= 0.95
                    or mean_fully_clipped_round_fraction[group] <= full_clipped_cap
                )
                for group in GROUPS
            }
            fixed_final = [float(row["final_loss_fixed"]) for row in subset]
            candidate_final = [float(row["final_loss_slaclip"]) for row in subset]
            fixed_final_sd = statistics.stdev(fixed_final)
            candidate_final_sd = statistics.stdev(candidate_final)
            fixed_excess_tv = statistics.fmean(
                float(row["loss_excess_total_variation_fixed"]) for row in subset
            )
            candidate_excess_tv = statistics.fmean(
                float(row["loss_excess_total_variation_slaclip"]) for row in subset
            )
            fixed_final_minus_best = statistics.fmean(
                float(row["final_minus_best_fixed"]) for row in subset
            )
            candidate_final_minus_best = statistics.fmean(
                float(row["final_minus_best_slaclip"]) for row in subset
            )
            stability = {
                "paired_loss_excess_total_variation_delta_mean_not_positive": (
                    float(excess_variation["mean"]) <= 0.0
                ),
                "paired_loss_excess_total_variation_delta_ci95_high_below_zero": (
                    float(excess_variation["ci95_high"]) < 0.0
                ),
                "paired_final_minus_best_delta_mean_not_positive": (
                    float(final_minus_best["mean"]) <= 0.0
                ),
                "paired_final_minus_best_delta_ci95_high_below_zero": (
                    float(final_minus_best["ci95_high"]) < 0.0
                ),
            }
            mprd = float(
                spec["confirmation"]["minimum_practically_relevant_improvement"][
                    model
                ]
            )
            primary = {
                "paired_final_loss_mean_below_negative_model_MPRD": (
                    float(final["mean"]) <= -mprd
                ),
                "paired_final_loss_ci95_high_below_zero": (
                    float(final["ci95_high"]) < 0.0
                ),
                "exact_sign_flip_p_below_bonferroni_0.005": (
                    float(final["exact_sign_flip_p"]) < alpha
                ),
                "paired_normalized_loss_auc_mean_not_positive": (
                    float(auc["mean"]) <= 0.0
                ),
                "hard_target_achieved_for_both_groups": all(
                    target_achieved.values()
                ),
                "controller_surrogate_tracking_achieved_for_both_groups": all(
                    controller_tracking_achieved.values()
                ),
                "full_clipped_round_fraction_cap_achieved_for_both_groups": all(
                    full_clipped_cap_achieved.values()
                ),
            }
            accuracy_gate = {
                "paired_final_internal_supervised_token_accuracy_delta_ci95_low_above_zero": (
                    float(accuracy["ci95_low"]) > 0.0
                ),
                "paired_final_internal_supervised_token_accuracy_exact_sign_flip_p_below_bonferroni_0.005": (
                    float(accuracy["exact_sign_flip_p"]) < alpha
                ),
            }
            primary_passed = all(primary.values())
            stability_passed = all(stability.values())
            accuracy_passed = all(accuracy_gate.values())
            records.append({
                "model": model, "profile_index": profile_index,
                "target_profile_position": subset[0]["target_profile_position"],
                "seed_count": len(subset), "selected_eta": subset[0]["eta"],
                "beta_A": subset[0]["beta_A"],
                "beta_B": subset[0]["beta_B"],
                "desired_hard_clipped_fraction_A": desired["A"],
                "desired_hard_clipped_fraction_B": desired["B"],
                "confirmation_mean_actual_clipped_fraction_A": actual["A"],
                "confirmation_mean_actual_clipped_fraction_B": actual["B"],
                "target_error_A": target_error["A"],
                "target_error_B": target_error["B"],
                "target_achieved_A": target_achieved["A"],
                "target_achieved_B": target_achieved["B"],
                "target_achieved_both_groups": all(target_achieved.values()),
                "target_achievement_absolute_tolerance": tolerance,
                "controller_tracking_error_A": controller_tracking_error["A"],
                "controller_tracking_error_B": controller_tracking_error["B"],
                "controller_tracking_achieved_A": (
                    controller_tracking_achieved["A"]
                ),
                "controller_tracking_achieved_B": (
                    controller_tracking_achieved["B"]
                ),
                "controller_tracking_achieved_both_groups": all(
                    controller_tracking_achieved.values()
                ),
                "controller_tracking_absolute_error_tolerance": (
                    controller_tolerance
                ),
                "mean_fully_clipped_round_fraction_A": (
                    mean_fully_clipped_round_fraction["A"]
                ),
                "mean_fully_clipped_round_fraction_B": (
                    mean_fully_clipped_round_fraction["B"]
                ),
                "full_clipped_round_fraction_cap_achieved_A": (
                    full_clipped_cap_achieved["A"]
                ),
                "full_clipped_round_fraction_cap_achieved_B": (
                    full_clipped_cap_achieved["B"]
                ),
                "full_clipped_round_fraction_cap_achieved_both_groups": all(
                    full_clipped_cap_achieved.values()
                ),
                "full_clipped_round_fraction_cap": full_clipped_cap,
                "minimum_practically_relevant_improvement": mprd,
                "bonferroni_alpha_per_hypothesis": alpha,
                **{f"final_loss_delta_{key}": value for key, value in final.items()},
                **{f"normalized_loss_auc_delta_{key}": value for key, value in auc.items()},
                **{
                    f"final_token_accuracy_delta_{key}": value
                    for key, value in accuracy.items()
                },
                **{
                    f"loss_excess_total_variation_delta_{key}": value
                    for key, value in excess_variation.items()
                },
                **{
                    f"final_minus_best_delta_{key}": value
                    for key, value in final_minus_best.items()
                },
                "fixed_cross_seed_final_loss_sample_std": fixed_final_sd,
                "slaclip_cross_seed_final_loss_sample_std": candidate_final_sd,
                "fixed_mean_loss_excess_total_variation": fixed_excess_tv,
                "slaclip_mean_loss_excess_total_variation": candidate_excess_tv,
                "fixed_mean_final_minus_best": fixed_final_minus_best,
                "slaclip_mean_final_minus_best": candidate_final_minus_best,
                "primary_criteria": primary,
                "utility_stability_criteria": stability,
                "token_accuracy_criteria": accuracy_gate,
                "primary_gate_passed": primary_passed,
                "utility_stability_gate_passed": stability_passed,
                "token_accuracy_secondary_gate_passed": accuracy_passed,
                "joint_claim_supported": (
                    primary_passed and stability_passed and accuracy_passed
                ),
            })
    if len(records) != 10:
        raise RuntimeError("confirmation hypothesis count differs")
    return records


def _validate_gate_lock(
    root: Path, lock: Mapping[str, Any], master: Mapping[str, Any],
    calibration_lock: Mapping[str, Any], development_lock: Mapping[str, Any],
    confirmation: Mapping[str, Any], spec: Mapping[str, Any],
    expected_records: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    staged._validate_lock(dict(lock), "target-profile gate lock")
    if (
        lock.get("status")
        not in {"TARGET_PROFILE_CLAIM_SUPPORTED", "TARGET_PROFILE_CLAIM_NOT_SUPPORTED"}
        or lock.get("master_runtime_manifest_sha256") != master["manifest_sha256"]
        or lock.get("calibration_lock_sha256") != calibration_lock["lock_sha256"]
        or lock.get("development_selection_lock_sha256")
        != development_lock["lock_sha256"]
        or lock.get("confirmation_runtime_manifest_sha256")
        != confirmation["manifest_sha256"]
        or lock.get("confirmation_seeds") != spec["confirmation"]["seeds"]
        or lock.get("hypothesis_count") != 10
        or lock.get("bonferroni_alpha_per_hypothesis") != 0.005
    ):
        raise RuntimeError("target-profile gate lock identity differs")
    staged._verify_locked_evidence(root, confirmation, lock.get("source_evidence"))
    if expected_records is None:
        _rows, paired = _confirmation_paired_rows(root, confirmation)
        expected_records = _gate_records(spec, paired)
    expected_records = list(expected_records)
    supported = [
        {"model": row["model"], "profile_index": row["profile_index"]}
        for row in expected_records if row["joint_claim_supported"]
    ]
    expected_status = (
        "TARGET_PROFILE_CLAIM_SUPPORTED"
        if supported else "TARGET_PROFILE_CLAIM_NOT_SUPPORTED"
    )
    if (
        lock.get("hypotheses") != expected_records
        or lock.get("supported_model_profiles") != supported
        or lock.get("status") != expected_status
    ):
        raise RuntimeError("target-profile gate result differs from evidence")


def lock_gate(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, master, calibration_lock, upstream_fixed, upstream = _identity(args)
    development_lock = _load_lock(
        root / DEVELOPMENT_LOCK_NAME, "target-profile development lock"
    )
    confirmation = full.load_runtime(_ensure_confirmation(
        root, spec_path, spec, master, calibration_lock,
        upstream_fixed, upstream, development_lock,
    ))
    path = root / GATE_LOCK_NAME
    if path.is_file():
        lock = _load_lock(path, "target-profile gate lock")
        _validate_gate_lock(
            root, lock, master, calibration_lock, development_lock,
            confirmation, spec,
        )
        print(f"target_profile_gate_reused={path}")
        return
    _rows, paired = _confirmation_paired_rows(root, confirmation)
    records = _gate_records(spec, paired)
    supported = [
        {"model": row["model"], "profile_index": row["profile_index"]}
        for row in records if row["joint_claim_supported"]
    ]
    lock = staged._lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": (
            "TARGET_PROFILE_CLAIM_SUPPORTED"
            if supported else "TARGET_PROFILE_CLAIM_NOT_SUPPORTED"
        ),
        "campaign_name": master["campaign_name"],
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "calibration_lock_sha256": calibration_lock["lock_sha256"],
        "development_selection_lock_sha256": development_lock["lock_sha256"],
        "confirmation_runtime_manifest_sha256": confirmation["manifest_sha256"],
        "confirmation_seeds": spec["confirmation"]["seeds"],
        "hypothesis_count": spec["confirmation"]["hypothesis_count"],
        "familywise_alpha": spec["confirmation"]["familywise_alpha"],
        "bonferroni_alpha_per_hypothesis": spec["confirmation"][
            "bonferroni_alpha_per_hypothesis"
        ],
        "accuracy_metric_scope": "internal_supervised_token_accuracy_not_external_task_accuracy",
        "stability_scope": "utility_outcome_stability_not_threshold_stability",
        "supported_model_profiles": supported,
        "hypotheses": records,
        "source_evidence": staged._arm_evidence(root, confirmation),
        "created_at_utc": full.utc_now(),
    })
    full.atomic_json(path, lock)
    lock = _load_lock(path, "target-profile gate lock")
    _validate_gate_lock(
        root, lock, master, calibration_lock, development_lock,
        confirmation, spec, expected_records=records,
    )
    print(f"target_profile_gate_lock={path}")
    print(f"target_profile_gate_status={lock['status']}")


def _materialized_stages(
    root: Path, master: Mapping[str, Any],
) -> list[tuple[dict[str, Any], Sequence[Mapping[str, Any]]]]:
    stages: list[tuple[dict[str, Any], Sequence[Mapping[str, Any]]]] = [
        (dict(master), master["arms"])
    ]
    confirmation_path = root / CONFIRMATION_RUNTIME_NAME
    if confirmation_path.is_file():
        confirmation = full.load_runtime(confirmation_path)
        if (
            confirmation.get("stage") != CONFIRMATION_STAGE
            or confirmation.get("expected_arm_count")
            != EXPECTED_COUNTS[CONFIRMATION_STAGE]
            or confirmation.get("repository_sha") != master["repository_sha"]
            or confirmation.get("spec_sha256") != master["spec_sha256"]
            or confirmation.get("input_manifest_sha256")
            != master["input_manifest_sha256"]
            or confirmation.get("target_profile_calibration_lock_sha256")
            != master["target_profile_calibration_lock_sha256"]
        ):
            raise RuntimeError("materialized confirmation manifest identity differs")
        stages.append((confirmation, confirmation["arms"]))
    return stages


def aggregate_campaign(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec = load_spec(args.spec.resolve())
    master = full.load_runtime(root / MASTER_RUNTIME_NAME)
    calibration_lock = _load_lock(
        root / CALIBRATION_LOCK_NAME, "target-profile calibration lock"
    )
    if (
        master.get("campaign_name") != spec["campaign_name"]
        or master.get("spec_sha256") != full.sha256_file(args.spec.resolve())
        or master.get("target_profile_calibration_lock_sha256")
        != calibration_lock["lock_sha256"]
    ):
        raise RuntimeError("aggregate target-profile identity differs")
    stages = _materialized_stages(root, master)
    status_counts = {
        "COMPLETED": 0, "FAILED": 0, "CHECKPOINTED_STOP": 0,
        "NOT_STARTED": 0, "OTHER": 0,
    }
    rows: list[dict[str, Any]] = []
    for manifest, arms in stages:
        for arm in arms:
            status_path = root / "arm-status" / f"{arm['arm_id']}.json"
            status = "NOT_STARTED"
            if status_path.is_file():
                status = str(
                    full.load_object(status_path, "arm status").get(
                        "status", "OTHER"
                    )
                )
            status_counts[status if status in status_counts else "OTHER"] += 1
            if status == "COMPLETED":
                rows.append(_metric_row(root, manifest, arm))
    paired: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    if len(stages) == 2 and all(
        staged._status_completed(root, stages[-1][0], arm)
        for arm in stages[-1][1]
    ):
        _confirmation_rows, paired = _confirmation_paired_rows(
            root, stages[-1][0]
        )
        records = _gate_records(spec, paired)
    full.atomic_csv(
        root / "campaign_metrics.csv", rows,
        staged._columns(rows, (
            "stage", "arm_id", "method", "model", "seed",
            "target_profile_index", "target_profile_position",
            "initial_clip_norm_A", "initial_clip_norm_B", "slaclip_eta",
            "desired_hard_clipped_fraction_A",
            "desired_hard_clipped_fraction_B", "slaclip_beta_A",
            "slaclip_beta_B", "final_loss", "normalized_loss_auc",
            "final_token_accuracy", "loss_excess_total_variation",
            "final_minus_best", "actual_clipped_fraction_A",
            "actual_clipped_fraction_B",
        )),
    )
    full.atomic_csv(
        root / "confirmation_paired_metrics.csv", paired,
        staged._columns(paired, (
            "model", "profile_index", "target_profile_position", "seed",
            "desired_hard_clipped_fraction_A",
            "desired_hard_clipped_fraction_B", "beta_A", "beta_B", "eta",
            "final_loss_delta_slaclip_minus_fixed",
            "normalized_loss_auc_delta_slaclip_minus_fixed",
            "final_token_accuracy_delta_slaclip_minus_fixed",
            "loss_excess_total_variation_fixed",
            "loss_excess_total_variation_slaclip",
            "final_minus_best_fixed", "final_minus_best_slaclip",
            "mean_actual_clipped_fraction_A_observation",
            "mean_actual_clipped_fraction_B_observation",
            "actual_target_absolute_error_median_A",
            "actual_target_absolute_error_median_B",
            "fully_clipped_round_fraction_A",
            "fully_clipped_round_fraction_B",
            "controller_instability_events",
        )),
    )
    full.atomic_csv(
        root / "confirmation_hypothesis_metrics.csv", records,
        staged._columns(records, (
            "model", "profile_index", "target_profile_position", "seed_count",
            "desired_hard_clipped_fraction_A",
            "desired_hard_clipped_fraction_B",
            "confirmation_mean_actual_clipped_fraction_A",
            "confirmation_mean_actual_clipped_fraction_B",
            "target_error_A", "target_error_B", "target_achieved_A",
            "target_achieved_B", "target_achieved_both_groups",
            "controller_tracking_error_A", "controller_tracking_error_B",
            "controller_tracking_achieved_A", "controller_tracking_achieved_B",
            "controller_tracking_achieved_both_groups",
            "mean_fully_clipped_round_fraction_A",
            "mean_fully_clipped_round_fraction_B",
            "full_clipped_round_fraction_cap_achieved_A",
            "full_clipped_round_fraction_cap_achieved_B",
            "full_clipped_round_fraction_cap_achieved_both_groups",
            "final_loss_delta_mean", "final_loss_delta_ci95_low",
            "final_loss_delta_ci95_high", "final_loss_delta_exact_sign_flip_p",
            "normalized_loss_auc_delta_mean",
            "final_token_accuracy_delta_mean",
            "final_token_accuracy_delta_ci95_low",
            "final_token_accuracy_delta_exact_sign_flip_p",
            "fixed_cross_seed_final_loss_sample_std",
            "slaclip_cross_seed_final_loss_sample_std",
            "fixed_mean_loss_excess_total_variation",
            "slaclip_mean_loss_excess_total_variation",
            "loss_excess_total_variation_delta_mean",
            "loss_excess_total_variation_delta_ci95_high",
            "fixed_mean_final_minus_best", "slaclip_mean_final_minus_best",
            "final_minus_best_delta_mean", "final_minus_best_delta_ci95_high",
            "primary_gate_passed", "utility_stability_gate_passed",
            "token_accuracy_secondary_gate_passed", "joint_claim_supported",
        )),
    )
    groupwise._write_trajectories(root, stages)
    development_lock = (
        _load_lock(root / DEVELOPMENT_LOCK_NAME, "target-profile development lock")
        if (root / DEVELOPMENT_LOCK_NAME).is_file() else None
    )
    gate_lock = (
        _load_lock(root / GATE_LOCK_NAME, "target-profile gate lock")
        if (root / GATE_LOCK_NAME).is_file() else None
    )
    if gate_lock is not None:
        if len(stages) != 2 or development_lock is None:
            raise RuntimeError("target-profile gate exists before stage materialization")
        _validate_gate_lock(
            root, gate_lock, master, calibration_lock, development_lock,
            stages[-1][0], spec, expected_records=records,
        )
    complete = (
        len(rows) == EXPECTED_COUNTS["total"]
        and status_counts["COMPLETED"] == EXPECTED_COUNTS["total"]
        and len(paired) == 200
        and len(records) == 10
        and gate_lock is not None
    )
    summary = {
        "schema_version": 1,
        "status": "COMPLETED" if complete else "IN_PROGRESS",
        "campaign_name": master["campaign_name"],
        "expected_total_model_arms": EXPECTED_COUNTS["total"],
        "completed_model_arms": len(rows),
        "status_counts_for_materialized_arms": status_counts,
        "upstream_strong_fixed_provenance": master[
            "upstream_strong_fixed_provenance"
        ],
        "target_profile_calibration": calibration_lock["models"],
        "selected_eta_by_model_and_profile": (
            development_lock.get("models") if development_lock else None
        ),
        "confirmation_paired_rows": len(paired),
        "confirmation_hypotheses": records,
        "gate_status": gate_lock.get("status") if gate_lock else None,
        "supported_model_profiles": (
            gate_lock.get("supported_model_profiles") if gate_lock else []
        ),
        "scientific_boundary": spec["scientific_boundary"],
        "warning": (
            "Desired hard clipping rates are fixed-trajectory P10--P90 profile "
            "labels, not beta or dynamic controller targets. Bracketing hard-rate "
            "strata map each label to the stationary endpoint surrogate 1-q, and "
            "beta is fit to beta*(1-z) ~= 1-q. HARD_TARGET_ACHIEVED requires "
            "confirmation mean hard clipping within +/-0.075 for both groups; "
            "controller surrogate tracking additionally requires mean per-seed "
            "actual-target absolute-error medians <=0.20 for both groups. "
            "When a desired hard rate is below 0.95, mean fully-clipped-round "
            "fraction must also remain <=0.20 for both groups. "
            "Token accuracy is internal supervised-token accuracy, not external "
            "task accuracy. Stability is utility/outcome stability, never "
            "threshold stability versus a constant fixed-C comparator."
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
        "development_model_arm_count": EXPECTED_COUNTS[DEVELOPMENT_STAGE],
        "confirmation_model_arm_count": EXPECTED_COUNTS[CONFIRMATION_STAGE],
        "total_model_arm_count": EXPECTED_COUNTS["total"],
        "models": list(MODELS), "K": 5, "N": 5,
        "target_profiles_per_model": 5,
        "hypotheses": 10, "bonferroni_alpha": 0.005,
        "excluded_method": "SlaClip-Q",
    }, indent=2, sort_keys=True))


def print_waves(args: argparse.Namespace) -> None:
    runtime = full.load_runtime(args.manifest.resolve())
    if len(runtime["arms"]) % 2:
        raise RuntimeError("sequential target-profile stage has an odd arm count")
    for index in range(0, len(runtime["arms"]), 2):
        print(f"{index}\t{index + 1}")


def _identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--upstream-campaign-root", type=Path, required=True)
    parser.add_argument("--upstream-repository", type=Path, required=True)
    parser.add_argument("--upstream-expected-code-sha", required=True)
    parser.add_argument("--upstream-spec", type=Path, required=True)
    parser.add_argument("--upstream-input-manifest", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-spec")
    validate.add_argument("--spec", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    _identity_args(prepare)
    prepare.add_argument("--private-key", type=Path, required=True)
    prepare.add_argument("--resume", action="store_true")
    development = commands.add_parser("lock-development")
    _identity_args(development)
    gate = commands.add_parser("lock-gate")
    _identity_args(gate)
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
    elif args.command == "lock-development":
        lock_development(args)
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
