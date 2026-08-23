#!/usr/bin/env python3
"""Identifiable N=128 Full SlaClip extension downstream of job 1425084.

The coordinator is deliberately staged and fail closed.  It first rebuilds
and hash-checks every fixed-baseline metric and round shard from the completed
180-arm parent campaign.  It then chooses exactly one eligible GPT-2 domain,
screens all ten parent fixed threshold pairs at N=128, freezes both the
strongest screened fixed C* and an independently eligible calibration C0,
calibrates five per-group hard-rate profiles, and finally confirms one frozen
Full SlaClip controller on independent seeds.  Hard clipping rate, beta, and
the per-round dynamic target beta*(1-z) remain distinct throughout.

Only ``paper_dp_lora`` and the canonical noisy ``slaclip_dp_lora`` method are
executable.  No quantile-controller variant is planned by this module.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from paper_repro import baseline_followup_campaign as baseline
    from paper_repro import full_slaclip_campaign as full
    from paper_repro import oracle_ceiling_campaign as oracle
    from paper_repro import staged_slaclip_campaign as staged
    from paper_repro import target_profile_campaign as target_profile
    from paper_repro.slaclip import automatic_num_slots, build_slack_vector, normalize_noisy_slack
except ModuleNotFoundError:  # direct-script execution
    import baseline_followup_campaign as baseline  # type: ignore[no-redef]
    import full_slaclip_campaign as full  # type: ignore[no-redef]
    import oracle_ceiling_campaign as oracle  # type: ignore[no-redef]
    import staged_slaclip_campaign as staged  # type: ignore[no-redef]
    import target_profile_campaign as target_profile  # type: ignore[no-redef]
    from slaclip import automatic_num_slots, build_slack_vector, normalize_noisy_slack  # type: ignore[no-redef]


SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1
GROUPS = ("A", "B")
MODEL = "gpt2"
DOMAINS = ("meddialog", "slimpajama", "finance")
FIXED_METHOD = full.FIXED_DP_METHOD
SLACLIP_METHOD = full.FULL_SLACLIP_METHOD
FIXED_STAGE = "fixed_screen"
DEVELOPMENT_STAGE = "profile_development"
CONFIRMATION_STAGE = "confirmation"
EXPECTED_COUNTS = {FIXED_STAGE: 20, DEVELOPMENT_STAGE: 10, CONFIRMATION_STAGE: 12, "total": 42}

PLAN_NAME = "identified-campaign-plan.lock.json"
UPSTREAM_LOCK_NAME = "upstream-gpt2-eligibility-selection.lock.json"
FIXED_MANIFEST_NAME = "stage1-fixed-screen-runtime-manifest.json"
FIXED_LOCK_NAME = "stage1-n128-fixed-selection.lock.json"
DEVELOPMENT_MANIFEST_NAME = "stage2-profile-development-runtime-manifest.json"
DEVELOPMENT_LOCK_NAME = "stage2-profile-selection.lock.json"
CONFIRMATION_MANIFEST_NAME = "stage3-confirmation-runtime-manifest.json"
GATE_LOCK_NAME = "confirmation-gate.lock.json"
SUMMARY_NAME = "campaign_summary.json"

CALIBRATION_PROVENANCE = (
    "exact_per_group_NON_DP_fixed_trajectory_hard_rate_strata_to_stationary_"
    "endpoint_surrogate_frozen_before_full_slaclip"
)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_spec(path: Path) -> dict[str, Any]:
    spec = full.load_object(path.resolve(), "identified N128 campaign specification")
    _exact_keys(
        spec,
        {
            "schema_version", "campaign_name", "description",
            "expected_stage_arm_counts", "upstream", "common", "eligibility",
            "fixed_screen", "profile_development", "confirmation",
            "scientific_boundary",
        },
        "identified N128 specification",
    )
    if spec["schema_version"] != SCHEMA_VERSION or spec["expected_stage_arm_counts"] != EXPECTED_COUNTS:
        raise ValueError("identified N128 schema/counts differ")

    upstream = spec["upstream"]
    _exact_keys(
        upstream,
        {
            "required_slurm_job_id", "campaign_name", "repository_sha",
            "expected_arm_count", "expected_setting_count", "eligible_model",
            "domain_order", "candidate_policy",
        },
        "upstream preregistration",
    )
    if (
        upstream.get("required_slurm_job_id") != "1425084"
        or upstream.get("campaign_name") != "paper_default_baseline_groupwise_fixed_followup_v1"
        or upstream.get("repository_sha") != "fcfbc490f664c64e4463c501cc0631a599f2cb25"
        or upstream.get("expected_arm_count") != 180
        or upstream.get("expected_setting_count") != 9
        or upstream.get("eligible_model") != MODEL
        or tuple(upstream.get("domain_order", [])) != DOMAINS
        or upstream.get("candidate_policy")
        != "a_domain_is_eligible_when_at_least_one_of_its_ten_ranked_fixed_candidates_passes_both_group_gates;_use_the_highest_upstream_ranked_eligible_candidate_only_for_domain_ranking_evidence"
    ):
        raise ValueError("upstream preregistration differs")

    common = spec["common"]
    expected_common = {
        "model": MODEL, "num_clients": 128, "rounds": 50,
        "batch_size": 8, "noise_multiplier": 2.0,
        "learning_rate": 5e-4, "rank": 512, "max_seq_length": 128,
        "max_validation_records": 512, "eval_every": 10,
        "checkpoint_every": 10, "data_split_seed": 1729,
        "evaluation_seed": 2718, "delta": 1e-5,
        "slaclip_num_slots": 5, "slaclip_eta": 0.05,
        "automatic_num_slots_upper_bound": 5,
        "theoretical_normalized_endpoint_noise_std": 0.39528470752104744,
        "slaclip_c_min": 0.025, "slaclip_c_max": 50.0,
        "slaclip_endpoint_epsilon": 1e-6,
    }
    if common != expected_common:
        raise ValueError("N128 mechanism/training constants differ")
    automatic_bound = automatic_num_slots(
        common["num_clients"], common["noise_multiplier"]
    )
    endpoint_noise = float(common["noise_multiplier"]) * math.sqrt(
        float(common["slaclip_num_slots"]) / float(common["num_clients"])
    )
    if (
        automatic_bound != common["automatic_num_slots_upper_bound"]
        or int(common["slaclip_num_slots"]) > automatic_bound
        or not math.isclose(
            endpoint_noise,
            float(common["theoretical_normalized_endpoint_noise_std"]),
            rel_tol=0.0, abs_tol=1e-15,
        )
    ):
        raise ValueError("N128/K5 automatic slot/noise mechanism lock differs")

    eligibility = spec["eligibility"]
    _exact_keys(
        eligibility,
        {
            "groups", "calibration_round_min", "calibration_round_max",
            "quantiles", "hard_rate_floor", "hard_rate_ceiling",
            "minimum_robust_interval_width",
            "minimum_distinct_pooled_hard_rate_strata",
            "maximum_fully_clipped_round_fraction",
            "maximum_mean_hard_clipped_fraction",
            "minimum_remaining_non_small_gradient_mass",
            "require_positive_cross_seed_overlap", "require_beta_identifiable",
            "require_box_constraint_feasible",
            "require_beta_strictly_inside_bounds",
            "profile_interpolation_positions", "domain_selection_rule",
        },
        "eligibility preregistration",
    )
    expected_domain_rule = [
        "largest_minimum_group_robust_interval_width",
        "smallest_maximum_group_fully_clipped_round_fraction",
        "largest_minimum_group_distinct_hard_rate_strata",
        "largest_minimum_group_P10_remaining_mass",
        "lowest_upstream_rank_of_first_eligible_candidate",
        "lowest_upstream_mean_final_loss",
        "preregistered_domain_order",
    ]
    if (
        eligibility.get("groups") != ["A", "B"]
        or eligibility.get("calibration_round_min") != 2
        or eligibility.get("calibration_round_max") != 50
        or eligibility.get("quantiles") != [0.1, 0.9]
        or eligibility.get("hard_rate_floor") != 0.05
        or eligibility.get("hard_rate_ceiling") != 0.85
        or eligibility.get("minimum_robust_interval_width") != 0.2
        or eligibility.get("minimum_distinct_pooled_hard_rate_strata") != 5
        or eligibility.get("maximum_fully_clipped_round_fraction") != 0.2
        or eligibility.get("maximum_mean_hard_clipped_fraction") != 0.9
        or eligibility.get("minimum_remaining_non_small_gradient_mass") != 0.1
        or eligibility.get("profile_interpolation_positions") != [0.0, 0.25, 0.5, 0.75, 1.0]
        or eligibility.get("domain_selection_rule") != expected_domain_rule
        or any(eligibility.get(name) is not True for name in (
            "require_positive_cross_seed_overlap", "require_beta_identifiable",
            "require_box_constraint_feasible", "require_beta_strictly_inside_bounds",
        ))
    ):
        raise ValueError("eligibility preregistration differs")

    fixed_screen = spec["fixed_screen"]
    _exact_keys(
        fixed_screen,
        {
            "seeds", "candidate_policy", "strongest_screened_fixed_selection_rule",
            "calibration_C0_policy",
        },
        "fixed-screen preregistration",
    )
    expected_fixed_rule = [
        "lowest_mean_final_internal_validation_loss",
        "lowest_mean_normalized_internal_validation_loss_auc",
        "lowest_final_loss_sample_std", "lowest_groupwise_noise_scale_l2",
        "smaller_C_A", "smaller_C_B",
    ]
    if (
        fixed_screen.get("seeds") != [1400, 1401]
        or fixed_screen.get("candidate_policy")
        != "all_ten_hash_locked_upstream_groupwise_fixed_profiles"
        or fixed_screen.get("strongest_screened_fixed_selection_rule") != expected_fixed_rule
        or fixed_screen.get("calibration_C0_policy")
        != "highest_best_of_10_screened_fixed_rank_that_repasses_all_N128_per_group_eligibility_and_five_profile_beta_feasibility_gates"
    ):
        raise ValueError("fixed-screen seeds differ")
    development = spec["profile_development"]
    _exact_keys(
        development,
        {
            "seeds", "profile_count", "eta", "initial_C_policy",
            "comparison_policy", "selection_rule",
            "hard_target_absolute_tolerance",
        },
        "profile-development preregistration",
    )
    expected_development_rule = [
        "feasible_beta_and_maximum_absolute_per_seed_both_group_target_error_at_most_0.15",
        "lowest_mean_paired_final_validation_loss_delta_vs_strongest_screened_fixed_C_star",
        "lowest_mean_paired_normalized_loss_auc_delta_vs_strongest_screened_fixed_C_star",
        "fewest_controller_instability_events", "lower_profile_index",
    ]
    if (
        development.get("seeds") != [1400, 1401]
        or development.get("profile_count") != 5
        or development.get("eta") != 0.05
        or development.get("hard_target_absolute_tolerance") != 0.15
        or development.get("initial_C_policy") != "N128_locked_calibration_C0"
        or development.get("comparison_policy")
        != "pair_each_profile_with_the_existing_same_seed_strongest_screened_fixed_C_star_arm_using_a_shared_sample_supervision_and_standard_Gaussian_noise_schedule;_evaluate_rounds_2_to_50_hard_target_error_per_seed_without_cross_seed_cancellation"
        or development.get("selection_rule") != expected_development_rule
    ):
        raise ValueError("profile-development preregistration differs")
    confirmation = spec["confirmation"]
    _exact_keys(
        confirmation,
        {
            "seeds", "methods", "primary_metric", "secondary_metric",
            "alpha", "minimum_wins", "mechanism_rounds",
            "mechanism_mean_abs_error_tolerance",
            "mechanism_seed_abs_error_tolerance",
            "mechanism_minimum_passing_seeds_per_group",
            "overall_positive_claim_rule", "success_gate",
        },
        "confirmation preregistration",
    )
    expected_confirmation_gate = [
        "mean_final_loss_delta_below_zero",
        "paired_two_sided_95pct_CI_upper_below_zero",
        "exact_two_sided_magnitude_preserving_sign_flip_p_below_0.05",
        "6_of_6_final_loss_wins",
        "mean_normalized_loss_auc_delta_not_positive",
        "mean_final_token_accuracy_delta_not_negative",
    ]
    if (
        confirmation.get("seeds") != [1500, 1501, 1502, 1503, 1504, 1505]
        or confirmation.get("methods") != [FIXED_METHOD, SLACLIP_METHOD]
        or confirmation.get("alpha") != 0.05
        or confirmation.get("minimum_wins") != 6
        or confirmation.get("mechanism_rounds") != [2, 50]
        or confirmation.get("mechanism_mean_abs_error_tolerance") != 0.15
        or confirmation.get("mechanism_seed_abs_error_tolerance") != 0.2
        or confirmation.get("mechanism_minimum_passing_seeds_per_group") != 5
        or confirmation.get("overall_positive_claim_rule")
        != "utility_success_gate_AND_mechanism_validity_gate"
        or confirmation.get("primary_metric")
        != "paired_final_internal_validation_loss_delta_SlaClip_minus_fixed"
        or confirmation.get("secondary_metric")
        != "paired_normalized_internal_validation_loss_auc_delta_SlaClip_minus_fixed"
        or confirmation.get("success_gate") != expected_confirmation_gate
    ):
        raise ValueError("confirmation preregistration differs")
    boundary = spec["scientific_boundary"]
    _exact_keys(
        boundary,
        {
            "full_slaclip_only",
            "desired_hard_clip_rate_is_distinct_from_beta_and_dynamic_target",
            "beta_source_is_per_group_only",
            "any_group_clipped_fraction_never_used_for_beta",
            "cdf_reconstruction_evidence_role",
            "private_diagnostic_label", "external_test_set_used",
            "paper_result_reproduced", "end_to_end_dp_certified",
            "single_allocation", "nested_sbatch_or_array",
            "all_arms_sequential", "comparator_scope",
        },
        "scientific boundary",
    )
    if (
        boundary.get("full_slaclip_only") is not True
        or boundary.get("desired_hard_clip_rate_is_distinct_from_beta_and_dynamic_target") is not True
        or boundary.get("beta_source_is_per_group_only") is not True
        or boundary.get("any_group_clipped_fraction_never_used_for_beta") is not True
        or boundary.get("cdf_reconstruction_evidence_role")
        != "descriptive_NON_DP_exact_vs_noisy_telemetry_without_accuracy_gate"
        or boundary.get("single_allocation") is not True
        or boundary.get("nested_sbatch_or_array") is not False
        or boundary.get("all_arms_sequential") is not True
        or boundary.get("private_diagnostic_label") != "NON_DP_PRIVATE_DIAGNOSTIC"
        or boundary.get("external_test_set_used") is not False
        or boundary.get("paper_result_reproduced") is not False
        or boundary.get("end_to_end_dp_certified") is not False
        or boundary.get("comparator_scope")
        != "strongest_screened_fixed_C_star_is_best_of_10_predeclared_groupwise_candidates_not_a_global_optimum"
    ):
        raise ValueError("scientific boundary differs")
    return spec


def _lock_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["lock_sha256"] = full.sha256_bytes(full.canonical_bytes(value))
    return value


def _load_lock(path: Path, label: str) -> dict[str, Any]:
    value = full.load_object(path.resolve(), label)
    supplied = value.get("lock_sha256")
    unsigned = {key: item for key, item in value.items() if key != "lock_sha256"}
    if supplied != full.sha256_bytes(full.canonical_bytes(unsigned)):
        raise RuntimeError(f"{label} self-hash differs")
    return value


def _write_or_verify(path: Path, candidate: Mapping[str, Any], label: str) -> None:
    if path.is_file():
        if full.load_object(path, label) != candidate:
            raise RuntimeError(f"existing {label} differs from immutable inputs")
    else:
        full.atomic_json(path, candidate)


def _csv_columns(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()) -> tuple[str, ...]:
    keys = {key for row in rows for key in row}
    return tuple([key for key in preferred if key in keys] + sorted(keys - set(preferred)))


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_or_verify_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str], label: str
) -> str:
    payload = _csv_bytes(rows, columns)
    digest = full.sha256_bytes(payload)
    if path.is_file():
        if full.sha256_file(path) != digest:
            raise RuntimeError(f"existing {label} differs from reconstructed evidence")
    else:
        full.atomic_write_text(path, payload.decode("utf-8"), mode=0o600)
    return digest


def _round_path(root: Path, arm: Mapping[str, Any], round_index: int) -> Path:
    model = str(arm.get("model", arm.get("models", [None])[0]))
    return (
        root / "arms" / str(arm["arm_id"]) / model / "private_diagnostics" /
        "rounds" / f"round-{round_index:05d}.json"
    )


def _fixed_trajectory_rows(
    root: Path, arms: Sequence[Mapping[str, Any]], *, slots: int, epsilon: float,
    round_min: int = 2, round_max: int = 50,
) -> list[dict[str, Any]]:
    """Recompute per-group exact endpoint evidence from raw client norms."""

    rows: list[dict[str, Any]] = []
    for arm in arms:
        thresholds = arm["initial_clip_norm_by_group"]
        model = str(arm.get("model", arm["models"][0]))
        if round_min != 2 or round_max != 50 or int(arm["rounds"]) != 50:
            raise RuntimeError("calibration trajectory must consume exactly rounds 2--50")
        for round_index in range(round_min, round_max + 1):
            shard = full.load_object(
                _round_path(root, arm, round_index), "fixed trajectory round shard"
            )
            records = shard.get("client_records")
            summary = shard.get("round_summary")
            if (
                shard.get("round") != round_index
                or shard.get("model") != model
                or shard.get("method") != FIXED_METHOD
                or not isinstance(records, list)
                or len(records) != int(arm["num_clients"])
                or not isinstance(summary, dict)
            ):
                raise RuntimeError(f"fixed trajectory identity differs: {arm['arm_id']}/{round_index}")
            for group in GROUPS:
                threshold = float(thresholds[group])
                norms: list[float] = []
                for record in records:
                    telemetry = record.get("gradient_groups", {}).get(group, {})
                    norm = float(telemetry.get("raw_norm", math.nan))
                    if (
                        not math.isfinite(norm)
                        or float(telemetry.get("clip_threshold", math.nan)) != threshold
                        or float(telemetry.get("noise_std_per_coordinate", math.nan))
                        != float(arm["noise_multiplier"]) * threshold
                    ):
                        raise RuntimeError(
                            f"fixed raw-norm/threshold/noise evidence differs: "
                            f"{arm['arm_id']}/{round_index}/{group}"
                        )
                    norms.append(norm)
                actual = float(summary[group]["clipped_fraction"])
                recomputed = sum(norm > threshold for norm in norms) / len(norms)
                if not math.isclose(actual, recomputed, rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError("persisted hard-clipping fraction differs from raw norms")
                signals = [build_slack_vector(norm, threshold, slots) for norm in norms]
                exact = normalize_noisy_slack(
                    [math.fsum(vector[index] for vector in signals) for index in range(slots)],
                    threshold, slots, len(norms),
                )
                raw_z = float(exact[-1]) / (threshold + epsilon)
                if not -1e-12 <= raw_z <= 1.0 + 1e-12:
                    raise RuntimeError("exact endpoint z lies outside its domain")
                z_value = min(1.0, max(0.0, raw_z))
                surrogate = min(1.0, max(0.0, 1.0 - float(exact[0])))
                rows.append({
                    "arm_id": arm["arm_id"], "seed": int(arm["seed"]),
                    "round": round_index, "group": group,
                    "fixed_C_group": threshold,
                    "actual_clipped_fraction": actual,
                    "exact_q_endpoint_1": float(exact[0]),
                    "exact_r_endpoint_K": float(exact[-1]),
                    "z_r_over_C_plus_epsilon": z_value,
                    "remaining_non_small_gradient_fraction": 1.0 - z_value,
                    "stationary_surrogate_target_clipped": surrogate,
                    "hard_minus_stationary_surrogate_target": actual - surrogate,
                })
    return rows


def assess_fixed_eligibility(
    rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply every preregistered A/B gate and derive five feasible profiles."""

    policy = spec["eligibility"]
    positions = [float(value) for value in policy["profile_interpolation_positions"]]
    expected_seeds = {int(seed) for seed in seeds}
    if {int(row["seed"]) for row in rows} != expected_seeds:
        raise RuntimeError("eligibility trajectory seed coverage differs")
    groups: dict[str, Any] = {}
    for group in GROUPS:
        subset = [dict(row) for row in rows if row["group"] == group]
        if not subset:
            raise RuntimeError(f"eligibility trajectory is empty for group {group}")
        actual = [float(row["actual_clipped_fraction"]) for row in subset]
        pooled_p10 = float(staged._linear_quantile(actual, 0.1))
        pooled_p90 = float(staged._linear_quantile(actual, 0.9))
        pooled_lower = max(pooled_p10, float(policy["hard_rate_floor"]))
        pooled_upper = min(pooled_p90, float(policy["hard_rate_ceiling"]))
        seed_intervals: dict[str, Any] = {}
        seed_lowers: list[float] = []
        seed_uppers: list[float] = []
        for seed in sorted(expected_seeds):
            values = [
                float(row["actual_clipped_fraction"])
                for row in subset if int(row["seed"]) == seed
            ]
            if not values:
                raise RuntimeError("eligibility cross-seed trajectory is incomplete")
            lower = max(float(staged._linear_quantile(values, 0.1)), float(policy["hard_rate_floor"]))
            upper = min(float(staged._linear_quantile(values, 0.9)), float(policy["hard_rate_ceiling"]))
            seed_lowers.append(lower)
            seed_uppers.append(upper)
            seed_intervals[str(seed)] = {"lower": lower, "upper": upper, "width": upper - lower}
        robust_lower = max([pooled_lower, *seed_lowers])
        robust_upper = min([pooled_upper, *seed_uppers])
        robust_width = robust_upper - robust_lower
        distinct = len(set(actual))
        fully_fraction = sum(value == 1.0 for value in actual) / len(actual)
        mean_hard = statistics.fmean(actual)
        remaining_p10 = float(staged._linear_quantile(
            [float(row["remaining_non_small_gradient_fraction"]) for row in subset], 0.1
        ))
        profiles: list[dict[str, Any]] = []
        calibration_error: str | None = None
        if robust_lower <= robust_upper:
            for position in positions:
                desired = robust_lower + position * (robust_upper - robust_lower)
                try:
                    fit = target_profile.calibrate_beta_for_hard_rate(subset, desired)
                except (RuntimeError, ValueError) as error:
                    calibration_error = f"{type(error).__name__}:{error}"
                    profiles = []
                    break
                profiles.append({"position": position, **fit})
        else:
            calibration_error = "negative_cross_seed_overlap"
        beta_feasible = len(profiles) == 5 and all(
            bool(profile["beta_identifiable"])
            and bool(profile["box_constraint_feasible"])
            and 0.0 < float(profile["beta"]) < 1.0
            for profile in profiles
        )
        gates = {
            "mean_hard_clip_below_0p90": mean_hard < float(policy["maximum_mean_hard_clipped_fraction"]),
            "robust_interval_width_at_least_0p20": robust_width >= float(policy["minimum_robust_interval_width"]),
            "at_least_five_distinct_pooled_hard_rate_strata": distinct >= int(policy["minimum_distinct_pooled_hard_rate_strata"]),
            "fully_clipped_round_fraction_below_0p20": fully_fraction < float(policy["maximum_fully_clipped_round_fraction"]),
            "positive_cross_seed_overlap": robust_lower < robust_upper,
            "P10_remaining_non_small_gradient_mass_above_0p10": remaining_p10 > float(policy["minimum_remaining_non_small_gradient_mass"]),
            "all_five_betas_identifiable_feasible_and_non_bound": beta_feasible,
        }
        groups[group] = {
            "pooled_P10": pooled_p10, "pooled_P90": pooled_p90,
            "bounded_pooled_lower": pooled_lower, "bounded_pooled_upper": pooled_upper,
            "cross_seed_intervals": seed_intervals,
            "robust_lower": robust_lower, "robust_upper": robust_upper,
            "robust_width": robust_width, "mean_hard_clipped_fraction": mean_hard,
            "distinct_pooled_hard_rate_strata": distinct,
            "fully_clipped_round_fraction": fully_fraction,
            "P10_remaining_non_small_gradient_mass": remaining_p10,
            "profiles": profiles, "calibration_error": calibration_error,
            "gates": gates, "eligible": all(gates.values()),
        }
    profiles_by_index: list[dict[str, Any]] = []
    if all(groups[group]["eligible"] for group in GROUPS):
        for index, position in enumerate(positions, start=1):
            profiles_by_index.append({
                "profile_index": index, "interpolation_position": position,
                "groups": {
                    group: groups[group]["profiles"][index - 1] for group in GROUPS
                },
            })
    return {
        "groups": groups, "profiles": profiles_by_index,
        "eligible": all(groups[group]["eligible"] for group in GROUPS),
        "any_group_clipped_fraction_consumed_for_beta": False,
    }


def _validate_upstream(
    *, root: Path, repository: Path, spec_path: Path,
    expected_repository_sha: str, required_job_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Rebuild the completed 180-arm parent lock from immutable evidence."""

    root = root.resolve()
    repository = repository.resolve()
    spec_path = spec_path.resolve()
    if (
        full.repository_sha(repository) != expected_repository_sha
        or full.repository_dirty(repository)
    ):
        raise RuntimeError("upstream repository SHA/cleanliness lock failed")
    upstream_spec = baseline.load_spec(spec_path)
    plan_path = root / baseline.PLAN_NAME
    selection_path = root / baseline.SELECTION_LOCK_NAME
    metrics_path = root / baseline.METRICS_NAME
    summary_path = root / baseline.SUMMARY_NAME
    for path in (plan_path, selection_path, metrics_path, summary_path):
        if not path.is_file():
            raise RuntimeError(f"completed upstream artifact is missing: {path}")
    plan = full.load_object(plan_path, "upstream fixed-followup plan")
    baseline.validate_plan(plan, upstream_spec)
    if (
        plan.get("repository_sha") != expected_repository_sha
        or plan.get("campaign_name") != upstream_spec["campaign_name"]
        or plan.get("expected_arm_count") != 180
        or plan.get("spec_sha256") != full.sha256_file(spec_path)
    ):
        raise RuntimeError("upstream plan provenance differs")
    summary = full.load_object(summary_path, "upstream campaign summary")
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("expected_arm_count") != 180
        or summary.get("status_counts", {}).get("COMPLETED") != 180
        or summary.get("candidate_plan_sha256") != plan["manifest_sha256"]
    ):
        raise RuntimeError("upstream fixed-followup campaign is incomplete")
    selection = baseline._validate_selection_artifacts(selection_path, metrics_path, 180)
    if (
        summary.get("campaign_name") != upstream_spec["campaign_name"]
        or summary.get("selection_lock_sha256") != selection.get("lock_sha256")
    ):
        raise RuntimeError("upstream summary/selection lock binding differs")
    rebuilt_base, rebuilt_rows = baseline.build_selection_lock(
        root=root, plan=plan, spec=upstream_spec,
        created_at_utc=str(selection.get("created_at_utc")),
    )
    rebuilt = baseline._bind_metrics_to_lock(rebuilt_base, metrics_path, len(rebuilt_rows))
    if rebuilt != selection:
        raise RuntimeError("upstream selection lock does not reproduce from all round shards")
    if (
        selection.get("status") != "GROUPWISE_FIXED_FOLLOWUP_SELECTION_LOCKED"
        or selection.get("all_candidate_count") != 180
        or len(selection.get("settings", [])) != 9
    ):
        raise RuntimeError("upstream selection identity differs")

    for arm in plan["arms"]:
        status_path = root / "arm-status" / f"{arm['arm_id']}-status.json"
        summary_file = root / "arms" / str(arm["arm_id"]) / "final_summary.json"
        status = baseline._verified_completed_status(
            status_path, summary_file, arm, plan, smoke=False
        )
        # The parent runner intentionally omits Slurm ownership from its final
        # COMPLETED status.  Dependency job provenance is preregistered, while
        # scientific identity is proven by the immutable artifacts below.
        if status is None:
            raise RuntimeError(
                f"upstream arm does not have a hash-locked completed status: {arm['arm_id']}"
            )

    input_records = plan.get("upstream", {}).get("input_manifests")
    if not isinstance(input_records, dict) or set(input_records) != set(DOMAINS):
        raise RuntimeError("upstream input-manifest provenance differs")
    for domain, record in input_records.items():
        path = Path(str(record.get("path"))).resolve()
        if not path.is_file() or full.sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"upstream locked input manifest changed: {domain}")
        value = full.load_object(path, f"upstream {domain} input manifest")
        if value.get("inventory_sha256") != record.get("inventory_sha256"):
            raise RuntimeError(f"upstream locked input inventory changed: {domain}")

    source = {
        "required_dependency_slurm_job_id": required_job_id,
        "dependency_job_id_is_preregistered_not_arm_status_attested": True,
        "campaign_root": str(root),
        "repository_path": str(repository),
        "repository_sha": expected_repository_sha,
        "spec_path": str(spec_path),
        "spec_sha256": full.sha256_file(spec_path),
        "candidate_plan_path": str(plan_path),
        "candidate_plan_sha256": plan["manifest_sha256"],
        "candidate_plan_file_sha256": full.sha256_file(plan_path),
        "selection_lock_path": str(selection_path),
        "selection_lock_sha256": selection["lock_sha256"],
        "selection_lock_file_sha256": full.sha256_file(selection_path),
        "campaign_metrics_path": str(metrics_path),
        "campaign_metrics_sha256": full.sha256_file(metrics_path),
        "campaign_summary_path": str(summary_path),
        "campaign_summary_sha256": full.sha256_file(summary_path),
        "verified_completed_arm_count": 180,
        "deep_rebuild_from_round_shards": True,
        "input_manifests": input_records,
    }
    return plan, selection, source, rebuilt_rows


def _candidate_arms(
    plan: Mapping[str, Any], *, domain: str, c_a: float, c_b: float,
) -> list[Mapping[str, Any]]:
    arms = [
        arm for arm in plan["arms"]
        if arm["domain"] == domain and arm["model"] == MODEL
        and float(arm["initial_clip_norm_by_group"]["A"]) == float(c_a)
        and float(arm["initial_clip_norm_by_group"]["B"]) == float(c_b)
    ]
    if len(arms) != 2 or {int(arm["seed"]) for arm in arms} != {1300, 1301}:
        raise RuntimeError(f"upstream candidate trajectory coverage differs: {domain}/{c_a}/{c_b}")
    return sorted(arms, key=lambda arm: int(arm["seed"]))


def derive_upstream_eligibility(
    *, upstream_root: Path, plan: Mapping[str, Any], selection: Mapping[str, Any],
    source: Mapping[str, Any], spec: Mapping[str, Any], created_at_utc: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    settings = {
        (str(record["domain"]), str(record["model"])): record
        for record in selection["settings"]
    }
    domain_records: list[dict[str, Any]] = []
    all_trajectory: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(DOMAINS):
        setting = settings.get((domain, MODEL))
        if not isinstance(setting, dict):
            raise RuntimeError(f"upstream GPT-2 setting is missing: {domain}")
        ordered = setting.get("ordered_candidates")
        if not isinstance(ordered, list) or len(ordered) != 10:
            raise RuntimeError(f"upstream GPT-2 candidate ranking differs: {domain}")
        candidates: list[dict[str, Any]] = []
        seen_thresholds: set[tuple[float, float]] = set()
        for rank, candidate in enumerate(ordered, start=1):
            c_a, c_b = float(candidate["C_A"]), float(candidate["C_B"])
            pair = (c_a, c_b)
            if pair in seen_thresholds:
                raise RuntimeError(f"upstream candidate thresholds are duplicated: {domain}/{pair}")
            seen_thresholds.add(pair)
            arms = _candidate_arms(plan, domain=domain, c_a=c_a, c_b=c_b)
            trajectory = _fixed_trajectory_rows(
                upstream_root, arms,
                slots=int(spec["common"]["slaclip_num_slots"]),
                epsilon=float(spec["common"]["slaclip_endpoint_epsilon"]),
                round_min=int(spec["eligibility"]["calibration_round_min"]),
                round_max=int(spec["eligibility"]["calibration_round_max"]),
            )
            eligibility = assess_fixed_eligibility(trajectory, [1300, 1301], spec)
            trajectory_sha = full.sha256_bytes(full.canonical_bytes(trajectory))
            all_trajectory.extend({"domain": domain, "candidate_rank": rank, **row} for row in trajectory)
            record = {
                "upstream_rank": rank, "C_A": c_a, "C_B": c_b,
                "upstream_mean_final_loss": float(candidate["mean_final_loss"]),
                "upstream_mean_normalized_loss_auc": float(candidate["mean_normalized_loss_auc"]),
                "trajectory_rows": len(trajectory),
                "trajectory_canonical_sha256": trajectory_sha,
                "eligibility": eligibility,
            }
            candidates.append(record)
            for group in GROUPS:
                evidence = eligibility["groups"][group]
                summary_rows.append({
                    "domain": domain, "model": MODEL, "candidate_rank": rank,
                    "C_A": c_a, "C_B": c_b, "group": group,
                    "eligible": evidence["eligible"],
                    "mean_hard_clipped_fraction": evidence["mean_hard_clipped_fraction"],
                    "pooled_P10": evidence["pooled_P10"], "pooled_P90": evidence["pooled_P90"],
                    "robust_lower": evidence["robust_lower"], "robust_upper": evidence["robust_upper"],
                    "robust_width": evidence["robust_width"],
                    "distinct_hard_rate_strata": evidence["distinct_pooled_hard_rate_strata"],
                    "fully_clipped_round_fraction": evidence["fully_clipped_round_fraction"],
                    "P10_remaining_non_small_gradient_mass": evidence["P10_remaining_non_small_gradient_mass"],
                })
        eligible_candidates = [record for record in candidates if record["eligibility"]["eligible"]]
        first = eligible_candidates[0] if eligible_candidates else None
        if first is None:
            rank_key: list[Any] | None = None
        else:
            groups = first["eligibility"]["groups"]
            rank_key = [
                -min(float(groups[group]["robust_width"]) for group in GROUPS),
                max(float(groups[group]["fully_clipped_round_fraction"]) for group in GROUPS),
                -min(int(groups[group]["distinct_pooled_hard_rate_strata"]) for group in GROUPS),
                -min(float(groups[group]["P10_remaining_non_small_gradient_mass"]) for group in GROUPS),
                int(first["upstream_rank"]), float(first["upstream_mean_final_loss"]),
                domain_index,
            ]
        domain_records.append({
            "domain": domain, "model": MODEL, "eligible": first is not None,
            "first_eligible_candidate": first, "ordered_candidates": candidates,
            "preregistered_rank_key": rank_key,
        })
    eligible_domains = [record for record in domain_records if record["eligible"]]
    if not eligible_domains:
        raise RuntimeError("no GPT-2 domain passes the preregistered per-group eligibility gates")
    eligible_domains.sort(key=lambda record: tuple(record["preregistered_rank_key"]))
    selected = eligible_domains[0]
    lock = _lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "UPSTREAM_GPT2_DOMAIN_ELIGIBILITY_LOCKED",
        "campaign_name": spec["campaign_name"],
        "source": dict(source),
        "eligibility_policy": spec["eligibility"],
        "domain_selection_rule": spec["eligibility"]["domain_selection_rule"],
        "eligible_domain_count": len(eligible_domains),
        "selected_domain": selected["domain"],
        "selected_domain_first_eligible_candidate": selected["first_eligible_candidate"],
        "ordered_domains": sorted(
            domain_records,
            key=lambda record: (
                0 if record["eligible"] else 1,
                tuple(record["preregistered_rank_key"] or [math.inf]),
            ),
        ),
        "all_trajectory_rows": len(all_trajectory),
        "all_trajectory_canonical_sha256": full.sha256_bytes(full.canonical_bytes(all_trajectory)),
        "any_group_clipped_fraction_consumed_for_beta": False,
        "created_at_utc": created_at_utc,
    })
    return lock, all_trajectory, summary_rows


def _token(value: float) -> str:
    return full.number_token(float(value))


def _base_arm(
    spec: Mapping[str, Any], *, arm_id: str, stage: str, domain: str,
    method: str, seed: int, thresholds: Mapping[str, float],
    reference_arm_id: str | None, rng_domain: str,
    profile: Mapping[str, Any] | None = None,
    calibration_lock_sha256: str | None = None,
    candidate_rank: int | None = None,
) -> dict[str, Any]:
    adaptive = method == SLACLIP_METHOD
    if method not in {FIXED_METHOD, SLACLIP_METHOD}:
        raise ValueError("identified campaign only supports fixed DP-LoRA and Full SlaClip")
    if adaptive != (profile is not None):
        raise ValueError("adaptive arm/profile arguments disagree")
    if stage not in {FIXED_STAGE, DEVELOPMENT_STAGE, CONFIRMATION_STAGE}:
        raise ValueError("identified campaign stage differs")
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
        or any(not 0.0 < value < 1.0 for value in (betas or {}).values())
    ):
        raise ValueError("adaptive calibration lock/beta differs")
    common = spec["common"]
    roles = {
        FIXED_STAGE: "N128_all_ten_parent_groupwise_fixed_ceiling_and_calibration_screen",
        DEVELOPMENT_STAGE: "paired_five_profile_Full_SlaClip_development",
        CONFIRMATION_STAGE: "independent_frozen_fixed_vs_Full_SlaClip_confirmation",
    }
    return {
        "arm_id": arm_id, "stage": stage, "family": stage,
        "analysis_role": roles[stage], "domain": domain, "model": MODEL,
        "candidate_rank": candidate_rank,
        "target_profile_index": int(profile["profile_index"]) if profile else None,
        "target_profile_position": float(profile["interpolation_position"]) if profile else None,
        "desired_hard_clipped_fraction_by_group": desired,
        "method": method, "seed": int(seed),
        "initial_clip_norm": c_b,
        "initial_clip_norm_by_group": {"A": c_a, "B": c_b},
        "slaclip_eta": float(common["slaclip_eta"]) if adaptive else None,
        "slaclip_base_target_clipped_fraction": None,
        "slaclip_beta": None,
        "slaclip_base_target_clipped_fraction_by_group": betas,
        "slaclip_beta_by_group": dict(betas) if betas else None,
        "slaclip_baseline_calibration_lock_sha256": calibration_lock_sha256 if adaptive else None,
        "acknowledge_slaclip_baseline_calibration_is_non_dp": adaptive,
        "slaclip_calibration_provenance": CALIBRATION_PROVENANCE if adaptive else None,
        "controller_input": full.CONTROLLER_INPUT_BY_METHOD.get(method),
        "reference_arm_id": reference_arm_id, "rng_domain": rng_domain,
        "models": [MODEL], "num_clients": common["num_clients"],
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


def fixed_screen_arms(
    spec: Mapping[str, Any], domain: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(candidates) != 10:
        raise RuntimeError("fixed screen requires all ten upstream candidates")
    pairs = [(float(row["C_A"]), float(row["C_B"])) for row in candidates]
    if len(set(pairs)) != 10:
        raise RuntimeError("fixed screen refuses deduplicated upstream threshold pairs")
    arms: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        if int(candidate.get("upstream_rank", rank)) != rank:
            raise RuntimeError("upstream candidate rank order differs")
        thresholds = {"A": float(candidate["C_A"]), "B": float(candidate["C_B"])}
        for seed in spec["fixed_screen"]["seeds"]:
            arms.append(_base_arm(
                spec,
                arm_id=(
                    f"n128-fixed-p{rank:02d}-ca{_token(thresholds['A'])}-"
                    f"cb{_token(thresholds['B'])}-s{seed}"
                ),
                stage=FIXED_STAGE, domain=domain, method=FIXED_METHOD,
                seed=int(seed), thresholds=thresholds, reference_arm_id=None,
                rng_domain=f"identified-n128:{domain}:{MODEL}:s{seed}",
                candidate_rank=rank,
            ))
    return staged._indexed(arms)


def development_arms(
    spec: Mapping[str, Any], domain: str, fixed_lock: Mapping[str, Any]
) -> list[dict[str, Any]]:
    c0 = fixed_lock["selected_calibration_C0_by_group"]
    c_star_refs = {
        int(record["seed"]): str(record["arm_id"])
        for record in fixed_lock["strongest_screened_fixed_C_star_seed_evidence"]
    }
    profiles = fixed_lock["calibrated_profiles"]
    if len(profiles) != 5:
        raise RuntimeError("N128 calibration lock must contain five profiles")
    arms: list[dict[str, Any]] = []
    for profile in profiles:
        index = int(profile["profile_index"])
        for seed in spec["profile_development"]["seeds"]:
            arms.append(_base_arm(
                spec,
                arm_id=(
                    f"n128-dev-fullslaclip-p{index}-ca{_token(float(c0['A']))}-"
                    f"cb{_token(float(c0['B']))}-s{seed}"
                ),
                stage=DEVELOPMENT_STAGE, domain=domain, method=SLACLIP_METHOD,
                seed=int(seed), thresholds=c0,
                reference_arm_id=c_star_refs[int(seed)],
                rng_domain=f"identified-n128:{domain}:{MODEL}:s{seed}",
                profile=profile,
                calibration_lock_sha256=str(fixed_lock["lock_sha256"]),
            ))
    return staged._indexed(arms)


def confirmation_arms(
    spec: Mapping[str, Any], domain: str, fixed_lock: Mapping[str, Any],
    development_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    c_star = fixed_lock["strongest_screened_fixed_C_star_by_group"]
    c0 = fixed_lock["selected_calibration_C0_by_group"]
    profile = development_lock["selected_profile"]
    arms: list[dict[str, Any]] = []
    for seed in spec["confirmation"]["seeds"]:
        fixed_id = (
            f"n128-confirm-fixed-ca{_token(float(c_star['A']))}-"
            f"cb{_token(float(c_star['B']))}-s{seed}"
        )
        rng_domain = f"identified-n128:{domain}:{MODEL}:s{seed}"
        arms.append(_base_arm(
            spec, arm_id=fixed_id, stage=CONFIRMATION_STAGE, domain=domain,
            method=FIXED_METHOD, seed=int(seed), thresholds=c_star,
            reference_arm_id=None, rng_domain=rng_domain,
        ))
        arms.append(_base_arm(
            spec,
            arm_id=(
                f"n128-confirm-fullslaclip-p{profile['profile_index']}-"
                f"ca{_token(float(c0['A']))}-cb{_token(float(c0['B']))}-s{seed}"
            ),
            stage=CONFIRMATION_STAGE, domain=domain, method=SLACLIP_METHOD,
            seed=int(seed), thresholds=c0, reference_arm_id=fixed_id,
            rng_domain=rng_domain, profile=profile,
            calibration_lock_sha256=str(fixed_lock["lock_sha256"]),
        ))
    return staged._indexed(arms)


def _runtime_manifest(
    *, spec: Mapping[str, Any], spec_path: Path, repository_sha: str,
    input_record: Mapping[str, Any], stage: str, arms: list[dict[str, Any]],
    created_at_utc: str, parent_lock_sha256: str,
    private_key_commitment: str,
) -> dict[str, Any]:
    if (
        len(private_key_commitment) != 64
        or any(character not in "0123456789abcdef" for character in private_key_commitment)
    ):
        raise ValueError("runtime private-key commitment is invalid")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "IDENTIFIED_STAGE_RUNTIME_LOCKED",
        "campaign_name": spec["campaign_name"], "stage": stage,
        "created_at_utc": created_at_utc, "repository_sha": repository_sha,
        "spec_path": str(spec_path.resolve()), "spec_sha256": full.sha256_file(spec_path.resolve()),
        "input_manifest_path": str(Path(str(input_record["path"])).resolve()),
        "input_manifest_sha256": str(input_record["sha256"]),
        "input_inventory_sha256": str(input_record["inventory_sha256"]),
        "parent_lock_sha256": parent_lock_sha256,
        "private_key_commitment": private_key_commitment,
        "expected_arm_count": len(arms),
        "scientific_boundary": spec["scientific_boundary"], "arms": arms,
    }
    manifest["manifest_sha256"] = full.sha256_bytes(full.canonical_bytes(manifest))
    full.validate_runtime_manifest(manifest)
    return manifest


def _plan_payload(
    *, spec: Mapping[str, Any], spec_path: Path, repository_sha: str,
    private_key: Path, upstream_lock: Mapping[str, Any],
    fixed_manifest: Mapping[str, Any], created_at_utc: str,
) -> dict[str, Any]:
    selected_domain = str(upstream_lock["selected_domain"])
    selected_record = next(
        row for row in upstream_lock["ordered_domains"] if row["domain"] == selected_domain
    )
    candidates = [
        {"upstream_rank": row["upstream_rank"], "C_A": row["C_A"], "C_B": row["C_B"]}
        for row in selected_record["ordered_candidates"]
    ]
    source_input = upstream_lock["source"]["input_manifests"][selected_domain]
    return _lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "IDENTIFIED_N128_CAMPAIGN_PREREGISTERED",
        "campaign_name": spec["campaign_name"], "created_at_utc": created_at_utc,
        "repository_sha": repository_sha, "spec_path": str(spec_path.resolve()),
        "spec_sha256": full.sha256_file(spec_path.resolve()),
        "private_key_commitment": full.sha256_file(private_key.resolve()),
        "upstream_eligibility_lock_sha256": upstream_lock["lock_sha256"],
        "selected_domain": selected_domain, "model": MODEL,
        "selected_domain_input_manifest": source_input,
        "all_ten_fixed_candidates": candidates,
        "fixed_stage_runtime_manifest_sha256": fixed_manifest["manifest_sha256"],
        "expected_stage_arm_counts": spec["expected_stage_arm_counts"],
        "mechanism_lock": {
            "N": spec["common"]["num_clients"],
            "K": spec["common"]["slaclip_num_slots"],
            "sigma": spec["common"]["noise_multiplier"],
            "automatic_num_slots_upper_bound": automatic_num_slots(
                spec["common"]["num_clients"], spec["common"]["noise_multiplier"]
            ),
            "theoretical_normalized_endpoint_noise_std": spec["common"]["theoretical_normalized_endpoint_noise_std"],
        },
        "scientific_boundary": spec["scientific_boundary"],
    })


def prepare(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    repository = args.repository.resolve()
    upstream_root = args.upstream_root.resolve()
    upstream_repository = args.upstream_repository.resolve()
    upstream_spec_path = args.upstream_spec.resolve()
    spec = load_spec(spec_path)
    if (
        full.repository_sha(repository) != args.expected_code_sha
        or full.repository_dirty(repository)
    ):
        raise RuntimeError("repository snapshot SHA/cleanliness gate failed")
    full.validate_or_create_key(args.private_key.resolve(), create=False)
    if root.exists() and not args.resume:
        raise RuntimeError(f"refusing to overwrite campaign root: {root}")
    if args.resume:
        existing_plan = _load_lock(root / PLAN_NAME, "identified campaign plan")
        created_at_utc = str(existing_plan["created_at_utc"])
    else:
        if root.exists():
            raise RuntimeError("resume campaign root does not have a locked plan")
        created_at_utc = full.utc_now()

    upstream_plan, upstream_selection, source, _rows = _validate_upstream(
        root=upstream_root, repository=upstream_repository,
        spec_path=upstream_spec_path,
        expected_repository_sha=str(spec["upstream"]["repository_sha"]),
        required_job_id=str(spec["upstream"]["required_slurm_job_id"]),
    )
    eligibility_lock, trajectory_rows, eligibility_rows = derive_upstream_eligibility(
        upstream_root=upstream_root, plan=upstream_plan,
        selection=upstream_selection, source=source, spec=spec,
        created_at_utc=created_at_utc,
    )
    trajectory_columns = _csv_columns(
        trajectory_rows,
        ("domain", "candidate_rank", "arm_id", "seed", "round", "group"),
    )
    eligibility_columns = _csv_columns(
        eligibility_rows,
        ("domain", "model", "candidate_rank", "C_A", "C_B", "group", "eligible"),
    )
    trajectory_bytes = _csv_bytes(trajectory_rows, trajectory_columns)
    eligibility_bytes = _csv_bytes(eligibility_rows, eligibility_columns)
    unsigned = {key: value for key, value in eligibility_lock.items() if key != "lock_sha256"}
    unsigned.update({
        "trajectory_csv": "upstream_gpt2_candidate_trajectory.csv",
        "trajectory_csv_sha256": full.sha256_bytes(trajectory_bytes),
        "eligibility_csv": "upstream_gpt2_candidate_eligibility.csv",
        "eligibility_csv_sha256": full.sha256_bytes(eligibility_bytes),
    })
    eligibility_lock = _lock_payload(unsigned)

    selected_domain = str(eligibility_lock["selected_domain"])
    selected_record = next(
        row for row in eligibility_lock["ordered_domains"]
        if row["domain"] == selected_domain
    )
    fixed_arms = fixed_screen_arms(spec, selected_domain, selected_record["ordered_candidates"])
    input_record = source["input_manifests"][selected_domain]
    fixed_manifest = _runtime_manifest(
        spec=spec, spec_path=spec_path, repository_sha=args.expected_code_sha,
        input_record=input_record, stage=FIXED_STAGE, arms=fixed_arms,
        created_at_utc=created_at_utc,
        parent_lock_sha256=str(eligibility_lock["lock_sha256"]),
        private_key_commitment=full.sha256_file(args.private_key.resolve()),
    )
    plan = _plan_payload(
        spec=spec, spec_path=spec_path, repository_sha=args.expected_code_sha,
        private_key=args.private_key.resolve(), upstream_lock=eligibility_lock,
        fixed_manifest=fixed_manifest, created_at_utc=created_at_utc,
    )

    if not root.exists():
        root.mkdir(parents=True, mode=0o700)
    for name in ("arms", "arm-status", "arm-logs", "control", "preflight", "tmp"):
        (root / name).mkdir(mode=0o700, exist_ok=True)
    _write_or_verify_csv(
        root / "upstream_gpt2_candidate_trajectory.csv", trajectory_rows,
        trajectory_columns, "upstream GPT-2 trajectory CSV",
    )
    _write_or_verify_csv(
        root / "upstream_gpt2_candidate_eligibility.csv", eligibility_rows,
        eligibility_columns, "upstream GPT-2 eligibility CSV",
    )
    _write_or_verify(root / UPSTREAM_LOCK_NAME, eligibility_lock, "upstream eligibility lock")
    _write_or_verify(root / FIXED_MANIFEST_NAME, fixed_manifest, "fixed-screen runtime manifest")
    _write_or_verify(root / PLAN_NAME, plan, "identified campaign plan")
    stop = root / "control" / "stop.request"
    if stop.exists():
        stop.unlink()
    print(json.dumps({
        "status": "READY", "campaign_plan": str(root / PLAN_NAME),
        "selected_domain": selected_domain, "model": MODEL,
        "fixed_screen_arms": len(fixed_arms), "total_preregistered_arms": 42,
    }, indent=2, sort_keys=True))


def _identity(
    root: Path, spec_path: Path, repository: Path, expected_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    spec_path = spec_path.resolve()
    repository = repository.resolve()
    spec = load_spec(spec_path)
    if full.repository_sha(repository) != expected_sha or full.repository_dirty(repository):
        raise RuntimeError("repository snapshot SHA/cleanliness gate failed")
    plan = _load_lock(root / PLAN_NAME, "identified campaign plan")
    upstream_lock = _load_lock(root / UPSTREAM_LOCK_NAME, "upstream eligibility lock")
    fixed_manifest = full.load_runtime(root / FIXED_MANIFEST_NAME)
    if (
        plan.get("status") != "IDENTIFIED_N128_CAMPAIGN_PREREGISTERED"
        or plan.get("repository_sha") != expected_sha
        or plan.get("spec_sha256") != full.sha256_file(spec_path)
        or plan.get("upstream_eligibility_lock_sha256") != upstream_lock["lock_sha256"]
        or plan.get("fixed_stage_runtime_manifest_sha256") != fixed_manifest["manifest_sha256"]
        or fixed_manifest.get("repository_sha") != expected_sha
        or fixed_manifest.get("parent_lock_sha256") != upstream_lock["lock_sha256"]
        or fixed_manifest.get("private_key_commitment") != plan["private_key_commitment"]
        or len(fixed_manifest["arms"]) != EXPECTED_COUNTS[FIXED_STAGE]
    ):
        raise RuntimeError("identified campaign immutable identity differs")
    for name in (
        "trajectory_csv", "eligibility_csv",
    ):
        path = root / str(upstream_lock[name])
        if not path.is_file() or full.sha256_file(path) != upstream_lock[f"{name}_sha256"]:
            raise RuntimeError(f"upstream eligibility artifact changed: {name}")
    source = upstream_lock["source"]
    source_hashes = {
        "spec_path": "spec_sha256", "candidate_plan_path": "candidate_plan_file_sha256",
        "selection_lock_path": "selection_lock_file_sha256",
        "campaign_metrics_path": "campaign_metrics_sha256",
        "campaign_summary_path": "campaign_summary_sha256",
    }
    for path_key, hash_key in source_hashes.items():
        path = Path(str(source[path_key])).resolve()
        if not path.is_file() or full.sha256_file(path) != source[hash_key]:
            raise RuntimeError(f"upstream source artifact changed: {path_key}")
    input_record = plan["selected_domain_input_manifest"]
    input_path = Path(str(input_record["path"])).resolve()
    if (
        not input_path.is_file()
        or full.sha256_file(input_path) != input_record["sha256"]
        or full.load_object(input_path, "selected input manifest").get("inventory_sha256")
        != input_record["inventory_sha256"]
    ):
        raise RuntimeError("selected domain input manifest/inventory changed")
    return spec, plan, upstream_lock, fixed_manifest


def _metrics_for_manifest(
    root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in manifest["arms"]:
        row = oracle._metric_row(root, manifest, arm)
        row.update({
            "stage": arm["stage"], "domain": arm["domain"],
            "candidate_rank": arm.get("candidate_rank"),
            "target_profile_index": arm.get("target_profile_index"),
            "target_profile_position": arm.get("target_profile_position"),
            "desired_hard_clipped_fraction_A": (
                arm.get("desired_hard_clipped_fraction_by_group", {}).get("A")
                if isinstance(arm.get("desired_hard_clipped_fraction_by_group"), dict)
                else None
            ),
            "desired_hard_clipped_fraction_B": (
                arm.get("desired_hard_clipped_fraction_by_group", {}).get("B")
                if isinstance(arm.get("desired_hard_clipped_fraction_by_group"), dict)
                else None
            ),
        })
        rows.append(row)
    return rows


def _fixed_rankings(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    expected_seeds = set(spec["fixed_screen"]["seeds"])
    for rank in range(1, 11):
        subset = [row for row in rows if int(row["candidate_rank"]) == rank]
        arms = [arm for arm in manifest["arms"] if int(arm["candidate_rank"]) == rank]
        if (
            len(subset) != 2 or {int(row["seed"]) for row in subset} != expected_seeds
            or len(arms) != 2
        ):
            raise RuntimeError(f"N128 fixed candidate coverage differs: rank {rank}")
        thresholds = arms[0]["initial_clip_norm_by_group"]
        finals = [float(row["final_loss"]) for row in subset]
        aucs = [float(row["normalized_loss_auc"]) for row in subset]
        rankings.append({
            "upstream_candidate_rank": rank,
            "C_A": float(thresholds["A"]), "C_B": float(thresholds["B"]),
            "seed_count": 2, "mean_final_loss": statistics.fmean(finals),
            "mean_normalized_loss_auc": statistics.fmean(aucs),
            "final_loss_sample_std": statistics.stdev(finals),
            "groupwise_noise_scale_l2": float(spec["common"]["noise_multiplier"])
            * math.sqrt(float(thresholds["A"]) ** 2 + float(thresholds["B"]) ** 2),
            "mean_actual_clipped_fraction_A": statistics.fmean(
                float(row["actual_clipped_fraction_A"]) for row in subset
            ),
            "mean_actual_clipped_fraction_B": statistics.fmean(
                float(row["actual_clipped_fraction_B"]) for row in subset
            ),
            "seed_evidence": sorted(
                [dict(row) for row in subset], key=lambda row: int(row["seed"])
            ),
        })
    rankings.sort(key=lambda row: (
        row["mean_final_loss"], row["mean_normalized_loss_auc"],
        row["final_loss_sample_std"], row["groupwise_noise_scale_l2"],
        row["C_A"], row["C_B"],
    ))
    return rankings


def _ensure_development_manifest(
    root: Path, spec_path: Path, spec: Mapping[str, Any], plan: Mapping[str, Any],
    fixed_lock: Mapping[str, Any],
) -> Path:
    arms = development_arms(spec, str(plan["selected_domain"]), fixed_lock)
    if len(arms) != EXPECTED_COUNTS[DEVELOPMENT_STAGE]:
        raise RuntimeError("profile-development arm count differs")
    manifest = _runtime_manifest(
        spec=spec, spec_path=spec_path,
        repository_sha=str(plan["repository_sha"]),
        input_record=plan["selected_domain_input_manifest"],
        stage=DEVELOPMENT_STAGE, arms=arms,
        created_at_utc=str(plan["created_at_utc"]),
        parent_lock_sha256=str(fixed_lock["lock_sha256"]),
        private_key_commitment=str(plan["private_key_commitment"]),
    )
    path = root / DEVELOPMENT_MANIFEST_NAME
    _write_or_verify(path, manifest, "profile-development runtime manifest")
    return path


def lock_fixed(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, plan, _upstream_lock, manifest = _identity(
        root, spec_path, args.repository, args.expected_code_sha
    )
    lock_path = root / FIXED_LOCK_NAME
    existing = _load_lock(lock_path, "N128 fixed selection lock") if lock_path.is_file() else None
    rows = _metrics_for_manifest(root, manifest)
    evidence = staged._arm_evidence(root, manifest)
    rankings = _fixed_rankings(rows, manifest, spec)
    strongest = rankings[0]
    by_id = {str(arm["arm_id"]): arm for arm in manifest["arms"]}
    calibration_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for candidate in rankings:
        source_arms = [by_id[str(row["arm_id"])] for row in candidate["seed_evidence"]]
        trajectory = _fixed_trajectory_rows(
            root, source_arms,
            slots=int(spec["common"]["slaclip_num_slots"]),
            epsilon=float(spec["common"]["slaclip_endpoint_epsilon"]),
            round_min=int(spec["eligibility"]["calibration_round_min"]),
            round_max=int(spec["eligibility"]["calibration_round_max"]),
        )
        eligibility = assess_fixed_eligibility(
            trajectory, spec["fixed_screen"]["seeds"], spec
        )
        candidate["N128_eligibility"] = eligibility
        candidate["N128_calibration_trajectory_rows"] = len(trajectory)
        candidate["N128_calibration_trajectory_canonical_sha256"] = full.sha256_bytes(
            full.canonical_bytes(trajectory)
        )
        trajectory_rows.extend({
            "upstream_candidate_rank": candidate["upstream_candidate_rank"],
            "strongest_screened_fixed_rank": rankings.index(candidate) + 1, **row,
        } for row in trajectory)
        for group in GROUPS:
            group_record = eligibility["groups"][group]
            calibration_rows.append({
                "strongest_screened_fixed_rank": rankings.index(candidate) + 1,
                "upstream_candidate_rank": candidate["upstream_candidate_rank"],
                "C_A": candidate["C_A"], "C_B": candidate["C_B"],
                "group": group, "eligible": group_record["eligible"],
                "mean_hard_clipped_fraction": group_record["mean_hard_clipped_fraction"],
                "pooled_P10": group_record["pooled_P10"], "pooled_P90": group_record["pooled_P90"],
                "robust_lower": group_record["robust_lower"],
                "robust_upper": group_record["robust_upper"],
                "robust_width": group_record["robust_width"],
                "distinct_hard_rate_strata": group_record["distinct_pooled_hard_rate_strata"],
                "fully_clipped_round_fraction": group_record["fully_clipped_round_fraction"],
                "P10_remaining_non_small_gradient_mass": group_record["P10_remaining_non_small_gradient_mass"],
            })
    eligible = [row for row in rankings if row["N128_eligibility"]["eligible"]]
    if not eligible:
        raise RuntimeError("no N128 fixed candidate passes calibration eligibility")
    calibration = eligible[0]
    if len(calibration["N128_eligibility"]["profiles"]) != 5:
        raise RuntimeError("selected N128 calibration C0 lacks five profiles")

    metric_columns = _csv_columns(rows, ("stage", "domain", "model", "arm_id", "seed", "method"))
    ranking_rows = [
        {
            "strongest_screened_fixed_rank": index,
            "upstream_candidate_rank": row["upstream_candidate_rank"],
            "C_A": row["C_A"], "C_B": row["C_B"],
            "mean_final_loss": row["mean_final_loss"],
            "mean_normalized_loss_auc": row["mean_normalized_loss_auc"],
            "final_loss_sample_std": row["final_loss_sample_std"],
            "groupwise_noise_scale_l2": row["groupwise_noise_scale_l2"],
            "N128_calibration_eligible": row["N128_eligibility"]["eligible"],
        }
        for index, row in enumerate(rankings, start=1)
    ]
    artifacts = {
        "fixed_screen_metrics.csv": (rows, metric_columns),
        "fixed_screen_ranking.csv": (ranking_rows, _csv_columns(ranking_rows)),
        "n128_calibration_eligibility.csv": (calibration_rows, _csv_columns(calibration_rows)),
        "n128_calibration_trajectory.csv": (
            trajectory_rows,
            _csv_columns(trajectory_rows, ("strongest_screened_fixed_rank", "upstream_candidate_rank", "arm_id", "seed", "round", "group")),
        ),
    }
    profile_rows = [
        {
            "profile_index": profile["profile_index"],
            "interpolation_position": profile["interpolation_position"],
            "group": group,
            "desired_hard_clipped_fraction": profile["groups"][group]["desired_hard_clipped_fraction"],
            "beta": profile["groups"][group]["beta"],
            "weighted_stationary_surrogate_target_mean": profile["groups"][group]["weighted_stationary_surrogate_target_mean"],
            "predicted_dynamic_target_mean": profile["groups"][group]["predicted_dynamic_target_mean"],
            "surrogate_fit_mae": profile["groups"][group]["surrogate_fit_mae"],
            "surrogate_fit_rmse": profile["groups"][group]["surrogate_fit_rmse"],
            "beta_identifiable": profile["groups"][group]["beta_identifiable"],
            "box_constraint_feasible": profile["groups"][group]["box_constraint_feasible"],
            "beta_hit_lower_bound": profile["groups"][group]["beta_hit_lower_bound"],
            "beta_hit_upper_bound": profile["groups"][group]["beta_hit_upper_bound"],
        }
        for profile in calibration["N128_eligibility"]["profiles"]
        for group in GROUPS
    ]
    artifacts["n128_target_profile_calibration.csv"] = (
        profile_rows,
        _csv_columns(profile_rows, ("profile_index", "interpolation_position", "group")),
    )
    artifact_hashes = {
        name: full.sha256_bytes(_csv_bytes(values, columns))
        for name, (values, columns) in artifacts.items()
    }
    created = str(existing["created_at_utc"]) if existing else full.utc_now()
    candidate_lock = _lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "N128_FIXED_CEILING_AND_CALIBRATION_LOCKED",
        "campaign_name": spec["campaign_name"], "campaign_plan_sha256": plan["lock_sha256"],
        "fixed_runtime_manifest_sha256": manifest["manifest_sha256"],
        "selection_rule": spec["fixed_screen"]["strongest_screened_fixed_selection_rule"],
        "calibration_C0_policy": spec["fixed_screen"]["calibration_C0_policy"],
        "development_seeds": spec["fixed_screen"]["seeds"],
        "calibration_rounds": [2, 50],
        "strongest_screened_fixed_C_star_by_group": {"A": strongest["C_A"], "B": strongest["C_B"]},
        "strongest_screened_fixed_C_star_seed_evidence": strongest["seed_evidence"],
        "selected_calibration_C0_by_group": {"A": calibration["C_A"], "B": calibration["C_B"]},
        "selected_calibration_C0_strongest_screened_rank": rankings.index(calibration) + 1,
        "selected_calibration_C0_upstream_rank": calibration["upstream_candidate_rank"],
        "calibrated_profiles": calibration["N128_eligibility"]["profiles"],
        "ordered_fixed_candidates": rankings,
        "source_evidence": evidence,
        "artifact_sha256": artifact_hashes,
        "any_group_clipped_fraction_consumed_for_beta": False,
        "confirmation_data_accessed": False,
        "created_at_utc": created,
    })
    if existing is not None and existing != candidate_lock:
        raise RuntimeError("existing N128 fixed selection lock differs from completed evidence")
    for name, (values, columns) in artifacts.items():
        _write_or_verify_csv(root / name, values, columns, name)
    _write_or_verify(lock_path, candidate_lock, "N128 fixed selection lock")
    path = _ensure_development_manifest(root, spec_path, spec, plan, candidate_lock)
    print(json.dumps({
        "status": candidate_lock["status"],
        "strongest_screened_fixed_C_star": candidate_lock["strongest_screened_fixed_C_star_by_group"],
        "calibration_C0": candidate_lock["selected_calibration_C0_by_group"],
        "development_manifest": str(path),
    }, indent=2, sort_keys=True))


def _validate_fixed_lock(
    root: Path, plan: Mapping[str, Any], manifest: Mapping[str, Any],
    lock: Mapping[str, Any], spec: Mapping[str, Any],
) -> None:
    if (
        lock.get("status") != "N128_FIXED_CEILING_AND_CALIBRATION_LOCKED"
        or lock.get("campaign_plan_sha256") != plan["lock_sha256"]
        or lock.get("fixed_runtime_manifest_sha256") != manifest["manifest_sha256"]
        or lock.get("development_seeds") != spec["fixed_screen"]["seeds"]
        or lock.get("calibration_rounds") != [2, 50]
        or len(lock.get("ordered_fixed_candidates", [])) != 10
        or len(lock.get("calibrated_profiles", [])) != 5
        or lock.get("any_group_clipped_fraction_consumed_for_beta") is not False
        or lock.get("confirmation_data_accessed") is not False
    ):
        raise RuntimeError("N128 fixed lock identity differs")
    staged._verify_locked_evidence(root, manifest, lock["source_evidence"])
    for name, digest in lock.get("artifact_sha256", {}).items():
        path = root / str(name)
        if not path.is_file() or full.sha256_file(path) != digest:
            raise RuntimeError(f"N128 fixed lock artifact changed: {name}")
    if len({
        (float(row["C_A"]), float(row["C_B"]))
        for row in lock["ordered_fixed_candidates"]
    }) != 10:
        raise RuntimeError("N128 fixed candidate set deduplicated")
    for profile_index, profile in enumerate(lock["calibrated_profiles"], start=1):
        if profile.get("profile_index") != profile_index or set(profile.get("groups", {})) != set(GROUPS):
            raise RuntimeError("N128 calibrated profile identity differs")
        for group in GROUPS:
            record = profile["groups"][group]
            if (
                not bool(record.get("beta_identifiable"))
                or not bool(record.get("box_constraint_feasible"))
                or not 0.0 < float(record.get("beta", math.nan)) < 1.0
                or not math.isclose(
                    float(record["desired_hard_clipped_fraction"]),
                    float(record["weighted_hard_clipped_fraction"]),
                    rel_tol=0.0, abs_tol=1e-12,
                )
            ):
                raise RuntimeError("N128 calibrated profile beta semantics differ")


def _ensure_confirmation_manifest(
    root: Path, spec_path: Path, spec: Mapping[str, Any], plan: Mapping[str, Any],
    fixed_lock: Mapping[str, Any], development_lock: Mapping[str, Any],
) -> Path:
    arms = confirmation_arms(
        spec, str(plan["selected_domain"]), fixed_lock, development_lock
    )
    if len(arms) != EXPECTED_COUNTS[CONFIRMATION_STAGE]:
        raise RuntimeError("confirmation arm count differs")
    manifest = _runtime_manifest(
        spec=spec, spec_path=spec_path,
        repository_sha=str(plan["repository_sha"]),
        input_record=plan["selected_domain_input_manifest"],
        stage=CONFIRMATION_STAGE, arms=arms,
        created_at_utc=str(plan["created_at_utc"]),
        parent_lock_sha256=str(development_lock["lock_sha256"]),
        private_key_commitment=str(plan["private_key_commitment"]),
    )
    path = root / CONFIRMATION_MANIFEST_NAME
    _write_or_verify(path, manifest, "confirmation runtime manifest")
    return path


def _round_2_to_50_hard_rate_means(
    root: Path, arm: Mapping[str, Any]
) -> dict[str, float]:
    values = {group: [] for group in GROUPS}
    for round_index in range(2, 51):
        shard = full.load_object(
            _round_path(root, arm, round_index), "development hard-rate round shard"
        )
        summary = shard.get("round_summary")
        if not isinstance(summary, dict) or summary.get("round") != round_index:
            raise RuntimeError("development hard-rate shard identity differs")
        for group in GROUPS:
            values[group].append(float(summary[group]["clipped_fraction"]))
    return {group: statistics.fmean(values[group]) for group in GROUPS}


def development_target_feasibility(
    per_seed_observed: Sequence[Mapping[str, float]],
    desired: Mapping[str, float], tolerance: float,
) -> dict[str, Any]:
    if not per_seed_observed:
        raise ValueError("development target feasibility requires seed observations")
    seed_errors = [
        {group: float(observed[group]) - float(desired[group]) for group in GROUPS}
        for observed in per_seed_observed
    ]
    mean_observed = {
        group: statistics.fmean(float(observed[group]) for observed in per_seed_observed)
        for group in GROUPS
    }
    max_abs = {
        group: max(abs(float(error[group])) for error in seed_errors)
        for group in GROUPS
    }
    return {
        "per_seed_error_by_group": seed_errors,
        "mean_observed_by_group": mean_observed,
        "mean_error_by_group": {
            group: mean_observed[group] - float(desired[group]) for group in GROUPS
        },
        "max_abs_seed_error_by_group": max_abs,
        "feasible": all(max_abs[group] <= float(tolerance) for group in GROUPS),
    }


def lock_development(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, plan, _upstream_lock, fixed_manifest = _identity(
        root, spec_path, args.repository, args.expected_code_sha
    )
    fixed_lock = _load_lock(root / FIXED_LOCK_NAME, "N128 fixed selection lock")
    _validate_fixed_lock(root, plan, fixed_manifest, fixed_lock, spec)
    development_path = _ensure_development_manifest(
        root, spec_path, spec, plan, fixed_lock
    )
    manifest = full.load_runtime(development_path)
    lock_path = root / DEVELOPMENT_LOCK_NAME
    existing = _load_lock(lock_path, "profile development lock") if lock_path.is_file() else None
    rows = _metrics_for_manifest(root, manifest)
    arms_by_id = {str(arm["arm_id"]): arm for arm in manifest["arms"]}
    fixed_rows = {
        str(row["arm_id"]): row for row in _metrics_for_manifest(root, fixed_manifest)
    }
    candidates: list[dict[str, Any]] = []
    for profile in fixed_lock["calibrated_profiles"]:
        index = int(profile["profile_index"])
        subset = [row for row in rows if int(row["target_profile_index"]) == index]
        if len(subset) != 2 or {int(row["seed"]) for row in subset} != set(
            spec["profile_development"]["seeds"]
        ):
            raise RuntimeError(f"profile development coverage differs: p{index}")
        final_deltas: list[float] = []
        auc_deltas: list[float] = []
        instability = 0
        paired_seed_evidence: list[dict[str, Any]] = []
        desired = {
            group: float(profile["groups"][group]["desired_hard_clipped_fraction"])
            for group in GROUPS
        }
        seed_observations: list[dict[str, float]] = []
        for row in sorted(subset, key=lambda value: int(value["seed"])):
            reference = fixed_rows.get(str(row["reference_arm_id"]))
            if reference is None:
                raise RuntimeError("profile development fixed C* reference is missing")
            oracle._paired_identity(row, reference, f"development/p{index}/s{row['seed']}")
            final_delta = float(row["final_loss"]) - float(reference["final_loss"])
            auc_delta = float(row["normalized_loss_auc"]) - float(reference["normalized_loss_auc"])
            final_deltas.append(final_delta)
            auc_deltas.append(auc_delta)
            instability += staged._controller_instability_events(row)
            hard_means = _round_2_to_50_hard_rate_means(
                root, arms_by_id[str(row["arm_id"])]
            )
            seed_observations.append(hard_means)
            paired_seed_evidence.append({
                "seed": int(row["seed"]), "fixed_arm_id": reference["arm_id"],
                "slaclip_arm_id": row["arm_id"], "final_loss_delta": final_delta,
                "normalized_loss_auc_delta": auc_delta,
                "sample_schedule_sha256": row["sample_schedule_sha256"],
                "supervision_schedule_sha256": row["supervision_schedule_sha256"],
                "private_key_commitment": row["private_key_commitment"],
                "rng_domain": row["rng_domain"],
                "round_2_to_50_observed_hard_clipped_fraction_A": hard_means["A"],
                "round_2_to_50_observed_hard_clipped_fraction_B": hard_means["B"],
                "round_2_to_50_hard_target_error_A": hard_means["A"] - desired["A"],
                "round_2_to_50_hard_target_error_B": hard_means["B"] - desired["B"],
            })
        target_assessment = development_target_feasibility(
            seed_observations, desired,
            float(spec["profile_development"]["hard_target_absolute_tolerance"]),
        )
        candidates.append({
            "profile_index": index,
            "target_profile_position": profile["interpolation_position"],
            "desired_hard_clipped_fraction_by_group": desired,
            "beta_by_group": {group: profile["groups"][group]["beta"] for group in GROUPS},
            "mean_observed_hard_clipped_fraction_by_group": target_assessment["mean_observed_by_group"],
            "mean_hard_target_error_by_group": target_assessment["mean_error_by_group"],
            "max_abs_seed_hard_target_error_by_group": target_assessment["max_abs_seed_error_by_group"],
            "per_seed_hard_target_error_by_group": target_assessment["per_seed_error_by_group"],
            "target_feasible": target_assessment["feasible"], "seed_count": 2,
            "mean_paired_final_loss_delta": statistics.fmean(final_deltas),
            "mean_paired_normalized_loss_auc_delta": statistics.fmean(auc_deltas),
            "controller_instability_event_count": instability,
            "paired_seed_evidence": paired_seed_evidence,
        })
    candidates.sort(key=lambda row: (
        0 if row["target_feasible"] else 1,
        row["mean_paired_final_loss_delta"],
        row["mean_paired_normalized_loss_auc_delta"],
        row["controller_instability_event_count"], row["profile_index"],
    ))
    if not candidates[0]["target_feasible"]:
        raise RuntimeError("no Full SlaClip profile meets the preregistered development feasibility gate")
    selected_index = int(candidates[0]["profile_index"])
    selected_profile = next(
        profile for profile in fixed_lock["calibrated_profiles"]
        if int(profile["profile_index"]) == selected_index
    )
    evidence = staged._arm_evidence(root, manifest)
    paired_rows = [
        {"profile_index": candidate["profile_index"], **row}
        for candidate in candidates for row in candidate["paired_seed_evidence"]
    ]
    artifacts = {
        "profile_development_metrics.csv": (
            rows, _csv_columns(rows, ("stage", "domain", "model", "arm_id", "seed", "method", "target_profile_index"))
        ),
        "profile_development_paired_metrics.csv": (
            paired_rows, _csv_columns(paired_rows, ("profile_index", "seed", "fixed_arm_id", "slaclip_arm_id"))
        ),
    }
    artifact_hashes = {
        name: full.sha256_bytes(_csv_bytes(values, columns))
        for name, (values, columns) in artifacts.items()
    }
    created = str(existing["created_at_utc"]) if existing else full.utc_now()
    lock = _lock_payload({
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "FULL_SLACLIP_PROFILE_DEVELOPMENT_LOCKED",
        "campaign_name": spec["campaign_name"], "campaign_plan_sha256": plan["lock_sha256"],
        "fixed_selection_lock_sha256": fixed_lock["lock_sha256"],
        "development_runtime_manifest_sha256": manifest["manifest_sha256"],
        "selection_rule": spec["profile_development"]["selection_rule"],
        "development_seeds": spec["profile_development"]["seeds"],
        "eta": spec["profile_development"]["eta"],
        "selected_profile": selected_profile,
        "ordered_profile_candidates": candidates,
        "source_evidence": evidence, "fixed_C_star_source_evidence": fixed_lock["strongest_screened_fixed_C_star_seed_evidence"],
        "artifact_sha256": artifact_hashes,
        "confirmation_data_accessed": False, "created_at_utc": created,
    })
    if existing is not None and existing != lock:
        raise RuntimeError("existing profile development lock differs from completed evidence")
    for name, (values, columns) in artifacts.items():
        _write_or_verify_csv(root / name, values, columns, name)
    _write_or_verify(lock_path, lock, "profile development lock")
    confirmation_path = _ensure_confirmation_manifest(
        root, spec_path, spec, plan, fixed_lock, lock
    )
    print(json.dumps({
        "status": lock["status"], "selected_profile_index": selected_index,
        "confirmation_manifest": str(confirmation_path),
    }, indent=2, sort_keys=True))


def _validate_development_lock(
    root: Path, plan: Mapping[str, Any], fixed_lock: Mapping[str, Any],
    manifest: Mapping[str, Any], lock: Mapping[str, Any], spec: Mapping[str, Any],
) -> None:
    if (
        lock.get("status") != "FULL_SLACLIP_PROFILE_DEVELOPMENT_LOCKED"
        or lock.get("campaign_plan_sha256") != plan["lock_sha256"]
        or lock.get("fixed_selection_lock_sha256") != fixed_lock["lock_sha256"]
        or lock.get("development_runtime_manifest_sha256") != manifest["manifest_sha256"]
        or lock.get("development_seeds") != spec["profile_development"]["seeds"]
        or float(lock.get("eta", math.nan)) != 0.05
        or len(lock.get("ordered_profile_candidates", [])) != 5
        or not lock["ordered_profile_candidates"][0].get("target_feasible")
        or lock.get("confirmation_data_accessed") is not False
    ):
        raise RuntimeError("profile development lock identity differs")
    staged._verify_locked_evidence(root, manifest, lock["source_evidence"])
    for name, digest in lock.get("artifact_sha256", {}).items():
        path = root / str(name)
        if not path.is_file() or full.sha256_file(path) != digest:
            raise RuntimeError(f"profile development lock artifact changed: {name}")


def _paired_confirmation_rows(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_id = {str(row["arm_id"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for arm in manifest["arms"]:
        if arm["method"] != SLACLIP_METHOD:
            continue
        candidate = by_id.get(str(arm["arm_id"]))
        reference = by_id.get(str(arm["reference_arm_id"]))
        if candidate is None or reference is None:
            raise RuntimeError(f"confirmation pair is incomplete: {arm['arm_id']}")
        oracle._paired_identity(candidate, reference, f"confirmation/s{arm['seed']}")
        output.append({
            "domain": arm["domain"], "model": MODEL, "seed": int(arm["seed"]),
            "fixed_arm_id": reference["arm_id"], "slaclip_arm_id": candidate["arm_id"],
            "fixed_C_A": reference["initial_clip_norm_A"],
            "fixed_C_B": reference["initial_clip_norm_B"],
            "slaclip_initial_C_A": candidate["initial_clip_norm_A"],
            "slaclip_initial_C_B": candidate["initial_clip_norm_B"],
            "profile_index": candidate["target_profile_index"],
            "beta_A": candidate["slaclip_beta_A"], "beta_B": candidate["slaclip_beta_B"],
            "desired_hard_clipped_fraction_A": candidate["desired_hard_clipped_fraction_A"],
            "desired_hard_clipped_fraction_B": candidate["desired_hard_clipped_fraction_B"],
            "fixed_final_loss": reference["final_loss"],
            "slaclip_final_loss": candidate["final_loss"],
            "final_loss_delta_slaclip_minus_fixed": float(candidate["final_loss"]) - float(reference["final_loss"]),
            "fixed_normalized_loss_auc": reference["normalized_loss_auc"],
            "slaclip_normalized_loss_auc": candidate["normalized_loss_auc"],
            "normalized_loss_auc_delta_slaclip_minus_fixed": float(candidate["normalized_loss_auc"]) - float(reference["normalized_loss_auc"]),
            "fixed_final_token_accuracy": reference["final_token_accuracy"],
            "slaclip_final_token_accuracy": candidate["final_token_accuracy"],
            "final_token_accuracy_delta_slaclip_minus_fixed": float(candidate["final_token_accuracy"]) - float(reference["final_token_accuracy"]),
            "slaclip_actual_clipped_fraction_A": candidate["actual_clipped_fraction_A"],
            "slaclip_actual_clipped_fraction_B": candidate["actual_clipped_fraction_B"],
            "fixed_actual_clipped_fraction_A": reference["actual_clipped_fraction_A"],
            "fixed_actual_clipped_fraction_B": reference["actual_clipped_fraction_B"],
            "sample_schedule_sha256": candidate["sample_schedule_sha256"],
            "supervision_schedule_sha256": candidate["supervision_schedule_sha256"],
            "private_key_commitment": candidate["private_key_commitment"],
            "rng_domain": candidate["rng_domain"],
            **{
                f"lower_C_bound_hit_fraction_{group}": (
                    float(candidate[f"lower_bound_hits_{group}"]) / float(arm["rounds"])
                )
                for group in GROUPS
            },
            **{
                f"upper_C_bound_hit_fraction_{group}": (
                    float(candidate[f"upper_bound_hits_{group}"]) / float(arm["rounds"])
                )
                for group in GROUPS
            },
            **{
                f"dynamic_target_clamp_fraction_{group}": (
                    (
                        float(candidate[f"gamma_clamped_low_count_{group}"])
                        + float(candidate[f"gamma_clamped_high_count_{group}"])
                    ) / float(arm["rounds"])
                )
                for group in GROUPS
            },
            **{
                f"log_step_bounded_fraction_{group}": (
                    float(candidate[f"log_step_bounded_count_{group}"]) / float(arm["rounds"])
                )
                for group in GROUPS
            },
        })
    return sorted(output, key=lambda row: int(row["seed"]))


def _round_trajectory_rows(
    root: Path, manifests: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for manifest in manifests:
        for arm in manifest["arms"]:
            for round_index in range(1, int(arm["rounds"]) + 1):
                shard = full.load_object(_round_path(root, arm, round_index), "journal round trajectory")
                summary = shard.get("round_summary")
                if not isinstance(summary, dict) or summary.get("round") != round_index:
                    raise RuntimeError("journal round trajectory identity differs")
                validation = summary.get("validation")
                validation = validation if isinstance(validation, dict) else {}
                controller = summary.get("slaclip_controller")
                update = summary.get("federated_update")
                for group in GROUPS:
                    group_summary = summary[group]
                    control = controller.get(group) if isinstance(controller, dict) else None
                    update_group = update.get(group) if isinstance(update, dict) else {}
                    threshold = (
                        float(control["clip_threshold_used"])
                        if isinstance(control, dict)
                        else float(arm["initial_clip_norm_by_group"][group])
                    )
                    desired_map = arm.get("desired_hard_clipped_fraction_by_group")
                    desired_hard = (
                        float(desired_map[group]) if isinstance(desired_map, dict) else None
                    )
                    output.append({
                        "stage": arm["stage"], "domain": arm["domain"],
                        "model": MODEL, "arm_id": arm["arm_id"],
                        "method": arm["method"], "seed": arm["seed"],
                        "candidate_rank": arm.get("candidate_rank"),
                        "profile_index": arm.get("target_profile_index"),
                        "round": round_index, "group": group,
                        "mean_training_loss": summary.get("mean_training_loss"),
                        "internal_validation_loss": validation.get("loss"),
                        "internal_validation_supervised_token_accuracy": validation.get("token_accuracy"),
                        "internal_validation_supervised_tokens": validation.get("supervised_tokens"),
                        "hard_clipped_fraction": group_summary.get("clipped_fraction"),
                        "desired_hard_clipped_fraction": desired_hard,
                        "hard_minus_desired_target": (
                            float(group_summary["clipped_fraction"]) - desired_hard
                            if desired_hard is not None else None
                        ),
                        "would_clip_fraction": group_summary.get("would_clip_fraction"),
                        "mean_raw_norm": group_summary.get("mean_raw_norm"),
                        "max_raw_norm": group_summary.get("max_raw_norm"),
                        "mean_clip_factor": group_summary.get("mean_clip_factor"),
                        "clip_threshold_used": threshold,
                        "next_clip_threshold": control.get("next_clip_threshold") if isinstance(control, dict) else threshold,
                        "beta": control.get("base_target_clipped_fraction") if isinstance(control, dict) else None,
                        "near_threshold_proxy": control.get("near_threshold_proxy") if isinstance(control, dict) else None,
                        "near_zero_proxy": control.get("near_zero_proxy") if isinstance(control, dict) else None,
                        "remaining_non_small_gradient_fraction": control.get("remaining_non_small_gradient_fraction") if isinstance(control, dict) else None,
                        "raw_dynamic_target_clipped": control.get("raw_dynamic_target_clipped") if isinstance(control, dict) else None,
                        "dynamic_target_clipped": control.get("dynamic_target_clipped") if isinstance(control, dict) else None,
                        "actual_minus_dynamic_target": (
                            float(group_summary["clipped_fraction"]) - float(control["dynamic_target_clipped"])
                            if isinstance(control, dict) else None
                        ),
                        "noisy_cdf_endpoint_near_threshold": (
                            control.get("noisy_cdf_proxy_by_slot", [None])[0]
                            if isinstance(control, dict) else None
                        ),
                        "noisy_cdf_endpoint_near_zero": (
                            control.get("noisy_cdf_proxy_by_slot", [None])[-1]
                            if isinstance(control, dict) else None
                        ),
                        "exact_cdf_endpoint_near_threshold": (
                            control.get("exact_cdf_proxy_by_slot", [None])[0]
                            if isinstance(control, dict) else None
                        ),
                        "exact_cdf_endpoint_near_zero": (
                            control.get("exact_cdf_proxy_by_slot", [None])[-1]
                            if isinstance(control, dict) else None
                        ),
                        "noisy_near_threshold_minus_exact": control.get("noisy_near_threshold_minus_exact") if isinstance(control, dict) else None,
                        "noisy_near_zero_minus_exact": control.get("noisy_near_zero_minus_exact") if isinstance(control, dict) else None,
                        "cdf_error_mae": control.get("cdf_error_mae") if isinstance(control, dict) else None,
                        "cdf_error_rmse": control.get("cdf_error_rmse") if isinstance(control, dict) else None,
                        "cdf_error_z_rmse": control.get("cdf_error_z_rmse") if isinstance(control, dict) else None,
                        "cdf_error_max_abs": control.get("cdf_error_max_abs") if isinstance(control, dict) else None,
                        "normalized_proxy_noise_std_per_slot": control.get("normalized_proxy_noise_std_per_slot") if isinstance(control, dict) else None,
                        "theoretical_normalized_endpoint_noise_std": (
                            float(arm["noise_multiplier"]) * math.sqrt(
                                float(arm["slaclip_num_slots"]) / float(arm["num_clients"])
                            ) if isinstance(control, dict) else None
                        ),
                        "noisy_cdf_out_of_range_count": control.get("noisy_cdf_out_of_range_count") if isinstance(control, dict) else None,
                        "noisy_cdf_out_of_range_fraction": control.get("noisy_cdf_out_of_range_fraction") if isinstance(control, dict) else None,
                        "noisy_adjacent_monotonicity_violations": control.get("noisy_adjacent_monotonicity_violations") if isinstance(control, dict) else None,
                        "exact_adjacent_monotonicity_violations": control.get("exact_adjacent_monotonicity_violations") if isinstance(control, dict) else None,
                        "controller_error": control.get("controller_error") if isinstance(control, dict) else None,
                        "raw_log_step": control.get("raw_log_step") if isinstance(control, dict) else None,
                        "bounded_log_step": control.get("bounded_log_step") if isinstance(control, dict) else None,
                        "log_step_was_bounded": control.get("log_step_was_bounded") if isinstance(control, dict) else None,
                        "dynamic_target_was_clamped": (
                            bool(control.get("gamma_clamped_low") or control.get("gamma_clamped_high"))
                            if isinstance(control, dict) else None
                        ),
                        "hit_min_clip_threshold": control.get("hit_min_clip_norm") if isinstance(control, dict) else None,
                        "hit_max_clip_threshold": control.get("hit_max_clip_norm") if isinstance(control, dict) else None,
                        "oracle_dynamic_target_clipped": control.get("oracle_dynamic_target_clipped") if isinstance(control, dict) else None,
                        "oracle_next_clip_threshold": control.get("oracle_next_clip_threshold") if isinstance(control, dict) else None,
                        "noisy_minus_oracle_raw_log_step": control.get("noisy_minus_oracle_raw_log_step") if isinstance(control, dict) else None,
                        "noisy_oracle_log_threshold_error": control.get("noisy_oracle_log_threshold_error") if isinstance(control, dict) else None,
                        "noisy_oracle_update_direction_agrees": control.get("update_direction_agrees") if isinstance(control, dict) else None,
                        "exact_oracle_fields_privacy_label": (
                            "NON_DP_PRIVATE_DIAGNOSTIC" if isinstance(control, dict) else None
                        ),
                        "aggregate_signal_gradient_l2": update_group.get("aggregate_signal_gradient_l2"),
                        "aggregate_noise_gradient_l2": update_group.get("aggregate_noise_gradient_l2"),
                        "signal_to_noise_l2_ratio": update_group.get("signal_to_noise_l2_ratio"),
                        "signal_noise_cosine": update_group.get("signal_noise_cosine"),
                        "actual_global_update_l2": update_group.get("actual_global_update_l2"),
                        "relative_global_update": update_group.get("relative_global_update"),
                    })
    return output


def _completed_count(root: Path, manifest: Mapping[str, Any]) -> int:
    count = 0
    for arm in manifest["arms"]:
        try:
            staged._completed_summary(root, manifest, arm)
        except RuntimeError:
            continue
        count += 1
    return count


def confirmation_gates(
    final_inference: Mapping[str, Any], auc_inference: Mapping[str, Any],
    accuracy_inference: Mapping[str, Any], wins: int, spec: Mapping[str, Any],
) -> dict[str, bool]:
    if (
        final_inference.get("n") != 6
        or auc_inference.get("n") != 6
        or accuracy_inference.get("n") != 6
    ):
        raise RuntimeError("confirmation gate requires exactly six paired seeds")
    return {
        "mean_final_loss_delta_below_zero": float(final_inference["mean"]) < 0.0,
        "paired_two_sided_95pct_CI_upper_below_zero": float(final_inference["ci95_high"]) < 0.0,
        "exact_two_sided_magnitude_preserving_sign_flip_p_below_0p05": float(final_inference["exact_sign_flip_p"]) < float(spec["confirmation"]["alpha"]),
        "six_of_six_final_loss_wins": wins >= int(spec["confirmation"]["minimum_wins"]),
        "mean_normalized_loss_auc_delta_not_positive": float(auc_inference["mean"]) <= 0.0,
        "mean_final_token_accuracy_delta_not_negative": float(accuracy_inference["mean"]) >= 0.0,
    }


def confirmation_mechanism_validity(
    paired_rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> dict[str, Any]:
    if len(paired_rows) != 6:
        raise RuntimeError("mechanism validity requires exactly six paired seeds")
    policy = spec["confirmation"]
    groups: dict[str, Any] = {}
    for group in GROUPS:
        errors = [
            float(row[f"round_2_to_50_hard_target_error_{group}"])
            for row in paired_rows
        ]
        mean_abs = statistics.fmean(abs(value) for value in errors)
        passing = sum(
            abs(value) <= float(policy["mechanism_seed_abs_error_tolerance"])
            for value in errors
        )
        gates = {
            "across_seed_mean_absolute_error_at_most_0p15": (
                mean_abs <= float(policy["mechanism_mean_abs_error_tolerance"])
            ),
            "at_least_5_of_6_seeds_absolute_error_at_most_0p20": (
                passing >= int(policy["mechanism_minimum_passing_seeds_per_group"])
            ),
        }
        groups[group] = {
            "per_seed_errors": errors, "mean_absolute_error": mean_abs,
            "passing_seed_count": passing, "gates": gates,
            "passed": all(gates.values()),
        }
    return {
        "rounds": policy["mechanism_rounds"], "groups": groups,
        "passed": all(groups[group]["passed"] for group in GROUPS),
    }


def aggregate(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, plan, _upstream_lock, fixed_manifest = _identity(
        root, spec_path, args.repository, args.expected_code_sha
    )
    manifests: list[dict[str, Any]] = [fixed_manifest]
    stage_counts = {FIXED_STAGE: _completed_count(root, fixed_manifest), DEVELOPMENT_STAGE: 0, CONFIRMATION_STAGE: 0}
    fixed_lock = None
    development_lock = None
    if (root / FIXED_LOCK_NAME).is_file():
        fixed_lock = _load_lock(root / FIXED_LOCK_NAME, "N128 fixed selection lock")
        _validate_fixed_lock(root, plan, fixed_manifest, fixed_lock, spec)
        dev_manifest = full.load_runtime(_ensure_development_manifest(root, spec_path, spec, plan, fixed_lock))
        manifests.append(dev_manifest)
        stage_counts[DEVELOPMENT_STAGE] = _completed_count(root, dev_manifest)
        if (root / DEVELOPMENT_LOCK_NAME).is_file():
            development_lock = _load_lock(root / DEVELOPMENT_LOCK_NAME, "profile development lock")
            _validate_development_lock(root, plan, fixed_lock, dev_manifest, development_lock, spec)
            confirmation_manifest = full.load_runtime(_ensure_confirmation_manifest(
                root, spec_path, spec, plan, fixed_lock, development_lock
            ))
            manifests.append(confirmation_manifest)
            stage_counts[CONFIRMATION_STAGE] = _completed_count(root, confirmation_manifest)
    complete = (
        fixed_lock is not None and development_lock is not None
        and stage_counts == {
            FIXED_STAGE: EXPECTED_COUNTS[FIXED_STAGE],
            DEVELOPMENT_STAGE: EXPECTED_COUNTS[DEVELOPMENT_STAGE],
            CONFIRMATION_STAGE: EXPECTED_COUNTS[CONFIRMATION_STAGE],
        }
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED" if complete else "IN_PROGRESS",
        "campaign_name": spec["campaign_name"], "campaign_plan_sha256": plan["lock_sha256"],
        "selected_domain": plan["selected_domain"], "model": MODEL,
        "expected_stage_arm_counts": spec["expected_stage_arm_counts"],
        "completed_stage_arm_counts": stage_counts,
        "completed_total_arm_count": sum(stage_counts.values()),
        "mechanism_lock": plan["mechanism_lock"],
        "updated_at_utc": full.utc_now(),
    }
    if complete:
        confirmation_manifest = manifests[-1]
        all_metrics = [row for manifest in manifests for row in _metrics_for_manifest(root, manifest)]
        if len(all_metrics) != EXPECTED_COUNTS["total"]:
            raise RuntimeError("strict aggregate metric arm count differs")
        paired = _paired_confirmation_rows(
            [row for row in all_metrics if row["stage"] == CONFIRMATION_STAGE],
            confirmation_manifest,
        )
        if len(paired) != 6 or {int(row["seed"]) for row in paired} != set(spec["confirmation"]["seeds"]):
            raise RuntimeError("confirmation paired seed coverage differs")
        desired_confirmation = {
            group: float(development_lock["selected_profile"]["groups"][group]["desired_hard_clipped_fraction"])
            for group in GROUPS
        }
        confirmation_arm_by_id = {
            str(arm["arm_id"]): arm for arm in confirmation_manifest["arms"]
        }
        for row in paired:
            hard_means = _round_2_to_50_hard_rate_means(
                root, confirmation_arm_by_id[str(row["slaclip_arm_id"])]
            )
            for group in GROUPS:
                row[f"round_2_to_50_observed_hard_clipped_fraction_{group}"] = hard_means[group]
                row[f"round_2_to_50_hard_target_error_{group}"] = (
                    hard_means[group] - desired_confirmation[group]
                )
                row[f"round_2_to_50_absolute_hard_target_error_{group}"] = abs(
                    hard_means[group] - desired_confirmation[group]
                )
        mechanism_validity = confirmation_mechanism_validity(paired, spec)
        final_inference = full.paired_inference([
            float(row["final_loss_delta_slaclip_minus_fixed"]) for row in paired
        ])
        auc_inference = full.paired_inference([
            float(row["normalized_loss_auc_delta_slaclip_minus_fixed"]) for row in paired
        ])
        accuracy_inference = full.paired_inference([
            float(row["final_token_accuracy_delta_slaclip_minus_fixed"]) for row in paired
        ])
        if final_inference["n"] != 6 or auc_inference["n"] != 6 or accuracy_inference["n"] != 6:
            raise RuntimeError("confirmation inference dropped paired seeds")
        wins = sum(float(row["final_loss_delta_slaclip_minus_fixed"]) < 0.0 for row in paired)
        gates = confirmation_gates(
            final_inference, auc_inference, accuracy_inference, wins, spec
        )
        confirmation_rows = [
            row for row in all_metrics if row["stage"] == CONFIRMATION_STAGE
        ]
        fixed_confirmation = [
            row for row in confirmation_rows if row["method"] == FIXED_METHOD
        ]
        slaclip_confirmation = [
            row for row in confirmation_rows if row["method"] == SLACLIP_METHOD
        ]
        if len(fixed_confirmation) != 6 or len(slaclip_confirmation) != 6:
            raise RuntimeError("confirmation per-method coverage differs")
        method_means = {
            "fixed": {
                "final_loss": statistics.fmean(float(row["final_loss"]) for row in fixed_confirmation),
                "normalized_loss_auc": statistics.fmean(float(row["normalized_loss_auc"]) for row in fixed_confirmation),
                "final_token_accuracy": statistics.fmean(float(row["final_token_accuracy"]) for row in fixed_confirmation),
            },
            "slaclip": {
                "final_loss": statistics.fmean(float(row["final_loss"]) for row in slaclip_confirmation),
                "normalized_loss_auc": statistics.fmean(float(row["normalized_loss_auc"]) for row in slaclip_confirmation),
                "final_token_accuracy": statistics.fmean(float(row["final_token_accuracy"]) for row in slaclip_confirmation),
            },
        }
        observed_confirmation = {
            group: statistics.fmean(
                float(row[f"round_2_to_50_observed_hard_clipped_fraction_{group}"])
                for row in paired
            ) for group in GROUPS
        }
        final_threshold_summary = {
            group: {
                "mean": statistics.fmean(
                    float(row[f"final_threshold_{group}"]) for row in slaclip_confirmation
                ),
                "sample_std": statistics.stdev(
                    float(row[f"final_threshold_{group}"]) for row in slaclip_confirmation
                ),
                "mean_log_threshold_total_variation": statistics.fmean(
                    float(row[f"log_threshold_total_variation_{group}"])
                    for row in slaclip_confirmation
                ),
            } for group in GROUPS
        }
        controller_diagnostics = {
            group: {
                "mean_lower_C_bound_hit_fraction": statistics.fmean(
                    float(row[f"lower_bound_hits_{group}"]) / float(spec["common"]["rounds"])
                    for row in slaclip_confirmation
                ),
                "mean_upper_C_bound_hit_fraction": statistics.fmean(
                    float(row[f"upper_bound_hits_{group}"]) / float(spec["common"]["rounds"])
                    for row in slaclip_confirmation
                ),
                "mean_dynamic_target_clamp_fraction": statistics.fmean(
                    (
                        float(row[f"gamma_clamped_low_count_{group}"])
                        + float(row[f"gamma_clamped_high_count_{group}"])
                    ) / float(spec["common"]["rounds"])
                    for row in slaclip_confirmation
                ),
                "mean_log_step_bounded_fraction": statistics.fmean(
                    float(row[f"log_step_bounded_count_{group}"]) / float(spec["common"]["rounds"])
                    for row in slaclip_confirmation
                ),
            } for group in GROUPS
        }
        utility_success = all(gates.values())
        overall_positive = utility_success and bool(mechanism_validity["passed"])
        trajectory = _round_trajectory_rows(root, manifests)
        artifacts = {
            "campaign_metrics.csv": (
                all_metrics, _csv_columns(all_metrics, ("stage", "domain", "model", "arm_id", "seed", "method"))
            ),
            "confirmation_paired_metrics.csv": (
                paired, _csv_columns(paired, ("domain", "model", "seed", "fixed_arm_id", "slaclip_arm_id"))
            ),
            "round_trajectory.csv": (
                trajectory, _csv_columns(trajectory, ("stage", "domain", "model", "arm_id", "method", "seed", "round", "group"))
            ),
        }
        aggregate_row = {
            "Domain": plan["selected_domain"], "Model": "GPT-2",
            "Clients": 128, "K": 5, "Noise Multiplier": 2.0,
            "Fixed C_A": fixed_lock["strongest_screened_fixed_C_star_by_group"]["A"],
            "Fixed C_B": fixed_lock["strongest_screened_fixed_C_star_by_group"]["B"],
            "SlaClip Initial C_A": fixed_lock["selected_calibration_C0_by_group"]["A"],
            "SlaClip Initial C_B": fixed_lock["selected_calibration_C0_by_group"]["B"],
            "Selected Profile": development_lock["selected_profile"]["profile_index"],
            "Beta A": development_lock["selected_profile"]["groups"]["A"]["beta"],
            "Beta B": development_lock["selected_profile"]["groups"]["B"]["beta"],
            "Seeds": 6, "Mean Final Loss Delta": final_inference["mean"],
            "Final Loss Delta CI95 Low": final_inference["ci95_low"],
            "Final Loss Delta CI95 High": final_inference["ci95_high"],
            "Exact Sign-Flip P": final_inference["exact_sign_flip_p"],
            "Wins": wins, "Mean Loss AUC Delta": auc_inference["mean"],
            "Token Accuracy Delta Mean": accuracy_inference["mean"],
            "Token Accuracy Delta CI95 Low": accuracy_inference["ci95_low"],
            "Token Accuracy Delta CI95 High": accuracy_inference["ci95_high"],
            "Token Accuracy Exact Sign-Flip P": accuracy_inference["exact_sign_flip_p"],
            "Fixed Mean Final Loss": method_means["fixed"]["final_loss"],
            "SlaClip Mean Final Loss": method_means["slaclip"]["final_loss"],
            "Fixed Mean Loss AUC": method_means["fixed"]["normalized_loss_auc"],
            "SlaClip Mean Loss AUC": method_means["slaclip"]["normalized_loss_auc"],
            "Fixed Mean Token Accuracy": method_means["fixed"]["final_token_accuracy"],
            "SlaClip Mean Token Accuracy": method_means["slaclip"]["final_token_accuracy"],
            "Desired Hard Clip A": desired_confirmation["A"],
            "Desired Hard Clip B": desired_confirmation["B"],
            "R2-50 Observed Hard Clip A": observed_confirmation["A"],
            "R2-50 Observed Hard Clip B": observed_confirmation["B"],
            "R2-50 Observed Minus Desired A": observed_confirmation["A"] - desired_confirmation["A"],
            "R2-50 Observed Minus Desired B": observed_confirmation["B"] - desired_confirmation["B"],
            "Mean Final C A": final_threshold_summary["A"]["mean"],
            "Mean Final C B": final_threshold_summary["B"]["mean"],
            "Final C Sample Std A": final_threshold_summary["A"]["sample_std"],
            "Final C Sample Std B": final_threshold_summary["B"]["sample_std"],
            "Mean Log C Total Variation A": final_threshold_summary["A"]["mean_log_threshold_total_variation"],
            "Mean Log C Total Variation B": final_threshold_summary["B"]["mean_log_threshold_total_variation"],
            "Mechanism Mean Absolute Error A": mechanism_validity["groups"]["A"]["mean_absolute_error"],
            "Mechanism Mean Absolute Error B": mechanism_validity["groups"]["B"]["mean_absolute_error"],
            "Mechanism Passing Seeds A": mechanism_validity["groups"]["A"]["passing_seed_count"],
            "Mechanism Passing Seeds B": mechanism_validity["groups"]["B"]["passing_seed_count"],
            "Mean Lower C Bound Hit Fraction A": controller_diagnostics["A"]["mean_lower_C_bound_hit_fraction"],
            "Mean Lower C Bound Hit Fraction B": controller_diagnostics["B"]["mean_lower_C_bound_hit_fraction"],
            "Mean Upper C Bound Hit Fraction A": controller_diagnostics["A"]["mean_upper_C_bound_hit_fraction"],
            "Mean Upper C Bound Hit Fraction B": controller_diagnostics["B"]["mean_upper_C_bound_hit_fraction"],
            "Mean Target Clamp Fraction A": controller_diagnostics["A"]["mean_dynamic_target_clamp_fraction"],
            "Mean Target Clamp Fraction B": controller_diagnostics["B"]["mean_dynamic_target_clamp_fraction"],
            "Utility Success Gate Passed": utility_success,
            "Mechanism Validity Gate Passed": mechanism_validity["passed"],
            "Overall Positive Claim": overall_positive,
        }
        artifacts["journal_main_table.csv"] = ([aggregate_row], tuple(aggregate_row))
        artifact_hashes: dict[str, str] = {}
        for name, (values, columns) in artifacts.items():
            artifact_hashes[name] = _write_or_verify_csv(root / name, values, columns, name)
        gate_path = root / GATE_LOCK_NAME
        existing_gate = _load_lock(gate_path, "confirmation gate lock") if gate_path.is_file() else None
        created = str(existing_gate["created_at_utc"]) if existing_gate else full.utc_now()
        gate_lock = _lock_payload({
            "schema_version": LOCK_SCHEMA_VERSION,
            "status": "INITIAL_CONFIRMATION_EVALUATED",
            "campaign_name": spec["campaign_name"],
            "campaign_plan_sha256": plan["lock_sha256"],
            "fixed_selection_lock_sha256": fixed_lock["lock_sha256"],
            "development_selection_lock_sha256": development_lock["lock_sha256"],
            "confirmation_runtime_manifest_sha256": confirmation_manifest["manifest_sha256"],
            "confirmation_seeds": spec["confirmation"]["seeds"],
            "paired_final_loss_inference": final_inference,
            "paired_normalized_loss_auc_inference": auc_inference,
            "paired_final_token_accuracy_inference": accuracy_inference,
            "confirmation_method_means": method_means,
            "slaclip_desired_hard_clipped_fraction_by_group": desired_confirmation,
            "slaclip_round_2_to_50_observed_hard_clipped_fraction_by_group": observed_confirmation,
            "slaclip_round_2_to_50_observed_minus_desired_by_group": {
                group: observed_confirmation[group] - desired_confirmation[group]
                for group in GROUPS
            },
            "slaclip_final_threshold_summary_by_group": final_threshold_summary,
            "slaclip_controller_descriptive_diagnostics_by_group": controller_diagnostics,
            "final_loss_win_count": wins,
            "utility_success_gate": {"gates": gates, "passed": utility_success},
            "mechanism_validity_gate": mechanism_validity,
            "overall_positive_claim": overall_positive,
            "mean_hard_rate_target_achievement_supported": bool(
                mechanism_validity["passed"]
            ),
            "within_seed_temporal_tracking_stability_evidence_is_descriptive": True,
            "cdf_reconstruction_evidence_is_descriptive": True,
            "cdf_reconstruction_accuracy_preregistered_gate": False,
            "cdf_reconstruction_exact_reference_privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
            "attribution_statement": (
                "utility_result_may_be_interpreted_with_mean_hard_rate_target_achievement_support"
                if mechanism_validity["passed"]
                else "mechanism_gate_failed_so_utility_result_cannot_be_attributed_to_mean_hard_rate_target_achievement"
            ),
            "journal_grade_positive_claim_requires_future_seed_expansion": True,
            "recommended_minimum_total_confirmation_seeds": 10,
            "preferred_total_confirmation_seeds": 20,
            "source_evidence": staged._arm_evidence(root, confirmation_manifest),
            "artifact_sha256": artifact_hashes, "created_at_utc": created,
        })
        if existing_gate is not None and existing_gate != gate_lock:
            raise RuntimeError("existing confirmation gate lock differs from evidence")
        _write_or_verify(gate_path, gate_lock, "confirmation gate lock")
        summary.update({
            "utility_success_gate_passed": gate_lock["utility_success_gate"]["passed"],
            "mechanism_validity_gate_passed": gate_lock["mechanism_validity_gate"]["passed"],
            "overall_positive_claim": gate_lock["overall_positive_claim"],
            "mean_hard_rate_target_achievement_supported": gate_lock[
                "mean_hard_rate_target_achievement_supported"
            ],
            "within_seed_temporal_tracking_stability_evidence_is_descriptive": True,
            "cdf_reconstruction_evidence_is_descriptive": True,
            "cdf_reconstruction_accuracy_preregistered_gate": False,
            "attribution_statement": gate_lock["attribution_statement"],
            "confirmation_gate_lock_sha256": gate_lock["lock_sha256"],
            "journal_grade_positive_claim_requires_future_seed_expansion": True,
        })
    full.atomic_json(root / SUMMARY_NAME, summary)
    if args.require_complete and not complete:
        raise RuntimeError("identified N128 campaign is incomplete")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_arm(args: argparse.Namespace) -> int:
    runtime = full.load_runtime(args.manifest.resolve())
    if full.sha256_file(args.private_key.resolve()) != runtime.get("private_key_commitment"):
        raise RuntimeError("runtime private RNG key commitment changed before arm execution")
    input_path = Path(str(runtime["input_manifest_path"])).resolve()
    if not input_path.is_file() or full.sha256_file(input_path) != runtime["input_manifest_sha256"]:
        raise RuntimeError("runtime input manifest SHA-256 changed before arm execution")
    input_value = full.load_object(input_path, "runtime input manifest")
    if input_value.get("inventory_sha256") != runtime.get("input_inventory_sha256"):
        raise RuntimeError("runtime input inventory changed before arm execution")
    return full.run_arm(argparse.Namespace(
        manifest=args.manifest, repository=args.repository,
        python_bin=args.python_bin, private_key=args.private_key,
        arm_index=args.arm_index,
    ))


def validate_resume_state(
    plan_path: Path, private_key: Path, expected_code_sha: str,
    expected_spec_sha: str,
) -> dict[str, Any]:
    plan = _load_lock(plan_path.resolve(), "resume campaign plan")
    if (
        plan.get("status") != "IDENTIFIED_N128_CAMPAIGN_PREREGISTERED"
        or plan.get("repository_sha") != expected_code_sha
        or plan.get("spec_sha256") != expected_spec_sha
    ):
        raise RuntimeError("resume plan repository/spec identity differs")
    key_path = private_key.resolve()
    if not key_path.is_file() or full.sha256_file(key_path) != plan.get("private_key_commitment"):
        raise RuntimeError("resume private RNG key commitment differs")
    return plan


def run_smoke(args: argparse.Namespace) -> int:
    manifest = full.load_runtime(args.manifest.resolve())
    repository = args.repository.resolve()
    if full.repository_sha(repository) != manifest["repository_sha"] or full.repository_dirty(repository):
        raise RuntimeError("repository snapshot changed before smoke")
    if full.sha256_file(args.private_key.resolve()) != manifest.get("private_key_commitment"):
        raise RuntimeError("runtime private RNG key commitment changed before smoke")
    input_path = Path(str(manifest["input_manifest_path"])).resolve()
    if (
        not input_path.is_file()
        or full.sha256_file(input_path) != manifest["input_manifest_sha256"]
        or full.load_object(input_path, "smoke input manifest").get("inventory_sha256")
        != manifest.get("input_inventory_sha256")
    ):
        raise RuntimeError("runtime input manifest/inventory changed before smoke")
    if args.arm_index < 0 or args.arm_index >= len(manifest["arms"]):
        raise ValueError("smoke arm index is outside the manifest")
    arm = manifest["arms"][args.arm_index]
    label = str(args.label)
    if not label or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in label):
        raise ValueError("unsafe smoke label")
    output_dir = args.campaign_root.resolve() / "preflight" / label
    stop_file = args.campaign_root.resolve() / "control" / f"{label}.stop"
    final_summary = output_dir / "final_summary.json"
    if final_summary.is_file():
        summary = full.load_object(final_summary, "smoke final summary")
        if summary.get("status") == "COMPLETED" and summary.get("method") == arm["method"]:
            print(f"smoke_reused={label}")
            return 0
    command = full._arm_command(
        arm, repository=repository, python_bin=args.python_bin.resolve(),
        input_manifest=Path(str(manifest["input_manifest_path"])),
        output_dir=output_dir, private_key=args.private_key.resolve(),
        stop_file=stop_file,
    )
    command.append("--smoke")
    if any("quantile" in part.lower() for part in command):
        raise RuntimeError("identified campaign command contains an excluded controller")
    completed = subprocess.run(command, cwd=repository, check=False)
    return int(completed.returncode)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-spec")
    validate.add_argument("--spec", type=Path, required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--spec", type=Path, required=True)
    prepare_parser.add_argument("--repository", type=Path, required=True)
    prepare_parser.add_argument("--expected-code-sha", required=True)
    prepare_parser.add_argument("--upstream-root", type=Path, required=True)
    prepare_parser.add_argument("--upstream-repository", type=Path, required=True)
    prepare_parser.add_argument("--upstream-spec", type=Path, required=True)
    prepare_parser.add_argument("--campaign-root", type=Path, required=True)
    prepare_parser.add_argument("--private-key", type=Path, required=True)
    prepare_parser.add_argument("--resume", action="store_true")

    for name in ("lock-fixed", "lock-development", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--expected-code-sha", required=True)
        command.add_argument("--campaign-root", type=Path, required=True)
        if name == "aggregate":
            command.add_argument("--require-complete", action="store_true")

    for name in ("run-arm", "run-smoke"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--python-bin", type=Path, required=True)
        command.add_argument("--private-key", type=Path, required=True)
        command.add_argument("--arm-index", type=int, required=True)
        if name == "run-smoke":
            command.add_argument("--campaign-root", type=Path, required=True)
            command.add_argument("--label", required=True)

    indices = commands.add_parser("print-arm-indices")
    indices.add_argument("--manifest", type=Path, required=True)
    resume_state = commands.add_parser("validate-resume-state")
    resume_state.add_argument("--plan", type=Path, required=True)
    resume_state.add_argument("--private-key", type=Path, required=True)
    resume_state.add_argument("--expected-code-sha", required=True)
    resume_state.add_argument("--expected-spec-sha", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    if args.command == "validate-spec":
        spec = load_spec(args.spec)
        print(json.dumps({
            "status": "VALID", "expected_stage_arm_counts": spec["expected_stage_arm_counts"],
            "model": MODEL, "N": 128, "K": 5,
            "automatic_num_slots_upper_bound": automatic_num_slots(128, 2.0),
            "theoretical_normalized_endpoint_noise_std": 2.0 * math.sqrt(5.0 / 128.0),
        }, indent=2, sort_keys=True))
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "lock-fixed":
        lock_fixed(args)
    elif args.command == "lock-development":
        lock_development(args)
    elif args.command == "aggregate":
        aggregate(args)
    elif args.command == "run-arm":
        raise SystemExit(run_arm(args))
    elif args.command == "run-smoke":
        raise SystemExit(run_smoke(args))
    elif args.command == "print-arm-indices":
        runtime = full.load_runtime(args.manifest.resolve())
        for index in range(len(runtime["arms"])):
            print(index)
    elif args.command == "validate-resume-state":
        plan = validate_resume_state(
            args.plan, args.private_key, args.expected_code_sha,
            args.expected_spec_sha,
        )
        print(json.dumps({
            "status": "VALID", "campaign_plan_sha256": plan["lock_sha256"],
            "private_key_commitment": plan["private_key_commitment"],
        }, indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
