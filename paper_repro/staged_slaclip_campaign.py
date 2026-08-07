#!/usr/bin/env python3
"""Coordinate one staged, single-allocation fixed-C versus full-SlaClip study.

The campaign has three immutable scientific stages:

1. tune a strong fixed clipping threshold on development seeds;
2. derive five full-SlaClip ``beta`` values from the selected fixed-C
   clipping trajectory and tune ``(C0, beta)`` on the same development seeds;
3. freeze one configuration per model before evaluating either method on
   disjoint confirmation seeds.

Stage manifests are materialized only after the preceding selection lock has
been durably published.  A resumed allocation verifies an existing lock and
never re-runs its selection rule.  This file delegates model execution to the
already tested ``full_slaclip_campaign.run_arm`` path; it does not implement a
new trainer or a SlaClip-Q controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from paper_repro import full_slaclip_campaign as full
    from paper_repro.slaclip import (
        build_slack_vector,
        normalize_noisy_slack,
        stationary_beta_from_exact_endpoints,
    )
except ModuleNotFoundError:  # Support direct ``python paper_repro/...py`` use.
    import full_slaclip_campaign as full  # type: ignore[no-redef]
    from slaclip import (  # type: ignore[no-redef]
        build_slack_vector,
        normalize_noisy_slack,
        stationary_beta_from_exact_endpoints,
    )


SPEC_SCHEMA_VERSION = 1
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
EXPECTED_MODELS = ("bert", "gpt2")
FIXED_METHOD = full.FIXED_DP_METHOD
SLACLIP_METHOD = full.FULL_SLACLIP_METHOD
NOISY_CONTROLLER_INPUT = full.NOISY_CONTROLLER_INPUT


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{label} keys differ; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _integer(value: Any, label: str, *, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{label} has an invalid sign")
    return value


def _unique(values: Sequence[Any], label: str) -> None:
    encoded = [full.canonical_bytes(value) for value in values]
    if len(encoded) != len(set(encoded)):
        raise ValueError(f"{label} contains duplicates")


def load_spec(path: Path) -> dict[str, Any]:
    value = full.load_object(path, "staged campaign specification")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "campaign_name",
            "description",
            "expected_stage_arm_counts",
            "common",
            "fixed_development",
            "slaclip_development",
            "confirmation",
            "scientific_boundary",
        },
        "campaign specification",
    )
    if value["schema_version"] != SPEC_SCHEMA_VERSION:
        raise ValueError("unsupported staged campaign specification schema")
    if not isinstance(value["campaign_name"], str) or not value["campaign_name"]:
        raise ValueError("campaign_name is invalid")
    if not isinstance(value["description"], str) or not value["description"]:
        raise ValueError("description is invalid")

    common = value["common"]
    if not isinstance(common, dict):
        raise ValueError("common must be an object")
    _require_exact_keys(
        common,
        {
            "models",
            "num_clients",
            "rounds",
            "batch_size",
            "noise_multiplier",
            "learning_rate",
            "rank",
            "max_seq_length",
            "max_validation_records",
            "eval_every",
            "checkpoint_every",
            "data_split_seed",
            "evaluation_seed",
            "delta",
            "slaclip_num_slots",
            "slaclip_c_min",
            "slaclip_c_max",
            "slaclip_eta",
            "slaclip_endpoint_epsilon",
        },
        "common",
    )
    if tuple(common["models"]) != EXPECTED_MODELS:
        raise ValueError("models must be bert then gpt2")
    for name in (
        "num_clients",
        "rounds",
        "batch_size",
        "rank",
        "max_seq_length",
        "max_validation_records",
        "eval_every",
        "checkpoint_every",
        "slaclip_num_slots",
    ):
        _integer(common[name], f"common.{name}")
    for name in ("data_split_seed", "evaluation_seed"):
        _integer(common[name], f"common.{name}", positive=False)
    for name in (
        "noise_multiplier",
        "learning_rate",
        "delta",
        "slaclip_c_min",
        "slaclip_c_max",
        "slaclip_endpoint_epsilon",
    ):
        _number(common[name], f"common.{name}", positive=True)
    _number(common["slaclip_eta"], "common.slaclip_eta", positive=True)
    if common["slaclip_num_slots"] != 5:
        raise ValueError("this campaign is full SlaClip with K=5")
    if float(common["slaclip_endpoint_epsilon"]) != 1e-6:
        raise ValueError("full SlaClip endpoint epsilon must be exactly 1e-6")
    if common["checkpoint_every"] != 25 or common["eval_every"] != 10:
        raise ValueError("campaign requires checkpoint_every=25 and eval_every=10")
    if common["slaclip_c_max"] < common["slaclip_c_min"]:
        raise ValueError("SlaClip threshold bounds are reversed")

    fixed = value["fixed_development"]
    if not isinstance(fixed, dict):
        raise ValueError("fixed_development must be an object")
    _require_exact_keys(
        fixed,
        {"method", "clip_norm_grid", "seeds", "selection_rule"},
        "fixed_development",
    )
    if fixed["method"] != FIXED_METHOD:
        raise ValueError("fixed development must use paper_dp_lora")
    grid = [_number(item, "fixed C", positive=True) for item in fixed["clip_norm_grid"]]
    seeds = [_integer(item, "development seed", positive=False) for item in fixed["seeds"]]
    _unique(grid, "fixed C grid")
    _unique(seeds, "development seeds")
    if grid != sorted(grid) or len(grid) < 3 or 10.0 not in grid:
        raise ValueError("fixed C grid must be sorted, contain C=10, and have neighbours")
    if any(not common["slaclip_c_min"] <= item <= common["slaclip_c_max"] for item in grid):
        raise ValueError("fixed C grid lies outside the public threshold bounds")
    if fixed["selection_rule"] != [
        "lowest_mean_final_internal_validation_loss",
        "lowest_mean_normalized_internal_validation_loss_auc",
        "lowest_final_loss_sample_std",
        "smaller_fixed_C",
    ]:
        raise ValueError("fixed-C selection rule differs from the preregistration")

    adaptive = value["slaclip_development"]
    if not isinstance(adaptive, dict):
        raise ValueError("slaclip_development must be an object")
    _require_exact_keys(
        adaptive,
        {
            "method",
            "seeds",
            "initial_C_policy",
            "beta_derivation",
            "selection_rule",
        },
        "slaclip_development",
    )
    if adaptive["method"] != SLACLIP_METHOD or adaptive["seeds"] != fixed["seeds"]:
        raise ValueError("SlaClip development must reuse the fixed development seeds")
    if adaptive["initial_C_policy"] != "selected_fixed_C_and_immediate_grid_neighbours":
        raise ValueError("unexpected SlaClip initial-C policy")
    beta = adaptive["beta_derivation"]
    if not isinstance(beta, dict):
        raise ValueError("beta_derivation must be an object")
    _require_exact_keys(
        beta,
        {
            "calibration_group",
            "actual_clipped_fraction_field",
            "exact_near_zero_formula",
            "conditional_beta_formula",
            "quantile_interval",
            "num_points",
            "degenerate_interval_policy",
        },
        "beta_derivation",
    )
    if (
        beta["calibration_group"] != "B"
        or beta["actual_clipped_fraction_field"] != "B.clipped_fraction"
        or beta["exact_near_zero_formula"]
        != "z=exact_normalized_slack_endpoint_K/(C+epsilon)"
        or beta["conditional_beta_formula"]
        != "beta_stationary_t=(1-q_exact_t)/(1-z_exact_t)"
        or beta["quantile_interval"] != [0.1, 0.9]
        or beta["num_points"] != 5
        or beta["degenerate_interval_policy"] != "fail_closed"
    ):
        raise ValueError("beta derivation differs from the preregistration")
    if adaptive["selection_rule"] != [
        "lowest_mean_paired_final_loss_delta_vs_selected_fixed_C",
        "lowest_mean_paired_normalized_loss_auc_delta",
        "fewest_controller_instability_events",
        "smaller_beta",
        "smaller_initial_C",
    ]:
        raise ValueError("SlaClip selection rule differs from the preregistration")

    confirmation = value["confirmation"]
    if not isinstance(confirmation, dict):
        raise ValueError("confirmation must be an object")
    _require_exact_keys(
        confirmation,
        {"methods", "seeds", "primary_metric", "multiplicity", "success_rule"},
        "confirmation",
    )
    if confirmation["methods"] != [FIXED_METHOD, SLACLIP_METHOD]:
        raise ValueError("confirmation methods/order is invalid")
    confirmation_seeds = [
        _integer(item, "confirmation seed", positive=False)
        for item in confirmation["seeds"]
    ]
    _unique(confirmation_seeds, "confirmation seeds")
    if set(confirmation_seeds) & set(seeds):
        raise ValueError("development and confirmation seeds must be disjoint")
    if len(confirmation_seeds) != 20:
        raise ValueError("confirmation requires exactly twenty independent seeds")
    if (
        confirmation["primary_metric"] != "final_internal_validation_loss"
        or confirmation["multiplicity"] != "Holm_across_two_models"
    ):
        raise ValueError("confirmation inference contract differs")
    if not isinstance(confirmation["success_rule"], dict):
        raise ValueError("confirmation.success_rule must be an object")
    success_rule = confirmation["success_rule"]
    _require_exact_keys(
        success_rule,
        {
            "mean_SlaClip_minus_fixed_below_zero",
            "paired_two_sided_95pct_CI_upper_below_zero",
            "holm_alpha",
            "normalized_loss_auc_not_worse",
            "minimum_seed_win_fraction",
        },
        "confirmation.success_rule",
    )
    if (
        success_rule["mean_SlaClip_minus_fixed_below_zero"] is not True
        or success_rule["paired_two_sided_95pct_CI_upper_below_zero"] is not True
        or _number(success_rule["holm_alpha"], "confirmation Holm alpha", positive=True)
        != 0.05
        or success_rule["normalized_loss_auc_not_worse"] is not True
        or _number(
            success_rule["minimum_seed_win_fraction"],
            "minimum seed win fraction",
            positive=True,
        )
        != 0.7
    ):
        raise ValueError("confirmation success rule differs from the preregistration")

    counts = value["expected_stage_arm_counts"]
    if not isinstance(counts, dict):
        raise ValueError("expected_stage_arm_counts must be an object")
    _require_exact_keys(counts, {FIXED_STAGE, SLACLIP_STAGE, CONFIRMATION_STAGE, "total"}, "counts")
    expected_fixed = len(grid) * len(seeds) * len(EXPECTED_MODELS)
    expected_slaclip = 3 * 5 * len(seeds) * len(EXPECTED_MODELS)
    expected_confirmation = 2 * len(confirmation_seeds) * len(EXPECTED_MODELS)
    expected = {
        FIXED_STAGE: expected_fixed,
        SLACLIP_STAGE: expected_slaclip,
        CONFIRMATION_STAGE: expected_confirmation,
        "total": expected_fixed + expected_slaclip + expected_confirmation,
    }
    if counts != expected:
        raise ValueError(f"expected stage counts differ: {counts} != {expected}")
    if not isinstance(value["scientific_boundary"], dict):
        raise ValueError("scientific_boundary must be an object")
    return value


def _token(value: float) -> str:
    return full.number_token(float(value))


def _base_arm(
    spec: Mapping[str, Any],
    *,
    arm_id: str,
    stage: str,
    method: str,
    model: str,
    seed: int,
    clip_norm: float,
    reference_arm_id: str | None,
    beta: float | None = None,
) -> dict[str, Any]:
    common = spec["common"]
    adaptive = method == SLACLIP_METHOD
    if adaptive != (beta is not None):
        raise ValueError("adaptive method and beta disagree")
    roles = {
        FIXED_STAGE: "development_strong_fixed_C_selection",
        SLACLIP_STAGE: "development_full_SlaClip_C0_beta_selection",
        CONFIRMATION_STAGE: "independent_training_seed_confirmation",
    }
    arm = {
        "arm_id": arm_id,
        "stage": stage,
        "family": stage,
        "analysis_role": roles[stage],
        "method": method,
        "seed": seed,
        "initial_clip_norm": float(clip_norm),
        "slaclip_eta": float(common["slaclip_eta"]) if adaptive else None,
        "slaclip_base_target_clipped_fraction": float(beta) if adaptive else None,
        "slaclip_beta": float(beta) if adaptive else None,
        "controller_input": NOISY_CONTROLLER_INPUT if adaptive else None,
        "reference_arm_id": reference_arm_id,
        "rng_domain": f"staged-full-slaclip:s{seed}",
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
    return arm


def _indexed(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(arms) % 2:
        raise ValueError("two-lane stage must contain an even number of arms")
    ids = [arm["arm_id"] for arm in arms]
    if len(ids) != len(set(ids)):
        raise ValueError("stage arm IDs are not unique")
    for index, arm in enumerate(arms):
        arm["index"] = index
        arm["wave"] = index // 2
        arm["lane"] = index % 2
    return arms


def fixed_stage_arms(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for model in EXPECTED_MODELS:
        for clip_norm in spec["fixed_development"]["clip_norm_grid"]:
            for seed in spec["fixed_development"]["seeds"]:
                arms.append(
                    _base_arm(
                        spec,
                        arm_id=f"dev-fixed-{model}-c{_token(clip_norm)}-s{seed}",
                        stage=FIXED_STAGE,
                        method=FIXED_METHOD,
                        model=model,
                        seed=seed,
                        clip_norm=clip_norm,
                        reference_arm_id=None,
                    )
                )
    return _indexed(arms)


def _runtime_manifest(
    *,
    spec: Mapping[str, Any],
    spec_path: Path,
    repository_sha: str,
    input_manifest: Path,
    created_at_utc: str,
    arms: list[dict[str, Any]],
    stage: str,
    parent_lock_sha256: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    payload["manifest_sha256"] = full.sha256_bytes(full.canonical_bytes(payload))
    full.validate_runtime_manifest(payload)
    return payload


def _preflight_manifest(master: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    common = spec["common"]
    seed = int(spec["fixed_development"]["seeds"][0])
    fixed = _base_arm(
        spec,
        arm_id="preflight-template-fixed",
        stage=FIXED_STAGE,
        method=FIXED_METHOD,
        model="bert",
        seed=seed,
        clip_norm=10.0,
        reference_arm_id=None,
    )
    adaptive = _base_arm(
        spec,
        arm_id="preflight-template-slaclip",
        stage=SLACLIP_STAGE,
        method=SLACLIP_METHOD,
        model="bert",
        seed=seed,
        clip_norm=10.0,
        reference_arm_id=fixed["arm_id"],
        beta=0.5,
    )
    for arm, method in ((fixed, FIXED_METHOD), (adaptive, SLACLIP_METHOD)):
        arm["family"] = "primary"
        arm["models"] = list(EXPECTED_MODELS)
        arm["method"] = method
    arms = _indexed([fixed, adaptive])
    payload = {
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
    payload["manifest_sha256"] = full.sha256_bytes(full.canonical_bytes(payload))
    full.validate_runtime_manifest(payload)
    if common["slaclip_num_slots"] != 5:
        raise AssertionError("preflight lost the K=5 contract")
    return payload


def _lock_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["lock_sha256"] = full.sha256_bytes(full.canonical_bytes(payload))
    return payload


def _validate_lock(value: dict[str, Any], label: str) -> None:
    fingerprint = value.get("lock_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise RuntimeError(f"{label} fingerprint is missing")
    unsigned = dict(value)
    del unsigned["lock_sha256"]
    if full.sha256_bytes(full.canonical_bytes(unsigned)) != fingerprint:
        raise RuntimeError(f"{label} fingerprint mismatch")
    if value.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise RuntimeError(f"{label} schema is unsupported")


def _write_or_verify(path: Path, value: Mapping[str, Any], label: str) -> None:
    if path.is_file():
        existing = full.load_object(path, label)
        if existing != value:
            raise RuntimeError(f"existing {label} differs from the immutable candidate")
        return
    if os.path.lexists(path):
        raise RuntimeError(f"{label} path is not a regular file: {path}")
    full.atomic_json(path, value)


def _completed_summary(
    campaign_root: Path,
    manifest: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    status_path = campaign_root / "arm-status" / f"{arm['arm_id']}.json"
    final_path = campaign_root / "arms" / str(arm["arm_id"]) / "final_summary.json"
    if not status_path.is_file() or not final_path.is_file():
        raise RuntimeError(f"arm is incomplete: {arm['arm_id']}")
    status = full.load_object(status_path, "arm status")
    summary = full.load_object(final_path, "arm final summary")
    if (
        status.get("status") != "COMPLETED"
        or status.get("runtime_manifest_sha256") != manifest["manifest_sha256"]
        or status.get("arm_spec_sha256")
        != full.sha256_bytes(full.canonical_bytes(arm))
        or status.get("final_summary_sha256") != full.sha256_file(final_path)
        or summary.get("status") != "COMPLETED"
        or summary.get("method") != arm["method"]
    ):
        raise RuntimeError(f"completed-arm identity failed: {arm['arm_id']}")
    models = summary.get("models")
    if not isinstance(models, dict) or tuple(models) != tuple(arm["models"]):
        raise RuntimeError(f"completed-arm model set differs: {arm['arm_id']}")
    model = str(arm["models"][0])
    model_summary = models.get(model)
    if len(arm["models"]) != 1 or not isinstance(model_summary, dict):
        raise RuntimeError(f"model-specific arm summary is invalid: {arm['arm_id']}")
    return summary, model_summary, full.sha256_file(final_path)


def _arm_evidence(
    campaign_root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm in manifest["arms"]:
        _summary, model_summary, final_sha = _completed_summary(
            campaign_root, manifest, arm
        )
        prefix = model_summary.get("round_shard_prefix_sha256")
        if not isinstance(prefix, str) or len(prefix) != 64:
            raise RuntimeError(f"round-shard prefix digest is missing: {arm['arm_id']}")
        computed_prefix = _computed_round_shard_prefix(campaign_root, arm)
        if computed_prefix != prefix:
            raise RuntimeError(f"round-shard prefix does not reconcile: {arm['arm_id']}")
        records.append(
            {
                "arm_id": arm["arm_id"],
                "final_summary_sha256": final_sha,
                "round_shard_prefix_sha256": prefix,
            }
        )
    return records


def _verify_locked_evidence(
    campaign_root: Path,
    manifest: Mapping[str, Any],
    evidence: Any,
) -> None:
    if not isinstance(evidence, list) or len(evidence) != len(manifest["arms"]):
        raise RuntimeError("selection-lock evidence list is incomplete")
    by_id = {
        item.get("arm_id"): item for item in evidence if isinstance(item, dict)
    }
    if set(by_id) != {arm["arm_id"] for arm in manifest["arms"]}:
        raise RuntimeError("selection-lock evidence arm set differs")
    for arm in manifest["arms"]:
        _summary, model_summary, final_sha = _completed_summary(
            campaign_root, manifest, arm
        )
        item = by_id[arm["arm_id"]]
        computed_prefix = _computed_round_shard_prefix(campaign_root, arm)
        if (
            item.get("final_summary_sha256") != final_sha
            or item.get("round_shard_prefix_sha256")
            != model_summary.get("round_shard_prefix_sha256")
            or item.get("round_shard_prefix_sha256") != computed_prefix
        ):
            raise RuntimeError(f"locked source evidence changed: {arm['arm_id']}")


def _computed_round_shard_prefix(
    campaign_root: Path, arm: Mapping[str, Any]
) -> str:
    try:
        from paper_repro.train_federated import round_shard_prefix_sha256
    except ModuleNotFoundError:  # Direct-script execution from paper_repro/.
        from train_federated import round_shard_prefix_sha256  # type: ignore[no-redef]
    model = str(arm["models"][0])
    directory = (
        campaign_root
        / "arms"
        / str(arm["arm_id"])
        / model
        / "private_diagnostics"
        / "rounds"
    )
    return round_shard_prefix_sha256(
        directory, completed_round=int(arm["rounds"])
    )


def _metric_row(
    campaign_root: Path,
    manifest: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> dict[str, Any]:
    _summary, model_summary, _sha = _completed_summary(campaign_root, manifest, arm)
    model = str(arm["models"][0])
    row = full._model_metrics(arm, model, model_summary)
    privacy_randomness = model_summary.get("privacy_randomness")
    if (
        not isinstance(privacy_randomness, dict)
        or privacy_randomness.get("pair_noise_across_methods") is not True
        or privacy_randomness.get("rng_domain") != arm["rng_domain"]
    ):
        raise RuntimeError(f"paired private-randomness contract failed: {arm['arm_id']}")
    commitment = privacy_randomness.get("private_key_commitment")
    if not isinstance(commitment, str) or len(commitment) != 64:
        raise RuntimeError(f"private RNG commitment is invalid: {arm['arm_id']}")
    row["stage"] = arm["stage"]
    row["runtime_manifest_sha256"] = manifest["manifest_sha256"]
    row["pair_noise_across_methods"] = True
    row["rng_domain"] = arm["rng_domain"]
    row["private_key_commitment"] = commitment
    return row


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        raise RuntimeError("cannot take a quantile of no finite values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability lies outside [0,1]")
    position = (len(finite) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def derive_beta_grid(
    conditional_values: Sequence[float],
    *,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
    points: int = 5,
) -> tuple[list[float], float, float]:
    if points < 2:
        raise ValueError("beta grid needs at least two points")
    values = [float(value) for value in conditional_values]
    if not values or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise RuntimeError("conditional beta calibration values are invalid")
    low = _linear_quantile(values, lower_quantile)
    high = _linear_quantile(values, upper_quantile)
    if high - low <= max(1e-12, 64.0 * math.ulp(max(abs(low), abs(high), 1.0))):
        raise RuntimeError("conditional beta q10-q90 interval is degenerate")
    grid = [low + (high - low) * index / (points - 1) for index in range(points)]
    grid[0] = low
    grid[-1] = high
    if len({full.canonical_bytes(value) for value in grid}) != points:
        raise RuntimeError("conditional beta grid contains duplicate points")
    if any(not 0.0 <= value <= 1.0 for value in grid):
        raise RuntimeError("derived beta grid lies outside [0,1]")
    return grid, low, high


def _round_shard_path(campaign_root: Path, arm: Mapping[str, Any], round_index: int) -> Path:
    model = str(arm["models"][0])
    return (
        campaign_root
        / "arms"
        / str(arm["arm_id"])
        / model
        / "private_diagnostics"
        / "rounds"
        / f"round-{round_index:05d}.json"
    )


def _fixed_beta_calibration(
    campaign_root: Path,
    arms: Sequence[Mapping[str, Any]],
    *,
    clip_norm: float,
    num_slots: int,
    epsilon: float,
    rounds: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    conditional_values: list[float] = []
    rows: list[dict[str, Any]] = []
    for arm in arms:
        model = str(arm["models"][0])
        for round_index in range(1, rounds + 1):
            shard = full.load_object(
                _round_shard_path(campaign_root, arm, round_index),
                "fixed-C round shard",
            )
            if (
                shard.get("round") != round_index
                or shard.get("model") != model
                or shard.get("method") != FIXED_METHOD
            ):
                raise RuntimeError(f"fixed calibration shard identity failed: {arm['arm_id']}")
            records = shard.get("client_records")
            summary = shard.get("round_summary")
            if not isinstance(records, list) or not records or not isinstance(summary, dict):
                raise RuntimeError("fixed calibration shard content is invalid")
            raw_norms: list[float] = []
            slack_vectors: list[tuple[float, ...]] = []
            for record in records:
                try:
                    norm = float(record["gradient_groups"]["B"]["raw_norm"])
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError("fixed calibration raw B norm is missing") from error
                raw_norms.append(norm)
                slack_vectors.append(build_slack_vector(norm, clip_norm, num_slots))
            signal_sum = [
                math.fsum(vector[slot] for vector in slack_vectors)
                for slot in range(num_slots)
            ]
            exact_proxy = normalize_noisy_slack(
                signal_sum, clip_norm, num_slots, len(records)
            )
            exact_near_threshold = float(exact_proxy[0])
            exact_endpoint = float(exact_proxy[-1])
            stationary_calibration = stationary_beta_from_exact_endpoints(
                clip_norm,
                exact_near_threshold,
                exact_endpoint,
                epsilon=epsilon,
            )
            z_value = stationary_calibration["near_zero_adjusted"]
            remaining = stationary_calibration[
                "one_minus_threshold_adjusted_near_zero_signal"
            ]
            try:
                clipped_fraction = float(summary["B"]["clipped_fraction"])
                any_fraction = float(summary["any_group_clipped_fraction"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("fixed calibration clipping fraction is missing") from error
            if not 0.0 < remaining <= 1.0 + 1e-12:
                raise RuntimeError("fixed calibration has no valid remaining mass")
            # The controller compares gamma with the first slack endpoint q;
            # it does not compare gamma with the exact unclipped fraction
            # 1-p.  Therefore the beta that makes this fixed-C point
            # stationary is (1-q)/(1-z).  p/(1-z) is retained only as a
            # finite-K tracking diagnostic.
            stationary = stationary_calibration["stationary_beta"]
            actual_target_calibration = clipped_fraction / remaining
            for label, value in (
                ("stationary beta", stationary),
                ("actual-clipping diagnostic beta", actual_target_calibration),
            ):
                if not -1e-12 <= value <= 1.0 + 1e-12:
                    raise RuntimeError(
                        f"fixed calibration {label} lies outside [0,1]"
                    )
            stationary = max(0.0, min(1.0, stationary))
            actual_target_calibration = max(
                0.0, min(1.0, actual_target_calibration)
            )
            conditional_values.append(stationary)
            rows.append(
                {
                    "arm_id": arm["arm_id"],
                    "model": model,
                    "seed": arm["seed"],
                    "round": round_index,
                    "selected_fixed_C": clip_norm,
                    "B_clipped_fraction": clipped_fraction,
                    "any_group_clipped_fraction": any_fraction,
                    "exact_normalized_slack_endpoint_1": exact_near_threshold,
                    "exact_normalized_slack_endpoint_K": exact_endpoint,
                    "near_zero_adjusted_z": z_value,
                    "remaining_non_small_gradient_fraction": remaining,
                    "stationary_target_clipped_surrogate": (
                        1.0 - exact_near_threshold
                    ),
                    "actual_clipped_fraction_tracking_bias": (
                        (1.0 - exact_near_threshold) - clipped_fraction
                    ),
                    "actual_clipped_fraction_calibrated_beta": (
                        actual_target_calibration
                    ),
                    "stationary_beta": stationary,
                    # Compatibility field for old analysis readers.  Its
                    # value now has the explicitly documented stationary
                    # controller semantics above.
                    "conditional_beta": stationary,
                }
            )
    return conditional_values, rows


def _fixed_arm_id(model: str, clip_norm: float, seed: int) -> str:
    return f"dev-fixed-{model}-c{_token(clip_norm)}-s{seed}"


def _c0_candidates(grid: Sequence[float], selected: float) -> tuple[list[float], bool]:
    normalized = [float(value) for value in grid]
    try:
        index = normalized.index(float(selected))
    except ValueError as error:
        raise RuntimeError("selected fixed C is absent from its grid") from error
    boundary = index in {0, len(normalized) - 1}
    if index == 0:
        values = normalized[:3]
    elif index == len(normalized) - 1:
        values = normalized[-3:]
    else:
        values = normalized[index - 1 : index + 2]
    if len(values) != 3 or selected not in values:
        raise RuntimeError("could not construct three initial-C candidates")
    return values, boundary


def _build_master_runtime(
    spec: Mapping[str, Any],
    spec_path: Path,
    *,
    repository_sha: str,
    input_manifest: Path,
    created_at_utc: str,
) -> dict[str, Any]:
    return _runtime_manifest(
        spec=spec,
        spec_path=spec_path,
        repository_sha=repository_sha,
        input_manifest=input_manifest,
        created_at_utc=created_at_utc,
        arms=fixed_stage_arms(spec),
        stage=FIXED_STAGE,
        parent_lock_sha256=None,
    )


def _verify_master_identity(
    campaign_root: Path,
    spec_path: Path,
    repository: Path,
    expected_code_sha: str,
    input_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = load_spec(spec_path)
    if full.repository_sha(repository) != expected_code_sha or full.repository_dirty(repository):
        raise RuntimeError("repository snapshot SHA/cleanliness gate failed")
    master_path = campaign_root / MASTER_RUNTIME_NAME
    master = full.load_runtime(master_path)
    candidate = _build_master_runtime(
        spec,
        spec_path,
        repository_sha=expected_code_sha,
        input_manifest=input_manifest,
        created_at_utc=str(master.get("created_at_utc")),
    )
    if master != candidate:
        raise RuntimeError("master runtime manifest differs from immutable inputs")
    return spec, master


def prepare_campaign(args: argparse.Namespace) -> None:
    campaign_root = args.campaign_root.resolve()
    repository = args.repository.resolve()
    spec_path = args.spec.resolve()
    input_manifest = args.input_manifest.resolve()
    spec = load_spec(spec_path)
    if full.repository_sha(repository) != args.expected_code_sha or full.repository_dirty(repository):
        raise RuntimeError("repository snapshot SHA/cleanliness gate failed")
    master_path = campaign_root / MASTER_RUNTIME_NAME
    if args.resume:
        if not campaign_root.is_dir() or not master_path.is_file():
            raise RuntimeError("resume requires an existing staged campaign")
        master = full.load_runtime(master_path)
        candidate = _build_master_runtime(
            spec,
            spec_path,
            repository_sha=args.expected_code_sha,
            input_manifest=input_manifest,
            created_at_utc=str(master.get("created_at_utc")),
        )
        if master != candidate:
            raise RuntimeError("resume inputs differ from the immutable master manifest")
    else:
        if campaign_root.exists():
            raise RuntimeError(f"refusing to overwrite campaign root: {campaign_root}")
        campaign_root.mkdir(parents=True, mode=0o700)
        master = _build_master_runtime(
            spec,
            spec_path,
            repository_sha=args.expected_code_sha,
            input_manifest=input_manifest,
            created_at_utc=full.utc_now(),
        )
        full.atomic_json(master_path, master)
    for directory in (
        "arms",
        "arm-status",
        "arm-logs",
        "control",
        "tmp",
        "preflight",
        "selection",
    ):
        (campaign_root / directory).mkdir(mode=0o700, exist_ok=True)
    stop = campaign_root / "control" / "stop.request"
    if stop.exists():
        stop.unlink()
    full.validate_or_create_key(full.absolute_path(args.private_key), create=not args.resume)
    preflight_path = campaign_root / PREFLIGHT_RUNTIME_NAME
    _write_or_verify(
        preflight_path,
        _preflight_manifest(master, spec),
        "preflight runtime manifest",
    )

    fixed_lock_path = campaign_root / FIXED_LOCK_NAME
    if fixed_lock_path.is_file():
        fixed_lock = full.load_object(fixed_lock_path, "fixed selection lock")
        _validate_lock(fixed_lock, "fixed selection lock")
        _validate_fixed_lock_identity(fixed_lock, master, spec)
        _verify_locked_evidence(campaign_root, master, fixed_lock.get("source_evidence"))
        _verify_beta_calibration_artifact(campaign_root, fixed_lock)
        _ensure_stage2_manifest(campaign_root, spec_path, spec, master, fixed_lock)
    slaclip_lock_path = campaign_root / SLACLIP_LOCK_NAME
    if slaclip_lock_path.is_file():
        if not fixed_lock_path.is_file():
            raise RuntimeError("SlaClip lock exists without a fixed selection lock")
        fixed_lock = full.load_object(fixed_lock_path, "fixed selection lock")
        stage2 = full.load_runtime(campaign_root / STAGE2_RUNTIME_NAME)
        slaclip_lock = full.load_object(slaclip_lock_path, "SlaClip selection lock")
        _validate_lock(slaclip_lock, "SlaClip selection lock")
        _validate_slaclip_lock_identity(slaclip_lock, master, fixed_lock, stage2, spec)
        _verify_locked_evidence(campaign_root, stage2, slaclip_lock.get("source_evidence"))
        _ensure_stage3_manifest(
            campaign_root, spec_path, spec, master, fixed_lock, slaclip_lock
        )
    print(f"runtime_manifest={master_path}")
    print(f"preflight_runtime_manifest={preflight_path}")


def _validate_fixed_lock_identity(
    lock: Mapping[str, Any],
    master: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    if (
        lock.get("status") != "FIXED_DEVELOPMENT_SELECTION_LOCKED"
        or lock.get("campaign_name") != master["campaign_name"]
        or lock.get("master_runtime_manifest_sha256") != master["manifest_sha256"]
        or lock.get("stage1_runtime_manifest_sha256") != master["manifest_sha256"]
        or lock.get("spec_sha256") != master["spec_sha256"]
        or lock.get("selection_rule") != spec["fixed_development"]["selection_rule"]
    ):
        raise RuntimeError("fixed selection lock identity differs")
    calibration_sha = lock.get("beta_calibration_csv_sha256")
    if not isinstance(calibration_sha, str) or len(calibration_sha) != 64:
        raise RuntimeError("fixed selection lock lacks its calibration-table digest")
    models = lock.get("models")
    if not isinstance(models, dict) or tuple(models) != EXPECTED_MODELS:
        raise RuntimeError("fixed selection lock model set differs")
    grid = [float(value) for value in spec["fixed_development"]["clip_norm_grid"]]
    for model in EXPECTED_MODELS:
        record = models.get(model)
        if not isinstance(record, dict):
            raise RuntimeError("fixed selection lock model record is invalid")
        selected = record.get("selected_fixed_C")
        candidates = record.get("slaclip_initial_C_candidates")
        beta_grid = record.get("derived_beta_grid")
        if (
            not isinstance(selected, (int, float))
            or float(selected) not in grid
            or not isinstance(candidates, list)
            or len(candidates) != 3
            or float(selected) not in [float(value) for value in candidates]
            or not isinstance(beta_grid, list)
            or len(beta_grid) != 5
            or len({full.canonical_bytes(value) for value in beta_grid}) != 5
            or any(not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0 for value in beta_grid)
        ):
            raise RuntimeError("fixed selection lock contains invalid candidates")


def _verify_beta_calibration_artifact(
    campaign_root: Path, fixed_lock: Mapping[str, Any]
) -> None:
    path = campaign_root / "fixed_beta_calibration.csv"
    if not path.is_file():
        raise RuntimeError("locked fixed-beta calibration table is missing")
    if full.sha256_file(path) != fixed_lock["beta_calibration_csv_sha256"]:
        raise RuntimeError("locked fixed-beta calibration table digest differs")


def lock_fixed_selection(args: argparse.Namespace) -> None:
    campaign_root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    repository = args.repository.resolve()
    input_manifest = args.input_manifest.resolve()
    spec, master = _verify_master_identity(
        campaign_root,
        spec_path,
        repository,
        args.expected_code_sha,
        input_manifest,
    )
    lock_path = campaign_root / FIXED_LOCK_NAME
    if lock_path.is_file():
        lock = full.load_object(lock_path, "fixed selection lock")
        _validate_lock(lock, "fixed selection lock")
        _validate_fixed_lock_identity(lock, master, spec)
        _verify_locked_evidence(campaign_root, master, lock.get("source_evidence"))
        _verify_beta_calibration_artifact(campaign_root, lock)
        stage2 = _ensure_stage2_manifest(campaign_root, spec_path, spec, master, lock)
        print(f"fixed_selection_reused={lock_path}")
        print(f"stage2_runtime_manifest={stage2}")
        return

    evidence = _arm_evidence(campaign_root, master)
    metric_rows = [
        _metric_row(campaign_root, master, arm) for arm in master["arms"]
    ]
    grid = [float(value) for value in spec["fixed_development"]["clip_norm_grid"]]
    development_seeds = [int(value) for value in spec["fixed_development"]["seeds"]]
    model_records: dict[str, Any] = {}
    beta_calibration_rows: list[dict[str, Any]] = []
    for model in EXPECTED_MODELS:
        rankings: list[dict[str, Any]] = []
        for clip_norm in grid:
            rows = [
                row
                for row in metric_rows
                if row["model"] == model and float(row["initial_clip_norm"]) == clip_norm
            ]
            if {int(row["seed"]) for row in rows} != set(development_seeds):
                raise RuntimeError(f"fixed-C evidence is incomplete: {model}/C={clip_norm}")
            final_values = [float(row["final_loss"]) for row in rows]
            auc_values = [float(row["normalized_loss_auc"]) for row in rows]
            if any(not math.isfinite(value) for value in (*final_values, *auc_values)):
                raise RuntimeError("fixed-C selection metric is non-finite")
            rankings.append(
                {
                    "fixed_C": clip_norm,
                    "seed_count": len(rows),
                    "mean_final_loss": statistics.fmean(final_values),
                    "mean_normalized_loss_auc": statistics.fmean(auc_values),
                    "final_loss_sample_std": statistics.stdev(final_values),
                    "mean_actual_clipped_fraction": statistics.fmean(
                        float(row["actual_clipped_fraction"]) for row in rows
                    ),
                }
            )
        rankings.sort(
            key=lambda row: (
                row["mean_final_loss"],
                row["mean_normalized_loss_auc"],
                row["final_loss_sample_std"],
                row["fixed_C"],
            )
        )
        selected = float(rankings[0]["fixed_C"])
        c0_values, boundary_hit = _c0_candidates(grid, selected)
        selected_arms = [
            arm
            for arm in master["arms"]
            if arm["models"] == [model]
            and float(arm["initial_clip_norm"]) == selected
        ]
        if {int(arm["seed"]) for arm in selected_arms} != set(development_seeds):
            raise RuntimeError("selected fixed-C arm set is incomplete")
        conditional, calibration = _fixed_beta_calibration(
            campaign_root,
            selected_arms,
            clip_norm=selected,
            num_slots=int(spec["common"]["slaclip_num_slots"]),
            epsilon=float(spec["common"]["slaclip_endpoint_epsilon"]),
            rounds=int(spec["common"]["rounds"]),
        )
        beta_grid, q10, q90 = derive_beta_grid(conditional)
        beta_calibration_rows.extend(calibration)
        model_records[model] = {
            "selected_fixed_C": selected,
            "fixed_grid_boundary_hit": boundary_hit,
            "slaclip_initial_C_candidates": c0_values,
            "derived_beta_grid": beta_grid,
            "beta_calibration": {
                "group": "B",
                "sample_count": len(conditional),
                "stationary_beta_min": min(conditional),
                "stationary_beta_q10": q10,
                "stationary_beta_median": statistics.median(conditional),
                "stationary_beta_q90": q90,
                "stationary_beta_max": max(conditional),
                "formula": "(1-q_exact_t)/(1-z_exact_t)",
                "z_formula": "exact_normalized_slack_endpoint_K/(C+epsilon)",
                "num_slots": spec["common"]["slaclip_num_slots"],
            },
            "ordered_fixed_C_candidates": rankings,
        }
    calibration_path = campaign_root / "fixed_beta_calibration.csv"
    full.atomic_csv(
        calibration_path,
        beta_calibration_rows,
        (
            "arm_id",
            "model",
            "seed",
            "round",
            "selected_fixed_C",
            "B_clipped_fraction",
            "any_group_clipped_fraction",
            "exact_normalized_slack_endpoint_1",
            "exact_normalized_slack_endpoint_K",
            "near_zero_adjusted_z",
            "remaining_non_small_gradient_fraction",
            "stationary_target_clipped_surrogate",
            "actual_clipped_fraction_tracking_bias",
            "actual_clipped_fraction_calibrated_beta",
            "stationary_beta",
            "conditional_beta",
        ),
    )
    lock = _lock_payload(
        {
            "schema_version": LOCK_SCHEMA_VERSION,
            "status": "FIXED_DEVELOPMENT_SELECTION_LOCKED",
            "campaign_name": master["campaign_name"],
            "master_runtime_manifest_sha256": master["manifest_sha256"],
            "stage1_runtime_manifest_sha256": master["manifest_sha256"],
            "spec_sha256": master["spec_sha256"],
            "selection_rule": spec["fixed_development"]["selection_rule"],
            "development_seeds": development_seeds,
            "confirmation_data_accessed": False,
            "models": model_records,
            "source_evidence": evidence,
            "beta_calibration_csv_sha256": full.sha256_file(calibration_path),
            "created_at_utc": full.utc_now(),
        }
    )
    full.atomic_json(lock_path, lock)
    _validate_lock(full.load_object(lock_path, "fixed selection lock"), "fixed selection lock")
    _verify_beta_calibration_artifact(campaign_root, lock)
    stage2 = _ensure_stage2_manifest(campaign_root, spec_path, spec, master, lock)
    _write_trajectories(campaign_root, [(master, master["arms"])])
    print(f"fixed_selection_lock={lock_path}")
    print(f"stage2_runtime_manifest={stage2}")


def stage2_arms(
    spec: Mapping[str, Any], fixed_lock: Mapping[str, Any]
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    seeds = [int(value) for value in spec["slaclip_development"]["seeds"]]
    for model in EXPECTED_MODELS:
        model_lock = fixed_lock["models"][model]
        selected_fixed = float(model_lock["selected_fixed_C"])
        for initial_c in model_lock["slaclip_initial_C_candidates"]:
            for beta in model_lock["derived_beta_grid"]:
                for seed in seeds:
                    reference = _fixed_arm_id(model, selected_fixed, seed)
                    arms.append(
                        _base_arm(
                            spec,
                            arm_id=(
                                f"dev-slaclip-{model}-c0{_token(initial_c)}-"
                                f"b{_token(beta)}-s{seed}"
                            ),
                            stage=SLACLIP_STAGE,
                            method=SLACLIP_METHOD,
                            model=model,
                            seed=seed,
                            clip_norm=float(initial_c),
                            beta=float(beta),
                            reference_arm_id=reference,
                        )
                    )
    return _indexed(arms)


def _ensure_stage2_manifest(
    campaign_root: Path,
    spec_path: Path,
    spec: Mapping[str, Any],
    master: Mapping[str, Any],
    fixed_lock: Mapping[str, Any],
) -> Path:
    _validate_fixed_lock_identity(fixed_lock, master, spec)
    path = campaign_root / STAGE2_RUNTIME_NAME
    candidate = _runtime_manifest(
        spec=spec,
        spec_path=spec_path,
        repository_sha=str(master["repository_sha"]),
        input_manifest=Path(str(master["input_manifest_path"])),
        created_at_utc=str(master["created_at_utc"]),
        arms=stage2_arms(spec, fixed_lock),
        stage=SLACLIP_STAGE,
        parent_lock_sha256=str(fixed_lock["lock_sha256"]),
    )
    expected = int(spec["expected_stage_arm_counts"][SLACLIP_STAGE])
    if len(candidate["arms"]) != expected:
        raise RuntimeError("materialized Stage 2 arm count differs")
    _write_or_verify(path, candidate, "Stage 2 runtime manifest")
    return path


def _controller_instability_events(row: Mapping[str, Any]) -> int:
    names = (
        "gamma_clamped_low_count",
        "gamma_clamped_high_count",
        "log_step_bounded_count",
        "lower_bound_hits",
        "upper_bound_hits",
    )
    total = 0
    for group in ("A", "B"):
        for name in names:
            value = row.get(f"{name}_{group}")
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RuntimeError(f"controller instability metric is invalid: {name}_{group}")
            total += value
    return total


def _validate_slaclip_lock_identity(
    lock: Mapping[str, Any],
    master: Mapping[str, Any],
    fixed_lock: Mapping[str, Any],
    stage2: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    if (
        lock.get("status") != "SLACLIP_DEVELOPMENT_SELECTION_LOCKED"
        or lock.get("campaign_name") != master["campaign_name"]
        or lock.get("master_runtime_manifest_sha256") != master["manifest_sha256"]
        or lock.get("fixed_selection_lock_sha256") != fixed_lock["lock_sha256"]
        or lock.get("stage2_runtime_manifest_sha256") != stage2["manifest_sha256"]
        or lock.get("selection_rule") != spec["slaclip_development"]["selection_rule"]
    ):
        raise RuntimeError("SlaClip selection lock identity differs")
    models = lock.get("models")
    if not isinstance(models, dict) or tuple(models) != EXPECTED_MODELS:
        raise RuntimeError("SlaClip selection lock model set differs")
    for model in EXPECTED_MODELS:
        record = models.get(model)
        fixed_record = fixed_lock["models"][model]
        if not isinstance(record, dict):
            raise RuntimeError("SlaClip lock model record is invalid")
        if (
            float(record.get("selected_fixed_C", math.nan))
            != float(fixed_record["selected_fixed_C"])
            or float(record.get("selected_initial_C", math.nan))
            not in [float(value) for value in fixed_record["slaclip_initial_C_candidates"]]
            or float(record.get("selected_beta", math.nan))
            not in [float(value) for value in fixed_record["derived_beta_grid"]]
        ):
            raise RuntimeError("SlaClip lock selected configuration is invalid")


def lock_slaclip_selection(args: argparse.Namespace) -> None:
    campaign_root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec, master = _verify_master_identity(
        campaign_root,
        spec_path,
        args.repository.resolve(),
        args.expected_code_sha,
        args.input_manifest.resolve(),
    )
    fixed_lock = full.load_object(campaign_root / FIXED_LOCK_NAME, "fixed selection lock")
    _validate_lock(fixed_lock, "fixed selection lock")
    _validate_fixed_lock_identity(fixed_lock, master, spec)
    _verify_locked_evidence(campaign_root, master, fixed_lock.get("source_evidence"))
    _verify_beta_calibration_artifact(campaign_root, fixed_lock)
    stage2_path = _ensure_stage2_manifest(
        campaign_root, spec_path, spec, master, fixed_lock
    )
    stage2 = full.load_runtime(stage2_path)
    lock_path = campaign_root / SLACLIP_LOCK_NAME
    if lock_path.is_file():
        lock = full.load_object(lock_path, "SlaClip selection lock")
        _validate_lock(lock, "SlaClip selection lock")
        _validate_slaclip_lock_identity(lock, master, fixed_lock, stage2, spec)
        _verify_locked_evidence(campaign_root, stage2, lock.get("source_evidence"))
        stage3 = _ensure_stage3_manifest(
            campaign_root, spec_path, spec, master, fixed_lock, lock
        )
        print(f"slaclip_selection_reused={lock_path}")
        print(f"stage3_runtime_manifest={stage3}")
        return

    evidence = _arm_evidence(campaign_root, stage2)
    fixed_rows = {
        (row["model"], int(row["seed"])): row
        for row in (
            _metric_row(campaign_root, master, arm)
            for arm in master["arms"]
            if float(arm["initial_clip_norm"])
            == float(fixed_lock["models"][str(arm["models"][0])]["selected_fixed_C"])
        )
    }
    adaptive_rows = [
        _metric_row(campaign_root, stage2, arm) for arm in stage2["arms"]
    ]
    development_seeds = {int(value) for value in spec["slaclip_development"]["seeds"]}
    model_records: dict[str, Any] = {}
    for model in EXPECTED_MODELS:
        candidates: list[dict[str, Any]] = []
        c0_values = [
            float(value)
            for value in fixed_lock["models"][model]["slaclip_initial_C_candidates"]
        ]
        beta_values = [
            float(value) for value in fixed_lock["models"][model]["derived_beta_grid"]
        ]
        for initial_c in c0_values:
            for beta in beta_values:
                rows = [
                    row
                    for row in adaptive_rows
                    if row["model"] == model
                    and float(row["initial_clip_norm"]) == initial_c
                    and float(row["slaclip_beta"]) == beta
                ]
                if {int(row["seed"]) for row in rows} != development_seeds:
                    raise RuntimeError("SlaClip development candidate is incomplete")
                final_deltas: list[float] = []
                auc_deltas: list[float] = []
                instability = 0
                for row in rows:
                    reference = fixed_rows.get((model, int(row["seed"])))
                    if reference is None:
                        raise RuntimeError("SlaClip candidate has no selected fixed reference")
                    for digest in (
                        "sample_schedule_sha256",
                        "supervision_schedule_sha256",
                        "private_key_commitment",
                        "rng_domain",
                    ):
                        if row.get(digest) != reference.get(digest):
                            raise RuntimeError(
                                f"paired development schedule differs: {model}/{row['seed']}/{digest}"
                            )
                    if (
                        row.get("pair_noise_across_methods") is not True
                        or reference.get("pair_noise_across_methods") is not True
                    ):
                        raise RuntimeError("paired development noise-sharing flag differs")
                    final_deltas.append(float(row["final_loss"]) - float(reference["final_loss"]))
                    auc_deltas.append(
                        float(row["normalized_loss_auc"])
                        - float(reference["normalized_loss_auc"])
                    )
                    instability += _controller_instability_events(row)
                candidates.append(
                    {
                        "initial_C": initial_c,
                        "beta": beta,
                        "seed_count": len(rows),
                        "mean_paired_final_loss_delta": statistics.fmean(final_deltas),
                        "mean_paired_normalized_loss_auc_delta": statistics.fmean(auc_deltas),
                        "controller_instability_event_count": instability,
                        "paired_final_loss_deltas": final_deltas,
                        "paired_normalized_loss_auc_deltas": auc_deltas,
                    }
                )
        candidates.sort(
            key=lambda row: (
                row["mean_paired_final_loss_delta"],
                row["mean_paired_normalized_loss_auc_delta"],
                row["controller_instability_event_count"],
                row["beta"],
                row["initial_C"],
            )
        )
        selected = candidates[0]
        model_records[model] = {
            "selected_fixed_C": float(fixed_lock["models"][model]["selected_fixed_C"]),
            "selected_initial_C": selected["initial_C"],
            "selected_beta": selected["beta"],
            "selected_eta": float(spec["common"]["slaclip_eta"]),
            "selected_num_slots": int(spec["common"]["slaclip_num_slots"]),
            "beta_grid_boundary_hit": selected["beta"] in {min(beta_values), max(beta_values)},
            "initial_C_candidate_boundary_hit": selected["initial_C"] in {min(c0_values), max(c0_values)},
            "ordered_candidates": candidates,
        }
    lock = _lock_payload(
        {
            "schema_version": LOCK_SCHEMA_VERSION,
            "status": "SLACLIP_DEVELOPMENT_SELECTION_LOCKED",
            "campaign_name": master["campaign_name"],
            "master_runtime_manifest_sha256": master["manifest_sha256"],
            "fixed_selection_lock_sha256": fixed_lock["lock_sha256"],
            "stage2_runtime_manifest_sha256": stage2["manifest_sha256"],
            "selection_rule": spec["slaclip_development"]["selection_rule"],
            "development_seeds": sorted(development_seeds),
            "confirmation_data_accessed": False,
            "models": model_records,
            "source_evidence": evidence,
            "created_at_utc": full.utc_now(),
        }
    )
    full.atomic_json(lock_path, lock)
    _validate_lock(full.load_object(lock_path, "SlaClip selection lock"), "SlaClip selection lock")
    stage3 = _ensure_stage3_manifest(
        campaign_root, spec_path, spec, master, fixed_lock, lock
    )
    _write_trajectories(
        campaign_root,
        [(master, master["arms"]), (stage2, stage2["arms"])],
    )
    print(f"slaclip_selection_lock={lock_path}")
    print(f"stage3_runtime_manifest={stage3}")


def stage3_arms(
    spec: Mapping[str, Any],
    fixed_lock: Mapping[str, Any],
    slaclip_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for model in EXPECTED_MODELS:
        selected_fixed = float(fixed_lock["models"][model]["selected_fixed_C"])
        selected = slaclip_lock["models"][model]
        for seed in spec["confirmation"]["seeds"]:
            fixed_id = f"confirm-fixed-{model}-c{_token(selected_fixed)}-s{seed}"
            arms.append(
                _base_arm(
                    spec,
                    arm_id=fixed_id,
                    stage=CONFIRMATION_STAGE,
                    method=FIXED_METHOD,
                    model=model,
                    seed=int(seed),
                    clip_norm=selected_fixed,
                    reference_arm_id=None,
                )
            )
            arms.append(
                _base_arm(
                    spec,
                    arm_id=(
                        f"confirm-slaclip-{model}-c0{_token(selected['selected_initial_C'])}-"
                        f"b{_token(selected['selected_beta'])}-s{seed}"
                    ),
                    stage=CONFIRMATION_STAGE,
                    method=SLACLIP_METHOD,
                    model=model,
                    seed=int(seed),
                    clip_norm=float(selected["selected_initial_C"]),
                    beta=float(selected["selected_beta"]),
                    reference_arm_id=fixed_id,
                )
            )
    return _indexed(arms)


def _ensure_stage3_manifest(
    campaign_root: Path,
    spec_path: Path,
    spec: Mapping[str, Any],
    master: Mapping[str, Any],
    fixed_lock: Mapping[str, Any],
    slaclip_lock: Mapping[str, Any],
) -> Path:
    path = campaign_root / STAGE3_RUNTIME_NAME
    candidate = _runtime_manifest(
        spec=spec,
        spec_path=spec_path,
        repository_sha=str(master["repository_sha"]),
        input_manifest=Path(str(master["input_manifest_path"])),
        created_at_utc=str(master["created_at_utc"]),
        arms=stage3_arms(spec, fixed_lock, slaclip_lock),
        stage=CONFIRMATION_STAGE,
        parent_lock_sha256=str(slaclip_lock["lock_sha256"]),
    )
    expected = int(spec["expected_stage_arm_counts"][CONFIRMATION_STAGE])
    if len(candidate["arms"]) != expected:
        raise RuntimeError("materialized Stage 3 arm count differs")
    _write_or_verify(path, candidate, "Stage 3 runtime manifest")
    return path


def _quantile_summary(values: Iterable[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "q25": None, "median": None, "q75": None, "max": None}
    return {
        "mean": statistics.fmean(finite),
        "q25": _linear_quantile(finite, 0.25),
        "median": _linear_quantile(finite, 0.5),
        "q75": _linear_quantile(finite, 0.75),
        "max": max(finite),
    }


TRAJECTORY_COLUMNS = (
    "stage",
    "arm_id",
    "method",
    "privacy_label",
    "controller_input",
    "model",
    "seed",
    "round",
    "group",
    "initial_C",
    "base_target_clipped_fraction_beta",
    "clip_threshold_used",
    "next_clip_threshold",
    "actual_clipped_count",
    "actual_clipped_fraction",
    "would_clip_count",
    "would_clip_fraction",
    "raw_gradient_l2_mean",
    "raw_gradient_l2_q25",
    "raw_gradient_l2_median",
    "raw_gradient_l2_q75",
    "raw_gradient_l2_max",
    "raw_to_threshold_ratio_median",
    "removed_gradient_l2_median",
    "retained_energy_fraction_median",
    "noise_gradient_l2_mean",
    "aggregate_signal_gradient_l2",
    "aggregate_noise_gradient_l2",
    "aggregate_signal_to_noise_l2_ratio",
    "near_threshold_proxy",
    "near_zero_proxy",
    "near_zero_adjusted",
    "remaining_non_small_gradient_fraction",
    "raw_dynamic_target_clipped",
    "dynamic_target_clipped",
    "noisy_dynamic_target_clipped",
    "oracle_dynamic_target_clipped",
    "actual_minus_dynamic_target_clipped",
    "actual_target_absolute_error",
    "controller_error",
    "raw_log_step",
    "bounded_log_step",
    "noisy_raw_log_step",
    "oracle_raw_log_step",
    "noisy_next_clip_threshold",
    "oracle_next_clip_threshold",
    "noisy_minus_oracle_raw_log_step",
    "noisy_oracle_log_threshold_error",
    "update_direction_agrees",
    "cdf_error_mae",
    "cdf_error_rmse",
    "noisy_cdf_out_of_range_count",
    "noisy_cdf_out_of_range_fraction",
    "noisy_adjacent_monotonicity_violations",
    "exact_adjacent_monotonicity_violations",
    "gamma_clamped_low",
    "gamma_clamped_high",
    "log_step_was_bounded",
    "hit_min_clip_norm",
    "hit_max_clip_norm",
    "validation_loss",
    "round_total_elapsed_seconds",
    "round_training_elapsed_seconds",
    "cuda_max_memory_allocated_bytes",
    "cuda_max_memory_reserved_bytes",
    "host_max_rss_bytes",
)


def _round_trajectory_rows(
    campaign_root: Path,
    arm: Mapping[str, Any],
) -> list[dict[str, Any]]:
    model = str(arm["models"][0])
    rows: list[dict[str, Any]] = []
    for round_index in range(1, int(arm["rounds"]) + 1):
        shard = full.load_object(
            _round_shard_path(campaign_root, arm, round_index),
            "round trajectory shard",
        )
        clients = shard.get("client_records")
        summary = shard.get("round_summary")
        if not isinstance(clients, list) or not isinstance(summary, dict):
            raise RuntimeError(f"trajectory shard is incomplete: {arm['arm_id']}")
        validation = summary.get("validation")
        validation_loss = (
            validation.get("loss") if isinstance(validation, dict) else None
        )
        controller_root = summary.get("slaclip_controller")
        controller_root = controller_root if isinstance(controller_root, dict) else {}
        resources = summary.get("resource_telemetry")
        resources = resources if isinstance(resources, dict) else {}
        for group in ("A", "B"):
            group_summary = summary.get(group)
            update = summary.get("federated_update", {}).get(group)
            controller = controller_root.get(group)
            if not isinstance(group_summary, dict) or not isinstance(update, dict):
                raise RuntimeError("trajectory group summary is incomplete")
            controller = controller if isinstance(controller, dict) else {}
            client_groups = [record["gradient_groups"][group] for record in clients]
            raw = _quantile_summary(float(value["raw_norm"]) for value in client_groups)
            ratio = _quantile_summary(
                float(value["raw_to_threshold_ratio"]) for value in client_groups
            )
            removed = _quantile_summary(
                float(value["removed_gradient_l2"]) for value in client_groups
            )
            retained = _quantile_summary(
                float(value["retained_energy_fraction"]) for value in client_groups
            )
            noise = _quantile_summary(
                float(value["noise_l2_norm"]) for value in client_groups
            )
            rows.append(
                {
                    "stage": arm["stage"],
                    "arm_id": arm["arm_id"],
                    "method": arm["method"],
                    "privacy_label": summary.get("privacy_label"),
                    "controller_input": controller.get("controller_input"),
                    "model": model,
                    "seed": arm["seed"],
                    "round": round_index,
                    "group": group,
                    "initial_C": arm["initial_clip_norm"],
                    "base_target_clipped_fraction_beta": arm["slaclip_beta"],
                    "clip_threshold_used": controller.get(
                        "clip_threshold_used", arm["initial_clip_norm"]
                    ),
                    "next_clip_threshold": controller.get(
                        "next_clip_threshold", arm["initial_clip_norm"]
                    ),
                    "actual_clipped_count": group_summary.get("clipped_count"),
                    "actual_clipped_fraction": group_summary.get("clipped_fraction"),
                    "would_clip_count": group_summary.get("would_clip_count"),
                    "would_clip_fraction": group_summary.get("would_clip_fraction"),
                    "raw_gradient_l2_mean": raw["mean"],
                    "raw_gradient_l2_q25": raw["q25"],
                    "raw_gradient_l2_median": raw["median"],
                    "raw_gradient_l2_q75": raw["q75"],
                    "raw_gradient_l2_max": raw["max"],
                    "raw_to_threshold_ratio_median": ratio["median"],
                    "removed_gradient_l2_median": removed["median"],
                    "retained_energy_fraction_median": retained["median"],
                    "noise_gradient_l2_mean": noise["mean"],
                    "aggregate_signal_gradient_l2": update.get(
                        "aggregate_signal_gradient_l2"
                    ),
                    "aggregate_noise_gradient_l2": update.get(
                        "aggregate_noise_gradient_l2"
                    ),
                    "aggregate_signal_to_noise_l2_ratio": update.get(
                        "signal_to_noise_l2_ratio"
                    ),
                    "near_threshold_proxy": controller.get("near_threshold_proxy"),
                    "near_zero_proxy": controller.get("near_zero_proxy"),
                    "near_zero_adjusted": controller.get("near_zero_adjusted"),
                    "remaining_non_small_gradient_fraction": controller.get(
                        "remaining_non_small_gradient_fraction"
                    ),
                    "raw_dynamic_target_clipped": controller.get(
                        "raw_dynamic_target_clipped"
                    ),
                    "dynamic_target_clipped": controller.get(
                        "dynamic_target_clipped"
                    ),
                    "noisy_dynamic_target_clipped": controller.get(
                        "noisy_dynamic_target_clipped"
                    ),
                    "oracle_dynamic_target_clipped": controller.get(
                        "oracle_dynamic_target_clipped"
                    ),
                    "actual_minus_dynamic_target_clipped": controller.get(
                        "actual_minus_dynamic_target_clipped"
                    ),
                    "actual_target_absolute_error": controller.get(
                        "actual_target_absolute_error"
                    ),
                    "controller_error": controller.get("controller_error"),
                    "raw_log_step": controller.get("raw_log_step"),
                    "bounded_log_step": controller.get("bounded_log_step"),
                    "noisy_raw_log_step": controller.get("noisy_raw_log_step"),
                    "oracle_raw_log_step": controller.get("oracle_raw_log_step"),
                    "noisy_next_clip_threshold": controller.get(
                        "noisy_next_clip_threshold"
                    ),
                    "oracle_next_clip_threshold": controller.get(
                        "oracle_next_clip_threshold"
                    ),
                    "noisy_minus_oracle_raw_log_step": controller.get(
                        "noisy_minus_oracle_raw_log_step"
                    ),
                    "noisy_oracle_log_threshold_error": controller.get(
                        "noisy_oracle_log_threshold_error"
                    ),
                    "update_direction_agrees": controller.get(
                        "update_direction_agrees"
                    ),
                    "cdf_error_mae": controller.get("cdf_error_mae"),
                    "cdf_error_rmse": controller.get("cdf_error_rmse"),
                    "noisy_cdf_out_of_range_count": controller.get(
                        "noisy_cdf_out_of_range_count"
                    ),
                    "noisy_cdf_out_of_range_fraction": controller.get(
                        "noisy_cdf_out_of_range_fraction"
                    ),
                    "noisy_adjacent_monotonicity_violations": controller.get(
                        "noisy_adjacent_monotonicity_violations"
                    ),
                    "exact_adjacent_monotonicity_violations": controller.get(
                        "exact_adjacent_monotonicity_violations"
                    ),
                    "gamma_clamped_low": controller.get("gamma_clamped_low"),
                    "gamma_clamped_high": controller.get("gamma_clamped_high"),
                    "log_step_was_bounded": controller.get(
                        "log_step_was_bounded"
                    ),
                    "hit_min_clip_norm": controller.get("hit_min_clip_norm"),
                    "hit_max_clip_norm": controller.get("hit_max_clip_norm"),
                    "validation_loss": validation_loss,
                    "round_total_elapsed_seconds": summary.get(
                        "total_elapsed_seconds"
                    ),
                    "round_training_elapsed_seconds": summary.get(
                        "training_elapsed_seconds"
                    ),
                    "cuda_max_memory_allocated_bytes": resources.get(
                        "cuda_max_memory_allocated_bytes"
                    ),
                    "cuda_max_memory_reserved_bytes": resources.get(
                        "cuda_max_memory_reserved_bytes"
                    ),
                    "host_max_rss_bytes": resources.get("host_max_rss_bytes"),
                }
            )
        rows.append(
            {
                "stage": arm["stage"],
                "arm_id": arm["arm_id"],
                "method": arm["method"],
                "privacy_label": summary.get("privacy_label"),
                "controller_input": controller_root.get("controller_input"),
                "model": model,
                "seed": arm["seed"],
                "round": round_index,
                "group": "any",
                "initial_C": arm["initial_clip_norm"],
                "base_target_clipped_fraction_beta": arm["slaclip_beta"],
                "actual_clipped_count": summary.get("any_group_clipped_count"),
                "actual_clipped_fraction": summary.get("any_group_clipped_fraction"),
                "would_clip_count": summary.get("any_group_would_clip_count"),
                "would_clip_fraction": summary.get("any_group_would_clip_fraction"),
                "validation_loss": validation_loss,
                "round_total_elapsed_seconds": summary.get("total_elapsed_seconds"),
                "round_training_elapsed_seconds": summary.get(
                    "training_elapsed_seconds"
                ),
                "cuda_max_memory_allocated_bytes": resources.get(
                    "cuda_max_memory_allocated_bytes"
                ),
                "cuda_max_memory_reserved_bytes": resources.get(
                    "cuda_max_memory_reserved_bytes"
                ),
                "host_max_rss_bytes": resources.get("host_max_rss_bytes"),
            }
        )
    return rows


def _status_completed(
    campaign_root: Path, manifest: Mapping[str, Any], arm: Mapping[str, Any]
) -> bool:
    path = campaign_root / "arm-status" / f"{arm['arm_id']}.json"
    if not path.is_file():
        return False
    try:
        value = full.load_object(path, "arm status")
    except RuntimeError:
        return False
    return (
        value.get("status") == "COMPLETED"
        and value.get("runtime_manifest_sha256") == manifest["manifest_sha256"]
    )


def _write_trajectories(
    campaign_root: Path,
    stages: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
) -> None:
    fixed_rows: list[dict[str, Any]] = []
    slaclip_rows: list[dict[str, Any]] = []
    for manifest, arms in stages:
        for arm in arms:
            if not _status_completed(campaign_root, manifest, arm):
                continue
            target = fixed_rows if arm["method"] == FIXED_METHOD else slaclip_rows
            target.extend(_round_trajectory_rows(campaign_root, arm))
    full.atomic_csv(campaign_root / "fixed_trajectory.csv", fixed_rows, TRAJECTORY_COLUMNS)
    full.atomic_csv(
        campaign_root / "slaclip_trajectory.csv", slaclip_rows, TRAJECTORY_COLUMNS
    )


def _columns(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> list[str]:
    names = {str(name) for row in rows for name in row}
    ordered = [name for name in preferred if name in names]
    ordered.extend(sorted(names - set(ordered)))
    return ordered


def _paired_confirmation_rows(
    metric_rows: Sequence[Mapping[str, Any]],
    confirmation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_arm = {str(row["arm_id"]): row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for adaptive in confirmation["arms"]:
        if adaptive["method"] != SLACLIP_METHOD:
            continue
        candidate = by_arm.get(str(adaptive["arm_id"]))
        reference = by_arm.get(str(adaptive["reference_arm_id"]))
        if candidate is None or reference is None:
            continue
        if (
            candidate["model"] != reference["model"]
            or int(candidate["seed"]) != int(reference["seed"])
        ):
            raise RuntimeError("confirmation pair identity differs")
        for digest in (
            "sample_schedule_sha256",
            "supervision_schedule_sha256",
            "private_key_commitment",
            "rng_domain",
        ):
            if candidate.get(digest) != reference.get(digest):
                raise RuntimeError(
                    f"confirmation schedule differs: {candidate['model']}/{candidate['seed']}/{digest}"
                )
        if (
            candidate.get("pair_noise_across_methods") is not True
            or reference.get("pair_noise_across_methods") is not True
        ):
            raise RuntimeError("confirmation pair-noise flag differs")
        rows.append(
            {
                "model": candidate["model"],
                "seed": candidate["seed"],
                "fixed_arm_id": reference["arm_id"],
                "slaclip_arm_id": candidate["arm_id"],
                "selected_fixed_C": reference["initial_clip_norm"],
                "selected_slaclip_initial_C": candidate["initial_clip_norm"],
                "selected_beta": candidate["slaclip_beta"],
                "fixed_final_loss": reference["final_loss"],
                "slaclip_final_loss": candidate["final_loss"],
                "final_loss_difference_slaclip_minus_fixed": (
                    float(candidate["final_loss"]) - float(reference["final_loss"])
                ),
                "fixed_best_loss": reference["best_loss"],
                "slaclip_best_loss": candidate["best_loss"],
                "best_loss_difference_slaclip_minus_fixed": (
                    float(candidate["best_loss"]) - float(reference["best_loss"])
                ),
                "fixed_normalized_loss_auc": reference["normalized_loss_auc"],
                "slaclip_normalized_loss_auc": candidate["normalized_loss_auc"],
                "normalized_loss_auc_difference_slaclip_minus_fixed": (
                    float(candidate["normalized_loss_auc"])
                    - float(reference["normalized_loss_auc"])
                ),
                "fixed_actual_clipped_fraction": reference["actual_clipped_fraction"],
                "slaclip_actual_clipped_fraction": candidate["actual_clipped_fraction"],
                "actual_clipped_fraction_difference_slaclip_minus_fixed": (
                    float(candidate["actual_clipped_fraction"])
                    - float(reference["actual_clipped_fraction"])
                ),
                "fixed_elapsed_seconds": reference["elapsed_seconds"],
                "slaclip_elapsed_seconds": candidate["elapsed_seconds"],
                "sample_schedule_sha256": candidate["sample_schedule_sha256"],
                "supervision_schedule_sha256": candidate["supervision_schedule_sha256"],
                "pair_noise_across_methods": True,
                "rng_domain": candidate["rng_domain"],
                "private_key_commitment": candidate["private_key_commitment"],
            }
        )
    return rows


def _confirmation_aggregate(
    paired_rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    difference_metrics = (
        "final_loss_difference_slaclip_minus_fixed",
        "best_loss_difference_slaclip_minus_fixed",
        "normalized_loss_auc_difference_slaclip_minus_fixed",
        "actual_clipped_fraction_difference_slaclip_minus_fixed",
    )
    expected_seeds = {int(value) for value in spec["confirmation"]["seeds"]}
    for model in EXPECTED_MODELS:
        rows = [row for row in paired_rows if row["model"] == model]
        if rows and {int(row["seed"]) for row in rows} != expected_seeds:
            raise RuntimeError(f"confirmation paired seeds are incomplete for {model}")
        output: dict[str, Any] = {
            "model": model,
            "seed_count": len(rows),
        }
        for metric in difference_metrics:
            values = [float(row[metric]) for row in rows]
            for name, value in full.paired_inference(values).items():
                output[f"{metric}_{name}"] = value
        aggregates.append(output)
    p_key = "final_loss_difference_slaclip_minus_fixed_exact_sign_flip_p"
    valid = sorted(
        (
            (index, float(row[p_key]))
            for index, row in enumerate(aggregates)
            if isinstance(row.get(p_key), (int, float))
        ),
        key=lambda item: item[1],
    )
    running = 0.0
    for rank, (index, p_value) in enumerate(valid, start=1):
        running = max(running, min(1.0, (len(valid) - rank + 1) * p_value))
        aggregates[index]["final_loss_difference_slaclip_minus_fixed_holm_p"] = running
    rule = spec["confirmation"]["success_rule"]
    for output in aggregates:
        output["success_mean_final_loss_improved"] = (
            isinstance(output.get("final_loss_difference_slaclip_minus_fixed_mean"), (int, float))
            and float(output["final_loss_difference_slaclip_minus_fixed_mean"]) < 0.0
        )
        output["success_final_loss_ci95_excludes_zero"] = (
            isinstance(output.get("final_loss_difference_slaclip_minus_fixed_ci95_high"), (int, float))
            and float(output["final_loss_difference_slaclip_minus_fixed_ci95_high"]) < 0.0
        )
        output["success_holm_p_below_alpha"] = (
            isinstance(output.get("final_loss_difference_slaclip_minus_fixed_holm_p"), (int, float))
            and float(output["final_loss_difference_slaclip_minus_fixed_holm_p"])
            < float(rule["holm_alpha"])
        )
        output["success_auc_not_worse"] = (
            isinstance(output.get("normalized_loss_auc_difference_slaclip_minus_fixed_mean"), (int, float))
            and float(output["normalized_loss_auc_difference_slaclip_minus_fixed_mean"])
            <= 0.0
        )
        output["success_seed_win_fraction"] = (
            isinstance(output.get("final_loss_difference_slaclip_minus_fixed_negative_fraction"), (int, float))
            and float(output["final_loss_difference_slaclip_minus_fixed_negative_fraction"])
            >= float(rule["minimum_seed_win_fraction"])
        )
        output["primary_success"] = all(
            output[name]
            for name in (
                "success_mean_final_loss_improved",
                "success_final_loss_ci95_excludes_zero",
                "success_holm_p_below_alpha",
                "success_auc_not_worse",
                "success_seed_win_fraction",
            )
        )
    return aggregates


def aggregate_campaign(args: argparse.Namespace) -> None:
    campaign_root = args.campaign_root.resolve()
    spec_path = args.spec.resolve()
    spec = load_spec(spec_path)
    master = full.load_runtime(campaign_root / MASTER_RUNTIME_NAME)
    if master.get("spec_sha256") != full.sha256_file(spec_path):
        raise RuntimeError("aggregate spec differs from the master manifest")
    expected_master = _build_master_runtime(
        spec,
        spec_path,
        repository_sha=str(master.get("repository_sha")),
        input_manifest=Path(str(master.get("input_manifest_path"))),
        created_at_utc=str(master.get("created_at_utc")),
    )
    if master != expected_master:
        raise RuntimeError("aggregate master manifest differs from immutable inputs")
    stages: list[tuple[dict[str, Any], Sequence[Mapping[str, Any]]]] = [
        (master, master["arms"])
    ]
    fixed_lock: dict[str, Any] | None = None
    slaclip_lock: dict[str, Any] | None = None
    stage2: dict[str, Any] | None = None
    stage3: dict[str, Any] | None = None
    if (campaign_root / FIXED_LOCK_NAME).is_file():
        fixed_lock = full.load_object(campaign_root / FIXED_LOCK_NAME, "fixed selection lock")
        _validate_lock(fixed_lock, "fixed selection lock")
        _validate_fixed_lock_identity(fixed_lock, master, spec)
        _verify_beta_calibration_artifact(campaign_root, fixed_lock)
        stage2_path = _ensure_stage2_manifest(
            campaign_root,
            spec_path,
            spec,
            master,
            fixed_lock,
        )
        stage2 = full.load_runtime(stage2_path)
        stages.append((stage2, stage2["arms"]))
    if (campaign_root / SLACLIP_LOCK_NAME).is_file():
        if fixed_lock is None or stage2 is None:
            raise RuntimeError("SlaClip lock exists before the fixed stage is locked")
        slaclip_lock = full.load_object(campaign_root / SLACLIP_LOCK_NAME, "SlaClip selection lock")
        _validate_lock(slaclip_lock, "SlaClip selection lock")
        _validate_slaclip_lock_identity(slaclip_lock, master, fixed_lock, stage2, spec)
        stage3_path = _ensure_stage3_manifest(
            campaign_root,
            spec_path,
            spec,
            master,
            fixed_lock,
            slaclip_lock,
        )
        stage3 = full.load_runtime(stage3_path)
        stages.append((stage3, stage3["arms"]))

    status_counts = {"COMPLETED": 0, "FAILED": 0, "CHECKPOINTED_STOP": 0, "NOT_STARTED": 0, "OTHER": 0}
    metric_rows: list[dict[str, Any]] = []
    for manifest, arms in stages:
        for arm in arms:
            status_path = campaign_root / "arm-status" / f"{arm['arm_id']}.json"
            status = "NOT_STARTED"
            if status_path.is_file():
                status = str(full.load_object(status_path, "arm status").get("status", "OTHER"))
            status_counts[status if status in status_counts else "OTHER"] += 1
            if status == "COMPLETED":
                metric_rows.append(_metric_row(campaign_root, manifest, arm))
    expected_total = int(spec["expected_stage_arm_counts"]["total"])
    if args.require_complete:
        if fixed_lock is None or slaclip_lock is None or stage2 is None or stage3 is None:
            raise RuntimeError("complete aggregate requires both immutable selection locks")
        if len(metric_rows) != expected_total or status_counts["COMPLETED"] != expected_total:
            raise RuntimeError(
                f"complete aggregate has {len(metric_rows)} model arms, expected {expected_total}"
            )
        _verify_locked_evidence(campaign_root, master, fixed_lock["source_evidence"])
        _verify_locked_evidence(campaign_root, stage2, slaclip_lock["source_evidence"])

    paired_rows = (
        _paired_confirmation_rows(metric_rows, stage3)
        if stage3 is not None
        else []
    )
    confirmation_aggregate = _confirmation_aggregate(paired_rows, spec)
    if args.require_complete:
        expected_pairs = len(EXPECTED_MODELS) * len(spec["confirmation"]["seeds"])
        if len(paired_rows) != expected_pairs:
            raise RuntimeError(
                f"confirmation has {len(paired_rows)} pairs, expected {expected_pairs}"
            )
        if any(row["seed_count"] != len(spec["confirmation"]["seeds"]) for row in confirmation_aggregate):
            raise RuntimeError("confirmation aggregate seed count differs")

    full.atomic_csv(
        campaign_root / "campaign_metrics.csv",
        metric_rows,
        _columns(
            metric_rows,
            (
                "stage",
                "arm_id",
                "family",
                "method",
                "model",
                "seed",
                "initial_clip_norm",
                "slaclip_beta",
                "final_loss",
                "best_loss",
                "normalized_loss_auc",
                "actual_clipped_fraction",
            ),
        ),
    )
    full.atomic_csv(
        campaign_root / "confirmation_paired_metrics.csv",
        paired_rows,
        _columns(
            paired_rows,
            (
                "model",
                "seed",
                "selected_fixed_C",
                "selected_slaclip_initial_C",
                "selected_beta",
                "final_loss_difference_slaclip_minus_fixed",
                "normalized_loss_auc_difference_slaclip_minus_fixed",
            ),
        ),
    )
    full.atomic_csv(
        campaign_root / "confirmation_aggregate_metrics.csv",
        confirmation_aggregate,
        _columns(
            confirmation_aggregate,
            (
                "model",
                "seed_count",
                "final_loss_difference_slaclip_minus_fixed_mean",
                "final_loss_difference_slaclip_minus_fixed_ci95_low",
                "final_loss_difference_slaclip_minus_fixed_ci95_high",
                "final_loss_difference_slaclip_minus_fixed_negative_fraction",
                "final_loss_difference_slaclip_minus_fixed_exact_sign_flip_p",
                "final_loss_difference_slaclip_minus_fixed_holm_p",
                "primary_success",
            ),
        ),
    )
    _write_trajectories(campaign_root, stages)

    paper_anchor: dict[str, Any] = {}
    if fixed_lock is not None:
        for model in EXPECTED_MODELS:
            ranking = fixed_lock["models"][model]["ordered_fixed_C_candidates"]
            selected = ranking[0]
            c10 = next(row for row in ranking if float(row["fixed_C"]) == 10.0)
            paper_anchor[model] = {
                "selected_fixed_C": selected["fixed_C"],
                "selected_mean_final_loss": selected["mean_final_loss"],
                "paper_C10_mean_final_loss": c10["mean_final_loss"],
                "selected_minus_paper_C10_mean_final_loss": (
                    selected["mean_final_loss"] - c10["mean_final_loss"]
                ),
            }
    complete = (
        fixed_lock is not None
        and slaclip_lock is not None
        and stage3 is not None
        and len(metric_rows) == expected_total
        and len(paired_rows)
        == len(EXPECTED_MODELS) * len(spec["confirmation"]["seeds"])
    )
    summary = {
        "schema_version": 1,
        "status": "COMPLETED" if complete else "IN_PROGRESS",
        "campaign_name": master["campaign_name"],
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "fixed_selection_lock_sha256": fixed_lock.get("lock_sha256") if fixed_lock else None,
        "slaclip_selection_lock_sha256": slaclip_lock.get("lock_sha256") if slaclip_lock else None,
        "expected_total_model_arms": expected_total,
        "completed_model_arms": len(metric_rows),
        "status_counts_for_materialized_arms": status_counts,
        "materialized_stage_count": len(stages),
        "confirmation_paired_rows": len(paired_rows),
        "selected_models": slaclip_lock.get("models") if slaclip_lock else None,
        "fixed_vs_paper_C10_development": paper_anchor,
        "confirmation": confirmation_aggregate,
        "scientific_boundary": spec["scientific_boundary"],
        "warning": (
            "Development selections are not test evidence. Confirmation uses "
            "disjoint training seeds but the same internal LM-loss holdout; it "
            "is not an external downstream benchmark or a certified DP claim."
        ),
        "updated_at_utc": full.utc_now(),
    }
    full.atomic_json(campaign_root / "campaign_summary.json", summary)
    if args.require_complete and not complete:
        raise RuntimeError("strict final aggregate did not reach COMPLETED")
    print(f"campaign_summary={campaign_root / 'campaign_summary.json'}")


def validate_spec_command(args: argparse.Namespace) -> None:
    spec = load_spec(args.spec.resolve())
    fixed_arms = fixed_stage_arms(spec)
    print(
        json.dumps(
            {
                "status": "VALID",
                "spec_sha256": full.sha256_file(args.spec.resolve()),
                "stage1_model_arm_count": len(fixed_arms),
                "stage2_model_arm_count": spec["expected_stage_arm_counts"][SLACLIP_STAGE],
                "stage3_model_arm_count": spec["expected_stage_arm_counts"][CONFIRMATION_STAGE],
                "total_model_arm_count": spec["expected_stage_arm_counts"]["total"],
                "models": list(EXPECTED_MODELS),
                "K": spec["common"]["slaclip_num_slots"],
                "excluded_method": "SlaClip-Q",
            },
            indent=2,
            sort_keys=True,
        )
    )


def print_waves(args: argparse.Namespace) -> None:
    runtime = full.load_runtime(args.manifest.resolve())
    arms = runtime["arms"]
    if len(arms) % 2:
        raise RuntimeError("stage manifest has an odd arm count")
    for index in range(0, len(arms), 2):
        print(f"{index}\t{index + 1}")


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-spec")
    validate.add_argument("--spec", type=Path, required=True)

    prepare = subparsers.add_parser("prepare")
    _add_identity_arguments(prepare)
    prepare.add_argument("--private-key", type=Path, required=True)
    prepare.add_argument("--resume", action="store_true")

    fixed = subparsers.add_parser("lock-fixed")
    _add_identity_arguments(fixed)

    adaptive = subparsers.add_parser("lock-slaclip")
    _add_identity_arguments(adaptive)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--campaign-root", type=Path, required=True)
    aggregate.add_argument("--spec", type=Path, required=True)
    aggregate.add_argument("--require-complete", action="store_true")

    waves = subparsers.add_parser("waves")
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
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
