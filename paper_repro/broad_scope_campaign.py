#!/usr/bin/env python3
"""Plan and analyse a broad, staged Full-SlaClip DP-LoRA study.

The module deliberately separates three activities that must not share test
information:

1. a fixed-C development scan for every training-domain/model setting;
2. derivation of five Full-SlaClip beta values from the selected fixed-C
   clipping/slack trajectories; and
3. fresh-seed paired confirmation plus clipping-regime analysis.

It does not implement SlaClip-Q.  ``beta`` always has the generalized Full
SlaClip meaning used by :mod:`paper_repro.slaclip`: the requested clipped
fraction is multiplied by the mass remaining after the near-zero endpoint has
been removed.  Consequently a total clipped-fraction target cannot be mapped
to beta from clipping telemetry alone; the matching near-zero telemetry is a
required input and the planner fails closed when it is missing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from paper_repro.slaclip import (
        build_slack_vector,
        normalize_noisy_slack,
        stationary_beta_from_exact_endpoints,
    )
except ModuleNotFoundError:  # direct-script execution
    from slaclip import (  # type: ignore[no-redef]
        build_slack_vector,
        normalize_noisy_slack,
        stationary_beta_from_exact_endpoints,
    )


SCHEMA_VERSION = 1
FIXED_METHOD = "paper_dp_lora"
FULL_SLACLIP_METHOD = "slaclip_dp_lora"
ALLOWED_MODEL_IDS = frozenset({"bert", "gpt2", "chatglm2", "llama2"})
ALLOWED_DOMAIN_IDS = frozenset({"meddialog", "slimpajama", "finance"})
REQUIRED_GROUPS = ("A", "B")
ENDPOINT_EPSILON = 1e-6


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], description: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{description} keys differ; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _positive_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _finite_number(value: Any, description: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = " finite and positive" if positive else " finite"
        raise ValueError(f"{description} must be{qualifier}")
    return result


def _unique(values: Sequence[Any], description: str) -> None:
    encoded = [canonical_bytes(item) for item in values]
    if len(encoded) != len(set(encoded)):
        raise ValueError(f"{description} contains duplicates")


def validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a broad-scope campaign specification."""

    raw_spec = dict(spec)
    supplied_fingerprint = raw_spec.pop("spec_sha256", None)
    _require_exact_keys(
        raw_spec,
        {
            "schema_version",
            "campaign_name",
            "description",
            "domains",
            "models",
            "common",
            "fixed_development",
            "adaptive_development",
            "confirmation",
            "analysis",
            "scientific_boundary",
        },
        "broad-scope specification",
    )
    spec = raw_spec
    if spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported broad-scope specification schema")
    if not isinstance(spec["campaign_name"], str) or not spec["campaign_name"]:
        raise ValueError("campaign_name must be a non-empty string")

    domains = spec["domains"]
    models = spec["models"]
    if not isinstance(domains, list) or not domains:
        raise ValueError("domains must be a non-empty list")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    _unique(domains, "domains")
    _unique(models, "models")
    domain_ids: list[str] = []
    for index, domain in enumerate(domains):
        if not isinstance(domain, dict):
            raise ValueError(f"domains[{index}] must be an object")
        _require_exact_keys(
            domain,
            {
                "id",
                "paper_training_role",
                "input_manifest_env",
                "paper_exactness",
                "required_evaluation_tasks",
            },
            f"domains[{index}]",
        )
        domain_id = domain["id"]
        if domain_id not in ALLOWED_DOMAIN_IDS:
            raise ValueError(f"unsupported paper domain: {domain_id!r}")
        if domain["paper_exactness"] not in {
            "public_reconstruction",
            "public_subset_reconstruction",
            "paper_source_not_released",
        }:
            raise ValueError(f"invalid paper_exactness for {domain_id}")
        tasks = domain["required_evaluation_tasks"]
        if not isinstance(tasks, list) or not tasks or any(
            not isinstance(task, str) or not task for task in tasks
        ):
            raise ValueError(f"{domain_id} evaluation tasks must be non-empty strings")
        _unique(tasks, f"{domain_id} evaluation tasks")
        domain_ids.append(domain_id)
    if set(domain_ids) != ALLOWED_DOMAIN_IDS:
        raise ValueError(
            "the broad paper-domain scope must include meddialog, slimpajama, and finance"
        )

    model_ids: list[str] = []
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"models[{index}] must be an object")
        _require_exact_keys(
            model,
            {
                "id",
                "paper_name",
                "manifest_key",
                "access",
                "minimum_vram_gib",
            },
            f"models[{index}]",
        )
        model_id = model["id"]
        if model_id not in ALLOWED_MODEL_IDS:
            raise ValueError(f"unsupported paper model: {model_id!r}")
        if model["access"] not in {"public", "gated_user_acceptance_required"}:
            raise ValueError(f"invalid access policy for {model_id}")
        _positive_int(model["minimum_vram_gib"], f"{model_id}.minimum_vram_gib")
        model_ids.append(model_id)
    if set(model_ids) != ALLOWED_MODEL_IDS:
        raise ValueError(
            "the broad paper-model scope must include bert, gpt2, chatglm2, and llama2"
        )

    common = spec["common"]
    _require_exact_keys(
        common,
        {
            "num_clients",
            "rounds",
            "batch_size",
            "noise_multiplier",
            "learning_rate",
            "rank",
            "max_seq_length",
            "delta",
            "slaclip_num_slots",
            "slaclip_c_min",
            "slaclip_c_max",
        },
        "common",
    )
    for key in ("num_clients", "rounds", "batch_size", "rank", "max_seq_length"):
        _positive_int(common[key], f"common.{key}")
    for key in (
        "noise_multiplier",
        "learning_rate",
        "delta",
        "slaclip_c_min",
        "slaclip_c_max",
    ):
        _finite_number(common[key], f"common.{key}", positive=True)
    if common["slaclip_num_slots"] != 5:
        raise ValueError("this campaign is Full SlaClip K=5 only")
    if float(common["slaclip_c_min"]) >= float(common["slaclip_c_max"]):
        raise ValueError("SlaClip threshold bounds are reversed")

    fixed = spec["fixed_development"]
    _require_exact_keys(fixed, {"clip_norms", "seeds"}, "fixed_development")
    if not isinstance(fixed["clip_norms"], list) or len(fixed["clip_norms"]) < 3:
        raise ValueError("fixed_development.clip_norms needs at least three values")
    clip_norms = [
        _finite_number(value, "fixed clip norm", positive=True)
        for value in fixed["clip_norms"]
    ]
    if clip_norms != sorted(clip_norms):
        raise ValueError("fixed clip norms must be strictly increasing")
    _unique(clip_norms, "fixed clip norms")
    if not isinstance(fixed["seeds"], list) or len(fixed["seeds"]) < 3:
        raise ValueError("fixed development requires at least three seeds")
    for seed in fixed["seeds"]:
        _positive_int(seed, "fixed development seed")
    _unique(fixed["seeds"], "fixed development seeds")

    adaptive = spec["adaptive_development"]
    _require_exact_keys(
        adaptive,
        {
            "trajectory_quantiles",
            "eta_values",
            "seeds",
            "target_mapping",
        },
        "adaptive_development",
    )
    quantiles = adaptive["trajectory_quantiles"]
    if not isinstance(quantiles, list) or len(quantiles) != 5:
        raise ValueError("exactly five trajectory quantiles are required")
    normalized_quantiles = [
        _finite_number(value, "trajectory quantile") for value in quantiles
    ]
    if any(not 0 <= value <= 1 for value in normalized_quantiles):
        raise ValueError("trajectory quantiles must lie in [0, 1]")
    if normalized_quantiles != sorted(normalized_quantiles):
        raise ValueError("trajectory quantiles must be increasing")
    _unique(normalized_quantiles, "trajectory quantiles")
    if adaptive["target_mapping"] != "beta_equals_total_target_over_remaining_non_small_mass":
        raise ValueError("unsupported Full SlaClip beta mapping")
    for key in ("eta_values", "seeds"):
        if not isinstance(adaptive[key], list) or not adaptive[key]:
            raise ValueError(f"adaptive_development.{key} must be non-empty")
        _unique(adaptive[key], f"adaptive_development.{key}")
    for eta in adaptive["eta_values"]:
        _finite_number(eta, "SlaClip eta", positive=True)
    for seed in adaptive["seeds"]:
        _positive_int(seed, "adaptive development seed")
    if set(fixed["seeds"]) & set(adaptive["seeds"]):
        raise ValueError("fixed and adaptive development seeds must be disjoint")

    confirmation = spec["confirmation"]
    _require_exact_keys(
        confirmation,
        {"seeds", "methods", "selection_lock_required"},
        "confirmation",
    )
    if confirmation["methods"] != [FIXED_METHOD, FULL_SLACLIP_METHOD]:
        raise ValueError("confirmation must pair tuned fixed DP-LoRA and Full SlaClip")
    if confirmation["selection_lock_required"] is not True:
        raise ValueError("confirmation must require an immutable selection lock")
    seeds = confirmation["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 10:
        raise ValueError("confirmation requires at least ten fresh seeds")
    for seed in seeds:
        _positive_int(seed, "confirmation seed")
    _unique(seeds, "confirmation seeds")
    if (set(fixed["seeds"]) | set(adaptive["seeds"])) & set(seeds):
        raise ValueError("confirmation seeds must be disjoint from all development seeds")

    analysis = spec["analysis"]
    _require_exact_keys(
        analysis,
        {
            "clipping_fraction_bin_edges",
            "minimum_seed_pairs_per_setting",
            "minimum_dataset_model_settings_per_bin",
            "primary_metric",
            "secondary_metrics",
        },
        "analysis",
    )
    edges = analysis["clipping_fraction_bin_edges"]
    if not isinstance(edges, list) or len(edges) < 3:
        raise ValueError("analysis needs at least two clipping-fraction bins")
    normalized_edges = [_finite_number(edge, "clipping bin edge") for edge in edges]
    if normalized_edges[0] != 0 or normalized_edges[-1] != 1:
        raise ValueError("clipping bins must cover exactly [0, 1]")
    if any(left >= right for left, right in zip(normalized_edges, normalized_edges[1:])):
        raise ValueError("clipping bin edges must be strictly increasing")
    _positive_int(
        analysis["minimum_seed_pairs_per_setting"],
        "minimum_seed_pairs_per_setting",
    )
    _positive_int(
        analysis["minimum_dataset_model_settings_per_bin"],
        "minimum_dataset_model_settings_per_bin",
    )
    if analysis["primary_metric"] != "final_loss_difference_slaclip_minus_fixed":
        raise ValueError("the primary metric must be the paired final-loss difference")
    if not isinstance(analysis["secondary_metrics"], list):
        raise ValueError("secondary_metrics must be a list")

    boundary = spec["scientific_boundary"]
    required_boundary = {
        "paper_result_reproduced",
        "end_to_end_dp_certified",
        "external_test_set_required",
        "full_slaclip_only",
        "universal_clipping_boundary_claim_allowed",
        "finance_training_data_warning",
    }
    _require_exact_keys(boundary, required_boundary, "scientific_boundary")
    if boundary["full_slaclip_only"] is not True:
        raise ValueError("SlaClip-Q is forbidden in this campaign")
    if boundary["universal_clipping_boundary_claim_allowed"] is not False:
        raise ValueError("a universal clipping threshold may not be assumed in advance")

    normalized = json.loads(json.dumps(spec))
    normalized["spec_sha256"] = fingerprint(spec)
    if (
        supplied_fingerprint is not None
        and supplied_fingerprint != normalized["spec_sha256"]
    ):
        raise ValueError("supplied broad-scope specification fingerprint is invalid")
    return normalized


def expand_fixed_development(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    validated = validate_spec(spec)
    common = validated["common"]
    arms: list[dict[str, Any]] = []
    for domain in validated["domains"]:
        for model in validated["models"]:
            for clip_norm in validated["fixed_development"]["clip_norms"]:
                for seed in validated["fixed_development"]["seeds"]:
                    arm = {
                        "stage": "fixed_development",
                        "domain": domain["id"],
                        "model": model["id"],
                        "method": FIXED_METHOD,
                        "clip_norm": float(clip_norm),
                        "seed": int(seed),
                        "input_manifest_env": domain["input_manifest_env"],
                        **common,
                    }
                    arm["rng_domain"] = (
                        f"broad-{domain['id']}-{model['id']}-"
                        f"c{format(float(clip_norm), '.12g')}-s{seed}"
                    )
                    arm["arm_id"] = arm["rng_domain"] + "-fixed"
                    arms.append(arm)
    return arms


def _float(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"baseline telemetry has invalid {key!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"baseline telemetry has non-finite {key!r}")
    return value


def _integer(row: Mapping[str, str], key: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"baseline telemetry has invalid {key!r}") from error
    return value


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _concomitant_quantile(
    pairs: Sequence[tuple[float, float]], q: float
) -> tuple[float, float]:
    """Interpolate a value and its paired covariate at a value quantile."""

    if not pairs:
        raise ValueError("cannot calculate a concomitant quantile of no pairs")
    ordered = sorted(pairs, key=lambda pair: (pair[0], pair[1]))
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return tuple(
        left * (1 - fraction) + right * fraction
        for left, right in zip(ordered[lower], ordered[upper])
    )  # type: ignore[return-value]


def derive_adaptive_plan(
    spec: Mapping[str, Any], rows: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    """Select fixed C and derive five groupwise betas per domain/model.

    Rows are round-level fixed-development telemetry.  They must include
    ``actual_clipped_fraction_{A,B}`` and
    ``near_zero_adjusted_fraction_{A,B}``; the latter prevents the common but
    incorrect substitution of total clipping fraction directly for Full
    SlaClip beta.
    """

    validated = validate_spec(spec)
    expected_seeds = set(validated["fixed_development"]["seeds"])
    expected_clip_norms = {
        float(value) for value in validated["fixed_development"]["clip_norms"]
    }
    grouped: dict[tuple[str, str, float, int], list[Mapping[str, str]]] = {}
    allowed_settings = {
        (domain["id"], model["id"])
        for domain in validated["domains"]
        for model in validated["models"]
    }
    for row in rows:
        if row.get("method") != FIXED_METHOD:
            raise ValueError("adaptive derivation accepts fixed DP-LoRA rows only")
        domain = str(row.get("domain", ""))
        model = str(row.get("model", ""))
        setting = (domain, model)
        if setting not in allowed_settings:
            raise ValueError(f"unexpected baseline setting: {setting}")
        clip_norm = _float(row, "clip_norm")
        seed = _integer(row, "seed")
        if clip_norm not in expected_clip_norms or seed not in expected_seeds:
            raise ValueError("baseline row is outside the frozen development matrix")
        round_index = _integer(row, "round")
        if round_index <= 0:
            raise ValueError("baseline telemetry round must be positive")
        for group in REQUIRED_GROUPS:
            clipped = _float(row, f"actual_clipped_fraction_{group}")
            near_zero = _float(row, f"near_zero_adjusted_fraction_{group}")
            if not 0 <= clipped <= 1 or not 0 <= near_zero < 1:
                raise ValueError("clipping and near-zero fractions are outside [0, 1]")
        _float(row, "final_loss")
        grouped.setdefault((domain, model, clip_norm, seed), []).append(row)

    selections: list[dict[str, Any]] = []
    adaptive_arms: list[dict[str, Any]] = []
    quantiles = [float(value) for value in validated["adaptive_development"]["trajectory_quantiles"]]
    for domain, model in sorted(allowed_settings):
        mean_final_loss_by_c: dict[float, float] = {}
        for clip_norm in sorted(expected_clip_norms):
            seed_losses = []
            for seed in sorted(expected_seeds):
                key = (domain, model, clip_norm, seed)
                trajectory = grouped.get(key)
                if not trajectory:
                    raise ValueError(f"missing fixed trajectory: {key}")
                final_round = max(_integer(row, "round") for row in trajectory)
                final_rows = [
                    row for row in trajectory if _integer(row, "round") == final_round
                ]
                if len(final_rows) != 1:
                    raise ValueError(f"fixed trajectory has duplicate final round: {key}")
                seed_losses.append(_float(final_rows[0], "final_loss"))
            mean_final_loss_by_c[clip_norm] = statistics.fmean(seed_losses)
        selected_c = min(
            mean_final_loss_by_c,
            key=lambda value: (mean_final_loss_by_c[value], value),
        )
        selected_rows = [
            row
            for (d, m, c, _), trajectory in grouped.items()
            if d == domain and m == model and c == selected_c
            for row in trajectory
        ]
        group_targets: dict[str, list[dict[str, Any]]] = {}
        for group in REQUIRED_GROUPS:
            paired_trajectory = [
                (
                    _float(row, f"actual_clipped_fraction_{group}"),
                    _float(row, f"near_zero_adjusted_fraction_{group}"),
                )
                for row in selected_rows
            ]
            targets = []
            for q in quantiles:
                # Keep the near-zero endpoint paired with the clipping-trajectory
                # quantile.  Dividing a clipping quantile by an unrelated median
                # endpoint would silently break the generalized Full-SlaClip
                # semantics when endpoint mass changes through training.
                total_target, near_zero_at_target = _concomitant_quantile(
                    paired_trajectory, q
                )
                remaining = 1.0 - near_zero_at_target
                if remaining <= 0:
                    raise ValueError(
                        "no non-small-gradient mass remains for "
                        f"{domain}/{model}/{group} at quantile {q}"
                    )
                unclamped_beta = total_target / remaining
                beta = min(1.0, max(0.0, unclamped_beta))
                targets.append(
                    {
                        "trajectory_quantile": q,
                        "total_clipped_fraction_target": total_target,
                        "near_zero_adjusted_fraction_at_target": near_zero_at_target,
                        "remaining_non_small_mass": remaining,
                        "unclamped_base_target_clipped_fraction_beta": unclamped_beta,
                        "base_target_clipped_fraction_beta": beta,
                        "target_was_clamped": beta != unclamped_beta,
                    }
                )
            group_targets[group] = targets
        selection = {
            "domain": domain,
            "model": model,
            "selected_fixed_clip_norm": selected_c,
            "mean_final_loss_by_clip_norm": {
                format(key, ".12g"): value
                for key, value in sorted(mean_final_loss_by_c.items())
            },
            "group_targets": group_targets,
            "source_row_count": len(selected_rows),
        }
        selection["selection_sha256"] = fingerprint(selection)
        selections.append(selection)
        for index, (target_a, target_b) in enumerate(
            zip(group_targets["A"], group_targets["B"])
        ):
            for eta in validated["adaptive_development"]["eta_values"]:
                for seed in validated["adaptive_development"]["seeds"]:
                    arm = {
                        "stage": "adaptive_development",
                        "domain": domain,
                        "model": model,
                        "method": FULL_SLACLIP_METHOD,
                        "initial_clip_norm": selected_c,
                        "slaclip_base_target_clipped_fraction_by_group": {
                            "A": target_a["base_target_clipped_fraction_beta"],
                            "B": target_b["base_target_clipped_fraction_beta"],
                        },
                        "target_total_clipped_fraction_by_group": {
                            "A": target_a["total_clipped_fraction_target"],
                            "B": target_b["total_clipped_fraction_target"],
                        },
                        "target_quantile_index": index,
                        "slaclip_eta": float(eta),
                        "seed": int(seed),
                        "selection_sha256": selection["selection_sha256"],
                        **validated["common"],
                    }
                    arm["rng_domain"] = (
                        f"broad-{domain}-{model}-adaptive-q{index}-"
                        f"e{format(float(eta), '.12g')}-s{seed}"
                    )
                    arm["arm_id"] = arm["rng_domain"] + "-slaclip"
                    adaptive_arms.append(arm)
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "ADAPTIVE_DEVELOPMENT_PLAN_LOCKED",
        "spec_sha256": validated["spec_sha256"],
        "selection_count": len(selections),
        "selections": selections,
        "adaptive_arm_count": len(adaptive_arms),
        "adaptive_arms": adaptive_arms,
        "beta_semantics": (
            "dynamic_target_clipped=beta*(1-near_zero_adjusted_fraction); "
            "beta is not the unadjusted total clipping fraction"
        ),
    }
    output["plan_sha256"] = fingerprint(output)
    return output


def _completed_final_loss(model_directory: Path, arm: Mapping[str, Any]) -> float:
    summary_path = model_directory / "final_summary.json"
    summary = load_object(summary_path, "fixed arm final summary")
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("method") != FIXED_METHOD
        or summary.get("model") != arm["model"]
    ):
        raise ValueError(f"fixed arm did not complete cleanly: {arm['arm_id']}")
    evaluation = summary.get("final_evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"fixed arm has no final evaluation: {arm['arm_id']}")
    return _finite_number(
        evaluation.get("loss"), f"{arm['arm_id']} final evaluation loss"
    )


def collect_fixed_telemetry(
    spec: Mapping[str, Any], campaign_root: Path
) -> list[dict[str, Any]]:
    """Recover exact Full-SlaClip endpoint telemetry from fixed-C shards.

    This is deliberately a development-only, NON-DP diagnostic.  It consumes
    per-client raw gradient norms and must never be described as a released DP
    statistic.  The resulting near-zero mass is necessary to map a desired
    *total* clipping fraction to generalized Full-SlaClip beta correctly.
    """

    validated = validate_spec(spec)
    root = campaign_root.resolve()
    rounds = int(validated["common"]["rounds"])
    num_slots = int(validated["common"]["slaclip_num_slots"])
    rows: list[dict[str, Any]] = []
    for arm in expand_fixed_development(spec):
        model_directory = root / "arms" / arm["arm_id"] / arm["model"]
        final_loss = _completed_final_loss(model_directory, arm)
        for round_index in range(1, rounds + 1):
            shard_path = (
                model_directory
                / "private_diagnostics"
                / "rounds"
                / f"round-{round_index:05d}.json"
            )
            shard = load_object(shard_path, "fixed round diagnostic shard")
            records = shard.get("client_records")
            round_summary = shard.get("round_summary")
            if (
                shard.get("round") != round_index
                or shard.get("model") != arm["model"]
                or shard.get("method") != FIXED_METHOD
                or not isinstance(records, list)
                or not records
                or not isinstance(round_summary, dict)
            ):
                raise ValueError(f"invalid fixed round shard: {shard_path}")
            row: dict[str, Any] = {
                "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
                "method": FIXED_METHOD,
                "domain": arm["domain"],
                "model": arm["model"],
                "clip_norm": arm["clip_norm"],
                "seed": arm["seed"],
                "round": round_index,
                "final_loss": final_loss,
            }
            for group in REQUIRED_GROUPS:
                raw_norms: list[float] = []
                for record in records:
                    try:
                        raw_norm = float(record["gradient_groups"][group]["raw_norm"])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"missing raw {group} norm in {shard_path}"
                        ) from error
                    if not math.isfinite(raw_norm) or raw_norm < 0:
                        raise ValueError(f"invalid raw {group} norm in {shard_path}")
                    raw_norms.append(raw_norm)
                signals = [
                    build_slack_vector(norm, float(arm["clip_norm"]), num_slots)
                    for norm in raw_norms
                ]
                signal_sum = [
                    math.fsum(vector[slot] for vector in signals)
                    for slot in range(num_slots)
                ]
                endpoints = normalize_noisy_slack(
                    signal_sum,
                    float(arm["clip_norm"]),
                    num_slots,
                    len(raw_norms),
                )
                stationary = stationary_beta_from_exact_endpoints(
                    float(arm["clip_norm"]),
                    float(endpoints[0]),
                    float(endpoints[-1]),
                    epsilon=ENDPOINT_EPSILON,
                )
                try:
                    clipped = float(round_summary[group]["clipped_fraction"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"missing actual {group} clipping fraction in {shard_path}"
                    ) from error
                if not 0 <= clipped <= 1:
                    raise ValueError(f"invalid clipping fraction in {shard_path}")
                row[f"exact_q_endpoint_1_{group}"] = float(endpoints[0])
                row[f"exact_r_endpoint_K_{group}"] = float(endpoints[-1])
                row[f"actual_clipped_fraction_{group}"] = clipped
                row[f"near_zero_adjusted_fraction_{group}"] = float(
                    stationary["near_zero_adjusted"]
                )
                row[f"stationary_beta_{group}"] = float(
                    stationary["stationary_beta"]
                )
            rows.append(row)
    return rows


def validate_input_index(
    spec: Mapping[str, Any], input_index: Mapping[str, Any]
) -> dict[str, Path]:
    validated = validate_spec(spec)
    _require_exact_keys(input_index, {"schema_version", "domains"}, "input index")
    if input_index["schema_version"] != 1:
        raise ValueError("unsupported broad input-index schema")
    domains = input_index["domains"]
    if not isinstance(domains, dict):
        raise ValueError("input-index domains must be an object")
    expected = {str(item["id"]) for item in validated["domains"]}
    if set(domains) != expected:
        raise ValueError(
            "input-index domain set differs; "
            f"missing={sorted(expected-set(domains))}, "
            f"extra={sorted(set(domains)-expected)}"
        )
    result: dict[str, Path] = {}
    for domain, raw_path in domains.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"input manifest path is invalid for {domain}")
        path = Path(raw_path)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"input manifest does not exist for {domain}: {path}")
        result[domain] = path.resolve()
    return result


def _arm_command(
    arm: Mapping[str, Any],
    *,
    repository: Path,
    python_bin: Path,
    input_manifest: Path,
    output_dir: Path,
    private_key: Path,
    calibration_lock_sha256: str | None = None,
) -> list[str]:
    command = [
        str(python_bin),
        str(repository / "paper_repro" / "train_federated.py"),
        "--input-manifest",
        str(input_manifest),
        "--output-dir",
        str(output_dir),
        "--models",
        str(arm["model"]),
        "--device",
        "cuda",
        "--method",
        str(arm["method"]),
        "--num-clients",
        str(arm["num_clients"]),
        "--rounds",
        str(arm["rounds"]),
        "--batch-size",
        str(arm["batch_size"]),
        "--noise-multiplier",
        str(arm["noise_multiplier"]),
        "--learning-rate",
        str(arm["learning_rate"]),
        "--rank",
        str(arm["rank"]),
        "--max-seq-length",
        str(arm["max_seq_length"]),
        "--delta",
        str(arm["delta"]),
        "--seed",
        str(arm["seed"]),
        "--private-rng-key",
        str(private_key),
        "--rng-domain",
        str(arm["rng_domain"]),
        "--acknowledge-non-dp-diagnostics",
    ]
    if arm["method"] == FIXED_METHOD:
        command.extend(["--clip-norm", str(arm["clip_norm"])])
    elif arm["method"] == FULL_SLACLIP_METHOD:
        targets = arm.get("slaclip_base_target_clipped_fraction_by_group")
        if not isinstance(targets, dict) or set(targets) != set(REQUIRED_GROUPS):
            raise ValueError(f"adaptive arm has invalid group targets: {arm['arm_id']}")
        if calibration_lock_sha256 is None:
            raise ValueError("adaptive execution requires the locked development plan")
        command.extend(
            [
                "--clip-norm",
                str(arm["initial_clip_norm"]),
                "--slaclip-eta",
                str(arm["slaclip_eta"]),
                "--slaclip-base-target-clipped-fraction-a",
                str(targets["A"]),
                "--slaclip-base-target-clipped-fraction-b",
                str(targets["B"]),
                "--slaclip-num-slots",
                str(arm["slaclip_num_slots"]),
                "--slaclip-c-min",
                str(arm["slaclip_c_min"]),
                "--slaclip-c-max",
                str(arm["slaclip_c_max"]),
                "--slaclip-baseline-calibration-lock-sha256",
                calibration_lock_sha256,
                "--acknowledge-slaclip-baseline-calibration-is-non-dp",
            ]
        )
    else:  # pragma: no cover - validated plans constrain this
        raise ValueError(f"unsupported arm method: {arm['method']}")
    if output_dir.exists():
        command.append("--resume")
    return command


def _completed_root_summary(path: Path, arm: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        value = load_object(path, "arm root final summary")
    except ValueError:
        return False
    models = value.get("models")
    return bool(
        value.get("status") == "COMPLETED"
        and value.get("method") == arm["method"]
        and isinstance(models, dict)
        and set(models) == {arm["model"]}
        and models[arm["model"]].get("status") == "COMPLETED"
    )


def run_stage(
    spec: Mapping[str, Any],
    *,
    stage: str,
    input_index: Mapping[str, Any],
    campaign_root: Path,
    repository: Path,
    python_bin: Path,
    private_key: Path,
    adaptive_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one sequential, resumable stage inside an existing GPU allocation."""

    validated = validate_spec(spec)
    manifests = validate_input_index(spec, input_index)
    repository = repository.resolve()
    python_bin = python_bin.resolve()
    private_key = private_key.resolve()
    if not (repository / "paper_repro" / "train_federated.py").is_file():
        raise ValueError("repository has no broad-scope trainer")
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise ValueError(f"Python executable is invalid: {python_bin}")
    if not private_key.is_file() or private_key.stat().st_size < 32:
        raise ValueError("private RNG key is missing or too short")
    if stage == "fixed":
        arms = expand_fixed_development(spec)
        plan_sha256 = None
    elif stage == "adaptive":
        if not isinstance(adaptive_plan, Mapping):
            raise ValueError("adaptive stage requires --adaptive-plan")
        if (
            adaptive_plan.get("status") != "ADAPTIVE_DEVELOPMENT_PLAN_LOCKED"
            or adaptive_plan.get("spec_sha256") != validated["spec_sha256"]
            or adaptive_plan.get("plan_sha256")
            != fingerprint({k: v for k, v in adaptive_plan.items() if k != "plan_sha256"})
        ):
            raise ValueError("adaptive plan identity is invalid")
        raw_arms = adaptive_plan.get("adaptive_arms")
        if not isinstance(raw_arms, list) or not raw_arms:
            raise ValueError("adaptive plan has no arms")
        arms = [dict(arm) for arm in raw_arms]
        plan_sha256 = str(adaptive_plan["plan_sha256"])
    else:
        raise ValueError("stage must be fixed or adaptive")

    root = campaign_root.resolve()
    (root / "arms").mkdir(parents=True, exist_ok=True, mode=0o700)
    (root / "arm-logs").mkdir(parents=True, exist_ok=True, mode=0o700)
    (root / "arm-status").mkdir(parents=True, exist_ok=True, mode=0o700)
    completed = 0
    for index, arm in enumerate(arms):
        arm_id = str(arm["arm_id"])
        output_dir = root / "arms" / arm_id
        root_summary = output_dir / "final_summary.json"
        status_path = root / "arm-status" / f"{arm_id}.json"
        if _completed_root_summary(root_summary, arm):
            completed += 1
            continue
        command = _arm_command(
            arm,
            repository=repository,
            python_bin=python_bin,
            input_manifest=manifests[str(arm["domain"])],
            output_dir=output_dir,
            private_key=private_key,
            calibration_lock_sha256=plan_sha256,
        )
        _write_json(
            status_path,
            {
                "status": "RUNNING",
                "stage": stage,
                "arm_index": index,
                "arm_count": len(arms),
                "arm_id": arm_id,
                "arm_sha256": fingerprint(arm),
                "spec_sha256": validated["spec_sha256"],
                "adaptive_plan_sha256": plan_sha256,
            },
        )
        stdout_path = root / "arm-logs" / f"{arm_id}.out"
        stderr_path = root / "arm-logs" / f"{arm_id}.err"
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            result = subprocess.run(
                command,
                cwd=repository,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if result.returncode != 0 or not _completed_root_summary(root_summary, arm):
            _write_json(
                status_path,
                {
                    "status": "FAILED_OR_CHECKPOINTED",
                    "stage": stage,
                    "arm_index": index,
                    "arm_count": len(arms),
                    "arm_id": arm_id,
                    "return_code": result.returncode,
                    "arm_sha256": fingerprint(arm),
                    "spec_sha256": validated["spec_sha256"],
                    "adaptive_plan_sha256": plan_sha256,
                },
            )
            raise RuntimeError(
                f"arm stopped before validated completion: {arm_id}; "
                f"see {stderr_path}"
            )
        completed += 1
        _write_json(
            status_path,
            {
                "status": "COMPLETED",
                "stage": stage,
                "arm_index": index,
                "arm_count": len(arms),
                "arm_id": arm_id,
                "arm_sha256": fingerprint(arm),
                "spec_sha256": validated["spec_sha256"],
                "adaptive_plan_sha256": plan_sha256,
            },
        )
    return {
        "status": "COMPLETED",
        "stage": stage,
        "arm_count": len(arms),
        "completed_arm_count": completed,
        "spec_sha256": validated["spec_sha256"],
        "adaptive_plan_sha256": plan_sha256,
    }


@dataclass(frozen=True)
class BinSummary:
    lower: float
    upper: float
    pair_count: int
    setting_count: int
    mean_delta: float | None
    median_delta: float | None
    win_fraction: float | None
    ci95_low: float | None
    ci95_high: float | None
    evidence_gate_passed: bool


def _mean_ci95(values: Sequence[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    # Deliberately normal-approximate for a descriptive regime plot.  Formal
    # setting-level confirmation retains exact paired randomization tests.
    margin = 1.96 * standard_error
    return mean - margin, mean + margin


def analyse_clipping_regimes(
    spec: Mapping[str, Any], rows: Sequence[Mapping[str, str]]
) -> list[BinSummary]:
    """Summarize paired Full-SlaClip effects by baseline clipping regime."""

    validated = validate_spec(spec)
    edges = [float(value) for value in validated["analysis"]["clipping_fraction_bin_edges"]]
    minimum_pairs = int(validated["analysis"]["minimum_seed_pairs_per_setting"])
    minimum_settings = int(
        validated["analysis"]["minimum_dataset_model_settings_per_bin"]
    )
    buckets: list[list[tuple[tuple[str, str], int, float]]] = [
        [] for _ in range(len(edges) - 1)
    ]
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        domain = str(row.get("domain", ""))
        model = str(row.get("model", ""))
        seed = _integer(row, "seed")
        key = (domain, model, seed)
        if key in seen:
            raise ValueError(f"duplicate paired result: {key}")
        seen.add(key)
        clipped = _float(row, "baseline_actual_clipped_fraction")
        delta = _float(row, "final_loss_difference_slaclip_minus_fixed")
        if not 0 <= clipped <= 1:
            raise ValueError("baseline clipping fraction lies outside [0, 1]")
        index = len(edges) - 2 if clipped == 1 else next(
            (
                i
                for i, (lower, upper) in enumerate(zip(edges, edges[1:]))
                if lower <= clipped < upper
            ),
            -1,
        )
        if index < 0:
            raise ValueError("could not assign baseline clipping fraction to a bin")
        buckets[index].append(((domain, model), seed, delta))

    summaries = []
    for lower, upper, bucket in zip(edges, edges[1:], buckets):
        deltas = [delta for _, _, delta in bucket]
        setting_counts: dict[tuple[str, str], int] = {}
        for setting, _, _ in bucket:
            setting_counts[setting] = setting_counts.get(setting, 0) + 1
        qualified_settings = {
            setting for setting, count in setting_counts.items() if count >= minimum_pairs
        }
        ci_low, ci_high = _mean_ci95(deltas)
        summaries.append(
            BinSummary(
                lower=lower,
                upper=upper,
                pair_count=len(deltas),
                setting_count=len(setting_counts),
                mean_delta=statistics.fmean(deltas) if deltas else None,
                median_delta=statistics.median(deltas) if deltas else None,
                win_fraction=(sum(delta < 0 for delta in deltas) / len(deltas))
                if deltas
                else None,
                ci95_low=ci_low,
                ci95_high=ci_high,
                evidence_gate_passed=len(qualified_settings) >= minimum_settings,
            )
        )
    return summaries


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"could not read CSV: {path}") from error


def _write_json(path: Path | None, value: Any) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if path is None:
        print(encoded, end="")
        return
    _atomic_write(path, encoded.encode("utf-8"))


def _atomic_write(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty telemetry CSV")
    fieldnames = list(rows[0])
    for row in rows:
        if set(row) != set(fieldnames):
            raise ValueError("telemetry rows have inconsistent columns")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-spec", "expand-fixed"):
        command = subparsers.add_parser(name)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--output", type=Path)
    derive = subparsers.add_parser("derive-adaptive")
    derive.add_argument("--spec", type=Path, required=True)
    derive.add_argument("--baseline-telemetry", type=Path, required=True)
    derive.add_argument("--output", type=Path)
    collect = subparsers.add_parser("collect-fixed")
    collect.add_argument("--spec", type=Path, required=True)
    collect.add_argument("--campaign-root", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    analyse = subparsers.add_parser("analyse-regimes")
    analyse.add_argument("--spec", type=Path, required=True)
    analyse.add_argument("--paired-results", type=Path, required=True)
    analyse.add_argument("--output", type=Path)
    run = subparsers.add_parser("run-stage")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--stage", choices=["fixed", "adaptive"], required=True)
    run.add_argument("--input-index", type=Path, required=True)
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--repository", type=Path, required=True)
    run.add_argument("--python-bin", type=Path, required=True)
    run.add_argument("--private-key", type=Path, required=True)
    run.add_argument("--adaptive-plan", type=Path)
    run.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_object(args.spec.resolve(), "broad-scope specification")
    if args.command == "validate-spec":
        validated = validate_spec(spec)
        output: Any = {
            "status": "VALID",
            "spec_sha256": validated["spec_sha256"],
            "fixed_development_arm_count": len(expand_fixed_development(spec)),
            "dataset_model_setting_count": len(validated["domains"])
            * len(validated["models"]),
            "full_slaclip_only": True,
        }
    elif args.command == "expand-fixed":
        arms = expand_fixed_development(spec)
        output = {"arm_count": len(arms), "arms": arms}
    elif args.command == "derive-adaptive":
        output = derive_adaptive_plan(spec, _read_csv(args.baseline_telemetry))
    elif args.command == "collect-fixed":
        rows = collect_fixed_telemetry(spec, args.campaign_root)
        _write_csv(args.output.resolve(), rows)
        print(f"telemetry_rows={len(rows)}")
        print(f"telemetry_csv={args.output.resolve()}")
        return
    elif args.command == "analyse-regimes":
        output = {
            "schema_version": SCHEMA_VERSION,
            "interpretation": (
                "descriptive clipping-regime evidence; a universal transition "
                "threshold requires multiple qualified dataset/model settings"
            ),
            "bins": [
                summary.__dict__
                for summary in analyse_clipping_regimes(
                    spec, _read_csv(args.paired_results)
                )
            ],
        }
    elif args.command == "run-stage":
        adaptive_plan = (
            load_object(args.adaptive_plan.resolve(), "adaptive development plan")
            if args.adaptive_plan
            else None
        )
        output = run_stage(
            spec,
            stage=args.stage,
            input_index=load_object(args.input_index.resolve(), "broad input index"),
            campaign_root=args.campaign_root,
            repository=args.repository,
            python_bin=args.python_bin,
            private_key=args.private_key,
            adaptive_plan=adaptive_plan,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _write_json(args.output.resolve() if args.output else None, output)


if __name__ == "__main__":
    main()
