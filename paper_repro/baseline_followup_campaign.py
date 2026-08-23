#!/usr/bin/env python3
"""Hash-locked, fixed-only follow-up to the 27-arm DP-LoRA baseline.

The campaign consumes only rounds 2--10 of the completed paper-default
baseline's per-client raw-norm telemetry.  It derives three thresholds for
LoRA-A and LoRA-B independently, evaluates their Cartesian product plus the
exact ``(10, 10)`` paper-default anchor, and freezes the top three candidates
for every dataset/model setting.  This module contains no adaptive method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from paper_repro import broad_scope_campaign as broad
    from paper_repro import full_slaclip_campaign as full
except ModuleNotFoundError:  # direct-script execution
    import broad_scope_campaign as broad  # type: ignore[no-redef]
    import full_slaclip_campaign as full  # type: ignore[no-redef]


SCHEMA_VERSION = 1
PLAN_NAME = "runtime-manifest.json"
SELECTION_LOCK_NAME = "strong-groupwise-fixed-selection.lock.json"
SUMMARY_NAME = "campaign_summary.json"
METRICS_NAME = "campaign_metrics.csv"
FIXED_METHOD = "paper_dp_lora"
DOMAINS = ("meddialog", "slimpajama", "finance")
MODELS = ("bert", "gpt2", "chatglm2")
GROUPS = ("A", "B")
SOURCE_SEEDS = (1200, 1201, 1202)
SOURCE_FILES = {
    "campaign_summary": "campaign-summary.json",
    "telemetry_manifest": "telemetry/baseline_telemetry_manifest.json",
    "arm_integrity": "telemetry/baseline_arm_integrity.json",
    "round_telemetry": "telemetry/baseline_round_telemetry.csv",
    "client_telemetry": "telemetry/baseline_client_telemetry.csv",
    "evaluation_telemetry": "telemetry/baseline_evaluation_telemetry.csv",
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def load_spec(path: Path) -> dict[str, Any]:
    spec = full.load_object(path.resolve(), "baseline follow-up specification")
    _exact_keys(
        spec,
        {
            "schema_version", "campaign_name", "description",
            "expected_arm_count", "source_calibration", "development",
            "paper_defaults", "scientific_boundary",
        },
        "baseline follow-up specification",
    )
    if spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported baseline follow-up schema")
    if spec["expected_arm_count"] != 180:
        raise ValueError("follow-up campaign must contain exactly 180 arms")

    source = spec["source_calibration"]
    _exact_keys(
        source,
        {
            "round_min", "round_max", "groups", "raw_norm_quantiles",
            "anchor_clip_norm_by_group", "source_privacy_label",
        },
        "source_calibration",
    )
    if (
        source["round_min"] != 2
        or source["round_max"] != 10
        or tuple(source["groups"]) != GROUPS
        or source["raw_norm_quantiles"] != [0.05, 0.2, 0.5]
        or source["anchor_clip_norm_by_group"] != {"A": 10.0, "B": 10.0}
        or source["source_privacy_label"] != "NON_DP_PRIVATE_DIAGNOSTIC"
    ):
        raise ValueError("source calibration policy differs")

    development = spec["development"]
    _exact_keys(
        development,
        {"domains", "models", "seeds", "top_k_per_setting", "selection_rule"},
        "development",
    )
    if (
        tuple(development["domains"]) != DOMAINS
        or tuple(development["models"]) != MODELS
        or development["seeds"] != [1300, 1301]
        or development["top_k_per_setting"] != 3
    ):
        raise ValueError("development matrix differs")
    expected_rule = [
        "lowest_mean_final_internal_validation_loss",
        "lowest_mean_normalized_internal_validation_loss_auc",
        "lowest_final_loss_sample_std",
        "lowest_groupwise_noise_scale_l2",
        "smaller_C_A",
        "smaller_C_B",
    ]
    if development["selection_rule"] != expected_rule:
        raise ValueError("selection rule differs")

    defaults = spec["paper_defaults"]
    expected_defaults = {
        "method": FIXED_METHOD,
        "num_clients": 5,
        "rounds": 50,
        "batch_size": 8,
        "noise_multiplier": 2.0,
        "learning_rate": 5e-4,
        "rank": 512,
        "max_seq_length": 128,
        "max_validation_records": 512,
        "eval_every": 10,
        "checkpoint_every": 10,
        "data_split_seed": 1729,
        "evaluation_seed": 2718,
        "delta": 1e-5,
    }
    if defaults != expected_defaults:
        raise ValueError("paper-default training contract differs")
    for name in (
        "num_clients", "rounds", "batch_size", "rank", "max_seq_length",
        "max_validation_records", "eval_every", "checkpoint_every",
        "data_split_seed", "evaluation_seed",
    ):
        _positive_int(defaults[name], f"paper_defaults.{name}")
    for name in ("noise_multiplier", "learning_rate", "delta"):
        _finite(defaults[name], f"paper_defaults.{name}", positive=True)

    boundary = spec["scientific_boundary"]
    _exact_keys(
        boundary,
        {
            "analysis_role", "full_slaclip_run", "slaclip_q_run",
            "adaptive_threshold_run", "confirmation_data_used",
            "external_test_set_used", "paper_result_reproduced",
            "end_to_end_dp_certified", "single_allocation",
            "nested_sbatch_or_array", "all_candidates_reported",
        },
        "scientific_boundary",
    )
    if (
        boundary.get("analysis_role")
        != "development_only_groupwise_fixed_comparator_discovery"
        or boundary.get("full_slaclip_run") is not False
        or boundary.get("slaclip_q_run") is not False
        or boundary.get("adaptive_threshold_run") is not False
        or boundary.get("confirmation_data_used") is not False
        or boundary.get("external_test_set_used") is not False
        or boundary.get("paper_result_reproduced") is not False
        or boundary.get("end_to_end_dp_certified") is not False
        or boundary.get("single_allocation") is not True
        or boundary.get("nested_sbatch_or_array") is not False
        or boundary.get("all_candidates_reported") is not True
    ):
        raise ValueError("scientific boundary differs")
    return spec


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"could not read {label}: {path}") from error
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot derive a threshold from no raw norms")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _source_paths(root: Path) -> dict[str, Path]:
    resolved = root.resolve()
    paths = {name: resolved / relative for name, relative in SOURCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"completed baseline artifacts are missing: {missing}")
    return paths


def _validate_input_index(path: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    value = full.load_object(path.resolve(), "baseline input index")
    _exact_keys(value, {"schema_version", "domains"}, "baseline input index")
    if value["schema_version"] != 1 or not isinstance(value["domains"], dict):
        raise ValueError("baseline input index schema differs")
    if set(value["domains"]) != set(DOMAINS):
        raise ValueError("baseline input index domain set differs")
    manifests: dict[str, Path] = {}
    for domain, raw in value["domains"].items():
        candidate = Path(str(raw))
        if not candidate.is_absolute() or not candidate.is_file():
            raise ValueError(f"input manifest is missing for {domain}: {candidate}")
        manifests[domain] = candidate.resolve()
    return manifests, value


def load_upstream_source(
    upstream_root: Path, input_index: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the completed baseline and return hash-locked source evidence."""

    paths = _source_paths(upstream_root)
    campaign = full.load_object(paths["campaign_summary"], "baseline campaign summary")
    telemetry = full.load_object(paths["telemetry_manifest"], "baseline telemetry manifest")
    integrity = full.load_object(paths["arm_integrity"], "baseline arm integrity")
    if campaign != {
        "arm_count": 27,
        "completed_arm_count": 27,
        "smoke": False,
        "spec_sha256": campaign.get("spec_sha256"),
        "status": "COMPLETED",
    }:
        raise ValueError("upstream baseline campaign is not a completed 27-arm run")
    if not isinstance(campaign.get("spec_sha256"), str) or len(campaign["spec_sha256"]) != 64:
        raise ValueError("upstream baseline specification fingerprint is invalid")
    if (
        telemetry.get("status") != "COMPLETE"
        or telemetry.get("spec_sha256") != campaign["spec_sha256"]
        or telemetry.get("round_rows") != 2700
        or telemetry.get("client_rows") != 13500
        or telemetry.get("evaluation_rows") != 162
        or telemetry.get("arm_integrity_rows") != 27
        or telemetry.get("privacy_label") != "NON_DP_PRIVATE_DIAGNOSTIC"
        or telemetry.get("initialization_transient_policy") is None
    ):
        raise ValueError("upstream baseline telemetry manifest differs")
    if telemetry.get("arm_integrity_sha256") != full.sha256_file(paths["arm_integrity"]):
        raise ValueError("upstream arm-integrity hash differs")

    arms = integrity.get("arms")
    if (
        integrity.get("status") != "COMPLETE"
        or integrity.get("spec_sha256") != campaign["spec_sha256"]
        or not isinstance(arms, list)
        or len(arms) != 27
    ):
        raise ValueError("upstream arm-integrity inventory differs")
    expected_settings = {
        (domain, model, seed)
        for domain in DOMAINS for model in MODELS for seed in SOURCE_SEEDS
    }
    observed_settings: set[tuple[str, str, int]] = set()
    repository_shas: set[str] = set()
    manifest_hashes: dict[str, set[str]] = {domain: set() for domain in DOMAINS}
    inventory_hashes: dict[str, set[str]] = {domain: set() for domain in DOMAINS}
    arm_contracts: dict[tuple[str, str, int], dict[str, str]] = {}
    for arm in arms:
        if not isinstance(arm, dict):
            raise ValueError("upstream arm-integrity entry is invalid")
        key = (str(arm.get("domain")), str(arm.get("model")), int(arm.get("seed", -1)))
        if (
            key not in expected_settings
            or arm.get("method") != FIXED_METHOD
            or arm.get("status") != "COMPLETED"
            or arm.get("client_steps") != 250
            or arm.get("arm_id")
            != f"paper-default-{key[0]}-{key[1]}-s{key[2]}-fixed-c10"
            or arm.get("rng_domain")
            != f"paper-default-{key[0]}-{key[1]}-s{key[2]}"
            or not isinstance(arm.get("run_config_fingerprint"), str)
            or len(arm["run_config_fingerprint"]) != 64
        ):
            raise ValueError(f"upstream arm-integrity contract differs: {key}")
        observed_settings.add(key)
        repository_shas.add(str(arm.get("repository_sha")))
        manifest_hashes[key[0]].add(str(arm.get("input_manifest_sha256")))
        inventory_hashes[key[0]].add(str(arm.get("input_inventory_sha256")))
        arm_contracts[key] = {
            "arm_id": str(arm["arm_id"]),
            "rng_domain": str(arm["rng_domain"]),
            "repository_sha": str(arm["repository_sha"]),
            "run_config_fingerprint": str(arm["run_config_fingerprint"]),
            "input_manifest_sha256": str(arm["input_manifest_sha256"]),
            "input_inventory_sha256": str(arm["input_inventory_sha256"]),
        }
    if observed_settings != expected_settings or len(repository_shas) != 1:
        raise ValueError("upstream 27-arm identity is incomplete")

    manifests, input_index_value = _validate_input_index(input_index)
    for domain, manifest in manifests.items():
        if manifest_hashes[domain] != {full.sha256_file(manifest)}:
            raise ValueError(f"current input manifest differs from upstream: {domain}")
        manifest_value = full.load_object(manifest, f"{domain} input manifest")
        if (
            len(inventory_hashes[domain]) != 1
            or manifest_value.get("inventory_sha256")
            != next(iter(inventory_hashes[domain]))
        ):
            raise ValueError(f"current input inventory differs from upstream: {domain}")

    round_rows = _read_csv(paths["round_telemetry"], "round telemetry")
    client_rows = _read_csv(paths["client_telemetry"], "client telemetry")
    evaluation_rows = _read_csv(paths["evaluation_telemetry"], "evaluation telemetry")
    if (len(round_rows), len(client_rows), len(evaluation_rows)) != (2700, 13500, 162):
        raise ValueError("upstream telemetry row counts differ from their manifest")

    source = spec["source_calibration"]
    raw_norms: dict[tuple[str, str, str], list[float]] = {
        (domain, model, group): []
        for domain in DOMAINS for model in MODELS for group in GROUPS
    }
    seen: set[tuple[str, str, int, int, int, str]] = set()
    for row in client_rows:
        try:
            domain = str(row["domain"])
            model = str(row["model"])
            seed = int(row["seed"])
            round_index = int(row["round"])
            client = int(row["client"])
            group = str(row["group"])
            raw_norm = float(row["raw_norm"])
            clip_norm = float(row["clip_norm"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("upstream client telemetry schema differs") from error
        if not math.isfinite(raw_norm) or raw_norm < 0.0:
            raise ValueError("upstream raw norm is outside its domain")
        if row.get("privacy_label") != source["source_privacy_label"]:
            raise ValueError("upstream client telemetry privacy label differs")
        if not (
            domain in DOMAINS and model in MODELS and seed in SOURCE_SEEDS
            and 1 <= round_index <= 50 and 0 <= client < 5 and group in GROUPS
        ):
            raise ValueError("upstream client telemetry identity differs")
        identity = (domain, model, seed, round_index, client, group)
        if identity in seen:
            raise ValueError(f"duplicate upstream client telemetry row: {identity}")
        seen.add(identity)
        contract = arm_contracts[(domain, model, seed)]
        if (
            row.get("arm_id") != contract["arm_id"]
            or row.get("rng_domain") != contract["rng_domain"]
            or row.get("method") != FIXED_METHOD
            or row.get("repository_sha") != contract["repository_sha"]
            or row.get("run_config_fingerprint")
            != contract["run_config_fingerprint"]
            or row.get("input_manifest_sha256")
            != contract["input_manifest_sha256"]
            or row.get("input_inventory_sha256")
            != contract["input_inventory_sha256"]
            or clip_norm != 10.0
        ):
            raise ValueError(f"upstream client telemetry provenance differs: {identity}")
        if source["round_min"] <= round_index <= source["round_max"]:
            raw_norms[(domain, model, group)].append(raw_norm)
    if len(seen) != 13500:
        raise ValueError("upstream client telemetry identity coverage differs")
    expected_calibration_rows = len(SOURCE_SEEDS) * 9 * 5
    if any(len(values) != expected_calibration_rows for values in raw_norms.values()):
        raise ValueError("rounds 2--10 calibration window is incomplete")

    file_hashes = {
        name: {
            "path": str(path.resolve()),
            "sha256": full.sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    return {
        "upstream_root": str(upstream_root.resolve()),
        "upstream_repository_sha": next(iter(repository_shas)),
        "upstream_spec_sha256": campaign["spec_sha256"],
        "source_files": file_hashes,
        "input_index_path": str(input_index.resolve()),
        "input_index_sha256": full.sha256_file(input_index.resolve()),
        "input_index": input_index_value,
        "input_manifests": {
            domain: {
                "path": str(path),
                "sha256": full.sha256_file(path),
                "inventory_sha256": next(iter(inventory_hashes[domain])),
            }
            for domain, path in manifests.items()
        },
        "source_round_window": [source["round_min"], source["round_max"]],
        "round_1_excluded": True,
        "late_round_noise_feedback_excluded": True,
        "calibration_rows_per_domain_model_group": expected_calibration_rows,
        "raw_norms": raw_norms,
    }


def _token(value: float) -> str:
    return full.number_token(float(value))


def _candidate_grid(
    source: Mapping[str, Any], spec: Mapping[str, Any], domain: str, model: str
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    quantiles = [float(value) for value in spec["source_calibration"]["raw_norm_quantiles"]]
    thresholds = {
        group: {
            format(probability, ".12g"): _quantile(
                source["raw_norms"][(domain, model, group)], probability
            )
            for probability in quantiles
        }
        for group in GROUPS
    }
    candidates: list[dict[str, Any]] = []
    for index_a, probability_a in enumerate(quantiles):
        for index_b, probability_b in enumerate(quantiles):
            candidates.append({
                "candidate_index": index_a * len(quantiles) + index_b,
                "candidate_kind": "source_raw_norm_quantile_cartesian",
                "source_quantile_by_group": {"A": probability_a, "B": probability_b},
                "clip_norm_by_group": {
                    "A": thresholds["A"][format(probability_a, ".12g")],
                    "B": thresholds["B"][format(probability_b, ".12g")],
                },
            })
    candidates.append({
        "candidate_index": 9,
        "candidate_kind": "exact_paper_default_anchor",
        "source_quantile_by_group": {"A": None, "B": None},
        "clip_norm_by_group": {"A": 10.0, "B": 10.0},
    })
    pairs = [
        (float(item["clip_norm_by_group"]["A"]), float(item["clip_norm_by_group"]["B"]))
        for item in candidates
    ]
    if len(candidates) != 10 or len(set(pairs)) != 10:
        raise ValueError(f"derived fixed-C grid is degenerate: {domain}/{model}")
    if any(not math.isfinite(value) or value <= 0.0 for pair in pairs for value in pair):
        raise ValueError(f"derived fixed-C grid is invalid: {domain}/{model}")
    return thresholds, candidates


def build_plan(
    *,
    spec_path: Path,
    repository: Path,
    expected_code_sha: str,
    upstream_root: Path,
    input_index: Path,
    private_key: Path,
    created_at_utc: str,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    if full.repository_sha(repository.resolve()) != expected_code_sha:
        raise RuntimeError("repository SHA differs from the requested immutable snapshot")
    if full.repository_dirty(repository.resolve()):
        raise RuntimeError("repository snapshot is dirty")
    full.validate_or_create_key(private_key.resolve(), create=False)
    source = load_upstream_source(upstream_root, input_index, spec)
    settings: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    defaults = spec["paper_defaults"]
    for domain in DOMAINS:
        for model in MODELS:
            thresholds, candidates = _candidate_grid(source, spec, domain, model)
            setting = {
                "domain": domain,
                "model": model,
                "source_threshold_quantiles_by_group": thresholds,
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
            setting["setting_sha256"] = full.sha256_bytes(full.canonical_bytes(setting))
            settings.append(setting)
            for candidate in candidates:
                c_a = float(candidate["clip_norm_by_group"]["A"])
                c_b = float(candidate["clip_norm_by_group"]["B"])
                for seed in spec["development"]["seeds"]:
                    arm_id = (
                        f"dev-fixed-{domain}-{model}-p{candidate['candidate_index']}-"
                        f"ca{_token(c_a)}-cb{_token(c_b)}-s{seed}"
                    )
                    arm = {
                        "arm_id": arm_id,
                        "stage": "groupwise_fixed_development",
                        "analysis_role": "development_only_strong_fixed_candidate",
                        "domain": domain,
                        "model": model,
                        "models": [model],
                        "method": FIXED_METHOD,
                        "candidate_index": candidate["candidate_index"],
                        "candidate_kind": candidate["candidate_kind"],
                        "source_quantile_by_group": candidate["source_quantile_by_group"],
                        "initial_clip_norm": c_b,
                        "initial_clip_norm_by_group": {"A": c_a, "B": c_b},
                        "seed": int(seed),
                        "rng_domain": f"baseline-followup:{domain}:{model}:s{seed}",
                        "input_manifest_sha256": source["input_manifests"][domain]["sha256"],
                        "input_inventory_sha256": source["input_manifests"][domain]["inventory_sha256"],
                        **defaults,
                    }
                    # ``method`` is already explicit above and must never be
                    # shadowed by a future specification edit.
                    arm["method"] = FIXED_METHOD
                    arm["arm_sha256"] = full.sha256_bytes(full.canonical_bytes(arm))
                    arms.append(arm)
    source_record = {key: value for key, value in source.items() if key != "raw_norms"}
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "GROUPWISE_FIXED_CANDIDATE_PLAN_LOCKED",
        "campaign_name": spec["campaign_name"],
        "created_at_utc": created_at_utc,
        "repository_sha": expected_code_sha,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": full.sha256_file(spec_path.resolve()),
        "private_key_commitment": full.sha256_file(private_key.resolve()),
        "upstream": source_record,
        "expected_arm_count": spec["expected_arm_count"],
        "selection_rule": spec["development"]["selection_rule"],
        "scientific_boundary": spec["scientific_boundary"],
        "settings": settings,
        "arms": arms,
    }
    plan["manifest_sha256"] = full.sha256_bytes(full.canonical_bytes(plan))
    validate_plan(plan, spec)
    return plan


def validate_plan(plan: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    if plan.get("status") != "GROUPWISE_FIXED_CANDIDATE_PLAN_LOCKED":
        raise RuntimeError("candidate plan status differs")
    supplied = plan.get("manifest_sha256")
    payload = {key: value for key, value in plan.items() if key != "manifest_sha256"}
    if supplied != full.sha256_bytes(full.canonical_bytes(payload)):
        raise RuntimeError("candidate plan self-hash differs")
    arms = plan.get("arms")
    if not isinstance(arms, list) or len(arms) != spec["expected_arm_count"]:
        raise RuntimeError("candidate plan arm count differs")
    ids: set[str] = set()
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for arm in arms:
        if not isinstance(arm, dict):
            raise RuntimeError("candidate plan arm is invalid")
        arm_payload = {key: value for key, value in arm.items() if key != "arm_sha256"}
        if arm.get("arm_sha256") != full.sha256_bytes(full.canonical_bytes(arm_payload)):
            raise RuntimeError("candidate arm self-hash differs")
        arm_id = str(arm.get("arm_id"))
        if arm_id in ids:
            raise RuntimeError("candidate arm IDs are not unique")
        ids.add(arm_id)
        if (
            arm.get("method") != FIXED_METHOD
            or arm.get("domain") not in DOMAINS
            or arm.get("model") not in MODELS
            or arm.get("models") != [arm.get("model")]
            or arm.get("seed") not in spec["development"]["seeds"]
            or set(arm.get("initial_clip_norm_by_group", {})) != set(GROUPS)
            or not isinstance(arm.get("input_manifest_sha256"), str)
            or len(arm["input_manifest_sha256"]) != 64
            or not isinstance(arm.get("input_inventory_sha256"), str)
            or len(arm["input_inventory_sha256"]) != 64
            or any("slaclip" in key.lower() for key in arm)
        ):
            raise RuntimeError("candidate arm is not fixed-only")
        if float(arm["initial_clip_norm"]) != float(arm["initial_clip_norm_by_group"]["B"]):
            raise RuntimeError("legacy and groupwise B thresholds differ")
        key = (str(arm["domain"]), str(arm["model"]), int(arm["seed"]))
        grouped.setdefault(key, []).append(arm)
    expected_groups = {
        (domain, model, seed)
        for domain in DOMAINS for model in MODELS
        for seed in spec["development"]["seeds"]
    }
    if set(grouped) != expected_groups or any(len(values) != 10 for values in grouped.values()):
        raise RuntimeError("candidate plan setting/seed coverage differs")
    for key, values in grouped.items():
        rng_domains = {str(value["rng_domain"]) for value in values}
        if len(rng_domains) != 1:
            raise RuntimeError(f"paired RNG domain differs within {key}")


def _plan_identity(
    *, plan_path: Path, spec_path: Path, repository: Path,
    expected_code_sha: str, upstream_root: Path, input_index: Path,
    private_key: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = full.load_object(plan_path.resolve(), "fixed follow-up candidate plan")
    spec = load_spec(spec_path.resolve())
    validate_plan(plan, spec)
    candidate = build_plan(
        spec_path=spec_path.resolve(), repository=repository.resolve(),
        expected_code_sha=expected_code_sha, upstream_root=upstream_root.resolve(),
        input_index=input_index.resolve(), private_key=private_key.resolve(),
        created_at_utc=str(plan.get("created_at_utc")),
    )
    if candidate != plan:
        raise RuntimeError("candidate plan differs from its immutable inputs")
    return spec, plan


def prepare(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    plan_path = root / PLAN_NAME
    if args.resume:
        if not plan_path.is_file():
            raise RuntimeError("resume requires an existing candidate plan")
        spec, plan = _plan_identity(
            plan_path=plan_path, spec_path=args.spec, repository=args.repository,
            expected_code_sha=args.expected_code_sha,
            upstream_root=args.upstream_root, input_index=args.input_index,
            private_key=args.private_key,
        )
        del spec
    else:
        if root.exists():
            raise RuntimeError(f"refusing to overwrite campaign root: {root}")
        root.mkdir(parents=True, mode=0o700)
        plan = build_plan(
            spec_path=args.spec.resolve(), repository=args.repository.resolve(),
            expected_code_sha=args.expected_code_sha,
            upstream_root=args.upstream_root.resolve(),
            input_index=args.input_index.resolve(),
            private_key=args.private_key.resolve(), created_at_utc=full.utc_now(),
        )
        full.atomic_json(plan_path, plan)
    for name in ("arms", "arm-status", "arm-logs", "control", "preflight", "tmp"):
        (root / name).mkdir(mode=0o700, exist_ok=True)
    stop = root / "control" / "stop.request"
    if stop.exists():
        stop.unlink()
    print(json.dumps({
        "status": "READY", "candidate_plan": str(plan_path),
        "manifest_sha256": plan["manifest_sha256"],
        "arm_count": len(plan["arms"]), "fixed_only": True,
    }, indent=2, sort_keys=True))


def _arm_command(
    arm: Mapping[str, Any], *, repository: Path, python_bin: Path,
    input_manifest: Path, output_dir: Path, private_key: Path,
    stop_file: Path, smoke: bool,
) -> list[str]:
    thresholds = arm["initial_clip_norm_by_group"]
    command = [
        str(python_bin), str(repository / "paper_repro" / "train_federated.py"),
        "--input-manifest", str(input_manifest), "--output-dir", str(output_dir),
        "--models", str(arm["model"]), "--device", "cuda",
        "--method", FIXED_METHOD, "--num-clients", str(arm["num_clients"]),
        "--rounds", str(arm["rounds"]), "--batch-size", str(arm["batch_size"]),
        "--noise-multiplier", str(arm["noise_multiplier"]),
        "--learning-rate", str(arm["learning_rate"]),
        "--clip-norm", str(arm["initial_clip_norm"]),
        "--clip-norm-a", str(thresholds["A"]),
        "--clip-norm-b", str(thresholds["B"]),
        "--rank", str(arm["rank"]), "--max-seq-length", str(arm["max_seq_length"]),
        "--max-validation-records", str(arm["max_validation_records"]),
        "--seed", str(arm["seed"]), "--data-split-seed", str(arm["data_split_seed"]),
        "--evaluation-seed", str(arm["evaluation_seed"]),
        "--delta", str(arm["delta"]), "--eval-every", str(arm["eval_every"]),
        "--checkpoint-every", str(arm["checkpoint_every"]),
        "--private-rng-key", str(private_key), "--rng-domain", str(arm["rng_domain"]),
        "--pair-noise-across-methods", "--stop-file", str(stop_file),
        "--acknowledge-non-dp-diagnostics",
    ]
    if smoke:
        command.append("--smoke")
    if output_dir.exists():
        command.append("--resume")
    forbidden = [part for part in command if "slaclip" in part.lower()]
    if forbidden:
        raise RuntimeError(f"fixed-only command contains an adaptive option: {forbidden}")
    return command


def _completed_summary(path: Path, arm: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = full.load_object(path, "arm final summary")
    except RuntimeError:
        return None
    models = value.get("models")
    fingerprint = value.get("run_config_fingerprint")
    if not (
        value.get("status") == "COMPLETED"
        and value.get("method") == FIXED_METHOD
        and isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and isinstance(models, dict)
        and set(models) == {arm["model"]}
        and isinstance(models[arm["model"]], dict)
        and models[arm["model"]].get("status") == "COMPLETED"
        and models[arm["model"]].get("method") == FIXED_METHOD
        and models[arm["model"]].get("run_config_fingerprint") == fingerprint
    ):
        return None
    return value


def _verified_completed_status(
    status_path: Path, summary_path: Path, arm: Mapping[str, Any],
    plan: Mapping[str, Any], *, smoke: bool,
) -> dict[str, Any] | None:
    """Return a completed status only when every persisted identity still matches."""

    if not status_path.is_file() or _completed_summary(summary_path, arm) is None:
        return None
    try:
        status = full.load_object(status_path, "arm status")
    except RuntimeError:
        return None
    if (
        status.get("schema_version") != 1
        or status.get("status") != "COMPLETED"
        or status.get("smoke") is not smoke
        or status.get("arm_id") != arm["arm_id"]
        or status.get("arm_sha256") != arm["arm_sha256"]
        or status.get("manifest_sha256") != plan["manifest_sha256"]
        or status.get("final_summary_sha256") != full.sha256_file(summary_path)
    ):
        return None
    return status


def _validate_locked_input_manifest(
    plan: Mapping[str, Any], arm: Mapping[str, Any]
) -> Path:
    """Fail closed if a per-domain input manifest changed between arms."""

    record = plan["upstream"]["input_manifests"][arm["domain"]]
    path = Path(str(record["path"])).resolve()
    if (
        not path.is_file()
        or full.sha256_file(path) != record["sha256"]
        or arm["input_manifest_sha256"] != record["sha256"]
        or arm["input_inventory_sha256"] != record["inventory_sha256"]
    ):
        raise RuntimeError(f"locked input manifest changed: {arm['domain']}")
    value = full.load_object(path, f"{arm['domain']} locked input manifest")
    if value.get("inventory_sha256") != record["inventory_sha256"]:
        raise RuntimeError(f"locked input inventory changed: {arm['domain']}")
    return path


def _execute(
    *, plan: Mapping[str, Any], arm: Mapping[str, Any], campaign_root: Path,
    repository: Path, python_bin: Path, private_key: Path, smoke_label: str | None,
) -> int:
    smoke = smoke_label is not None
    section = "preflight" if smoke else "arms"
    identity = smoke_label or str(arm["arm_id"])
    output_dir = campaign_root / section / identity
    status_path = campaign_root / ("preflight" if smoke else "arm-status") / f"{identity}-status.json"
    summary_path = output_dir / "final_summary.json"
    input_manifest = _validate_locked_input_manifest(plan, arm)
    prior_status = _verified_completed_status(
        status_path, summary_path, arm, plan, smoke=smoke
    )
    if prior_status is not None:
        # Formal arms receive a deep all-round threshold/noise validation before
        # a completed result is ever reused.  A valid-looking orphan summary is
        # intentionally sent through the trainer's strict --resume path below,
        # which checks the complete scientific-contract fingerprint and repairs
        # the missing status without re-running completed training.
        if not smoke:
            _metric_row(campaign_root, arm)
        if smoke:
            print(f"smoke_reused={identity}")
        else:
            print(
                f"arm_reused={arm['arm_id']} "
                f"final_summary_sha256={prior_status['final_summary_sha256']}"
            )
        return 0
    stop_file = campaign_root / "control" / "stop.request"
    if stop_file.exists():
        return 75
    command = _arm_command(
        arm, repository=repository, python_bin=python_bin,
        input_manifest=input_manifest, output_dir=output_dir,
        private_key=private_key, stop_file=stop_file, smoke=smoke,
    )
    full.atomic_json(status_path, {
        "schema_version": 1, "status": "RUNNING", "smoke": smoke,
        "arm_id": arm["arm_id"], "arm_sha256": arm["arm_sha256"],
        "manifest_sha256": plan["manifest_sha256"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "updated_at_utc": full.utc_now(),
    })
    log_prefix = campaign_root / ("preflight" if smoke else "arm-logs") / identity
    arm_tmp = campaign_root / "tmp" / identity
    arm_tmp.mkdir(parents=True, mode=0o700, exist_ok=True)
    environment = dict(os.environ)
    environment.update({
        "TMPDIR": str(arm_tmp),
        "TORCH_EXTENSIONS_DIR": str(arm_tmp / "torch-extensions"),
        "TRITON_CACHE_DIR": str(arm_tmp / "triton"),
        "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
    })
    (arm_tmp / "torch-extensions").mkdir(mode=0o700, exist_ok=True)
    (arm_tmp / "triton").mkdir(mode=0o700, exist_ok=True)
    with (log_prefix.with_suffix(".out")).open("ab") as stdout, (
        log_prefix.with_suffix(".err")
    ).open("ab") as stderr:
        result = subprocess.run(
            command, cwd=repository, env=environment, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, check=False,
        )
    return_code = int(result.returncode)
    if return_code == 0:
        if _completed_summary(summary_path, arm) is None:
            return_code = 1
        elif not smoke:
            # Refuse to bless a zero exit until all 50 persisted round shards
            # prove the requested fixed A/B thresholds and noise scales.
            _metric_row(campaign_root, arm)
    state = "COMPLETED" if return_code == 0 else "CHECKPOINTED_STOP" if return_code == 75 else "FAILED"
    record = {
        "schema_version": 1, "status": state, "smoke": smoke,
        "arm_id": arm["arm_id"], "arm_sha256": arm["arm_sha256"],
        "manifest_sha256": plan["manifest_sha256"], "exit_code": return_code,
        "updated_at_utc": full.utc_now(),
    }
    if return_code == 0:
        record["final_summary_sha256"] = full.sha256_file(summary_path)
    full.atomic_json(status_path, record)
    return return_code


def run_arm(args: argparse.Namespace) -> int:
    plan = full.load_object(args.plan.resolve(), "fixed follow-up candidate plan")
    spec = load_spec(args.spec.resolve())
    validate_plan(plan, spec)
    if full.repository_sha(args.repository.resolve()) != plan["repository_sha"] or full.repository_dirty(args.repository.resolve()):
        raise RuntimeError("repository changed before arm execution")
    full.validate_or_create_key(args.private_key.resolve(), create=False)
    if full.sha256_file(args.private_key.resolve()) != plan["private_key_commitment"]:
        raise RuntimeError("private RNG key differs from candidate plan")
    if args.arm_index < 0 or args.arm_index >= len(plan["arms"]):
        raise ValueError("arm index is outside the candidate plan")
    return _execute(
        plan=plan, arm=plan["arms"][args.arm_index],
        campaign_root=args.campaign_root.resolve(), repository=args.repository.resolve(),
        python_bin=args.python_bin.resolve(), private_key=args.private_key.resolve(),
        smoke_label=None,
    )


def run_smokes(args: argparse.Namespace) -> int:
    plan = full.load_object(args.plan.resolve(), "fixed follow-up candidate plan")
    spec = load_spec(args.spec.resolve())
    validate_plan(plan, spec)
    selections: list[tuple[str, str]] = [
        ("meddialog", "bert"), ("meddialog", "gpt2"),
        ("meddialog", "chatglm2"), ("slimpajama", "chatglm2"),
        ("finance", "chatglm2"),
    ]
    for domain, model in selections:
        arm = next(
            candidate for candidate in plan["arms"]
            if candidate["domain"] == domain and candidate["model"] == model
            and candidate["candidate_index"] == 0
            and candidate["seed"] == spec["development"]["seeds"][0]
        )
        return_code = _execute(
            plan=plan, arm=arm, campaign_root=args.campaign_root.resolve(),
            repository=args.repository.resolve(), python_bin=args.python_bin.resolve(),
            private_key=args.private_key.resolve(), smoke_label=f"{domain}-{model}",
        )
        if return_code != 0:
            return return_code
    return 0


def _normalized_loss_auc(evaluations: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    points = [
        (int(value["round"]), float(value["loss"])) for value in evaluations
        if isinstance(value, dict) and isinstance(value.get("round"), int)
        and isinstance(value.get("loss"), (int, float))
        and not isinstance(value.get("loss"), bool)
        and math.isfinite(float(value["loss"]))
    ]
    if len(points) != len(evaluations) or len(points) < 2 or points != sorted(points):
        raise RuntimeError("arm evaluation trajectory is incomplete")
    span = points[-1][0] - points[0][0]
    if span != 50:
        raise RuntimeError("arm evaluation trajectory does not span rounds 0--50")
    auc = math.fsum(
        (right_round - left_round) * (left_loss + right_loss) / 2.0
        for (left_round, left_loss), (right_round, right_loss) in zip(points, points[1:])
    ) / span
    return points[0][1], points[-1][1], auc


def _metric_row(root: Path, arm: Mapping[str, Any]) -> dict[str, Any]:
    # Reuse the mature all-round threshold/noise evidence gate without using
    # its hard-coded BERT/GPT-2 campaign planner.
    try:
        from paper_repro.oracle_ceiling_campaign import (
            _validate_fixed_groupwise_threshold_evidence,
        )
    except ModuleNotFoundError:  # direct-script execution
        from oracle_ceiling_campaign import (  # type: ignore[no-redef]
            _validate_fixed_groupwise_threshold_evidence,
        )
    _validate_fixed_groupwise_threshold_evidence(root, arm)
    arm_root = root / "arms" / str(arm["arm_id"])
    run_config = full.load_object(arm_root / "run_config.json", "fixed run config")
    scientific_contract = run_config.get("scientific_contract")
    if (
        not isinstance(scientific_contract, dict)
        or scientific_contract.get("input_manifest_sha256")
        != arm["input_manifest_sha256"]
        or scientific_contract.get("input_inventory_sha256")
        != arm["input_inventory_sha256"]
    ):
        raise RuntimeError(f"fixed development input identity differs: {arm['arm_id']}")
    root_summary_path = arm_root / "final_summary.json"
    root_summary = _completed_summary(root_summary_path, arm)
    if root_summary is None:
        raise RuntimeError(f"fixed development arm is incomplete: {arm['arm_id']}")
    model_summary = root_summary["models"][arm["model"]]
    initial_loss, final_loss, auc = _normalized_loss_auc(model_summary["evaluations"])
    behavior = model_summary.get("behavior_summary")
    randomness = model_summary.get("privacy_randomness")
    if not isinstance(behavior, dict) or not isinstance(randomness, dict):
        raise RuntimeError("fixed development evidence is incomplete")
    if (
        randomness.get("pair_noise_across_methods") is not True
        or randomness.get("rng_domain") != arm["rng_domain"]
        or randomness.get("private_key_commitment") is None
    ):
        raise RuntimeError("paired standard-Gaussian contract differs")
    groups = behavior.get("groups")
    if not isinstance(groups, dict) or set(groups) != set(GROUPS):
        raise RuntimeError("fixed development behavior groups differ")
    thresholds = arm["initial_clip_norm_by_group"]
    final_evaluation = model_summary["evaluations"][-1]
    return {
        "domain": arm["domain"], "model": arm["model"],
        "candidate_index": arm["candidate_index"],
        "candidate_kind": arm["candidate_kind"], "seed": arm["seed"],
        "arm_id": arm["arm_id"], "arm_sha256": arm["arm_sha256"],
        "C_A": float(thresholds["A"]), "C_B": float(thresholds["B"]),
        "initial_loss": initial_loss, "final_loss": final_loss,
        "normalized_loss_auc": auc,
        "final_token_accuracy": float(final_evaluation["token_accuracy"]),
        "actual_clipped_fraction": float(behavior["any_group_actual_clipped_fraction"]),
        "actual_clipped_fraction_A": float(groups["A"]["actual_clipped_fraction"]),
        "actual_clipped_fraction_B": float(groups["B"]["actual_clipped_fraction"]),
        "sample_schedule_sha256": behavior["sample_schedule_sha256"],
        "supervision_schedule_sha256": behavior["supervision_schedule_sha256"],
        "private_key_commitment": randomness["private_key_commitment"],
        "rng_domain": randomness["rng_domain"],
        "client_partition_sha256": model_summary["client_partition_sha256"],
        "round_shard_prefix_sha256": model_summary["round_shard_prefix_sha256"],
        "run_config_fingerprint": model_summary["run_config_fingerprint"],
        "root_final_summary_sha256": full.sha256_file(root_summary_path),
        "model_final_summary_sha256": full.sha256_file(
            arm_root / arm["model"] / "final_summary.json"
        ),
    }


def _candidate_rankings(
    rows: Sequence[Mapping[str, Any]], *, seeds: Sequence[int],
    noise_multiplier: float,
) -> list[dict[str, Any]]:
    pairs = sorted({(float(row["C_A"]), float(row["C_B"])) for row in rows})
    rankings: list[dict[str, Any]] = []
    for c_a, c_b in pairs:
        subset = [
            row for row in rows
            if float(row["C_A"]) == c_a and float(row["C_B"]) == c_b
        ]
        if {int(row["seed"]) for row in subset} != set(seeds) or len(subset) != len(seeds):
            raise RuntimeError(f"fixed candidate is incomplete: {c_a}/{c_b}")
        final = [float(row["final_loss"]) for row in subset]
        auc = [float(row["normalized_loss_auc"]) for row in subset]
        rankings.append({
            "C_A": c_a, "C_B": c_b, "seed_count": len(subset),
            "mean_final_loss": statistics.fmean(final),
            "mean_normalized_loss_auc": statistics.fmean(auc),
            "final_loss_sample_std": statistics.stdev(final),
            "groupwise_noise_scale_l2": noise_multiplier * math.sqrt(c_a * c_a + c_b * c_b),
            "mean_actual_clipped_fraction": statistics.fmean(
                float(row["actual_clipped_fraction"]) for row in subset
            ),
            "seed_evidence": sorted(
                [dict(row) for row in subset], key=lambda item: int(item["seed"])
            ),
        })
    rankings.sort(key=lambda row: (
        row["mean_final_loss"], row["mean_normalized_loss_auc"],
        row["final_loss_sample_std"], row["groupwise_noise_scale_l2"],
        row["C_A"], row["C_B"],
    ))
    return rankings


def _validate_pairing(rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["domain"]), str(row["model"]), int(row["seed"])), []
        ).append(row)
    if len(grouped) != 18 or any(len(values) != 10 for values in grouped.values()):
        raise RuntimeError("paired fixed candidate coverage differs")
    fields = (
        "initial_loss", "sample_schedule_sha256", "supervision_schedule_sha256",
        "private_key_commitment", "rng_domain", "client_partition_sha256",
    )
    for key, values in grouped.items():
        reference = values[0]
        for candidate in values[1:]:
            if any(candidate[field] != reference[field] for field in fields):
                raise RuntimeError(f"paired fixed evidence differs: {key}")


def build_selection_lock(
    *, root: Path, plan: Mapping[str, Any], spec: Mapping[str, Any],
    created_at_utc: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [_metric_row(root, arm) for arm in plan["arms"]]
    _validate_pairing(rows)
    setting_records: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for model in MODELS:
            subset = [
                row for row in rows
                if row["domain"] == domain and row["model"] == model
            ]
            rankings = _candidate_rankings(
                subset, seeds=spec["development"]["seeds"],
                noise_multiplier=float(spec["paper_defaults"]["noise_multiplier"]),
            )
            if len(rankings) != 10:
                raise RuntimeError(f"fixed candidate ranking is incomplete: {domain}/{model}")
            setting_records.append({
                "domain": domain, "model": model,
                "ordered_candidates": rankings,
                "top_candidates": rankings[: spec["development"]["top_k_per_setting"]],
            })
    lock: dict[str, Any] = {
        "schema_version": 1,
        "status": "GROUPWISE_FIXED_FOLLOWUP_SELECTION_LOCKED",
        "campaign_name": plan["campaign_name"],
        "created_at_utc": created_at_utc,
        "candidate_plan_sha256": plan["manifest_sha256"],
        "repository_sha": plan["repository_sha"],
        "spec_sha256": plan["spec_sha256"],
        "upstream_source_evidence": plan["upstream"],
        "development_seeds": spec["development"]["seeds"],
        "selection_rule": spec["development"]["selection_rule"],
        "top_k_per_setting": spec["development"]["top_k_per_setting"],
        "development_only": True,
        "confirmation_or_adaptive_data_accessed": False,
        "full_slaclip_run": False,
        "slaclip_q_run": False,
        "all_candidate_count": len(rows),
        "settings": setting_records,
    }
    lock["lock_sha256"] = full.sha256_bytes(full.canonical_bytes(lock))
    return lock, rows


METRIC_COLUMNS = (
    "domain", "model", "candidate_index", "candidate_kind", "seed", "arm_id",
    "arm_sha256", "C_A", "C_B", "initial_loss", "final_loss",
    "normalized_loss_auc", "final_token_accuracy", "actual_clipped_fraction",
    "actual_clipped_fraction_A", "actual_clipped_fraction_B",
    "sample_schedule_sha256", "supervision_schedule_sha256",
    "private_key_commitment", "rng_domain", "client_partition_sha256",
    "round_shard_prefix_sha256", "run_config_fingerprint",
    "root_final_summary_sha256", "model_final_summary_sha256",
)


def _bind_metrics_to_lock(
    lock: Mapping[str, Any], metrics_path: Path, row_count: int
) -> dict[str, Any]:
    if not metrics_path.is_file():
        raise RuntimeError("fixed follow-up metrics are missing")
    value = {key: item for key, item in lock.items() if key != "lock_sha256"}
    value["campaign_metrics_sha256"] = full.sha256_file(metrics_path)
    value["campaign_metrics_rows"] = row_count
    value["lock_sha256"] = full.sha256_bytes(full.canonical_bytes(value))
    return value


def _validate_selection_artifacts(
    selection_path: Path, metrics_path: Path, expected_rows: int
) -> dict[str, Any]:
    lock = full.load_object(selection_path, "fixed follow-up selection lock")
    supplied = lock.get("lock_sha256")
    payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if supplied != full.sha256_bytes(full.canonical_bytes(payload)):
        raise RuntimeError("existing fixed selection lock self-hash differs")
    if (
        lock.get("campaign_metrics_rows") != expected_rows
        or not metrics_path.is_file()
        or lock.get("campaign_metrics_sha256") != full.sha256_file(metrics_path)
    ):
        raise RuntimeError("fixed selection metrics identity differs")
    return lock


def lock_selection(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec = load_spec(args.spec.resolve())
    plan = full.load_object(args.plan.resolve(), "fixed follow-up candidate plan")
    validate_plan(plan, spec)
    path = root / SELECTION_LOCK_NAME
    metrics_path = root / METRICS_NAME
    if path.is_file():
        existing = _validate_selection_artifacts(
            path, metrics_path, len(plan["arms"])
        )
        candidate_base, rows = build_selection_lock(
            root=root, plan=plan, spec=spec,
            created_at_utc=str(existing.get("created_at_utc")),
        )
        candidate = _bind_metrics_to_lock(candidate_base, metrics_path, len(rows))
        if candidate != existing:
            raise RuntimeError("existing fixed selection lock differs from completed evidence")
        lock = existing
    else:
        lock_base, rows = build_selection_lock(
            root=root, plan=plan, spec=spec, created_at_utc=full.utc_now()
        )
        full.atomic_csv(metrics_path, rows, METRIC_COLUMNS)
        lock = _bind_metrics_to_lock(lock_base, metrics_path, len(rows))
        full.atomic_json(path, lock)
    print(json.dumps({
        "status": lock["status"], "selection_lock": str(path),
        "lock_sha256": lock["lock_sha256"],
        "settings": len(lock["settings"]), "top_k_per_setting": 3,
    }, indent=2, sort_keys=True))


def aggregate(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    spec = load_spec(args.spec.resolve())
    plan = full.load_object(args.plan.resolve(), "fixed follow-up candidate plan")
    validate_plan(plan, spec)
    status_counts = {"COMPLETED": 0, "CHECKPOINTED_STOP": 0, "FAILED": 0, "NOT_STARTED": 0}
    for arm in plan["arms"]:
        status_path = root / "arm-status" / f"{arm['arm_id']}-status.json"
        state = "NOT_STARTED"
        if status_path.is_file():
            state = str(full.load_object(status_path, "arm status").get("status"))
        status_counts[state if state in status_counts else "FAILED"] += 1
    selection_path = root / SELECTION_LOCK_NAME
    selection_lock = None
    if selection_path.is_file():
        selection_lock = _validate_selection_artifacts(
            selection_path, root / METRICS_NAME, len(plan["arms"])
        )
    complete = status_counts["COMPLETED"] == len(plan["arms"]) and selection_lock is not None
    summary = {
        "schema_version": 1, "status": "COMPLETED" if complete else "IN_PROGRESS",
        "campaign_name": plan["campaign_name"],
        "candidate_plan_sha256": plan["manifest_sha256"],
        "expected_arm_count": len(plan["arms"]), "status_counts": status_counts,
        "selection_lock_path": str(selection_path) if selection_path.is_file() else None,
        "selection_lock_sha256": (
            selection_lock.get("lock_sha256") if selection_lock is not None else None
        ),
        "development_only": True, "full_slaclip_run": False,
        "slaclip_q_run": False, "updated_at_utc": full.utc_now(),
    }
    full.atomic_json(root / SUMMARY_NAME, summary)
    if args.require_complete and not complete:
        raise RuntimeError("fixed follow-up campaign is incomplete")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-spec")
    validate.add_argument("--spec", type=Path, required=True)
    prepare_parser = commands.add_parser("prepare")
    for command in (prepare_parser,):
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--expected-code-sha", required=True)
        command.add_argument("--upstream-root", type=Path, required=True)
        command.add_argument("--input-index", type=Path, required=True)
        command.add_argument("--campaign-root", type=Path, required=True)
        command.add_argument("--private-key", type=Path, required=True)
        command.add_argument("--resume", action="store_true")
    verify = commands.add_parser("validate-locked-plan")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--spec", type=Path, required=True)
    verify.add_argument("--repository", type=Path, required=True)
    verify.add_argument("--expected-code-sha", required=True)
    verify.add_argument("--upstream-root", type=Path, required=True)
    verify.add_argument("--input-index", type=Path, required=True)
    verify.add_argument("--private-key", type=Path, required=True)
    for name in ("run-arm", "run-smokes"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--campaign-root", type=Path, required=True)
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--python-bin", type=Path, required=True)
        command.add_argument("--private-key", type=Path, required=True)
        if name == "run-arm":
            command.add_argument("--arm-index", type=int, required=True)
    for name in ("lock-selection", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--campaign-root", type=Path, required=True)
        if name == "aggregate":
            command.add_argument("--require-complete", action="store_true")
    indices = commands.add_parser("print-arm-indices")
    indices.add_argument("--plan", type=Path, required=True)
    inputs = commands.add_parser("print-input-manifests")
    inputs.add_argument("--plan", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    if args.command == "validate-spec":
        spec = load_spec(args.spec)
        print(json.dumps({
            "status": "VALID", "expected_arm_count": spec["expected_arm_count"],
            "fixed_only": True, "full_slaclip": False, "slaclip_q": False,
        }, indent=2, sort_keys=True))
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "validate-locked-plan":
        spec, plan = _plan_identity(
            plan_path=args.plan, spec_path=args.spec,
            repository=args.repository, expected_code_sha=args.expected_code_sha,
            upstream_root=args.upstream_root, input_index=args.input_index,
            private_key=args.private_key,
        )
        del spec
        print(json.dumps({
            "status": "VALID", "manifest_sha256": plan["manifest_sha256"],
            "arm_count": len(plan["arms"]),
        }, indent=2, sort_keys=True))
    elif args.command == "run-arm":
        raise SystemExit(run_arm(args))
    elif args.command == "run-smokes":
        raise SystemExit(run_smokes(args))
    elif args.command == "lock-selection":
        lock_selection(args)
    elif args.command == "aggregate":
        aggregate(args)
    elif args.command == "print-arm-indices":
        plan = full.load_object(args.plan.resolve(), "candidate plan")
        for index in range(len(plan.get("arms", []))):
            print(index)
    elif args.command == "print-input-manifests":
        plan = full.load_object(args.plan.resolve(), "candidate plan")
        for domain in DOMAINS:
            print(plan["upstream"]["input_manifests"][domain]["path"])
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
