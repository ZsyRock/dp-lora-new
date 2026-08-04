#!/usr/bin/env python3
"""Build, execute, resume, summarize, and archive one full-SlaClip campaign.

The Slurm worker invokes one arm at a time through ``run-arm``.  Every arm
contains both paper-reconstruction models, while two independent arms occupy
the allocation's two GPU lanes.  The checked-in declarative specification is
expanded before training and then treated as immutable.

This coordinator intentionally does not implement SlaClip.  It only invokes
``train_federated.py`` with the full controller's public CLI.  In particular,
it never accepts a fixed clipping target or a calibration file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import secrets
import shutil
import signal
import socket
import stat
import statistics
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
RUNTIME_MANIFEST_NAME = "runtime-manifest.json"
JOB_STATUS_NAME = "job-status.json"
CAMPAIGN_KEY_BYTES = 32
FULL_SLACLIP_METHOD = "slaclip_dp_lora"
ORACLE_SLACLIP_METHOD = "oracle_slaclip_control"
FIXED_DP_METHOD = "paper_dp_lora"
CONTROL_METHODS = ("no_dp_lora_control", "clip_only_control")
ADAPTIVE_METHODS = frozenset({FULL_SLACLIP_METHOD, ORACLE_SLACLIP_METHOD})
NOISY_CONTROLLER_INPUT = "noisy_endpoints"
EXACT_CONTROLLER_INPUT = "exact_endpoints"
CONTROLLER_INPUT_BY_METHOD = {
    FULL_SLACLIP_METHOD: NOISY_CONTROLLER_INPUT,
    ORACLE_SLACLIP_METHOD: EXACT_CONTROLLER_INPUT,
}
ORACLE_NOISY_MATCH_FIELDS = (
    "seed",
    "reference_arm_id",
    "rng_domain",
    "models",
    "num_clients",
    "rounds",
    "batch_size",
    "noise_multiplier",
    "learning_rate",
    "initial_clip_norm",
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
    "slaclip_base_target_clipped_fraction",
    "slaclip_beta",
)
ALLOWED_METHODS = {*ADAPTIVE_METHODS, FIXED_DP_METHOD, *CONTROL_METHODS}
EXPECTED_MODELS = ("bert", "gpt2")
SMALL_ARCHIVE_MAX_BYTES = 32 * 1024 * 1024
FULL_COMPARISON_STATUS = "FULL_SLACLIP_COMPARISON_COMPLETE"
REQUIRED_STEP_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "SLURM_EXPORT_ENV": "ALL",
}
TERMINAL_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, value: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    atomic_bytes(path, encoded + b"\n")


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not load {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}")


def require_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def require_int(value: Any, label: str, *, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{label} has an invalid sign")
    return value


def number_token(value: float) -> str:
    text = format(value, ".12g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _unique(values: Sequence[Any], label: str) -> None:
    encoded = [canonical_bytes(value) for value in values]
    if len(encoded) != len(set(encoded)):
        raise ValueError(f"{label} contains duplicates")


def _common_values(spec: Mapping[str, Any]) -> dict[str, Any]:
    common = spec.get("common")
    if not isinstance(common, dict):
        raise ValueError("common must be an object")
    expected = {
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
        "default_eta",
        "default_beta",
    }
    require_exact_keys(common, expected, "common")
    models = common["models"]
    if not isinstance(models, list) or tuple(models) != EXPECTED_MODELS:
        raise ValueError("common.models must be exactly ['bert', 'gpt2']")
    integers = (
        "num_clients",
        "rounds",
        "batch_size",
        "rank",
        "max_seq_length",
        "max_validation_records",
        "eval_every",
        "checkpoint_every",
        "data_split_seed",
        "evaluation_seed",
        "slaclip_num_slots",
    )
    normalized = dict(common)
    for name in integers:
        normalized[name] = require_int(common[name], f"common.{name}")
    for name in (
        "noise_multiplier",
        "learning_rate",
        "delta",
        "slaclip_c_min",
        "slaclip_c_max",
        "default_eta",
        "default_beta",
    ):
        normalized[name] = require_number(common[name], f"common.{name}", positive=True)
    if normalized["slaclip_c_max"] < normalized["slaclip_c_min"]:
        raise ValueError("SlaClip threshold bounds are reversed")
    if not 0 < normalized["delta"] < 1:
        raise ValueError("common.delta must be in (0, 1)")
    if not 0 < normalized["default_beta"] <= 1:
        raise ValueError("default_beta must be in (0, 1]")
    if normalized["rounds"] != 50 or normalized["batch_size"] != 8:
        raise ValueError("formal campaign must retain T=50 and B=8")
    if normalized["slaclip_num_slots"] < 2:
        raise ValueError("full SlaClip requires at least two CDF endpoint slots")
    return normalized


def load_spec(path: Path) -> dict[str, Any]:
    spec = load_object(path, "campaign specification")
    require_exact_keys(
        spec,
        {
            "schema_version",
            "campaign_name",
            "description",
            "expected_arm_count",
            "common",
            "primary",
            "threshold_robustness",
            "sensitivity",
            "noise_sensitivity",
            "client_sensitivity",
            "controls",
            "oracle_controls",
            "scientific_boundary",
        },
        "campaign specification",
    )
    if spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported campaign-spec schema")
    if not isinstance(spec["campaign_name"], str) or not spec["campaign_name"]:
        raise ValueError("campaign_name is invalid")
    require_int(spec["expected_arm_count"], "expected_arm_count")
    _common_values(spec)
    if not isinstance(spec["scientific_boundary"], dict):
        raise ValueError("scientific_boundary must be an object")
    return spec


def _base_arm(
    *,
    common: Mapping[str, Any],
    arm_id: str,
    family: str,
    method: str,
    seed: int,
    clip_norm: float,
    eta: float | None,
    beta: float | None,
    reference_arm_id: str | None,
    num_clients: int | None = None,
    noise_multiplier: float | None = None,
) -> dict[str, Any]:
    if method not in ALLOWED_METHODS:
        raise ValueError(f"unsupported method: {method}")
    adaptive = method in ADAPTIVE_METHODS
    if adaptive != (eta is not None and beta is not None):
        raise ValueError("adaptive controller parameters and method disagree")
    if family == "primary":
        analysis_role = "paper_setting_confirmatory_seed_replication"
    elif family == "threshold_robustness":
        analysis_role = "pre_registered_initial_threshold_robustness"
    elif family == "sensitivity":
        analysis_role = "pre_registered_controller_hyperparameter_sensitivity"
    elif family == "noise_sensitivity":
        analysis_role = "pre_registered_noise_multiplier_sensitivity"
    elif family == "client_sensitivity":
        analysis_role = "pre_registered_cdf_record_count_sensitivity"
    elif family == "control":
        analysis_role = "mechanism_control"
    elif family == "oracle_control":
        analysis_role = "non_dp_exact_endpoint_oracle_controller_diagnostic"
    else:
        raise ValueError(f"unsupported arm family: {family}")
    return {
        "arm_id": arm_id,
        "family": family,
        "analysis_role": analysis_role,
        "method": method,
        "seed": seed,
        "initial_clip_norm": clip_norm,
        "slaclip_eta": eta,
        # ``beta`` is the full-controller base target clipped fraction.  The
        # per-round target remains dynamic:
        # beta * (1 - noisy_near_zero / (C_t + epsilon)).
        "slaclip_base_target_clipped_fraction": beta,
        # Retain the mathematical name in manifests for backwards-compatible
        # analysis of snapshots created before the semantic CLI was added.
        "slaclip_beta": beta,
        "controller_input": CONTROLLER_INPUT_BY_METHOD.get(method),
        "reference_arm_id": reference_arm_id,
        # Common random numbers isolate method/C/noise effects within a seed.
        "rng_domain": f"full-slaclip-cdf:s{seed}",
        "models": list(common["models"]),
        "num_clients": (
            common["num_clients"] if num_clients is None else num_clients
        ),
        "rounds": common["rounds"],
        "batch_size": common["batch_size"],
        "noise_multiplier": (
            common["noise_multiplier"]
            if noise_multiplier is None
            else noise_multiplier
        ),
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


def expand_spec(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    common = _common_values(spec)
    primary = spec.get("primary")
    threshold_robustness = spec.get("threshold_robustness")
    sensitivity = spec.get("sensitivity")
    noise_sensitivity = spec.get("noise_sensitivity")
    client_sensitivity = spec.get("client_sensitivity")
    controls = spec.get("controls")
    oracle_controls = spec.get("oracle_controls")
    named_sections = {
        "primary": primary,
        "threshold_robustness": threshold_robustness,
        "sensitivity": sensitivity,
        "noise_sensitivity": noise_sensitivity,
        "client_sensitivity": client_sensitivity,
        "controls": controls,
        "oracle_controls": oracle_controls,
    }
    if any(not isinstance(value, dict) for value in named_sections.values()):
        raise ValueError("all campaign matrix sections must be objects")
    assert isinstance(primary, dict)
    assert isinstance(threshold_robustness, dict)
    assert isinstance(sensitivity, dict)
    assert isinstance(noise_sensitivity, dict)
    assert isinstance(client_sensitivity, dict)
    assert isinstance(controls, dict)
    assert isinstance(oracle_controls, dict)
    require_exact_keys(primary, {"initial_clip_norms", "seeds", "methods"}, "primary")
    require_exact_keys(
        threshold_robustness,
        {"initial_clip_norms", "seeds", "methods"},
        "threshold_robustness",
    )
    require_exact_keys(
        sensitivity,
        {"initial_clip_norm", "seeds", "etas", "betas", "method", "exclude_primary_default"},
        "sensitivity",
    )
    require_exact_keys(
        noise_sensitivity,
        {"initial_clip_norm", "seeds", "noise_multipliers", "methods"},
        "noise_sensitivity",
    )
    require_exact_keys(
        client_sensitivity,
        {"initial_clip_norm", "seeds", "num_clients", "methods"},
        "client_sensitivity",
    )
    require_exact_keys(controls, {"initial_clip_norm", "seeds", "methods"}, "controls")
    require_exact_keys(
        oracle_controls,
        {"initial_clip_norm", "seeds", "etas", "betas", "method"},
        "oracle_controls",
    )

    primary_clip_norms = primary["initial_clip_norms"]
    primary_seeds = primary["seeds"]
    primary_methods = primary["methods"]
    if (
        not isinstance(primary_clip_norms, list)
        or not isinstance(primary_seeds, list)
        or not isinstance(primary_methods, list)
    ):
        raise ValueError("primary axes must be arrays")
    primary_clip_norms = [
        require_number(value, "primary.initial_clip_norm", positive=True)
        for value in primary_clip_norms
    ]
    primary_seeds = [require_int(value, "primary.seed", positive=False) for value in primary_seeds]
    if tuple(primary_methods) != (FIXED_DP_METHOD, FULL_SLACLIP_METHOD):
        raise ValueError("primary methods must be fixed DP then full SlaClip")
    if primary_clip_norms != [10.0]:
        raise ValueError("confirmatory primary must be the paper setting C=10")
    _unique(primary_clip_norms, "primary.initial_clip_norms")
    _unique(primary_seeds, "primary.seeds")

    arms: list[dict[str, Any]] = []
    default_eta = float(common["default_eta"])
    default_beta = float(common["default_beta"])

    def append_pair(
        *,
        family: str,
        stem: str,
        seed: int,
        clip_norm: float,
        num_clients: int | None = None,
        noise_multiplier: float | None = None,
    ) -> None:
        fixed_id = f"{stem}-fixed"
        arms.append(
            _base_arm(
                common=common,
                arm_id=fixed_id,
                family=family,
                method=FIXED_DP_METHOD,
                seed=seed,
                clip_norm=clip_norm,
                eta=None,
                beta=None,
                reference_arm_id=None,
                num_clients=num_clients,
                noise_multiplier=noise_multiplier,
            )
        )
        arms.append(
            _base_arm(
                common=common,
                arm_id=f"{stem}-slaclip",
                family=family,
                method=FULL_SLACLIP_METHOD,
                seed=seed,
                clip_norm=clip_norm,
                eta=default_eta,
                beta=default_beta,
                reference_arm_id=fixed_id,
                num_clients=num_clients,
                noise_multiplier=noise_multiplier,
            )
        )

    for clip_norm in primary_clip_norms:
        if not common["slaclip_c_min"] <= clip_norm <= common["slaclip_c_max"]:
            raise ValueError("primary initial threshold is outside controller bounds")
        clip_token = number_token(clip_norm)
        for seed in primary_seeds:
            append_pair(
                family="primary",
                stem=f"primary-c{clip_token}-s{seed}",
                seed=seed,
                clip_norm=clip_norm,
            )

    robustness_clips = threshold_robustness["initial_clip_norms"]
    robustness_seeds = threshold_robustness["seeds"]
    robustness_methods = threshold_robustness["methods"]
    if (
        not isinstance(robustness_clips, list)
        or not isinstance(robustness_seeds, list)
        or not isinstance(robustness_methods, list)
    ):
        raise ValueError("threshold robustness axes must be arrays")
    robustness_clips = [
        require_number(value, "threshold_robustness.initial_clip_norm", positive=True)
        for value in robustness_clips
    ]
    robustness_seeds = [
        require_int(value, "threshold_robustness.seed", positive=False)
        for value in robustness_seeds
    ]
    if tuple(robustness_methods) != (FIXED_DP_METHOD, FULL_SLACLIP_METHOD):
        raise ValueError("threshold robustness methods/order is invalid")
    _unique(robustness_clips, "threshold_robustness.initial_clip_norms")
    _unique(robustness_seeds, "threshold_robustness.seeds")
    if set(robustness_clips) & set(primary_clip_norms):
        raise ValueError("threshold robustness must not duplicate the primary C")
    for clip_norm in robustness_clips:
        if not common["slaclip_c_min"] <= clip_norm <= common["slaclip_c_max"]:
            raise ValueError("robustness threshold is outside controller bounds")
        clip_token = number_token(clip_norm)
        for seed in robustness_seeds:
            append_pair(
                family="threshold_robustness",
                stem=f"robust-c{clip_token}-s{seed}",
                seed=seed,
                clip_norm=clip_norm,
            )

    sensitivity_clip = require_number(
        sensitivity["initial_clip_norm"], "sensitivity.initial_clip_norm", positive=True
    )
    sensitivity_seeds = sensitivity["seeds"]
    etas = sensitivity["etas"]
    betas = sensitivity["betas"]
    if (
        not isinstance(sensitivity_seeds, list)
        or not isinstance(etas, list)
        or not isinstance(betas, list)
    ):
        raise ValueError("sensitivity axes must be arrays")
    sensitivity_seeds = [
        require_int(value, "sensitivity.seed", positive=False)
        for value in sensitivity_seeds
    ]
    etas = [require_number(value, "sensitivity.eta", positive=True) for value in etas]
    betas = [require_number(value, "sensitivity.beta") for value in betas]
    if any(value < 0 or value > 1 for value in betas):
        raise ValueError("sensitivity beta must lie in [0, 1]")
    _unique(sensitivity_seeds, "sensitivity.seeds")
    _unique(etas, "sensitivity.etas")
    _unique(betas, "sensitivity.betas")
    if (
        sensitivity["method"] != FULL_SLACLIP_METHOD
        or sensitivity["exclude_primary_default"] is not True
    ):
        raise ValueError("sensitivity must be de-duplicated full SlaClip")
    if sensitivity_clip not in primary_clip_norms or not set(
        sensitivity_seeds
    ).issubset(primary_seeds):
        raise ValueError("sensitivity references require matching primary baselines")
    clip_token = number_token(sensitivity_clip)
    for seed in sensitivity_seeds:
        reference_id = f"primary-c{clip_token}-s{seed}-fixed"
        for eta in etas:
            for beta in betas:
                if eta == default_eta and beta == default_beta:
                    continue
                arm_id = (
                    f"sensitivity-c{clip_token}-s{seed}-"
                    f"e{number_token(eta)}-b{number_token(beta)}"
                )
                arms.append(
                    _base_arm(
                        common=common,
                        arm_id=arm_id,
                        family="sensitivity",
                        method=FULL_SLACLIP_METHOD,
                        seed=seed,
                        clip_norm=sensitivity_clip,
                        eta=eta,
                        beta=beta,
                        reference_arm_id=reference_id,
                    )
                )

    noise_clip = require_number(
        noise_sensitivity["initial_clip_norm"],
        "noise_sensitivity.initial_clip_norm",
        positive=True,
    )
    noise_seeds = noise_sensitivity["seeds"]
    noise_values = noise_sensitivity["noise_multipliers"]
    noise_methods = noise_sensitivity["methods"]
    if (
        not isinstance(noise_seeds, list)
        or not isinstance(noise_values, list)
        or not isinstance(noise_methods, list)
    ):
        raise ValueError("noise sensitivity axes must be arrays")
    noise_seeds = [
        require_int(value, "noise_sensitivity.seed", positive=False)
        for value in noise_seeds
    ]
    noise_values = [
        require_number(
            value,
            "noise_sensitivity.noise_multiplier",
            positive=True,
        )
        for value in noise_values
    ]
    if tuple(noise_methods) != (FIXED_DP_METHOD, FULL_SLACLIP_METHOD):
        raise ValueError("noise sensitivity methods/order is invalid")
    _unique(noise_seeds, "noise_sensitivity.seeds")
    _unique(noise_values, "noise_sensitivity.noise_multipliers")
    if float(common["noise_multiplier"]) in noise_values:
        raise ValueError("noise sensitivity must not duplicate common sigma")
    for noise_multiplier in noise_values:
        sigma_token = number_token(noise_multiplier)
        for seed in noise_seeds:
            append_pair(
                family="noise_sensitivity",
                stem=(
                    f"noise-sigma{sigma_token}-c{number_token(noise_clip)}-s{seed}"
                ),
                seed=seed,
                clip_norm=noise_clip,
                noise_multiplier=noise_multiplier,
            )

    client_clip = require_number(
        client_sensitivity["initial_clip_norm"],
        "client_sensitivity.initial_clip_norm",
        positive=True,
    )
    client_seeds = client_sensitivity["seeds"]
    client_values = client_sensitivity["num_clients"]
    client_methods = client_sensitivity["methods"]
    if (
        not isinstance(client_seeds, list)
        or not isinstance(client_values, list)
        or not isinstance(client_methods, list)
    ):
        raise ValueError("client sensitivity axes must be arrays")
    client_seeds = [
        require_int(value, "client_sensitivity.seed", positive=False)
        for value in client_seeds
    ]
    client_values = [
        require_int(value, "client_sensitivity.num_clients")
        for value in client_values
    ]
    if tuple(client_methods) != (FIXED_DP_METHOD, FULL_SLACLIP_METHOD):
        raise ValueError("client sensitivity methods/order is invalid")
    _unique(client_seeds, "client_sensitivity.seeds")
    _unique(client_values, "client_sensitivity.num_clients")
    if int(common["num_clients"]) in client_values:
        raise ValueError("client sensitivity must not duplicate common N")
    for num_clients in client_values:
        for seed in client_seeds:
            append_pair(
                family="client_sensitivity",
                stem=(
                    f"clients-n{num_clients}-c{number_token(client_clip)}-s{seed}"
                ),
                seed=seed,
                clip_norm=client_clip,
                num_clients=num_clients,
            )

    control_clip = require_number(
        controls["initial_clip_norm"],
        "controls.initial_clip_norm",
        positive=True,
    )
    control_seeds = controls["seeds"]
    control_methods = controls["methods"]
    if not isinstance(control_seeds, list) or not isinstance(control_methods, list):
        raise ValueError("control axes must be arrays")
    control_seeds = [require_int(value, "controls.seed", positive=False) for value in control_seeds]
    if tuple(control_methods) != CONTROL_METHODS:
        raise ValueError("control methods/order is invalid")
    if control_clip not in primary_clip_norms or not set(control_seeds).issubset(
        primary_seeds
    ):
        raise ValueError("controls require matching primary configurations")
    control_token = number_token(control_clip)
    for seed in control_seeds:
        for method in control_methods:
            suffix = "nodp" if method == "no_dp_lora_control" else "cliponly"
            arms.append(
                _base_arm(
                    common=common,
                    arm_id=f"control-c{control_token}-s{seed}-{suffix}",
                    family="control",
                    method=method,
                    seed=seed,
                    clip_norm=control_clip,
                    eta=None,
                    beta=None,
                    reference_arm_id=f"primary-c{control_token}-s{seed}-fixed",
                )
            )

    oracle_clip = require_number(
        oracle_controls["initial_clip_norm"],
        "oracle_controls.initial_clip_norm",
        positive=True,
    )
    oracle_seeds = oracle_controls["seeds"]
    oracle_etas = oracle_controls["etas"]
    oracle_betas = oracle_controls["betas"]
    if (
        not isinstance(oracle_seeds, list)
        or not isinstance(oracle_etas, list)
        or not isinstance(oracle_betas, list)
    ):
        raise ValueError("oracle-control axes must be arrays")
    oracle_seeds = [
        require_int(value, "oracle_controls.seed", positive=False)
        for value in oracle_seeds
    ]
    oracle_etas = [
        require_number(value, "oracle_controls.eta") for value in oracle_etas
    ]
    oracle_betas = [
        require_number(value, "oracle_controls.beta") for value in oracle_betas
    ]
    if any(value < 0 for value in oracle_etas):
        raise ValueError("oracle-control eta must be non-negative")
    if any(value < 0 or value > 1 for value in oracle_betas):
        raise ValueError("oracle-control beta must lie in [0, 1]")
    _unique(oracle_seeds, "oracle_controls.seeds")
    _unique(oracle_etas, "oracle_controls.etas")
    _unique(oracle_betas, "oracle_controls.betas")
    if oracle_controls["method"] != ORACLE_SLACLIP_METHOD:
        raise ValueError("oracle_controls must use oracle_slaclip_control")
    if oracle_clip not in primary_clip_norms or not set(oracle_seeds).issubset(
        primary_seeds
    ):
        raise ValueError("oracle controls require matching primary fixed baselines")
    oracle_token = number_token(oracle_clip)
    for seed in oracle_seeds:
        reference_id = f"primary-c{oracle_token}-s{seed}-fixed"
        for eta in oracle_etas:
            for beta in oracle_betas:
                arms.append(
                    _base_arm(
                        common=common,
                        arm_id=(
                            f"oracle-c{oracle_token}-s{seed}-"
                            f"e{number_token(eta)}-b{number_token(beta)}"
                        ),
                        family="oracle_control",
                        method=ORACLE_SLACLIP_METHOD,
                        seed=seed,
                        clip_norm=oracle_clip,
                        eta=eta,
                        beta=beta,
                        reference_arm_id=reference_id,
                    )
                )

    expected = require_int(spec["expected_arm_count"], "expected_arm_count")
    ids = [arm["arm_id"] for arm in arms]
    if len(arms) != expected or len(ids) != len(set(ids)):
        raise ValueError(
            f"expanded matrix has {len(arms)} arms, expected {expected}, "
            "or duplicate IDs"
        )
    if len(arms) % 2:
        raise ValueError("two-lane campaign requires an even number of arms")
    for index, arm in enumerate(arms):
        arm["index"] = index
        arm["wave"] = index // 2
        arm["lane"] = index % 2
    return arms


def build_runtime_manifest(
    spec_path: Path,
    *,
    repository_sha: str,
    input_manifest: Path,
    created_at_utc: str,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    arms = expand_spec(spec)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_name": spec["campaign_name"],
        "created_at_utc": created_at_utc,
        "repository_sha": repository_sha,
        "spec_sha256": sha256_file(spec_path),
        "input_manifest_path": str(input_manifest.resolve()),
        "input_manifest_sha256": sha256_file(input_manifest),
        "expected_arm_count": spec["expected_arm_count"],
        "scientific_boundary": spec["scientific_boundary"],
        "arms": arms,
    }
    payload["manifest_sha256"] = sha256_bytes(canonical_bytes(payload))
    return payload


def validate_runtime_manifest(value: dict[str, Any]) -> None:
    fingerprint = value.get("manifest_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise RuntimeError("runtime manifest fingerprint is missing")
    unsigned = dict(value)
    del unsigned["manifest_sha256"]
    if sha256_bytes(canonical_bytes(unsigned)) != fingerprint:
        raise RuntimeError("runtime manifest fingerprint mismatch")
    arms = value.get("arms")
    expected = value.get("expected_arm_count")
    if not isinstance(arms, list) or len(arms) != expected:
        raise RuntimeError("runtime manifest arm count mismatch")
    ids = [arm.get("arm_id") for arm in arms if isinstance(arm, dict)]
    if len(ids) != len(arms) or len(ids) != len(set(ids)):
        raise RuntimeError("runtime manifest arm identities are invalid")


def repository_sha(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def repository_dirty(repository: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return bool(result.stdout.strip())


def absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def validate_or_create_key(path: Path, *, create: bool) -> None:
    path = absolute_path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if not path.exists():
        if not create:
            raise RuntimeError(f"campaign RNG key is missing: {path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(secrets.token_bytes(CAMPAIGN_KEY_BYTES))
                handle.flush()
                os.fsync(handle.fileno())
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            if path.exists():
                path.unlink()
            raise
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or metadata.st_size != CAMPAIGN_KEY_BYTES
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise RuntimeError("campaign RNG key must be user-owned, 32 bytes, and mode 0600")


def prepare_campaign(args: argparse.Namespace) -> None:
    campaign_root = args.campaign_root.resolve()
    repository = args.repository.resolve()
    input_manifest = args.input_manifest.resolve()
    spec_path = args.spec.resolve()
    if repository_sha(repository) != args.expected_code_sha or repository_dirty(repository):
        raise RuntimeError("repository snapshot SHA/cleanliness gate failed")
    runtime_path = campaign_root / RUNTIME_MANIFEST_NAME
    if args.resume:
        if not campaign_root.is_dir() or not runtime_path.is_file():
            raise RuntimeError("resume requires an existing runtime manifest")
        existing = load_object(runtime_path, "runtime campaign manifest")
        validate_runtime_manifest(existing)
        candidate = build_runtime_manifest(
            spec_path,
            repository_sha=args.expected_code_sha,
            input_manifest=input_manifest,
            created_at_utc=str(existing.get("created_at_utc")),
        )
        if existing != candidate:
            raise RuntimeError("resume inputs do not match the immutable runtime manifest")
    else:
        if campaign_root.exists():
            raise RuntimeError(f"refusing to overwrite campaign root: {campaign_root}")
        campaign_root.mkdir(parents=True, mode=0o700)
        runtime = build_runtime_manifest(
            spec_path,
            repository_sha=args.expected_code_sha,
            input_manifest=input_manifest,
            created_at_utc=utc_now(),
        )
        atomic_json(runtime_path, runtime)
    for directory in ("arms", "arm-status", "arm-logs", "control", "tmp", "preflight"):
        (campaign_root / directory).mkdir(mode=0o700, exist_ok=True)
    global_stop = campaign_root / "control" / "stop.request"
    if global_stop.exists():
        global_stop.unlink()
    validate_or_create_key(absolute_path(args.private_key), create=not args.resume)
    print(f"runtime_manifest={runtime_path}")


def load_runtime(path: Path) -> dict[str, Any]:
    value = load_object(path, "runtime campaign manifest")
    validate_runtime_manifest(value)
    return value


def _oracle_noisy_match_key(arm: Mapping[str, Any]) -> bytes:
    missing = [name for name in ORACLE_NOISY_MATCH_FIELDS if name not in arm]
    if missing:
        raise RuntimeError(
            f"oracle/noisy arm is missing match fields: "
            f"{arm.get('arm_id', '<unknown>')} {missing}"
        )
    return canonical_bytes(
        {name: arm[name] for name in ORACLE_NOISY_MATCH_FIELDS}
    )


def resolve_oracle_noisy_arm_pairs(
    runtime: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Resolve exact-CDF oracle arms to unique noisy-CDF full-SlaClip arms.

    The method and controller input intentionally differ.  Every scientific
    and randomization setting that can otherwise affect the trajectory must
    be identical.  Missing or ambiguous manifest counterparts are rejected
    before any completed metrics are interpreted.
    """

    raw_arms = runtime.get("arms")
    if not isinstance(raw_arms, list) or any(
        not isinstance(arm, dict) for arm in raw_arms
    ):
        raise RuntimeError("runtime manifest arms are invalid")
    arms = list(raw_arms)
    noisy_by_key: dict[bytes, list[Mapping[str, Any]]] = {}
    for arm in arms:
        if arm["method"] != FULL_SLACLIP_METHOD:
            continue
        if arm.get("controller_input") != NOISY_CONTROLLER_INPUT:
            raise RuntimeError(
                f"full SlaClip arm has invalid controller input: {arm['arm_id']}"
            )
        noisy_by_key.setdefault(_oracle_noisy_match_key(arm), []).append(arm)

    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for oracle in arms:
        if oracle["method"] != ORACLE_SLACLIP_METHOD:
            continue
        if oracle.get("controller_input") != EXACT_CONTROLLER_INPUT:
            raise RuntimeError(
                f"oracle SlaClip arm has invalid controller input: {oracle['arm_id']}"
            )
        matches = noisy_by_key.get(_oracle_noisy_match_key(oracle), [])
        if len(matches) != 1:
            raise RuntimeError(
                "oracle/noisy manifest match must be unique: "
                f"oracle={oracle['arm_id']} matched_noisy_arms="
                f"{[arm['arm_id'] for arm in matches]}"
            )
        pairs.append((oracle, matches[0]))
    return pairs


def mark_job_status(args: argparse.Namespace) -> None:
    """Atomically record the lifecycle of one Slurm campaign allocation.

    ``campaign_summary.json`` is intentionally an incremental scientific
    aggregate and can therefore remain ``IN_PROGRESS`` after a batch failure.
    This separate marker is authoritative for the allocation lifecycle.  A
    job may only terminate its own RUNNING attempt, preventing a late trap from
    an older allocation from overwriting a newer resumed allocation.
    """

    campaign_root = args.campaign_root.resolve()
    runtime_manifest = args.runtime_manifest.resolve()
    status_path = campaign_root / JOB_STATUS_NAME
    if not campaign_root.is_dir() or not runtime_manifest.is_file():
        raise RuntimeError("job status requires a prepared campaign manifest")
    if args.status not in {"RUNNING", "COMPLETED", "FAILED"}:
        raise ValueError(f"unsupported campaign job status: {args.status}")
    if not args.slurm_job_id or any(
        character not in "0123456789_" for character in args.slurm_job_id
    ):
        raise ValueError("Slurm job ID must contain only digits and underscores")
    if (
        len(args.repository_sha) != 40
        or any(character not in "0123456789abcdef" for character in args.repository_sha)
    ):
        raise ValueError("repository SHA must be 40 lowercase hexadecimal characters")
    if not args.reason or any(character in "\r\n" for character in args.reason):
        raise ValueError("job-status reason must be a non-empty single line")
    if args.status == "RUNNING":
        if args.exit_code is not None:
            raise ValueError("RUNNING job status cannot have an exit code")
    else:
        if args.exit_code is None or args.exit_code < 0:
            raise ValueError("terminal job status requires a non-negative exit code")
        if args.status == "COMPLETED" and args.exit_code != 0:
            raise ValueError("COMPLETED job status requires exit code zero")
        if args.status == "FAILED" and args.exit_code == 0:
            raise ValueError("FAILED job status requires a nonzero exit code")

    runtime = load_runtime(runtime_manifest)
    if runtime["repository_sha"] != args.repository_sha:
        raise RuntimeError("job status repository SHA differs from runtime manifest")
    manifest_sha256 = sha256_file(runtime_manifest)
    now = utc_now()
    existing: dict[str, Any] | None = None
    if status_path.exists():
        existing = load_object(status_path, "campaign job status")
        required_existing = {
            "schema_version",
            "status",
            "attempt_number",
            "slurm_job_id",
            "repository_sha",
            "runtime_manifest_sha256",
            "started_at_utc",
        }
        if not required_existing.issubset(existing):
            raise RuntimeError("existing campaign job status is incomplete")
        if (
            existing["schema_version"] != SCHEMA_VERSION
            or existing["repository_sha"] != args.repository_sha
            or existing["runtime_manifest_sha256"] != manifest_sha256
        ):
            raise RuntimeError("existing campaign job status identity differs")

    if args.status == "RUNNING":
        if existing is not None and existing["status"] == "RUNNING":
            if existing["slurm_job_id"] != args.slurm_job_id:
                raise RuntimeError("another Slurm allocation already owns this campaign")
            attempt_number = int(existing["attempt_number"])
            started_at_utc = str(existing["started_at_utc"])
        else:
            attempt_number = (
                int(existing["attempt_number"]) + 1 if existing is not None else 1
            )
            started_at_utc = now
        terminal_at_utc = None
        exit_code = None
        resumable = False
    else:
        if (
            existing is None
            or existing["status"] != "RUNNING"
            or existing["slurm_job_id"] != args.slurm_job_id
        ):
            raise RuntimeError("only the owning RUNNING allocation may write a terminal status")
        attempt_number = int(existing["attempt_number"])
        started_at_utc = str(existing["started_at_utc"])
        terminal_at_utc = now
        exit_code = int(args.exit_code)
        resumable = args.status == "FAILED" and exit_code == 75

    atomic_json(
        status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": args.status,
            "attempt_number": attempt_number,
            "slurm_job_id": args.slurm_job_id,
            "hostname": socket.gethostname(),
            "repository_sha": args.repository_sha,
            "runtime_manifest": str(runtime_manifest),
            "runtime_manifest_sha256": manifest_sha256,
            "campaign_root": str(campaign_root),
            "started_at_utc": started_at_utc,
            "updated_at_utc": now,
            "terminal_at_utc": terminal_at_utc,
            "exit_code": exit_code,
            "reason": args.reason,
            "resumable": resumable,
        },
    )


def _confirmed_terminal_slurm_state(slurm_job_id: str) -> str:
    """Return a scheduler-confirmed terminal state or fail closed.

    ``squeue`` is checked first so an active owner can never be taken over.
    ``sacct -X`` then supplies allocation-level terminal evidence.  A state in
    transition, missing accounting row, or scheduler command failure requires
    the operator to retry rather than guessing that the owner is stale.
    """

    try:
        queued = subprocess.run(
            [
                "squeue",
                "--noheader",
                f"--jobs={slurm_job_id}",
                "--format=%i|%T",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError("could not query squeue for stale-owner recovery") from error
    queue_reports_absent = (
        queued.returncode != 0
        and not queued.stdout.strip()
        and "invalid job id" in queued.stderr.lower()
    )
    if queued.returncode != 0 and not queue_reports_absent:
        raise RuntimeError(
            "squeue failed during stale-owner recovery: "
            + queued.stderr.strip()
        )
    active_rows = [line.strip() for line in queued.stdout.splitlines() if line.strip()]
    if active_rows:
        raise RuntimeError(
            "the existing RUNNING campaign owner is still present in squeue: "
            + ";".join(active_rows)
        )

    try:
        accounted = subprocess.run(
            [
                "sacct",
                "-X",
                f"--jobs={slurm_job_id}",
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError("could not query sacct for stale-owner recovery") from error
    if accounted.returncode != 0:
        raise RuntimeError(
            "sacct failed during stale-owner recovery: "
            + accounted.stderr.strip()
        )
    exact_states: list[str] = []
    for raw_line in accounted.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split("|", 1)
        if len(fields) != 2 or fields[0] != slurm_job_id:
            continue
        state = fields[1].strip().split()[0].rstrip("+") if fields[1].strip() else ""
        if state:
            exact_states.append(state)
    if len(exact_states) != 1:
        raise RuntimeError(
            "sacct did not return exactly one allocation-level stale-owner row"
        )
    state = exact_states[0]
    if state not in TERMINAL_SLURM_STATES:
        raise RuntimeError(
            f"the previous Slurm owner is not terminal in sacct: {state}"
        )
    return state


def recover_stale_job_status(args: argparse.Namespace) -> None:
    """Close a stale RUNNING marker only after scheduler terminal evidence."""

    campaign_root = args.campaign_root.resolve()
    runtime_manifest = args.runtime_manifest.resolve()
    status_path = campaign_root / JOB_STATUS_NAME
    if not campaign_root.is_dir() or not runtime_manifest.is_file():
        raise RuntimeError("stale-owner recovery requires a prepared campaign")
    runtime = load_runtime(runtime_manifest)
    if runtime["repository_sha"] != args.repository_sha:
        raise RuntimeError(
            "stale-owner recovery repository SHA differs from runtime manifest"
        )
    runtime_manifest_sha256 = sha256_file(runtime_manifest)
    if not status_path.exists():
        print(json.dumps({"status": "NO_EXISTING_JOB_STATUS"}, sort_keys=True))
        return
    existing = load_object(status_path, "campaign job status")
    if (
        existing.get("schema_version") != SCHEMA_VERSION
        or existing.get("repository_sha") != args.repository_sha
        or existing.get("runtime_manifest_sha256") != runtime_manifest_sha256
    ):
        raise RuntimeError("stale campaign owner identity differs")
    if existing.get("status") != "RUNNING":
        print(
            json.dumps(
                {
                    "status": "NO_STALE_RUNNING_OWNER",
                    "existing_status": existing.get("status"),
                },
                sort_keys=True,
            )
        )
        return
    stale_job_id = existing.get("slurm_job_id")
    if not isinstance(stale_job_id, str) or not stale_job_id or any(
        character not in "0123456789_" for character in stale_job_id
    ):
        raise RuntimeError("stale campaign owner has an invalid Slurm job ID")
    scheduler_state = _confirmed_terminal_slurm_state(stale_job_id)
    reason = f"scheduler_confirmed_{scheduler_state.lower()}_before_resume"
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "STALE_RUNNING_OWNER_TERMINAL_CONFIRMED",
                    "stale_slurm_job_id": stale_job_id,
                    "scheduler_state": scheduler_state,
                },
                sort_keys=True,
            )
        )
        return
    mark_job_status(
        argparse.Namespace(
            campaign_root=campaign_root,
            runtime_manifest=runtime_manifest,
            status="FAILED",
            slurm_job_id=stale_job_id,
            repository_sha=args.repository_sha,
            reason=reason,
            exit_code=75,
        )
    )
    print(
        json.dumps(
            {
                "status": "STALE_RUNNING_OWNER_RECOVERED",
                "stale_slurm_job_id": stale_job_id,
                "scheduler_state": scheduler_state,
                "job_status": str(status_path),
            },
            sort_keys=True,
        )
    )


def _arm_command(
    arm: Mapping[str, Any],
    *,
    repository: Path,
    python_bin: Path,
    input_manifest: Path,
    output_dir: Path,
    private_key: Path,
    stop_file: Path,
) -> list[str]:
    command = [
        str(python_bin),
        str(repository / "paper_repro" / "train_federated.py"),
        "--input-manifest",
        str(input_manifest),
        "--output-dir",
        str(output_dir),
        "--models",
        *[str(model) for model in arm["models"]],
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
        "--clip-norm",
        str(arm["initial_clip_norm"]),
        "--rank",
        str(arm["rank"]),
        "--max-seq-length",
        str(arm["max_seq_length"]),
        "--max-validation-records",
        str(arm["max_validation_records"]),
        "--seed",
        str(arm["seed"]),
        "--data-split-seed",
        str(arm["data_split_seed"]),
        "--evaluation-seed",
        str(arm["evaluation_seed"]),
        "--delta",
        str(arm["delta"]),
        "--eval-every",
        str(arm["eval_every"]),
        "--checkpoint-every",
        str(arm["checkpoint_every"]),
        "--private-rng-key",
        str(private_key),
        "--rng-domain",
        str(arm["rng_domain"]),
        "--pair-noise-across-methods",
        "--stop-file",
        str(stop_file),
        "--acknowledge-non-dp-diagnostics",
    ]
    if arm["method"] in ADAPTIVE_METHODS:
        command.extend(
            [
                "--slaclip-eta",
                str(arm["slaclip_eta"]),
                "--slaclip-base-target-clipped-fraction",
                str(arm["slaclip_base_target_clipped_fraction"]),
                "--slaclip-num-slots",
                str(arm["slaclip_num_slots"]),
                "--slaclip-c-min",
                str(arm["slaclip_c_min"]),
                "--slaclip-c-max",
                str(arm["slaclip_c_max"]),
            ]
        )
    if output_dir.exists():
        command.append("--resume")
    return command


def _write_arm_status(path: Path, arm: Mapping[str, Any], status: str, **extra: Any) -> None:
    prior_started = None
    if path.exists():
        try:
            prior_started = load_object(path, "arm status").get("first_started_at_utc")
        except RuntimeError:
            prior_started = None
    value = {
        "schema_version": SCHEMA_VERSION,
        "arm_id": arm["arm_id"],
        "index": arm["index"],
        "method": arm["method"],
        "status": status,
        "first_started_at_utc": prior_started or utc_now(),
        "updated_at_utc": utc_now(),
        **extra,
    }
    atomic_json(path, value)


def run_arm(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    runtime = load_runtime(manifest_path)
    repository = args.repository.resolve()
    if repository_sha(repository) != runtime["repository_sha"] or repository_dirty(repository):
        raise RuntimeError("repository snapshot changed after campaign preparation")
    index = args.arm_index
    arms = runtime["arms"]
    if index < 0 or index >= len(arms):
        raise ValueError(f"arm index is out of range: {index}")
    arm = arms[index]
    status_identity = {
        "runtime_manifest_sha256": runtime["manifest_sha256"],
        "repository_sha": runtime["repository_sha"],
        "arm_spec_sha256": sha256_bytes(canonical_bytes(arm)),
    }
    campaign_root = manifest_path.parent
    output_dir = campaign_root / "arms" / arm["arm_id"]
    status_path = campaign_root / "arm-status" / f"{arm['arm_id']}.json"
    log_out = campaign_root / "arm-logs" / f"{arm['arm_id']}.out"
    log_err = campaign_root / "arm-logs" / f"{arm['arm_id']}.err"
    stop_file = campaign_root / "control" / f"{arm['arm_id']}.stop"
    global_stop = campaign_root / "control" / "stop.request"
    final_summary_path = output_dir / "final_summary.json"
    if status_path.is_file() and final_summary_path.is_file():
        try:
            prior_status = load_object(status_path, "arm status")
            final_summary = load_object(final_summary_path, "arm final summary")
        except RuntimeError:
            prior_status = {}
            final_summary = {}
        if (
            prior_status.get("status") == "COMPLETED"
            and prior_status.get("arm_id") == arm["arm_id"]
            and prior_status.get("index") == arm["index"]
            and prior_status.get("method") == arm["method"]
            and all(
                prior_status.get(name) == value
                for name, value in status_identity.items()
            )
            and prior_status.get("final_summary_sha256")
            == sha256_file(final_summary_path)
            and final_summary.get("status") == "COMPLETED"
            and final_summary.get("method") == arm["method"]
        ):
            print(
                f"arm_reused_without_model_reload={arm['arm_id']} "
                f"final_summary_sha256={prior_status['final_summary_sha256']}"
            )
            return 0
    if stop_file.exists():
        stop_file.unlink()
    private_key = absolute_path(args.private_key)
    validate_or_create_key(private_key, create=False)
    command = _arm_command(
        arm,
        repository=repository,
        python_bin=args.python_bin.resolve(),
        input_manifest=Path(runtime["input_manifest_path"]),
        output_dir=output_dir,
        private_key=private_key,
        stop_file=stop_file,
    )
    _write_arm_status(
        status_path,
        arm,
        "RUNNING",
        **status_identity,
        resumed=output_dir.exists(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        hostname=socket.gethostname(),
    )

    stop_watcher_done = threading.Event()

    def request_stop(_signum: int | None = None, _frame: Any = None) -> None:
        stop_file.touch(mode=0o600, exist_ok=True)

    def watch_global_stop() -> None:
        while not stop_watcher_done.wait(1.0):
            if global_stop.exists():
                request_stop()
                return

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    watcher = threading.Thread(target=watch_global_stop, daemon=True)
    watcher.start()

    arm_tmp = campaign_root / "tmp" / arm["arm_id"]
    arm_tmp.mkdir(parents=True, mode=0o700, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "TMPDIR": str(arm_tmp),
            "TORCH_EXTENSIONS_DIR": str(arm_tmp / "torch-extensions"),
            "TRITON_CACHE_DIR": str(arm_tmp / "triton"),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    (arm_tmp / "torch-extensions").mkdir(mode=0o700, exist_ok=True)
    (arm_tmp / "triton").mkdir(mode=0o700, exist_ok=True)
    try:
        with log_out.open("ab") as stdout_handle, log_err.open("ab") as stderr_handle:
            line = (
                f"\n[{utc_now()}] starting arm={arm['arm_id']} "
                f"resume={output_dir.exists()}\n"
            )
            stdout_handle.write(line.encode())
            stdout_handle.flush()
            completed = subprocess.run(
                command,
                cwd=repository,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        return_code = int(completed.returncode)
    except BaseException as error:
        _write_arm_status(
            status_path,
            arm,
            "FAILED",
            **status_identity,
            exit_code=None,
            exception_type=type(error).__name__,
        )
        raise
    finally:
        stop_watcher_done.set()
        watcher.join(timeout=2.0)

    if return_code == 0:
        final_summary = output_dir / "final_summary.json"
        summary = load_object(final_summary, "arm final summary")
        if summary.get("status") != "COMPLETED" or summary.get("method") != arm["method"]:
            _write_arm_status(
                status_path,
                arm,
                "FAILED",
                **status_identity,
                exit_code=0,
                validation_error="zero_exit_without_matching_completed_summary",
            )
            raise RuntimeError(
                "arm returned zero without a matching completed summary: "
                f"{arm['arm_id']}"
            )
        _write_arm_status(
            status_path,
            arm,
            "COMPLETED",
            **status_identity,
            exit_code=0,
            final_summary_sha256=sha256_file(final_summary),
        )
    elif return_code == 75:
        _write_arm_status(
            status_path,
            arm,
            "CHECKPOINTED_STOP",
            **status_identity,
            exit_code=75,
        )
    else:
        _write_arm_status(
            status_path,
            arm,
            "FAILED",
            **status_identity,
            exit_code=return_code,
        )
    return return_code


def run_preflight_smoke(args: argparse.Namespace) -> int:
    """Run or deeply revalidate one real-model smoke inside the allocation."""

    manifest_path = args.manifest.resolve()
    runtime = load_runtime(manifest_path)
    repository = args.repository.resolve()
    if repository_sha(repository) != runtime["repository_sha"] or repository_dirty(repository):
        raise RuntimeError("repository snapshot changed before real-model smoke")
    desired_method = (
        args.method
        if args.method is not None
        else FIXED_DP_METHOD
        if args.lane == 0
        else FULL_SLACLIP_METHOD
    )
    template_method = (
        FULL_SLACLIP_METHOD
        if desired_method == ORACLE_SLACLIP_METHOD
        else desired_method
    )
    candidates = [
        arm
        for arm in runtime["arms"]
        if arm["family"] == "primary"
        and arm["method"] == template_method
        and arm["initial_clip_norm"] == 10.0
    ]
    if not candidates:
        raise RuntimeError("could not resolve a paper-style smoke template")
    smoke_seed = min(int(arm["seed"]) for arm in candidates)
    selected = [arm for arm in candidates if int(arm["seed"]) == smoke_seed]
    if len(selected) != 1:
        raise RuntimeError("paper-style smoke template is not unique at the first seed")
    arm = dict(selected[0])
    labels = {
        FIXED_DP_METHOD: "paper-baseline",
        FULL_SLACLIP_METHOD: "full-slaclip",
        ORACLE_SLACLIP_METHOD: "oracle-slaclip-control",
    }
    label = labels[desired_method]
    if desired_method == ORACLE_SLACLIP_METHOD:
        arm["method"] = ORACLE_SLACLIP_METHOD
        arm["analysis_role"] = (
            "non_dp_exact_endpoint_oracle_controller_diagnostic"
        )
        arm["controller_input"] = EXACT_CONTROLLER_INPUT
    arm["arm_id"] = f"preflight-{label}"
    arm["rng_domain"] = "full-slaclip-cdf:preflight:c10"
    campaign_root = manifest_path.parent
    output_dir = campaign_root / "preflight" / label
    status_path = campaign_root / "preflight" / f"{label}-status.json"
    log_out = campaign_root / "preflight" / f"{label}.out"
    log_err = campaign_root / "preflight" / f"{label}.err"
    stop_file = campaign_root / "control" / f"preflight-{label}.stop"
    global_stop = campaign_root / "control" / "stop.request"
    if stop_file.exists():
        stop_file.unlink()
    private_key = absolute_path(args.private_key)
    validate_or_create_key(private_key, create=False)
    command = _arm_command(
        arm,
        repository=repository,
        python_bin=args.python_bin.resolve(),
        input_manifest=Path(runtime["input_manifest_path"]),
        output_dir=output_dir,
        private_key=private_key,
        stop_file=stop_file,
    )
    command.append("--smoke")
    atomic_json(
        status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "RUNNING",
            "label": label,
            "method": desired_method,
            "repository_sha": runtime["repository_sha"],
            "runtime_manifest_sha256": runtime["manifest_sha256"],
            "resumed": output_dir.exists(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": socket.gethostname(),
            "updated_at_utc": utc_now(),
        },
    )

    watcher_done = threading.Event()

    def request_stop(_signum: int | None = None, _frame: Any = None) -> None:
        stop_file.touch(mode=0o600, exist_ok=True)

    def watch_global_stop() -> None:
        while not watcher_done.wait(1.0):
            if global_stop.exists():
                request_stop()
                return

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    watcher = threading.Thread(target=watch_global_stop, daemon=True)
    watcher.start()
    smoke_tmp = campaign_root / "tmp" / f"preflight-{label}"
    smoke_tmp.mkdir(parents=True, mode=0o700, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "TMPDIR": str(smoke_tmp),
            "TORCH_EXTENSIONS_DIR": str(smoke_tmp / "torch-extensions"),
            "TRITON_CACHE_DIR": str(smoke_tmp / "triton"),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    (smoke_tmp / "torch-extensions").mkdir(mode=0o700, exist_ok=True)
    (smoke_tmp / "triton").mkdir(mode=0o700, exist_ok=True)
    try:
        with log_out.open("ab") as stdout_handle, log_err.open("ab") as stderr_handle:
            line = (
                f"\n[{utc_now()}] real-model smoke method={desired_method} "
                f"resume={output_dir.exists()}\n"
            )
            stdout_handle.write(line.encode())
            stdout_handle.flush()
            completed = subprocess.run(
                command,
                cwd=repository,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        return_code = int(completed.returncode)
    finally:
        watcher_done.set()
        watcher.join(timeout=2.0)

    status = "FAILED"
    extra: dict[str, Any] = {"exit_code": return_code}
    if return_code == 0:
        final_summary = output_dir / "final_summary.json"
        summary = load_object(final_summary, "real-model smoke final summary")
        if summary.get("status") != "COMPLETED" or summary.get("method") != desired_method:
            raise RuntimeError("real-model smoke returned zero without a valid summary")
        status = "COMPLETED"
        extra["final_summary_sha256"] = sha256_file(final_summary)
    elif return_code == 75:
        status = "CHECKPOINTED_STOP"
    atomic_json(
        status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "label": label,
            "method": desired_method,
            "repository_sha": runtime["repository_sha"],
            "runtime_manifest_sha256": runtime["manifest_sha256"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": socket.gethostname(),
            "updated_at_utc": utc_now(),
            **extra,
        },
    )
    return return_code


def _quantile_median(container: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = container.get(name)
        if isinstance(value, dict):
            quantiles = value.get("quantiles")
            if isinstance(quantiles, dict):
                median = quantiles.get("0.5")
                if (
                    isinstance(median, (int, float))
                    and not isinstance(median, bool)
                    and math.isfinite(float(median))
                ):
                    return float(median)
    return None


def _model_metrics(
    arm: Mapping[str, Any],
    model: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    evaluations = summary.get("evaluations")
    initial_loss = final_loss = best_loss = normalized_loss_auc = None
    best_round = final_minus_best = None
    if isinstance(evaluations, list) and evaluations:
        points = [
            (int(value["round"]), float(value["loss"]))
            for value in evaluations
            if isinstance(value, dict)
            and isinstance(value.get("round"), int)
            and isinstance(value.get("loss"), (int, float))
            and not isinstance(value.get("loss"), bool)
            and math.isfinite(float(value["loss"]))
        ]
        if len(points) == len(evaluations) and points:
            first, last = points[0], points[-1]
            initial_loss = first[1]
            final_loss = last[1]
            best_round, best_loss = min(points, key=lambda item: (item[1], item[0]))
            final_minus_best = final_loss - best_loss
            span = last[0] - first[0]
            if span > 0:
                normalized_loss_auc = math.fsum(
                    (right_round - left_round) * (left_loss + right_loss) / 2.0
                    for (left_round, left_loss), (right_round, right_loss) in zip(
                        points,
                        points[1:],
                    )
                ) / span
    clipping = summary.get("clipping")
    clipping = clipping if isinstance(clipping, dict) else {}
    any_group = clipping.get("any_group")
    any_group = any_group if isinstance(any_group, dict) else {}
    behavior = summary.get("behavior_summary")
    behavior = behavior if isinstance(behavior, dict) else {}
    behavior_groups = behavior.get("groups")
    behavior_groups = behavior_groups if isinstance(behavior_groups, dict) else {}
    extension = summary.get("slaclip")
    extension = extension if isinstance(extension, dict) else {}
    controller = extension.get("controller_summary")
    controller = controller if isinstance(controller, dict) else {}
    groups = controller.get("groups")
    groups = groups if isinstance(groups, dict) else {}
    result: dict[str, Any] = {
        "arm_id": arm["arm_id"],
        "family": arm["family"],
        "analysis_role": arm["analysis_role"],
        "method": arm["method"],
        "controller_input": arm["controller_input"],
        "seed": arm["seed"],
        "num_clients": arm["num_clients"],
        "noise_multiplier": arm["noise_multiplier"],
        "effective_gradient_noise_multiplier": (
            0.0 if arm["method"] in CONTROL_METHODS else arm["noise_multiplier"]
        ),
        "initial_clip_norm": arm["initial_clip_norm"],
        "slaclip_num_slots": arm["slaclip_num_slots"],
        "slaclip_eta": arm["slaclip_eta"],
        "slaclip_base_target_clipped_fraction": arm[
            "slaclip_base_target_clipped_fraction"
        ],
        "slaclip_beta": arm["slaclip_beta"],
        "reference_arm_id": arm["reference_arm_id"],
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_round": best_round,
        "final_minus_best": final_minus_best,
        "normalized_loss_auc": normalized_loss_auc,
        "loss_delta": (
            final_loss - initial_loss
            if initial_loss is not None and final_loss is not None
            else None
        ),
        "actual_clipped_fraction": any_group.get("fraction"),
        "would_clip_fraction": any_group.get("would_fraction"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "sample_schedule_sha256": behavior.get("sample_schedule_sha256"),
        "supervision_schedule_sha256": behavior.get(
            "supervision_schedule_sha256"
        ),
        "cdf_endpoint_noise_std_theoretical": (
            float(
                arm["noise_multiplier"]
                * math.sqrt(arm["slaclip_num_slots"] / arm["num_clients"])
            )
            if arm["method"] in ADAPTIVE_METHODS
            else None
        ),
    }
    aliases = {
        "cdf_near_threshold_median": ("near_threshold_proxy",),
        "cdf_near_zero_median": ("near_zero_proxy",),
        "near_zero_adjusted_median": ("near_zero_adjusted",),
        "remaining_non_small_gradient_fraction_median": (
            "remaining_non_small_gradient_fraction",
        ),
        "raw_dynamic_target_clipped_median": (
            "raw_dynamic_target_clipped",
        ),
        "dynamic_target_median": ("dynamic_target_unclipped",),
        "dynamic_target_clipped_median": ("dynamic_target_clipped",),
        "controller_error_median": ("controller_error",),
        "near_threshold_proxy_error_median": ("near_threshold_proxy_error",),
        "near_zero_proxy_error_median": ("near_zero_proxy_error",),
        "raw_log_step_median": ("raw_log_step",),
        "cdf_error_mae_median": ("cdf_error_mae",),
        "cdf_error_rmse_median": ("cdf_error_rmse",),
        "cdf_error_max_abs_median": ("cdf_error_max_abs",),
        "cdf_error_z_rmse_median": ("cdf_error_z_rmse",),
        "oracle_target_clipped_median": ("oracle_dynamic_target_clipped",),
        "oracle_log_step_median": ("oracle_raw_log_step",),
        "oracle_next_threshold_median": ("oracle_next_clip_threshold",),
        "noisy_minus_oracle_log_step_median": (
            "noisy_minus_oracle_raw_log_step",
        ),
        "noisy_oracle_log_threshold_error_median": (
            "noisy_oracle_log_threshold_error",
        ),
        "actual_target_absolute_error_median": (
            "actual_target_absolute_error",
        ),
        "threshold_used_median": ("clip_threshold_used",),
        "next_threshold_median": ("next_clip_threshold",),
    }
    behavior_metric_names = (
        "raw_gradient_l2",
        "raw_to_threshold_ratio",
        "removed_gradient_l2",
        "retained_energy_fraction",
        "clipped_signal_gradient_l2",
        "noise_gradient_l2",
        "signal_to_noise_l2_ratio",
        "signal_noise_cosine",
        "global_update_l2",
        "aggregate_signal_gradient_l2",
        "aggregate_noise_gradient_l2",
        "aggregate_signal_to_noise_l2_ratio",
        "relative_global_update",
    )
    controller_count_names = (
        "gamma_clamped_low_count",
        "gamma_clamped_high_count",
        "log_step_bounded_count",
        "lower_bound_hits",
        "upper_bound_hits",
        "noisy_adjacent_monotonicity_violations",
        "exact_adjacent_monotonicity_violations",
        "log_step_direction_flip_count",
        "oracle_direction_agreement_count",
        "noisy_cdf_out_of_range_count",
    )
    for group_name in ("A", "B"):
        group = groups.get(group_name)
        group = group if isinstance(group, dict) else {}
        behavior_group = behavior_groups.get(group_name)
        behavior_group = behavior_group if isinstance(behavior_group, dict) else {}
        for output_name, names in aliases.items():
            result[f"{output_name}_{group_name}"] = _quantile_median(group, names)
        for metric_name in behavior_metric_names:
            result[f"{metric_name}_median_{group_name}"] = _quantile_median(
                behavior_group, (metric_name,)
            )
        result[f"actual_clipped_fraction_{group_name}"] = behavior_group.get(
            "actual_clipped_fraction"
        )
        result[f"would_clip_fraction_{group_name}"] = behavior_group.get(
            "would_clip_fraction"
        )
        for count_name in controller_count_names:
            result[f"{count_name}_{group_name}"] = group.get(count_name)
        for scalar_name in (
            "log_threshold_total_variation",
            "oracle_direction_agreement_fraction",
            "noisy_cdf_out_of_range_fraction",
        ):
            result[f"{scalar_name}_{group_name}"] = group.get(scalar_name)
        result[f"final_threshold_{group_name}"] = group.get("final_next_clip_threshold")
    return result


BASE_METRIC_COLUMNS = (
    "arm_id",
    "family",
    "analysis_role",
    "method",
    "controller_input",
    "seed",
    "num_clients",
    "noise_multiplier",
    "effective_gradient_noise_multiplier",
    "initial_clip_norm",
    "slaclip_num_slots",
    "slaclip_eta",
    "slaclip_base_target_clipped_fraction",
    "slaclip_beta",
    "reference_arm_id",
    "model",
    "initial_loss",
    "final_loss",
    "best_loss",
    "best_round",
    "final_minus_best",
    "normalized_loss_auc",
    "loss_delta",
    "actual_clipped_fraction",
    "would_clip_fraction",
    "elapsed_seconds",
    "sample_schedule_sha256",
    "supervision_schedule_sha256",
    "cdf_endpoint_noise_std_theoretical",
)

GROUP_METRIC_COLUMNS = (
    "cdf_near_threshold_median",
    "cdf_near_zero_median",
    "near_zero_adjusted_median",
    "remaining_non_small_gradient_fraction_median",
    "raw_dynamic_target_clipped_median",
    "dynamic_target_median",
    "dynamic_target_clipped_median",
    "controller_error_median",
    "near_threshold_proxy_error_median",
    "near_zero_proxy_error_median",
    "raw_log_step_median",
    "cdf_error_mae_median",
    "cdf_error_rmse_median",
    "cdf_error_max_abs_median",
    "cdf_error_z_rmse_median",
    "oracle_target_clipped_median",
    "oracle_log_step_median",
    "oracle_next_threshold_median",
    "noisy_minus_oracle_log_step_median",
    "noisy_oracle_log_threshold_error_median",
    "actual_target_absolute_error_median",
    "threshold_used_median",
    "next_threshold_median",
    "final_threshold",
    "actual_clipped_fraction",
    "would_clip_fraction",
    "raw_gradient_l2_median",
    "raw_to_threshold_ratio_median",
    "removed_gradient_l2_median",
    "retained_energy_fraction_median",
    "clipped_signal_gradient_l2_median",
    "noise_gradient_l2_median",
    "signal_to_noise_l2_ratio_median",
    "signal_noise_cosine_median",
    "global_update_l2_median",
    "aggregate_signal_gradient_l2_median",
    "aggregate_noise_gradient_l2_median",
    "aggregate_signal_to_noise_l2_ratio_median",
    "relative_global_update_median",
    "gamma_clamped_low_count",
    "gamma_clamped_high_count",
    "log_step_bounded_count",
    "lower_bound_hits",
    "upper_bound_hits",
    "noisy_adjacent_monotonicity_violations",
    "exact_adjacent_monotonicity_violations",
    "log_step_direction_flip_count",
    "oracle_direction_agreement_count",
    "noisy_cdf_out_of_range_count",
    "log_threshold_total_variation",
    "oracle_direction_agreement_fraction",
    "noisy_cdf_out_of_range_fraction",
)

METRIC_COLUMNS = BASE_METRIC_COLUMNS + tuple(
    f"{metric}_{group}"
    for group in ("A", "B")
    for metric in GROUP_METRIC_COLUMNS
)

AGGREGATED_METRICS = (
    "final_loss",
    "best_loss",
    "best_round",
    "final_minus_best",
    "normalized_loss_auc",
    "loss_delta",
    "actual_clipped_fraction",
    "elapsed_seconds",
    "cdf_endpoint_noise_std_theoretical",
) + tuple(
    f"{metric}_{group}"
    for group in ("A", "B")
    for metric in GROUP_METRIC_COLUMNS
)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            values.append(float(value))
    return values


def paired_inference(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return deterministic paired descriptive and exact sign-flip statistics."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    count = len(finite)
    if count == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "sample_std": None,
            "standard_error": None,
            "ci95_low": None,
            "ci95_high": None,
            "cohens_dz": None,
            "negative_fraction": None,
            "zero_fraction": None,
            "exact_sign_flip_p": None,
        }
    mean = statistics.fmean(finite)
    median = statistics.median(finite)
    sample_std = statistics.stdev(finite) if count > 1 else None
    standard_error = sample_std / math.sqrt(count) if sample_std is not None else None
    t_critical_975 = {
        1: 12.7062047364,
        2: 4.30265272975,
        3: 3.18244630528,
        4: 2.7764451052,
        5: 2.57058183564,
        6: 2.44691184879,
        7: 2.36462425101,
        8: 2.3060041352,
        9: 2.26215716285,
        10: 2.22813885196,
        11: 2.20098516008,
        12: 2.17881282966,
        13: 2.16036865646,
        14: 2.14478668792,
        15: 2.13144954556,
        16: 2.11990529922,
        17: 2.10981557783,
        18: 2.10092204024,
        19: 2.09302405441,
        20: 2.08596344727,
        21: 2.07961384473,
        22: 2.0738730679,
        23: 2.06865761042,
        24: 2.06389856163,
        25: 2.05953855275,
        26: 2.05552943864,
        27: 2.05183051648,
        28: 2.0484071418,
        29: 2.04522964213,
        30: 2.0422724563,
    }
    if standard_error is not None:
        critical = t_critical_975.get(count - 1, 1.95996398454)
        half_width = critical * standard_error
        ci_low = mean - half_width
        ci_high = mean + half_width
    else:
        ci_low = ci_high = None
    observed = abs(mean)
    extreme = 0
    assignments = 1 << count
    for mask in range(assignments):
        permuted = math.fsum(
            (-value if mask & (1 << index) else value)
            for index, value in enumerate(finite)
        ) / count
        if abs(permuted) >= observed - 1e-15:
            extreme += 1
    return {
        "n": count,
        "mean": mean,
        "median": median,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "cohens_dz": (
            mean / sample_std
            if sample_std is not None and sample_std > 0.0
            else None
        ),
        "negative_fraction": sum(value < 0.0 for value in finite) / count,
        "zero_fraction": sum(value == 0.0 for value in finite) / count,
        "exact_sign_flip_p": extreme / assignments,
    }


def write_development_beta_selection(
    runtime: Mapping[str, Any],
    root: Path,
    *,
    metric_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> Path | None:
    """Freeze a transparent development ranking without calling it test evidence."""

    boundary = runtime.get("scientific_boundary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("analysis_role")
        != "development_hyperparameter_screen_not_confirmatory_test"
    ):
        return None
    candidate_values = boundary.get("candidate_base_target_clipped_fractions")
    development_seeds = boundary.get("development_seeds")
    if (
        not isinstance(candidate_values, list)
        or len(candidate_values) != 5
        or not isinstance(development_seeds, list)
        or not development_seeds
    ):
        raise RuntimeError("beta-development boundary is incomplete")
    candidates = [require_number(value, "development beta") for value in candidate_values]
    if len(set(candidates)) != 5 or any(value < 0.0 or value > 1.0 for value in candidates):
        raise RuntimeError("beta-development candidates must be five unique values in [0,1]")
    expected_seeds = {
        require_int(value, "development seed", positive=False)
        for value in development_seeds
    }
    rankings: dict[str, Any] = {}
    for model in EXPECTED_MODELS:
        candidate_records: list[dict[str, Any]] = []
        for beta in candidates:
            paired = [
                row
                for row in paired_rows
                if row.get("model") == model
                and row.get("slaclip_beta") == beta
                and row.get("seed") in expected_seeds
            ]
            if {row.get("seed") for row in paired} != expected_seeds:
                raise RuntimeError(
                    f"beta-development results are incomplete for {model}/beta={beta}"
                )
            final_differences = _finite_values(
                paired,
                "final_loss_difference_slaclip_minus_fixed",
            )
            auc_differences = _finite_values(
                paired,
                "normalized_loss_auc_difference_slaclip_minus_fixed",
            )
            if len(final_differences) != len(expected_seeds) or len(
                auc_differences
            ) != len(expected_seeds):
                raise RuntimeError(
                    f"beta-development loss evidence is non-finite for {model}/beta={beta}"
                )
            adaptive = [
                row
                for row in metric_rows
                if row.get("model") == model
                and row.get("method") == FULL_SLACLIP_METHOD
                and row.get("slaclip_beta") == beta
                and row.get("seed") in expected_seeds
            ]
            if {row.get("seed") for row in adaptive} != expected_seeds:
                raise RuntimeError(
                    f"beta-development controller evidence is incomplete for {model}/beta={beta}"
                )
            instability_fields = tuple(
                f"{name}_{group}"
                for group in ("A", "B")
                for name in (
                    "gamma_clamped_low_count",
                    "gamma_clamped_high_count",
                    "lower_bound_hits",
                    "upper_bound_hits",
                )
            )
            instability_values = [
                row.get(field)
                for row in adaptive
                for field in instability_fields
            ]
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in instability_values
            ):
                raise RuntimeError(
                    f"beta-development controller evidence is invalid for {model}/beta={beta}"
                )
            candidate_records.append(
                {
                    "base_target_clipped_fraction": beta,
                    "seed_count": len(expected_seeds),
                    "mean_paired_final_loss_delta_vs_fixed": statistics.fmean(
                        final_differences
                    ),
                    "mean_paired_normalized_loss_auc_delta_vs_fixed": (
                        statistics.fmean(auc_differences)
                    ),
                    "controller_instability_event_count": sum(
                        int(value) for value in instability_values
                    ),
                }
            )
        ordered = sorted(
            candidate_records,
            key=lambda row: (
                row["mean_paired_final_loss_delta_vs_fixed"],
                row["mean_paired_normalized_loss_auc_delta_vs_fixed"],
                row["controller_instability_event_count"],
                row["base_target_clipped_fraction"],
            ),
        )
        rankings[model] = {
            "selected_base_target_clipped_fraction_for_future_confirmation": ordered[
                0
            ]["base_target_clipped_fraction"],
            "ordered_candidates": ordered,
        }
    output = root / "development_beta_selection.json"
    atomic_json(
        output,
        {
            "schema_version": 1,
            "status": "DEVELOPMENT_SELECTION_ONLY_NOT_TEST_EVIDENCE",
            "campaign_name": runtime["campaign_name"],
            "runtime_manifest_sha256": runtime["manifest_sha256"],
            "models": rankings,
            "selection_rule": {
                "primary": "lowest_mean_paired_final_internal_validation_loss_delta",
                "tiebreakers": [
                    "lowest_mean_paired_normalized_internal_validation_loss_AUC_delta",
                    "fewest_controller_clamp_or_threshold-bound-hit_events",
                    "smaller_base_target_clipped_fraction",
                ],
            },
            "warning": (
                "The selected values must be frozen before an independent "
                "confirmation run; this artifact is not test-set evidence."
            ),
            "created_at_utc": utc_now(),
        },
    )
    return output


def _load_comparison_record(
    path: Path,
    *,
    arm_id: str,
    validation: str,
) -> dict[str, Any]:
    value = load_object(path, f"full-SlaClip comparison for {arm_id}")
    fingerprint = value.get("comparison_fingerprint")
    if (
        value.get("status") != FULL_COMPARISON_STATUS
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
    ):
        raise RuntimeError(f"comparison identity is invalid: {arm_id}")
    return {
        "arm_id": arm_id,
        "path": str(path),
        "sha256": sha256_file(path),
        "comparison_fingerprint": fingerprint,
        "validation": validation,
    }


def _run_full_comparator(
    *,
    baseline_dir: Path,
    adaptive_dir: Path,
    output: Path,
    verify_existing: bool,
) -> None:
    comparator = Path(__file__).resolve().with_name("compare_slaclip.py")
    command = [
        sys.executable,
        str(comparator),
        "--baseline-dir",
        str(baseline_dir),
        "--adaptive-dir",
        str(adaptive_dir),
        "--output",
        str(output),
    ]
    if verify_existing:
        command.append("--verify-existing")
    completed = subprocess.run(
        command,
        cwd=comparator.parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "full-SlaClip comparison failed for "
            f"{adaptive_dir.name}: exit={completed.returncode}; {detail[-2000:]}"
        )


def ensure_full_comparisons(
    runtime: Mapping[str, Any],
    campaign_root: Path,
    *,
    completed_arm_ids: set[str],
    require_complete: bool,
) -> list[dict[str, Any]]:
    """Create missing comparisons incrementally and reverify all at completion."""

    adaptive_arms = [
        arm for arm in runtime["arms"] if arm["method"] == FULL_SLACLIP_METHOD
    ]
    if not adaptive_arms:
        raise RuntimeError("runtime manifest contains no full-SlaClip arms")
    expected_adaptive_ids = {arm["arm_id"] for arm in adaptive_arms}
    if require_complete and not expected_adaptive_ids.issubset(completed_arm_ids):
        missing = sorted(expected_adaptive_ids - completed_arm_ids)
        raise RuntimeError(
            f"final comparison gate is missing completed adaptive arms: {missing[:3]}"
        )
    comparison_root = campaign_root / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(comparison_root, 0o700)
    records: list[dict[str, Any]] = []
    for arm in adaptive_arms:
        arm_id = str(arm["arm_id"])
        if arm_id not in completed_arm_ids:
            continue
        reference_id = arm.get("reference_arm_id")
        if not isinstance(reference_id, str) or reference_id not in completed_arm_ids:
            if require_complete:
                raise RuntimeError(
                    f"final comparison gate is missing reference arm for {arm_id}"
                )
            continue
        output = comparison_root / f"{arm_id}.json"
        baseline_dir = campaign_root / "arms" / reference_id
        adaptive_dir = campaign_root / "arms" / arm_id
        if require_complete:
            if not output.is_file():
                _run_full_comparator(
                    baseline_dir=baseline_dir,
                    adaptive_dir=adaptive_dir,
                    output=output,
                    verify_existing=False,
                )
            _run_full_comparator(
                baseline_dir=baseline_dir,
                adaptive_dir=adaptive_dir,
                output=output,
                verify_existing=True,
            )
            validation = "REVERIFIED_THIS_PASS"
        elif output.is_file():
            validation = "PRESENT_PENDING_FINAL_REVERIFICATION"
        else:
            _run_full_comparator(
                baseline_dir=baseline_dir,
                adaptive_dir=adaptive_dir,
                output=output,
                verify_existing=False,
            )
            validation = "CREATED_AND_VERIFIED_THIS_PASS"
        records.append(
            _load_comparison_record(
                output,
                arm_id=arm_id,
                validation=validation,
            )
        )
    if require_complete and len(records) != len(adaptive_arms):
        raise RuntimeError(
            "final comparison evidence count is "
            f"{len(records)}, expected {len(adaptive_arms)}"
        )
    return records


def aggregate_campaign(args: argparse.Namespace) -> bool:
    manifest_path = args.manifest.resolve()
    runtime = load_runtime(manifest_path)
    oracle_noisy_arm_pairs = resolve_oracle_noisy_arm_pairs(runtime)
    root = manifest_path.parent
    rows: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    completed_arm_ids: list[str] = []
    for arm in runtime["arms"]:
        arm_id = arm["arm_id"]
        status_path = root / "arm-status" / f"{arm_id}.json"
        status = "NOT_STARTED"
        if status_path.is_file():
            status_value = load_object(status_path, "arm status")
            status = str(status_value.get("status", "INVALID"))
        statuses[arm_id] = status
        final_path = root / "arms" / arm_id / "final_summary.json"
        if not final_path.is_file():
            continue
        summary = load_object(final_path, "arm final summary")
        if summary.get("status") != "COMPLETED" or summary.get("method") != arm["method"]:
            continue
        models = summary.get("models")
        if not isinstance(models, dict) or tuple(models) != EXPECTED_MODELS:
            raise RuntimeError(f"completed arm has a mismatched model set/order: {arm_id}")
        completed_arm_ids.append(arm_id)
        for model in EXPECTED_MODELS:
            model_summary = models.get(model)
            if not isinstance(model_summary, dict):
                raise RuntimeError(f"model summary is missing: {arm_id}/{model}")
            rows.append(_model_metrics(arm, model, model_summary))

    if args.require_complete:
        for row in rows:
            required_metrics = [
                "initial_loss",
                "final_loss",
                "best_loss",
                "normalized_loss_auc",
                "elapsed_seconds",
                "actual_clipped_fraction",
                "raw_gradient_l2_median_A",
                "raw_gradient_l2_median_B",
            ]
            if row["method"] in {FIXED_DP_METHOD, *ADAPTIVE_METHODS}:
                required_metrics.extend(
                    (
                        "aggregate_signal_to_noise_l2_ratio_median_A",
                        "aggregate_signal_to_noise_l2_ratio_median_B",
                    )
                )
            if row["method"] in ADAPTIVE_METHODS:
                required_metrics.extend(
                    f"{name}_{group}"
                    for group in ("A", "B")
                    for name in (
                        "cdf_error_mae_median",
                        "cdf_error_rmse_median",
                        "oracle_next_threshold_median",
                        "actual_target_absolute_error_median",
                        "actual_clipped_fraction",
                        "retained_energy_fraction_median",
                        "final_threshold",
                        "log_threshold_total_variation",
                        "oracle_direction_agreement_fraction",
                    )
                )
            invalid = [
                name
                for name in required_metrics
                if not isinstance(row.get(name), (int, float))
                or isinstance(row.get(name), bool)
                or not math.isfinite(float(row[name]))
            ]
            if invalid:
                raise RuntimeError(
                    f"completed metric row has missing/non-finite evidence: "
                    f"{row['arm_id']}/{row['model']} {invalid[:4]}"
                )

    comparison_records = ensure_full_comparisons(
        runtime,
        root,
        completed_arm_ids=set(completed_arm_ids),
        require_complete=bool(args.require_complete),
    )

    paired_rows: list[dict[str, Any]] = []
    rows_by_arm_model = {(row["arm_id"], row["model"]): row for row in rows}
    for row in rows:
        reference_id = row["reference_arm_id"]
        if row["method"] != FULL_SLACLIP_METHOD or not isinstance(reference_id, str):
            continue
        reference = rows_by_arm_model.get((reference_id, row["model"]))
        if reference is None:
            continue
        paired_rows.append(
            {
                "arm_id": row["arm_id"],
                "reference_arm_id": reference_id,
                "family": row["family"],
                "method": row["method"],
                "controller_input": row["controller_input"],
                "seed": row["seed"],
                "num_clients": row["num_clients"],
                "noise_multiplier": row["noise_multiplier"],
                "effective_gradient_noise_multiplier": row[
                    "effective_gradient_noise_multiplier"
                ],
                "initial_clip_norm": row["initial_clip_norm"],
                "slaclip_num_slots": row["slaclip_num_slots"],
                "slaclip_eta": row["slaclip_eta"],
                "slaclip_beta": row["slaclip_beta"],
                "model": row["model"],
                "slaclip_final_loss": row["final_loss"],
                "fixed_final_loss": reference["final_loss"],
                "final_loss_difference_slaclip_minus_fixed": (
                    row["final_loss"] - reference["final_loss"]
                    if row["final_loss"] is not None and reference["final_loss"] is not None
                    else None
                ),
                "perplexity_ratio_slaclip_over_fixed": (
                    math.exp(
                        max(
                            -50.0,
                            min(
                                50.0,
                                row["final_loss"] - reference["final_loss"],
                            ),
                        )
                    )
                    if row["final_loss"] is not None
                    and reference["final_loss"] is not None
                    else None
                ),
                "slaclip_best_loss": row["best_loss"],
                "fixed_best_loss": reference["best_loss"],
                "best_loss_difference_slaclip_minus_fixed": (
                    row["best_loss"] - reference["best_loss"]
                    if row["best_loss"] is not None
                    and reference["best_loss"] is not None
                    else None
                ),
                "slaclip_normalized_loss_auc": row["normalized_loss_auc"],
                "fixed_normalized_loss_auc": reference["normalized_loss_auc"],
                "normalized_loss_auc_difference_slaclip_minus_fixed": (
                    row["normalized_loss_auc"] - reference["normalized_loss_auc"]
                    if row["normalized_loss_auc"] is not None
                    and reference["normalized_loss_auc"] is not None
                    else None
                ),
                "slaclip_final_minus_best": row["final_minus_best"],
                "fixed_final_minus_best": reference["final_minus_best"],
                "final_minus_best_difference": (
                    row["final_minus_best"] - reference["final_minus_best"]
                    if row["final_minus_best"] is not None
                    and reference["final_minus_best"] is not None
                    else None
                ),
                "slaclip_elapsed_seconds": row["elapsed_seconds"],
                "fixed_elapsed_seconds": reference["elapsed_seconds"],
                "elapsed_seconds_difference": (
                    row["elapsed_seconds"] - reference["elapsed_seconds"]
                    if isinstance(row["elapsed_seconds"], (int, float))
                    and isinstance(reference["elapsed_seconds"], (int, float))
                    else None
                ),
                "slaclip_actual_clipped_fraction": row["actual_clipped_fraction"],
                "fixed_actual_clipped_fraction": reference["actual_clipped_fraction"],
                "actual_clipped_fraction_difference": (
                    row["actual_clipped_fraction"] - reference["actual_clipped_fraction"]
                    if isinstance(row["actual_clipped_fraction"], (int, float))
                    and isinstance(reference["actual_clipped_fraction"], (int, float))
                    else None
                ),
            }
        )

    diagnostic_paired_rows: list[dict[str, Any]] = []
    diagnostic_methods = {ORACLE_SLACLIP_METHOD, *CONTROL_METHODS}
    for row in rows:
        reference_id = row["reference_arm_id"]
        if row["method"] not in diagnostic_methods or not isinstance(
            reference_id, str
        ):
            continue
        reference = rows_by_arm_model.get((reference_id, row["model"]))
        if reference is None:
            continue

        def difference(metric: str) -> float | None:
            candidate = row.get(metric)
            fixed = reference.get(metric)
            if (
                isinstance(candidate, (int, float))
                and not isinstance(candidate, bool)
                and isinstance(fixed, (int, float))
                and not isinstance(fixed, bool)
            ):
                return float(candidate) - float(fixed)
            return None

        diagnostic_paired_rows.append(
            {
                "arm_id": row["arm_id"],
                "reference_arm_id": reference_id,
                "family": row["family"],
                "analysis_role": row["analysis_role"],
                "comparison_role": (
                    "NON_DP_EXACT_ENDPOINT_ORACLE_DIAGNOSTIC"
                    if row["method"] == ORACLE_SLACLIP_METHOD
                    else "NON_DP_MECHANISM_CONTROL_EXPLORATORY"
                ),
                "method": row["method"],
                "controller_input": row["controller_input"],
                "seed": row["seed"],
                "num_clients": row["num_clients"],
                "noise_multiplier": row["noise_multiplier"],
                "effective_gradient_noise_multiplier": row[
                    "effective_gradient_noise_multiplier"
                ],
                "initial_clip_norm": row["initial_clip_norm"],
                "slaclip_num_slots": row["slaclip_num_slots"],
                "slaclip_eta": row["slaclip_eta"],
                "slaclip_beta": row["slaclip_beta"],
                "model": row["model"],
                "candidate_final_loss": row["final_loss"],
                "fixed_final_loss": reference["final_loss"],
                "final_loss_difference_candidate_minus_fixed": difference(
                    "final_loss"
                ),
                "candidate_best_loss": row["best_loss"],
                "fixed_best_loss": reference["best_loss"],
                "best_loss_difference_candidate_minus_fixed": difference(
                    "best_loss"
                ),
                "candidate_normalized_loss_auc": row["normalized_loss_auc"],
                "fixed_normalized_loss_auc": reference["normalized_loss_auc"],
                "normalized_loss_auc_difference_candidate_minus_fixed": difference(
                    "normalized_loss_auc"
                ),
                "candidate_final_minus_best": row["final_minus_best"],
                "fixed_final_minus_best": reference["final_minus_best"],
                "final_minus_best_difference_candidate_minus_fixed": difference(
                    "final_minus_best"
                ),
                "candidate_actual_clipped_fraction": row[
                    "actual_clipped_fraction"
                ],
                "fixed_actual_clipped_fraction": reference[
                    "actual_clipped_fraction"
                ],
                "actual_clipped_fraction_difference_candidate_minus_fixed": difference(
                    "actual_clipped_fraction"
                ),
                "candidate_elapsed_seconds": row["elapsed_seconds"],
                "fixed_elapsed_seconds": reference["elapsed_seconds"],
                "elapsed_seconds_difference_candidate_minus_fixed": difference(
                    "elapsed_seconds"
                ),
            }
        )

    oracle_vs_noisy_paired_rows: list[dict[str, Any]] = []
    oracle_vs_noisy_overall_metrics = (
        "final_loss",
        "best_loss",
        "normalized_loss_auc",
        "final_minus_best",
        "actual_clipped_fraction",
        "elapsed_seconds",
    )
    oracle_vs_noisy_group_metrics = (
        "final_threshold",
        "log_threshold_total_variation",
        "actual_clipped_fraction",
        "retained_energy_fraction_median",
        "aggregate_signal_to_noise_l2_ratio_median",
    )

    def oracle_minus_noisy(
        candidate: Mapping[str, Any],
        noisy: Mapping[str, Any],
        metric: str,
    ) -> float | None:
        candidate_value = candidate.get(metric)
        noisy_value = noisy.get(metric)
        if (
            isinstance(candidate_value, (int, float))
            and not isinstance(candidate_value, bool)
            and math.isfinite(float(candidate_value))
            and isinstance(noisy_value, (int, float))
            and not isinstance(noisy_value, bool)
            and math.isfinite(float(noisy_value))
        ):
            return float(candidate_value) - float(noisy_value)
        return None

    for oracle_arm, noisy_arm in oracle_noisy_arm_pairs:
        for model in EXPECTED_MODELS:
            candidate = rows_by_arm_model.get((oracle_arm["arm_id"], model))
            noisy = rows_by_arm_model.get((noisy_arm["arm_id"], model))
            if candidate is None or noisy is None:
                continue
            for digest_name in (
                "sample_schedule_sha256",
                "supervision_schedule_sha256",
            ):
                candidate_digest = candidate.get(digest_name)
                noisy_digest = noisy.get(digest_name)
                if (
                    not isinstance(candidate_digest, str)
                    or len(candidate_digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in candidate_digest
                    )
                    or candidate_digest != noisy_digest
                ):
                    raise RuntimeError(
                        "oracle/noisy completed pair has mismatched schedule evidence: "
                        f"{oracle_arm['arm_id']}/{noisy_arm['arm_id']}/{model}/"
                        f"{digest_name}"
                    )
            output: dict[str, Any] = {
                "oracle_arm_id": oracle_arm["arm_id"],
                "noisy_arm_id": noisy_arm["arm_id"],
                "oracle_family": oracle_arm["family"],
                "noisy_family": noisy_arm["family"],
                "comparison_role": (
                    "NON_DP_ORACLE_VS_NOISY_CDF_CONTROLLER_DIAGNOSTIC"
                ),
                "candidate_method": candidate["method"],
                "noisy_method": noisy["method"],
                "candidate_controller_input": candidate["controller_input"],
                "noisy_controller_input": noisy["controller_input"],
                "seed": candidate["seed"],
                "num_clients": candidate["num_clients"],
                "noise_multiplier": candidate["noise_multiplier"],
                "effective_gradient_noise_multiplier": candidate[
                    "effective_gradient_noise_multiplier"
                ],
                "initial_clip_norm": candidate["initial_clip_norm"],
                "slaclip_num_slots": candidate["slaclip_num_slots"],
                "slaclip_eta": candidate["slaclip_eta"],
                "slaclip_beta": candidate["slaclip_beta"],
                "model": model,
                "sample_schedule_sha256": candidate["sample_schedule_sha256"],
                "supervision_schedule_sha256": candidate[
                    "supervision_schedule_sha256"
                ],
            }
            for metric in oracle_vs_noisy_overall_metrics:
                output[f"candidate_{metric}"] = candidate.get(metric)
                output[f"noisy_{metric}"] = noisy.get(metric)
                output[f"{metric}_difference_candidate_minus_noisy"] = (
                    oracle_minus_noisy(candidate, noisy, metric)
                )
            for group_name in ("A", "B"):
                for metric in oracle_vs_noisy_group_metrics:
                    grouped_metric = f"{metric}_{group_name}"
                    output[f"candidate_{grouped_metric}"] = candidate.get(
                        grouped_metric
                    )
                    output[f"noisy_{grouped_metric}"] = noisy.get(grouped_metric)
                    output[
                        f"{grouped_metric}_difference_candidate_minus_noisy"
                    ] = oracle_minus_noisy(candidate, noisy, grouped_metric)
            oracle_vs_noisy_paired_rows.append(output)

    group_keys = (
        "family",
        "method",
        "controller_input",
        "num_clients",
        "noise_multiplier",
        "effective_gradient_noise_multiplier",
        "initial_clip_norm",
        "slaclip_num_slots",
        "slaclip_eta",
        "slaclip_beta",
        "model",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    )
    for key, group_rows in ordered_groups:
        output = dict(zip(group_keys, key))
        output["seed_count"] = len({row["seed"] for row in group_rows})
        for metric in AGGREGATED_METRICS:
            values = _finite_values(group_rows, metric)
            output[f"{metric}_n"] = len(values)
            output[f"{metric}_mean"] = statistics.fmean(values) if values else None
            output[f"{metric}_sample_std"] = statistics.stdev(values) if len(values) > 1 else None
        aggregate_rows.append(output)

    paired_group_keys = (
        "family",
        "method",
        "controller_input",
        "num_clients",
        "noise_multiplier",
        "effective_gradient_noise_multiplier",
        "initial_clip_norm",
        "slaclip_num_slots",
        "slaclip_eta",
        "slaclip_beta",
        "model",
    )
    paired_grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in paired_rows:
        paired_grouped.setdefault(
            tuple(row[key] for key in paired_group_keys), []
        ).append(row)
    paired_aggregate_rows: list[dict[str, Any]] = []
    paired_ordered_groups = sorted(
        paired_grouped.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    )
    paired_difference_metrics = (
        "final_loss_difference_slaclip_minus_fixed",
        "best_loss_difference_slaclip_minus_fixed",
        "normalized_loss_auc_difference_slaclip_minus_fixed",
        "final_minus_best_difference",
        "actual_clipped_fraction_difference",
        "elapsed_seconds_difference",
    )
    for key, group_rows in paired_ordered_groups:
        output = dict(zip(paired_group_keys, key))
        output["seed_count"] = len({row["seed"] for row in group_rows})
        for metric in paired_difference_metrics:
            values = _finite_values(group_rows, metric)
            for statistic_name, statistic_value in paired_inference(values).items():
                output[f"{metric}_{statistic_name}"] = statistic_value
        paired_aggregate_rows.append(output)

    p_value_key = (
        "final_loss_difference_slaclip_minus_fixed_exact_sign_flip_p"
    )
    ordered_p_values = sorted(
        (
            (index, float(row[p_value_key]))
            for index, row in enumerate(paired_aggregate_rows)
            if isinstance(row.get(p_value_key), (int, float))
        ),
        key=lambda item: item[1],
    )
    running_adjusted = 0.0
    total_tests = len(ordered_p_values)
    for rank, (index, p_value) in enumerate(ordered_p_values, start=1):
        running_adjusted = max(
            running_adjusted,
            min(1.0, (total_tests - rank + 1) * p_value),
        )
        paired_aggregate_rows[index][
            "final_loss_difference_slaclip_minus_fixed_holm_p"
        ] = running_adjusted

    diagnostic_group_keys = (
        "family",
        "analysis_role",
        "comparison_role",
        "method",
        "controller_input",
        "num_clients",
        "noise_multiplier",
        "effective_gradient_noise_multiplier",
        "initial_clip_norm",
        "slaclip_num_slots",
        "slaclip_eta",
        "slaclip_beta",
        "model",
    )
    diagnostic_difference_metrics = (
        "final_loss_difference_candidate_minus_fixed",
        "best_loss_difference_candidate_minus_fixed",
        "normalized_loss_auc_difference_candidate_minus_fixed",
        "final_minus_best_difference_candidate_minus_fixed",
        "actual_clipped_fraction_difference_candidate_minus_fixed",
        "elapsed_seconds_difference_candidate_minus_fixed",
    )
    diagnostic_grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in diagnostic_paired_rows:
        diagnostic_grouped.setdefault(
            tuple(row[key] for key in diagnostic_group_keys), []
        ).append(row)
    diagnostic_paired_aggregate_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(
        diagnostic_grouped.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        output = dict(zip(diagnostic_group_keys, key))
        output["seed_count"] = len({row["seed"] for row in group_rows})
        for metric in diagnostic_difference_metrics:
            values = _finite_values(group_rows, metric)
            output[f"{metric}_n"] = len(values)
            output[f"{metric}_mean"] = (
                statistics.fmean(values) if values else None
            )
            output[f"{metric}_median"] = (
                statistics.median(values) if values else None
            )
            output[f"{metric}_sample_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
            output[f"{metric}_negative_fraction"] = (
                sum(value < 0.0 for value in values) / len(values)
                if values
                else None
            )
        diagnostic_paired_aggregate_rows.append(output)

    oracle_vs_noisy_group_keys = (
        "num_clients",
        "noise_multiplier",
        "effective_gradient_noise_multiplier",
        "initial_clip_norm",
        "slaclip_num_slots",
        "slaclip_eta",
        "slaclip_beta",
        "model",
    )
    oracle_vs_noisy_difference_metrics = tuple(
        f"{metric}_difference_candidate_minus_noisy"
        for metric in oracle_vs_noisy_overall_metrics
    ) + tuple(
        f"{metric}_{group_name}_difference_candidate_minus_noisy"
        for group_name in ("A", "B")
        for metric in oracle_vs_noisy_group_metrics
    )
    oracle_vs_noisy_grouped: dict[
        tuple[Any, ...], list[Mapping[str, Any]]
    ] = {}
    for row in oracle_vs_noisy_paired_rows:
        oracle_vs_noisy_grouped.setdefault(
            tuple(row[key] for key in oracle_vs_noisy_group_keys), []
        ).append(row)
    oracle_vs_noisy_paired_aggregate_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(
        oracle_vs_noisy_grouped.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        output = dict(zip(oracle_vs_noisy_group_keys, key))
        output["seed_count"] = len({row["seed"] for row in group_rows})
        for metric in oracle_vs_noisy_difference_metrics:
            values = _finite_values(group_rows, metric)
            output[f"{metric}_n"] = len(values)
            output[f"{metric}_mean"] = (
                statistics.fmean(values) if values else None
            )
            output[f"{metric}_median"] = (
                statistics.median(values) if values else None
            )
            output[f"{metric}_sample_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
            output[f"{metric}_negative_fraction"] = (
                sum(value < 0.0 for value in values) / len(values)
                if values
                else None
            )
        oracle_vs_noisy_paired_aggregate_rows.append(output)

    expected_oracle_vs_noisy_paired_metric_row_count = (
        len(oracle_noisy_arm_pairs) * len(EXPECTED_MODELS)
    )
    expected_oracle_vs_noisy_aggregate_keys = {
        (
            oracle_arm["num_clients"],
            oracle_arm["noise_multiplier"],
            oracle_arm["noise_multiplier"],
            oracle_arm["initial_clip_norm"],
            oracle_arm["slaclip_num_slots"],
            oracle_arm["slaclip_eta"],
            oracle_arm["slaclip_beta"],
            model,
        )
        for oracle_arm, _noisy_arm in oracle_noisy_arm_pairs
        for model in EXPECTED_MODELS
    }
    expected_oracle_vs_noisy_paired_aggregate_row_count = len(
        expected_oracle_vs_noisy_aggregate_keys
    )
    if args.require_complete:
        if (
            len(oracle_vs_noisy_paired_rows)
            != expected_oracle_vs_noisy_paired_metric_row_count
        ):
            raise RuntimeError(
                "complete campaign oracle/noisy paired row count is "
                f"{len(oracle_vs_noisy_paired_rows)}, expected "
                f"{expected_oracle_vs_noisy_paired_metric_row_count}"
            )
        actual_oracle_vs_noisy_aggregate_keys = {
            tuple(row[key] for key in oracle_vs_noisy_group_keys)
            for row in oracle_vs_noisy_paired_aggregate_rows
        }
        if (
            actual_oracle_vs_noisy_aggregate_keys
            != expected_oracle_vs_noisy_aggregate_keys
        ):
            raise RuntimeError(
                "complete campaign oracle/noisy aggregate groups differ from "
                "the unique manifest configurations"
            )

    paired_columns = (
        "arm_id",
        "reference_arm_id",
        "family",
        "method",
        "controller_input",
        "seed",
        "num_clients",
        "noise_multiplier",
        "effective_gradient_noise_multiplier",
        "initial_clip_norm",
        "slaclip_num_slots",
        "slaclip_eta",
        "slaclip_beta",
        "model",
        "slaclip_final_loss",
        "fixed_final_loss",
        "final_loss_difference_slaclip_minus_fixed",
        "perplexity_ratio_slaclip_over_fixed",
        "slaclip_best_loss",
        "fixed_best_loss",
        "best_loss_difference_slaclip_minus_fixed",
        "slaclip_normalized_loss_auc",
        "fixed_normalized_loss_auc",
        "normalized_loss_auc_difference_slaclip_minus_fixed",
        "slaclip_final_minus_best",
        "fixed_final_minus_best",
        "final_minus_best_difference",
        "slaclip_elapsed_seconds",
        "fixed_elapsed_seconds",
        "elapsed_seconds_difference",
        "slaclip_actual_clipped_fraction",
        "fixed_actual_clipped_fraction",
        "actual_clipped_fraction_difference",
    )
    diagnostic_paired_columns = (
        "arm_id",
        "reference_arm_id",
        "family",
        "analysis_role",
        "comparison_role",
        "method",
        "controller_input",
        "seed",
        "num_clients",
        "noise_multiplier",
        "effective_gradient_noise_multiplier",
        "initial_clip_norm",
        "slaclip_num_slots",
        "slaclip_eta",
        "slaclip_beta",
        "model",
        "candidate_final_loss",
        "fixed_final_loss",
        "final_loss_difference_candidate_minus_fixed",
        "candidate_best_loss",
        "fixed_best_loss",
        "best_loss_difference_candidate_minus_fixed",
        "candidate_normalized_loss_auc",
        "fixed_normalized_loss_auc",
        "normalized_loss_auc_difference_candidate_minus_fixed",
        "candidate_final_minus_best",
        "fixed_final_minus_best",
        "final_minus_best_difference_candidate_minus_fixed",
        "candidate_actual_clipped_fraction",
        "fixed_actual_clipped_fraction",
        "actual_clipped_fraction_difference_candidate_minus_fixed",
        "candidate_elapsed_seconds",
        "fixed_elapsed_seconds",
        "elapsed_seconds_difference_candidate_minus_fixed",
    )
    oracle_vs_noisy_paired_columns: list[str] = [
        "oracle_arm_id",
        "noisy_arm_id",
        "oracle_family",
        "noisy_family",
        "comparison_role",
        "candidate_method",
        "noisy_method",
        "candidate_controller_input",
        "noisy_controller_input",
        "seed",
        "num_clients",
        "noise_multiplier",
        "effective_gradient_noise_multiplier",
        "initial_clip_norm",
        "slaclip_num_slots",
        "slaclip_eta",
        "slaclip_beta",
        "model",
        "sample_schedule_sha256",
        "supervision_schedule_sha256",
    ]
    for metric in oracle_vs_noisy_overall_metrics:
        oracle_vs_noisy_paired_columns.extend(
            (
                f"candidate_{metric}",
                f"noisy_{metric}",
                f"{metric}_difference_candidate_minus_noisy",
            )
        )
    for group_name in ("A", "B"):
        for metric in oracle_vs_noisy_group_metrics:
            grouped_metric = f"{metric}_{group_name}"
            oracle_vs_noisy_paired_columns.extend(
                (
                    f"candidate_{grouped_metric}",
                    f"noisy_{grouped_metric}",
                    f"{grouped_metric}_difference_candidate_minus_noisy",
                )
            )
    aggregate_columns = list(group_keys) + ["seed_count"]
    for metric in AGGREGATED_METRICS:
        aggregate_columns.extend((f"{metric}_n", f"{metric}_mean", f"{metric}_sample_std"))
    paired_aggregate_columns = list(paired_group_keys) + ["seed_count"]
    for metric in paired_difference_metrics:
        paired_aggregate_columns.extend(
            f"{metric}_{name}"
            for name in (
                "n",
                "mean",
                "median",
                "sample_std",
                "standard_error",
                "ci95_low",
                "ci95_high",
                "cohens_dz",
                "negative_fraction",
                "zero_fraction",
                "exact_sign_flip_p",
            )
        )
    paired_aggregate_columns.append(
        "final_loss_difference_slaclip_minus_fixed_holm_p"
    )
    diagnostic_paired_aggregate_columns = list(diagnostic_group_keys) + [
        "seed_count"
    ]
    for metric in diagnostic_difference_metrics:
        diagnostic_paired_aggregate_columns.extend(
            f"{metric}_{name}"
            for name in (
                "n",
                "mean",
                "median",
                "sample_std",
                "negative_fraction",
            )
        )
    oracle_vs_noisy_paired_aggregate_columns = list(
        oracle_vs_noisy_group_keys
    ) + ["seed_count"]
    for metric in oracle_vs_noisy_difference_metrics:
        oracle_vs_noisy_paired_aggregate_columns.extend(
            f"{metric}_{name}"
            for name in (
                "n",
                "mean",
                "median",
                "sample_std",
                "negative_fraction",
            )
        )
    atomic_csv(root / "campaign_metrics.csv", rows, METRIC_COLUMNS)
    atomic_csv(root / "paired_metrics.csv", paired_rows, paired_columns)
    atomic_csv(root / "aggregate_metrics.csv", aggregate_rows, aggregate_columns)
    atomic_csv(
        root / "paired_aggregate_metrics.csv",
        paired_aggregate_rows,
        paired_aggregate_columns,
    )
    atomic_csv(
        root / "diagnostic_paired_metrics.csv",
        diagnostic_paired_rows,
        diagnostic_paired_columns,
    )
    atomic_csv(
        root / "diagnostic_paired_aggregate_metrics.csv",
        diagnostic_paired_aggregate_rows,
        diagnostic_paired_aggregate_columns,
    )
    atomic_csv(
        root / "oracle_vs_noisy_paired_metrics.csv",
        oracle_vs_noisy_paired_rows,
        oracle_vs_noisy_paired_columns,
    )
    atomic_csv(
        root / "oracle_vs_noisy_paired_aggregate_metrics.csv",
        oracle_vs_noisy_paired_aggregate_rows,
        oracle_vs_noisy_paired_aggregate_columns,
    )
    expected_comparisons = sum(
        arm["method"] == FULL_SLACLIP_METHOD for arm in runtime["arms"]
    )
    completed = (
        len(completed_arm_ids) == runtime["expected_arm_count"]
        and len(comparison_records) == expected_comparisons
    )
    beta_selection_path = (
        write_development_beta_selection(
            runtime,
            root,
            metric_rows=rows,
            paired_rows=paired_rows,
        )
        if completed
        else None
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED" if completed else "IN_PROGRESS",
        "campaign_name": runtime["campaign_name"],
        "runtime_manifest_sha256": runtime["manifest_sha256"],
        "repository_sha": runtime["repository_sha"],
        "expected_arm_count": runtime["expected_arm_count"],
        "completed_arm_count": len(completed_arm_ids),
        "completed_arm_ids": completed_arm_ids,
        "arm_statuses": statuses,
        "metric_row_count": len(rows),
        "paired_metric_row_count": len(paired_rows),
        "diagnostic_paired_metric_row_count": len(diagnostic_paired_rows),
        "aggregate_row_count": len(aggregate_rows),
        "paired_aggregate_row_count": len(paired_aggregate_rows),
        "diagnostic_paired_aggregate_row_count": len(
            diagnostic_paired_aggregate_rows
        ),
        "expected_oracle_vs_noisy_paired_metric_row_count": (
            expected_oracle_vs_noisy_paired_metric_row_count
        ),
        "oracle_vs_noisy_paired_metric_row_count": len(
            oracle_vs_noisy_paired_rows
        ),
        "oracle_vs_noisy_paired_aggregate_row_count": len(
            oracle_vs_noisy_paired_aggregate_rows
        ),
        "expected_oracle_vs_noisy_paired_aggregate_row_count": (
            expected_oracle_vs_noisy_paired_aggregate_row_count
        ),
        "diagnostic_pairing_policy": (
            "Oracle and non-DP mechanism controls are descriptive diagnostics "
            "against matched fixed-C arms; they are excluded from SlaClip "
            "efficacy selection and Holm correction."
        ),
        "oracle_vs_noisy_pairing_policy": (
            "Exact-endpoint oracle SlaClip candidates are paired one-to-one "
            "with noisy-endpoint full SlaClip only when all training, DP, "
            "controller, RNG-domain, K, seed, and model settings match and "
            "completed sample/supervision schedules have identical hashes. "
            "These non-private oracle comparisons are descriptive mechanism "
            "diagnostics only and are excluded from efficacy claims, Holm "
            "correction, and beta selection."
        ),
        "expected_comparison_count": expected_comparisons,
        "comparison_evidence_count": len(comparison_records),
        "comparisons_verified_this_pass": sum(
            record["validation"]
            in {"CREATED_AND_VERIFIED_THIS_PASS", "REVERIFIED_THIS_PASS"}
            for record in comparison_records
        ),
        "all_comparisons_reverified_this_pass": bool(
            args.require_complete
            and len(comparison_records) == expected_comparisons
            and all(
                record["validation"] == "REVERIFIED_THIS_PASS"
                for record in comparison_records
            )
        ),
        "development_beta_selection": (
            {
                "path": str(beta_selection_path),
                "sha256": sha256_file(beta_selection_path),
                "status": "DEVELOPMENT_SELECTION_ONLY_NOT_TEST_EVIDENCE",
            }
            if beta_selection_path is not None
            else None
        ),
        "comparisons": comparison_records,
        "scientific_boundary": runtime["scientific_boundary"],
        "interpretation": (
            "Internal MedDialog holdout loss and mechanism diagnostics only; "
            "not a published DP-LoRA benchmark reproduction or privacy certificate."
        ),
        "updated_at_utc": utc_now(),
    }
    atomic_json(root / "campaign_summary.json", summary)
    if args.require_complete and not completed:
        raise RuntimeError(
            f"campaign is incomplete: {len(completed_arm_ids)}/{runtime['expected_arm_count']} arms"
        )
    return completed


def _archive_candidate(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > SMALL_ARCHIVE_MAX_BYTES:
        return False
    if any(part in {"final_adapter", "checkpoints", "private"} for part in relative.parts):
        return False
    if relative.name in {
        RUNTIME_MANIFEST_NAME,
        JOB_STATUS_NAME,
        "campaign_summary.json",
        "campaign_metrics.csv",
        "paired_metrics.csv",
        "aggregate_metrics.csv",
        "paired_aggregate_metrics.csv",
        "diagnostic_paired_metrics.csv",
        "diagnostic_paired_aggregate_metrics.csv",
        "oracle_vs_noisy_paired_metrics.csv",
        "oracle_vs_noisy_paired_aggregate_metrics.csv",
        "development_beta_selection.json",
    }:
        return True
    if relative.parts[0] == "arm-status" and relative.suffix == ".json":
        return True
    if relative.parts[0] == "preflight" and relative.suffix in {".json", ".jsonl", ".out", ".err"}:
        return True
    if relative.parts[0] == "comparisons" and relative.suffix == ".json":
        return True
    if relative.parts[0] == "arm-logs" and relative.suffix in {".out", ".err"}:
        return True
    if relative.parts[0] != "arms":
        return False
    if relative.suffix == ".jsonl":
        return True
    if relative.suffix != ".json":
        return False
    return relative.name in {
        "run_config.json",
        "final_summary.json",
        "progress.json",
        "attempt.json",
    } or "private_diagnostics" in relative.parts


def archive_small(args: argparse.Namespace) -> None:
    source = args.campaign_root.resolve()
    destination = args.archive_root.resolve()
    if destination == source or source in destination.parents:
        raise RuntimeError("incremental archive must be outside the scratch campaign")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    copied: list[dict[str, Any]] = []
    for source_path in sorted(source.rglob("*")):
        if not _archive_candidate(source_path, source):
            continue
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination_path.exists() and sha256_file(destination_path) == sha256_file(source_path):
            pass
        else:
            temporary = destination_path.with_name(f".{destination_path.name}.tmp.{os.getpid()}")
            shutil.copyfile(source_path, temporary)
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination_path)
        copied.append(
            {
                "path": relative.as_posix(),
                "bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            }
        )
    atomic_json(
        destination / "archive-inventory.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_campaign": str(source),
            "file_count": len(copied),
            "files": copied,
            "updated_at_utc": utc_now(),
            "excludes": [
                "private RNG key",
                "adapter tensors",
                "checkpoint tensors",
                "files larger than 32 MiB",
            ],
        },
    )


def cuda_smoke(args: argparse.Namespace) -> None:
    step_environment = validated_step_environment()
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch import failed") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each Slurm smoke lane must expose exactly one CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    expected_name = args.expected_device_name.strip().upper()
    if not expected_name or expected_name not in properties.name.upper():
        raise RuntimeError(
            f"campaign requires a {args.expected_device_name} lane, "
            f"found {properties.name}"
        )
    minimum_vram_bytes = int(args.min_vram_gib) * 1024**3
    if int(properties.total_memory) < minimum_vram_bytes:
        raise RuntimeError(
            f"visible {properties.name} reports less than "
            f"{args.min_vram_gib} GiB VRAM"
        )
    left = torch.randn(1024, 1024, device="cuda")
    right = torch.randn(1024, 1024, device="cuda")
    checksum = float((left @ right).mean().item())
    if not math.isfinite(checksum):
        raise RuntimeError("CUDA matrix smoke produced a non-finite result")
    atomic_json(
        args.output.resolve(),
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASSED",
            "lane": args.lane,
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_name": properties.name,
            "total_vram_bytes": int(properties.total_memory),
            "expected_device_name_substring": args.expected_device_name,
            "minimum_vram_bytes": minimum_vram_bytes,
            "visible_device_count": torch.cuda.device_count(),
            "step_environment": step_environment,
            "smoke_checksum": checksum,
            "completed_at_utc": utc_now(),
        },
    )


def validated_step_environment() -> dict[str, str]:
    """Fail before model loading if an ``srun`` lost the batch contract."""

    mismatches = {
        name: {"expected": expected, "actual": os.environ.get(name)}
        for name, expected in REQUIRED_STEP_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    for name in ("HF_HOME", "TMPDIR"):
        value = os.environ.get(name)
        if not value or not Path(value).is_absolute():
            mismatches[name] = {"expected": "absolute path", "actual": value}
    threads = os.environ.get("OMP_NUM_THREADS")
    if not threads or not threads.isdecimal() or int(threads) <= 0:
        mismatches["OMP_NUM_THREADS"] = {
            "expected": "positive integer",
            "actual": threads,
        }
    if mismatches:
        raise RuntimeError(
            "Slurm step environment contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        **REQUIRED_STEP_ENVIRONMENT,
        "HF_HOME": str(Path(os.environ["HF_HOME"]).resolve()),
        "TMPDIR": str(Path(os.environ["TMPDIR"]).resolve()),
        "OMP_NUM_THREADS": str(int(os.environ["OMP_NUM_THREADS"])),
    }


def validate_spec_command(args: argparse.Namespace) -> None:
    spec = load_spec(args.spec.resolve())
    arms = expand_spec(spec)
    families: dict[str, int] = {}
    methods: dict[str, int] = {}
    for arm in arms:
        families[arm["family"]] = families.get(arm["family"], 0) + 1
        methods[arm["method"]] = methods.get(arm["method"], 0) + 1
    print(
        json.dumps(
            {
                "status": "VALID",
                "spec_sha256": sha256_file(args.spec.resolve()),
                "arm_count": len(arms),
                "wave_count": len(arms) // 2,
                "families": families,
                "methods": methods,
            },
            indent=2,
            sort_keys=True,
        )
    )


def print_waves(args: argparse.Namespace) -> None:
    runtime = load_runtime(args.manifest.resolve())
    arms = runtime["arms"]
    for index in range(0, len(arms), 2):
        print(f"{index}\t{index + 1}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-spec")
    validate.add_argument("--spec", type=Path, required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--repository", type=Path, required=True)
    prepare.add_argument("--expected-code-sha", required=True)
    prepare.add_argument("--input-manifest", type=Path, required=True)
    prepare.add_argument("--campaign-root", type=Path, required=True)
    prepare.add_argument("--private-key", type=Path, required=True)
    prepare.add_argument("--resume", action="store_true")

    arm = subparsers.add_parser("run-arm")
    arm.add_argument("--manifest", type=Path, required=True)
    arm.add_argument("--arm-index", type=int, required=True)
    arm.add_argument("--repository", type=Path, required=True)
    arm.add_argument("--python-bin", type=Path, required=True)
    arm.add_argument("--private-key", type=Path, required=True)

    preflight_smoke = subparsers.add_parser("run-smoke")
    preflight_smoke.add_argument("--manifest", type=Path, required=True)
    preflight_smoke.add_argument("--lane", type=int, choices=(0, 1), required=True)
    preflight_smoke.add_argument(
        "--method",
        choices=(FIXED_DP_METHOD, FULL_SLACLIP_METHOD, ORACLE_SLACLIP_METHOD),
    )
    preflight_smoke.add_argument("--repository", type=Path, required=True)
    preflight_smoke.add_argument("--python-bin", type=Path, required=True)
    preflight_smoke.add_argument("--private-key", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--require-complete", action="store_true")

    job_status = subparsers.add_parser("mark-job-status")
    job_status.add_argument("--campaign-root", type=Path, required=True)
    job_status.add_argument("--runtime-manifest", type=Path, required=True)
    job_status.add_argument(
        "--status",
        choices=("RUNNING", "COMPLETED", "FAILED"),
        required=True,
    )
    job_status.add_argument("--slurm-job-id", required=True)
    job_status.add_argument("--repository-sha", required=True)
    job_status.add_argument("--reason", required=True)
    job_status.add_argument("--exit-code", type=int)

    stale_status = subparsers.add_parser("recover-stale-job-status")
    stale_status.add_argument("--campaign-root", type=Path, required=True)
    stale_status.add_argument("--runtime-manifest", type=Path, required=True)
    stale_status.add_argument("--repository-sha", required=True)
    stale_status.add_argument("--check-only", action="store_true")

    archive = subparsers.add_parser("archive-small")
    archive.add_argument("--campaign-root", type=Path, required=True)
    archive.add_argument("--archive-root", type=Path, required=True)

    waves = subparsers.add_parser("waves")
    waves.add_argument("--manifest", type=Path, required=True)

    smoke = subparsers.add_parser("cuda-smoke")
    smoke.add_argument("--lane", type=int, choices=(0, 1), required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--expected-device-name", required=True)
    smoke.add_argument("--min-vram-gib", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    args = parse_args(argv)
    if args.command == "validate-spec":
        validate_spec_command(args)
        return 0
    if args.command == "prepare":
        prepare_campaign(args)
        return 0
    if args.command == "run-arm":
        return run_arm(args)
    if args.command == "run-smoke":
        return run_preflight_smoke(args)
    if args.command == "aggregate":
        aggregate_campaign(args)
        return 0
    if args.command == "mark-job-status":
        mark_job_status(args)
        return 0
    if args.command == "recover-stale-job-status":
        recover_stale_job_status(args)
        return 0
    if args.command == "archive-small":
        archive_small(args)
        return 0
    if args.command == "waves":
        print_waves(args)
        return 0
    if args.command == "cuda-smoke":
        cuda_smoke(args)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
