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
CAMPAIGN_KEY_BYTES = 32
FULL_SLACLIP_METHOD = "slaclip_dp_lora"
FIXED_DP_METHOD = "paper_dp_lora"
CONTROL_METHODS = ("no_dp_lora_control", "clip_only_control")
ALLOWED_METHODS = {FULL_SLACLIP_METHOD, FIXED_DP_METHOD, *CONTROL_METHODS}
EXPECTED_MODELS = ("bert", "gpt2")
SMALL_ARCHIVE_MAX_BYTES = 32 * 1024 * 1024
FULL_COMPARISON_STATUS = "FULL_SLACLIP_COMPARISON_COMPLETE"


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
        "eval_every",
        "checkpoint_every",
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
        "eval_every",
        "checkpoint_every",
        "slaclip_num_slots",
    )
    normalized = dict(common)
    for name in integers:
        normalized[name] = require_int(common[name], f"common.{name}")
    for name in (
        "noise_multiplier",
        "learning_rate",
        "slaclip_c_min",
        "slaclip_c_max",
        "default_eta",
        "default_beta",
    ):
        normalized[name] = require_number(common[name], f"common.{name}", positive=True)
    if normalized["slaclip_c_max"] < normalized["slaclip_c_min"]:
        raise ValueError("SlaClip threshold bounds are reversed")
    if not 0 < normalized["default_beta"] <= 1:
        raise ValueError("default_beta must be in (0, 1]")
    if normalized["rounds"] != 50 or normalized["batch_size"] != 8:
        raise ValueError("formal campaign must retain T=50 and B=8")
    if normalized["slaclip_num_slots"] != 15:
        raise ValueError("this campaign freezes the requested K_slots=15 policy")
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
            "sensitivity",
            "controls",
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
) -> dict[str, Any]:
    if method not in ALLOWED_METHODS:
        raise ValueError(f"unsupported method: {method}")
    adaptive = method == FULL_SLACLIP_METHOD
    if adaptive != (eta is not None and beta is not None):
        raise ValueError("adaptive controller parameters and method disagree")
    if family == "primary" and clip_norm == 10.0:
        analysis_role = "paper_style_primary"
    elif family == "primary":
        analysis_role = "pre_registered_initial_threshold_sensitivity"
    elif family == "sensitivity":
        analysis_role = "pre_registered_controller_hyperparameter_sensitivity"
    elif family == "control":
        analysis_role = "mechanism_control"
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
        "slaclip_beta": beta,
        "reference_arm_id": reference_arm_id,
        "rng_domain": f"full-slaclip-cdf:s{seed}:c{number_token(clip_norm)}",
        "models": list(common["models"]),
        "num_clients": common["num_clients"],
        "rounds": common["rounds"],
        "batch_size": common["batch_size"],
        "noise_multiplier": common["noise_multiplier"],
        "learning_rate": common["learning_rate"],
        "rank": common["rank"],
        "max_seq_length": common["max_seq_length"],
        "eval_every": common["eval_every"],
        "checkpoint_every": common["checkpoint_every"],
        "slaclip_num_slots": common["slaclip_num_slots"] if adaptive else None,
        "slaclip_c_min": common["slaclip_c_min"] if adaptive else None,
        "slaclip_c_max": common["slaclip_c_max"] if adaptive else None,
    }


def expand_spec(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    common = _common_values(spec)
    primary = spec.get("primary")
    sensitivity = spec.get("sensitivity")
    controls = spec.get("controls")
    if (
        not isinstance(primary, dict)
        or not isinstance(sensitivity, dict)
        or not isinstance(controls, dict)
    ):
        raise ValueError("primary, sensitivity, and controls must be objects")
    require_exact_keys(primary, {"initial_clip_norms", "seeds", "methods"}, "primary")
    require_exact_keys(
        sensitivity,
        {"initial_clip_norm", "seeds", "etas", "betas", "method", "exclude_primary_default"},
        "sensitivity",
    )
    require_exact_keys(controls, {"initial_clip_norm", "seeds", "methods"}, "controls")

    clip_norms = primary["initial_clip_norms"]
    primary_seeds = primary["seeds"]
    primary_methods = primary["methods"]
    if (
        not isinstance(clip_norms, list)
        or not isinstance(primary_seeds, list)
        or not isinstance(primary_methods, list)
    ):
        raise ValueError("primary axes must be arrays")
    clip_norms = [
        require_number(value, "primary.initial_clip_norm", positive=True)
        for value in clip_norms
    ]
    primary_seeds = [require_int(value, "primary.seed", positive=False) for value in primary_seeds]
    if tuple(primary_methods) != (FIXED_DP_METHOD, FULL_SLACLIP_METHOD):
        raise ValueError("primary methods must be fixed DP then full SlaClip")
    _unique(clip_norms, "primary.initial_clip_norms")
    _unique(primary_seeds, "primary.seeds")

    arms: list[dict[str, Any]] = []
    default_eta = float(common["default_eta"])
    default_beta = float(common["default_beta"])
    for clip_norm in clip_norms:
        if not common["slaclip_c_min"] <= clip_norm <= common["slaclip_c_max"]:
            raise ValueError("primary initial threshold is outside controller bounds")
        clip_token = number_token(clip_norm)
        for seed in primary_seeds:
            fixed_id = f"primary-c{clip_token}-s{seed}-fixed"
            adaptive_id = f"primary-c{clip_token}-s{seed}-slaclip"
            arms.append(
                _base_arm(
                    common=common,
                    arm_id=fixed_id,
                    family="primary",
                    method=FIXED_DP_METHOD,
                    seed=seed,
                    clip_norm=clip_norm,
                    eta=None,
                    beta=None,
                    reference_arm_id=None,
                )
            )
            arms.append(
                _base_arm(
                    common=common,
                    arm_id=adaptive_id,
                    family="primary",
                    method=FULL_SLACLIP_METHOD,
                    seed=seed,
                    clip_norm=clip_norm,
                    eta=default_eta,
                    beta=default_beta,
                    reference_arm_id=fixed_id,
                )
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
    betas = [require_number(value, "sensitivity.beta", positive=True) for value in betas]
    if any(value > 1 for value in betas):
        raise ValueError("sensitivity beta must not exceed one")
    _unique(sensitivity_seeds, "sensitivity.seeds")
    _unique(etas, "sensitivity.etas")
    _unique(betas, "sensitivity.betas")
    if (
        sensitivity["method"] != FULL_SLACLIP_METHOD
        or sensitivity["exclude_primary_default"] is not True
    ):
        raise ValueError("sensitivity must be de-duplicated full SlaClip")
    if sensitivity_clip not in clip_norms or not set(sensitivity_seeds).issubset(primary_seeds):
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
    if control_clip not in clip_norms or not set(control_seeds).issubset(primary_seeds):
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
        "--seed",
        str(arm["seed"]),
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
    if arm["method"] == FULL_SLACLIP_METHOD:
        command.extend(
            [
                "--slaclip-eta",
                str(arm["slaclip_eta"]),
                "--slaclip-beta",
                str(arm["slaclip_beta"]),
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
    finally:
        stop_watcher_done.set()
        watcher.join(timeout=2.0)

    if return_code == 0:
        final_summary = output_dir / "final_summary.json"
        summary = load_object(final_summary, "arm final summary")
        if summary.get("status") != "COMPLETED" or summary.get("method") != arm["method"]:
            raise RuntimeError(
                "arm returned zero without a matching completed summary: "
                f"{arm['arm_id']}"
            )
        _write_arm_status(
            status_path,
            arm,
            "COMPLETED",
            exit_code=0,
            final_summary_sha256=sha256_file(final_summary),
        )
    elif return_code == 75:
        _write_arm_status(status_path, arm, "CHECKPOINTED_STOP", exit_code=75)
    else:
        _write_arm_status(status_path, arm, "FAILED", exit_code=return_code)
    return return_code


def run_preflight_smoke(args: argparse.Namespace) -> int:
    """Run or deeply revalidate one real-model smoke inside the allocation."""

    manifest_path = args.manifest.resolve()
    runtime = load_runtime(manifest_path)
    repository = args.repository.resolve()
    if repository_sha(repository) != runtime["repository_sha"] or repository_dirty(repository):
        raise RuntimeError("repository snapshot changed before real-model smoke")
    desired_method = FIXED_DP_METHOD if args.lane == 0 else FULL_SLACLIP_METHOD
    candidates = [
        arm
        for arm in runtime["arms"]
        if arm["family"] == "primary"
        and arm["method"] == desired_method
        and arm["seed"] == 42
        and arm["initial_clip_norm"] == 10.0
    ]
    if len(candidates) != 1:
        raise RuntimeError("could not resolve the unique paper-style smoke template")
    arm = dict(candidates[0])
    label = "paper-baseline" if args.lane == 0 else "full-slaclip"
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
    initial_loss = final_loss = None
    if isinstance(evaluations, list) and evaluations:
        first, last = evaluations[0], evaluations[-1]
        if isinstance(first, dict) and isinstance(first.get("loss"), (int, float)):
            initial_loss = float(first["loss"])
        if isinstance(last, dict) and isinstance(last.get("loss"), (int, float)):
            final_loss = float(last["loss"])
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
        "seed": arm["seed"],
        "initial_clip_norm": arm["initial_clip_norm"],
        "slaclip_eta": arm["slaclip_eta"],
        "slaclip_beta": arm["slaclip_beta"],
        "reference_arm_id": arm["reference_arm_id"],
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_delta": (
            final_loss - initial_loss
            if initial_loss is not None and final_loss is not None
            else None
        ),
        "actual_clipped_fraction": any_group.get("fraction"),
        "would_clip_fraction": any_group.get("would_fraction"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
    }
    aliases = {
        "cdf_near_threshold_median": ("near_threshold_proxy",),
        "cdf_near_zero_median": ("near_zero_proxy",),
        "near_zero_adjusted_median": ("near_zero_adjusted",),
        "dynamic_target_median": ("dynamic_target_unclipped",),
        "dynamic_target_clipped_median": ("dynamic_target_clipped",),
        "controller_error_median": ("controller_error",),
        "near_threshold_proxy_error_median": ("near_threshold_proxy_error",),
        "near_zero_proxy_error_median": ("near_zero_proxy_error",),
        "raw_log_step_median": ("raw_log_step",),
        "threshold_used_median": ("clip_threshold_used",),
        "next_threshold_median": ("next_clip_threshold",),
    }
    behavior_metric_names = (
        "raw_gradient_l2",
        "clipped_signal_gradient_l2",
        "noise_gradient_l2",
        "signal_to_noise_l2_ratio",
        "signal_noise_cosine",
        "global_update_l2",
    )
    controller_count_names = (
        "gamma_clamped_low_count",
        "gamma_clamped_high_count",
        "log_step_bounded_count",
        "lower_bound_hits",
        "upper_bound_hits",
        "noisy_adjacent_monotonicity_violations",
        "exact_adjacent_monotonicity_violations",
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
        result[f"final_threshold_{group_name}"] = group.get("final_next_clip_threshold")
    return result


METRIC_COLUMNS = (
    "arm_id",
    "family",
    "analysis_role",
    "method",
    "seed",
    "initial_clip_norm",
    "slaclip_eta",
    "slaclip_beta",
    "reference_arm_id",
    "model",
    "initial_loss",
    "final_loss",
    "loss_delta",
    "actual_clipped_fraction",
    "would_clip_fraction",
    "elapsed_seconds",
    "cdf_near_threshold_median_A",
    "cdf_near_zero_median_A",
    "near_zero_adjusted_median_A",
    "dynamic_target_median_A",
    "dynamic_target_clipped_median_A",
    "controller_error_median_A",
    "near_threshold_proxy_error_median_A",
    "near_zero_proxy_error_median_A",
    "raw_log_step_median_A",
    "threshold_used_median_A",
    "next_threshold_median_A",
    "final_threshold_A",
    "actual_clipped_fraction_A",
    "would_clip_fraction_A",
    "raw_gradient_l2_median_A",
    "clipped_signal_gradient_l2_median_A",
    "noise_gradient_l2_median_A",
    "signal_to_noise_l2_ratio_median_A",
    "signal_noise_cosine_median_A",
    "global_update_l2_median_A",
    "gamma_clamped_low_count_A",
    "gamma_clamped_high_count_A",
    "log_step_bounded_count_A",
    "lower_bound_hits_A",
    "upper_bound_hits_A",
    "noisy_adjacent_monotonicity_violations_A",
    "exact_adjacent_monotonicity_violations_A",
    "cdf_near_threshold_median_B",
    "cdf_near_zero_median_B",
    "near_zero_adjusted_median_B",
    "dynamic_target_median_B",
    "dynamic_target_clipped_median_B",
    "controller_error_median_B",
    "near_threshold_proxy_error_median_B",
    "near_zero_proxy_error_median_B",
    "raw_log_step_median_B",
    "threshold_used_median_B",
    "next_threshold_median_B",
    "final_threshold_B",
    "actual_clipped_fraction_B",
    "would_clip_fraction_B",
    "raw_gradient_l2_median_B",
    "clipped_signal_gradient_l2_median_B",
    "noise_gradient_l2_median_B",
    "signal_to_noise_l2_ratio_median_B",
    "signal_noise_cosine_median_B",
    "global_update_l2_median_B",
    "gamma_clamped_low_count_B",
    "gamma_clamped_high_count_B",
    "log_step_bounded_count_B",
    "lower_bound_hits_B",
    "upper_bound_hits_B",
    "noisy_adjacent_monotonicity_violations_B",
    "exact_adjacent_monotonicity_violations_B",
)

AGGREGATED_METRICS = (
    "final_loss",
    "loss_delta",
    "actual_clipped_fraction",
    "actual_clipped_fraction_A",
    "actual_clipped_fraction_B",
    "raw_gradient_l2_median_A",
    "raw_gradient_l2_median_B",
    "clipped_signal_gradient_l2_median_A",
    "clipped_signal_gradient_l2_median_B",
    "noise_gradient_l2_median_A",
    "noise_gradient_l2_median_B",
    "signal_to_noise_l2_ratio_median_A",
    "signal_to_noise_l2_ratio_median_B",
    "signal_noise_cosine_median_A",
    "signal_noise_cosine_median_B",
    "global_update_l2_median_A",
    "global_update_l2_median_B",
    "cdf_near_threshold_median_A",
    "cdf_near_threshold_median_B",
    "cdf_near_zero_median_A",
    "cdf_near_zero_median_B",
    "dynamic_target_median_A",
    "dynamic_target_median_B",
    "controller_error_median_A",
    "controller_error_median_B",
    "near_threshold_proxy_error_median_A",
    "near_threshold_proxy_error_median_B",
    "near_zero_proxy_error_median_A",
    "near_zero_proxy_error_median_B",
    "final_threshold_A",
    "final_threshold_B",
    "lower_bound_hits_A",
    "lower_bound_hits_B",
    "upper_bound_hits_A",
    "upper_bound_hits_B",
    "noisy_adjacent_monotonicity_violations_A",
    "noisy_adjacent_monotonicity_violations_B",
    "exact_adjacent_monotonicity_violations_A",
    "exact_adjacent_monotonicity_violations_B",
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
    if len(adaptive_arms) != 39:
        raise RuntimeError(
            f"runtime manifest has {len(adaptive_arms)} adaptive arms, expected 39"
        )
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
            f"final comparison evidence count is {len(records)}, expected 39"
        )
    return records


def aggregate_campaign(args: argparse.Namespace) -> bool:
    manifest_path = args.manifest.resolve()
    runtime = load_runtime(manifest_path)
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
                "seed": row["seed"],
                "initial_clip_norm": row["initial_clip_norm"],
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

    group_keys = (
        "family",
        "method",
        "initial_clip_norm",
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
        "initial_clip_norm",
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
        "actual_clipped_fraction_difference",
    )
    for key, group_rows in paired_ordered_groups:
        output = dict(zip(paired_group_keys, key))
        output["seed_count"] = len({row["seed"] for row in group_rows})
        for metric in paired_difference_metrics:
            values = _finite_values(group_rows, metric)
            output[f"{metric}_n"] = len(values)
            output[f"{metric}_mean"] = statistics.fmean(values) if values else None
            output[f"{metric}_sample_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        paired_aggregate_rows.append(output)

    paired_columns = (
        "arm_id",
        "reference_arm_id",
        "family",
        "seed",
        "initial_clip_norm",
        "slaclip_eta",
        "slaclip_beta",
        "model",
        "slaclip_final_loss",
        "fixed_final_loss",
        "final_loss_difference_slaclip_minus_fixed",
        "slaclip_actual_clipped_fraction",
        "fixed_actual_clipped_fraction",
        "actual_clipped_fraction_difference",
    )
    aggregate_columns = list(group_keys) + ["seed_count"]
    for metric in AGGREGATED_METRICS:
        aggregate_columns.extend((f"{metric}_n", f"{metric}_mean", f"{metric}_sample_std"))
    paired_aggregate_columns = list(paired_group_keys) + ["seed_count"]
    for metric in paired_difference_metrics:
        paired_aggregate_columns.extend(
            (f"{metric}_n", f"{metric}_mean", f"{metric}_sample_std")
        )
    atomic_csv(root / "campaign_metrics.csv", rows, METRIC_COLUMNS)
    atomic_csv(root / "paired_metrics.csv", paired_rows, paired_columns)
    atomic_csv(root / "aggregate_metrics.csv", aggregate_rows, aggregate_columns)
    atomic_csv(
        root / "paired_aggregate_metrics.csv",
        paired_aggregate_rows,
        paired_aggregate_columns,
    )
    expected_comparisons = sum(
        arm["method"] == FULL_SLACLIP_METHOD for arm in runtime["arms"]
    )
    completed = (
        len(completed_arm_ids) == runtime["expected_arm_count"]
        and len(comparison_records) == expected_comparisons
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
        "aggregate_row_count": len(aggregate_rows),
        "paired_aggregate_row_count": len(paired_aggregate_rows),
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
        "campaign_summary.json",
        "campaign_metrics.csv",
        "paired_metrics.csv",
        "aggregate_metrics.csv",
        "paired_aggregate_metrics.csv",
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
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch import failed") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each Slurm smoke lane must expose exactly one CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    if "H200" not in properties.name.upper():
        raise RuntimeError(f"campaign requires an H200 lane, found {properties.name}")
    if int(properties.total_memory) < 100 * 1024**3:
        raise RuntimeError("visible H200 reports unexpectedly low VRAM")
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
            "visible_device_count": torch.cuda.device_count(),
            "smoke_checksum": checksum,
            "completed_at_utc": utc_now(),
        },
    )


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
    preflight_smoke.add_argument("--repository", type=Path, required=True)
    preflight_smoke.add_argument("--python-bin", type=Path, required=True)
    preflight_smoke.add_argument("--private-key", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--require-complete", action="store_true")

    archive = subparsers.add_parser("archive-small")
    archive.add_argument("--campaign-root", type=Path, required=True)
    archive.add_argument("--archive-root", type=Path, required=True)

    waves = subparsers.add_parser("waves")
    waves.add_argument("--manifest", type=Path, required=True)

    smoke = subparsers.add_parser("cuda-smoke")
    smoke.add_argument("--lane", type=int, choices=(0, 1), required=True)
    smoke.add_argument("--output", type=Path, required=True)
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
