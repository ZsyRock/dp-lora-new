#!/usr/bin/env python3
"""Validate and summarize one matched DP-LoRA/full-SlaClip Level-1 pair.

Full SlaClip reconstructs a noisy binned CDF and updates the clipping norm
from its endpoint nearest the current threshold and its endpoint nearest zero.
The historical shared-beta contract has no baseline-derived calibration.  The
groupwise generalized-beta contract may freeze A/B beta hyperparameters chosen
from an immutable NON-DP fixed-baseline development lock, but never consumes
that calibration data at controller runtime.  Neither contract uses a fixed
target clipping rate: exact CDF indicators remain NON-DP diagnostics and only
the noisy endpoints are permitted to drive the online controller.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from paper_repro.reproducibility import (
        METHOD_SPECS,
        canonical_json_fingerprint,
        safe_quantiles,
    )
except ModuleNotFoundError:  # Support direct execution.
    from reproducibility import (  # type: ignore[no-redef]
        METHOD_SPECS,
        canonical_json_fingerprint,
        safe_quantiles,
    )


BASELINE_METHOD = "paper_dp_lora"
ADAPTIVE_METHOD = "slaclip_dp_lora"
EXPECTED_METHODS = (BASELINE_METHOD, ADAPTIVE_METHOD)
DEFAULT_MODELS = ("bert", "gpt2")
LORA_GROUPS = ("A", "B")
COMPARISON_SCHEMA_VERSION = 2
COMPARISON_STATUS = "FULL_SLACLIP_COMPARISON_COMPLETE"
SLACLIP_CONTRACT_SCHEMA = "full_slaclip_contract_v1"
SLACLIP_VARIANT = "full_slaclip_cdf_endpoints"
GROUPWISE_SLACLIP_CONTRACT_SCHEMA = (
    "groupwise_generalized_full_slaclip_beta_contract_v1"
)
GROUPWISE_SLACLIP_VARIANT = "groupwise_generalized_full_slaclip_beta"
SLACLIP_VARIANT_BY_CONTRACT_SCHEMA = {
    SLACLIP_CONTRACT_SCHEMA: SLACLIP_VARIANT,
    GROUPWISE_SLACLIP_CONTRACT_SCHEMA: GROUPWISE_SLACLIP_VARIANT,
}
NOISY_CONTROLLER_INPUT = "noisy_endpoints"
ROUND_PREFIX_DOMAIN = b"dp-lora-round-shard-prefix-v1\0"
EXACT_CDF_FLOAT32_TOLERANCE = 1e-6
SLACLIP_Z_0995 = 2.5758293035489004
MAX_ABS_LOG_STEP = 50.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    raw = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if os.path.lexists(raw):
        # Reject a symlink at the requested object itself, while allowing the
        # cluster's canonicalized /scratch ancestor alias.
        if stat.S_ISLNK(raw.lstat().st_mode):
            return raw
        return raw.resolve(strict=True)
    return raw.parent.resolve(strict=True) / raw.name


def _validate_private_directory(path: Path, description: str) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{description} is not a real directory: {path}")
    if metadata.st_uid != os.getuid():
        raise RuntimeError(f"{description} is not owned by this user: {path}")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != 0o700:
        raise RuntimeError(
            f"{description} must have mode 0700, found {mode:04o}: {path}"
        )
    if path.resolve(strict=True) != path:
        raise RuntimeError(f"{description} contains a symlink component: {path}")


def _validate_private_file_metadata(
    metadata: os.stat_result, path: Path, description: str
) -> None:
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{description} is not a real regular file: {path}")
    if metadata.st_uid != os.getuid():
        raise RuntimeError(f"{description} is not owned by this user: {path}")
    if metadata.st_nlink != 1:
        raise RuntimeError(
            f"{description} must have exactly one hard link, found "
            f"{metadata.st_nlink}: {path}"
        )
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != 0o600:
        raise RuntimeError(
            f"{description} must have mode 0600, found {mode:04o}: {path}"
        )


def _read_private_bytes(path: Path, description: str) -> bytes:
    before = path.lstat()
    _validate_private_file_metadata(before, path, description)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _validate_private_file_metadata(opened, path, description)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"{description} changed during validation: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        finished = os.fstat(descriptor)
        _validate_private_file_metadata(finished, path, description)
        before_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        after_identity = (
            finished.st_dev,
            finished.st_ino,
            finished.st_size,
            finished.st_mtime_ns,
            finished.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise RuntimeError(f"{description} changed while reading: {path}")
        value = b"".join(chunks)
        if len(value) != finished.st_size:
            raise RuntimeError(f"{description} read was incomplete: {path}")
        return value
    finally:
        os.close(descriptor)


def _load_private_object(
    path: Path, description: str
) -> tuple[dict[str, Any], bytes]:
    encoded = _read_private_bytes(path, description)
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value, encoded


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{description} is not a lowercase SHA-256 digest")
    return value


def _finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{description} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{description} must be a finite number")
    return result


def _fraction(value: Any, description: str) -> float:
    result = _finite_number(value, description)
    if not 0.0 <= result <= 1.0:
        raise RuntimeError(f"{description} must be in [0, 1]")
    return result


def _group_fraction_mapping(value: Any, description: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(LORA_GROUPS):
        raise RuntimeError(f"{description} must contain exactly A and B")
    return {
        group: _fraction(value[group], f"{description}[{group}]")
        for group in LORA_GROUPS
    }


def _positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{description} must be a positive integer")
    return value


def _independent_automatic_num_slots(
    expected_records: Any, noise_multiplier: Any
) -> int:
    """Recompute the paper's automatic K rule without production helpers."""

    records = _finite_number(expected_records, "SlaClip expected records")
    sigma = _finite_number(noise_multiplier, "SlaClip noise multiplier")
    if records <= 0.0 or sigma <= 0.0:
        raise RuntimeError("SlaClip K-rule inputs must be positive")
    ratio = records / (2.0 * SLACLIP_Z_0995 * sigma)
    upper_bound = ratio ** (2.0 / 3.0)
    if not math.isfinite(upper_bound):
        raise RuntimeError("SlaClip automatic slot bound is not finite")
    return max(1, math.floor(upper_bound))


def _require_claims(
    run_config: Mapping[str, Any],
    root_summary: Mapping[str, Any],
    *,
    adaptive: bool,
) -> None:
    reproduction = run_config.get("reproduction_claim")
    if not isinstance(reproduction, dict):
        raise RuntimeError("run config has no reproduction claim")
    for key, expected in {
        "level": 1,
        "paper_result_reproduced": False,
        "paper_benchmarks_evaluated": False,
    }.items():
        if reproduction.get(key) != expected:
            raise RuntimeError(f"unsafe reproduction claim: {key}")

    privacy = run_config.get("privacy_claim")
    if not isinstance(privacy, dict):
        raise RuntimeError("run config has no privacy claim")
    scientific_contract = run_config.get("scientific_contract")
    algorithm = (
        scientific_contract.get("algorithm_contract")
        if isinstance(scientific_contract, dict)
        else None
    )
    slaclip_contract = (
        algorithm.get("slaclip") if isinstance(algorithm, dict) else None
    )
    groupwise = bool(
        adaptive
        and isinstance(slaclip_contract, dict)
        and slaclip_contract.get("schema_version")
        == GROUPWISE_SLACLIP_CONTRACT_SCHEMA
    )
    for key, expected in {
        "end_to_end_dp_certified": False,
        "epsilon": None,
        "sigma_is_not_epsilon": True,
        "diagnostics_are_private_non_dp_data": True,
        "baseline_derived_calibration_is_non_dp": groupwise,
        "exact_cdf_diagnostics_are_non_dp": adaptive,
    }.items():
        if privacy.get(key) != expected:
            raise RuntimeError(f"unsafe privacy claim: {key}")
    if groupwise:
        _require_sha256(
            privacy.get("baseline_calibration_lock_sha256"),
            "groupwise SlaClip baseline calibration lock",
        )
        if privacy.get("baseline_calibration_consumed_at_runtime") is not False:
            raise RuntimeError(
                "unsafe privacy claim: baseline_calibration_consumed_at_runtime"
            )
    else:
        if privacy.get("baseline_calibration_lock_sha256") not in (None,):
            raise RuntimeError(
                "shared SlaClip privacy claim unexpectedly contains a calibration lock"
            )
        if privacy.get("baseline_calibration_consumed_at_runtime", False) is not False:
            raise RuntimeError(
                "unsafe privacy claim: baseline_calibration_consumed_at_runtime"
            )

    for key, expected in {
        "reproduction_claim_level": 1,
        "paper_result_reproduced": False,
        "paper_benchmarks_evaluated": False,
        "privacy_claim": False,
    }.items():
        if root_summary.get(key) != expected:
            raise RuntimeError(f"unsafe root claim: {key}")


def _evaluation_delta(model_summary: Mapping[str, Any], rounds: int) -> dict[str, Any]:
    evaluations = model_summary.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) < 2:
        raise RuntimeError("model summary has no initial/final evaluation pair")
    if any(not isinstance(item, dict) for item in evaluations):
        raise RuntimeError("model evaluation entry is invalid")
    initial = evaluations[0]
    final = evaluations[-1]
    if initial.get("round") != 0 or final.get("round") != rounds:
        raise RuntimeError("model evaluation round range is incomplete")
    if model_summary.get("final_evaluation") != final:
        raise RuntimeError("model final evaluation does not match its evaluation series")
    initial_loss = _finite_number(initial.get("loss"), "initial evaluation loss")
    final_loss = _finite_number(final.get("loss"), "final evaluation loss")
    return {
        "metric": "internal_disjoint_meddialog_lm_loss_not_paper_benchmark",
        "initial": initial_loss,
        "final": final_loss,
        "final_minus_initial": final_loss - initial_loss,
        "improved": final_loss < initial_loss,
    }


def _round_shard_prefix_sha256(rounds_directory: Path, rounds: int) -> str:
    _validate_private_directory(rounds_directory, "round diagnostics directory")
    expected_paths = [
        rounds_directory / f"round-{round_index:05d}.json"
        for round_index in range(1, rounds + 1)
    ]
    actual_paths = set(rounds_directory.glob("round-*.json"))
    if actual_paths != set(expected_paths):
        raise RuntimeError("round shard set mismatch")
    digest = hashlib.sha256(ROUND_PREFIX_DOMAIN)
    for path in expected_paths:
        encoded = _read_private_bytes(path, "round diagnostic shard")
        digest.update(path.name.encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(encoded).digest())
    digest.update(rounds.to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _load_arm(directory: Path, method: str, *, adaptive: bool) -> dict[str, Any]:
    _validate_private_directory(directory, f"{method} run directory")
    run_config, run_config_bytes = _load_private_object(
        directory / "run_config.json", f"{method} run config"
    )
    root_summary, root_summary_bytes = _load_private_object(
        directory / "final_summary.json", f"{method} root summary"
    )
    expected_method_spec = asdict(METHOD_SPECS[method])
    for key, expected in {
        "schema_version": 2,
        "method": method,
        "method_spec": expected_method_spec,
        "contains_slaclip": adaptive,
    }.items():
        if run_config.get(key) != expected:
            raise RuntimeError(f"{method} run-config identity mismatch: {key}")
    contract = run_config.get("scientific_contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != 2:
        raise RuntimeError(f"{method} scientific contract is invalid")
    run_fingerprint = canonical_json_fingerprint(contract)
    if run_config.get("run_config_fingerprint") != run_fingerprint:
        raise RuntimeError(f"{method} scientific-contract fingerprint mismatch")
    if contract.get("method") != expected_method_spec:
        raise RuntimeError(f"{method} scientific method contract mismatch")
    effective = contract.get("effective_config")
    if (
        not isinstance(effective, dict)
        or effective.get("method") != method
        or run_config.get("effective_config") != effective
    ):
        raise RuntimeError(f"{method} effective configuration mismatch")
    models = run_config.get("models")
    if models != list(DEFAULT_MODELS) or contract.get("models") != models:
        raise RuntimeError(f"{method} model set/order mismatch")
    algorithm = contract.get("algorithm_contract")
    if not isinstance(algorithm, dict):
        raise RuntimeError(f"{method} algorithm contract is missing")
    if algorithm.get("contains_slaclip") is not adaptive:
        raise RuntimeError(f"{method} algorithm SlaClip flag mismatch")
    slaclip_contract = algorithm.get("slaclip")
    if adaptive != isinstance(slaclip_contract, dict):
        raise RuntimeError(f"{method} full-SlaClip contract presence mismatch")
    if algorithm.get("slaclip_q") is not None:
        raise RuntimeError(f"{method} unexpectedly enables legacy SlaClip-Q")

    for key, expected in {
        "schema_version": 2,
        "status": "COMPLETED",
        "method": method,
        "method_spec": expected_method_spec,
        "contains_slaclip": adaptive,
        "run_config_fingerprint": run_fingerprint,
    }.items():
        if root_summary.get(key) != expected:
            raise RuntimeError(f"{method} root-summary identity mismatch: {key}")
    _require_claims(run_config, root_summary, adaptive=adaptive)
    root_models = root_summary.get("models")
    if not isinstance(root_models, dict) or tuple(root_models) != DEFAULT_MODELS:
        raise RuntimeError(f"{method} root-summary model set/order mismatch")
    root_schedules = root_summary.get("sample_schedule_sha256_by_model")
    if not isinstance(root_schedules, dict) or tuple(root_schedules) != DEFAULT_MODELS:
        raise RuntimeError(f"{method} root sample-schedule map is invalid")

    rounds = _positive_integer(effective.get("rounds"), f"{method} rounds")
    clients = _positive_integer(effective.get("num_clients"), f"{method} clients")
    model_records: dict[str, Any] = {}
    for model in DEFAULT_MODELS:
        model_dir = directory / model
        _validate_private_directory(model_dir, f"{method}/{model} output directory")
        model_summary, model_summary_bytes = _load_private_object(
            model_dir / "final_summary.json", f"{method}/{model} final summary"
        )
        if root_models[model] != model_summary:
            raise RuntimeError(f"{method}/{model} root/model summary mismatch")
        for key, expected in {
            "schema_version": 2,
            "status": "COMPLETED",
            "method": method,
            "model": model,
            "privacy_claim": False,
            "run_config_fingerprint": run_fingerprint,
            "client_steps": rounds * clients,
        }.items():
            if model_summary.get(key) != expected:
                raise RuntimeError(f"{method}/{model} summary mismatch: {key}")
        behavior = model_summary.get("behavior_summary")
        if not isinstance(behavior, dict):
            raise RuntimeError(f"{method}/{model} behavior summary is missing")
        sample_schedule = _require_sha256(
            behavior.get("sample_schedule_sha256"),
            f"{method}/{model} sample schedule",
        )
        supervision_schedule = _require_sha256(
            behavior.get("supervision_schedule_sha256"),
            f"{method}/{model} supervision schedule",
        )
        if root_schedules.get(model) != sample_schedule:
            raise RuntimeError(f"{method}/{model} root sample schedule mismatch")
        partitions = model_summary.get("client_partition_sha256")
        if (
            not isinstance(partitions, list)
            or len(partitions) != clients
            or any(
                _require_sha256(value, f"{method}/{model} client partition") != value
                for value in partitions
            )
        ):
            raise RuntimeError(f"{method}/{model} client partitions are invalid")
        accounting = model_summary.get("privacy_accounting")
        if (
            not isinstance(accounting, dict)
            or accounting.get("status") != "NOT_CERTIFIED"
            or accounting.get("epsilon") is not None
            or accounting.get("sigma_is_not_epsilon") is not True
        ):
            raise RuntimeError(f"{method}/{model} privacy accounting is unsafe")

        adapter_dir = model_dir / "final_adapter"
        _validate_private_directory(adapter_dir, f"{method}/{model} adapter directory")
        adapter_bytes = _read_private_bytes(
            adapter_dir / "adapter_model.safetensors", f"{method}/{model} adapter"
        )
        adapter_config_bytes = _read_private_bytes(
            adapter_dir / "adapter_config.json", f"{method}/{model} adapter config"
        )
        adapter_sha = _sha256_bytes(adapter_bytes)
        adapter_config_sha = _sha256_bytes(adapter_config_bytes)
        if model_summary.get("adapter_sha256") != adapter_sha:
            raise RuntimeError(f"{method}/{model} adapter checksum mismatch")
        if model_summary.get("adapter_config_sha256") != adapter_config_sha:
            raise RuntimeError(f"{method}/{model} adapter-config checksum mismatch")
        integrity = model_summary.get("adapter_integrity")
        if not isinstance(integrity, dict) or integrity.get("all_finite") is not True:
            raise RuntimeError(f"{method}/{model} adapter integrity failed")
        model_records[model] = {
            "summary": model_summary,
            "summary_sha256": _sha256_bytes(model_summary_bytes),
            "sample_schedule_sha256": sample_schedule,
            "supervision_schedule_sha256": supervision_schedule,
            "client_partition_sha256": list(partitions),
            "adapter_sha256": adapter_sha,
            "adapter_config_sha256": adapter_config_sha,
            "evaluation": _evaluation_delta(model_summary, rounds),
        }
    return {
        "directory": directory,
        "run_config_sha256": _sha256_bytes(run_config_bytes),
        "root_summary_sha256": _sha256_bytes(root_summary_bytes),
        "contract": contract,
        "run_config_fingerprint": run_fingerprint,
        "effective_config": effective,
        "algorithm_contract": algorithm,
        "slaclip_contract": slaclip_contract,
        "privacy_claim": run_config["privacy_claim"],
        "rounds": rounds,
        "clients": clients,
        "models": model_records,
    }


def _normalized_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(contract))
    normalized.pop("method", None)
    effective = normalized.get("effective_config")
    algorithm = normalized.get("algorithm_contract")
    if not isinstance(effective, dict) or not isinstance(algorithm, dict):
        raise RuntimeError("scientific contract cannot be normalized")
    effective.pop("method", None)
    algorithm.pop("contains_slaclip", None)
    algorithm.pop("slaclip", None)
    algorithm.pop("slaclip_q", None)
    return normalized


def _validate_contract(
    adaptive: Mapping[str, Any], *, baseline_clip_norm: float
) -> dict[str, Any]:
    contract = adaptive["slaclip_contract"]
    assert isinstance(contract, dict)
    schema = contract.get("schema_version")
    if schema not in SLACLIP_VARIANT_BY_CONTRACT_SCHEMA:
        raise RuntimeError("full-SlaClip contract schema mismatch")
    variant = contract.get("variant")
    if variant != SLACLIP_VARIANT_BY_CONTRACT_SCHEMA[schema]:
        raise RuntimeError("full-SlaClip contract variant mismatch")
    groupwise = schema == GROUPWISE_SLACLIP_CONTRACT_SCHEMA
    forbidden_fields = {
        "calibration",
        "target_spec",
        "target_clip_fraction",
        "target_clip_fraction_by_group",
    }
    pending: list[Any] = [contract]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            forbidden = forbidden_fields.intersection(value)
            if forbidden:
                field = sorted(forbidden)[0]
                raise RuntimeError(
                    f"full SlaClip must not contain fixed-target field: {field}"
                )
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    if contract.get("independently_privacy_certified") is not False:
        raise RuntimeError("full-SlaClip contract makes an unsafe privacy claim")
    provenance = contract.get("hyperparameter_provenance")
    calibration_lock: str | None = None
    if groupwise:
        if not isinstance(provenance, dict):
            raise RuntimeError(
                "groupwise full-SlaClip hyperparameter provenance is missing"
            )
        calibration_lock = _require_sha256(
            provenance.get("calibration_lock_sha256"),
            "groupwise SlaClip calibration lock",
        )
        expected_provenance = {
            "source": "fixed_baseline_exact_endpoint_development_selection_lock",
            "baseline_derived_calibration_is_non_dp": True,
            "calibration_lock_sha256": calibration_lock,
            "calibration_data_consumed_at_controller_runtime": False,
            "frozen_scalar_hyperparameters_only": True,
        }
        if provenance != expected_provenance:
            raise RuntimeError(
                "groupwise full-SlaClip hyperparameter provenance mismatch"
            )
        privacy = adaptive.get("privacy_claim")
        if not isinstance(privacy, dict):
            raise RuntimeError("groupwise full-SlaClip privacy claim is missing")
        if (
            privacy.get("baseline_derived_calibration_is_non_dp") is not True
            or privacy.get("baseline_calibration_lock_sha256")
            != calibration_lock
            or privacy.get("baseline_calibration_consumed_at_runtime") is not False
        ):
            raise RuntimeError(
                "groupwise full-SlaClip calibration/privacy provenance mismatch"
            )
    elif provenance is not None:
        # Newly generated shared-v1 runs carry explicit direct-CLI provenance;
        # historical v1 artifacts did not.  Accept absence, but never accept a
        # baseline-derived lock under the shared schema.
        expected_provenance = {
            "source": "direct_cli_not_declared_baseline_derived",
            "baseline_derived_calibration_is_non_dp": False,
            "calibration_lock_sha256": None,
            "calibration_data_consumed_at_controller_runtime": False,
            "frozen_scalar_hyperparameters_only": True,
        }
        if provenance != expected_provenance:
            raise RuntimeError(
                "shared full-SlaClip hyperparameter provenance mismatch"
            )
    # Historical shared-beta contracts predate these explicit input labels, so
    # absence remains compatible.  If present, however, they must describe the
    # noisy-endpoint controller used by this comparator.  The new groupwise
    # schema requires the labels rather than silently defaulting them.
    if contract.get("controller_input", NOISY_CONTROLLER_INPUT) != (
        NOISY_CONTROLLER_INPUT
    ):
        raise RuntimeError("full-SlaClip controller input is not noisy endpoints")
    if contract.get("non_private_oracle_control", False) is not False:
        raise RuntimeError("full-SlaClip contract unexpectedly enables oracle control")
    if groupwise:
        for key, expected in {
            "controller_input": NOISY_CONTROLLER_INPUT,
            "non_private_oracle_control": False,
        }.items():
            if contract.get(key) != expected:
                raise RuntimeError(
                    f"groupwise full-SlaClip contract mismatch: {key}"
                )
    controller = contract.get("controller")
    if not isinstance(controller, dict):
        raise RuntimeError("full-SlaClip controller contract is missing")
    num_slots = _positive_integer(controller.get("num_slots"), "SlaClip slots")
    eta = _finite_number(controller.get("eta"), "SlaClip eta")
    if groupwise:
        if "beta" in controller or "base_target_clipped_fraction" in controller:
            raise RuntimeError(
                "groupwise full-SlaClip contract contains a shared target alias"
            )
        base_targets = _group_fraction_mapping(
            controller.get("base_target_clipped_fraction_by_group"),
            "SlaClip base target clipped fraction by group",
        )
        betas = _group_fraction_mapping(
            controller.get("beta_by_group"), "SlaClip beta by group"
        )
        if base_targets != betas:
            raise RuntimeError(
                "groupwise full-SlaClip beta/base-target mappings disagree"
            )
        expected_groupwise_contract = {
            "controller_input": NOISY_CONTROLLER_INPUT,
            "slack_noise_method_scope": ADAPTIVE_METHOD,
            "target_parameterization": "per_gradient_group",
            "generalized_full_slaclip_beta": True,
            "shared_beta_compatibility_alias_active": False,
            "base_target_clipped_fraction_source": "groupwise_canonical_cli",
        }
        for key, expected in expected_groupwise_contract.items():
            if controller.get(key) != expected:
                raise RuntimeError(
                    f"groupwise full-SlaClip controller contract mismatch: {key}"
                )
        expected_formula = (
            "for each g in {A,B}:q_g=s_hat_g[0];"
            "r_g=s_hat_g[K-1];z_g=r_g/(C_g+1e-6);"
            "remaining_g=1-z_g;"
            "target_clip_g=clip(beta_g*remaining_g,0,1);"
            "gamma_g=1-target_clip_g;"
            "C_g_next=clip(C_g*exp(eta*(gamma_g-q_g)),C_min,C_max)"
        )
        if controller.get("update_formula") != expected_formula:
            raise RuntimeError("groupwise full-SlaClip update formula mismatch")
    else:
        unexpected_groupwise_fields = {
            "base_target_clipped_fraction_by_group",
            "beta_by_group",
            "target_parameterization",
            "generalized_full_slaclip_beta",
            "shared_beta_compatibility_alias_active",
        }.intersection(controller)
        if unexpected_groupwise_fields:
            raise RuntimeError(
                "shared full-SlaClip contract contains groupwise target fields"
            )
        beta = _fraction(controller.get("beta"), "SlaClip beta")
        base_target = _fraction(
            controller.get("base_target_clipped_fraction"),
            "SlaClip base target clipped fraction",
        )
        if base_target != beta:
            raise RuntimeError("full-SlaClip beta/base-target aliases disagree")
        base_targets = {group: beta for group in LORA_GROUPS}
        betas = dict(base_targets)
        if controller.get("controller_input", NOISY_CONTROLLER_INPUT) != (
            NOISY_CONTROLLER_INPUT
        ):
            raise RuntimeError("full-SlaClip controller input is not noisy endpoints")
    lower = _finite_number(controller.get("c_min"), "SlaClip lower bound")
    upper = _finite_number(controller.get("c_max"), "SlaClip upper bound")
    epsilon = _finite_number(controller.get("epsilon"), "SlaClip endpoint epsilon")
    initial = _finite_number(
        controller.get("initial_clip_threshold"), "SlaClip initial threshold"
    )
    if eta < 0.0 or epsilon <= 0.0:
        raise RuntimeError("full-SlaClip controller parameters are invalid")
    if groupwise and epsilon != 1e-6:
        raise RuntimeError(
            "groupwise full-SlaClip epsilon/update-formula contract mismatch"
        )
    if lower <= 0.0 or upper < lower or not lower <= initial <= upper:
        raise RuntimeError("full-SlaClip threshold contract is invalid")
    adaptive_initial = _finite_number(
        adaptive["effective_config"].get("clip_norm"),
        "adaptive effective initial clip norm",
    )
    if initial != adaptive_initial:
        raise RuntimeError(
            "adaptive initial C differs between effective config and contract"
        )
    if not groupwise and initial != baseline_clip_norm:
        raise RuntimeError("adaptive initial C differs from baseline C")
    if controller.get("near_threshold_index") != 0:
        raise RuntimeError("full-SlaClip near-threshold endpoint index mismatch")
    if controller.get("near_zero_index") != num_slots - 1:
        raise RuntimeError("full-SlaClip near-zero endpoint index mismatch")
    if controller.get("numerical_log_step_bounds") != [
        -MAX_ABS_LOG_STEP,
        MAX_ABS_LOG_STEP,
    ]:
        raise RuntimeError("full-SlaClip numerical log-step bound mismatch")
    if controller.get("expected_release_records") != adaptive["clients"]:
        raise RuntimeError("full-SlaClip release count mismatch")
    effective = adaptive["effective_config"]
    local_batch_size = _positive_integer(
        effective.get("batch_size"), "SlaClip local batch size"
    )
    noise_multiplier = _finite_number(
        effective.get("noise_multiplier"), "SlaClip noise multiplier"
    )
    automatic_slots = _independent_automatic_num_slots(
        adaptive["clients"], noise_multiplier
    )
    theoretical_noise = noise_multiplier * math.sqrt(
        num_slots / adaptive["clients"]
    )
    expected_audit = {
        "local_batch_size": local_batch_size,
        "num_clients": adaptive["clients"],
        "automatic_release_num_slots": automatic_slots,
        "explicit_num_slots_exceeds_automatic_release_bound": bool(
            controller.get("num_slots_selection") != "automatic_monotonicity_rule"
            and num_slots > automatic_slots
        ),
        "normalized_proxy_noise_std_formula": (
            "noise_multiplier*sqrt(num_slots/num_clients)"
        ),
        "controller_inputs": "noisy_joint_release_endpoints_only",
    }
    for key, expected in expected_audit.items():
        if controller.get(key) != expected:
            raise RuntimeError(f"full-SlaClip K/noise audit mismatch: {key}")
    recorded_noise = _finite_number(
        controller.get("normalized_proxy_noise_std_per_slot_theoretical"),
        "SlaClip theoretical endpoint noise",
    )
    if not math.isclose(
        recorded_noise, theoretical_noise, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RuntimeError("full-SlaClip K/noise audit mismatch: theoretical noise")
    return {
        "schema_version": schema,
        "variant": variant,
        "groupwise": groupwise,
        "num_slots": num_slots,
        "eta": eta,
        "beta_by_group": betas,
        "base_target_clipped_fraction_by_group": base_targets,
        "calibration_lock_sha256": calibration_lock,
        "min_clip_norm": lower,
        "max_clip_norm": upper,
        "epsilon": epsilon,
        "initial_clip_norm": initial,
        "theoretical_endpoint_noise_std": theoretical_noise,
    }


def _finite_vector(value: Any, length: int, description: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise RuntimeError(f"{description} must contain exactly {length} values")
    return [_finite_number(item, description) for item in value]


def _independent_normalize_noisy_slack(
    summed_slack: Sequence[Any],
    clip_norm: Any,
    num_slots: int,
    expected_records: Any,
) -> tuple[float, ...]:
    """Normalize the joint slack release using an independent specification."""

    threshold = _finite_number(clip_norm, "SlaClip normalization threshold")
    records = _finite_number(expected_records, "SlaClip normalization records")
    slots = _positive_integer(num_slots, "SlaClip normalization slots")
    if threshold <= 0.0 or records <= 0.0:
        raise RuntimeError("SlaClip normalization inputs must be positive")
    if len(summed_slack) != slots:
        raise RuntimeError(
            f"SlaClip summed slack must contain exactly {slots} values"
        )
    values = [
        _finite_number(value, f"SlaClip summed slack[{index}]")
        for index, value in enumerate(summed_slack)
    ]
    denominator = (threshold / math.sqrt(slots)) * records
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("SlaClip normalization denominator is not finite")
    normalized = tuple(value / denominator for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise RuntimeError("SlaClip normalized CDF proxy is not finite")
    return normalized


def _independent_full_slaclip_update(
    current_clip_norm: Any,
    noisy_cdf_near_threshold: Any,
    noisy_cdf_near_zero: Any,
    *,
    beta: Any,
    eta: Any,
    min_clip_norm: Any,
    max_clip_norm: Any,
    epsilon: Any,
) -> dict[str, Any]:
    """Recompute the endpoint controller without importing production code."""

    current = _finite_number(current_clip_norm, "SlaClip current threshold")
    near_threshold = _finite_number(
        noisy_cdf_near_threshold, "SlaClip near-threshold endpoint"
    )
    near_zero = _finite_number(
        noisy_cdf_near_zero, "SlaClip near-zero endpoint"
    )
    balance = _finite_number(beta, "SlaClip beta")
    gain = _finite_number(eta, "SlaClip eta")
    lower = _finite_number(min_clip_norm, "SlaClip lower threshold")
    upper = _finite_number(max_clip_norm, "SlaClip upper threshold")
    denominator_epsilon = _finite_number(epsilon, "SlaClip endpoint epsilon")
    if current <= 0.0 or lower <= 0.0 or denominator_epsilon <= 0.0:
        raise RuntimeError("SlaClip positive controller inputs are invalid")
    if not 0.0 <= balance <= 1.0 or gain < 0.0:
        raise RuntimeError("SlaClip beta/eta inputs are invalid")
    if upper < lower or not lower <= current <= upper:
        raise RuntimeError("SlaClip threshold bounds are invalid")

    endpoint_denominator = current + denominator_epsilon
    if not math.isfinite(endpoint_denominator) or endpoint_denominator <= 0.0:
        raise RuntimeError("SlaClip endpoint denominator is not finite")
    near_zero_adjusted = near_zero / endpoint_denominator
    remaining_non_small = 1.0 - near_zero_adjusted
    raw_target_clipped = balance * remaining_non_small
    if not math.isfinite(raw_target_clipped):
        raise RuntimeError("SlaClip dynamic target is not finite")
    target_clipped = max(0.0, min(1.0, raw_target_clipped))
    raw_target = 1.0 - raw_target_clipped
    target_unclipped = 1.0 - target_clipped
    controller_error = target_unclipped - near_threshold
    raw_log_step = gain * controller_error
    if not math.isfinite(raw_log_step):
        raise RuntimeError("SlaClip log update is not finite")
    bounded_log_step = max(
        -MAX_ABS_LOG_STEP, min(MAX_ABS_LOG_STEP, raw_log_step)
    )
    unbounded_next = current * math.exp(bounded_log_step)
    if not math.isfinite(unbounded_next) or unbounded_next <= 0.0:
        raise RuntimeError("SlaClip candidate threshold is not finite")
    next_threshold = max(lower, min(upper, unbounded_next))

    return {
        "current_clip_norm": float(current),
        "near_threshold_proxy": float(near_threshold),
        "near_zero_proxy": float(near_zero),
        "base_target_clipped_fraction": float(balance),
        "beta": float(balance),
        "epsilon": float(denominator_epsilon),
        "endpoint_denominator": float(endpoint_denominator),
        "near_zero_adjusted": float(near_zero_adjusted),
        "remaining_non_small_gradient_fraction": float(remaining_non_small),
        "raw_dynamic_target_clipped": float(raw_target_clipped),
        "clamped_dynamic_target_clipped": float(target_clipped),
        "raw_dynamic_target_unclipped": float(raw_target),
        "dynamic_target_unclipped": float(target_unclipped),
        "dynamic_target_clipped": float(target_clipped),
        "gamma_clamped_low": bool(raw_target < 0.0),
        "gamma_clamped_high": bool(raw_target > 1.0),
        "controller_error": float(controller_error),
        "eta": float(gain),
        "raw_log_step": float(raw_log_step),
        "bounded_log_step": float(bounded_log_step),
        "log_step_was_bounded": bool(raw_log_step != bounded_log_step),
        "unbounded_next_clip_norm": float(unbounded_next),
        "next_clip_norm": float(next_threshold),
        "min_clip_norm": float(lower),
        "max_clip_norm": float(upper),
        "hit_min_clip_norm": bool(unbounded_next < lower),
        "hit_max_clip_norm": bool(unbounded_next > upper),
    }


def _validate_exact_cdf_range(values: Sequence[float], description: str) -> None:
    """Accept only the tiny endpoint overshoot caused by float32 slack storage."""

    if any(
        item < -EXACT_CDF_FLOAT32_TOLERANCE
        or item > 1.0 + EXACT_CDF_FLOAT32_TOLERANCE
        for item in values
    ):
        raise RuntimeError(f"{description} must lie within float32 tolerance of [0, 1]")


def _reconstruct_round_release(
    client_records: Sequence[Mapping[str, Any]],
    *,
    group: str,
    threshold: float,
    num_slots: int,
    description: str,
) -> dict[str, Any]:
    """Rebuild both CDF proxies directly from the protected client releases."""

    noisy_releases: list[list[float]] = []
    signal_releases: list[list[float]] = []
    clipped: list[bool] = []
    slack_noise_stds: list[float] = []
    for client_index, record in enumerate(client_records):
        gradient_groups = record.get("gradient_groups")
        statistics = (
            gradient_groups.get(group)
            if isinstance(gradient_groups, dict)
            else None
        )
        if not isinstance(statistics, dict):
            raise RuntimeError(
                f"{description} client {client_index} gradient group is missing"
            )
        if statistics.get("clip_threshold") != threshold:
            raise RuntimeError(
                f"{description} client {client_index} threshold mismatch"
            )
        telemetry = statistics.get("slaclip")
        if (
            not isinstance(telemetry, dict)
            or telemetry.get("variant") != SLACLIP_VARIANT
            or telemetry.get("num_slots") != num_slots
            or telemetry.get("joint_sensitivity_bound_passed") is not True
        ):
            raise RuntimeError(
                f"{description} client {client_index} joint release is invalid"
            )
        noisy_releases.append(
            _finite_vector(
                telemetry.get("noisy_slack"),
                num_slots,
                f"{description} client {client_index} noisy slack",
            )
        )
        signal_releases.append(
            _finite_vector(
                telemetry.get("slack_signal"),
                num_slots,
                f"{description} client {client_index} slack signal",
            )
        )
        slack_noise_stds.append(
            _finite_number(
                telemetry.get("slack_noise_std_per_coordinate"),
                f"{description} client {client_index} slack noise std",
            )
        )
        if not isinstance(statistics.get("clipped"), bool):
            raise RuntimeError(
                f"{description} client {client_index} clipping flag is invalid"
            )
        clipped.append(bool(statistics["clipped"]))

    client_count = len(client_records)
    noisy_sum = [
        math.fsum(release[slot] for release in noisy_releases)
        for slot in range(num_slots)
    ]
    signal_sum = [
        math.fsum(release[slot] for release in signal_releases)
        for slot in range(num_slots)
    ]
    noisy = list(
        _independent_normalize_noisy_slack(
            noisy_sum, threshold, num_slots, client_count
        )
    )
    exact = list(
        _independent_normalize_noisy_slack(
            signal_sum, threshold, num_slots, client_count
        )
    )
    if any(value < 0.0 for value in slack_noise_stds) or len(
        set(slack_noise_stds)
    ) != 1:
        raise RuntimeError(f"{description} client slack noise scales disagree")
    slack_lambda = threshold / math.sqrt(num_slots)
    normalized_noise = (
        slack_noise_stds[0]
        * math.sqrt(client_count)
        / (slack_lambda * client_count)
    )
    return {
        "noisy": noisy,
        "exact": exact,
        "actual_clip_fraction": sum(clipped) / client_count,
        "normalized_proxy_noise_std_per_slot": normalized_noise,
    }


def _runner_update_fields(recomputed: Mapping[str, Any]) -> dict[str, Any]:
    expected = dict(recomputed)
    expected["unbounded_next_clip_threshold"] = expected.pop(
        "unbounded_next_clip_norm"
    )
    expected["next_clip_threshold"] = expected.pop("next_clip_norm")
    expected["c_min"] = expected.pop("min_clip_norm")
    expected["c_max"] = expected.pop("max_clip_norm")
    expected.pop("current_clip_norm", None)
    return expected


def _validate_update_fields(
    recorded: Mapping[str, Any], recomputed: Mapping[str, Any], description: str
) -> dict[str, Any]:
    expected_fields = _runner_update_fields(recomputed)
    for key, expected in expected_fields.items():
        if recorded.get(key) != expected:
            raise RuntimeError(f"{description} full-SlaClip formula mismatch: {key}")
    return expected_fields


def _validate_controller_trajectory(
    adaptive: Mapping[str, Any], *, model: str, baseline_clip_norm: float
) -> dict[str, Any]:
    parameters = _validate_contract(adaptive, baseline_clip_norm=baseline_clip_norm)
    num_slots = parameters["num_slots"]
    rounds_dir = adaptive["directory"] / model / "private_diagnostics" / "rounds"
    _validate_private_directory(rounds_dir, f"adaptive/{model} rounds directory")
    expected_thresholds = {
        group: parameters["initial_clip_norm"] for group in LORA_GROUPS
    }
    trajectory: list[dict[str, Any]] = []
    accumulators = {
        group: {
            "actual_clip_fractions": [],
            "dynamic_targets": [],
            "validated_round_telemetry": [],
            "gamma_clamped_rounds": 0,
            "threshold_bound_hit_rounds": 0,
        }
        for group in LORA_GROUPS
    }
    for round_index in range(1, adaptive["rounds"] + 1):
        shard, _ = _load_private_object(
            rounds_dir / f"round-{round_index:05d}.json",
            f"adaptive/{model} round shard",
        )
        for key, expected in {
            "schema_version": 2,
            "method": ADAPTIVE_METHOD,
            "model": model,
            "round": round_index,
        }.items():
            if shard.get(key) != expected:
                raise RuntimeError(
                    f"adaptive/{model} round shard identity mismatch: {key}"
                )
        round_summary = shard.get("round_summary")
        if not isinstance(round_summary, dict) or round_summary.get("round") != round_index:
            raise RuntimeError(f"adaptive/{model} round summary is invalid")
        client_records = shard.get("client_records")
        if (
            not isinstance(client_records, list)
            or len(client_records) != adaptive["clients"]
            or any(not isinstance(record, dict) for record in client_records)
            or [record.get("client") for record in client_records]
            != list(range(adaptive["clients"]))
        ):
            raise RuntimeError(
                f"adaptive/{model} protected client releases are missing or reordered"
            )
        value = round_summary.get("slaclip_controller")
        if not isinstance(value, dict):
            raise RuntimeError(f"adaptive/{model} controller round is missing")
        expected_header = {
            "variant": parameters["variant"],
            "update_timing": "once_after_all_clients_for_use_in_next_round",
            "clients": adaptive["clients"],
            "num_slots": parameters["num_slots"],
            "eta": parameters["eta"],
            "epsilon": parameters["epsilon"],
            "near_threshold_index": 0,
            "near_zero_index": parameters["num_slots"] - 1,
            "c_min": parameters["min_clip_norm"],
            "c_max": parameters["max_clip_norm"],
        }
        if parameters["groupwise"]:
            if "beta" in value or "base_target_clipped_fraction" in value:
                raise RuntimeError(
                    f"adaptive/{model} groupwise controller has a shared target"
                )
            round_targets = _group_fraction_mapping(
                value.get("base_target_clipped_fraction_by_group"),
                f"adaptive/{model} round base target by group",
            )
            round_betas = _group_fraction_mapping(
                value.get("beta_by_group"),
                f"adaptive/{model} round beta by group",
            )
            if (
                round_targets
                != parameters["base_target_clipped_fraction_by_group"]
                or round_betas != parameters["beta_by_group"]
                or round_targets != round_betas
            ):
                raise RuntimeError(
                    f"adaptive/{model} groupwise controller target mapping mismatch"
                )
            expected_header.update(
                {
                    "controller_input": NOISY_CONTROLLER_INPUT,
                    "controller_input_is_non_dp_exact": False,
                    "target_parameterization": "per_gradient_group",
                    "generalized_full_slaclip_beta": True,
                    "base_target_clipped_fraction_by_group": round_targets,
                    "beta_by_group": round_betas,
                }
            )
        else:
            if any(
                key in value
                for key in (
                    "base_target_clipped_fraction_by_group",
                    "beta_by_group",
                    "target_parameterization",
                    "generalized_full_slaclip_beta",
                )
            ):
                raise RuntimeError(
                    f"adaptive/{model} shared controller has groupwise targets"
                )
            expected_header.update(
                {
                    "base_target_clipped_fraction": parameters[
                        "base_target_clipped_fraction_by_group"
                    ]["A"],
                    "beta": parameters["beta_by_group"]["A"],
                }
            )
        for key, expected in expected_header.items():
            if value.get(key) != expected:
                raise RuntimeError(
                    f"adaptive/{model} controller header mismatch: {key}"
                )

        for group in LORA_GROUPS:
            group_value = value.get(group)
            if not isinstance(group_value, dict):
                raise RuntimeError(f"adaptive/{model}/{group} controller is missing")
            if parameters["groupwise"]:
                for key, expected in {
                    "controller_input": NOISY_CONTROLLER_INPUT,
                    "controller_input_is_non_dp_exact": False,
                }.items():
                    if group_value.get(key) != expected:
                        raise RuntimeError(
                            f"adaptive/{model}/{group} controller input mismatch: {key}"
                        )
            used = _finite_number(
                group_value.get("clip_threshold_used"),
                f"adaptive/{model}/{group} threshold",
            )
            if used != expected_thresholds[group]:
                raise RuntimeError(
                    f"adaptive/{model}/{group} threshold trajectory is discontinuous"
                )
            noisy = _finite_vector(
                group_value.get("noisy_cdf_proxy_by_slot"),
                num_slots,
                f"adaptive/{model}/{group} noisy CDF",
            )
            exact = _finite_vector(
                group_value.get("exact_cdf_proxy_by_slot"),
                num_slots,
                f"adaptive/{model}/{group} exact CDF",
            )
            reconstructed = _reconstruct_round_release(
                client_records,
                group=group,
                threshold=used,
                num_slots=num_slots,
                description=f"adaptive/{model}/{group}",
            )
            if noisy != reconstructed["noisy"]:
                raise RuntimeError(
                    f"adaptive/{model}/{group} noisy CDF does not reconcile "
                    "with client slack releases"
                )
            if exact != reconstructed["exact"]:
                raise RuntimeError(
                    f"adaptive/{model}/{group} exact CDF does not reconcile "
                    "with client slack signals"
                )
            _validate_exact_cdf_range(
                exact,
                f"adaptive/{model}/{group} exact CDF",
            )
            actual = _fraction(
                group_value.get("actual_clip_fraction"),
                f"adaptive/{model}/{group} actual clip fraction",
            )
            if actual != reconstructed["actual_clip_fraction"]:
                raise RuntimeError(
                    f"adaptive/{model}/{group} clipping fraction does not "
                    "reconcile with client releases"
                )
            recomputed = _independent_full_slaclip_update(
                used,
                noisy[0],
                noisy[-1],
                beta=parameters["beta_by_group"][group],
                eta=parameters["eta"],
                min_clip_norm=parameters["min_clip_norm"],
                max_clip_norm=parameters["max_clip_norm"],
                epsilon=parameters["epsilon"],
            )
            recorded_update = _validate_update_fields(
                group_value,
                recomputed,
                f"adaptive/{model}/{group}",
            )
            oracle = _independent_full_slaclip_update(
                used,
                exact[0],
                exact[-1],
                beta=parameters["beta_by_group"][group],
                eta=parameters["eta"],
                min_clip_norm=parameters["min_clip_norm"],
                max_clip_norm=parameters["max_clip_norm"],
                epsilon=parameters["epsilon"],
            )
            normalized_noise = _finite_number(
                group_value.get("normalized_proxy_noise_std_per_slot"),
                f"adaptive/{model}/{group} normalized endpoint noise",
            )
            if not math.isclose(
                normalized_noise,
                reconstructed["normalized_proxy_noise_std_per_slot"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"adaptive/{model}/{group} normalized endpoint noise does "
                    "not reconcile with client releases"
                )
            if normalized_noise < 0.0:
                raise RuntimeError(
                    f"adaptive/{model}/{group} endpoint noise must be non-negative"
                )
            if not math.isclose(
                normalized_noise,
                parameters["theoretical_endpoint_noise_std"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"adaptive/{model}/{group} endpoint noise audit mismatch"
                )

            cdf_errors = [
                noisy_value - exact_value
                for noisy_value, exact_value in zip(noisy, exact)
            ]
            cdf_error_mae = math.fsum(
                abs(error) for error in cdf_errors
            ) / num_slots
            cdf_error_rmse = math.sqrt(
                math.fsum(error * error for error in cdf_errors) / num_slots
            )
            noisy_out_of_range_count = sum(
                not 0.0 <= value <= 1.0 for value in noisy
            )
            noisy_log_step = float(recomputed["raw_log_step"])
            oracle_log_step = float(oracle["raw_log_step"])

            def update_direction(value: float) -> int:
                return -1 if value < 0.0 else (1 if value > 0.0 else 0)

            expected_endpoint_telemetry = {
                "cdf_error_mae": cdf_error_mae,
                "cdf_error_rmse": cdf_error_rmse,
                "cdf_error_max_abs": max(abs(error) for error in cdf_errors),
                "cdf_error_z_rmse": (
                    cdf_error_rmse / normalized_noise
                    if normalized_noise > 0.0
                    else None
                ),
                "noisy_cdf_out_of_range_count": noisy_out_of_range_count,
                "noisy_cdf_out_of_range_fraction": (
                    noisy_out_of_range_count / num_slots
                ),
                "noisy_near_threshold_minus_exact": noisy[0] - exact[0],
                "noisy_near_zero_minus_exact": noisy[-1] - exact[-1],
                "noisy_adjacent_monotonicity_violations": sum(
                    noisy[index + 1] > noisy[index]
                    for index in range(num_slots - 1)
                ),
                "exact_adjacent_monotonicity_violations": sum(
                    exact[index + 1] > exact[index]
                    for index in range(num_slots - 1)
                ),
                "actual_minus_dynamic_target_clipped": (
                    actual - float(recorded_update["dynamic_target_clipped"])
                ),
                "actual_target_absolute_error": abs(
                    actual - float(recorded_update["dynamic_target_clipped"])
                ),
                "oracle_dynamic_target_clipped": float(
                    oracle["dynamic_target_clipped"]
                ),
                "oracle_raw_log_step": oracle_log_step,
                "oracle_next_clip_threshold": float(oracle["next_clip_norm"]),
                "noisy_minus_oracle_raw_log_step": (
                    noisy_log_step - oracle_log_step
                ),
                "noisy_oracle_log_threshold_error": math.log(
                    float(recomputed["next_clip_norm"])
                    / float(oracle["next_clip_norm"])
                ),
                "update_direction_agrees": (
                    update_direction(noisy_log_step)
                    == update_direction(oracle_log_step)
                ),
            }
            if parameters["groupwise"]:
                expected_endpoint_telemetry.update(
                    {
                        "noisy_dynamic_target_clipped": float(
                            recomputed["dynamic_target_clipped"]
                        ),
                        "noisy_raw_log_step": noisy_log_step,
                        "noisy_next_clip_threshold": float(
                            recomputed["next_clip_norm"]
                        ),
                    }
                )
            for key, expected in expected_endpoint_telemetry.items():
                if key not in group_value or group_value[key] != expected:
                    raise RuntimeError(
                        f"adaptive/{model}/{group} endpoint telemetry mismatch: {key}"
                    )
            expected_thresholds[group] = _finite_number(
                recorded_update["next_clip_threshold"],
                f"adaptive/{model}/{group} next threshold",
            )
            accumulator = accumulators[group]
            accumulator["actual_clip_fractions"].append(actual)
            accumulator["dynamic_targets"].append(
                _fraction(
                    recomputed["dynamic_target_clipped"],
                    f"adaptive/{model}/{group} dynamic clipped target",
                )
            )
            accumulator["validated_round_telemetry"].append(
                {
                    **expected_endpoint_telemetry,
                    "clip_threshold_used": used,
                    "next_clip_threshold": float(recomputed["next_clip_norm"]),
                    "raw_log_step": noisy_log_step,
                }
            )
            if recomputed["gamma_clamped_low"] or recomputed["gamma_clamped_high"]:
                accumulator["gamma_clamped_rounds"] += 1
            if recomputed["hit_min_clip_norm"] or recomputed["hit_max_clip_norm"]:
                accumulator["threshold_bound_hit_rounds"] += 1
        trajectory.append(value)

    model_summary = adaptive["models"][model]["summary"]
    prefix = _round_shard_prefix_sha256(rounds_dir, adaptive["rounds"])
    if model_summary.get("round_shard_prefix_sha256") != prefix:
        raise RuntimeError(f"adaptive/{model} round-prefix mismatch")
    behavior = model_summary["behavior_summary"]
    controller_summary = behavior.get("slaclip_controller")
    if not isinstance(controller_summary, dict):
        raise RuntimeError(f"adaptive/{model} controller summary is missing")
    if (
        controller_summary.get("variant") != parameters["variant"]
        or controller_summary.get("rounds") != adaptive["rounds"]
        or controller_summary.get("trajectory_sha256")
        != canonical_json_fingerprint(trajectory)
    ):
        raise RuntimeError(f"adaptive/{model} controller trajectory digest mismatch")
    if parameters["groupwise"]:
        if "beta" in controller_summary or (
            "base_target_clipped_fraction" in controller_summary
        ):
            raise RuntimeError(
                f"adaptive/{model} groupwise controller summary has a shared target"
            )
        summary_targets = _group_fraction_mapping(
            controller_summary.get("base_target_clipped_fraction_by_group"),
            f"adaptive/{model} summary base target by group",
        )
        summary_betas = _group_fraction_mapping(
            controller_summary.get("beta_by_group"),
            f"adaptive/{model} summary beta by group",
        )
        expected_summary_header = {
            "controller_input": NOISY_CONTROLLER_INPUT,
            "controller_input_is_non_dp_exact": False,
            "target_parameterization": "per_gradient_group",
            "generalized_full_slaclip_beta": True,
            "base_target_clipped_fraction_by_group": parameters[
                "base_target_clipped_fraction_by_group"
            ],
            "beta_by_group": parameters["beta_by_group"],
        }
        if summary_targets != summary_betas:
            raise RuntimeError(
                f"adaptive/{model} controller summary target aliases disagree"
            )
        for key, expected in expected_summary_header.items():
            if controller_summary.get(key) != expected:
                raise RuntimeError(
                    f"adaptive/{model} controller summary header mismatch: {key}"
                )
    else:
        if any(
            key in controller_summary
            for key in (
                "base_target_clipped_fraction_by_group",
                "beta_by_group",
                "target_parameterization",
                "generalized_full_slaclip_beta",
            )
        ):
            raise RuntimeError(
                f"adaptive/{model} shared controller summary has groupwise targets"
            )
        # These scalar aliases were added after the v1 schema was deployed.
        # Keep old summaries valid, while fail-closing if either alias appears
        # with the wrong value or without its companion.
        scalar_summary_aliases = {
            key for key in ("base_target_clipped_fraction", "beta")
            if key in controller_summary
        }
        if scalar_summary_aliases and scalar_summary_aliases != {
            "base_target_clipped_fraction",
            "beta",
        }:
            raise RuntimeError(
                f"adaptive/{model} controller summary scalar aliases are incomplete"
            )
        if scalar_summary_aliases:
            expected_beta = parameters["beta_by_group"]["A"]
            if (
                controller_summary["base_target_clipped_fraction"]
                != expected_beta
                or controller_summary["beta"] != expected_beta
            ):
                raise RuntimeError(
                    f"adaptive/{model} controller summary scalar aliases disagree"
                )
    model_contract = model_summary.get("slaclip")
    if not isinstance(model_contract, dict):
        raise RuntimeError(f"adaptive/{model} full-SlaClip model contract is missing")
    if model_contract.get("contract") != adaptive["slaclip_contract"]:
        raise RuntimeError(f"adaptive/{model} full-SlaClip contract mismatch")
    if model_contract.get("controller_summary") != controller_summary:
        raise RuntimeError(f"adaptive/{model} controller-summary binding mismatch")
    if (
        model_contract.get("final_next_clip_threshold_by_group")
        != expected_thresholds
    ):
        raise RuntimeError(f"adaptive/{model} final threshold map mismatch")
    if "slaclip_q" in model_summary:
        raise RuntimeError(f"adaptive/{model} unexpectedly contains legacy SlaClip-Q")

    summary_groups = controller_summary.get("groups")
    if (
        not isinstance(summary_groups, dict)
        or tuple(summary_groups) != LORA_GROUPS
    ):
        raise RuntimeError(f"adaptive/{model} controller summary groups are invalid")
    quantile_telemetry = (
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
    for group in LORA_GROUPS:
        recorded_summary = summary_groups[group]
        if not isinstance(recorded_summary, dict):
            raise RuntimeError(
                f"adaptive/{model}/{group} controller summary is invalid"
            )
        validated_rounds = accumulators[group]["validated_round_telemetry"]
        raw_steps = [float(value["raw_log_step"]) for value in validated_rounds]
        nonzero_directions = [
            -1 if value < 0.0 else 1 for value in raw_steps if value != 0.0
        ]
        agreement_count = sum(
            bool(value["update_direction_agrees"]) for value in validated_rounds
        )
        out_of_range_count = sum(
            int(value["noisy_cdf_out_of_range_count"])
            for value in validated_rounds
        )
        expected_summary_telemetry: dict[str, Any] = {
            key: safe_quantiles(value[key] for value in validated_rounds)
            for key in quantile_telemetry
        }
        expected_summary_telemetry.update(
            {
                "log_threshold_total_variation": math.fsum(
                    abs(
                        math.log(
                            float(value["next_clip_threshold"])
                            / float(value["clip_threshold_used"])
                        )
                    )
                    for value in validated_rounds
                ),
                "log_step_direction_flip_count": sum(
                    left != right
                    for left, right in zip(
                        nonzero_directions, nonzero_directions[1:]
                    )
                ),
                "oracle_direction_agreement_count": agreement_count,
                "oracle_direction_agreement_fraction": (
                    agreement_count / len(validated_rounds)
                ),
                "noisy_cdf_out_of_range_count": out_of_range_count,
                "noisy_cdf_out_of_range_fraction": (
                    out_of_range_count / (len(validated_rounds) * num_slots)
                ),
                "final_next_clip_threshold": expected_thresholds[group],
            }
        )
        for key, expected in expected_summary_telemetry.items():
            if key not in recorded_summary or recorded_summary[key] != expected:
                raise RuntimeError(
                    f"adaptive/{model}/{group} controller summary telemetry "
                    f"mismatch: {key}"
                )

    group_results: dict[str, Any] = {}
    for group in LORA_GROUPS:
        accumulator = accumulators[group]
        actual_values = accumulator.pop("actual_clip_fractions")
        target_values = accumulator.pop("dynamic_targets")
        telemetry = accumulator.pop("validated_round_telemetry")
        group_results[group] = {
            "final_clip_norm": expected_thresholds[group],
            "mean_actual_clip_fraction": sum(actual_values) / len(actual_values),
            "mean_dynamic_target_clip_fraction": sum(target_values)
            / len(target_values),
            "mean_actual_target_absolute_error": math.fsum(
                value["actual_target_absolute_error"] for value in telemetry
            )
            / len(telemetry),
            "mean_cdf_error_mae": math.fsum(
                value["cdf_error_mae"] for value in telemetry
            )
            / len(telemetry),
            "mean_cdf_error_rmse": math.fsum(
                value["cdf_error_rmse"] for value in telemetry
            )
            / len(telemetry),
            "oracle_direction_agreement_fraction": (
                sum(bool(value["update_direction_agrees"]) for value in telemetry)
                / len(telemetry)
            ),
            "noisy_cdf_out_of_range_fraction": (
                sum(
                    int(value["noisy_cdf_out_of_range_count"])
                    for value in telemetry
                )
                / (len(telemetry) * num_slots)
            ),
            **accumulator,
        }
    result = {
        "rounds": adaptive["rounds"],
        "trajectory_sha256": controller_summary["trajectory_sha256"],
        "initial_clip_norm": parameters["initial_clip_norm"],
        "final_clip_norm_by_group": dict(expected_thresholds),
        "num_slots": parameters["num_slots"],
        "eta": parameters["eta"],
        "epsilon": parameters["epsilon"],
        "near_threshold_index": 0,
        "near_zero_index": parameters["num_slots"] - 1,
        "groups": group_results,
    }
    if parameters["groupwise"]:
        result.update(
            {
                "target_parameterization": "per_gradient_group",
                "generalized_full_slaclip_beta": True,
                "base_target_clipped_fraction_by_group": dict(
                    parameters["base_target_clipped_fraction_by_group"]
                ),
                "beta_by_group": dict(parameters["beta_by_group"]),
            }
        )
    else:
        result.update(
            {
                "base_target_clipped_fraction": parameters["beta_by_group"]["A"],
                "beta": parameters["beta_by_group"]["A"],
            }
        )
    return result


def build_comparison(
    baseline_dir: str | os.PathLike[str],
    adaptive_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate both arms and return a fingerprinted comparison core."""

    baseline_path = _absolute_path(baseline_dir)
    adaptive_path = _absolute_path(adaptive_dir)
    baseline = _load_arm(baseline_path, BASELINE_METHOD, adaptive=False)
    adaptive = _load_arm(adaptive_path, ADAPTIVE_METHOD, adaptive=True)
    slaclip_contract = adaptive["slaclip_contract"]
    assert isinstance(slaclip_contract, dict)
    slaclip_schema = slaclip_contract.get("schema_version")
    slaclip_variant = slaclip_contract.get("variant")
    groupwise = slaclip_schema == GROUPWISE_SLACLIP_CONTRACT_SCHEMA
    normalized_baseline = _normalized_contract(baseline["contract"])
    normalized_adaptive = _normalized_contract(adaptive["contract"])
    if groupwise:
        # The generalized development protocol compares a tuned fixed-C arm
        # with an independently selected adaptive C0.  Remove only this one
        # explicitly reported hyperparameter before enforcing all other paired
        # scientific-contract fields.
        for normalized in (normalized_baseline, normalized_adaptive):
            effective = normalized.get("effective_config")
            if not isinstance(effective, dict):
                raise RuntimeError("normalized effective config is missing")
            effective.pop("clip_norm", None)
    if normalized_baseline != normalized_adaptive:
        raise RuntimeError("baseline and adaptive arms differ outside full SlaClip")
    normalized_fingerprint = canonical_json_fingerprint(normalized_baseline)
    baseline_clip_norm = _finite_number(
        baseline["effective_config"].get("clip_norm"), "baseline clip norm"
    )
    adaptive_initial_clip_norm = _finite_number(
        adaptive["effective_config"].get("clip_norm"),
        "adaptive initial clip norm",
    )
    if not groupwise and adaptive_initial_clip_norm != baseline_clip_norm:
        raise RuntimeError("adaptive initial C differs from baseline C")

    model_comparisons: dict[str, Any] = {}
    for model in DEFAULT_MODELS:
        baseline_model = baseline["models"][model]
        adaptive_model = adaptive["models"][model]
        for field, label in (
            ("sample_schedule_sha256", "sample schedules"),
            ("supervision_schedule_sha256", "supervision schedules"),
            ("client_partition_sha256", "client partitions"),
        ):
            if baseline_model[field] != adaptive_model[field]:
                raise RuntimeError(f"{model} {label} do not match")
        baseline_summary = baseline_model["summary"]
        if (
            "slaclip" in baseline_summary
            or "slaclip_q" in baseline_summary
            or "slaclip_controller" in baseline_summary.get("behavior_summary", {})
        ):
            raise RuntimeError(f"baseline/{model} unexpectedly contains SlaClip")
        controller = _validate_controller_trajectory(
            adaptive,
            model=model,
            baseline_clip_norm=baseline_clip_norm,
        )
        baseline_evaluation = baseline_model["evaluation"]
        adaptive_evaluation = adaptive_model["evaluation"]
        if baseline_evaluation["initial"] != adaptive_evaluation["initial"]:
            raise RuntimeError(f"{model} initial evaluations do not match")
        paired_evaluation = {
            "metric": baseline_evaluation["metric"],
            "baseline_initial": baseline_evaluation["initial"],
            "baseline_final": baseline_evaluation["final"],
            "adaptive_initial": adaptive_evaluation["initial"],
            "adaptive_final": adaptive_evaluation["final"],
            "adaptive_minus_baseline_final": (
                adaptive_evaluation["final"] - baseline_evaluation["final"]
            ),
            "adaptive_minus_baseline_change": (
                adaptive_evaluation["final_minus_initial"]
                - baseline_evaluation["final_minus_initial"]
            ),
            "adaptive_has_lower_final_loss": (
                adaptive_evaluation["final"] < baseline_evaluation["final"]
            ),
        }
        model_comparisons[model] = {
            "matched_sample_schedule_sha256": baseline_model[
                "sample_schedule_sha256"
            ],
            "matched_supervision_schedule_sha256": baseline_model[
                "supervision_schedule_sha256"
            ],
            "matched_client_partition_sha256": baseline_model[
                "client_partition_sha256"
            ],
            "controller": controller,
            "paired_internal_holdout": paired_evaluation,
            "arms": {
                BASELINE_METHOD: {
                    "adapter_sha256": baseline_model["adapter_sha256"],
                    "adapter_config_sha256": baseline_model[
                        "adapter_config_sha256"
                    ],
                    "clipping": baseline_summary.get("clipping"),
                    "internal_holdout": baseline_model["evaluation"],
                },
                ADAPTIVE_METHOD: {
                    "adapter_sha256": adaptive_model["adapter_sha256"],
                    "adapter_config_sha256": adaptive_model[
                        "adapter_config_sha256"
                    ],
                    "clipping": adaptive_model["summary"].get("clipping"),
                    "internal_holdout": adaptive_model["evaluation"],
                },
            },
        }

    core = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": COMPARISON_STATUS,
        "claim_level": 1,
        "paper_result_reproduced": False,
        "paper_benchmarks_evaluated": False,
        "contains_slaclip": True,
        "slaclip_variant": slaclip_variant,
        "uses_baseline_calibration": groupwise,
        "uses_fixed_target_clip_fraction": False,
        "methods_in_order": list(EXPECTED_METHODS),
        "normalized_scientific_contract_sha256": normalized_fingerprint,
        "arm_evidence": {
            BASELINE_METHOD: {
                "directory": str(baseline_path),
                "run_config_sha256": baseline["run_config_sha256"],
                "root_final_summary_sha256": baseline["root_summary_sha256"],
                "run_config_fingerprint": baseline["run_config_fingerprint"],
            },
            ADAPTIVE_METHOD: {
                "directory": str(adaptive_path),
                "run_config_sha256": adaptive["run_config_sha256"],
                "root_final_summary_sha256": adaptive["root_summary_sha256"],
                "run_config_fingerprint": adaptive["run_config_fingerprint"],
            },
        },
        "models": model_comparisons,
        "privacy_notice": {
            "end_to_end_dp_certified": False,
            "epsilon": None,
            "sigma_is_not_epsilon": True,
            "baseline_calibration_consumed": groupwise,
            "controller_consumes_noisy_cdf_endpoints": True,
            "exact_cdf_diagnostics_are_non_dp_private_data": True,
            "adaptive_arm_independently_privacy_certified": False,
        },
    }
    if groupwise:
        core.update(
            {
                "slaclip_contract_schema": slaclip_schema,
                "target_parameterization": "per_gradient_group",
                "generalized_full_slaclip_beta": True,
                "baseline_clip_norm": baseline_clip_norm,
                "adaptive_initial_clip_norm": adaptive_initial_clip_norm,
                "baseline_calibration_lock_sha256": slaclip_contract[
                    "hyperparameter_provenance"
                ]["calibration_lock_sha256"],
                "baseline_calibration_consumed_at_runtime": False,
                "hyperparameters_derived_from_non_dp_fixed_development": True,
            }
        )
        core["privacy_notice"].update(
            {
                "baseline_calibration_consumed_at_runtime": False,
                "hyperparameters_derived_from_non_dp_fixed_development": True,
            }
        )
    core["comparison_fingerprint"] = canonical_json_fingerprint(core)
    return core


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = _absolute_path(path)
    _validate_private_directory(destination.parent, "comparison parent directory")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing comparison: {destination}")
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("failed to write full-SlaClip comparison")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _validate_private_file_metadata(
            temporary.lstat(), temporary, "temporary full-SlaClip comparison"
        )
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(destination.parent)
        _validate_private_file_metadata(
            destination.lstat(), destination, "full-SlaClip comparison"
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing comparison: {destination}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if os.path.lexists(temporary):
            temporary.unlink()
            _fsync_directory(destination.parent)


def _validate_existing_comparison(
    path: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    existing, _ = _load_private_object(path, "existing full-SlaClip comparison")
    if existing.get("created_at_utc") is None:
        raise RuntimeError("existing full-SlaClip comparison timestamp is missing")
    stored_fingerprint = existing.get("comparison_fingerprint")
    core = {
        key: value
        for key, value in existing.items()
        if key not in {"comparison_fingerprint", "created_at_utc"}
    }
    expected_fingerprint = canonical_json_fingerprint(core)
    if stored_fingerprint != expected_fingerprint:
        raise RuntimeError("existing full-SlaClip comparison fingerprint mismatch")
    if stored_fingerprint != expected.get("comparison_fingerprint"):
        raise RuntimeError("existing full-SlaClip comparison no longer matches the arms")
    return existing


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    comparison = build_comparison(args.baseline_dir, args.adaptive_dir)
    output = _absolute_path(args.output)
    if args.verify_existing:
        existing = _validate_existing_comparison(output, comparison)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "comparison_fingerprint": existing["comparison_fingerprint"],
                    "output": str(output),
                },
                indent=2,
            )
        )
        return
    payload = {**comparison, "created_at_utc": utc_now()}
    try:
        atomic_create_json(output, payload)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "status": COMPARISON_STATUS,
                "comparison_fingerprint": comparison["comparison_fingerprint"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
