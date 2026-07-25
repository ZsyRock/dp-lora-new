#!/usr/bin/env python3
"""Validate and summarize one matched DP-LoRA/SlaClip-Q Level-1 pair.

The calibration consumed by the adaptive arm is derived from exact baseline
clipping diagnostics.  Consequently this comparison is deliberately labelled
non-DP and cannot establish either a paper-result reproduction or an epsilon
guarantee.  The validator binds the calibration, both completed arms, their
matched stochastic schedules, and the complete adaptive threshold trajectory.
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
    from paper_repro.calibrate_slaclip import (
        CALIBRATION_PRIVACY_CLASS,
        CALIBRATION_REDUCER,
        DEFAULT_MODELS,
        LORA_GROUPS,
        load_and_validate_calibration,
        round_shard_prefix_sha256,
    )
    from paper_repro.reproducibility import (
        METHOD_SPECS,
        canonical_json_fingerprint,
    )
    from paper_repro.slaclip import slaclip_q_update
except ModuleNotFoundError:  # Support direct execution.
    from calibrate_slaclip import (  # type: ignore[no-redef]
        CALIBRATION_PRIVACY_CLASS,
        CALIBRATION_REDUCER,
        DEFAULT_MODELS,
        LORA_GROUPS,
        load_and_validate_calibration,
        round_shard_prefix_sha256,
    )
    from reproducibility import (  # type: ignore[no-redef]
        METHOD_SPECS,
        canonical_json_fingerprint,
    )
    from slaclip import slaclip_q_update  # type: ignore[no-redef]


BASELINE_METHOD = "paper_dp_lora"
ADAPTIVE_METHOD = "slaclip_q_dp_lora"
EXPECTED_METHODS = (BASELINE_METHOD, ADAPTIVE_METHOD)
COMPARISON_SCHEMA_VERSION = 1
COMPARISON_STATUS = "VALID_MATCHED_DP_LORA_SLACLIP_Q_LEVEL1_COMPARISON"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    raw = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if os.path.lexists(raw):
        # Reject a symlink at the requested object itself, while allowing the
        # canonicalized ancestor alias used by this cluster for /scratch.
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


def _positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{description} must be a positive integer")
    return value


def _require_claims(
    run_config: Mapping[str, Any], root_summary: Mapping[str, Any], *, adaptive: bool
) -> None:
    reproduction = run_config.get("reproduction_claim")
    if not isinstance(reproduction, dict):
        raise RuntimeError("run config has no reproduction claim")
    expected_reproduction = {
        "level": 1,
        "paper_result_reproduced": False,
        "paper_benchmarks_evaluated": False,
    }
    for key, expected in expected_reproduction.items():
        if reproduction.get(key) != expected:
            raise RuntimeError(f"unsafe reproduction claim: {key}")

    privacy = run_config.get("privacy_claim")
    if not isinstance(privacy, dict):
        raise RuntimeError("run config has no privacy claim")
    expected_privacy = {
        "end_to_end_dp_certified": False,
        "epsilon": None,
        "sigma_is_not_epsilon": True,
        "diagnostics_are_private_non_dp_data": True,
        "baseline_derived_calibration_is_non_dp": adaptive,
    }
    for key, expected in expected_privacy.items():
        if privacy.get(key) != expected:
            raise RuntimeError(f"unsafe privacy claim: {key}")

    expected_root = {
        "reproduction_claim_level": 1,
        "paper_result_reproduced": False,
        "paper_benchmarks_evaluated": False,
        "privacy_claim": False,
    }
    for key, expected in expected_root.items():
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


def _load_arm(directory: Path, method: str, *, adaptive: bool) -> dict[str, Any]:
    _validate_private_directory(directory, f"{method} run directory")
    run_config, run_config_bytes = _load_private_object(
        directory / "run_config.json", f"{method} run config"
    )
    root_summary, root_summary_bytes = _load_private_object(
        directory / "final_summary.json", f"{method} root summary"
    )
    expected_method_spec = asdict(METHOD_SPECS[method])
    run_identity = {
        "schema_version": 2,
        "method": method,
        "method_spec": expected_method_spec,
        "contains_slaclip": adaptive,
    }
    for key, expected in run_identity.items():
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
    slaclip_contract = algorithm.get("slaclip_q")
    if adaptive != isinstance(slaclip_contract, dict):
        raise RuntimeError(f"{method} SlaClip-Q contract presence mismatch")

    root_identity = {
        "schema_version": 2,
        "status": "COMPLETED",
        "method": method,
        "method_spec": expected_method_spec,
        "contains_slaclip": adaptive,
        "run_config_fingerprint": run_fingerprint,
    }
    for key, expected in root_identity.items():
        if root_summary.get(key) != expected:
            raise RuntimeError(f"{method} root-summary identity mismatch: {key}")
    _require_claims(run_config, root_summary, adaptive=adaptive)
    root_models = root_summary.get("models")
    if not isinstance(root_models, dict) or tuple(root_models) != DEFAULT_MODELS:
        raise RuntimeError(f"{method} root-summary model set/order mismatch")
    root_schedules = root_summary.get("sample_schedule_sha256_by_model")
    if not isinstance(root_schedules, dict) or tuple(root_schedules) != DEFAULT_MODELS:
        raise RuntimeError(f"{method} root sample-schedule map is invalid")

    model_records: dict[str, Any] = {}
    rounds = _positive_integer(effective.get("rounds"), f"{method} rounds")
    clients = _positive_integer(
        effective.get("num_clients"), f"{method} client count"
    )
    for model in DEFAULT_MODELS:
        model_dir = directory / model
        _validate_private_directory(model_dir, f"{method}/{model} output directory")
        model_summary, model_summary_bytes = _load_private_object(
            model_dir / "final_summary.json", f"{method}/{model} final summary"
        )
        if root_models[model] != model_summary:
            raise RuntimeError(f"{method}/{model} root/model summary mismatch")
        model_identity = {
            "schema_version": 2,
            "status": "COMPLETED",
            "method": method,
            "model": model,
            "privacy_claim": False,
            "run_config_fingerprint": run_fingerprint,
            "client_steps": rounds * clients,
        }
        for key, expected in model_identity.items():
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
            adapter_dir / "adapter_model.safetensors",
            f"{method}/{model} adapter",
        )
        adapter_config_bytes = _read_private_bytes(
            adapter_dir / "adapter_config.json",
            f"{method}/{model} adapter config",
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
        "run_config": run_config,
        "run_config_bytes": run_config_bytes,
        "run_config_sha256": _sha256_bytes(run_config_bytes),
        "root_summary": root_summary,
        "root_summary_bytes": root_summary_bytes,
        "root_summary_sha256": _sha256_bytes(root_summary_bytes),
        "contract": contract,
        "run_config_fingerprint": run_fingerprint,
        "effective_config": effective,
        "algorithm_contract": algorithm,
        "slaclip_contract": slaclip_contract,
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
    algorithm.pop("slaclip_q", None)
    return normalized


def _calibration_targets(calibration: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {
        model: {
            group: float(
                calibration["models"][model]["groups"][group][
                    "target_clip_fraction"
                ]
            )
            for group in LORA_GROUPS
        }
        for model in DEFAULT_MODELS
    }


def _validate_calibration_binding(
    calibration_path: Path,
    calibration: Mapping[str, Any],
    baseline: Mapping[str, Any],
    adaptive: Mapping[str, Any],
) -> tuple[str, dict[str, dict[str, float]]]:
    calibration_bytes = _read_private_bytes(
        calibration_path, "SlaClip calibration"
    )
    calibration_sha = _sha256_bytes(calibration_bytes)
    source = calibration["source"]
    baseline_source_expected = {
        "method": BASELINE_METHOD,
        "baseline_dir": str(baseline["directory"]),
        "run_config_fingerprint": baseline["run_config_fingerprint"],
        "run_config_sha256": baseline["run_config_sha256"],
        "root_final_summary_sha256": baseline["root_summary_sha256"],
        "repository_sha": baseline["contract"].get("repository_sha"),
    }
    if source != baseline_source_expected:
        raise RuntimeError("calibration source evidence does not match the baseline")
    if adaptive["contract"].get("repository_sha") != source["repository_sha"]:
        raise RuntimeError("adaptive repository SHA differs from calibration source")

    for model in DEFAULT_MODELS:
        calibration_model = calibration["models"][model]
        baseline_model = baseline["models"][model]
        if calibration_model["model_final_summary_sha256"] != baseline_model[
            "summary_sha256"
        ]:
            raise RuntimeError(f"calibration {model} model-summary evidence mismatch")
        baseline_prefix = baseline_model["summary"].get(
            "round_shard_prefix_sha256"
        )
        if calibration_model["round_shard_prefix_sha256"] != baseline_prefix:
            raise RuntimeError(f"calibration {model} round-prefix evidence mismatch")
        recomputed_prefix = round_shard_prefix_sha256(
            baseline["directory"] / model / "private_diagnostics" / "rounds",
            baseline["rounds"],
        )
        if recomputed_prefix != baseline_prefix:
            raise RuntimeError(f"baseline {model} round-prefix changed after calibration")
        if (
            calibration_model["rounds"] != baseline["rounds"]
            or calibration_model["clients_per_round"] != baseline["clients"]
        ):
            raise RuntimeError(f"calibration {model} schedule evidence mismatch")

    targets = _calibration_targets(calibration)
    slaclip_contract = adaptive["slaclip_contract"]
    assert isinstance(slaclip_contract, dict)
    expected_contract_calibration = {
        "privacy_class": CALIBRATION_PRIVACY_CLASS,
        "calibration_fingerprint": calibration["calibration_fingerprint"],
        "file_sha256": calibration_sha,
        "source": source,
        "reducer": CALIBRATION_REDUCER,
        "targets": targets,
    }
    if slaclip_contract.get("calibration") != expected_contract_calibration:
        raise RuntimeError("adaptive calibration contract does not match its file")
    if slaclip_contract.get("independently_privacy_certified") is not False:
        raise RuntimeError("adaptive SlaClip-Q contract makes an unsafe privacy claim")
    return calibration_sha, targets


def _validate_controller_trajectory(
    adaptive: Mapping[str, Any],
    *,
    model: str,
    targets: Mapping[str, float],
    baseline_clip_norm: float,
) -> dict[str, Any]:
    slaclip_contract = adaptive["slaclip_contract"]
    assert isinstance(slaclip_contract, dict)
    if slaclip_contract.get("variant") != "SlaClip-Q_fixed_target":
        raise RuntimeError("adaptive SlaClip-Q variant mismatch")
    controller = slaclip_contract.get("controller")
    if not isinstance(controller, dict):
        raise RuntimeError("adaptive controller contract is missing")
    eta = _finite_number(controller.get("eta"), "SlaClip-Q eta")
    if eta < 0.0:
        raise RuntimeError("SlaClip-Q eta must be non-negative")
    num_slots = _positive_integer(controller.get("num_slots"), "SlaClip-Q slots")
    c_min = _finite_number(controller.get("c_min"), "SlaClip-Q lower bound")
    c_max = _finite_number(controller.get("c_max"), "SlaClip-Q upper bound")
    initial = _finite_number(
        controller.get("initial_clip_threshold"), "SlaClip-Q initial threshold"
    )
    if c_min <= 0.0 or c_max < c_min or not c_min <= initial <= c_max:
        raise RuntimeError("SlaClip-Q threshold contract is invalid")
    if initial != baseline_clip_norm:
        raise RuntimeError("adaptive initial threshold differs from baseline C")
    if controller.get("expected_release_records") != adaptive["clients"]:
        raise RuntimeError("adaptive controller release count mismatch")

    rounds_dir = (
        adaptive["directory"] / model / "private_diagnostics" / "rounds"
    )
    _validate_private_directory(rounds_dir, f"adaptive/{model} rounds directory")
    expected_thresholds = {group: initial for group in LORA_GROUPS}
    trajectory: list[dict[str, Any]] = []
    for round_index in range(1, adaptive["rounds"] + 1):
        shard, _ = _load_private_object(
            rounds_dir / f"round-{round_index:05d}.json",
            f"adaptive/{model} round shard",
        )
        shard_identity = {
            "schema_version": 2,
            "method": ADAPTIVE_METHOD,
            "model": model,
            "round": round_index,
        }
        for key, expected in shard_identity.items():
            if shard.get(key) != expected:
                raise RuntimeError(
                    f"adaptive/{model} round shard identity mismatch: {key}"
                )
        round_summary = shard.get("round_summary")
        if not isinstance(round_summary, dict) or round_summary.get("round") != round_index:
            raise RuntimeError(f"adaptive/{model} round summary is invalid")
        value = round_summary.get("slaclip_controller")
        if not isinstance(value, dict):
            raise RuntimeError(f"adaptive/{model} controller round is missing")
        expected_header = {
            "variant": "SlaClip-Q",
            "update_timing": "once_after_all_clients_for_use_in_next_round",
            "clients": adaptive["clients"],
            "num_slots": num_slots,
            "eta": eta,
            "c_min": c_min,
            "c_max": c_max,
        }
        for key, expected in expected_header.items():
            if value.get(key) != expected:
                raise RuntimeError(
                    f"adaptive/{model} controller header mismatch: {key}"
                )
        for group in LORA_GROUPS:
            group_value = value.get(group)
            if not isinstance(group_value, dict):
                raise RuntimeError(f"adaptive/{model}/{group} controller is missing")
            used = _finite_number(
                group_value.get("clip_threshold_used"),
                f"adaptive/{model}/{group} threshold",
            )
            if used != expected_thresholds[group]:
                raise RuntimeError(
                    f"adaptive/{model}/{group} threshold trajectory is discontinuous"
                )
            target = _fraction(
                group_value.get("target_clip_fraction"),
                f"adaptive/{model}/{group} target",
            )
            if target != targets[group]:
                raise RuntimeError(
                    f"adaptive/{model}/{group} controller target mismatch"
                )
            if group_value.get("target_clipped_fraction") != target:
                raise RuntimeError(
                    f"adaptive/{model}/{group} update target mismatch"
                )
            target_unclipped = _finite_number(
                group_value.get("target_unclipped_proxy"),
                f"adaptive/{model}/{group} unclipped target",
            )
            if not math.isclose(target_unclipped, 1.0 - target, abs_tol=1e-15):
                raise RuntimeError(
                    f"adaptive/{model}/{group} target complement mismatch"
                )
            noisy_slots = group_value.get("noisy_unclipped_proxy_by_slot")
            if (
                not isinstance(noisy_slots, list)
                or len(noisy_slots) != num_slots
                or any(
                    not math.isfinite(
                        _finite_number(item, f"adaptive/{model}/{group} proxy")
                    )
                    for item in noisy_slots
                )
            ):
                raise RuntimeError(f"adaptive/{model}/{group} proxy vector is invalid")
            controller_error = _finite_number(
                group_value.get("controller_error"),
                f"adaptive/{model}/{group} controller error",
            )
            if not math.isclose(
                controller_error,
                target_unclipped - float(noisy_slots[0]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"adaptive/{model}/{group} controller error mismatch"
                )
            recomputed_update = slaclip_q_update(
                used,
                target,
                float(noisy_slots[0]),
                eta,
                c_min,
                c_max,
            )
            expected_update_fields = {
                "target_clipped_fraction": recomputed_update[
                    "target_clipped_fraction"
                ],
                "target_unclipped_proxy": recomputed_update[
                    "target_unclipped_proxy"
                ],
                "noisy_unclipped_proxy": recomputed_update[
                    "noisy_unclipped_proxy"
                ],
                "controller_error": recomputed_update["controller_error"],
                "eta": recomputed_update["eta"],
                "raw_log_update": recomputed_update["raw_log_update"],
                "bounded_log_update": recomputed_update["bounded_log_update"],
                "log_update_was_clamped": recomputed_update[
                    "log_update_was_clamped"
                ],
                "unbounded_next_clip_threshold": recomputed_update[
                    "unbounded_next_clip_norm"
                ],
                "next_clip_threshold": recomputed_update["next_clip_norm"],
                "c_min": recomputed_update["min_clip_norm"],
                "c_max": recomputed_update["max_clip_norm"],
                "hit_lower_bound": recomputed_update["hit_lower_bound"],
                "hit_upper_bound": recomputed_update["hit_upper_bound"],
            }
            for key, expected in expected_update_fields.items():
                if group_value.get(key) != expected:
                    raise RuntimeError(
                        f"adaptive/{model}/{group} SlaClip-Q formula mismatch: {key}"
                    )
            if group_value.get("c_min") != c_min or group_value.get("c_max") != c_max:
                raise RuntimeError(
                    f"adaptive/{model}/{group} controller bounds mismatch"
                )
            unbounded = _finite_number(
                group_value.get("unbounded_next_clip_threshold"),
                f"adaptive/{model}/{group} unbounded threshold",
            )
            next_threshold = _finite_number(
                group_value.get("next_clip_threshold"),
                f"adaptive/{model}/{group} next threshold",
            )
            expected_next = max(c_min, min(c_max, unbounded))
            if next_threshold != expected_next or not c_min <= next_threshold <= c_max:
                raise RuntimeError(
                    f"adaptive/{model}/{group} next threshold violates bounds"
                )
            expected_lower_hit = unbounded < c_min
            expected_upper_hit = unbounded > c_max
            if (
                group_value.get("hit_lower_bound") is not expected_lower_hit
                or group_value.get("hit_upper_bound") is not expected_upper_hit
            ):
                raise RuntimeError(
                    f"adaptive/{model}/{group} bound-hit telemetry mismatch"
                )
            expected_thresholds[group] = next_threshold
        trajectory.append(value)

    model_summary = adaptive["models"][model]["summary"]
    prefix = round_shard_prefix_sha256(rounds_dir, adaptive["rounds"])
    if model_summary.get("round_shard_prefix_sha256") != prefix:
        raise RuntimeError(f"adaptive/{model} round-prefix mismatch")
    behavior = model_summary["behavior_summary"]
    controller_summary = behavior.get("slaclip_controller")
    if not isinstance(controller_summary, dict):
        raise RuntimeError(f"adaptive/{model} controller summary is missing")
    if (
        controller_summary.get("variant") != "SlaClip-Q"
        or controller_summary.get("rounds") != adaptive["rounds"]
        or controller_summary.get("trajectory_sha256")
        != canonical_json_fingerprint(trajectory)
    ):
        raise RuntimeError(f"adaptive/{model} controller trajectory digest mismatch")
    controller_groups = controller_summary.get("groups")
    if not isinstance(controller_groups, dict) or tuple(controller_groups) != LORA_GROUPS:
        raise RuntimeError(f"adaptive/{model} controller group summary is invalid")
    for group in LORA_GROUPS:
        group_summary = controller_groups[group]
        if (
            not isinstance(group_summary, dict)
            or group_summary.get("target_clip_fraction") != targets[group]
            or group_summary.get("final_next_clip_threshold")
            != expected_thresholds[group]
        ):
            raise RuntimeError(f"adaptive/{model}/{group} controller summary mismatch")
    model_contract = model_summary.get("slaclip_q")
    expected_model_contract = {
        "contract": slaclip_contract,
        "target_clip_fraction_by_group": dict(targets),
        "final_next_clip_threshold_by_group": dict(expected_thresholds),
        "controller_summary": controller_summary,
    }
    if model_contract != expected_model_contract:
        raise RuntimeError(f"adaptive/{model} SlaClip-Q model contract mismatch")
    return {
        "rounds": adaptive["rounds"],
        "trajectory_sha256": controller_summary["trajectory_sha256"],
        "initial_clip_threshold": initial,
        "final_next_clip_threshold_by_group": dict(expected_thresholds),
        "target_clip_fraction_by_group": dict(targets),
        "c_min": c_min,
        "c_max": c_max,
        "eta": eta,
        "num_slots": num_slots,
    }


def build_comparison(
    baseline_dir: str | os.PathLike[str],
    adaptive_dir: str | os.PathLike[str],
    calibration_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate the two arms and return a fingerprinted comparison core."""

    baseline_path = _absolute_path(baseline_dir)
    adaptive_path = _absolute_path(adaptive_dir)
    calibration_file = _absolute_path(calibration_path)
    baseline = _load_arm(baseline_path, BASELINE_METHOD, adaptive=False)
    adaptive = _load_arm(adaptive_path, ADAPTIVE_METHOD, adaptive=True)
    normalized_baseline = _normalized_contract(baseline["contract"])
    normalized_adaptive = _normalized_contract(adaptive["contract"])
    if normalized_baseline != normalized_adaptive:
        raise RuntimeError("baseline and adaptive arms differ outside SlaClip-Q")
    normalized_fingerprint = canonical_json_fingerprint(normalized_baseline)

    calibration = load_and_validate_calibration(
        calibration_file,
        expected_models=DEFAULT_MODELS,
        expected_source_dir=baseline_path,
    )
    calibration_sha, target_map = _validate_calibration_binding(
        calibration_file, calibration, baseline, adaptive
    )
    baseline_clip_norm = _finite_number(
        baseline["effective_config"].get("clip_norm"), "baseline clip norm"
    )
    adaptive_clip_norm = _finite_number(
        adaptive["effective_config"].get("clip_norm"), "adaptive initial clip norm"
    )
    if adaptive_clip_norm != baseline_clip_norm:
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
        if "slaclip_q" in baseline_model["summary"] or "slaclip_controller" in baseline_model[
            "summary"
        ].get("behavior_summary", {}):
            raise RuntimeError(f"baseline/{model} unexpectedly contains SlaClip")
        controller = _validate_controller_trajectory(
            adaptive,
            model=model,
            targets=target_map[model],
            baseline_clip_norm=baseline_clip_norm,
        )
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
            "target_clip_fraction_by_group": target_map[model],
            "controller": controller,
            "arms": {
                BASELINE_METHOD: {
                    "adapter_sha256": baseline_model["adapter_sha256"],
                    "adapter_config_sha256": baseline_model[
                        "adapter_config_sha256"
                    ],
                    "clipping": baseline_model["summary"].get("clipping"),
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
        "methods_in_order": list(EXPECTED_METHODS),
        "normalized_scientific_contract_sha256": normalized_fingerprint,
        "calibration_evidence": {
            "path": str(calibration_file),
            "file_sha256": calibration_sha,
            "calibration_fingerprint": calibration["calibration_fingerprint"],
            "privacy_class": CALIBRATION_PRIVACY_CLASS,
            "reducer": CALIBRATION_REDUCER,
            "source": calibration["source"],
            "target_clip_fraction_by_model_and_group": target_map,
        },
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
            "data_dependent_baseline_calibration_is_non_dp": True,
            "exact_diagnostics_are_non_dp_private_data": True,
            "adaptive_arm_independently_privacy_certified": False,
        },
    }
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
                raise OSError("failed to write SlaClip comparison")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _validate_private_file_metadata(
            temporary.lstat(), temporary, "temporary SlaClip comparison"
        )
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(destination.parent)
        _validate_private_file_metadata(
            destination.lstat(), destination, "SlaClip comparison"
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
    existing, _ = _load_private_object(path, "existing SlaClip comparison")
    if existing.get("created_at_utc") is None:
        raise RuntimeError("existing SlaClip comparison timestamp is missing")
    stored_fingerprint = existing.get("comparison_fingerprint")
    core = {
        key: value
        for key, value in existing.items()
        if key not in {"comparison_fingerprint", "created_at_utc"}
    }
    expected_fingerprint = canonical_json_fingerprint(core)
    if stored_fingerprint != expected_fingerprint:
        raise RuntimeError("existing SlaClip comparison fingerprint mismatch")
    if stored_fingerprint != expected.get("comparison_fingerprint"):
        raise RuntimeError("existing SlaClip comparison no longer matches the arms")
    return existing


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    comparison = build_comparison(
        args.baseline_dir, args.adaptive_dir, args.calibration
    )
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
