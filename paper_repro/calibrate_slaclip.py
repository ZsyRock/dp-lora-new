#!/usr/bin/env python3
"""Build a fixed-target SlaClip-Q calibration from a completed DP-LoRA run.

The source values are exact clipping diagnostics and are therefore private,
non-DP data.  This tool validates the completed paper-DP arm and its complete
round-shard prefix before taking, independently for every model and LoRA
factor group, the median of the per-round realized clipping fractions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from paper_repro.reproducibility import canonical_json_fingerprint
except ModuleNotFoundError:  # Support direct execution.
    from reproducibility import canonical_json_fingerprint  # type: ignore[no-redef]


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_STATUS = "VALIDATED_BASELINE_MEDIAN_CLIP_CALIBRATION"
CALIBRATION_PRIVACY_CLASS = "NON_DP_PRIVATE_DIAGNOSTIC_DERIVED_TARGET"
CALIBRATION_REDUCER = "statistics.median_of_per_round_actual_clipped_fraction"
SOURCE_METHOD = "paper_dp_lora"
DEFAULT_MODELS = ("bert", "gpt2")
LORA_GROUPS = ("A", "B")
ROUND_PREFIX_DOMAIN = b"dp-lora-round-shard-prefix-v1\0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    raw = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if os.path.lexists(raw):
        # Reject a symlink at the object being opened, but canonicalize trusted
        # ancestor aliases such as this cluster's /scratch -> /iridisfs/scratch.
        if stat.S_ISLNK(raw.lstat().st_mode):
            return raw
        return raw.resolve(strict=True)
    return raw.parent.resolve(strict=True) / raw.name


def _validate_private_directory(path: Path, description: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
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
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    _validate_private_file_metadata(before, path, description)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
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
        identity_before = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        identity_after = (
            finished.st_dev,
            finished.st_ino,
            finished.st_size,
            finished.st_mtime_ns,
            finished.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError(f"{description} changed while reading: {path}")
        value = b"".join(chunks)
        if len(value) != finished.st_size:
            raise RuntimeError(f"{description} read was incomplete: {path}")
        return value
    finally:
        os.close(descriptor)


def _load_private_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
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


def _require_fraction(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{description} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RuntimeError(f"{description} must be a finite number in [0, 1]")
    return result


def _require_positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{description} must be a positive integer")
    return value


def round_shard_prefix_sha256(rounds_directory: Path, rounds: int) -> str:
    """Return the runner-compatible digest of a complete round-shard prefix."""

    _validate_private_directory(rounds_directory, "round diagnostics directory")
    paths = _complete_round_shard_paths(rounds_directory, rounds)
    digest = hashlib.sha256(ROUND_PREFIX_DOMAIN)
    for path in paths:
        encoded = _read_private_bytes(path, "round diagnostic shard")
        digest.update(path.name.encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(encoded).digest())
    digest.update(rounds.to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _complete_round_shard_paths(
    rounds_directory: Path, rounds: int
) -> list[Path]:
    _validate_private_directory(rounds_directory, "round diagnostics directory")
    expected_paths = {
        rounds_directory / f"round-{round_index:05d}.json"
        for round_index in range(1, rounds + 1)
    }
    actual_paths = set(rounds_directory.glob("round-*.json"))
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise RuntimeError(
            f"round shard set mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    return [
        rounds_directory / f"round-{round_index:05d}.json"
        for round_index in range(1, rounds + 1)
    ]


def _validate_source_identity(
    run_config: dict[str, Any], root_summary: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if run_config.get("schema_version") != 2:
        raise RuntimeError("unsupported baseline run-config schema")
    if run_config.get("method") != SOURCE_METHOD:
        raise RuntimeError("baseline run config is not paper_dp_lora")
    contract = run_config.get("scientific_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("baseline run config has no scientific contract")
    fingerprint = canonical_json_fingerprint(contract)
    if run_config.get("run_config_fingerprint") != fingerprint:
        raise RuntimeError("baseline scientific-contract fingerprint mismatch")
    method_contract = contract.get("method")
    if not isinstance(method_contract, dict) or method_contract.get("name") != SOURCE_METHOD:
        raise RuntimeError("baseline scientific method contract is not paper_dp_lora")
    effective = run_config.get("effective_config")
    if not isinstance(effective, dict) or effective != contract.get("effective_config"):
        raise RuntimeError("baseline effective config does not match its contract")
    if effective.get("method") != SOURCE_METHOD:
        raise RuntimeError("baseline effective method is not paper_dp_lora")
    if root_summary.get("schema_version") != 2:
        raise RuntimeError("unsupported baseline root-summary schema")
    root_identity = {
        "status": "COMPLETED",
        "method": SOURCE_METHOD,
        "contains_slaclip": False,
        "run_config_fingerprint": fingerprint,
    }
    for key, expected in root_identity.items():
        if root_summary.get(key) != expected:
            raise RuntimeError(f"baseline root-summary identity mismatch: {key}")
    return contract, effective, fingerprint


def _validate_round_and_extract(
    shard: dict[str, Any],
    *,
    path: Path,
    model: str,
    round_index: int,
    clients_per_round: int,
) -> dict[str, float]:
    shard_identity = {
        "schema_version": 2,
        "method": SOURCE_METHOD,
        "model": model,
        "round": round_index,
    }
    for key, expected in shard_identity.items():
        if shard.get(key) != expected:
            raise RuntimeError(f"round shard identity mismatch for {key}: {path}")
    records = shard.get("client_records")
    if not isinstance(records, list) or len(records) != clients_per_round:
        raise RuntimeError(f"round shard client count mismatch: {path}")
    if [record.get("client") if isinstance(record, dict) else None for record in records] != list(
        range(clients_per_round)
    ):
        raise RuntimeError(f"round shard client IDs are invalid or reordered: {path}")
    round_summary = shard.get("round_summary")
    if not isinstance(round_summary, dict) or round_summary.get("round") != round_index:
        raise RuntimeError(f"round summary identity mismatch: {path}")
    if round_summary.get("clients") != clients_per_round:
        raise RuntimeError(f"round summary client count mismatch: {path}")

    fractions: dict[str, float] = {}
    for group in LORA_GROUPS:
        clipped_flags: list[bool] = []
        for record in records:
            assert isinstance(record, dict)
            record_identity = {
                "method": SOURCE_METHOD,
                "model": model,
                "round": round_index,
            }
            if any(record.get(key) != expected for key, expected in record_identity.items()):
                raise RuntimeError(f"round/client identity mismatch: {path}")
            gradient_groups = record.get("gradient_groups")
            if not isinstance(gradient_groups, dict):
                raise RuntimeError(f"round/client gradient groups are missing: {path}")
            group_statistics = gradient_groups.get(group)
            if not isinstance(group_statistics, dict) or not isinstance(
                group_statistics.get("clipped"), bool
            ):
                raise RuntimeError(f"round/client {group} clipping flag is invalid: {path}")
            clipped_flags.append(group_statistics["clipped"])
        clipped_count = sum(clipped_flags)
        clipped_fraction = clipped_count / clients_per_round
        summary_group = round_summary.get(group)
        if not isinstance(summary_group, dict):
            raise RuntimeError(f"round summary {group} group is missing: {path}")
        if summary_group.get("clipped_count") != clipped_count:
            raise RuntimeError(f"round summary {group} clipped count mismatch: {path}")
        recorded_fraction = _require_fraction(
            summary_group.get("clipped_fraction"),
            f"round summary {group} clipped fraction",
        )
        if recorded_fraction != clipped_fraction:
            raise RuntimeError(f"round summary {group} clipped fraction mismatch: {path}")
        fractions[group] = clipped_fraction
    return fractions


def _validate_model_totals(
    model_summary: dict[str, Any],
    *,
    model: str,
    rounds: int,
    clients_per_round: int,
    values: Mapping[str, Sequence[float]],
) -> None:
    identity = {
        "schema_version": 2,
        "status": "COMPLETED",
        "method": SOURCE_METHOD,
        "model": model,
        "client_steps": rounds * clients_per_round,
    }
    for key, expected in identity.items():
        if model_summary.get(key) != expected:
            raise RuntimeError(f"baseline {model} summary mismatch: {key}")
    clipping = model_summary.get("clipping")
    behavior = model_summary.get("behavior_summary")
    behavior_groups = behavior.get("groups") if isinstance(behavior, dict) else None
    if not isinstance(clipping, dict) or not isinstance(behavior_groups, dict):
        raise RuntimeError(f"baseline {model} clipping summaries are missing")
    total_steps = rounds * clients_per_round
    for group in LORA_GROUPS:
        count = sum(
            int(round(fraction * clients_per_round)) for fraction in values[group]
        )
        fraction = count / total_steps
        clipping_group = clipping.get(group)
        behavior_group = behavior_groups.get(group)
        if not isinstance(clipping_group, dict) or not isinstance(behavior_group, dict):
            raise RuntimeError(f"baseline {model}/{group} clipping summary is missing")
        if (
            clipping_group.get("count") != count
            or _require_fraction(
                clipping_group.get("fraction"),
                f"baseline {model}/{group} clipping fraction",
            )
            != fraction
            or behavior_group.get("actual_clipped_count") != count
            or _require_fraction(
                behavior_group.get("actual_clipped_fraction"),
                f"baseline {model}/{group} behavior clipping fraction",
            )
            != fraction
        ):
            raise RuntimeError(f"baseline {model}/{group} clipping totals do not reconcile")


def build_calibration(
    baseline_dir: str | os.PathLike[str],
    *,
    expected_models: Sequence[str] = DEFAULT_MODELS,
) -> dict[str, Any]:
    """Validate one completed baseline and return its median-target calibration."""

    models = tuple(expected_models)
    if models != DEFAULT_MODELS:
        raise ValueError(f"baseline calibration requires exactly {DEFAULT_MODELS}")
    source_dir = _absolute_path(baseline_dir)
    _validate_private_directory(source_dir, "baseline run directory")
    run_config_path = source_dir / "run_config.json"
    root_summary_path = source_dir / "final_summary.json"
    run_config, run_config_bytes = _load_private_object(
        run_config_path, "baseline run config"
    )
    root_summary, root_summary_bytes = _load_private_object(
        root_summary_path, "baseline root summary"
    )
    contract, effective, run_fingerprint = _validate_source_identity(
        run_config, root_summary
    )
    rounds = _require_positive_integer(effective.get("rounds"), "baseline rounds")
    clients_per_round = _require_positive_integer(
        effective.get("num_clients"), "baseline client count"
    )
    if run_config.get("models") != list(models):
        raise RuntimeError("baseline run-config model set/order mismatch")
    root_models = root_summary.get("models")
    if not isinstance(root_models, dict) or tuple(root_models) != models:
        raise RuntimeError("baseline root-summary model set/order mismatch")

    calibration_models: dict[str, Any] = {}
    for model in models:
        model_dir = source_dir / model
        diagnostics_dir = model_dir / "private_diagnostics"
        rounds_dir = diagnostics_dir / "rounds"
        _validate_private_directory(model_dir, f"baseline {model} output directory")
        _validate_private_directory(
            diagnostics_dir, f"baseline {model} diagnostics directory"
        )
        _validate_private_directory(rounds_dir, f"baseline {model} rounds directory")
        model_summary_path = model_dir / "final_summary.json"
        model_summary, model_summary_bytes = _load_private_object(
            model_summary_path, f"baseline {model} final summary"
        )
        if root_models[model] != model_summary:
            raise RuntimeError(f"baseline root and model summaries do not match: {model}")
        if model_summary.get("run_config_fingerprint") != run_fingerprint:
            raise RuntimeError(f"baseline {model} run fingerprint mismatch")

        shard_paths = _complete_round_shard_paths(rounds_dir, rounds)
        prefix = hashlib.sha256(ROUND_PREFIX_DOMAIN)
        per_group_values: dict[str, list[float]] = {group: [] for group in LORA_GROUPS}
        for round_index, shard_path in enumerate(shard_paths, start=1):
            shard, shard_bytes = _load_private_object(
                shard_path, "round diagnostic shard"
            )
            prefix.update(shard_path.name.encode("ascii"))
            prefix.update(b"\0")
            prefix.update(hashlib.sha256(shard_bytes).digest())
            fractions = _validate_round_and_extract(
                shard,
                path=shard_path,
                model=model,
                round_index=round_index,
                clients_per_round=clients_per_round,
            )
            for group in LORA_GROUPS:
                per_group_values[group].append(fractions[group])
        prefix.update(rounds.to_bytes(8, byteorder="little", signed=False))
        prefix_digest = prefix.hexdigest()
        if model_summary.get("round_shard_prefix_sha256") != prefix_digest:
            raise RuntimeError(f"baseline {model} round-shard prefix mismatch")
        _validate_model_totals(
            model_summary,
            model=model,
            rounds=rounds,
            clients_per_round=clients_per_round,
            values=per_group_values,
        )
        confirmed_prefix = round_shard_prefix_sha256(rounds_dir, rounds)
        if confirmed_prefix != prefix_digest:
            raise RuntimeError(
                f"baseline {model} round shards changed during calibration"
            )
        if _read_private_bytes(
            model_summary_path, f"baseline {model} final summary"
        ) != model_summary_bytes:
            raise RuntimeError(
                f"baseline {model} final summary changed during calibration"
            )
        calibration_models[model] = {
            "rounds": rounds,
            "clients_per_round": clients_per_round,
            "model_final_summary_sha256": _sha256_bytes(model_summary_bytes),
            "round_shard_prefix_sha256": confirmed_prefix,
            "groups": {
                group: {
                    "reducer": CALIBRATION_REDUCER,
                    "round_actual_clipped_fractions": per_group_values[group],
                    "round_count": len(per_group_values[group]),
                    "round_actual_clipped_fractions_sha256": canonical_json_fingerprint(
                        per_group_values[group]
                    ),
                    "target_clip_fraction": float(
                        statistics.median(per_group_values[group])
                    ),
                }
                for group in LORA_GROUPS
            },
        }

    if _read_private_bytes(run_config_path, "baseline run config") != run_config_bytes:
        raise RuntimeError("baseline run config changed during calibration")
    if (
        _read_private_bytes(root_summary_path, "baseline root summary")
        != root_summary_bytes
    ):
        raise RuntimeError("baseline root summary changed during calibration")

    repository_sha = contract.get("repository_sha")
    if (
        not isinstance(repository_sha, str)
        or len(repository_sha) != 40
        or any(character not in "0123456789abcdef" for character in repository_sha)
    ):
        raise RuntimeError("baseline repository SHA is invalid")
    core = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": CALIBRATION_STATUS,
        "privacy_class": CALIBRATION_PRIVACY_CLASS,
        "source": {
            "method": SOURCE_METHOD,
            "baseline_dir": str(source_dir),
            "run_config_fingerprint": run_fingerprint,
            "run_config_sha256": _sha256_bytes(run_config_bytes),
            "root_final_summary_sha256": _sha256_bytes(root_summary_bytes),
            "repository_sha": repository_sha,
        },
        "reducer": CALIBRATION_REDUCER,
        "models": calibration_models,
        "created_at_utc": utc_now(),
    }
    return {
        **core,
        "calibration_fingerprint": canonical_json_fingerprint(core),
    }


def _calibration_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key != "calibration_fingerprint"
    }


def _calibration_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"calibration_fingerprint", "created_at_utc"}
    }


def load_and_validate_calibration(
    path: str | os.PathLike[str],
    *,
    expected_models: Sequence[str] | None = None,
    expected_source_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load a private calibration file and validate all derived values."""

    calibration_path = _absolute_path(path)
    _validate_private_directory(calibration_path.parent, "calibration parent directory")
    value, _ = _load_private_object(calibration_path, "SlaClip calibration")
    expected_top_level = {
        "schema_version",
        "status",
        "privacy_class",
        "source",
        "reducer",
        "models",
        "calibration_fingerprint",
        "created_at_utc",
    }
    if set(value) != expected_top_level:
        raise RuntimeError("SlaClip calibration top-level schema mismatch")
    identity = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": CALIBRATION_STATUS,
        "privacy_class": CALIBRATION_PRIVACY_CLASS,
        "reducer": CALIBRATION_REDUCER,
    }
    for key, expected in identity.items():
        if value.get(key) != expected:
            raise RuntimeError(f"SlaClip calibration identity mismatch: {key}")
    timestamp = value.get("created_at_utc")
    if not isinstance(timestamp, str) or not timestamp:
        raise RuntimeError("SlaClip calibration timestamp is invalid")
    source = value.get("source")
    expected_source_keys = {
        "method",
        "baseline_dir",
        "run_config_fingerprint",
        "run_config_sha256",
        "root_final_summary_sha256",
        "repository_sha",
    }
    if not isinstance(source, dict) or set(source) != expected_source_keys:
        raise RuntimeError("SlaClip calibration source schema mismatch")
    if source.get("method") != SOURCE_METHOD:
        raise RuntimeError("SlaClip calibration source method mismatch")
    stored_source_dir = source.get("baseline_dir")
    if not isinstance(stored_source_dir, str) or not Path(stored_source_dir).is_absolute():
        raise RuntimeError("SlaClip calibration source directory is invalid")
    if expected_source_dir is not None:
        expected_source = _absolute_path(expected_source_dir).resolve(strict=True)
        if Path(stored_source_dir) != expected_source:
            raise RuntimeError("SlaClip calibration source directory mismatch")
    for name in (
        "run_config_fingerprint",
        "run_config_sha256",
        "root_final_summary_sha256",
    ):
        _require_sha256(source.get(name), f"SlaClip calibration source {name}")
    repository_sha = source.get("repository_sha")
    if (
        not isinstance(repository_sha, str)
        or len(repository_sha) != 40
        or any(character not in "0123456789abcdef" for character in repository_sha)
    ):
        raise RuntimeError("SlaClip calibration repository SHA is invalid")

    models_value = value.get("models")
    models = tuple(expected_models) if expected_models is not None else DEFAULT_MODELS
    if not isinstance(models_value, dict) or tuple(models_value) != models:
        raise RuntimeError("SlaClip calibration model set/order mismatch")
    for model in models:
        model_value = models_value.get(model)
        expected_model_keys = {
            "rounds",
            "clients_per_round",
            "model_final_summary_sha256",
            "round_shard_prefix_sha256",
            "groups",
        }
        if not isinstance(model_value, dict) or set(model_value) != expected_model_keys:
            raise RuntimeError(f"SlaClip calibration {model} schema mismatch")
        rounds = _require_positive_integer(
            model_value.get("rounds"), f"SlaClip calibration {model} rounds"
        )
        _require_positive_integer(
            model_value.get("clients_per_round"),
            f"SlaClip calibration {model} clients_per_round",
        )
        _require_sha256(
            model_value.get("model_final_summary_sha256"),
            f"SlaClip calibration {model} model summary",
        )
        _require_sha256(
            model_value.get("round_shard_prefix_sha256"),
            f"SlaClip calibration {model} round prefix",
        )
        groups = model_value.get("groups")
        if not isinstance(groups, dict) or tuple(groups) != LORA_GROUPS:
            raise RuntimeError(f"SlaClip calibration {model} group set/order mismatch")
        for group in LORA_GROUPS:
            group_value = groups[group]
            expected_group_keys = {
                "reducer",
                "round_actual_clipped_fractions",
                "round_count",
                "round_actual_clipped_fractions_sha256",
                "target_clip_fraction",
            }
            if not isinstance(group_value, dict) or set(group_value) != expected_group_keys:
                raise RuntimeError(f"SlaClip calibration {model}/{group} schema mismatch")
            if group_value.get("reducer") != CALIBRATION_REDUCER:
                raise RuntimeError(f"SlaClip calibration {model}/{group} reducer mismatch")
            raw_values = group_value.get("round_actual_clipped_fractions")
            if not isinstance(raw_values, list):
                raise RuntimeError(f"SlaClip calibration {model}/{group} values are invalid")
            fractions = [
                _require_fraction(item, f"SlaClip calibration {model}/{group} value")
                for item in raw_values
            ]
            if group_value.get("round_count") != rounds or len(fractions) != rounds:
                raise RuntimeError(f"SlaClip calibration {model}/{group} count mismatch")
            digest = canonical_json_fingerprint(fractions)
            if group_value.get("round_actual_clipped_fractions_sha256") != digest:
                raise RuntimeError(f"SlaClip calibration {model}/{group} values digest mismatch")
            target = _require_fraction(
                group_value.get("target_clip_fraction"),
                f"SlaClip calibration {model}/{group} target",
            )
            if target != float(statistics.median(fractions)):
                raise RuntimeError(f"SlaClip calibration {model}/{group} median mismatch")
    core = _calibration_core(value)
    expected_fingerprint = canonical_json_fingerprint(core)
    if value.get("calibration_fingerprint") != expected_fingerprint:
        raise RuntimeError("SlaClip calibration fingerprint mismatch")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_create_calibration(path: str | os.PathLike[str], value: Mapping[str, Any]) -> Path:
    """Publish one private calibration atomically without replacing a path."""

    destination = _absolute_path(path)
    _validate_private_directory(destination.parent, "calibration parent directory")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing calibration: {destination}")
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor: int | None = None
    linked = False
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
                raise OSError("failed to write SlaClip calibration")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _validate_private_file_metadata(
            temporary.lstat(), temporary, "temporary SlaClip calibration"
        )
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        temporary.unlink()
        _fsync_directory(destination.parent)
        _validate_private_file_metadata(
            destination.lstat(), destination, "SlaClip calibration"
        )
        return destination
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing calibration: {destination}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if os.path.lexists(temporary):
            temporary.unlink()
            _fsync_directory(destination.parent)
        if linked and not os.path.lexists(destination):
            raise RuntimeError("SlaClip calibration publication disappeared")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    calibration = build_calibration(args.baseline_dir)
    output = _absolute_path(args.output)
    if args.verify_existing:
        existing = load_and_validate_calibration(
            output,
            expected_models=DEFAULT_MODELS,
            expected_source_dir=args.baseline_dir,
        )
        if _calibration_semantics(existing) != _calibration_semantics(calibration):
            raise SystemExit("existing SlaClip calibration does not match baseline")
        print(
            json.dumps(
                {
                    "status": "verified",
                    "calibration_fingerprint": existing["calibration_fingerprint"],
                    "output": str(output),
                },
                indent=2,
            )
        )
        return
    try:
        atomic_create_calibration(output, calibration)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error
    load_and_validate_calibration(
        output,
        expected_models=DEFAULT_MODELS,
        expected_source_dir=args.baseline_dir,
    )
    print(
        json.dumps(
            {
                "status": CALIBRATION_STATUS,
                "calibration_fingerprint": calibration["calibration_fingerprint"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
