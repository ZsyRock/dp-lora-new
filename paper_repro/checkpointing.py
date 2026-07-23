"""Fail-closed, pickle-free checkpoints for the DP-LoRA reconstruction.

Only complete federated rounds are checkpointed.  Tensor payloads use
``safetensors`` and trainer metadata uses JSON so a resume never executes
serialized Python objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class LoadedCheckpoint:
    completed_round: int
    tensors: dict[str, torch.Tensor]
    trainer_state: dict[str, Any]
    directory: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_private_path(path: Path, *, directory: bool, mode: int) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if stat.S_ISLNK(metadata.st_mode) or not expected_type(metadata.st_mode):
        kind = "directory" if directory else "regular file"
        raise RuntimeError(f"private checkpoint path is not a real {kind}: {path}")
    if metadata.st_uid != os.getuid():
        raise RuntimeError(f"private checkpoint path is not user-owned: {path}")
    actual_mode = stat.S_IMODE(metadata.st_mode)
    if actual_mode != mode:
        raise RuntimeError(
            f"private checkpoint path mode mismatch: {actual_mode:04o} != {mode:04o}: {path}"
        )
    return metadata


def _validate_private_metadata(
    metadata: os.stat_result,
    path: Path,
    *,
    directory: bool,
    mode: int,
) -> None:
    """Validate metadata obtained from an already-open path.

    Archiving uses directory-relative descriptors after the initial ``lstat``
    checks.  Revalidating ``fstat``/``stat(..., follow_symlinks=False)`` results
    keeps a concurrent path replacement from turning a checked private path
    into an unchecked one.
    """

    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if stat.S_ISLNK(metadata.st_mode) or not expected_type(metadata.st_mode):
        kind = "directory" if directory else "regular file"
        raise RuntimeError(f"private checkpoint path is not a real {kind}: {path}")
    if metadata.st_uid != os.getuid():
        raise RuntimeError(f"private checkpoint path is not user-owned: {path}")
    actual_mode = stat.S_IMODE(metadata.st_mode)
    if actual_mode != mode:
        raise RuntimeError(
            f"private checkpoint path mode mismatch: {actual_mode:04o} != {mode:04o}: {path}"
        )


def _open_private_directory(path: Path) -> int:
    _validate_private_path(path, directory=True, mode=0o700)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        _validate_private_metadata(
            os.fstat(descriptor), path, directory=True, mode=0o700
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _copy_private_file(
    *,
    source_directory_fd: int,
    source_path: Path,
    destination_directory_fd: int,
    destination_path: Path,
) -> os.stat_result:
    """Durably copy one private shard without following either endpoint."""

    source_name = source_path.name
    destination_name = destination_path.name
    source_metadata = os.stat(
        source_name, dir_fd=source_directory_fd, follow_symlinks=False
    )
    _validate_private_metadata(
        source_metadata, source_path, directory=False, mode=0o600
    )
    source_descriptor = os.open(
        source_name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=source_directory_fd,
    )
    destination_descriptor: int | None = None
    try:
        opened_source_metadata = os.fstat(source_descriptor)
        _validate_private_metadata(
            opened_source_metadata, source_path, directory=False, mode=0o600
        )
        if (
            opened_source_metadata.st_dev,
            opened_source_metadata.st_ino,
        ) != (source_metadata.st_dev, source_metadata.st_ino):
            raise RuntimeError(
                f"round diagnostic changed during validation: {source_path}"
            )
        destination_descriptor = os.open(
            destination_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=destination_directory_fd,
        )
        # A restrictive process umask may narrow the requested mode; restore
        # the exact private-file contract before the copy becomes durable.
        os.fchmod(destination_descriptor, 0o600)
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError(f"failed to copy round diagnostic: {source_path}")
                view = view[written:]
        finished_source_metadata = os.fstat(source_descriptor)
        _validate_private_metadata(
            finished_source_metadata, source_path, directory=False, mode=0o600
        )
        initial_identity = (
            opened_source_metadata.st_dev,
            opened_source_metadata.st_ino,
            opened_source_metadata.st_size,
            opened_source_metadata.st_mtime_ns,
            opened_source_metadata.st_ctime_ns,
        )
        finished_identity = (
            finished_source_metadata.st_dev,
            finished_source_metadata.st_ino,
            finished_source_metadata.st_size,
            finished_source_metadata.st_mtime_ns,
            finished_source_metadata.st_ctime_ns,
        )
        if finished_identity != initial_identity:
            raise RuntimeError(f"round diagnostic changed while copying: {source_path}")
        destination_metadata = os.fstat(destination_descriptor)
        if destination_metadata.st_size != finished_source_metadata.st_size:
            raise RuntimeError(f"round diagnostic copy is incomplete: {source_path}")
        os.fsync(destination_descriptor)
        _validate_private_metadata(
            destination_metadata,
            destination_path,
            directory=False,
            mode=0o600,
        )
        return finished_source_metadata
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)


def _prepare_checkpoint_root(checkpoint_root: Path) -> None:
    if os.path.lexists(checkpoint_root):
        _validate_private_path(checkpoint_root, directory=True, mode=0o700)
        return
    checkpoint_root.mkdir(parents=True, mode=0o700)
    _validate_private_path(checkpoint_root, directory=True, mode=0o700)


def write_checkpoint(
    checkpoint_root: Path,
    *,
    completed_round: int,
    tensors: Mapping[str, torch.Tensor],
    trainer_state: Mapping[str, Any],
    config_fingerprint: str,
) -> Path:
    """Atomically commit one complete-round checkpoint.

    Existing checkpoint directories are never overwritten.  The ``latest``
    pointer is updated only after both payload files are durable.
    """

    if completed_round <= 0:
        raise ValueError("completed_round must be positive")
    _prepare_checkpoint_root(checkpoint_root)
    name = f"round-{completed_round:05d}"
    destination = checkpoint_root / name
    if os.path.lexists(destination):
        raise FileExistsError(f"checkpoint already exists: {destination}")
    temporary = checkpoint_root / f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700)
    tensor_path = temporary / "adapter_state.safetensors"
    state_path = temporary / "trainer_state.json"
    try:
        contiguous = {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in tensors.items()
        }
        if not contiguous:
            raise ValueError("checkpoint must contain at least one tensor")
        non_finite = [
            name
            for name, tensor in contiguous.items()
            if not bool(torch.isfinite(tensor).all())
        ]
        if non_finite:
            raise FloatingPointError(
                f"checkpoint tensors contain non-finite values: {non_finite[:5]}"
            )
        save_file(
            contiguous,
            tensor_path,
            metadata={
                "config_fingerprint": config_fingerprint,
                "completed_round": str(completed_round),
            },
        )
        os.chmod(tensor_path, 0o600)
        with tensor_path.open("rb") as handle:
            os.fsync(handle.fileno())
        tensor_sha = _sha256(tensor_path)
        payload = dict(trainer_state)
        payload.update(
            {
                "schema_version": 1,
                "completed_round": completed_round,
                "config_fingerprint": config_fingerprint,
                "adapter_state_sha256": tensor_sha,
            }
        )
        _atomic_json(state_path, payload)
        state_sha = _sha256(state_path)
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(checkpoint_root)
        _atomic_json(
            checkpoint_root / "latest.json",
            {
                "schema_version": 1,
                "completed_round": completed_round,
                "directory": name,
                "config_fingerprint": config_fingerprint,
                "adapter_state_sha256": tensor_sha,
                "trainer_state_sha256": state_sha,
            },
        )
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    _validate_private_path(path, directory=False, mode=0o600)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def _inspect_checkpoint_directory(
    directory: Path, *, expected_config_fingerprint: str
) -> tuple[int, dict[str, Any], str, str]:
    _validate_private_path(directory, directory=True, mode=0o700)
    name = directory.name
    if len(name) != len("round-00000") or not name.startswith("round-"):
        raise RuntimeError(f"invalid checkpoint directory name: {directory}")
    try:
        name_round = int(name.removeprefix("round-"))
    except ValueError as error:
        raise RuntimeError(f"invalid checkpoint directory name: {directory}") from error
    if name_round <= 0 or name != f"round-{name_round:05d}":
        raise RuntimeError(f"invalid checkpoint directory round: {directory}")
    tensor_path = directory / "adapter_state.safetensors"
    state_path = directory / "trainer_state.json"
    _validate_private_path(tensor_path, directory=False, mode=0o600)
    trainer_state = _load_json_object(state_path, "checkpoint trainer state")
    if trainer_state.get("schema_version") != 1:
        raise RuntimeError("unsupported checkpoint trainer-state schema")
    if trainer_state.get("completed_round") != name_round:
        raise RuntimeError("checkpoint directory/round metadata mismatch")
    if trainer_state.get("config_fingerprint") != expected_config_fingerprint:
        raise RuntimeError("checkpoint trainer-state fingerprint mismatch")
    tensor_sha = _sha256(tensor_path)
    if trainer_state.get("adapter_state_sha256") != tensor_sha:
        raise RuntimeError("checkpoint tensor checksum metadata mismatch")
    try:
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            tensor_names = list(handle.keys())
    except Exception as error:
        raise RuntimeError(f"invalid checkpoint safetensors: {tensor_path}") from error
    if not tensor_names:
        raise RuntimeError("checkpoint safetensors contains no tensors")
    if metadata.get("config_fingerprint") != expected_config_fingerprint:
        raise RuntimeError("checkpoint safetensors fingerprint mismatch")
    if metadata.get("completed_round") != str(name_round):
        raise RuntimeError("checkpoint safetensors round mismatch")
    return name_round, trainer_state, tensor_sha, _sha256(state_path)


def _checkpoint_pointer_payload(
    *,
    completed_round: int,
    directory: Path,
    config_fingerprint: str,
    tensor_sha: str,
    state_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "completed_round": completed_round,
        "directory": directory.name,
        "config_fingerprint": config_fingerprint,
        "adapter_state_sha256": tensor_sha,
        "trainer_state_sha256": state_sha,
    }


def load_latest_checkpoint(
    checkpoint_root: Path, *, expected_config_fingerprint: str
) -> LoadedCheckpoint | None:
    """Load the newest committed checkpoint, promoting a durable orphan.

    A process can die after atomically renaming a complete round directory but
    before updating ``latest.json``.  Every committed round directory is
    therefore validated and the newest one is promoted before training resumes.
    """

    if not os.path.lexists(checkpoint_root):
        return None
    _validate_private_path(checkpoint_root, directory=True, mode=0o700)
    candidates: list[tuple[int, Path, dict[str, Any], str, str]] = []
    for path in sorted(checkpoint_root.iterdir()):
        if path.name.startswith(".round-"):
            # An interrupted pre-commit temporary directory is never trusted.
            continue
        if path.name.startswith("round-"):
            completed_round, state, tensor_sha, state_sha = (
                _inspect_checkpoint_directory(
                    path, expected_config_fingerprint=expected_config_fingerprint
                )
            )
            candidates.append(
                (completed_round, path, state, tensor_sha, state_sha)
            )
    pointer_path = checkpoint_root / "latest.json"
    pointer: dict[str, Any] | None = None
    if os.path.lexists(pointer_path):
        pointer = _load_json_object(pointer_path, "checkpoint pointer")
        if pointer.get("schema_version") != 1:
            raise RuntimeError("unsupported checkpoint pointer schema")
        if pointer.get("config_fingerprint") != expected_config_fingerprint:
            raise RuntimeError("checkpoint configuration fingerprint mismatch")
        pointer_round = pointer.get("completed_round")
        pointer_directory = pointer.get("directory")
        if (
            not isinstance(pointer_round, int)
            or pointer_round <= 0
            or pointer_directory != f"round-{pointer_round:05d}"
        ):
            raise RuntimeError("checkpoint pointer round/directory mismatch")
        pointed = [item for item in candidates if item[0] == pointer_round]
        if len(pointed) != 1:
            raise RuntimeError("checkpoint pointer target is missing or ambiguous")
        _, _, _, pointed_tensor_sha, pointed_state_sha = pointed[0]
        if pointer.get("adapter_state_sha256") != pointed_tensor_sha:
            raise RuntimeError("checkpoint pointer tensor checksum mismatch")
        if pointer.get("trainer_state_sha256") != pointed_state_sha:
            raise RuntimeError("checkpoint pointer trainer-state checksum mismatch")
    if not candidates:
        if pointer is not None:
            raise RuntimeError("checkpoint pointer exists without a committed round")
        return None
    candidates.sort(key=lambda item: item[0])
    completed_round, directory, trainer_state, tensor_sha, state_sha = candidates[-1]
    expected_pointer = _checkpoint_pointer_payload(
        completed_round=completed_round,
        directory=directory,
        config_fingerprint=expected_config_fingerprint,
        tensor_sha=tensor_sha,
        state_sha=state_sha,
    )
    if pointer != expected_pointer:
        _atomic_json(pointer_path, expected_pointer)
    tensor_path = directory / "adapter_state.safetensors"
    tensors = load_file(tensor_path, device="cpu")
    non_finite = [
        name
        for name, tensor in tensors.items()
        if not bool(torch.isfinite(tensor).all())
    ]
    if non_finite:
        raise RuntimeError(
            f"checkpoint tensors contain non-finite values: {non_finite[:5]}"
        )
    return LoadedCheckpoint(
        completed_round=completed_round,
        tensors=tensors,
        trainer_state=trainer_state,
        directory=directory,
    )


def archive_round_shards_after(
    rounds_directory: Path, *, completed_round: int
) -> list[Path]:
    """Archive diagnostics newer than a resume checkpoint without data loss.

    Shards are copied into a private staging directory and made durable before
    the complete archive is atomically published.  Only then are the originals
    unlinked.  A crash can therefore leave duplicate shards, but never remove
    the only durable copy.
    """

    if completed_round < 0:
        raise ValueError("completed_round must be non-negative")
    if not os.path.lexists(rounds_directory):
        return []

    # The helper derives archive destinations from ``rounds_directory.parent``.
    # Reject a symlink in any ancestor before using that lexical parent, even
    # when the final ``rounds`` component itself is a real private directory.
    lexical_absolute = Path(os.path.abspath(os.fspath(rounds_directory)))
    try:
        resolved = rounds_directory.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"cannot resolve round diagnostics path safely: {rounds_directory}"
        ) from error
    if resolved != lexical_absolute:
        raise RuntimeError(
            f"round diagnostics path contains a symlink component: {rounds_directory}"
        )
    _validate_private_path(
        rounds_directory.parent, directory=True, mode=0o700
    )

    rounds_descriptor = _open_private_directory(rounds_directory)
    try:
        stale: list[Path] = []
        for name in sorted(os.listdir(rounds_descriptor)):
            if not (name.startswith("round-") and name.endswith(".json")):
                continue
            path = rounds_directory / name
            try:
                round_number = int(name[len("round-") : -len(".json")])
            except ValueError as error:
                raise RuntimeError(
                    f"invalid round diagnostic filename: {path}"
                ) from error
            if round_number <= 0 or name != f"round-{round_number:05d}.json":
                raise RuntimeError(f"invalid round diagnostic filename: {path}")
            metadata = os.stat(
                name, dir_fd=rounds_descriptor, follow_symlinks=False
            )
            _validate_private_metadata(
                metadata, path, directory=False, mode=0o600
            )
            if round_number > completed_round:
                stale.append(path)
        if not stale:
            return []

        superseded = rounds_directory.parent / "superseded"
        if os.path.lexists(superseded):
            _validate_private_path(superseded, directory=True, mode=0o700)
        else:
            try:
                superseded.mkdir(mode=0o700)
            except FileExistsError:
                # A concurrent creator is accepted only if it satisfies the
                # same fail-closed private-directory contract.
                pass
            _validate_private_path(superseded, directory=True, mode=0o700)
        superseded_descriptor = _open_private_directory(superseded)
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive_name = (
                f"after-round-{completed_round:05d}-{timestamp}-"
                f"{uuid.uuid4().hex[:8]}"
            )
            staging_name = f".{archive_name}.{uuid.uuid4().hex}.tmp"
            os.mkdir(staging_name, mode=0o700, dir_fd=superseded_descriptor)
            staging_path = superseded / staging_name
            staging_descriptor = os.open(
                staging_name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=superseded_descriptor,
            )
            archive_published = False
            try:
                _validate_private_metadata(
                    os.fstat(staging_descriptor),
                    staging_path,
                    directory=True,
                    mode=0o700,
                )
                source_metadata: dict[str, os.stat_result] = {}
                for path in stale:
                    source_metadata[path.name] = _copy_private_file(
                        source_directory_fd=rounds_descriptor,
                        source_path=path,
                        destination_directory_fd=staging_descriptor,
                        destination_path=staging_path / path.name,
                    )
                os.fsync(staging_descriptor)
                try:
                    os.stat(
                        archive_name,
                        dir_fd=superseded_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    raise FileExistsError(
                        f"round diagnostic archive already exists: "
                        f"{superseded / archive_name}"
                    )
                os.rename(
                    staging_name,
                    archive_name,
                    src_dir_fd=superseded_descriptor,
                    dst_dir_fd=superseded_descriptor,
                )
                archive_published = True
                os.fsync(superseded_descriptor)

                archive = superseded / archive_name
                _validate_private_path(archive, directory=True, mode=0o700)
                archived_metadata: dict[str, os.stat_result] = {}
                for path in stale:
                    target = archive / path.name
                    archived_metadata[path.name] = _validate_private_path(
                        target, directory=False, mode=0o600
                    )
                    if (
                        archived_metadata[path.name].st_size
                        != source_metadata[path.name].st_size
                    ):
                        raise RuntimeError(
                            f"round diagnostic archive is incomplete: {target}"
                        )
                moved: list[Path] = []
                for path in stale:
                    current = os.stat(
                        path.name,
                        dir_fd=rounds_descriptor,
                        follow_symlinks=False,
                    )
                    _validate_private_metadata(
                        current, path, directory=False, mode=0o600
                    )
                    original = source_metadata[path.name]
                    current_identity = (
                        current.st_dev,
                        current.st_ino,
                        current.st_size,
                        current.st_mtime_ns,
                        current.st_ctime_ns,
                    )
                    original_identity = (
                        original.st_dev,
                        original.st_ino,
                        original.st_size,
                        original.st_mtime_ns,
                        original.st_ctime_ns,
                    )
                    if current_identity != original_identity:
                        raise RuntimeError(
                            f"round diagnostic changed before archival cleanup: {path}"
                        )
                    os.unlink(path.name, dir_fd=rounds_descriptor)
                    target = archive / path.name
                    moved.append(target)
                os.fsync(rounds_descriptor)
                return moved
            except BaseException:
                if not archive_published:
                    # Originals are still intact.  Remove only the private
                    # regular files created by this transaction, then the
                    # private staging directory itself.
                    for name in os.listdir(staging_descriptor):
                        staged_path = staging_path / name
                        metadata = os.stat(
                            name,
                            dir_fd=staging_descriptor,
                            follow_symlinks=False,
                        )
                        _validate_private_metadata(
                            metadata,
                            staged_path,
                            directory=False,
                            mode=0o600,
                        )
                        os.unlink(name, dir_fd=staging_descriptor)
                    os.fsync(staging_descriptor)
                raise
            finally:
                os.close(staging_descriptor)
                if not archive_published:
                    os.rmdir(staging_name, dir_fd=superseded_descriptor)
                    os.fsync(superseded_descriptor)
        finally:
            os.close(superseded_descriptor)
    finally:
        os.close(rounds_descriptor)
