"""Small, runner-independent reproducibility and audit helpers.

This module deliberately has no dependency on the training runner.  It owns the
stable method contract, private run-key handling, deterministic seed derivation,
and JSON-safe scalar summaries that checkpoints and runners can share later.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import operator
import os
import secrets
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


PRIVATE_KEY_BYTES = 32
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_KEY_MODE = 0o600
SEED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MethodSpec:
    """Auditable behavior of one experiment arm.

    ``independently_accounted`` is intentionally false for every current arm.
    In particular, the paper-literal reconstruction must not be mistaken for a
    separately audited end-to-end privacy guarantee.
    """

    name: str
    clipping_enabled: bool
    gaussian_noise_enabled: bool
    is_control: bool
    independently_accounted: bool
    release_class: str


METHOD_SPECS: Mapping[str, MethodSpec] = MappingProxyType(
    {
        "no_dp_lora_control": MethodSpec(
            name="no_dp_lora_control",
            clipping_enabled=False,
            gaussian_noise_enabled=False,
            is_control=True,
            independently_accounted=False,
            release_class="non_private_control",
        ),
        "clip_only_control": MethodSpec(
            name="clip_only_control",
            clipping_enabled=True,
            gaussian_noise_enabled=False,
            is_control=True,
            independently_accounted=False,
            release_class="non_private_control",
        ),
        "paper_dp_lora": MethodSpec(
            name="paper_dp_lora",
            clipping_enabled=True,
            gaussian_noise_enabled=True,
            is_control=False,
            independently_accounted=False,
            release_class="paper_literal_dp_reconstruction",
        ),
    }
)


class PrivateKeyError(RuntimeError):
    """A key path failed a fail-closed ownership or permission check."""


def _path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.name:
        raise ValueError("private key path must name a file")
    return path


def _validate_private_parent(parent: Path, *, create: bool) -> None:
    if create:
        parent.mkdir(parents=True, mode=PRIVATE_DIRECTORY_MODE, exist_ok=True)
    try:
        metadata = parent.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PrivateKeyError(f"private key parent is not a real directory: {parent}")
    if metadata.st_uid != os.getuid():
        raise PrivateKeyError(f"private key parent is not owned by this user: {parent}")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != PRIVATE_DIRECTORY_MODE:
        raise PrivateKeyError(
            f"private key parent must have mode 0700, found {mode:04o}: {parent}"
        )


def _validate_open_key(descriptor: int, path: Path) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise PrivateKeyError(f"private key is not a regular file: {path}")
    if metadata.st_uid != os.getuid():
        raise PrivateKeyError(f"private key is not owned by this user: {path}")
    if metadata.st_nlink != 1:
        raise PrivateKeyError(f"private key must have exactly one hard link: {path}")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != PRIVATE_KEY_MODE:
        raise PrivateKeyError(
            f"private key must have mode 0600, found {mode:04o}: {path}"
        )


def _open_flags(base: int) -> int:
    return base | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def load_private_key(path: str | os.PathLike[str]) -> bytes:
    """Load an exactly 32-byte key, rejecting unsafe paths and permissions."""

    key_path = _path(path)
    _validate_private_parent(key_path.parent, create=False)
    try:
        descriptor = os.open(key_path, _open_flags(os.O_RDONLY))
    except OSError as error:
        if error.errno in {
            getattr(os, "ELOOP", -1),
            getattr(os, "EMLINK", -1),
        }:
            raise PrivateKeyError(f"private key may not be a symlink: {key_path}") from error
        raise
    try:
        _validate_open_key(descriptor, key_path)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            key = handle.read(PRIVATE_KEY_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(key) != PRIVATE_KEY_BYTES:
        raise PrivateKeyError(
            f"private key must contain exactly {PRIVATE_KEY_BYTES} bytes: {key_path}"
        )
    return key


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_private_key(path: str | os.PathLike[str]) -> bytes:
    """Atomically create a new private 32-byte key without overwriting a file."""

    key_path = _path(path)
    _validate_private_parent(key_path.parent, create=True)
    key = secrets.token_bytes(PRIVATE_KEY_BYTES)
    descriptor = os.open(
        key_path,
        _open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
        PRIVATE_KEY_MODE,
    )
    try:
        os.fchmod(descriptor, PRIVATE_KEY_MODE)
        _validate_open_key(descriptor, key_path)
        pending = memoryview(key)
        while pending:
            written = os.write(descriptor, pending)
            if written <= 0:
                raise OSError("failed to write private key")
            pending = pending[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(key_path.parent)
    return key


def load_or_create_private_key(path: str | os.PathLike[str]) -> bytes:
    """Load a valid private key or securely create it once.

    A concurrent creator is tolerated; unsafe permissions or malformed content
    are never repaired implicitly and always fail closed.
    """

    key_path = _path(path)
    _validate_private_parent(key_path.parent, create=True)
    try:
        return load_private_key(key_path)
    except FileNotFoundError:
        try:
            return create_private_key(key_path)
        except FileExistsError:
            return load_private_key(key_path)


def _require_seed_text(label: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_nonnegative_integer(label: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not bool")
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    if integer < 0:
        raise ValueError(f"{label} must be non-negative")
    return integer


def derive_seed(
    key: bytes,
    domain: str,
    purpose: str,
    model: str,
    round: int,
    client: int,
    method_scope: str | None = None,
) -> int:
    """Derive a stateless, non-negative 63-bit seed with HMAC-SHA256.

    Omit ``method_scope`` for randomness intentionally shared across experiment
    arms (for example client sampling), and provide an arm-specific scope for
    randomness that must be independent (for example injected noise).
    """

    if not isinstance(key, bytes) or len(key) != PRIVATE_KEY_BYTES:
        raise ValueError(f"key must be exactly {PRIVATE_KEY_BYTES} bytes")
    if method_scope is not None:
        method_scope = _require_seed_text("method_scope", method_scope)
    context = {
        "client": _require_nonnegative_integer("client", client),
        "domain": _require_seed_text("domain", domain),
        "method_scope": method_scope,
        "model": _require_seed_text("model", model),
        "purpose": _require_seed_text("purpose", purpose),
        "round": _require_nonnegative_integer("round", round),
        "schema_version": SEED_SCHEMA_VERSION,
    }
    encoded = _canonical_json_bytes(context)
    digest = hmac.new(key, encoded, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not permit NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            normalized[key] = _normalize_json(item)
        return normalized
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_fingerprint(value: Any) -> str:
    """Return a SHA-256 hex digest of deterministic, compact JSON."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def private_key_fingerprint(key: bytes) -> str:
    """Return a non-secret identifier for an exactly 32-byte run key."""

    if not isinstance(key, bytes) or len(key) != PRIVATE_KEY_BYTES:
        raise ValueError(f"key must be exactly {PRIVATE_KEY_BYTES} bytes")
    return hashlib.sha256(key).hexdigest()


def int64_index_digest(indices: Iterable[int]) -> str:
    """Digest an ordered sequence using a platform-independent int64 encoding."""

    digest = hashlib.sha256(b"dp-lora-index-vector-v1\0")
    count = 0
    for raw_value in indices:
        if isinstance(raw_value, bool):
            raise TypeError("boolean is not a valid index")
        try:
            value = operator.index(raw_value)
        except TypeError as error:
            raise TypeError("every index must be an integer") from error
        if value < -(1 << 63) or value >= (1 << 63):
            raise OverflowError(f"index is outside signed int64 range: {value}")
        digest.update(struct.pack("<q", value))
        count += 1
    digest.update(struct.pack("<Q", count))
    return digest.hexdigest()


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    """Return a finite ratio, or ``None`` for invalid/non-finite inputs."""

    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numerator_value) or not math.isfinite(denominator_value):
        return None
    if denominator_value == 0.0:
        return None
    ratio = numerator_value / denominator_value
    return ratio if math.isfinite(ratio) else None


def _quantile_label(probability: float) -> str:
    return format(probability, ".15g")


def safe_quantiles(
    values: Iterable[Any],
    probabilities: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
) -> dict[str, Any]:
    """Summarize finite values with deterministic linear-interpolated quantiles.

    Invalid and non-finite observations are counted but excluded.  Empty finite
    input yields ``None`` for each quantile, keeping the result strict-JSON safe.
    """

    requested: list[float] = []
    labels: set[str] = set()
    for raw_probability in probabilities:
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("quantile probabilities must be finite numbers") from error
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("quantile probabilities must lie in [0, 1]")
        label = _quantile_label(probability)
        if label in labels:
            raise ValueError(f"duplicate quantile probability: {probability}")
        labels.add(label)
        requested.append(probability)

    total_count = 0
    finite: list[float] = []
    for raw_value in values:
        total_count += 1
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            finite.append(value)
    finite.sort()

    quantiles: dict[str, float | None] = {}
    for probability in requested:
        label = _quantile_label(probability)
        if not finite:
            quantiles[label] = None
            continue
        position = probability * (len(finite) - 1)
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        if lower_index == upper_index:
            quantiles[label] = finite[lower_index]
        else:
            fraction = position - lower_index
            quantiles[label] = (
                finite[lower_index] * (1.0 - fraction)
                + finite[upper_index] * fraction
            )
    return {
        "count": total_count,
        "finite_count": len(finite),
        "non_finite_count": total_count - len(finite),
        "quantiles": quantiles,
    }


__all__ = [
    "METHOD_SPECS",
    "PRIVATE_KEY_BYTES",
    "MethodSpec",
    "PrivateKeyError",
    "canonical_json_fingerprint",
    "create_private_key",
    "derive_seed",
    "int64_index_digest",
    "load_or_create_private_key",
    "load_private_key",
    "private_key_fingerprint",
    "safe_quantiles",
    "safe_ratio",
]
