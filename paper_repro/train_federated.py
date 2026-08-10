#!/usr/bin/env python3
"""Single-GPU reconstruction of Algorithm 1 from arXiv:2312.17493.

This runner intentionally implements the paper's *federated* DP-LoRA update,
not the separate centralized RoBERTa/SST-2 implementation in the sibling
worktree.  Five logical clients are simulated sequentially in one process.

The paper leaves several reproduction details unspecified.  Every such choice
is recorded in ``run_config.json`` and no independently calibrated epsilon is
claimed.  Exact client/batch losses and gradient norms are private diagnostics,
not differentially-private release artifacts.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import stat
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

try:
    from paper_repro.checkpointing import (
        archive_round_shards_after,
        load_latest_checkpoint,
        write_checkpoint,
    )
    from paper_repro.durable_io import (
        atomic_write_text,
        fsync_directory,
        fsync_fd,
    )
    from paper_repro.reproducibility import (
        METHOD_SPECS,
        canonical_json_fingerprint,
        derive_seed,
        int64_index_digest,
        load_private_key,
        load_or_create_private_key,
        private_key_fingerprint,
        safe_ratio,
        safe_quantiles,
    )
    from paper_repro.slaclip import (
        DEFAULT_BASE_TARGET_CLIPPED_FRACTION,
        MAX_ABS_LOG_STEP,
        automatic_num_slots,
        build_slack_vector,
        full_slaclip_update,
        normalize_noisy_slack,
        resolve_base_target_clipped_fraction,
    )
except ModuleNotFoundError:  # Support direct ``python paper_repro/...py`` use.
    from checkpointing import (  # type: ignore[no-redef]
        archive_round_shards_after,
        load_latest_checkpoint,
        write_checkpoint,
    )
    from durable_io import (  # type: ignore[no-redef]
        atomic_write_text,
        fsync_directory,
        fsync_fd,
    )
    from reproducibility import (  # type: ignore[no-redef]
        METHOD_SPECS,
        canonical_json_fingerprint,
        derive_seed,
        int64_index_digest,
        load_private_key,
        load_or_create_private_key,
        private_key_fingerprint,
        safe_ratio,
        safe_quantiles,
    )
    from slaclip import (  # type: ignore[no-redef]
        DEFAULT_BASE_TARGET_CLIPPED_FRACTION,
        MAX_ABS_LOG_STEP,
        automatic_num_slots,
        build_slack_vector,
        full_slaclip_update,
        normalize_noisy_slack,
        resolve_base_target_clipped_fraction,
    )


PRIVACY_LABEL = "NON_DP_PRIVATE_DIAGNOSTIC"
EXPECTED_DATASET_ID = "lighteval/med_dialog"
EXPECTED_DATASET_REVISION = "ce8a234c92ea9a37743ad8154253ba897a4a70a5"
EXPECTED_MODELS = {
    "bert": {
        "manifest_key": "bert-base-uncased",
        "repo_id": "google-bert/bert-base-uncased",
        "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
    },
    "gpt2": {
        "manifest_key": "gpt2",
        "repo_id": "openai-community/gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    },
}
EXPECTED_LORA_TARGETS = {
    "bert": ["query", "key", "value"],
    "gpt2": ["c_attn"],
}
SLACLIP_METHOD = "slaclip_dp_lora"
ORACLE_SLACLIP_METHOD = "oracle_slaclip_control"
NOISY_CONTROLLER_INPUT = "noisy_endpoints"
EXACT_CONTROLLER_INPUT = "exact_endpoints"
ADAPTIVE_METHODS = frozenset({SLACLIP_METHOD, ORACLE_SLACLIP_METHOD})
CONTROLLER_INPUT_BY_METHOD = {
    SLACLIP_METHOD: NOISY_CONTROLLER_INPUT,
    ORACLE_SLACLIP_METHOD: EXACT_CONTROLLER_INPUT,
}
# The historical full-SlaClip domain is immutable for backwards-compatible
# confirmation.  A paired oracle reuses it so the controller input is the only
# stochastic difference; an explicitly unpaired oracle gets its own domain.
SLACLIP_PAIRED_SLACK_NOISE_SCOPE = SLACLIP_METHOD
SLACLIP_REFERENCE_REPOSITORY = "https://github.com/ZsyRock/SlaClip"
SLACLIP_REFERENCE_REVISION = "d48b8e07aef33c58a3595ee18b4dccf9c75fa1f3"
GROUPWISE_SLACLIP_VARIANT = "groupwise_generalized_full_slaclip_beta"


def slaclip_slack_noise_method_scope(
    method: str, *, pair_noise_across_methods: bool
) -> str:
    if method == SLACLIP_METHOD:
        return SLACLIP_METHOD
    if method == ORACLE_SLACLIP_METHOD:
        return (
            SLACLIP_PAIRED_SLACK_NOISE_SCOPE
            if pair_noise_across_methods
            else ORACLE_SLACLIP_METHOD
        )
    raise ValueError(f"method has no SlaClip slack-noise domain: {method}")


def resolve_slaclip_base_targets(
    *,
    base_target_clipped_fraction: float | None = None,
    beta: float | None = None,
    base_target_clipped_fraction_by_group: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], bool]:
    """Resolve either the historical shared beta or canonical A/B betas.

    The scalar path intentionally retains the pre-groupwise resolution rules.
    Groupwise targets are a separate, explicit contract: both A and B must be
    present and cannot be combined with either scalar spelling.
    """

    if base_target_clipped_fraction_by_group is None:
        shared = resolve_base_target_clipped_fraction(
            base_target_clipped_fraction=base_target_clipped_fraction,
            beta=beta,
        )
        return {"A": shared, "B": shared}, False
    if base_target_clipped_fraction is not None or beta is not None:
        raise ValueError(
            "groupwise SlaClip base targets cannot be combined with a shared "
            "base_target_clipped_fraction or beta"
        )
    if set(base_target_clipped_fraction_by_group) != {"A", "B"}:
        raise ValueError(
            "groupwise SlaClip base targets must contain exactly A and B"
        )
    return (
        {
            group: resolve_base_target_clipped_fraction(
                base_target_clipped_fraction=(
                    base_target_clipped_fraction_by_group[group]
                )
            )
            for group in ("A", "B")
        },
        True,
    )


def slaclip_controller_target_arguments(
    controller: Mapping[str, Any],
) -> dict[str, Any]:
    """Return mutually exclusive target arguments from a persisted contract."""

    groupwise = controller.get("base_target_clipped_fraction_by_group")
    if groupwise is not None:
        # Re-resolve here so malformed persisted contracts fail closed before
        # any controller update or shard reconciliation.
        targets, _ = resolve_slaclip_base_targets(
            base_target_clipped_fraction_by_group=groupwise
        )
        return {"base_target_clipped_fraction_by_group": targets}
    return {
        "base_target_clipped_fraction": resolve_base_target_clipped_fraction(
            base_target_clipped_fraction=controller.get(
                "base_target_clipped_fraction"
            ),
            beta=controller.get("beta"),
        )
    }


@dataclass(frozen=True)
class EffectiveConfig:
    method: str
    num_clients: int
    rounds: int
    batch_size: int
    noise_multiplier: float
    learning_rate: float
    clip_norm: float
    rank: int
    max_seq_length: int
    seed: int
    data_split_seed: int
    evaluation_seed: int
    max_validation_records: int
    eval_every: int
    checkpoint_every: int
    data_protocol: str
    delta: float
    pair_noise_across_methods: bool
    smoke: bool
    # ``clip_norm`` is retained as the paper/shared-C compatibility field.
    # Formal groupwise-fixed and groupwise-adaptive campaigns populate the two
    # optional fields below; older callers that omit them continue to resolve
    # both groups to the shared threshold bit-for-bit.
    clip_norm_A: float | None = None
    clip_norm_B: float | None = None

    @property
    def clip_norm_by_group(self) -> dict[str, float]:
        return {
            "A": float(self.clip_norm if self.clip_norm_A is None else self.clip_norm_A),
            "B": float(self.clip_norm if self.clip_norm_B is None else self.clip_norm_B),
        }


@dataclass(frozen=True)
class LoadedData:
    training: "ParquetTextTable"
    validation: "ParquetTextTable"
    training_pool: np.ndarray
    validation_indices: np.ndarray
    protocol: dict[str, Any]


class GracefulStop(RuntimeError):
    """A scheduler stop request observed at a committed round boundary."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def validate_private_directory(path: Path, description: str = "private directory") -> None:
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


def validate_private_run_directory(path: Path) -> None:
    validate_private_directory(path, "run output")


def prepare_private_directory(
    path: Path, *, exist_ok: bool, description: str = "private directory"
) -> None:
    if os.path.lexists(path):
        if not exist_ok:
            raise FileExistsError(f"{description} already exists: {path}")
        validate_private_directory(path, description)
        return
    path.mkdir(mode=0o700)
    validate_private_directory(path, description)


def validate_private_regular_file(
    path: Path, description: str = "private file"
) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{description} is not a real regular file: {path}")
    if metadata.st_uid != os.getuid():
        raise RuntimeError(f"{description} is not owned by this user: {path}")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != 0o600:
        raise RuntimeError(
            f"{description} must have mode 0600, found {mode:04o}: {path}"
        )


def write_new_private_text(path: Path, value: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            fsync_fd(handle.fileno(), path=path, operation="fsync_private_text")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    validate_private_regular_file(path)
    fsync_directory(path.parent)


def acquire_run_lock(output_dir: Path) -> Any:
    """Acquire a non-blocking single-writer lock for one root run directory."""

    validate_private_run_directory(output_dir)
    lock_path = output_dir / ".run.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(lock_path, flags, 0o600)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise RuntimeError(f"unsafe run lock file: {lock_path}")
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"another writer already owns this run: {output_dir}") from error
    return handle


def atomic_json(path: Path, value: Any) -> None:
    if os.path.lexists(path):
        validate_private_regular_file(path, "existing JSON output")
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=json_default,
            allow_nan=False,
        )
        + "\n"
    )
    atomic_write_text(path, encoded, mode=0o600)


def atomic_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    if os.path.lexists(path):
        validate_private_regular_file(path, "existing JSONL output")
    encoded = "".join(
        json.dumps(
            value,
            sort_keys=True,
            default=json_default,
            allow_nan=False,
        )
        + "\n"
        for value in values
    )
    atomic_write_text(path, encoded, mode=0o600)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_adapter_state_sha256(tensors: dict[str, torch.Tensor]) -> str:
    """Hash LoRA state independent of PEFT's removable ``default`` namespace."""

    canonical: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        normalized_name = name.replace(".lora_A.default.", ".lora_A.").replace(
            ".lora_B.default.", ".lora_B."
        )
        if normalized_name in canonical:
            raise RuntimeError(f"duplicate canonical LoRA tensor name: {normalized_name}")
        canonical[normalized_name] = tensor
    if not canonical:
        raise RuntimeError("cannot fingerprint an empty adapter state")
    digest = hashlib.sha256(b"dp-lora-canonical-adapter-state-v1\0")
    for name in sorted(canonical):
        tensor = canonical[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
        name_bytes = name.encode("utf-8")
        shape_bytes = json.dumps(list(tensor.shape), separators=(",", ":")).encode(
            "ascii"
        )
        digest.update(len(name_bytes).to_bytes(8, "little"))
        digest.update(name_bytes)
        digest.update(len(shape_bytes).to_bytes(8, "little"))
        digest.update(shape_bytes)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_adapter_artifact(
    path: Path,
    *,
    expected_parameter_elements: int,
    expected_parameter_tensors: int,
    expected_rank: int,
    expected_target_modules: Sequence[str],
    expected_base_model_path: Path,
    expected_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed on a missing, corrupt, non-finite, or inert LoRA adapter."""

    from safetensors.torch import load_file

    try:
        validate_private_regular_file(path, "adapter artifact")
    except FileNotFoundError as error:
        raise RuntimeError(f"adapter tensor file is missing: {path}") from error
    config_path = path.with_name("adapter_config.json")
    try:
        validate_private_regular_file(config_path, "adapter artifact")
    except FileNotFoundError as error:
        raise RuntimeError(f"adapter configuration is missing: {config_path}") from error
    config = load_json_object(config_path, "adapter configuration")
    config_digest = sha256_file(config_path)
    if (
        expected_config_sha256 is not None
        and config_digest != expected_config_sha256
    ):
        raise RuntimeError("completed adapter configuration checksum mismatch")
    semantic_expectations = {
        "r": expected_rank,
        "lora_alpha": expected_rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "peft_type": "LORA",
        "base_model_name_or_path": str(expected_base_model_path),
    }
    for key, expected in semantic_expectations.items():
        if config.get(key) != expected:
            raise RuntimeError(f"adapter configuration mismatch: {key}")
    target_modules = config.get("target_modules")
    if not isinstance(target_modules, list) or sorted(target_modules) != sorted(
        expected_target_modules
    ):
        raise RuntimeError("adapter target modules do not match the run contract")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError("completed adapter checksum mismatch")
    try:
        tensors = load_file(path, device="cpu")
    except Exception as error:
        raise RuntimeError(f"adapter is not a readable safetensors artifact: {path}") from error
    if len(tensors) != expected_parameter_tensors:
        raise RuntimeError(
            "adapter tensor count mismatch: "
            f"{len(tensors)} != {expected_parameter_tensors}"
        )
    elements = sum(tensor.numel() for tensor in tensors.values())
    if elements != expected_parameter_elements:
        raise RuntimeError(
            f"adapter parameter count mismatch: {elements} != {expected_parameter_elements}"
        )
    groups: dict[str, list[torch.Tensor]] = {"A": [], "B": []}
    unexpected = []
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"adapter contains non-finite values: {name}")
        if ".lora_A." in name:
            groups["A"].append(tensor)
        elif ".lora_B." in name:
            groups["B"].append(tensor)
        else:
            unexpected.append(name)
    if unexpected or not groups["A"] or not groups["B"]:
        raise RuntimeError(
            f"adapter does not contain only non-empty LoRA A/B groups: {unexpected[:5]}"
        )
    norms = {
        group: math.sqrt(
            sum(float(tensor.detach().float().square().sum().item()) for tensor in values)
        )
        for group, values in groups.items()
    }
    if not all(math.isfinite(norm) and norm > 0 for norm in norms.values()):
        raise RuntimeError(f"adapter contains an inert or non-finite LoRA group: {norms}")
    return {
        "format": "safetensors",
        "sha256": digest,
        "config_sha256": config_digest,
        "canonical_state_sha256": canonical_adapter_state_sha256(tensors),
        "config_semantics_verified": True,
        "parameter_tensors": len(tensors),
        "parameter_elements": elements,
        "group_tensor_counts": {group: len(values) for group, values in groups.items()},
        "group_parameter_l2": norms,
        "all_finite": True,
        "both_groups_nonzero": True,
    }


def save_final_adapter_atomically(
    model: Any, output_dir: Path, *, resume: bool
) -> Path:
    """Commit the final PEFT adapter as one private, durable directory."""

    adapter_dir = output_dir / "final_adapter"
    if os.path.lexists(adapter_dir):
        validate_private_directory(adapter_dir, "existing final adapter directory")
        if not resume:
            raise FileExistsError(f"final adapter already exists: {adapter_dir}")
        # A final adapter is written only after the final-round checkpoint.
        # On resume, preserve the published candidate and let the caller bind
        # it to that checkpoint through the full artifact and canonical-state
        # validation immediately below this function.  Re-saving here would
        # create a failure window in which a valid published adapter vanished.
        return adapter_dir

    temporary = output_dir / f".final_adapter.{os.getpid()}.{time.time_ns()}.tmp"
    prepare_private_directory(
        temporary, exist_ok=False, description="temporary final adapter directory"
    )
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        for path in sorted(temporary.rglob("*")):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"generated adapter contains a symlink: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                os.chmod(path, 0o700)
                validate_private_directory(path, "generated adapter directory")
            elif stat.S_ISREG(metadata.st_mode):
                os.chmod(path, 0o600)
                validate_private_regular_file(path, "generated adapter file")
                with path.open("rb") as handle:
                    fsync_fd(
                        handle.fileno(),
                        path=path,
                        operation="fsync_final_adapter_file",
                    )
            else:
                raise RuntimeError(f"generated adapter contains an unsafe path: {path}")
        fsync_directory(temporary)
        os.replace(temporary, adapter_dir)
        fsync_directory(output_dir)
        validate_private_directory(adapter_dir, "final adapter directory")
        return adapter_dir
    except BaseException:
        # Preserve a failed write for diagnosis, but move it out of the
        # publication namespace so a later resume cannot mistake it for a
        # candidate final adapter.  Cleanup is deliberately best-effort so it
        # never hides the original save/validation exception.
        with contextlib.suppress(Exception):
            if os.path.lexists(temporary):
                metadata = temporary.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
                    metadata.st_mode
                ):
                    raise RuntimeError(
                        f"failed adapter staging path is not a real directory: {temporary}"
                    )
                if metadata.st_uid != os.getuid():
                    raise RuntimeError(
                        f"failed adapter staging path is not user-owned: {temporary}"
                    )
                os.chmod(temporary, 0o700)
                failed_root = output_dir / "failed_final_adapter_writes"
                prepare_private_directory(
                    failed_root,
                    exist_ok=True,
                    description="failed adapter write directory",
                )
                failed_destination = failed_root / (
                    f"failed-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
                    f"-pid{os.getpid()}-{time.time_ns()}"
                )
                if os.path.lexists(failed_destination):
                    raise RuntimeError(
                        f"refusing to overwrite a failed adapter write: {failed_destination}"
                    )
                os.replace(temporary, failed_destination)
                fsync_directory(failed_root)
                fsync_directory(output_dir)
        raise


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "datasets",
        "pyarrow",
        "numpy",
        "safetensors",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def cpu_model_name() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
    except OSError:
        pass
    value = platform.processor().strip()
    return value or None


def backend_is_available(name: str) -> bool | None:
    backend = getattr(torch.backends, name, None)
    checker = getattr(backend, "is_available", None)
    return bool(checker()) if callable(checker) else None


def cuda_driver_versions() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to identify the CUDA driver with nvidia-smi") from error
    versions = sorted({line.strip() for line in output.splitlines() if line.strip()})
    if not versions:
        raise RuntimeError("nvidia-smi returned no CUDA driver version")
    return versions


def configure_deterministic_execution(device_type: str) -> None:
    """Make numeric reproducibility an enforced protocol, not an aspiration."""

    if device_type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise RuntimeError(
            "CUDA deterministic execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def execution_backend_contract(device: torch.device) -> dict[str, Any]:
    torch_build = torch.__config__.show()
    contract: dict[str, Any] = {
        "device_type": device.type,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_build_config_sha256": hashlib.sha256(
            torch_build.encode("utf-8")
        ).hexdigest(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "platform_machine": platform.machine(),
        "cpu_model": cpu_model_name(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkl_available": backend_is_available("mkl"),
        "mkldnn_available": backend_is_available("mkldnn"),
        "openmp_available": backend_is_available("openmp"),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "numeric_environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        contract.update(
            {
                "cuda_driver_versions": cuda_driver_versions(),
                "gpu_name": properties.name,
                "gpu_compute_capability": [properties.major, properties.minor],
                "cudnn_version": torch.backends.cudnn.version(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            }
        )
    return contract


def repository_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(root), "branch", "--show-current"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
        return {"root": str(root), "sha": sha, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"root": str(root), "sha": None, "branch": None, "dirty": None}


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def load_private_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    validate_private_regular_file(path, description)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"invalid {description}: {path}") from error
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"invalid {description} JSON at line {line_number}: {path}"
            ) from error
        if not isinstance(value, dict):
            raise RuntimeError(
                f"{description} line {line_number} is not an object: {path}"
            )
        values.append(value)
    return values


def validate_input_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json_object(path, "input manifest")
    if manifest.get("status") != "STAGING_COMPLETE_VERIFIED":
        raise RuntimeError("input manifest is not marked STAGING_COMPLETE_VERIFIED")
    dataset = manifest.get("formal_dataset")
    models = manifest.get("models")
    if not isinstance(dataset, dict) or not isinstance(models, dict):
        raise RuntimeError("input manifest is missing dataset/model metadata")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("input manifest has no immutable file inventory")
    if canonical_json_fingerprint(inventory) != manifest.get("inventory_sha256"):
        raise RuntimeError("input manifest inventory fingerprint is invalid")
    if manifest.get("inventory_files") != len(inventory):
        raise RuntimeError("input manifest inventory file count is invalid")
    seen_paths: set[Path] = set()
    inventory_entries: dict[Path, dict[str, Any]] = {}
    dataset_inventory_paths: set[Path] = set()
    model_inventory_paths: dict[str, set[Path]] = {
        expected["manifest_key"]: set() for expected in EXPECTED_MODELS.values()
    }
    inventory_bytes = 0
    for item in inventory:
        if not isinstance(item, dict):
            raise RuntimeError("input manifest inventory entry is not an object")
        file_path = Path(str(item.get("path")))
        if not file_path.is_absolute():
            raise RuntimeError(f"input inventory path is not absolute: {file_path}")
        if file_path in seen_paths:
            raise RuntimeError(f"duplicate input inventory path: {file_path}")
        seen_paths.add(file_path)
        inventory_entries[file_path] = item
        role = item.get("role")
        if role == "formal_dataset":
            if item.get("model") is not None:
                raise RuntimeError(f"dataset inventory entry has a model tag: {file_path}")
            dataset_inventory_paths.add(file_path)
        elif role == "model":
            model_key = item.get("model")
            if model_key not in model_inventory_paths:
                raise RuntimeError(f"unknown model inventory tag: {model_key!r}")
            model_inventory_paths[str(model_key)].add(file_path)
        else:
            raise RuntimeError(f"unsupported input inventory role: {role!r}")
        if not file_path.is_file():
            raise RuntimeError(f"missing staged inventory file: {file_path}")
        expected_bytes = item.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise RuntimeError(f"invalid inventory byte count: {file_path}")
        actual_bytes = file_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(f"staged input size mismatch: {file_path}")
        expected_sha256 = item.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise RuntimeError(f"invalid inventory checksum: {file_path}")
        if sha256_file(file_path) != expected_sha256:
            raise RuntimeError(f"staged input checksum mismatch: {file_path}")
        inventory_bytes += actual_bytes
    if manifest.get("inventory_bytes") != inventory_bytes:
        raise RuntimeError("input manifest inventory byte total is invalid")
    if dataset.get("repo_id") != EXPECTED_DATASET_ID:
        raise RuntimeError(f"unexpected dataset: {dataset.get('repo_id')!r}")
    if dataset.get("revision") != EXPECTED_DATASET_REVISION:
        raise RuntimeError(f"unexpected dataset revision: {dataset.get('revision')!r}")
    combined = dataset.get("combined_splits")
    if not isinstance(combined, dict) or set(combined) != {
        "train",
        "validation",
        "test",
    }:
        raise RuntimeError("input manifest has no combined dataset splits")
    referenced_dataset_paths: set[Path] = set()
    for split in ("train", "validation", "test"):
        entry = combined.get(split)
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), list):
            raise RuntimeError(f"input manifest has no file list for {split}")
        if int(entry.get("rows", 0)) <= 0:
            raise RuntimeError(f"input manifest has no rows for {split}")
        for raw_path in entry["files"]:
            file_path = Path(str(raw_path))
            if file_path in referenced_dataset_paths:
                raise RuntimeError(f"dataset file is referenced more than once: {file_path}")
            referenced_dataset_paths.add(file_path)
            if not file_path.is_file():
                raise RuntimeError(f"missing staged {split} file: {file_path}")
    if referenced_dataset_paths != dataset_inventory_paths:
        missing = sorted(str(path) for path in dataset_inventory_paths - referenced_dataset_paths)
        unhashed = sorted(str(path) for path in referenced_dataset_paths - dataset_inventory_paths)
        raise RuntimeError(
            "dataset references do not exactly match the hashed inventory; "
            f"unreferenced={missing[:3]}, unhashed={unhashed[:3]}"
        )
    expected_model_keys = {
        expected["manifest_key"] for expected in EXPECTED_MODELS.values()
    }
    if set(models) != expected_model_keys:
        raise RuntimeError("input manifest model set is not exact")
    for model_name, expected in EXPECTED_MODELS.items():
        model_key = expected["manifest_key"]
        entry = models.get(model_key)
        if not isinstance(entry, dict):
            raise RuntimeError(f"input manifest is missing {model_name}")
        if entry.get("repo_id") != expected["repo_id"]:
            raise RuntimeError(f"{model_name} repo ID mismatch")
        if entry.get("revision") != expected["revision"]:
            raise RuntimeError(f"{model_name} revision mismatch")
        snapshot = Path(str(entry.get("snapshot_path")))
        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            raise RuntimeError(f"invalid {model_name} snapshot: {snapshot}")
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeError(f"input manifest has no file list for {model_name}")
        referenced_model_paths: set[Path] = set()
        for file_entry in files:
            if not isinstance(file_entry, dict):
                raise RuntimeError(f"invalid {model_name} file entry")
            file_path = Path(str(file_entry.get("path")))
            if file_path in referenced_model_paths:
                raise RuntimeError(f"duplicate {model_name} file reference: {file_path}")
            referenced_model_paths.add(file_path)
            if file_path.parent != snapshot:
                raise RuntimeError(f"{model_name} file escapes its pinned snapshot: {file_path}")
            inventory_entry = inventory_entries.get(file_path)
            if inventory_entry is None:
                raise RuntimeError(f"unhashed {model_name} file reference: {file_path}")
            for field in ("bytes", "sha256"):
                if file_entry.get(field) != inventory_entry.get(field):
                    raise RuntimeError(f"{model_name} file metadata mismatch: {file_path}")
        if referenced_model_paths != model_inventory_paths[model_key]:
            missing = sorted(
                str(path)
                for path in model_inventory_paths[model_key] - referenced_model_paths
            )
            unhashed = sorted(
                str(path)
                for path in referenced_model_paths - model_inventory_paths[model_key]
            )
            raise RuntimeError(
                f"{model_name} references do not exactly match the hashed inventory; "
                f"unreferenced={missing[:3]}, unhashed={unhashed[:3]}"
            )
        for required_name in ("config.json", "model.safetensors"):
            if snapshot / required_name not in referenced_model_paths:
                raise RuntimeError(
                    f"{model_name} inventory is missing required file: {required_name}"
                )
    return manifest


def deep_input_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    """Check schemas, model-config readability, and safetensors headers."""

    try:
        import pyarrow.parquet as pq
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("training runtime dependencies are incomplete") from error
    split_rows: dict[str, int] = {}
    for split, entry in manifest["formal_dataset"]["combined_splits"].items():
        rows = 0
        for raw_path in entry["files"]:
            parquet = pq.ParquetFile(Path(raw_path))
            names = set(parquet.schema_arrow.names)
            if not {"src", "tgt"}.issubset(names):
                raise RuntimeError(f"Parquet schema is missing src/tgt: {raw_path}")
            rows += parquet.metadata.num_rows
        if rows != int(entry["rows"]):
            raise RuntimeError(f"Parquet row count mismatch for {split}")
        split_rows[split] = rows
    model_headers: dict[str, Any] = {}
    for model_kind, expected in EXPECTED_MODELS.items():
        snapshot = model_snapshot(manifest, model_kind)
        config = load_json_object(snapshot / "config.json", "base-model config")
        expected_model_type = "bert" if model_kind == "bert" else "gpt2"
        if config.get("model_type") != expected_model_type:
            raise RuntimeError(f"base-model config type mismatch: {model_kind}")
        weight_path = snapshot / "model.safetensors"
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            tensor_names = list(handle.keys())
        if not tensor_names:
            raise RuntimeError(f"base-model safetensors has no tensors: {weight_path}")
        model_headers[model_kind] = {
            "model_type": config["model_type"],
            "weight_tensors": len(tensor_names),
            "repo_id": expected["repo_id"],
            "revision": expected["revision"],
        }
    missing_packages = [
        name for name, version in package_versions().items() if version is None
    ]
    if missing_packages:
        raise RuntimeError(f"training runtime packages are missing: {missing_packages}")
    return {
        "scope": "all_inventory_bytes_parquet_metadata_model_configs_and_weight_headers",
        "split_rows": split_rows,
        "models": model_headers,
        "runtime_packages_present": True,
    }


def manifest_summary(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    dataset = manifest["formal_dataset"]
    return {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "inventory_sha256": manifest.get("inventory_sha256"),
        "inventory_files": manifest.get("inventory_files"),
        "inventory_bytes": manifest.get("inventory_bytes"),
        "dataset": {
            "repo_id": dataset["repo_id"],
            "revision": dataset["revision"],
            "combined_splits": dataset["combined_splits"],
            "total_rows": dataset.get("total_rows"),
        },
        "models": {
            model: {
                "repo_id": manifest["models"][spec["manifest_key"]]["repo_id"],
                "revision": manifest["models"][spec["manifest_key"]]["revision"],
                "snapshot_path": manifest["models"][spec["manifest_key"]][
                    "snapshot_path"
                ],
            }
            for model, spec in EXPECTED_MODELS.items()
        },
    }


class ParquetTextTable:
    """In-memory Arrow table with deterministic indexed text access."""

    def __init__(self, paths: Sequence[Path]):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("pyarrow is required for the staged MedDialog data") from error
        tables = []
        for path in paths:
            table = pq.read_table(path, columns=["src", "tgt"])
            if table.column_names != ["src", "tgt"]:
                raise RuntimeError(f"unexpected Parquet schema in {path}")
            tables.append(table)
        self._pa = pa
        self._table = pa.concat_tables(tables).combine_chunks()

    def __len__(self) -> int:
        return self._table.num_rows

    def _normalized_rows(
        self, indices: Sequence[int] | np.ndarray
    ) -> list[tuple[str, str]]:
        index_array = self._pa.array([int(index) for index in indices], type=self._pa.int64())
        rows = self._table.take(index_array).to_pylist()
        normalized = []
        for row in rows:
            source = str(row.get("src") or "").strip()
            target = str(row.get("tgt") or "").strip()
            if not source and not target:
                raise RuntimeError("MedDialog record contains no text")
            normalized.append((source, target))
        return normalized

    def texts(self, indices: Sequence[int] | np.ndarray) -> list[str]:
        return [
            f"Patient: {source}\nDoctor: {target}"
            for source, target in self._normalized_rows(indices)
        ]

    def content_keys(self, indices: Sequence[int] | np.ndarray) -> list[str]:
        keys = []
        for source, target in self._normalized_rows(indices):
            encoded = json.dumps(
                [source, target], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            keys.append(hashlib.sha256(b"dp-lora-record-v1\0" + encoded).hexdigest())
        return keys


def load_data_protocol(
    manifest: dict[str, Any], config: EffectiveConfig
) -> LoadedData:
    splits = manifest["formal_dataset"]["combined_splits"]
    if config.data_protocol != "paper_union_minus_fixed_holdout":
        raise ValueError(f"unsupported data protocol: {config.data_protocol}")
    # Keep the closest public approximation to the paper's 257,332-dialogue
    # corpus, but remove the exact fixed diagnostic examples from training.
    # This makes the internal loss genuinely held out without pretending it is
    # one of the paper's downstream benchmark metrics.
    training_split_names = ("train", "validation", "test")
    training = ParquetTextTable(
        [
            Path(path)
            for split_name in training_split_names
            for path in splits[split_name]["files"]
        ]
    )
    validation = ParquetTextTable(
        [Path(path) for path in splits["validation"]["files"]]
    )
    expected_training_rows = sum(
        int(splits[split_name]["rows"]) for split_name in training_split_names
    )
    if len(training) != expected_training_rows:
        raise RuntimeError("loaded all-split training row count differs from manifest")
    if len(validation) != int(splits["validation"]["rows"]):
        raise RuntimeError("loaded validation row count differs from manifest")
    requested_validation_count = min(
        len(validation),
        config.max_validation_records,
        4 if config.smoke else len(validation),
    )
    validation_permutation = np.random.default_rng(config.data_split_seed).permutation(
        len(validation)
    )
    all_validation_indices = np.arange(len(validation), dtype=np.int64)
    validation_content_keys = validation.content_keys(all_validation_indices)
    selected_content_keys: set[str] = set()
    selected_validation_indices: list[int] = []
    for raw_index in validation_permutation:
        index = int(raw_index)
        content_key = validation_content_keys[index]
        if content_key in selected_content_keys:
            continue
        selected_content_keys.add(content_key)
        selected_validation_indices.append(index)
        if len(selected_validation_indices) == requested_validation_count:
            break
    if len(selected_validation_indices) != requested_validation_count:
        raise RuntimeError("validation split has too few unique normalized records")
    validation_indices = np.asarray(selected_validation_indices, dtype=np.int64)
    validation_offset = int(splits["train"]["rows"])
    held_out_global = validation_indices + validation_offset
    in_training_pool = np.ones(len(training), dtype=bool)
    content_excluded_rows = 0
    scan_chunk = 4096
    for offset in range(0, len(training), scan_chunk):
        stop = min(offset + scan_chunk, len(training))
        indices = np.arange(offset, stop, dtype=np.int64)
        keys = training.content_keys(indices)
        excluded = np.fromiter(
            (key in selected_content_keys for key in keys),
            dtype=bool,
            count=len(keys),
        )
        in_training_pool[offset:stop] = ~excluded
        content_excluded_rows += int(excluded.sum())
    training_pool = np.flatnonzero(in_training_pool).astype(np.int64, copy=False)
    if np.intersect1d(training_pool, held_out_global).size:
        raise RuntimeError("training and internal holdout indices overlap")
    if content_excluded_rows < len(validation_indices):
        raise RuntimeError("not every selected holdout record was found in the union")
    protocol = {
        "name": config.data_protocol,
        "paper_benchmark_metric": False,
        "training_source_splits": list(training_split_names),
        "training_rows_before_holdout": len(training),
        "training_rows_after_holdout": len(training_pool),
        "holdout_source_split": "validation",
        "data_split_seed": config.data_split_seed,
        "holdout_requested_rows": requested_validation_count,
        "holdout_rows": len(validation_indices),
        "holdout_unique_normalized_contents": len(selected_content_keys),
        "holdout_overlaps_training": False,
        "holdout_content_overlaps_training": False,
        "content_excluded_training_rows": content_excluded_rows,
        "duplicate_content_rows_excluded": content_excluded_rows
        - len(validation_indices),
        "training_pool_sha256": int64_index_digest(training_pool),
        "holdout_local_indices_sha256": int64_index_digest(validation_indices),
        "holdout_global_indices_sha256": int64_index_digest(held_out_global),
        "holdout_content_set_sha256": canonical_json_fingerprint(
            sorted(selected_content_keys)
        ),
        "test_split_used_as_paper_corpus": True,
        "external_paper_benchmarks_evaluated": False,
    }
    return LoadedData(
        training=training,
        validation=validation,
        training_pool=training_pool,
        validation_indices=validation_indices,
        protocol=protocol,
    )


def partition_pool(
    pool: np.ndarray, num_clients: int, seed: int, limit: int | None = None
) -> list[np.ndarray]:
    if len(pool) < num_clients:
        raise ValueError("population must be at least the number of clients")
    usable = len(pool) if limit is None else min(len(pool), limit)
    if usable < num_clients:
        raise ValueError("limited population must be at least the number of clients")
    positions = np.random.default_rng(seed).permutation(len(pool))[:usable]
    permutation = pool[positions]
    partitions = [part.astype(np.int64, copy=False) for part in np.array_split(permutation, num_clients)]
    if any(len(part) == 0 for part in partitions):
        raise RuntimeError("a client received an empty data partition")
    return partitions


def partition_indices(
    population: int, num_clients: int, seed: int, limit: int | None = None
) -> list[np.ndarray]:
    """Compatibility wrapper used by unit tests and synthetic callers."""

    return partition_pool(
        np.arange(population, dtype=np.int64), num_clients, seed, limit
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_model_stochasticity(seed: int, device: torch.device) -> None:
    """Set the frozen base model's dropout stream for one stateless client step."""

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_effective_config(args: argparse.Namespace) -> EffectiveConfig:
    clip_norm_a = getattr(args, "clip_norm_a", None)
    clip_norm_b = getattr(args, "clip_norm_b", None)
    if (clip_norm_a is None) != (clip_norm_b is None):
        raise ValueError("groupwise clip norms must provide both A and B")
    resolved_clip_norm_a = float(
        args.clip_norm if clip_norm_a is None else clip_norm_a
    )
    resolved_clip_norm_b = float(
        args.clip_norm if clip_norm_b is None else clip_norm_b
    )
    if args.smoke:
        return EffectiveConfig(
            method=args.method,
            num_clients=2,
            rounds=1,
            batch_size=args.batch_size,
            noise_multiplier=args.noise_multiplier,
            learning_rate=args.learning_rate,
            clip_norm=args.clip_norm,
            rank=args.rank,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
            data_split_seed=args.data_split_seed,
            evaluation_seed=args.evaluation_seed,
            max_validation_records=args.batch_size,
            eval_every=1,
            checkpoint_every=1,
            data_protocol=args.data_protocol,
            delta=args.delta,
            pair_noise_across_methods=args.pair_noise_across_methods,
            smoke=True,
            clip_norm_A=resolved_clip_norm_a,
            clip_norm_B=resolved_clip_norm_b,
        )
    return EffectiveConfig(
        method=args.method,
        num_clients=args.num_clients,
        rounds=args.rounds,
        batch_size=args.batch_size,
        noise_multiplier=args.noise_multiplier,
        learning_rate=args.learning_rate,
        clip_norm=args.clip_norm,
        rank=args.rank,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        data_split_seed=args.data_split_seed,
        evaluation_seed=args.evaluation_seed,
        max_validation_records=args.max_validation_records,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        data_protocol=args.data_protocol,
        delta=args.delta,
        pair_noise_across_methods=args.pair_noise_across_methods,
        smoke=False,
        clip_norm_A=resolved_clip_norm_a,
        clip_norm_B=resolved_clip_norm_b,
    )


def validate_config(config: EffectiveConfig) -> None:
    for name in (
        "num_clients",
        "rounds",
        "batch_size",
        "rank",
        "max_seq_length",
        "checkpoint_every",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.method not in METHOD_SPECS:
        raise ValueError(f"unsupported method: {config.method}")
    if config.noise_multiplier < 0:
        raise ValueError("noise_multiplier must be non-negative")
    if (
        config.learning_rate <= 0
        or config.clip_norm <= 0
        or any(value <= 0 for value in config.clip_norm_by_group.values())
    ):
        raise ValueError("learning_rate and all clip norms must be positive")
    if config.max_validation_records <= 0 or config.eval_every <= 0:
        raise ValueError("validation size and eval interval must be positive")
    if config.data_split_seed < 0 or config.evaluation_seed < 0:
        raise ValueError("data and evaluation seeds must be non-negative")
    if not 0 < config.delta < 1:
        raise ValueError("delta must be between zero and one")


def parameter_groups(model: torch.nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {"A": [], "B": []}
    unexpected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_A." in name:
            groups["A"].append((name, parameter))
        elif "lora_B." in name:
            groups["B"].append((name, parameter))
        else:
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"non-LoRA trainable parameters are not covered: {unexpected}")
    if not groups["A"] or not groups["B"]:
        raise RuntimeError("both LoRA A and LoRA B must be trainable")
    return groups


def squared_norm(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    total: torch.Tensor | None = None
    for tensor in tensors:
        value = tensor.detach().float().square().sum()
        total = value if total is None else total + value
    if total is None:
        raise RuntimeError("cannot compute a norm over an empty tensor group")
    return total


def clip_noise_and_step(
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]],
    *,
    clip_norm: float | None = None,
    clip_norm_by_group: dict[str, float] | None = None,
    noise_multiplier: float,
    learning_rate: float,
    generator: torch.Generator,
    apply_clipping: bool = True,
    slaclip_num_slots: int | None = None,
    slack_noise_generators: dict[str, torch.Generator] | None = None,
    component_accumulators: dict[str, dict[str, torch.Tensor]] | None = None,
    component_weight: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Apply the paper's separate A/B batch-gradient mechanism in place."""

    if (clip_norm is None) == (clip_norm_by_group is None):
        raise ValueError("provide exactly one of clip_norm or clip_norm_by_group")
    if clip_norm_by_group is not None and set(clip_norm_by_group) != {"A", "B"}:
        raise ValueError("clip_norm_by_group must contain exactly A and B")
    adaptive = slaclip_num_slots is not None or slack_noise_generators is not None
    if adaptive:
        if slaclip_num_slots is None or slaclip_num_slots <= 0:
            raise ValueError("adaptive clipping requires a positive slaclip_num_slots")
        if slack_noise_generators is None or set(slack_noise_generators) != {"A", "B"}:
            raise ValueError("adaptive clipping requires A/B slack noise generators")
        if not apply_clipping or noise_multiplier <= 0:
            raise ValueError("SlaClip requires clipping and positive Gaussian noise")

    statistics: dict[str, dict[str, Any]] = {}
    for group_name in ("A", "B"):
        group_clip_norm = float(
            clip_norm_by_group[group_name]
            if clip_norm_by_group is not None
            else clip_norm
        )
        if not math.isfinite(group_clip_norm) or group_clip_norm <= 0:
            raise ValueError(f"invalid LoRA {group_name} clipping threshold")
        entries = groups[group_name]
        missing = [name for name, parameter in entries if parameter.grad is None]
        if missing:
            raise RuntimeError(f"missing gradients in LoRA {group_name}: {missing[:5]}")
        gradients = [parameter.grad for _, parameter in entries]
        assert all(gradient is not None for gradient in gradients)
        raw_norm = float(torch.sqrt(squared_norm(gradients)).item())
        if not math.isfinite(raw_norm):
            raise FloatingPointError(f"non-finite raw LoRA {group_name} gradient norm")
        counterfactual_factor = min(
            1.0, group_clip_norm / max(raw_norm, torch.finfo(torch.float32).tiny)
        )
        factor = counterfactual_factor if apply_clipping else 1.0
        signal_sq: torch.Tensor | None = None
        noise_sq: torch.Tensor | None = None
        signal_noise_dot: torch.Tensor | None = None
        parameter_sq: torch.Tensor | None = None
        tensor_statistics: list[dict[str, Any]] = []
        for name, parameter in entries:
            gradient = parameter.grad
            assert gradient is not None
            raw_tensor_norm = float(torch.sqrt(squared_norm([gradient])).item())
            parameter_norm_before = float(
                torch.sqrt(squared_norm([parameter])).item()
            )
            parameter_sq_value = parameter.detach().float().square().sum()
            parameter_sq = (
                parameter_sq_value
                if parameter_sq is None
                else parameter_sq + parameter_sq_value
            )
            gradient.mul_(factor)
            signal_sq_value = gradient.detach().float().square().sum()
            signal_sq = (
                signal_sq_value
                if signal_sq is None
                else signal_sq + signal_sq_value
            )
            clipped_tensor_norm = raw_tensor_norm * factor
            if component_accumulators is not None:
                component_accumulators["signal"][name].add_(
                    gradient.detach().float(), alpha=component_weight
                )
            if noise_multiplier > 0:
                noise = torch.randn(
                    gradient.shape,
                    generator=generator,
                    device=gradient.device,
                    dtype=torch.float32,
                ).mul_(noise_multiplier * group_clip_norm)
                noise_sq_value = noise.square().sum()
                dot_value = (gradient.detach().float() * noise).sum()
                noise_sq = (
                    noise_sq_value if noise_sq is None else noise_sq + noise_sq_value
                )
                signal_noise_dot = (
                    dot_value
                    if signal_noise_dot is None
                    else signal_noise_dot + dot_value
                )
                if component_accumulators is not None:
                    component_accumulators["noise"][name].add_(
                        noise, alpha=component_weight
                    )
                gradient.add_(noise.to(dtype=gradient.dtype))
                noise_tensor_norm = float(torch.sqrt(noise_sq_value).item())
                signal_noise_tensor_dot = float(dot_value.item())
            else:
                noise_tensor_norm = 0.0
                signal_noise_tensor_dot = 0.0
            private_tensor_norm = float(torch.sqrt(squared_norm([gradient])).item())
            tensor_statistics.append(
                {
                    "name": name,
                    "elements": parameter.numel(),
                    "raw_gradient_l2": raw_tensor_norm,
                    "clipped_gradient_l2": clipped_tensor_norm,
                    "noise_l2": noise_tensor_norm,
                    "noise_rms": safe_ratio(
                        noise_tensor_norm, math.sqrt(parameter.numel())
                    ),
                    "private_gradient_l2": private_tensor_norm,
                    "signal_noise_dot": signal_noise_tensor_dot,
                    "parameter_l2_before": parameter_norm_before,
                    "local_update_l2": learning_rate * private_tensor_norm,
                }
            )
        private_sq_value = float(squared_norm(gradients).item())
        signal_sq_float = float(signal_sq.item()) if signal_sq is not None else 0.0
        noise_sq_float = float(noise_sq.item()) if noise_sq is not None else 0.0
        noisy_norm = math.sqrt(private_sq_value)
        noise_norm = math.sqrt(noise_sq_float)
        clipped_norm = math.sqrt(signal_sq_float)
        dot = float(signal_noise_dot.item()) if signal_noise_dot is not None else 0.0
        identity_rhs = signal_sq_float + noise_sq_float + 2.0 * dot
        identity_absolute_error = abs(private_sq_value - identity_rhs)
        identity_relative_error = safe_ratio(
            identity_absolute_error,
            max(private_sq_value, abs(identity_rhs), 1.0),
        )
        if identity_relative_error is None or identity_relative_error > 5e-5:
            raise RuntimeError(
                f"LoRA {group_name} signal/noise norm identity failed: "
                f"relative_error={identity_relative_error}"
            )
        parameter_norm_before = (
            float(torch.sqrt(parameter_sq).item()) if parameter_sq is not None else 0.0
        )
        expected_noise_norm = (
            noise_multiplier
            * group_clip_norm
            * math.sqrt(sum(parameter.numel() for _, parameter in entries))
        )
        statistics[group_name] = {
            "clip_threshold": group_clip_norm,
            "noise_std_per_coordinate": noise_multiplier * group_clip_norm,
            "raw_norm": raw_norm,
            "raw_to_threshold_ratio": raw_norm / group_clip_norm,
            "clip_factor": factor,
            "removed_gradient_l2": max(raw_norm - group_clip_norm, 0.0)
            if apply_clipping
            else 0.0,
            "retained_energy_fraction": factor * factor,
            "counterfactual_clip_factor": counterfactual_factor,
            "clipping_applied": apply_clipping,
            "would_clip": counterfactual_factor < 1.0,
            "clipped": apply_clipping and factor < 1.0,
            "clipped_norm": clipped_norm,
            "noise_l2_norm": noise_norm,
            "expected_noise_l2_norm": expected_noise_norm,
            "noise_l2_to_expected": safe_ratio(noise_norm, expected_noise_norm),
            "noisy_gradient_l2_norm": noisy_norm,
            "signal_to_noise_l2_ratio": safe_ratio(clipped_norm, noise_norm),
            "noise_to_signal_l2_ratio": safe_ratio(noise_norm, clipped_norm),
            "signal_noise_dot": dot,
            "signal_noise_cosine": safe_ratio(dot, clipped_norm * noise_norm),
            "signal_noise_norm_identity_absolute_error": identity_absolute_error,
            "signal_noise_norm_identity_relative_error": identity_relative_error,
            "signal_noise_norm_identity_passed": True,
            "parameter_l2_before": parameter_norm_before,
            "clean_update_l2": learning_rate * clipped_norm,
            "noise_update_l2": learning_rate * noise_norm,
            "local_update_l2": learning_rate * noisy_norm,
            "parameter_tensors": len(entries),
            "parameter_elements": sum(parameter.numel() for _, parameter in entries),
            "tensor_statistics": tensor_statistics,
        }
        if adaptive:
            assert slaclip_num_slots is not None
            assert slack_noise_generators is not None
            slack_signal = build_slack_vector(
                raw_norm,
                group_clip_norm,
                slaclip_num_slots,
            )
            slack_tensor = torch.tensor(
                slack_signal,
                dtype=torch.float32,
                device=gradients[0].device,
            )
            slack_noise = torch.randn(
                slack_tensor.shape,
                generator=slack_noise_generators[group_name],
                device=slack_tensor.device,
                dtype=torch.float32,
            ).mul_(noise_multiplier * group_clip_norm)
            noisy_slack = slack_tensor + slack_noise
            joint_signal_l2 = math.sqrt(
                clipped_norm * clipped_norm + float(slack_tensor.square().sum().item())
            )
            bound_ratio = joint_signal_l2 / group_clip_norm
            if bound_ratio > 1.0 + 1e-6:
                raise RuntimeError(
                    f"LoRA {group_name} SlaClip joint sensitivity bound failed: "
                    f"ratio={bound_ratio}"
                )
            statistics[group_name]["slaclip"] = {
                "variant": "full_slaclip_cdf_endpoints",
                "num_slots": slaclip_num_slots,
                "slack_lambda": group_clip_norm / math.sqrt(slaclip_num_slots),
                "slack_signal": [float(value) for value in slack_tensor.cpu().tolist()],
                "slack_noise": [float(value) for value in slack_noise.cpu().tolist()],
                "noisy_slack": [float(value) for value in noisy_slack.cpu().tolist()],
                "joint_signal_l2": joint_signal_l2,
                "joint_sensitivity_bound_ratio": bound_ratio,
                "joint_sensitivity_bound_passed": True,
                "slack_noise_std_per_coordinate": noise_multiplier * group_clip_norm,
            }
    with torch.no_grad():
        for entries in groups.values():
            for _, parameter in entries:
                assert parameter.grad is not None
                parameter.add_(parameter.grad, alpha=-learning_rate)
    for group_name in ("A", "B"):
        entries = groups[group_name]
        parameter_norm_after = float(
            torch.sqrt(squared_norm(parameter for _, parameter in entries)).item()
        )
        statistics[group_name]["parameter_l2_after"] = parameter_norm_after
        statistics[group_name]["relative_local_update"] = safe_ratio(
            float(statistics[group_name]["local_update_l2"]),
            float(statistics[group_name]["parameter_l2_before"]),
        )
    return statistics


def clone_trainable_state(
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]]
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
        for entries in groups.values()
        for name, parameter in entries
    }


def restore_trainable_state(
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]],
    state: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for entries in groups.values():
            for name, parameter in entries:
                if name not in state or state[name].shape != parameter.shape:
                    raise RuntimeError(f"invalid global adapter state for {name}")
                parameter.copy_(state[name].to(parameter.device, dtype=parameter.dtype))


def empty_state_like(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(value) for name, value in state.items()}


def accumulate_state(
    destination: dict[str, torch.Tensor],
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]],
    weight: float,
) -> None:
    with torch.no_grad():
        for entries in groups.values():
            for name, parameter in entries:
                destination[name].add_(
                    parameter.detach().to(device="cpu", dtype=torch.float32), alpha=weight
                )


def empty_mechanism_components(
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]]
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        component: {
            name: torch.zeros_like(parameter, dtype=torch.float32)
            for entries in groups.values()
            for name, parameter in entries
        }
        for component in ("signal", "noise")
    }


def group_for_parameter(name: str) -> str:
    if "lora_A." in name:
        return "A"
    if "lora_B." in name:
        return "B"
    raise ValueError(f"parameter is not LoRA A/B: {name}")


def round_update_statistics(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    components: dict[str, dict[str, torch.Tensor]],
    *,
    learning_rate: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_name in ("A", "B"):
        actual_sq = signal_sq = noise_sq = private_sq = residual_sq = 0.0
        signal_noise_dot = before_sq = after_sq = 0.0
        for name, previous in before.items():
            if group_for_parameter(name) != group_name:
                continue
            current = after[name]
            device = components["signal"][name].device
            actual = (current - previous).to(device=device, dtype=torch.float32)
            signal = components["signal"][name]
            noise = components["noise"][name]
            predicted = (signal + noise).mul(-learning_rate)
            residual = actual - predicted
            actual_sq += float(actual.square().sum().item())
            signal_sq += float(signal.square().sum().item())
            noise_sq += float(noise.square().sum().item())
            private_sq += float((signal + noise).square().sum().item())
            residual_sq += float(residual.square().sum().item())
            signal_noise_dot += float((signal * noise).sum().item())
            before_sq += float(previous.float().square().sum().item())
            after_sq += float(current.float().square().sum().item())
        signal_norm = math.sqrt(signal_sq)
        noise_norm = math.sqrt(noise_sq)
        private_norm = math.sqrt(private_sq)
        actual_update_norm = math.sqrt(actual_sq)
        component_identity_rhs = signal_sq + noise_sq + 2.0 * signal_noise_dot
        component_identity_error = abs(private_sq - component_identity_rhs)
        component_identity_relative_error = safe_ratio(
            component_identity_error,
            max(private_sq, abs(component_identity_rhs), 1.0),
        )
        if (
            component_identity_relative_error is None
            or component_identity_relative_error > 5e-5
        ):
            raise RuntimeError(
                f"round LoRA {group_name} component norm identity failed"
            )
        residual_norm = math.sqrt(residual_sq)
        predicted_update_norm = learning_rate * private_norm
        residual_tolerance = 2e-5 + 2e-5 * max(
            actual_update_norm, predicted_update_norm
        )
        if residual_norm > residual_tolerance:
            raise RuntimeError(
                f"round LoRA {group_name} FedAvg reconstruction failed: "
                f"{residual_norm} > {residual_tolerance}"
            )
        result[group_name] = {
            "aggregate_signal_gradient_l2": signal_norm,
            "aggregate_noise_gradient_l2": noise_norm,
            "aggregate_private_gradient_l2": private_norm,
            "signal_noise_dot": signal_noise_dot,
            "signal_noise_cosine": safe_ratio(
                signal_noise_dot, signal_norm * noise_norm
            ),
            "signal_noise_norm_identity_absolute_error": component_identity_error,
            "signal_noise_norm_identity_relative_error": component_identity_relative_error,
            "signal_noise_norm_identity_passed": True,
            "signal_to_noise_l2_ratio": safe_ratio(signal_norm, noise_norm),
            "noise_to_signal_l2_ratio": safe_ratio(noise_norm, signal_norm),
            "predicted_signal_update_l2": learning_rate * signal_norm,
            "predicted_noise_update_l2": learning_rate * noise_norm,
            "predicted_private_update_l2": predicted_update_norm,
            "actual_global_update_l2": actual_update_norm,
            "fedavg_reconstruction_residual_l2": residual_norm,
            "fedavg_relative_residual": safe_ratio(
                residual_norm, actual_update_norm
            ),
            "fedavg_reconstruction_tolerance_l2": residual_tolerance,
            "fedavg_reconstruction_passed": True,
            "global_parameter_l2_before": math.sqrt(before_sq),
            "global_parameter_l2_after": math.sqrt(after_sq),
            "relative_global_update": safe_ratio(
                actual_update_norm, math.sqrt(before_sq)
            ),
        }
    return result


def effective_lora_statistics(
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]]
) -> dict[str, Any]:
    a_parameters = {name: parameter for name, parameter in groups["A"]}
    b_parameters = {name: parameter for name, parameter in groups["B"]}
    layers: list[dict[str, Any]] = []
    total_sq = 0.0
    with torch.no_grad():
        for a_name, a_parameter in sorted(a_parameters.items()):
            b_name = a_name.replace(".lora_A.", ".lora_B.")
            if b_name not in b_parameters:
                raise RuntimeError(f"LoRA B partner is missing for {a_name}")
            b_parameter = b_parameters[b_name]
            effective = b_parameter.detach().float() @ a_parameter.detach().float()
            norm = float(torch.linalg.vector_norm(effective).item())
            total_sq += norm * norm
            layers.append(
                {
                    "module": a_name.split(".lora_A.", 1)[0],
                    "frobenius_l2": norm,
                    "shape": list(effective.shape),
                }
            )
    return {
        "definition": "frobenius_norm_of_B_matmul_A_with_lora_scaling_1",
        "layers": layers,
        "aggregate_frobenius_l2": math.sqrt(total_sq),
    }


def synchronized_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def resource_telemetry(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
    }
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result.update(
            {
                "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(
                    device
                ),
                "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                    device
                ),
                "cuda_memory_free_bytes": free_bytes,
                "cuda_memory_total_bytes": total_bytes,
            }
        )
    return result


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def mask_bert_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    probability = torch.full(input_ids.shape, 0.15, dtype=torch.float32)
    probability.masked_fill_(attention_mask.eq(0), 0.0)
    for token_id in tokenizer.all_special_ids:
        probability.masked_fill_(input_ids.eq(int(token_id)), 0.0)
    masked = torch.rand(input_ids.shape, generator=generator).lt(probability)
    for row in range(masked.shape[0]):
        if not bool(masked[row].any()):
            candidates = torch.nonzero(probability[row].gt(0), as_tuple=False).flatten()
            if len(candidates):
                masked[row, int(candidates[0])] = True
    labels[~masked] = -100
    replace_draw = torch.rand(input_ids.shape, generator=generator)
    replace_mask = masked & replace_draw.lt(0.8)
    input_ids = input_ids.clone()
    input_ids[replace_mask] = int(tokenizer.mask_token_id)
    random_mask = masked & replace_draw.ge(0.8) & replace_draw.lt(0.9)
    random_tokens = torch.randint(
        low=0,
        high=len(tokenizer),
        size=input_ids.shape,
        generator=generator,
        dtype=torch.long,
    )
    input_ids[random_mask] = random_tokens[random_mask]
    return input_ids, labels


def make_batch(
    texts: Sequence[str],
    tokenizer: Any,
    model_kind: str,
    max_seq_length: int,
    device: torch.device,
    mask_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if model_kind == "bert":
        generator = torch.Generator(device="cpu").manual_seed(mask_seed)
        input_ids, labels = mask_bert_inputs(
            input_ids, attention_mask, tokenizer, generator
        )
    elif model_kind == "gpt2":
        labels = input_ids.clone()
        labels[attention_mask.eq(0)] = -100
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")
    supervised = labels[:, 1:] if model_kind == "gpt2" else labels
    sequence_lengths = attention_mask.sum(dim=1)
    telemetry = {
        "records": int(input_ids.shape[0]),
        "tensor_sequence_length": int(input_ids.shape[1]),
        "non_padding_tokens": int(attention_mask.sum().item()),
        "padding_tokens": int(attention_mask.eq(0).sum().item()),
        "supervised_tokens": int(supervised.ne(-100).sum().item()),
        "records_at_sequence_limit": int(
            sequence_lengths.ge(max_seq_length).sum().item()
        ),
        "mean_sequence_tokens": float(sequence_lengths.float().mean().item()),
        "input_ids_sha256": tensor_sha256(input_ids),
        "labels_sha256": tensor_sha256(labels),
    }
    batch = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }
    return batch, telemetry


def equal_record_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    model_kind: str,
) -> torch.Tensor:
    """Average token loss within each record, then average the B records."""

    outputs = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    logits, labels = align_supervised_logits_and_labels(
        outputs.logits, batch["labels"], model_kind
    )
    return _equal_record_loss_from_aligned(logits, labels)


def align_supervised_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model_kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align logits with the labels that each LM objective supervises."""

    if model_kind == "gpt2":
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
    elif model_kind != "bert":
        raise ValueError(f"unsupported model kind: {model_kind}")
    if logits.shape[:-1] != labels.shape:
        raise RuntimeError("aligned logits and labels have incompatible shapes")
    return logits, labels


def _equal_record_loss_from_aligned(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the established equal-record objective from aligned tensors."""

    import torch.nn.functional as functional

    token_losses = functional.cross_entropy(
        logits.float().view(-1, logits.shape[-1]),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape)
    valid = labels.ne(-100)
    counts = valid.sum(dim=1)
    if bool(counts.eq(0).any()):
        raise RuntimeError("a batch record has no supervised tokens")
    record_losses = (token_losses * valid).sum(dim=1) / counts
    return record_losses.mean()


def supervised_token_prediction_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model_kind: str,
) -> tuple[int, int]:
    """Return ``(correct, total)`` over valid masked/next-token labels."""

    aligned_logits, aligned_labels = align_supervised_logits_and_labels(
        logits, labels, model_kind
    )
    return _supervised_token_prediction_counts_from_aligned(
        aligned_logits, aligned_labels
    )


def _supervised_token_prediction_counts_from_aligned(
    aligned_logits: torch.Tensor,
    aligned_labels: torch.Tensor,
) -> tuple[int, int]:
    """Count correct supervised tokens after objective-specific alignment."""

    if aligned_logits.shape[:-1] != aligned_labels.shape:
        raise RuntimeError("aligned logits and labels have incompatible shapes")
    valid = aligned_labels.ne(-100)
    total = int(valid.sum().item())
    if total <= 0:
        raise RuntimeError("a validation batch has no supervised tokens")
    predictions = aligned_logits.argmax(dim=-1)
    correct = int(predictions.eq(aligned_labels).logical_and(valid).sum().item())
    return correct, total


def evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    model_kind: str,
    validation: ParquetTextTable,
    validation_indices: np.ndarray,
    config: EffectiveConfig,
    device: torch.device,
    seed: int,
) -> dict[str, float | int | str]:
    model.eval()
    weighted_loss_sum = 0.0
    evaluated_records = 0
    supervised_tokens = 0
    correct_tokens = 0
    batches = 0
    with torch.no_grad():
        for offset in range(0, len(validation_indices), config.batch_size):
            indices = validation_indices[offset : offset + config.batch_size]
            batch, _ = make_batch(
                validation.texts(indices),
                tokenizer,
                model_kind,
                config.max_seq_length,
                device,
                seed + offset,
            )
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            aligned_logits, aligned_labels = align_supervised_logits_and_labels(
                outputs.logits, batch["labels"], model_kind
            )
            loss = _equal_record_loss_from_aligned(
                aligned_logits, aligned_labels
            )
            batch_correct, batch_supervised = (
                _supervised_token_prediction_counts_from_aligned(
                    aligned_logits, aligned_labels
                )
            )
            value = float(loss.detach().float().item())
            if not math.isfinite(value):
                raise FloatingPointError("non-finite validation loss")
            batch_records = int(len(indices))
            weighted_loss_sum += value * batch_records
            evaluated_records += batch_records
            supervised_tokens += batch_supervised
            correct_tokens += batch_correct
            batches += 1
    if evaluated_records != len(validation_indices) or evaluated_records <= 0:
        raise RuntimeError("validation record accounting mismatch")
    if supervised_tokens <= 0 or not 0 <= correct_tokens <= supervised_tokens:
        raise RuntimeError("validation supervised-token accounting mismatch")
    mean_loss = float(weighted_loss_sum / evaluated_records)
    model.train()
    return {
        "objective": "masked_lm" if model_kind == "bert" else "causal_lm",
        "records": int(len(validation_indices)),
        "batches": batches,
        "supervised_tokens": supervised_tokens,
        "correct_tokens": correct_tokens,
        "token_accuracy": float(correct_tokens / supervised_tokens),
        "token_accuracy_definition": "supervised_token_top1_micro_accuracy",
        "loss": mean_loss,
        "exp_loss": math.exp(min(mean_loss, 20.0)),
        "exp_loss_capped_at_20": mean_loss > 20.0,
    }


def build_model(
    model_kind: str,
    snapshot_path: Path,
    rank: int,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, list[str]]:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path, local_files_only=True, use_fast=True
    )
    if model_kind == "bert":
        base = AutoModelForMaskedLM.from_pretrained(
            snapshot_path, local_files_only=True, torch_dtype=torch.float32
        )
        targets = ["query", "key", "value"]
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
        )
    elif model_kind == "gpt2":
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            snapshot_path, local_files_only=True, torch_dtype=torch.float32
        )
        base.config.use_cache = False
        targets = ["c_attn"]
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")
    model = get_peft_model(base, lora_config).to(device)
    model.train()
    return model, tokenizer, targets


def model_snapshot(manifest: dict[str, Any], model_kind: str) -> Path:
    key = EXPECTED_MODELS[model_kind]["manifest_key"]
    return Path(manifest["models"][key]["snapshot_path"])


def aggregate_round_statistics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "clients": len(records),
        "mean_training_loss": float(np.mean([record["loss"] for record in records])),
        "sample_draws": sum(record["batch_size"] for record in records),
        "supervised_tokens": sum(
            record["batch_telemetry"]["supervised_tokens"] for record in records
        ),
    }
    for group in ("A", "B"):
        values = [record["gradient_groups"][group] for record in records]
        clipped = sum(bool(value["clipped"]) for value in values)
        would_clip = sum(bool(value["would_clip"]) for value in values)
        relative_updates = [
            value["relative_local_update"]
            for value in values
            if value["relative_local_update"] is not None
        ]
        summary[group] = {
            "clipped_count": clipped,
            "clipped_fraction": clipped / len(values),
            "would_clip_count": would_clip,
            "would_clip_fraction": would_clip / len(values),
            "mean_raw_norm": float(np.mean([value["raw_norm"] for value in values])),
            "max_raw_norm": float(np.max([value["raw_norm"] for value in values])),
            "mean_clip_factor": float(
                np.mean([value["clip_factor"] for value in values])
            ),
            "mean_noise_l2_norm": float(
                np.mean([value["noise_l2_norm"] for value in values])
            ),
            "mean_private_gradient_l2_norm": float(
                np.mean([value["noisy_gradient_l2_norm"] for value in values])
            ),
            "mean_parameter_l2_before": float(
                np.mean([value["parameter_l2_before"] for value in values])
            ),
            "mean_relative_local_update": float(np.mean(relative_updates))
            if relative_updates
            else None,
            "relative_local_update_defined_count": len(relative_updates),
            "relative_local_update_undefined_count": len(values)
            - len(relative_updates),
        }
    summary["any_group_clipped_count"] = sum(
        record["gradient_groups"]["A"]["clipped"]
        or record["gradient_groups"]["B"]["clipped"]
        for record in records
    )
    summary["any_group_clipped_fraction"] = (
        summary["any_group_clipped_count"] / len(records)
    )
    summary["any_group_would_clip_count"] = sum(
        record["gradient_groups"]["A"]["would_clip"]
        or record["gradient_groups"]["B"]["would_clip"]
        for record in records
    )
    summary["any_group_would_clip_fraction"] = (
        summary["any_group_would_clip_count"] / len(records)
    )
    timing_names = sorted(
        {
            name
            for record in records
            for name in record["timings_seconds"]
        }
    )
    summary["timings_seconds"] = {
        name: {
            "sum": float(sum(record["timings_seconds"].get(name, 0.0) for record in records)),
            "mean": float(
                np.mean([record["timings_seconds"].get(name, 0.0) for record in records])
            ),
        }
        for name in timing_names
    }
    return summary


def slaclip_round_controller_summary(
    records: Sequence[dict[str, Any]],
    *,
    clip_thresholds: dict[str, float],
    num_slots: int,
    eta: float,
    controller_input: str = NOISY_CONTROLLER_INPUT,
    base_target_clipped_fraction: float | None = None,
    beta: float | None = None,
    base_target_clipped_fraction_by_group: Mapping[str, Any] | None = None,
    c_min: float,
    c_max: float,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Reconcile one adaptive update from noisy and exact endpoint signals.

    Formal SlaClip selects ``noisy_endpoints``.  The explicitly non-private
    oracle control selects ``exact_endpoints`` while retaining the same noisy
    gradient and slack-vector releases so that only the controller input is
    ablated.
    """

    if not records:
        raise ValueError("cannot update SlaClip without client releases")
    if num_slots < 2:
        raise ValueError("full SlaClip requires at least two CDF slots")
    if controller_input not in {NOISY_CONTROLLER_INPUT, EXACT_CONTROLLER_INPUT}:
        raise ValueError(f"unsupported controller input: {controller_input!r}")
    base_targets, groupwise_targets = resolve_slaclip_base_targets(
        base_target_clipped_fraction=base_target_clipped_fraction,
        beta=beta,
        base_target_clipped_fraction_by_group=(
            base_target_clipped_fraction_by_group
        ),
    )
    result: dict[str, Any] = {
        "variant": (
            GROUPWISE_SLACLIP_VARIANT
            if groupwise_targets
            else "full_slaclip_cdf_endpoints"
        ),
        "update_timing": "once_after_all_clients_for_use_in_next_round",
        "clients": len(records),
        "num_slots": num_slots,
        "eta": eta,
        "controller_input": controller_input,
        "controller_input_is_non_dp_exact": (
            controller_input == EXACT_CONTROLLER_INPUT
        ),
        "epsilon": epsilon,
        "near_threshold_index": 0,
        "near_zero_index": num_slots - 1,
        "c_min": c_min,
        "c_max": c_max,
    }
    if groupwise_targets:
        result.update(
            {
                "target_parameterization": "per_gradient_group",
                "generalized_full_slaclip_beta": True,
                "base_target_clipped_fraction_by_group": dict(base_targets),
                "beta_by_group": dict(base_targets),
            }
        )
    else:
        # Keep the historical shared-beta telemetry shape unchanged so old
        # round shards remain exactly reconcilable on resume/revalidation.
        result.update(
            {
                "base_target_clipped_fraction": base_targets["A"],
                "beta": base_targets["A"],
            }
        )
    for group in ("A", "B"):
        threshold = float(clip_thresholds[group])
        base_target = base_targets[group]
        releases = []
        signals = []
        for record in records:
            group_statistics = record["gradient_groups"][group]
            if group_statistics.get("clip_threshold") != threshold:
                raise RuntimeError(
                    f"SlaClip {group} threshold changed within a federated round"
                )
            telemetry = group_statistics.get("slaclip")
            if (
                not isinstance(telemetry, dict)
                or telemetry.get("num_slots") != num_slots
                or telemetry.get("joint_sensitivity_bound_passed") is not True
            ):
                raise RuntimeError(f"invalid SlaClip {group} client release")
            noisy_slack = telemetry.get("noisy_slack")
            slack_signal = telemetry.get("slack_signal")
            if (
                not isinstance(noisy_slack, list)
                or len(noisy_slack) != num_slots
                or not isinstance(slack_signal, list)
                or len(slack_signal) != num_slots
            ):
                raise RuntimeError(f"invalid SlaClip {group} slack vector")
            releases.append([float(value) for value in noisy_slack])
            signals.append([float(value) for value in slack_signal])
        noisy_sum = [
            math.fsum(release[slot] for release in releases)
            for slot in range(num_slots)
        ]
        signal_sum = [
            math.fsum(signal[slot] for signal in signals)
            for slot in range(num_slots)
        ]
        noisy_indicator = normalize_noisy_slack(
            noisy_sum,
            threshold,
            num_slots,
            len(records),
        )
        exact_indicator = normalize_noisy_slack(
            signal_sum,
            threshold,
            num_slots,
            len(records),
        )
        slack_lambda = threshold / math.sqrt(num_slots)
        per_client_slack_noise_std = float(
            records[0]["gradient_groups"][group]["slaclip"][
                "slack_noise_std_per_coordinate"
            ]
        )
        normalized_proxy_noise_std = (
            per_client_slack_noise_std
            * math.sqrt(len(records))
            / (slack_lambda * len(records))
        )
        cdf_errors = [
            float(noisy - exact)
            for noisy, exact in zip(noisy_indicator, exact_indicator)
        ]
        cdf_error_mae = math.fsum(abs(value) for value in cdf_errors) / num_slots
        cdf_error_rmse = math.sqrt(
            math.fsum(value * value for value in cdf_errors) / num_slots
        )
        noisy_update = full_slaclip_update(
            threshold,
            float(noisy_indicator[0]),
            float(noisy_indicator[-1]),
            base_target_clipped_fraction=base_target,
            eta=eta,
            min_clip_norm=c_min,
            max_clip_norm=c_max,
            epsilon=epsilon,
        )
        oracle_update = full_slaclip_update(
            threshold,
            float(exact_indicator[0]),
            float(exact_indicator[-1]),
            base_target_clipped_fraction=base_target,
            eta=eta,
            min_clip_norm=c_min,
            max_clip_norm=c_max,
            epsilon=epsilon,
        )

        def normalized_update_fields(update: Mapping[str, Any]) -> dict[str, Any]:
            fields = (
                asdict(update)
                if hasattr(update, "__dataclass_fields__")
                else dict(update)
            )
            fields["unbounded_next_clip_threshold"] = fields.pop(
                "unbounded_next_clip_norm"
            )
            fields["next_clip_threshold"] = fields.pop("next_clip_norm")
            fields["c_min"] = fields.pop("min_clip_norm")
            fields["c_max"] = fields.pop("max_clip_norm")
            fields.pop("current_clip_norm", None)
            return fields

        noisy_fields = normalized_update_fields(noisy_update)
        oracle_fields = normalized_update_fields(oracle_update)
        update_fields = dict(
            noisy_fields
            if controller_input == NOISY_CONTROLLER_INPUT
            else oracle_fields
        )
        oracle_next = float(oracle_fields["next_clip_threshold"])
        noisy_next = float(noisy_fields["next_clip_threshold"])
        noisy_log_step = float(noisy_fields["raw_log_step"])
        oracle_log_step = float(oracle_fields["raw_log_step"])

        def direction(value: float) -> int:
            return -1 if value < 0.0 else (1 if value > 0.0 else 0)

        actual_fraction = safe_ratio(
            sum(bool(record["gradient_groups"][group]["clipped"]) for record in records),
            len(records),
        )
        result[group] = {
            "clip_threshold_used": threshold,
            "noisy_cdf_proxy_by_slot": [
                float(value) for value in noisy_indicator
            ],
            "exact_cdf_proxy_by_slot": [
                float(value) for value in exact_indicator
            ],
            "controller_input": controller_input,
            "controller_input_is_non_dp_exact": (
                controller_input == EXACT_CONTROLLER_INPUT
            ),
            "normalized_proxy_noise_std_per_slot": normalized_proxy_noise_std,
            "cdf_error_mae": float(cdf_error_mae),
            "cdf_error_rmse": float(cdf_error_rmse),
            "cdf_error_max_abs": float(max(abs(value) for value in cdf_errors)),
            "cdf_error_z_rmse": (
                float(cdf_error_rmse / normalized_proxy_noise_std)
                if normalized_proxy_noise_std > 0.0
                else None
            ),
            "noisy_cdf_out_of_range_count": sum(
                not 0.0 <= value <= 1.0 for value in noisy_indicator
            ),
            "noisy_cdf_out_of_range_fraction": safe_ratio(
                sum(not 0.0 <= value <= 1.0 for value in noisy_indicator),
                num_slots,
            ),
            "noisy_near_threshold_minus_exact": float(
                noisy_indicator[0] - exact_indicator[0]
            ),
            "noisy_near_zero_minus_exact": float(
                noisy_indicator[-1] - exact_indicator[-1]
            ),
            "noisy_adjacent_monotonicity_violations": sum(
                noisy_indicator[index + 1] > noisy_indicator[index]
                for index in range(num_slots - 1)
            ),
            "exact_adjacent_monotonicity_violations": sum(
                exact_indicator[index + 1] > exact_indicator[index]
                for index in range(num_slots - 1)
            ),
            "actual_clip_fraction": actual_fraction,
            "actual_minus_dynamic_target_clipped": (
                None
                if actual_fraction is None
                else actual_fraction - float(update_fields["dynamic_target_clipped"])
            ),
            "actual_target_absolute_error": (
                None
                if actual_fraction is None
                else abs(
                    actual_fraction
                    - float(update_fields["dynamic_target_clipped"])
                )
            ),
            "noisy_dynamic_target_clipped": float(
                noisy_fields["dynamic_target_clipped"]
            ),
            "noisy_raw_log_step": noisy_log_step,
            "noisy_next_clip_threshold": noisy_next,
            "oracle_dynamic_target_clipped": float(
                oracle_fields["dynamic_target_clipped"]
            ),
            "oracle_raw_log_step": oracle_log_step,
            "oracle_next_clip_threshold": oracle_next,
            "noisy_minus_oracle_raw_log_step": float(
                noisy_log_step - oracle_log_step
            ),
            "noisy_oracle_log_threshold_error": float(
                math.log(noisy_next / oracle_next)
            ),
            "update_direction_agrees": (
                direction(noisy_log_step) == direction(oracle_log_step)
            ),
            **update_fields,
        }
    return result


def read_round_shards(
    rounds_directory: Path,
    *,
    expected_rounds: int,
    expected_model: str,
    expected_method: str,
    expected_clients: int,
    expected_batch_size: int,
    slaclip_contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validate_private_directory(rounds_directory, "round diagnostics directory")
    adaptive = expected_method in ADAPTIVE_METHODS
    if adaptive != (slaclip_contract is not None):
        raise RuntimeError("SlaClip round validation contract mismatch")
    current_thresholds: dict[str, float] | None = None
    controller: dict[str, Any] | None = None
    if slaclip_contract is not None:
        controller = slaclip_contract["controller"]
        groupwise_initial = controller.get("initial_clip_threshold_by_group")
        if groupwise_initial is None:
            current_thresholds = {
                "A": float(controller["initial_clip_threshold"]),
                "B": float(controller["initial_clip_threshold"]),
            }
        elif (
            not isinstance(groupwise_initial, dict)
            or set(groupwise_initial) != {"A", "B"}
        ):
            raise RuntimeError("SlaClip groupwise initial thresholds are invalid")
        else:
            current_thresholds = {
                group: float(groupwise_initial[group]) for group in ("A", "B")
            }
    shards: list[dict[str, Any]] = []
    expected_paths = {
        rounds_directory / f"round-{round_index:05d}.json"
        for round_index in range(1, expected_rounds + 1)
    }
    actual_paths = set(rounds_directory.glob("round-*.json"))
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise RuntimeError(
            f"round shard set mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    for round_index in range(1, expected_rounds + 1):
        path = rounds_directory / f"round-{round_index:05d}.json"
        validate_private_regular_file(path, "round diagnostic shard")
        payload = load_json_object(path, "round diagnostic shard")
        if payload.get("schema_version") != 2:
            raise RuntimeError(f"unsupported round shard schema: {path}")
        if payload.get("round") != round_index:
            raise RuntimeError(f"round shard index mismatch: {path}")
        if payload.get("model") != expected_model:
            raise RuntimeError(f"round shard model mismatch: {path}")
        if payload.get("method") != expected_method:
            raise RuntimeError(f"round shard method mismatch: {path}")
        client_records = payload.get("client_records")
        if not isinstance(client_records, list):
            raise RuntimeError(f"round shard has no client records: {path}")
        if len(client_records) != expected_clients:
            raise RuntimeError(f"round shard client count mismatch: {path}")
        if any(not isinstance(record, dict) for record in client_records):
            raise RuntimeError(f"round shard has an invalid client record: {path}")
        client_ids = [record.get("client") for record in client_records]
        if client_ids != list(range(expected_clients)):
            raise RuntimeError(f"round shard client IDs are missing or reordered: {path}")
        for record in client_records:
            if (
                record.get("round") != round_index
                or record.get("model") != expected_model
                or record.get("method") != expected_method
            ):
                raise RuntimeError(f"round/client record identity mismatch: {path}")
            if record.get("batch_size") != expected_batch_size:
                raise RuntimeError(f"round/client batch size mismatch: {path}")
            sample_indices = record.get("sample_indices")
            if not isinstance(sample_indices, list) or len(sample_indices) != expected_batch_size:
                raise RuntimeError(f"round/client sample list mismatch: {path}")
            if record.get("sample_indices_sha256") != int64_index_digest(
                sample_indices
            ):
                raise RuntimeError(f"round/client sample digest mismatch: {path}")
            gradient_groups = record.get("gradient_groups")
            if not isinstance(gradient_groups, dict):
                raise RuntimeError(f"round/client gradient groups are missing: {path}")
            for group in ("A", "B"):
                statistics = gradient_groups.get(group)
                if (
                    not isinstance(statistics, dict)
                    or statistics.get("signal_noise_norm_identity_passed") is not True
                ):
                    raise RuntimeError(
                        f"round/client mechanism identity failed for {group}: {path}"
                    )
        round_summary = payload.get("round_summary")
        if not isinstance(round_summary, dict) or round_summary.get("round") != round_index:
            raise RuntimeError(f"round summary identity mismatch: {path}")
        recomputed_summary = aggregate_round_statistics(client_records)
        for key, expected in recomputed_summary.items():
            if round_summary.get(key) != expected:
                raise RuntimeError(f"round summary does not reconcile for {key}: {path}")
        if adaptive:
            assert controller is not None
            assert current_thresholds is not None
            recomputed_controller = slaclip_round_controller_summary(
                client_records,
                clip_thresholds=current_thresholds,
                num_slots=int(controller["num_slots"]),
                eta=float(controller["eta"]),
                controller_input=str(controller["controller_input"]),
                c_min=float(controller["c_min"]),
                c_max=float(controller["c_max"]),
                epsilon=float(controller["epsilon"]),
                **slaclip_controller_target_arguments(controller),
            )
            if round_summary.get("slaclip_controller") != recomputed_controller:
                raise RuntimeError(
                    f"SlaClip controller summary does not reconcile: {path}"
                )
            current_thresholds = {
                group: float(recomputed_controller[group]["next_clip_threshold"])
                for group in ("A", "B")
            }
        elif "slaclip_controller" in round_summary:
            raise RuntimeError(f"fixed-threshold arm contains SlaClip telemetry: {path}")
        federated_update = round_summary.get("federated_update")
        if not isinstance(federated_update, dict):
            raise RuntimeError(f"round summary has no federated update: {path}")
        for group in ("A", "B"):
            update = federated_update.get(group)
            if (
                not isinstance(update, dict)
                or update.get("signal_noise_norm_identity_passed") is not True
                or update.get("fedavg_reconstruction_passed") is not True
            ):
                raise RuntimeError(f"round mechanism gate failed for {group}: {path}")
        shards.append(payload)
    return shards


def round_shard_prefix_sha256(
    rounds_directory: Path, *, completed_round: int
) -> str:
    validate_private_directory(rounds_directory, "round diagnostics directory")
    digest = hashlib.sha256(b"dp-lora-round-shard-prefix-v1\0")
    for round_index in range(1, completed_round + 1):
        path = rounds_directory / f"round-{round_index:05d}.json"
        try:
            validate_private_regular_file(path, "round diagnostic shard")
        except FileNotFoundError as error:
            raise RuntimeError(f"checkpointed round shard is missing: {path}") from error

        file_digest = sha256_file(path)
        digest.update(path.name.encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
    digest.update(completed_round.to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def behavior_summary(shards: Sequence[dict[str, Any]]) -> dict[str, Any]:
    clients = [
        record
        for shard in shards
        for record in shard["client_records"]
    ]
    rounds = [shard["round_summary"] for shard in shards]
    schedule = [
        {
            "round": record["round"],
            "client": record["client"],
            "sample_indices": record["sample_indices"],
        }
        for record in clients
    ]
    mask_schedule = [
        {
            "round": record["round"],
            "client": record["client"],
            "labels_sha256": record["batch_telemetry"]["labels_sha256"],
        }
        for record in clients
    ]
    group_summaries: dict[str, Any] = {}
    for group in ("A", "B"):
        values = [record["gradient_groups"][group] for record in clients]
        clipped_rounds = [
            record["round"] for record, value in zip(clients, values) if value["clipped"]
        ]
        fully_clipped_round_count = sum(
            float(round_record[group]["clipped_fraction"]) == 1.0
            for round_record in rounds
        )
        group_summaries[group] = {
            "actual_clipped_count": sum(bool(value["clipped"]) for value in values),
            "actual_clipped_fraction": safe_ratio(
                sum(bool(value["clipped"]) for value in values), len(values)
            ),
            "would_clip_count": sum(bool(value["would_clip"]) for value in values),
            "would_clip_fraction": safe_ratio(
                sum(bool(value["would_clip"]) for value in values), len(values)
            ),
            "first_actual_clipped_round": min(clipped_rounds)
            if clipped_rounds
            else None,
            "fully_clipped_round_count": fully_clipped_round_count,
            "fully_clipped_round_fraction": safe_ratio(
                fully_clipped_round_count, len(rounds)
            ),
            "round_actual_clipped_fraction": safe_quantiles(
                round_record[group]["clipped_fraction"]
                for round_record in rounds
            ),
            "raw_gradient_l2": safe_quantiles(value["raw_norm"] for value in values),
            "raw_to_threshold_ratio": safe_quantiles(
                value["raw_to_threshold_ratio"] for value in values
            ),
            "removed_gradient_l2": safe_quantiles(
                value["removed_gradient_l2"] for value in values
            ),
            "retained_energy_fraction": safe_quantiles(
                value["retained_energy_fraction"] for value in values
            ),
            "clipped_signal_gradient_l2": safe_quantiles(
                value["clipped_norm"] for value in values
            ),
            "noise_gradient_l2": safe_quantiles(
                value["noise_l2_norm"] for value in values
            ),
            "private_gradient_l2": safe_quantiles(
                value["noisy_gradient_l2_norm"] for value in values
            ),
            "signal_to_noise_l2_ratio": safe_quantiles(
                value["signal_to_noise_l2_ratio"] for value in values
            ),
            "noise_to_signal_l2_ratio": safe_quantiles(
                value["noise_to_signal_l2_ratio"] for value in values
            ),
            "signal_noise_cosine": safe_quantiles(
                value["signal_noise_cosine"] for value in values
            ),
            "signal_noise_norm_identity_relative_error": safe_quantiles(
                value["signal_noise_norm_identity_relative_error"]
                for value in values
            ),
            "local_update_l2": safe_quantiles(
                value["local_update_l2"] for value in values
            ),
            "relative_local_update": safe_quantiles(
                value["relative_local_update"] for value in values
            ),
            "global_update_l2": safe_quantiles(
                round_record["federated_update"][group]["actual_global_update_l2"]
                for round_record in rounds
            ),
            "aggregate_signal_gradient_l2": safe_quantiles(
                round_record["federated_update"][group][
                    "aggregate_signal_gradient_l2"
                ]
                for round_record in rounds
            ),
            "aggregate_noise_gradient_l2": safe_quantiles(
                round_record["federated_update"][group][
                    "aggregate_noise_gradient_l2"
                ]
                for round_record in rounds
            ),
            "aggregate_signal_to_noise_l2_ratio": safe_quantiles(
                round_record["federated_update"][group][
                    "signal_to_noise_l2_ratio"
                ]
                for round_record in rounds
            ),
            "relative_global_update": safe_quantiles(
                round_record["federated_update"][group]["relative_global_update"]
                for round_record in rounds
            ),
            "fedavg_relative_residual": safe_quantiles(
                round_record["federated_update"][group]["fedavg_relative_residual"]
                for round_record in rounds
            ),
        }
    timing_names = sorted(
        {
            name
            for record in clients
            for name in record["timings_seconds"]
        }
    )
    timing_totals = {
        name: float(sum(record["timings_seconds"].get(name, 0.0) for record in clients))
        for name in timing_names
    }
    any_actual_clipped_count = sum(
        bool(record["gradient_groups"]["A"]["clipped"])
        or bool(record["gradient_groups"]["B"]["clipped"])
        for record in clients
    )
    any_would_clip_count = sum(
        bool(record["gradient_groups"]["A"]["would_clip"])
        or bool(record["gradient_groups"]["B"]["would_clip"])
        for record in clients
    )
    result = {
        "client_steps": len(clients),
        "rounds": len(rounds),
        "sample_draws": sum(record["batch_size"] for record in clients),
        "sample_schedule_sha256": canonical_json_fingerprint(schedule),
        "supervision_schedule_sha256": canonical_json_fingerprint(mask_schedule),
        "loss": safe_quantiles(record["loss"] for record in clients),
        "groups": group_summaries,
        "any_group_actual_clipped_count": any_actual_clipped_count,
        "any_group_actual_clipped_fraction": safe_ratio(
            any_actual_clipped_count, len(clients)
        ),
        "any_group_would_clip_count": any_would_clip_count,
        "any_group_would_clip_fraction": safe_ratio(
            any_would_clip_count, len(clients)
        ),
        "timing_totals_seconds": timing_totals,
        "supervised_tokens": sum(
            record["batch_telemetry"]["supervised_tokens"] for record in clients
        ),
    }
    controller_rounds = [
        round_record.get("slaclip_controller") for round_record in rounds
    ]
    if any(value is not None for value in controller_rounds):
        if not all(isinstance(value, dict) for value in controller_rounds):
            raise RuntimeError("SlaClip controller telemetry is incomplete")
        controller_inputs = {
            str(value.get("controller_input")) for value in controller_rounds
        }
        if len(controller_inputs) != 1 or controller_inputs.pop() not in {
            NOISY_CONTROLLER_INPUT,
            EXACT_CONTROLLER_INPUT,
        }:
            raise RuntimeError("SlaClip controller-input telemetry is invalid")
        controller_input = str(controller_rounds[0]["controller_input"])
        controller_variants = {
            str(value.get("variant")) for value in controller_rounds
        }
        if len(controller_variants) != 1:
            raise RuntimeError("SlaClip controller-variant telemetry is invalid")
        controller_variant = controller_variants.pop()
        if controller_variant not in {
            "full_slaclip_cdf_endpoints",
            GROUPWISE_SLACLIP_VARIANT,
        }:
            raise RuntimeError("SlaClip controller variant is unsupported")
        groupwise_targets = controller_variant == GROUPWISE_SLACLIP_VARIANT
        if groupwise_targets:
            targets_by_round = [
                value.get("base_target_clipped_fraction_by_group")
                for value in controller_rounds
            ]
            if any(target != targets_by_round[0] for target in targets_by_round):
                raise RuntimeError(
                    "groupwise SlaClip target telemetry changed across rounds"
                )
            targets, resolved_groupwise = resolve_slaclip_base_targets(
                base_target_clipped_fraction_by_group=targets_by_round[0]
            )
            if not resolved_groupwise:
                raise RuntimeError("groupwise SlaClip targets are invalid")
        controller_groups: dict[str, Any] = {}
        for group in ("A", "B"):
            group_rounds = [value[group] for value in controller_rounds]
            raw_steps = [float(value["raw_log_step"]) for value in group_rounds]
            nonzero_directions = [
                -1 if value < 0.0 else 1 for value in raw_steps if value != 0.0
            ]
            direction_flips = sum(
                left != right
                for left, right in zip(
                    nonzero_directions,
                    nonzero_directions[1:],
                )
            )
            controller_groups[group] = {
                    "clip_threshold_used": safe_quantiles(
                        value["clip_threshold_used"] for value in group_rounds
                    ),
                    "next_clip_threshold": safe_quantiles(
                        value["next_clip_threshold"] for value in group_rounds
                    ),
                    "near_threshold_proxy": safe_quantiles(
                        value["near_threshold_proxy"] for value in group_rounds
                    ),
                    "near_zero_proxy": safe_quantiles(
                        value["near_zero_proxy"] for value in group_rounds
                    ),
                    "near_zero_adjusted": safe_quantiles(
                        value["near_zero_adjusted"] for value in group_rounds
                    ),
                    "remaining_non_small_gradient_fraction": safe_quantiles(
                        value["remaining_non_small_gradient_fraction"]
                        for value in group_rounds
                    ),
                    "raw_dynamic_target_clipped": safe_quantiles(
                        value["raw_dynamic_target_clipped"]
                        for value in group_rounds
                    ),
                    "clamped_dynamic_target_clipped": safe_quantiles(
                        value["clamped_dynamic_target_clipped"]
                        for value in group_rounds
                    ),
                    "dynamic_target_unclipped": safe_quantiles(
                        value["dynamic_target_unclipped"] for value in group_rounds
                    ),
                    "dynamic_target_clipped": safe_quantiles(
                        value["dynamic_target_clipped"] for value in group_rounds
                    ),
                    "controller_error": safe_quantiles(
                        value["controller_error"] for value in group_rounds
                    ),
                    "actual_clip_fraction": safe_quantiles(
                        value["actual_clip_fraction"] for value in group_rounds
                    ),
                    "actual_minus_dynamic_target_clipped": safe_quantiles(
                        value["actual_minus_dynamic_target_clipped"]
                        for value in group_rounds
                    ),
                    "actual_target_absolute_error": safe_quantiles(
                        value["actual_target_absolute_error"] for value in group_rounds
                    ),
                    "near_threshold_proxy_error": safe_quantiles(
                        value["noisy_near_threshold_minus_exact"]
                        for value in group_rounds
                    ),
                    "near_zero_proxy_error": safe_quantiles(
                        value["noisy_near_zero_minus_exact"] for value in group_rounds
                    ),
                    "cdf_error_mae": safe_quantiles(
                        value["cdf_error_mae"] for value in group_rounds
                    ),
                    "cdf_error_rmse": safe_quantiles(
                        value["cdf_error_rmse"] for value in group_rounds
                    ),
                    "cdf_error_max_abs": safe_quantiles(
                        value["cdf_error_max_abs"] for value in group_rounds
                    ),
                    "cdf_error_z_rmse": safe_quantiles(
                        value["cdf_error_z_rmse"] for value in group_rounds
                    ),
                    "oracle_dynamic_target_clipped": safe_quantiles(
                        value["oracle_dynamic_target_clipped"]
                        for value in group_rounds
                    ),
                    "oracle_raw_log_step": safe_quantiles(
                        value["oracle_raw_log_step"] for value in group_rounds
                    ),
                    "oracle_next_clip_threshold": safe_quantiles(
                        value["oracle_next_clip_threshold"] for value in group_rounds
                    ),
                    "noisy_minus_oracle_raw_log_step": safe_quantiles(
                        value["noisy_minus_oracle_raw_log_step"]
                        for value in group_rounds
                    ),
                    "noisy_oracle_log_threshold_error": safe_quantiles(
                        value["noisy_oracle_log_threshold_error"]
                        for value in group_rounds
                    ),
                    "raw_log_step": safe_quantiles(raw_steps),
                    "log_threshold_total_variation": float(
                        math.fsum(
                            abs(
                                math.log(
                                    float(value["next_clip_threshold"])
                                    / float(value["clip_threshold_used"])
                                )
                            )
                            for value in group_rounds
                        )
                    ),
                    "log_step_direction_flip_count": direction_flips,
                    "oracle_direction_agreement_count": sum(
                        bool(value["update_direction_agrees"])
                        for value in group_rounds
                    ),
                    "oracle_direction_agreement_fraction": safe_ratio(
                        sum(
                            bool(value["update_direction_agrees"])
                            for value in group_rounds
                        ),
                        len(group_rounds),
                    ),
                    "noisy_cdf_out_of_range_count": sum(
                        int(value["noisy_cdf_out_of_range_count"])
                        for value in group_rounds
                    ),
                    "noisy_cdf_out_of_range_fraction": safe_ratio(
                        sum(
                            int(value["noisy_cdf_out_of_range_count"])
                            for value in group_rounds
                        ),
                        len(group_rounds) * int(controller_rounds[0]["num_slots"]),
                    ),
                    "gamma_clamped_low_count": sum(
                        bool(value["gamma_clamped_low"]) for value in group_rounds
                    ),
                    "gamma_clamped_high_count": sum(
                        bool(value["gamma_clamped_high"]) for value in group_rounds
                    ),
                    "log_step_bounded_count": sum(
                        bool(value["log_step_was_bounded"]) for value in group_rounds
                    ),
                    "lower_bound_hits": sum(
                        bool(value["hit_min_clip_norm"]) for value in group_rounds
                    ),
                    "upper_bound_hits": sum(
                        bool(value["hit_max_clip_norm"]) for value in group_rounds
                    ),
                    "noisy_adjacent_monotonicity_violations": sum(
                        int(value["noisy_adjacent_monotonicity_violations"])
                        for value in group_rounds
                    ),
                    "exact_adjacent_monotonicity_violations": sum(
                        int(value["exact_adjacent_monotonicity_violations"])
                        for value in group_rounds
                    ),
                    "final_next_clip_threshold": group_rounds[-1][
                        "next_clip_threshold"
                    ],
                }
        controller_result = {
            "variant": controller_variant,
            "controller_input": controller_input,
            "controller_input_is_non_dp_exact": (
                controller_input == EXACT_CONTROLLER_INPUT
            ),
            "rounds": len(controller_rounds),
            "trajectory_sha256": canonical_json_fingerprint(controller_rounds),
            "groups": controller_groups,
        }
        if groupwise_targets:
            controller_result.update(
                {
                    "target_parameterization": "per_gradient_group",
                    "generalized_full_slaclip_beta": True,
                    "base_target_clipped_fraction_by_group": dict(targets),
                    "beta_by_group": dict(targets),
                }
            )
        else:
            controller_result.update(
                {
                    "base_target_clipped_fraction": controller_rounds[0][
                        "base_target_clipped_fraction"
                    ],
                    "beta": controller_rounds[0]["beta"],
                }
            )
        result["slaclip_controller"] = controller_result
    return result


def illustrative_accounting_diagnostic(
    *,
    client_partition_sizes: Sequence[int],
    batch_size: int,
    rounds: int,
    noise_multiplier: float,
    delta: float,
    contains_slaclip: bool = False,
    slaclip_num_slots: int | None = None,
    controller_input: str | None = None,
) -> dict[str, Any]:
    """Record why standard DP-SGD accounting cannot certify this mechanism."""

    reasons = [
        "The paper-literal mechanism clips one aggregate batch gradient, not per-example gradients.",
        "A and B are clipped separately and jointly released, so sensitivity is not established as C.",
        "Sampling is fixed-size without replacement, not the Poisson sampling assumed by common accountants.",
        "Publishing multiple models or paired arms requires an explicit composition analysis.",
    ]
    if contains_slaclip:
        if slaclip_num_slots is None or slaclip_num_slots < 2:
            raise ValueError(
                "a full-SlaClip accounting diagnostic requires at least two slots"
            )
        resolved_controller_input = (
            NOISY_CONTROLLER_INPUT
            if controller_input is None
            else controller_input
        )
        if resolved_controller_input not in {
            NOISY_CONTROLLER_INPUT,
            EXACT_CONTROLLER_INPUT,
        }:
            raise ValueError("invalid SlaClip accounting controller input")
        reasons.extend(
            [
                "This is a federated per-client adaptation of full SlaClip, not the paper's audited per-sample DP-SGD implementation.",
                (
                    "The controller uses noisy CDF endpoints; exact CDF and "
                    "clipping telemetry are retained only as non-DP private "
                    "diagnostics and are not controller inputs."
                    if resolved_controller_input == NOISY_CONTROLLER_INPUT
                    else "The oracle controller uses exact CDF endpoints to set "
                    "the next threshold and is explicitly NON-DP, even though "
                    "gradient noise and the noisy slack release remain enabled."
                ),
                (
                    f"The explicit K={slaclip_num_slots} release has no "
                    "independently reviewed end-to-end composition analysis "
                    "for this federated adaptation."
                ),
            ]
        )
    elif controller_input is not None:
        raise ValueError("controller_input requires contains_slaclip=True")
    return {
        "status": "NOT_CERTIFIED",
        "epsilon": None,
        "delta": delta,
        "noise_multiplier": noise_multiplier,
        "controller_input": (
            resolved_controller_input if contains_slaclip else None
        ),
        "mechanism_steps_per_client": rounds,
        "fixed_batch_sampling_rates": [
            batch_size / size for size in client_partition_sizes
        ],
        "reasons": reasons,
        "sigma_is_not_epsilon": True,
    }


def validate_checkpoint_trainer_state(
    state: dict[str, Any],
    *,
    completed_round: int,
    model_kind: str,
    config: EffectiveConfig,
    run_config_fingerprint: str,
    private_key_commitment: str,
    rng_domain: str,
    clients: Sequence[np.ndarray],
    slaclip_contract: dict[str, Any] | None = None,
) -> None:
    if not 0 < completed_round <= config.rounds:
        raise RuntimeError("checkpoint round is outside the configured run")
    expected_identity = {
        "model": model_kind,
        "method": config.method,
        "run_config_fingerprint": run_config_fingerprint,
        "private_key_commitment": private_key_commitment,
        "rng_domain": rng_domain,
    }
    for key, expected in expected_identity.items():
        if state.get(key) != expected:
            raise RuntimeError(f"checkpoint trainer-state identity mismatch: {key}")
    expected_steps = completed_round * config.num_clients
    if state.get("total_client_steps") != expected_steps:
        raise RuntimeError("checkpoint client-step count mismatch")
    for field in ("clipped_counts", "would_clip_counts"):
        counts = state.get(field)
        if not isinstance(counts, dict) or set(counts) != {"A", "B", "any"}:
            raise RuntimeError(f"checkpoint {field} is invalid")
        if any(
            not isinstance(value, int) or not 0 <= value <= expected_steps
            for value in counts.values()
        ):
            raise RuntimeError(f"checkpoint {field} values are invalid")
    evaluations = state.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise RuntimeError("checkpoint evaluation history is invalid")
    expected_evaluation_rounds = [0] + [
        round_index
        for round_index in range(1, completed_round + 1)
        if round_index % config.eval_every == 0 or round_index == config.rounds
    ]
    if any(not isinstance(evaluation, dict) for evaluation in evaluations):
        raise RuntimeError("checkpoint evaluation entries must be objects")
    if [evaluation.get("round") for evaluation in evaluations] != expected_evaluation_rounds:
        raise RuntimeError("checkpoint evaluation round sequence is invalid")
    for evaluation in evaluations:
        loss = evaluation.get("loss")
        if (
            not isinstance(loss, (int, float))
            or isinstance(loss, bool)
            or not math.isfinite(float(loss))
        ):
            raise RuntimeError("checkpoint evaluation loss is invalid")
        supervised_tokens = evaluation.get("supervised_tokens")
        correct_tokens = evaluation.get("correct_tokens")
        token_accuracy = evaluation.get("token_accuracy")
        if (
            not isinstance(supervised_tokens, int)
            or isinstance(supervised_tokens, bool)
            or supervised_tokens <= 0
            or not isinstance(correct_tokens, int)
            or isinstance(correct_tokens, bool)
            or not 0 <= correct_tokens <= supervised_tokens
            or not isinstance(token_accuracy, (int, float))
            or isinstance(token_accuracy, bool)
            or not math.isfinite(float(token_accuracy))
            or not 0.0 <= float(token_accuracy) <= 1.0
            or float(token_accuracy) != correct_tokens / supervised_tokens
        ):
            raise RuntimeError(
                "checkpoint evaluation supervised-token accuracy is invalid"
            )
        if (
            evaluation.get("token_accuracy_definition")
            != "supervised_token_top1_micro_accuracy"
        ):
            raise RuntimeError(
                "checkpoint evaluation token-accuracy definition is invalid"
            )
    sampled = state.get("sampled_unique_indices")
    if not isinstance(sampled, list) or len(sampled) != config.num_clients:
        raise RuntimeError("checkpoint sample-coverage state is invalid")
    for client_id, values in enumerate(sampled):
        if not isinstance(values, list) or any(
            not isinstance(index, int) for index in values
        ):
            raise RuntimeError("checkpoint sample-coverage indices are invalid")
        if not set(values).issubset(set(int(index) for index in clients[client_id])):
            raise RuntimeError("checkpoint sample coverage escapes its client partition")
    active_seconds = state.get("active_elapsed_seconds")
    if (
        not isinstance(active_seconds, (int, float))
        or not math.isfinite(float(active_seconds))
        or float(active_seconds) < 0
    ):
        raise RuntimeError("checkpoint active elapsed time is invalid")
    last_round = state.get("last_round_summary")
    if not isinstance(last_round, dict):
        raise RuntimeError("checkpoint last-round summary is invalid")
    expected_last_round_identity = {
        "privacy_label": PRIVACY_LABEL,
        "method": config.method,
        "model": model_kind,
        "round": completed_round,
    }
    if any(
        last_round.get(key) != expected
        for key, expected in expected_last_round_identity.items()
    ):
        raise RuntimeError("checkpoint last-round summary identity is invalid")
    mean_loss = last_round.get("mean_training_loss")
    if (
        not isinstance(mean_loss, (int, float))
        or isinstance(mean_loss, bool)
        or not math.isfinite(float(mean_loss))
    ):
        raise RuntimeError("checkpoint last-round training loss is invalid")
    shard_digest = state.get("round_shard_prefix_sha256")
    if (
        not isinstance(shard_digest, str)
        or len(shard_digest) != 64
        or any(character not in "0123456789abcdef" for character in shard_digest)
    ):
        raise RuntimeError("checkpoint round-shard digest is invalid")
    controller_state = state.get("slaclip_controller_state")
    if config.method in ADAPTIVE_METHODS:
        if slaclip_contract is None or not isinstance(controller_state, dict):
            raise RuntimeError("checkpoint SlaClip controller state is missing")
        expected_controller_identity = {
            "controller_contract_sha256": canonical_json_fingerprint(
                slaclip_contract
            ),
            "controller_input": slaclip_contract["controller_input"],
            "updates_completed": completed_round,
        }
        groupwise_targets = controller_state.get(
            "base_target_clipped_fraction_by_group"
        )
        contract_groupwise_targets = slaclip_contract["controller"].get(
            "base_target_clipped_fraction_by_group"
        )
        if contract_groupwise_targets is not None:
            expected_controller_identity[
                "base_target_clipped_fraction_by_group"
            ] = contract_groupwise_targets
        elif groupwise_targets is not None:
            raise RuntimeError(
                "checkpoint shared-beta SlaClip state contains groupwise targets"
            )
        for key, expected in expected_controller_identity.items():
            if controller_state.get(key) != expected:
                raise RuntimeError(f"checkpoint SlaClip state mismatch: {key}")
        thresholds = controller_state.get("next_clip_threshold_by_group")
        controller = slaclip_contract["controller"]
        if not isinstance(thresholds, dict) or set(thresholds) != {"A", "B"}:
            raise RuntimeError("checkpoint SlaClip thresholds are invalid")
        for group in ("A", "B"):
            value = thresholds[group]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not float(controller["c_min"])
                <= float(value)
                <= float(controller["c_max"])
            ):
                raise RuntimeError("checkpoint SlaClip threshold is out of bounds")
            last_controller = last_round.get("slaclip_controller")
            if (
                not isinstance(last_controller, dict)
                or last_controller.get(group, {}).get("next_clip_threshold") != value
            ):
                raise RuntimeError(
                    "checkpoint SlaClip threshold does not match the last shard"
                )
            if (
                contract_groupwise_targets is not None
                and last_controller[group].get(
                    "base_target_clipped_fraction"
                )
                != contract_groupwise_targets[group]
            ):
                raise RuntimeError(
                    "checkpoint groupwise SlaClip target does not match the contract"
                )
    elif controller_state is not None:
        raise RuntimeError("fixed-threshold checkpoint contains SlaClip state")


def train_one_model(
    *,
    model_kind: str,
    manifest: dict[str, Any],
    data: LoadedData,
    output_dir: Path,
    config: EffectiveConfig,
    device: torch.device,
    private_key: bytes,
    private_key_commitment: str,
    rng_domain: str,
    run_config_fingerprint: str,
    slaclip_contract: dict[str, Any] | None,
    resume: bool,
    stop_file: Path | None,
) -> dict[str, Any]:
    adaptive = config.method in ADAPTIVE_METHODS
    if adaptive != (slaclip_contract is not None):
        raise RuntimeError("SlaClip model contract mismatch")
    controller = slaclip_contract["controller"] if slaclip_contract else None
    model_seed = config.seed + (0 if model_kind == "bert" else 1_000_000)
    snapshot = model_snapshot(manifest, model_kind)
    target_modules = EXPECTED_LORA_TARGETS[model_kind]
    train_limit = 16 if config.smoke else None
    clients = partition_pool(
        data.training_pool, config.num_clients, model_seed + 11, train_limit
    )
    client_partition_sizes = [len(partition) for partition in clients]
    client_partition_sha256 = [
        int64_index_digest(partition) for partition in clients
    ]
    model_config_fingerprint = canonical_json_fingerprint(
        {
            "run_config_fingerprint": run_config_fingerprint,
            "model": model_kind,
            "snapshot": str(snapshot),
            "target_modules": target_modules,
            "client_partition_sha256": client_partition_sha256,
            "private_key_commitment": private_key_commitment,
            "rng_domain": rng_domain,
        }
    )
    diagnostics_dir = output_dir / "private_diagnostics"
    rounds_directory = diagnostics_dir / "rounds"
    notice_path = diagnostics_dir / "NON_DP_PRIVATE_DATA.txt"
    client_log_path = diagnostics_dir / "client_rounds.jsonl"
    round_log_path = diagnostics_dir / "round_summaries.jsonl"
    checkpoint_root = output_dir / "checkpoints"
    if os.path.lexists(output_dir):
        validate_private_directory(output_dir, "model output")
        if not resume:
            raise FileExistsError(f"model output already exists: {output_dir}")
        completed_summary_path = output_dir / "final_summary.json"
        if os.path.lexists(completed_summary_path):
            validate_private_regular_file(
                completed_summary_path, "model final summary"
            )
            completed = load_json_object(completed_summary_path, "model final summary")
            if completed.get("schema_version") != 2:
                raise RuntimeError("unsupported completed model summary schema")
            if completed.get("status") != "COMPLETED":
                raise RuntimeError("existing model summary is not COMPLETED")
            if completed.get("run_config_fingerprint") != run_config_fingerprint:
                raise RuntimeError("completed model configuration fingerprint mismatch")
            identity_expectations = {
                "model": model_kind,
                "method": config.method,
                "model_config_fingerprint": model_config_fingerprint,
                "client_partition_sha256": client_partition_sha256,
                "client_steps": config.rounds * config.num_clients,
            }
            for key, expected in identity_expectations.items():
                if completed.get(key) != expected:
                    raise RuntimeError(f"completed model identity mismatch: {key}")
            completed_last_round = completed.get("last_round")
            if (
                not isinstance(completed_last_round, dict)
                or completed_last_round.get("round") != config.rounds
            ):
                raise RuntimeError("completed model does not contain the final round")
            validate_private_directory(diagnostics_dir, "private diagnostics directory")
            validate_private_directory(rounds_directory, "round diagnostics directory")
            validate_private_regular_file(notice_path, "private-data notice")
            shards = read_round_shards(
                rounds_directory,
                expected_rounds=config.rounds,
                expected_model=model_kind,
                expected_method=config.method,
                expected_clients=config.num_clients,
                expected_batch_size=config.batch_size,
                slaclip_contract=slaclip_contract,
            )
            final_shard_digest = round_shard_prefix_sha256(
                rounds_directory, completed_round=config.rounds
            )
            if completed.get("round_shard_prefix_sha256") != final_shard_digest:
                raise RuntimeError("completed round diagnostics checksum mismatch")
            recomputed_behavior = behavior_summary(shards)
            if completed.get("behavior_summary") != recomputed_behavior:
                raise RuntimeError("completed behavior summary does not reconcile")
            if completed.get("last_round") != shards[-1]["round_summary"]:
                raise RuntimeError("completed last-round summary does not reconcile")
            expected_client_records = [
                record for shard in shards for record in shard["client_records"]
            ]
            if load_private_jsonl(
                client_log_path, "client-round diagnostic log"
            ) != expected_client_records:
                raise RuntimeError("completed client-round log does not reconcile")
            if load_private_jsonl(
                round_log_path, "round-summary diagnostic log"
            ) != [shard["round_summary"] for shard in shards]:
                raise RuntimeError("completed round-summary log does not reconcile")
            checkpoint = load_latest_checkpoint(
                checkpoint_root,
                expected_config_fingerprint=model_config_fingerprint,
            )
            if checkpoint is None or checkpoint.completed_round != config.rounds:
                raise RuntimeError("completed model has no final-round checkpoint")
            state = checkpoint.trainer_state
            validate_checkpoint_trainer_state(
                state,
                completed_round=checkpoint.completed_round,
                model_kind=model_kind,
                config=config,
                run_config_fingerprint=run_config_fingerprint,
                private_key_commitment=private_key_commitment,
                rng_domain=rng_domain,
                clients=clients,
                slaclip_contract=slaclip_contract,
            )
            if state.get("round_shard_prefix_sha256") != final_shard_digest:
                raise RuntimeError("completed checkpoint diagnostics checksum mismatch")
            if state.get("last_round_summary") != shards[-1]["round_summary"]:
                raise RuntimeError("completed checkpoint last-round summary mismatch")
            if state.get("evaluations") != completed.get("evaluations"):
                raise RuntimeError("completed checkpoint evaluation history mismatch")
            checkpoint_tensor_elements = sum(
                tensor.numel() for tensor in checkpoint.tensors.values()
            )
            if (
                len(checkpoint.tensors)
                != completed.get("trainable_parameter_tensors")
                or checkpoint_tensor_elements
                != completed.get("trainable_parameter_elements")
            ):
                raise RuntimeError("completed checkpoint tensor inventory mismatch")
            expected_client_steps = config.rounds * config.num_clients
            expected_counts = {
                "A": recomputed_behavior["groups"]["A"]["actual_clipped_count"],
                "B": recomputed_behavior["groups"]["B"]["actual_clipped_count"],
                "any": recomputed_behavior["any_group_actual_clipped_count"],
            }
            expected_would_counts = {
                "A": recomputed_behavior["groups"]["A"]["would_clip_count"],
                "B": recomputed_behavior["groups"]["B"]["would_clip_count"],
                "any": recomputed_behavior["any_group_would_clip_count"],
            }
            if state.get("clipped_counts") != expected_counts:
                raise RuntimeError("completed checkpoint clipping counts mismatch")
            if state.get("would_clip_counts") != expected_would_counts:
                raise RuntimeError(
                    "completed checkpoint counterfactual clipping counts mismatch"
                )
            adapter_dir = output_dir / "final_adapter"
            validate_private_directory(adapter_dir, "final adapter directory")
            adapter_file = adapter_dir / "adapter_model.safetensors"
            adapter_integrity = validate_adapter_artifact(
                adapter_file,
                expected_parameter_elements=int(
                    completed.get("trainable_parameter_elements", -1)
                ),
                expected_parameter_tensors=int(
                    completed.get("trainable_parameter_tensors", -1)
                ),
                expected_rank=config.rank,
                expected_target_modules=target_modules,
                expected_base_model_path=snapshot,
                expected_sha256=str(completed.get("adapter_sha256")),
                expected_config_sha256=str(
                    completed.get("adapter_config_sha256")
                ),
            )
            checkpoint_state_digest = canonical_adapter_state_sha256(
                checkpoint.tensors
            )
            if (
                adapter_integrity["canonical_state_sha256"]
                != checkpoint_state_digest
                or completed.get("adapter_state_sha256")
                != checkpoint_state_digest
            ):
                raise RuntimeError(
                    "completed adapter does not match the final checkpoint state"
                )
            completed_evaluations = completed.get("evaluations")
            if (
                not isinstance(completed_evaluations, list)
                or not completed_evaluations
                or completed.get("final_evaluation") != completed_evaluations[-1]
            ):
                raise RuntimeError(
                    "completed final evaluation does not match evaluation history"
                )
            expected_clipping = {
                "A": {
                    "count": expected_counts["A"],
                    "fraction": safe_ratio(
                        expected_counts["A"], expected_client_steps
                    ),
                    "would_count": expected_would_counts["A"],
                    "would_fraction": safe_ratio(
                        expected_would_counts["A"], expected_client_steps
                    ),
                },
                "B": {
                    "count": expected_counts["B"],
                    "fraction": safe_ratio(
                        expected_counts["B"], expected_client_steps
                    ),
                    "would_count": expected_would_counts["B"],
                    "would_fraction": safe_ratio(
                        expected_would_counts["B"], expected_client_steps
                    ),
                },
                "any_group": {
                    "count": expected_counts["any"],
                    "fraction": safe_ratio(
                        expected_counts["any"], expected_client_steps
                    ),
                    "would_count": expected_would_counts["any"],
                    "would_fraction": safe_ratio(
                        expected_would_counts["any"], expected_client_steps
                    ),
                },
            }
            if completed.get("clipping") != expected_clipping:
                raise RuntimeError("completed clipping summary does not reconcile")
            if adaptive:
                controller_behavior = recomputed_behavior.get("slaclip_controller")
                if not isinstance(controller_behavior, dict):
                    raise RuntimeError("completed SlaClip controller summary is missing")
                expected_slaclip_summary = {
                    "contract": slaclip_contract,
                    "final_next_clip_threshold_by_group": {
                        group: controller_behavior["groups"][group][
                            "final_next_clip_threshold"
                        ]
                        for group in ("A", "B")
                    },
                    "controller_summary": controller_behavior,
                }
                if completed.get("slaclip") != expected_slaclip_summary:
                    raise RuntimeError(
                        "completed SlaClip summary does not reconcile"
                    )
            elif "slaclip" in completed:
                raise RuntimeError("fixed-threshold summary contains SlaClip data")
            if completed.get("adapter_integrity") != adapter_integrity:
                raise RuntimeError("completed adapter integrity summary mismatch")
            return completed
    else:
        prepare_private_directory(
            output_dir, exist_ok=False, description="model output"
        )

    started = time.monotonic()
    seed_everything(model_seed)
    model, tokenizer, built_target_modules = build_model(
        model_kind, snapshot, config.rank, device
    )
    if sorted(built_target_modules) != sorted(target_modules):
        raise RuntimeError("built LoRA target modules differ from the run contract")
    groups = parameter_groups(model)
    global_state = clone_trainable_state(groups)
    trainable_elements = sum(value.numel() for value in global_state.values())
    method_spec = METHOD_SPECS[config.method]
    effective_noise_multiplier = (
        config.noise_multiplier if method_spec.gaussian_noise_enabled else 0.0
    )
    prepare_private_directory(
        diagnostics_dir,
        exist_ok=resume,
        description="private diagnostics directory",
    )
    prepare_private_directory(
        rounds_directory,
        exist_ok=resume,
        description="round diagnostics directory",
    )
    if not os.path.lexists(notice_path):
        write_new_private_text(
            notice_path,
            "NON_DP_PRIVATE_DIAGNOSTIC\n"
            "Exact samples, losses, gradients, controls, and checkpoints are not DP release artifacts.\n",
        )
    else:
        validate_private_regular_file(notice_path, "private-data notice")
    checkpoint = (
        load_latest_checkpoint(
            checkpoint_root,
            expected_config_fingerprint=model_config_fingerprint,
        )
        if resume
        else None
    )
    evaluation_seed = config.evaluation_seed + (
        0 if model_kind == "bert" else 1_000_000
    )
    if checkpoint is None:
        if resume:
            archive_round_shards_after(rounds_directory, completed_round=0)
        evaluations: list[dict[str, Any]] = []
        initial_eval = evaluate(
            model,
            tokenizer,
            model_kind,
            data.validation,
            data.validation_indices,
            config,
            device,
            evaluation_seed,
        )
        evaluations.append(
            {
                "round": 0,
                **initial_eval,
                "effective_lora": effective_lora_statistics(groups),
            }
        )
        clipped_counts = {"A": 0, "B": 0, "any": 0}
        would_clip_counts = {"A": 0, "B": 0, "any": 0}
        total_client_steps = 0
        sampled_unique_indices = [set() for _ in clients]
        completed_round = 0
        previous_active_seconds = 0.0
        checkpointed_shard_digest: str | None = None
        last_round_summary: dict[str, Any] | None = None
        current_clip_thresholds = dict(config.clip_norm_by_group)
    else:
        if set(checkpoint.tensors) != set(global_state):
            raise RuntimeError("checkpoint LoRA tensor names do not match the model")
        for name, tensor in checkpoint.tensors.items():
            if tensor.shape != global_state[name].shape:
                raise RuntimeError(f"checkpoint tensor shape mismatch for {name}")
        global_state = {
            name: tensor.detach().float().clone()
            for name, tensor in checkpoint.tensors.items()
        }
        restore_trainable_state(groups, global_state)
        state = checkpoint.trainer_state
        validate_checkpoint_trainer_state(
            state,
            completed_round=checkpoint.completed_round,
            model_kind=model_kind,
            config=config,
            run_config_fingerprint=run_config_fingerprint,
            private_key_commitment=private_key_commitment,
            rng_domain=rng_domain,
            clients=clients,
            slaclip_contract=slaclip_contract,
        )
        evaluations = list(state["evaluations"])
        clipped_counts = dict(state["clipped_counts"])
        would_clip_counts = dict(state["would_clip_counts"])
        total_client_steps = int(state["total_client_steps"])
        sampled_unique_indices = [
            set(int(index) for index in values)
            for values in state["sampled_unique_indices"]
        ]
        completed_round = checkpoint.completed_round
        previous_active_seconds = float(state["active_elapsed_seconds"])
        checkpointed_shard_digest = str(state["round_shard_prefix_sha256"])
        actual_shard_digest = round_shard_prefix_sha256(
            rounds_directory, completed_round=completed_round
        )
        if actual_shard_digest != checkpointed_shard_digest:
            raise RuntimeError("checkpointed round diagnostics checksum mismatch")
        last_round_summary = state.get("last_round_summary")
        if adaptive:
            saved_controller_state = state["slaclip_controller_state"]
            current_clip_thresholds = {
                group: float(
                    saved_controller_state["next_clip_threshold_by_group"][group]
                )
                for group in ("A", "B")
            }
        else:
            current_clip_thresholds = dict(config.clip_norm_by_group)
        last_shard_path = rounds_directory / f"round-{completed_round:05d}.json"
        validate_private_regular_file(last_shard_path, "round diagnostic shard")
        last_shard = load_json_object(last_shard_path, "round diagnostic shard")
        if (
            last_shard.get("schema_version") != 2
            or last_shard.get("round") != completed_round
            or last_shard.get("model") != model_kind
            or last_shard.get("method") != config.method
            or last_shard.get("round_summary") != last_round_summary
        ):
            raise RuntimeError("checkpoint last-round diagnostics do not reconcile")
        archive_round_shards_after(
            rounds_directory, completed_round=completed_round
        )

    for round_index in range(completed_round + 1, config.rounds + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        round_started = synchronized_time(device)
        previous_global_state = global_state
        aggregate = empty_state_like(previous_global_state)
        components = empty_mechanism_components(groups)
        records: list[dict[str, Any]] = []
        for client_id, client_indices in enumerate(clients):
            client_started = synchronized_time(device)
            restore_started = client_started
            restore_trainable_state(groups, previous_global_state)
            restore_ended = synchronized_time(device)
            model.zero_grad(set_to_none=True)

            sampling_started = time.perf_counter()
            replace = len(client_indices) < config.batch_size
            sampling_seed = derive_seed(
                private_key,
                rng_domain,
                "batch_sampling",
                model_kind,
                round_index,
                client_id,
            )
            sampled = np.random.default_rng(sampling_seed).choice(
                client_indices, size=config.batch_size, replace=replace
            ).astype(np.int64, copy=False)
            sampled_unique_indices[client_id].update(int(index) for index in sampled)
            sampling_ended = time.perf_counter()

            tokenization_started = synchronized_time(device)
            mask_seed = derive_seed(
                private_key,
                rng_domain,
                "supervision_mask",
                model_kind,
                round_index,
                client_id,
            )
            batch, batch_telemetry = make_batch(
                data.training.texts(sampled),
                tokenizer,
                model_kind,
                config.max_seq_length,
                device,
                mask_seed,
            )
            tokenization_ended = synchronized_time(device)

            forward_started = tokenization_ended
            forward_seed = derive_seed(
                private_key,
                rng_domain,
                "model_stochasticity",
                model_kind,
                round_index,
                client_id,
            )
            seed_model_stochasticity(forward_seed, device)
            loss = equal_record_loss(model, batch, model_kind)
            loss_value = float(loss.detach().float().item())
            forward_ended = synchronized_time(device)
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite loss at round {round_index}, client {client_id}"
                )
            backward_started = forward_ended
            loss.backward()
            backward_ended = synchronized_time(device)

            mechanism_started = backward_ended
            noise_scope = (
                None if config.pair_noise_across_methods else config.method
            )
            noise_seed = derive_seed(
                private_key,
                rng_domain,
                "gaussian_noise",
                model_kind,
                round_index,
                client_id,
                method_scope=noise_scope,
            )
            noise_generator = torch.Generator(device=device.type).manual_seed(
                noise_seed
            )
            slack_noise_generators: dict[str, torch.Generator] | None = None
            if adaptive:
                slack_noise_generators = {
                    group: torch.Generator(device=device.type).manual_seed(
                        derive_seed(
                            private_key,
                            rng_domain,
                            f"slaclip_slack_noise_{group}",
                            model_kind,
                            round_index,
                            client_id,
                            method_scope=str(
                                controller["slack_noise_method_scope"]
                            ),
                        )
                    )
                    for group in ("A", "B")
                }
            group_stats = clip_noise_and_step(
                groups,
                clip_norm_by_group=current_clip_thresholds,
                noise_multiplier=effective_noise_multiplier,
                learning_rate=config.learning_rate,
                generator=noise_generator,
                apply_clipping=method_spec.clipping_enabled,
                slaclip_num_slots=(
                    int(controller["num_slots"]) if controller is not None else None
                ),
                slack_noise_generators=slack_noise_generators,
                component_accumulators=components,
                component_weight=1.0 / config.num_clients,
            )
            mechanism_ended = synchronized_time(device)

            aggregation_started = mechanism_ended
            accumulate_state(aggregate, groups, 1.0 / config.num_clients)
            aggregation_ended = synchronized_time(device)
            record = {
                "privacy_label": PRIVACY_LABEL,
                "method": config.method,
                "model": model_kind,
                "round": round_index,
                "client": client_id,
                "batch_size": config.batch_size,
                "sample_indices": [int(index) for index in sampled],
                "sample_indices_sha256": int64_index_digest(sampled),
                "loss": loss_value,
                "batch_telemetry": batch_telemetry,
                "gradient_groups": group_stats,
                "timings_seconds": {
                    "restore_global_state": restore_ended - restore_started,
                    "batch_sampling": sampling_ended - sampling_started,
                    "tokenization_and_h2d": tokenization_ended
                    - tokenization_started,
                    "forward_and_loss": forward_ended - forward_started,
                    "backward": backward_ended - backward_started,
                    "clipping_noise_and_local_step": mechanism_ended
                    - mechanism_started,
                    "fedavg_state_transfer": aggregation_ended
                    - aggregation_started,
                    "client_total": aggregation_ended - client_started,
                },
            }
            records.append(record)
            total_client_steps += 1
            for group in ("A", "B"):
                clipped_counts[group] += int(bool(group_stats[group]["clipped"]))
                would_clip_counts[group] += int(
                    bool(group_stats[group]["would_clip"])
                )
            clipped_counts["any"] += int(
                bool(group_stats["A"]["clipped"])
                or bool(group_stats["B"]["clipped"])
            )
            would_clip_counts["any"] += int(
                bool(group_stats["A"]["would_clip"])
                or bool(group_stats["B"]["would_clip"])
            )
            model.zero_grad(set_to_none=True)
            del batch, loss

        global_state = aggregate
        restore_trainable_state(groups, global_state)
        federated_update = round_update_statistics(
            previous_global_state,
            global_state,
            components,
            learning_rate=config.learning_rate,
        )
        training_ended = synchronized_time(device)
        last_round_summary = {
            "privacy_label": PRIVACY_LABEL,
            "method": config.method,
            "model": model_kind,
            "round": round_index,
            "training_elapsed_seconds": training_ended - round_started,
            **aggregate_round_statistics(records),
            "federated_update": federated_update,
            "sampling_coverage": [
                {
                    "client": client_id,
                    "partition_records": len(clients[client_id]),
                    "draws_so_far": round_index * config.batch_size,
                    "unique_records_so_far": len(sampled_unique_indices[client_id]),
                    "coverage_fraction": safe_ratio(
                        len(sampled_unique_indices[client_id]), len(clients[client_id])
                    ),
                    "repeat_fraction": safe_ratio(
                        round_index * config.batch_size
                        - len(sampled_unique_indices[client_id]),
                        round_index * config.batch_size,
                    ),
                }
                for client_id in range(config.num_clients)
            ],
        }
        if adaptive:
            assert controller is not None
            controller_summary = slaclip_round_controller_summary(
                records,
                clip_thresholds=current_clip_thresholds,
                num_slots=int(controller["num_slots"]),
                eta=float(controller["eta"]),
                controller_input=str(controller["controller_input"]),
                c_min=float(controller["c_min"]),
                c_max=float(controller["c_max"]),
                epsilon=float(controller["epsilon"]),
                **slaclip_controller_target_arguments(controller),
            )
            last_round_summary["slaclip_controller"] = controller_summary
            current_clip_thresholds = {
                group: float(controller_summary[group]["next_clip_threshold"])
                for group in ("A", "B")
            }
        if round_index % config.eval_every == 0 or round_index == config.rounds:
            evaluation_started = synchronized_time(device)
            evaluation = evaluate(
                model,
                tokenizer,
                model_kind,
                data.validation,
                data.validation_indices,
                config,
                device,
                evaluation_seed,
            )
            evaluation["effective_lora"] = effective_lora_statistics(groups)
            evaluation["elapsed_seconds"] = (
                synchronized_time(device) - evaluation_started
            )
            evaluations.append({"round": round_index, **evaluation})
            last_round_summary["validation"] = evaluation
        last_round_summary["resource_telemetry"] = resource_telemetry(device)
        last_round_summary["total_elapsed_seconds"] = (
            synchronized_time(device) - round_started
        )
        shard = {
            "schema_version": 2,
            "privacy_label": PRIVACY_LABEL,
            "method": config.method,
            "model": model_kind,
            "round": round_index,
            "client_records": records,
            "round_summary": last_round_summary,
        }
        shard_path = rounds_directory / f"round-{round_index:05d}.json"
        if os.path.lexists(shard_path):
            raise RuntimeError(f"refusing to overwrite a round diagnostic shard: {shard_path}")
        atomic_json(shard_path, shard)

        stop_requested = stop_file is not None and stop_file.exists()
        if (
            round_index % config.checkpoint_every == 0
            or round_index == config.rounds
            or stop_requested
        ):
            checkpointed_shard_digest = round_shard_prefix_sha256(
                rounds_directory, completed_round=round_index
            )
            write_checkpoint(
                checkpoint_root,
                completed_round=round_index,
                tensors=global_state,
                config_fingerprint=model_config_fingerprint,
                trainer_state={
                    "model": model_kind,
                    "method": config.method,
                    "run_config_fingerprint": run_config_fingerprint,
                    "private_key_commitment": private_key_commitment,
                    "rng_domain": rng_domain,
                    "evaluations": evaluations,
                    "clipped_counts": clipped_counts,
                    "would_clip_counts": would_clip_counts,
                    "total_client_steps": total_client_steps,
                    "sampled_unique_indices": [
                        sorted(indices) for indices in sampled_unique_indices
                    ],
                    "last_round_summary": last_round_summary,
                    "round_shard_prefix_sha256": checkpointed_shard_digest,
                    "active_elapsed_seconds": previous_active_seconds
                    + (time.monotonic() - started),
                    **(
                        {
                            "slaclip_controller_state": {
                                "controller_contract_sha256": (
                                    canonical_json_fingerprint(slaclip_contract)
                                ),
                                "controller_input": slaclip_contract[
                                    "controller_input"
                                ],
                                "updates_completed": round_index,
                                "next_clip_threshold_by_group": dict(
                                    current_clip_thresholds
                                ),
                                **(
                                    {
                                        "base_target_clipped_fraction_by_group": dict(
                                            slaclip_contract["controller"][
                                                "base_target_clipped_fraction_by_group"
                                            ]
                                        )
                                    }
                                    if "base_target_clipped_fraction_by_group"
                                    in slaclip_contract["controller"]
                                    else {}
                                ),
                            }
                        }
                        if adaptive
                        else {}
                    ),
                },
            )
        atomic_json(
            output_dir / "progress.json",
            {
                "schema_version": 2,
                "status": "CHECKPOINTED_STOP" if stop_requested else "RUNNING",
                "method": config.method,
                "model": model_kind,
                "round": round_index,
                "rounds": config.rounds,
                "mean_training_loss": last_round_summary["mean_training_loss"],
                "actual_clip_fraction": {
                    group: last_round_summary[group]["clipped_fraction"]
                    for group in ("A", "B")
                },
                "would_clip_fraction": {
                    group: last_round_summary[group]["would_clip_fraction"]
                    for group in ("A", "B")
                },
                **(
                    {
                        "slaclip": {
                            "controller_input": slaclip_contract[
                                "controller_input"
                            ],
                            **(
                                {
                                    "variant": GROUPWISE_SLACLIP_VARIANT,
                                    "target_parameterization": (
                                        "per_gradient_group"
                                    ),
                                    "generalized_full_slaclip_beta": True,
                                    "base_target_clipped_fraction_by_group": dict(
                                        slaclip_contract["controller"][
                                            "base_target_clipped_fraction_by_group"
                                        ]
                                    ),
                                }
                                if "base_target_clipped_fraction_by_group"
                                in slaclip_contract["controller"]
                                else {}
                            ),
                            **{
                                group: {
                                "base_target_clipped_fraction": last_round_summary[
                                    "slaclip_controller"
                                ][group]["base_target_clipped_fraction"],
                                "clip_threshold_used": last_round_summary[
                                    "slaclip_controller"
                                ][group]["clip_threshold_used"],
                                "next_clip_threshold": current_clip_thresholds[group],
                                "near_threshold_proxy": last_round_summary[
                                    "slaclip_controller"
                                ][group]["near_threshold_proxy"],
                                "near_zero_proxy": last_round_summary[
                                    "slaclip_controller"
                                ][group]["near_zero_proxy"],
                                "remaining_non_small_gradient_fraction": (
                                    last_round_summary["slaclip_controller"][group][
                                        "remaining_non_small_gradient_fraction"
                                    ]
                                ),
                                "raw_dynamic_target_clipped": last_round_summary[
                                    "slaclip_controller"
                                ][group]["raw_dynamic_target_clipped"],
                                "clamped_dynamic_target_clipped": (
                                    last_round_summary["slaclip_controller"][group][
                                        "clamped_dynamic_target_clipped"
                                    ]
                                ),
                                "dynamic_target_unclipped": last_round_summary[
                                    "slaclip_controller"
                                ][group]["dynamic_target_unclipped"],
                                "dynamic_target_clipped": last_round_summary[
                                    "slaclip_controller"
                                ][group]["dynamic_target_clipped"],
                                }
                                for group in ("A", "B")
                            },
                        }
                    }
                    if adaptive
                    else {}
                ),
                "updated_at_utc": utc_now(),
            },
        )
        print(
            json.dumps(
                {
                    "method": config.method,
                    "model": model_kind,
                    "round": round_index,
                    "rounds": config.rounds,
                    "mean_loss": last_round_summary["mean_training_loss"],
                    "A_clip_fraction": last_round_summary["A"]["clipped_fraction"],
                    "B_clip_fraction": last_round_summary["B"]["clipped_fraction"],
                    **(
                        {
                            "A_clip_threshold_next": current_clip_thresholds["A"],
                            "B_clip_threshold_next": current_clip_thresholds["B"],
                        }
                        if adaptive
                        else {}
                    ),
                    "elapsed_seconds": last_round_summary["total_elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if stop_requested:
            raise GracefulStop(
                f"stop requested after committed round {round_index} of {model_kind}"
            )

    restore_trainable_state(groups, global_state)
    adapter_dir = save_final_adapter_atomically(model, output_dir, resume=resume)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_integrity = validate_adapter_artifact(
        adapter_file,
        expected_parameter_elements=trainable_elements,
        expected_parameter_tensors=len(global_state),
        expected_rank=config.rank,
        expected_target_modules=target_modules,
        expected_base_model_path=snapshot,
    )
    global_state_digest = canonical_adapter_state_sha256(global_state)
    if adapter_integrity["canonical_state_sha256"] != global_state_digest:
        raise RuntimeError("saved adapter does not match the final global state")
    shards = read_round_shards(
        rounds_directory,
        expected_rounds=config.rounds,
        expected_model=model_kind,
        expected_method=config.method,
        expected_clients=config.num_clients,
        expected_batch_size=config.batch_size,
        slaclip_contract=slaclip_contract,
    )
    final_shard_digest = round_shard_prefix_sha256(
        rounds_directory, completed_round=config.rounds
    )
    if checkpointed_shard_digest != final_shard_digest:
        raise RuntimeError("final round diagnostics do not match the final checkpoint")
    client_records = [
        record for shard in shards for record in shard["client_records"]
    ]
    round_summaries = [shard["round_summary"] for shard in shards]
    atomic_jsonl(client_log_path, client_records)
    atomic_jsonl(round_log_path, round_summaries)
    behavior = behavior_summary(shards)
    expected_client_steps = config.rounds * config.num_clients
    if total_client_steps != expected_client_steps or len(client_records) != expected_client_steps:
        raise RuntimeError("client-step count does not reconcile with rounds times clients")
    for group in ("A", "B"):
        if behavior["groups"][group]["actual_clipped_count"] != clipped_counts[group]:
            raise RuntimeError(f"{group} clipping count does not reconcile")
        if behavior["groups"][group]["would_clip_count"] != would_clip_counts[group]:
            raise RuntimeError(f"{group} counterfactual clipping count does not reconcile")
    if behavior["any_group_actual_clipped_count"] != clipped_counts["any"]:
        raise RuntimeError("any-group clipping count does not reconcile")
    if behavior["any_group_would_clip_count"] != would_clip_counts["any"]:
        raise RuntimeError("any-group counterfactual clipping count does not reconcile")
    if not evaluations or evaluations[-1].get("round") != config.rounds:
        raise RuntimeError("final evaluation is missing")
    final_evaluation = evaluations[-1]
    elapsed = previous_active_seconds + (time.monotonic() - started)
    summary = {
        "schema_version": 2,
        "status": "COMPLETED",
        "privacy_label": PRIVACY_LABEL,
        "method": config.method,
        "method_spec": asdict(method_spec),
        "privacy_claim": False,
        "run_config_fingerprint": run_config_fingerprint,
        "model_config_fingerprint": model_config_fingerprint,
        "model": model_kind,
        "base_model": EXPECTED_MODELS[model_kind],
        "snapshot_path": str(snapshot),
        "objective": "masked_lm" if model_kind == "bert" else "causal_lm",
        "target_modules": target_modules,
        "trainable_parameter_tensors": len(global_state),
        "trainable_parameter_elements": trainable_elements,
        "client_partition_sizes": client_partition_sizes,
        "client_partition_sha256": client_partition_sha256,
        "client_steps": total_client_steps,
        "clipping": {
            "A": {
                "count": clipped_counts["A"],
                "fraction": safe_ratio(clipped_counts["A"], total_client_steps),
                "would_count": would_clip_counts["A"],
                "would_fraction": safe_ratio(
                    would_clip_counts["A"], total_client_steps
                ),
            },
            "B": {
                "count": clipped_counts["B"],
                "fraction": safe_ratio(clipped_counts["B"], total_client_steps),
                "would_count": would_clip_counts["B"],
                "would_fraction": safe_ratio(
                    would_clip_counts["B"], total_client_steps
                ),
            },
            "any_group": {
                "count": clipped_counts["any"],
                "fraction": safe_ratio(clipped_counts["any"], total_client_steps),
                "would_count": would_clip_counts["any"],
                "would_fraction": safe_ratio(
                    would_clip_counts["any"], total_client_steps
                ),
            },
        },
        "behavior_summary": behavior,
        **(
            {
                "slaclip": {
                    "contract": slaclip_contract,
                    "final_next_clip_threshold_by_group": dict(
                        current_clip_thresholds
                    ),
                    "controller_summary": behavior["slaclip_controller"],
                }
            }
            if adaptive
            else {}
        ),
        "round_shard_prefix_sha256": final_shard_digest,
        "evaluations": evaluations,
        "final_evaluation": final_evaluation,
        "last_round": last_round_summary,
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": adapter_integrity["sha256"],
        "adapter_config_sha256": adapter_integrity["config_sha256"],
        "adapter_state_sha256": global_state_digest,
        "adapter_integrity": adapter_integrity,
        "privacy_randomness": {
            "scheme": "private_hmac_sha256_counter_streams",
            "stateless_purposes": [
                "batch_sampling",
                "supervision_mask",
                "model_stochasticity",
                "gaussian_noise",
            ]
            + (
                ["slaclip_slack_noise_A", "slaclip_slack_noise_B"]
                if adaptive
                else []
            ),
            "private_key_commitment": private_key_commitment,
            "rng_domain": rng_domain,
            "pair_noise_across_methods": config.pair_noise_across_methods,
            "slaclip_slack_noise_method_scope": (
                str(controller["slack_noise_method_scope"])
                if adaptive and controller is not None
                else None
            ),
            "raw_key_or_derived_seeds_logged": False,
        },
        "privacy_accounting": illustrative_accounting_diagnostic(
            client_partition_sizes=client_partition_sizes,
            batch_size=config.batch_size,
            rounds=config.rounds,
            noise_multiplier=effective_noise_multiplier,
            delta=config.delta,
            contains_slaclip=adaptive,
            slaclip_num_slots=(
                int(controller["num_slots"])
                if adaptive and controller is not None
                else None
            ),
            controller_input=(
                str(controller["controller_input"])
                if adaptive and controller is not None
                else None
            ),
        ),
        "elapsed_seconds": elapsed,
        "completed_at_utc": utc_now(),
    }
    atomic_json(output_dir / "final_summary.json", summary)
    atomic_json(
        output_dir / "progress.json",
        {
            "schema_version": 2,
            "status": "COMPLETED",
            "method": config.method,
            "controller_input": (
                slaclip_contract["controller_input"]
                if slaclip_contract is not None
                else None
            ),
            "model": model_kind,
            "round": config.rounds,
            "rounds": config.rounds,
            "completed_at_utc": summary["completed_at_utc"],
        },
    )
    del model, tokenizer, groups, global_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def validate_completed_root_summary(
    value: dict[str, Any],
    *,
    config: EffectiveConfig,
    run_config_fingerprint: str,
    results: dict[str, Any],
) -> None:
    expected_identity = {
        "schema_version": 2,
        "status": "COMPLETED",
        "privacy_label": PRIVACY_LABEL,
        "contains_slaclip": config.method in ADAPTIVE_METHODS,
        "method": config.method,
        "method_spec": asdict(METHOD_SPECS[config.method]),
        "run_config_fingerprint": run_config_fingerprint,
        "reproduction_claim_level": 1,
        "paper_result_reproduced": False,
        "paper_benchmarks_evaluated": False,
        "privacy_claim": False,
        "models": results,
        "sample_schedule_sha256_by_model": {
            model: result["behavior_summary"]["sample_schedule_sha256"]
            for model, result in results.items()
        },
    }
    for key, expected in expected_identity.items():
        if value.get(key) != expected:
            raise RuntimeError(f"completed root summary mismatch: {key}")
    elapsed = value.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
    ):
        raise RuntimeError("completed root summary elapsed time is invalid")
    completed_at = value.get("completed_at_utc")
    if not isinstance(completed_at, str) or not completed_at:
        raise RuntimeError("completed root summary timestamp is invalid")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated DP-LoRA reconstruction with optional full SlaClip"
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["bert", "gpt2"],
        default=["bert", "gpt2"],
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--method",
        choices=sorted(METHOD_SPECS),
        default="paper_dp_lora",
    )
    parser.add_argument("--bert-model", default=EXPECTED_MODELS["bert"]["repo_id"])
    parser.add_argument("--bert-revision", default=EXPECTED_MODELS["bert"]["revision"])
    parser.add_argument("--gpt2-model", default=EXPECTED_MODELS["gpt2"]["repo_id"])
    parser.add_argument("--gpt2-revision", default=EXPECTED_MODELS["gpt2"]["revision"])
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--noise-multiplier", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument(
        "--clip-norm-a",
        type=float,
        help=(
            "LoRA-A clipping threshold for a groupwise fixed threshold or "
            "groupwise adaptive initial threshold; requires --clip-norm-b"
        ),
    )
    parser.add_argument(
        "--clip-norm-b",
        type=float,
        help=(
            "LoRA-B clipping threshold for a groupwise fixed threshold or "
            "groupwise adaptive initial threshold; requires --clip-norm-a"
        ),
    )
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-split-seed", type=int, default=1729)
    parser.add_argument("--evaluation-seed", type=int, default=2718)
    parser.add_argument("--max-validation-records", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--data-protocol",
        choices=["paper_union_minus_fixed_holdout"],
        default="paper_union_minus_fixed_holdout",
    )
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--slaclip-eta", type=float, default=0.2)
    parser.add_argument(
        "--slaclip-base-target-clipped-fraction",
        type=float,
        help=(
            "full-SlaClip base clipped-fraction target before noisy near-zero "
            "CDF modulation "
            f"(default: {DEFAULT_BASE_TARGET_CLIPPED_FRACTION:g})"
        ),
    )
    parser.add_argument(
        "--slaclip-beta",
        type=float,
        help=(
            "compatibility alias for "
            "--slaclip-base-target-clipped-fraction"
        ),
    )
    parser.add_argument(
        "--slaclip-base-target-clipped-fraction-a",
        type=float,
        help=(
            "canonical generalized full-SlaClip beta for LoRA gradient group A; "
            "requires the corresponding B option and excludes shared-beta options"
        ),
    )
    parser.add_argument(
        "--slaclip-base-target-clipped-fraction-b",
        type=float,
        help=(
            "canonical generalized full-SlaClip beta for LoRA gradient group B; "
            "requires the corresponding A option and excludes shared-beta options"
        ),
    )
    parser.add_argument("--slaclip-num-slots", type=int, default=15)
    parser.add_argument("--slaclip-c-min", type=float, default=0.1)
    parser.add_argument("--slaclip-c-max", type=float, default=50.0)
    parser.add_argument(
        "--slaclip-baseline-calibration-lock-sha256",
        help=(
            "immutable selection-lock digest when beta hyperparameters were "
            "derived from exact, non-DP fixed-baseline diagnostics"
        ),
    )
    parser.add_argument(
        "--acknowledge-slaclip-baseline-calibration-is-non-dp",
        action="store_true",
        help=(
            "acknowledge that baseline-derived SlaClip hyperparameters used "
            "exact private diagnostics during development"
        ),
    )
    parser.add_argument("--private-rng-key", type=Path)
    parser.add_argument("--rng-domain")
    parser.add_argument("--pair-noise-across-methods", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--acknowledge-non-dp-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    if (args.clip_norm_a is None) != (args.clip_norm_b is None):
        parser.error("--clip-norm-a and --clip-norm-b must be supplied together")
    if args.clip_norm_a is not None and (
        not math.isfinite(args.clip_norm_a)
        or not math.isfinite(args.clip_norm_b)
        or args.clip_norm_a <= 0
        or args.clip_norm_b <= 0
    ):
        parser.error("--clip-norm-a and --clip-norm-b must be finite and positive")
    canonical = args.slaclip_base_target_clipped_fraction
    alias = args.slaclip_beta
    group_a = args.slaclip_base_target_clipped_fraction_a
    group_b = args.slaclip_base_target_clipped_fraction_b
    if (group_a is None) != (group_b is None):
        parser.error(
            "--slaclip-base-target-clipped-fraction-a and "
            "--slaclip-base-target-clipped-fraction-b must be supplied together"
        )
    if group_a is not None and (canonical is not None or alias is not None):
        parser.error(
            "groupwise SlaClip base targets cannot be combined with "
            "--slaclip-base-target-clipped-fraction or --slaclip-beta"
        )
    if canonical is not None and alias is not None and canonical != alias:
        parser.error(
            "--slaclip-base-target-clipped-fraction and --slaclip-beta "
            "specify different values"
        )
    try:
        if group_a is not None:
            resolved_by_group, _ = resolve_slaclip_base_targets(
                base_target_clipped_fraction_by_group={
                    "A": group_a,
                    "B": group_b,
                }
            )
            resolved = None
        else:
            resolved_by_group = None
            resolved = resolve_base_target_clipped_fraction(
                base_target_clipped_fraction=canonical,
                beta=alias,
            )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    args.slaclip_base_target_clipped_fraction = resolved
    # Preserve the old Namespace field for callers that inspect requested
    # arguments while making the canonical name authoritative internally.
    args.slaclip_beta = resolved
    args.slaclip_base_target_clipped_fraction_by_group = resolved_by_group
    args.slaclip_base_target_clipped_fraction_source = (
        "groupwise_canonical_cli"
        if resolved_by_group is not None
        else "canonical_and_compatible_alias"
        if canonical is not None and alias is not None
        else "canonical_cli"
        if canonical is not None
        else "compatibility_alias"
        if alias is not None
        else "default"
    )
    calibration_lock = args.slaclip_baseline_calibration_lock_sha256
    calibration_acknowledged = (
        args.acknowledge_slaclip_baseline_calibration_is_non_dp
    )
    if (calibration_lock is None) != (not calibration_acknowledged):
        parser.error(
            "--slaclip-baseline-calibration-lock-sha256 and "
            "--acknowledge-slaclip-baseline-calibration-is-non-dp must be "
            "supplied together"
        )
    if calibration_lock is not None and (
        len(calibration_lock) != 64
        or any(character not in "0123456789abcdef" for character in calibration_lock)
    ):
        parser.error(
            "--slaclip-baseline-calibration-lock-sha256 must be 64 lowercase "
            "hexadecimal characters"
        )
    return args


def verify_requested_models(args: argparse.Namespace) -> None:
    requested = {
        "bert": (args.bert_model, args.bert_revision),
        "gpt2": (args.gpt2_model, args.gpt2_revision),
    }
    for model_kind, (repo_id, revision) in requested.items():
        expected = EXPECTED_MODELS[model_kind]
        if repo_id != expected["repo_id"] or revision != expected["revision"]:
            raise RuntimeError(
                f"{model_kind} must stay pinned to "
                f"{expected['repo_id']}@{expected['revision']}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    args.input_manifest = args.input_manifest.expanduser().resolve()
    manifest = validate_input_manifest(args.input_manifest)
    verify_requested_models(args)
    summary = manifest_summary(args.input_manifest, manifest)
    if args.check_inputs:
        preflight = deep_input_preflight(manifest)
        print(
            json.dumps(
                {
                    "status": "IMMUTABLE_INPUT_BYTES_AND_RUNTIME_PREFLIGHT_VERIFIED",
                    "preflight": preflight,
                    **summary,
                },
                indent=2,
                default=json_default,
                allow_nan=False,
            )
        )
        return
    if not args.acknowledge_non_dp_diagnostics:
        raise SystemExit(
            "--acknowledge-non-dp-diagnostics is required because exact client "
            "losses and gradient norms are written"
        )
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --check-inputs is used")
    if args.private_rng_key is None:
        raise SystemExit("--private-rng-key is required for training")
    if not isinstance(args.rng_domain, str) or not args.rng_domain.strip():
        raise SystemExit("--rng-domain must be a non-empty run/pair identifier")
    output_dir = args.output_dir.expanduser().resolve()
    args.private_rng_key = args.private_rng_key.expanduser().resolve()
    args.stop_file = (
        args.stop_file.expanduser().resolve() if args.stop_file is not None else None
    )
    if args.resume:
        try:
            validate_private_run_directory(output_dir)
        except FileNotFoundError as error:
            raise SystemExit(
                f"resume output directory does not exist: {output_dir}"
            ) from error
    elif os.path.lexists(output_dir):
        raise SystemExit(f"refusing to overwrite existing output: {output_dir}")
    config = make_effective_config(args)
    validate_config(config)
    if (
        args.slaclip_baseline_calibration_lock_sha256 is not None
        and config.method not in ADAPTIVE_METHODS
    ):
        raise SystemExit(
            "baseline-derived SlaClip calibration provenance is only valid "
            "for an adaptive SlaClip method"
        )
    if config.method in {"paper_dp_lora", *ADAPTIVE_METHODS} and config.noise_multiplier <= 0:
        raise SystemExit(f"{config.method} requires --noise-multiplier > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    try:
        configure_deterministic_execution(args.device)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    device = torch.device(args.device)
    if len(set(args.models)) != len(args.models):
        raise SystemExit("--models must not contain duplicates")
    repo = repository_state()
    if repo.get("dirty") is True:
        raise SystemExit("repository is dirty; refusing a formalized run")
    repository_root = Path(str(repo.get("root"))).resolve()
    if output_dir == repository_root or repository_root in output_dir.parents:
        raise SystemExit("output directory must be outside the Git worktree")

    slaclip_contract: dict[str, Any] | None = None
    if config.method in ADAPTIVE_METHODS:
        controller_input = CONTROLLER_INPUT_BY_METHOD[config.method]
        if args.slaclip_base_target_clipped_fraction_by_group is not None:
            base_targets, groupwise_targets = resolve_slaclip_base_targets(
                base_target_clipped_fraction_by_group=(
                    args.slaclip_base_target_clipped_fraction_by_group
                )
            )
        else:
            base_targets, groupwise_targets = resolve_slaclip_base_targets(
                base_target_clipped_fraction=(
                    args.slaclip_base_target_clipped_fraction
                ),
                beta=args.slaclip_beta,
            )
        if args.slaclip_num_slots < 2:
            raise SystemExit(
                "--slaclip-num-slots must be at least two for the full "
                "two-endpoint controller"
            )
        num_slots = args.slaclip_num_slots
        automatic_release_num_slots = automatic_num_slots(
            config.num_clients, config.noise_multiplier
        )
        normalized_proxy_noise_std = float(
            config.noise_multiplier
            * math.sqrt(num_slots / config.num_clients)
        )
        exceeds_automatic_release_bound = bool(
            args.slaclip_num_slots > 0
            and num_slots > automatic_release_num_slots
        )
        initial_clip_norms = config.clip_norm_by_group
        if (
            not math.isfinite(args.slaclip_eta)
            or args.slaclip_eta < 0
            or any(
                not math.isfinite(target) or not 0.0 <= target <= 1.0
                for target in base_targets.values()
            )
            or not math.isfinite(args.slaclip_c_min)
            or not math.isfinite(args.slaclip_c_max)
            or args.slaclip_c_min <= 0
            or args.slaclip_c_max < args.slaclip_c_min
            or any(
                not args.slaclip_c_min <= value <= args.slaclip_c_max
                for value in initial_clip_norms.values()
            )
        ):
            raise SystemExit(
                "invalid full-SlaClip base target, eta, or threshold bounds"
            )
        controller_update_formula = (
            "for each g in {A,B}:q_g=s_hat_g[0];"
            "r_g=s_hat_g[K-1];z_g=r_g/(C_g+1e-6);"
            "remaining_g=1-z_g;"
            "target_clip_g=clip(beta_g*remaining_g,0,1);"
            "gamma_g=1-target_clip_g;"
            "C_g_next=clip(C_g*exp(eta*(gamma_g-q_g)),C_min,C_max)"
            if groupwise_targets
            else (
                "q=s_hat[0];r=s_hat[K-1];z=r/(C+1e-6);"
                "remaining=1-z;"
                "target_clip=clip(base_target_clipped_fraction*remaining,0,1);"
                "gamma=1-target_clip;"
                "C_next=clip(C*exp(eta*(gamma-q)),C_min,C_max)"
            )
        )
        slaclip_contract = {
            "schema_version": (
                "groupwise_generalized_full_slaclip_beta_contract_v1"
                if groupwise_targets
                else "full_slaclip_contract_v1"
            ),
            "variant": (
                GROUPWISE_SLACLIP_VARIANT
                if groupwise_targets
                else "full_slaclip_cdf_endpoints"
            ),
            "controller_input": controller_input,
            "non_private_oracle_control": (
                controller_input == EXACT_CONTROLLER_INPUT
            ),
            "federated_adaptation": (
                "per_client_joint_gradient_slack_release_then_equal_fedavg_"
                "and_one_controller_update_per_round"
            ),
            "reference": {
                "paper": "https://openreview.net/pdf?id=48suUeYKdb",
                "repository": SLACLIP_REFERENCE_REPOSITORY,
                "revision": SLACLIP_REFERENCE_REVISION,
            },
            "hyperparameter_provenance": {
                "source": (
                    "fixed_baseline_exact_endpoint_development_selection_lock"
                    if args.slaclip_baseline_calibration_lock_sha256 is not None
                    else "direct_cli_not_declared_baseline_derived"
                ),
                "baseline_derived_calibration_is_non_dp": (
                    args.slaclip_baseline_calibration_lock_sha256 is not None
                ),
                "calibration_lock_sha256": (
                    args.slaclip_baseline_calibration_lock_sha256
                ),
                "calibration_data_consumed_at_controller_runtime": False,
                "frozen_scalar_hyperparameters_only": True,
            },
            "controller": {
                "eta": float(args.slaclip_eta),
                "controller_input": controller_input,
                "slack_noise_method_scope": slaclip_slack_noise_method_scope(
                    config.method,
                    pair_noise_across_methods=config.pair_noise_across_methods,
                ),
                **(
                    {
                        "target_parameterization": "per_gradient_group",
                        "generalized_full_slaclip_beta": True,
                        "base_target_clipped_fraction_by_group": dict(
                            base_targets
                        ),
                        "base_target_clipped_fraction_source": (
                            args.slaclip_base_target_clipped_fraction_source
                        ),
                        "beta_by_group": dict(base_targets),
                        "shared_beta_compatibility_alias_active": False,
                    }
                    if groupwise_targets
                    else {
                        "base_target_clipped_fraction": float(
                            base_targets["A"]
                        ),
                        "base_target_clipped_fraction_source": (
                            args.slaclip_base_target_clipped_fraction_source
                        ),
                        "beta": float(base_targets["A"]),
                        "beta_compatibility_alias": True,
                    }
                ),
                "epsilon": 1e-6,
                "num_slots": int(num_slots),
                "num_slots_selection": "explicit",
                "explicit_num_slots": int(num_slots),
                "local_batch_size": config.batch_size,
                "num_clients": config.num_clients,
                "expected_release_records": config.num_clients,
                "automatic_release_num_slots": automatic_release_num_slots,
                "explicit_num_slots_exceeds_automatic_release_bound": (
                    exceeds_automatic_release_bound
                ),
                "normalized_proxy_noise_std_per_slot_theoretical": (
                    normalized_proxy_noise_std
                ),
                "normalized_proxy_noise_std_formula": (
                    "noise_multiplier*sqrt(num_slots/num_clients)"
                ),
                "near_threshold_index": 0,
                "near_zero_index": int(num_slots - 1),
                "c_min": float(args.slaclip_c_min),
                "c_max": float(args.slaclip_c_max),
                "initial_clip_threshold": config.clip_norm,
                **(
                    {
                        "initial_clip_threshold_by_group": dict(
                            initial_clip_norms
                        )
                    }
                    if len(set(initial_clip_norms.values())) > 1
                    else {}
                ),
                "update_formula": controller_update_formula,
                "numerical_log_step_bounds": [
                    -MAX_ABS_LOG_STEP,
                    MAX_ABS_LOG_STEP,
                ],
                "numerical_safeguard": (
                    "eta*(gamma-q) is bounded before exp only to avoid "
                    "floating-point overflow; every activation is logged"
                ),
                "controller_inputs": (
                    "noisy_joint_release_endpoints_only"
                    if controller_input == NOISY_CONTROLLER_INPUT
                    else "exact_non_dp_endpoint_signals"
                ),
            },
            "exact_cdf_and_clipping_diagnostics": PRIVACY_LABEL,
            "release_noise_warning": (
                f"With {config.num_clients} client contributions, "
                f"sigma={config.noise_multiplier:g}, and K={num_slots}, the "
                "normalized endpoint noise standard deviation is "
                f"{normalized_proxy_noise_std:.12g}. "
                + (
                    "This K exceeds the paper central-batch monotonicity-rule "
                    f"choice K={automatic_release_num_slots}."
                    if exceeds_automatic_release_bound
                    else "This K does not exceed the automatic monotonicity-rule "
                    f"choice K={automatic_release_num_slots}."
                )
            ),
            "independently_privacy_certified": False,
        }

    private_key = (
        load_private_key(args.private_rng_key)
        if args.resume
        else load_or_create_private_key(args.private_rng_key)
    )
    private_key_commitment = private_key_fingerprint(private_key)
    data = load_data_protocol(manifest, config)

    assumptions = [
        "The paper says GPT-2 has 12 layers/hidden 768 despite also saying 1.5B; this run uses GPT-2 small.",
        "The paper does not publish model revisions; immutable public revisions are pinned here.",
        "The paper does not publish LoRA targets; BERT query/key/value and GPT-2 fused c_attn are used.",
        "The paper does not publish LoRA alpha/dropout; alpha=rank and dropout=0 are used.",
        "The paper says gradient descent but no optimizer; one manual SGD step is used per client/round.",
        "The local objective averages token loss within each record, then equally averages the B records before group clipping.",
        "The paper does not publish its language-model objective; BERT uses masked LM and GPT-2 uses causal LM over the Patient/Doctor text.",
        "The paper clips aggregate A and B batch gradients separately; this run follows that literal grouping.",
        "Gaussian coordinates for A and B are independent and their secret seeds are never logged.",
        "The paper does not specify sequence length; 128 is used for formal runs.",
        "Torch deterministic algorithms are enforced, cuDNN benchmarking and TF32 are disabled, and CUDA uses a fixed cuBLAS workspace configuration.",
        "The public MedDialog mirror has 257,469 records versus 257,332 reported in the paper; all public train/validation/test splits are united for training because the paper calls the full corpus its training dataset and does not publish a split.",
        "A fixed subset of the public validation split is removed from the united corpus before it is used for the internal fixed-mask LM-loss diagnostic.",
        "The internal LM loss is not any of the six paper benchmark scores and cannot establish paper metric reproduction.",
        "No independent epsilon is reported because the paper does not provide the constants/calibration needed to map sigma=2 to its epsilon sweep.",
    ]
    if len(set(config.clip_norm_by_group.values())) > 1:
        assumptions.append(
            "This arm uses explicitly preregistered groupwise clipping "
            "thresholds C_A and C_B; each group therefore also receives "
            "Gaussian gradient noise scaled to its own threshold."
        )
    if slaclip_contract is not None:
        assert config.method in ADAPTIVE_METHODS
        controller_input = CONTROLLER_INPUT_BY_METHOD[config.method]
        assumptions.extend(
            [
                (
                    "The adaptive controller uses the first and final "
                    + (
                        "noisy CDF-proxy slots"
                        if controller_input == NOISY_CONTROLLER_INPUT
                        else "exact, explicitly non-private CDF-proxy slots"
                    )
                    + " to derive a dynamic target; "
                    + (
                        "each gradient group uses its own canonical "
                        "base_target_clipped_fraction_g (generalized full-"
                        "SlaClip beta_g), and each beta_g "
                        if groupwise_targets
                        else "base_target_clipped_fraction (the paper/reference "
                        "shared beta) "
                    )
                    + "is modulated by that group's estimated fraction of "
                    "remaining non-small gradients, so it is not a fixed "
                    "clipping target "
                    "and no calibration data enters the controller at runtime."
                ),
                "SlaClip is adapted to the paper-literal federated mechanism by releasing one joint gradient/slack vector per client and LoRA group, then updating each A/B threshold once after FedAvg.",
                (
                    "The controller consumes only noisy endpoints; exact endpoint "
                    "values and actual clipping fractions are retained solely as "
                    "explicitly labelled non-DP private diagnostics."
                    if controller_input == NOISY_CONTROLLER_INPUT
                    else "This oracle control uses exact endpoint signals to set "
                    "the next threshold and is explicitly NON-DP; the sigma=2 "
                    "gradient noise and noisy slack release remain enabled solely "
                    "to isolate controller-input noise."
                ),
                (
                    f"The requested K={num_slots} policy with "
                    f"N={config.num_clients} and sigma={config.noise_multiplier:g} "
                    "is retained exactly as specified by the experiment plan; "
                    f"its theoretical normalized endpoint noise standard "
                    f"deviation is {normalized_proxy_noise_std:.12g}, and the "
                    "exploratory federated adaptation is not independently "
                    "end-to-end DP certified."
                ),
            ]
        )
        if args.slaclip_baseline_calibration_lock_sha256 is not None:
            assumptions.append(
                "The frozen SlaClip beta hyperparameters were selected from "
                "exact, NON_DP_PRIVATE_DIAGNOSTIC fixed-baseline endpoints "
                "under the immutable calibration lock recorded in the "
                "algorithm contract; the controller consumes no calibration "
                "artifact during this training run."
            )
    dependency_versions = package_versions()
    backend_contract = execution_backend_contract(device)
    scientific_contract = {
        "schema_version": 2,
        "repository_sha": repo.get("sha"),
        "input_manifest_sha256": summary["manifest_sha256"],
        "input_inventory_sha256": summary["inventory_sha256"],
        "dataset": {
            "repo_id": EXPECTED_DATASET_ID,
            "revision": EXPECTED_DATASET_REVISION,
        },
        "data_protocol": data.protocol,
        "method": asdict(METHOD_SPECS[config.method]),
        "effective_config": asdict(config),
        "models": args.models,
        "model_revisions": EXPECTED_MODELS,
        "private_key_commitment": private_key_commitment,
        "rng_domain": args.rng_domain,
        "dependency_versions": dependency_versions,
        "execution_backend": backend_contract,
        "algorithm_contract": {
            "federated_clients_simulated_sequentially": True,
            "client_weighting": "equal_one_over_K",
            "gradient_grouping": "aggregate_batch_gradient_separate_A_and_B",
            "initial_clip_threshold_by_group": dict(
                config.clip_norm_by_group
            ),
            "groupwise_fixed_thresholds": bool(
                config.method not in ADAPTIVE_METHODS
                and len(set(config.clip_norm_by_group.values())) > 1
            ),
            "local_optimizer": "one_manual_sgd_step",
            "record_weighting": "equal_records_after_within_record_token_mean",
            "dropout_rng": "stateless_private_hmac_per_model_round_client",
            "contains_slaclip": slaclip_contract is not None,
            "slaclip": slaclip_contract,
        },
    }
    run_config_fingerprint = canonical_json_fingerprint(scientific_contract)
    requested_arguments = dict(vars(args))
    requested_arguments["private_rng_key"] = "<PRIVATE_PATH_REDACTED>"
    run_config = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "privacy_label": PRIVACY_LABEL,
        "method": config.method,
        "method_spec": asdict(METHOD_SPECS[config.method]),
        "contains_slaclip": slaclip_contract is not None,
        "run_config_fingerprint": run_config_fingerprint,
        "scientific_contract": scientific_contract,
        "reproduction_claim": {
            "level": 1,
            "name": "algorithm_execution_reconstruction",
            "paper_result_reproduced": False,
            "paper_benchmarks_evaluated": False,
            "reason": "The paper does not publish sufficient benchmark prompts/scoring details and the six benchmark data are not staged.",
        },
        "privacy_claim": {
            "end_to_end_dp_certified": False,
            "epsilon": None,
            "sigma_is_not_epsilon": True,
            "diagnostics_are_private_non_dp_data": True,
            "baseline_derived_calibration_is_non_dp": (
                args.slaclip_baseline_calibration_lock_sha256 is not None
            ),
            "baseline_calibration_lock_sha256": (
                args.slaclip_baseline_calibration_lock_sha256
            ),
            "baseline_calibration_consumed_at_runtime": False,
            "exact_cdf_diagnostics_are_non_dp": slaclip_contract is not None,
            "oracle_controller_uses_non_dp_exact_cdf": (
                config.method == ORACLE_SLACLIP_METHOD
            ),
        },
        "repository": repo,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": dependency_versions,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "execution_backend_contract": backend_contract,
        },
        "input": summary,
        "training_corpus": {
            "source_splits": ["train", "validation", "test"],
            "rows": manifest["formal_dataset"]["total_rows"],
            "reason": "paper labels the full 257,332-dialogue corpus as training data and publishes no split",
        },
        "internal_validation_diagnostic": {
            "source_split": "validation",
            "overlaps_training_corpus": False,
            "paper_benchmark_metric": False,
            "fixed_examples_and_bert_masks_across_rounds": True,
            "protocol": data.protocol,
        },
        "requested_arguments": requested_arguments,
        "effective_config": asdict(config),
        "models": args.models,
        "paper_reported_parameters": {
            "K": 5,
            "T": 50,
            "B": 8,
            "sigma": 2,
            "learning_rate": 5e-4,
            "C": 10,
            "rank": 512,
        },
        "reconstruction_assumptions": assumptions,
    }
    if not args.resume:
        output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            output_dir.mkdir(mode=0o700)
        except FileExistsError as error:
            raise SystemExit(
                f"another process created the output first: {output_dir}"
            ) from error
        validate_private_run_directory(output_dir)
    root_final_summary_path = output_dir / "final_summary.json"
    existing_completed_root: dict[str, Any] | None = None
    existing_completed_root_sha256: str | None = None
    lock_handle = acquire_run_lock(output_dir)
    try:
        run_config_path = output_dir / "run_config.json"
        if args.resume:
            validate_private_regular_file(run_config_path, "existing run config")
            validate_private_regular_file(
                output_dir / "PRIVACY-NOTICE.txt", "root privacy notice"
            )
            existing_config = load_json_object(run_config_path, "existing run config")
            if existing_config.get("run_config_fingerprint") != run_config_fingerprint:
                raise SystemExit("resume run configuration fingerprint mismatch")
            if existing_config.get("scientific_contract") != scientific_contract:
                raise SystemExit("resume scientific contract differs from the original run")
            if os.path.lexists(root_final_summary_path):
                validate_private_regular_file(
                    root_final_summary_path, "completed root summary"
                )
                existing_completed_root = load_json_object(
                    root_final_summary_path, "completed root summary"
                )
                root_identity = {
                    "schema_version": 2,
                    "status": "COMPLETED",
                    "method": config.method,
                    "run_config_fingerprint": run_config_fingerprint,
                }
                for key, expected in root_identity.items():
                    if existing_completed_root.get(key) != expected:
                        raise RuntimeError(
                            f"completed root summary mismatch: {key}"
                        )
                root_models = existing_completed_root.get("models")
                if not isinstance(root_models, dict) or set(root_models) != set(
                    args.models
                ):
                    raise RuntimeError("completed root summary model set mismatch")
                for model_kind in args.models:
                    validate_private_regular_file(
                        output_dir / model_kind / "final_summary.json",
                        f"completed {model_kind} summary",
                    )
                existing_completed_root_sha256 = sha256_file(
                    root_final_summary_path
                )
        else:
            atomic_json(run_config_path, run_config)
            write_new_private_text(
                output_dir / "PRIVACY-NOTICE.txt",
                "NON_DP_PRIVATE_DIAGNOSTIC\n"
                "Exact samples, losses, gradients, checkpoints, and control adapters are private diagnostics.\n"
                "They must not be published or described as differentially private outputs.\n",
            )

        attempts_dir = output_dir / "attempts"
        prepare_private_directory(
            attempts_dir,
            exist_ok=args.resume,
            description="run attempts directory",
        )
        attempt_id = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
            f"-pid{os.getpid()}"
        )
        attempt_path = attempts_dir / f"{attempt_id}.json"
        prior_failure_path = output_dir / "failure.json"
        if args.resume and os.path.lexists(prior_failure_path):
            validate_private_regular_file(prior_failure_path, "prior failure record")
            archived_failure = attempts_dir / f"superseded-failure-{attempt_id}.json"
            if os.path.lexists(archived_failure):
                raise RuntimeError(
                    f"refusing to overwrite an archived failure: {archived_failure}"
                )
            os.replace(prior_failure_path, archived_failure)
            fsync_directory(attempts_dir)
        attempt = {
            "schema_version": 2,
            "status": "RUNNING",
            "attempt_id": attempt_id,
            "resume": args.resume,
            "run_config_fingerprint": run_config_fingerprint,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "hostname": platform.node(),
            "execution_backend": backend_contract,
            "pid": os.getpid(),
            "started_at_utc": utc_now(),
        }
        atomic_json(attempt_path, attempt)
        atomic_json(
            output_dir / "progress.json",
            {
                "schema_version": 2,
                "status": "RUNNING",
                "method": config.method,
                "models": args.models,
                "resume": args.resume,
                "attempt_id": attempt_id,
                "updated_at_utc": utc_now(),
            },
        )
    except BaseException:
        lock_handle.close()
        raise

    results: dict[str, Any] = {}
    started = time.monotonic()
    try:
        for model_kind in args.models:
            results[model_kind] = train_one_model(
                model_kind=model_kind,
                manifest=manifest,
                data=data,
                output_dir=output_dir / model_kind,
                config=config,
                device=device,
                private_key=private_key,
                private_key_commitment=private_key_commitment,
                rng_domain=args.rng_domain,
                run_config_fingerprint=run_config_fingerprint,
                slaclip_contract=slaclip_contract,
                resume=args.resume,
                stop_file=args.stop_file,
            )
        root_revalidation = existing_completed_root is not None
        if root_revalidation:
            assert existing_completed_root is not None
            assert existing_completed_root_sha256 is not None
            validate_completed_root_summary(
                existing_completed_root,
                config=config,
                run_config_fingerprint=run_config_fingerprint,
                results=results,
            )
            if sha256_file(root_final_summary_path) != existing_completed_root_sha256:
                raise RuntimeError(
                    "completed root summary changed during revalidation"
                )
            final = existing_completed_root
        else:
            final = {
                "schema_version": 2,
                "status": "COMPLETED",
                "privacy_label": PRIVACY_LABEL,
                "contains_slaclip": slaclip_contract is not None,
                "method": config.method,
                "method_spec": asdict(METHOD_SPECS[config.method]),
                "run_config_fingerprint": run_config_fingerprint,
                "reproduction_claim_level": 1,
                "paper_result_reproduced": False,
                "paper_benchmarks_evaluated": False,
                "privacy_claim": False,
                "models": results,
                "sample_schedule_sha256_by_model": {
                    model: result["behavior_summary"]["sample_schedule_sha256"]
                    for model, result in results.items()
                },
                "elapsed_seconds": time.monotonic() - started,
                "completed_at_utc": utc_now(),
            }
            atomic_json(root_final_summary_path, final)
        progress = {
            "schema_version": 2,
            "status": "COMPLETED",
            "method": config.method,
            "models": args.models,
            "attempt_id": attempt_id,
            "completed_at_utc": final["completed_at_utc"],
            "root_final_summary_preserved": root_revalidation,
        }
        if root_revalidation:
            progress["revalidated_at_utc"] = utc_now()
        atomic_json(
            output_dir / "progress.json",
            progress,
        )
        attempt.update(
            {
                "status": "COMPLETED",
                "completed_root_revalidation": root_revalidation,
                "root_final_summary_preserved": root_revalidation,
                "elapsed_seconds": time.monotonic() - started,
                "finished_at_utc": utc_now(),
            }
        )
        atomic_json(attempt_path, attempt)
    except GracefulStop as error:
        stopped = {
            "schema_version": 2,
            "status": "CHECKPOINTED_STOP",
            "privacy_label": PRIVACY_LABEL,
            "method": config.method,
            "run_config_fingerprint": run_config_fingerprint,
            "attempt_id": attempt_id,
            "reason": str(error),
            "stopped_at_utc": utc_now(),
        }
        atomic_json(output_dir / "progress.json", stopped)
        attempt.update(
            {
                "status": "CHECKPOINTED_STOP",
                "reason": str(error),
                "elapsed_seconds": time.monotonic() - started,
                "finished_at_utc": utc_now(),
            }
        )
        atomic_json(attempt_path, attempt)
        raise SystemExit(75) from error
    except BaseException as error:
        failure = {
            "schema_version": 2,
            "status": "FAILED",
            "privacy_label": PRIVACY_LABEL,
            "method": config.method,
            "run_config_fingerprint": run_config_fingerprint,
            "attempt_id": attempt_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at_utc": utc_now(),
        }
        with contextlib.suppress(Exception):
            atomic_json(output_dir / "failure.json", failure)
            atomic_json(output_dir / "progress.json", failure)
            attempt.update(
                {
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.monotonic() - started,
                    "finished_at_utc": utc_now(),
                }
            )
            atomic_json(attempt_path, attempt)
        raise
    finally:
        lock_handle.close()


if __name__ == "__main__":
    main()
