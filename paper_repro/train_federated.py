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
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import secrets
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


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


@dataclass(frozen=True)
class EffectiveConfig:
    num_clients: int
    rounds: int
    batch_size: int
    noise_multiplier: float
    learning_rate: float
    clip_norm: float
    rank: int
    max_seq_length: int
    seed: int
    max_validation_records: int
    eval_every: int
    smoke: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True, default=json_default) + "\n")
    handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_input_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json_object(path, "input manifest")
    if manifest.get("status") != "STAGING_COMPLETE_VERIFIED":
        raise RuntimeError("input manifest is not marked STAGING_COMPLETE_VERIFIED")
    dataset = manifest.get("formal_dataset")
    models = manifest.get("models")
    if not isinstance(dataset, dict) or not isinstance(models, dict):
        raise RuntimeError("input manifest is missing dataset/model metadata")
    if dataset.get("repo_id") != EXPECTED_DATASET_ID:
        raise RuntimeError(f"unexpected dataset: {dataset.get('repo_id')!r}")
    if dataset.get("revision") != EXPECTED_DATASET_REVISION:
        raise RuntimeError(f"unexpected dataset revision: {dataset.get('revision')!r}")
    combined = dataset.get("combined_splits")
    if not isinstance(combined, dict):
        raise RuntimeError("input manifest has no combined dataset splits")
    for split in ("train", "validation", "test"):
        entry = combined.get(split)
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), list):
            raise RuntimeError(f"input manifest has no file list for {split}")
        if int(entry.get("rows", 0)) <= 0:
            raise RuntimeError(f"input manifest has no rows for {split}")
        for raw_path in entry["files"]:
            file_path = Path(str(raw_path))
            if not file_path.is_file():
                raise RuntimeError(f"missing staged {split} file: {file_path}")
    for model_name, expected in EXPECTED_MODELS.items():
        entry = models.get(expected["manifest_key"])
        if not isinstance(entry, dict):
            raise RuntimeError(f"input manifest is missing {model_name}")
        if entry.get("repo_id") != expected["repo_id"]:
            raise RuntimeError(f"{model_name} repo ID mismatch")
        if entry.get("revision") != expected["revision"]:
            raise RuntimeError(f"{model_name} revision mismatch")
        snapshot = Path(str(entry.get("snapshot_path")))
        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            raise RuntimeError(f"invalid {model_name} snapshot: {snapshot}")
    return manifest


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

    def texts(self, indices: Sequence[int] | np.ndarray) -> list[str]:
        index_array = self._pa.array([int(index) for index in indices], type=self._pa.int64())
        rows = self._table.take(index_array).to_pylist()
        texts = []
        for row in rows:
            source = str(row.get("src") or "").strip()
            target = str(row.get("tgt") or "").strip()
            if not source and not target:
                raise RuntimeError("MedDialog record contains no text")
            texts.append(f"Patient: {source}\nDoctor: {target}")
        return texts


def load_tables(manifest: dict[str, Any], smoke: bool) -> tuple[ParquetTextTable, ParquetTextTable]:
    splits = manifest["formal_dataset"]["combined_splits"]
    # The paper calls all 257,332 dialogues its training dataset and does not
    # describe a held-out split.  The pinned public mirror has 257,469 rows
    # divided into train/validation/test, so use their union for the closest
    # available reconstruction of the paper's training corpus.
    training_split_names = ("train", "validation", "test")
    train = ParquetTextTable(
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
    if len(train) != expected_training_rows:
        raise RuntimeError("loaded all-split training row count differs from manifest")
    if len(validation) != int(splits["validation"]["rows"]):
        raise RuntimeError("loaded validation row count differs from manifest")
    if smoke:
        # The Arrow table stays immutable; smoke restricts the index domain below.
        return train, validation
    return train, validation


def partition_indices(
    population: int, num_clients: int, seed: int, limit: int | None = None
) -> list[np.ndarray]:
    if population < num_clients:
        raise ValueError("population must be at least the number of clients")
    usable = population if limit is None else min(population, limit)
    if usable < num_clients:
        raise ValueError("limited population must be at least the number of clients")
    permutation = np.random.default_rng(seed).permutation(population)[:usable]
    partitions = [part.astype(np.int64, copy=False) for part in np.array_split(permutation, num_clients)]
    if any(len(part) == 0 for part in partitions):
        raise RuntimeError("a client received an empty data partition")
    return partitions


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_effective_config(args: argparse.Namespace) -> EffectiveConfig:
    if args.smoke:
        return EffectiveConfig(
            num_clients=2,
            rounds=1,
            batch_size=args.batch_size,
            noise_multiplier=args.noise_multiplier,
            learning_rate=args.learning_rate,
            clip_norm=args.clip_norm,
            rank=args.rank,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
            max_validation_records=args.batch_size,
            eval_every=1,
            smoke=True,
        )
    return EffectiveConfig(
        num_clients=args.num_clients,
        rounds=args.rounds,
        batch_size=args.batch_size,
        noise_multiplier=args.noise_multiplier,
        learning_rate=args.learning_rate,
        clip_norm=args.clip_norm,
        rank=args.rank,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        max_validation_records=args.max_validation_records,
        eval_every=args.eval_every,
        smoke=False,
    )


def validate_config(config: EffectiveConfig) -> None:
    for name in ("num_clients", "rounds", "batch_size", "rank", "max_seq_length"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.noise_multiplier < 0:
        raise ValueError("noise_multiplier must be non-negative")
    if config.learning_rate <= 0 or config.clip_norm <= 0:
        raise ValueError("learning_rate and clip_norm must be positive")
    if config.max_validation_records <= 0 or config.eval_every <= 0:
        raise ValueError("validation size and eval interval must be positive")


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
    clip_norm: float,
    noise_multiplier: float,
    learning_rate: float,
    generator: torch.Generator,
) -> dict[str, dict[str, float | bool | int]]:
    """Apply the paper's separate A/B batch-gradient mechanism in place."""

    statistics: dict[str, dict[str, float | bool | int]] = {}
    for group_name in ("A", "B"):
        entries = groups[group_name]
        missing = [name for name, parameter in entries if parameter.grad is None]
        if missing:
            raise RuntimeError(f"missing gradients in LoRA {group_name}: {missing[:5]}")
        gradients = [parameter.grad for _, parameter in entries]
        assert all(gradient is not None for gradient in gradients)
        raw_norm = float(torch.sqrt(squared_norm(gradients)).item())
        if not math.isfinite(raw_norm):
            raise FloatingPointError(f"non-finite raw LoRA {group_name} gradient norm")
        factor = min(1.0, clip_norm / max(raw_norm, torch.finfo(torch.float32).tiny))
        noise_sq: torch.Tensor | None = None
        for _, parameter in entries:
            gradient = parameter.grad
            assert gradient is not None
            gradient.mul_(factor)
            noise = torch.randn(
                gradient.shape,
                generator=generator,
                device=gradient.device,
                dtype=torch.float32,
            ).mul_(noise_multiplier * clip_norm)
            noise_sq_value = noise.square().sum()
            noise_sq = noise_sq_value if noise_sq is None else noise_sq + noise_sq_value
            gradient.add_(noise.to(dtype=gradient.dtype))
        noisy_norm = float(torch.sqrt(squared_norm(gradients)).item())
        noise_norm = float(torch.sqrt(noise_sq).item()) if noise_sq is not None else 0.0
        statistics[group_name] = {
            "raw_norm": raw_norm,
            "clip_factor": factor,
            "clipped": factor < 1.0,
            "clipped_norm": raw_norm * factor,
            "noise_l2_norm": noise_norm,
            "noisy_gradient_l2_norm": noisy_norm,
            "parameter_tensors": len(entries),
            "parameter_elements": sum(parameter.numel() for _, parameter in entries),
        }
    with torch.no_grad():
        for entries in groups.values():
            for _, parameter in entries:
                assert parameter.grad is not None
                parameter.add_(parameter.grad, alpha=-learning_rate)
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
) -> dict[str, torch.Tensor]:
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
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def equal_record_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    model_kind: str,
) -> torch.Tensor:
    """Average token loss within each record, then average the B records."""

    import torch.nn.functional as functional

    outputs = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    logits = outputs.logits
    labels = batch["labels"]
    if model_kind == "gpt2":
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
    elif model_kind != "bert":
        raise ValueError(f"unsupported model kind: {model_kind}")
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
    losses: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(validation_indices), config.batch_size):
            indices = validation_indices[offset : offset + config.batch_size]
            batch = make_batch(
                validation.texts(indices),
                tokenizer,
                model_kind,
                config.max_seq_length,
                device,
                seed + offset,
            )
            loss = equal_record_loss(model, batch, model_kind)
            value = float(loss.detach().float().item())
            if not math.isfinite(value):
                raise FloatingPointError("non-finite validation loss")
            losses.append(value)
    mean_loss = float(np.mean(losses))
    model.train()
    return {
        "objective": "masked_lm" if model_kind == "bert" else "causal_lm",
        "records": int(len(validation_indices)),
        "batches": len(losses),
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
    }
    for group in ("A", "B"):
        values = [record["gradient_groups"][group] for record in records]
        clipped = sum(bool(value["clipped"]) for value in values)
        summary[group] = {
            "clipped_count": clipped,
            "clipped_fraction": clipped / len(values),
            "mean_raw_norm": float(np.mean([value["raw_norm"] for value in values])),
            "max_raw_norm": float(np.max([value["raw_norm"] for value in values])),
            "mean_clip_factor": float(
                np.mean([value["clip_factor"] for value in values])
            ),
            "mean_noise_l2_norm": float(
                np.mean([value["noise_l2_norm"] for value in values])
            ),
        }
    summary["any_group_clipped_count"] = sum(
        record["gradient_groups"]["A"]["clipped"]
        or record["gradient_groups"]["B"]["clipped"]
        for record in records
    )
    summary["any_group_clipped_fraction"] = (
        summary["any_group_clipped_count"] / len(records)
    )
    return summary


def train_one_model(
    *,
    model_kind: str,
    manifest: dict[str, Any],
    train: ParquetTextTable,
    validation: ParquetTextTable,
    output_dir: Path,
    config: EffectiveConfig,
    device: torch.device,
) -> dict[str, Any]:
    started = time.monotonic()
    model_seed = config.seed + (0 if model_kind == "bert" else 1_000_000)
    seed_everything(model_seed)
    snapshot = model_snapshot(manifest, model_kind)
    model, tokenizer, target_modules = build_model(
        model_kind, snapshot, config.rank, device
    )
    groups = parameter_groups(model)
    global_state = clone_trainable_state(groups)
    trainable_elements = sum(value.numel() for value in global_state.values())

    train_limit = 16 if config.smoke else None
    clients = partition_indices(
        len(train), config.num_clients, model_seed + 11, train_limit
    )
    validation_count = min(
        len(validation), config.max_validation_records, 4 if config.smoke else len(validation)
    )
    validation_indices = np.random.default_rng(model_seed + 17).choice(
        len(validation), size=validation_count, replace=False
    )
    # Sampling and Gaussian-mechanism seeds are deliberately drawn from OS
    # entropy and never persisted.  The user seed controls initialization and
    # deterministic preprocessing only; publishing DP-noise seeds would make
    # the privacy mechanism removable from released weights.
    sampling_master_seed = secrets.randbits(63)
    client_rngs = [
        np.random.default_rng(sampling_master_seed + client_id)
        for client_id in range(config.num_clients)
    ]
    noise_generator = torch.Generator(device=device.type).manual_seed(
        secrets.randbits(63)
    )

    output_dir.mkdir(mode=0o700)
    diagnostics_dir = output_dir / "private_diagnostics"
    diagnostics_dir.mkdir(mode=0o700)
    (diagnostics_dir / "NON_DP_PRIVATE_DATA.txt").write_text(
        "NON_DP_PRIVATE_DIAGNOSTIC\n"
        "Exact client batch losses and gradient norms are not DP release artifacts.\n",
        encoding="utf-8",
    )
    client_log_path = diagnostics_dir / "client_rounds.jsonl"
    round_log_path = diagnostics_dir / "round_summaries.jsonl"

    evaluations: list[dict[str, Any]] = []
    initial_eval = evaluate(
        model,
        tokenizer,
        model_kind,
        validation,
        validation_indices,
        config,
        device,
        model_seed + 20_000,
    )
    evaluations.append({"round": 0, **initial_eval})

    clipped_counts = {"A": 0, "B": 0, "any": 0}
    total_client_steps = 0
    last_round_summary: dict[str, Any] | None = None
    with client_log_path.open("x", encoding="utf-8") as client_log, round_log_path.open(
        "x", encoding="utf-8"
    ) as round_log:
        for round_index in range(1, config.rounds + 1):
            round_started = time.monotonic()
            aggregate = empty_state_like(global_state)
            records: list[dict[str, Any]] = []
            for client_id, client_indices in enumerate(clients):
                restore_trainable_state(groups, global_state)
                model.zero_grad(set_to_none=True)
                replace = len(client_indices) < config.batch_size
                sampled = client_rngs[client_id].choice(
                    client_indices, size=config.batch_size, replace=replace
                )
                batch = make_batch(
                    train.texts(sampled),
                    tokenizer,
                    model_kind,
                    config.max_seq_length,
                    device,
                    model_seed + round_index * 1000 + client_id,
                )
                loss = equal_record_loss(model, batch, model_kind)
                loss_value = float(loss.detach().float().item())
                if not math.isfinite(loss_value):
                    raise FloatingPointError(
                        f"non-finite loss at round {round_index}, client {client_id}"
                    )
                loss.backward()
                group_stats = clip_noise_and_step(
                    groups,
                    clip_norm=config.clip_norm,
                    noise_multiplier=config.noise_multiplier,
                    learning_rate=config.learning_rate,
                    generator=noise_generator,
                )
                accumulate_state(aggregate, groups, 1.0 / config.num_clients)
                record = {
                    "privacy_label": PRIVACY_LABEL,
                    "model": model_kind,
                    "round": round_index,
                    "client": client_id,
                    "batch_size": config.batch_size,
                    "loss": loss_value,
                    "gradient_groups": group_stats,
                }
                append_jsonl(client_log, record)
                records.append(record)
                total_client_steps += 1
                for group in ("A", "B"):
                    clipped_counts[group] += int(bool(group_stats[group]["clipped"]))
                clipped_counts["any"] += int(
                    bool(group_stats["A"]["clipped"])
                    or bool(group_stats["B"]["clipped"])
                )
                model.zero_grad(set_to_none=True)
            global_state = aggregate
            restore_trainable_state(groups, global_state)
            last_round_summary = {
                "privacy_label": PRIVACY_LABEL,
                "model": model_kind,
                "round": round_index,
                "elapsed_seconds": time.monotonic() - round_started,
                **aggregate_round_statistics(records),
            }
            if round_index % config.eval_every == 0 or round_index == config.rounds:
                evaluation = evaluate(
                    model,
                    tokenizer,
                    model_kind,
                    validation,
                    validation_indices,
                    config,
                    device,
                    # Keep BERT's random MLM mask fixed across evaluations so
                    # the curve reflects model change rather than mask drift.
                    model_seed + 20_000,
                )
                evaluations.append({"round": round_index, **evaluation})
                last_round_summary["validation"] = evaluation
            append_jsonl(round_log, last_round_summary)
            print(
                json.dumps(
                    {
                        "model": model_kind,
                        "round": round_index,
                        "rounds": config.rounds,
                        "mean_loss": last_round_summary["mean_training_loss"],
                        "A_clip_fraction": last_round_summary["A"]["clipped_fraction"],
                        "B_clip_fraction": last_round_summary["B"]["clipped_fraction"],
                        "elapsed_seconds": last_round_summary["elapsed_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    restore_trainable_state(groups, global_state)
    adapter_dir = output_dir / "final_adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    final_evaluation = evaluations[-1]
    elapsed = time.monotonic() - started
    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "privacy_label": PRIVACY_LABEL,
        "model": model_kind,
        "base_model": EXPECTED_MODELS[model_kind],
        "snapshot_path": str(snapshot),
        "objective": "masked_lm" if model_kind == "bert" else "causal_lm",
        "target_modules": target_modules,
        "trainable_parameter_tensors": len(global_state),
        "trainable_parameter_elements": trainable_elements,
        "client_partition_sizes": [len(partition) for partition in clients],
        "client_steps": total_client_steps,
        "clipping": {
            "A": {
                "count": clipped_counts["A"],
                "fraction": clipped_counts["A"] / total_client_steps,
            },
            "B": {
                "count": clipped_counts["B"],
                "fraction": clipped_counts["B"] / total_client_steps,
            },
            "any_group": {
                "count": clipped_counts["any"],
                "fraction": clipped_counts["any"] / total_client_steps,
            },
        },
        "evaluations": evaluations,
        "final_evaluation": final_evaluation,
        "last_round": last_round_summary,
        "adapter_dir": str(adapter_dir),
        "privacy_randomness": {
            "batch_sampling_seed": "SECRET_NOT_LOGGED",
            "gaussian_noise_seed": "SECRET_NOT_LOGGED",
        },
        "elapsed_seconds": elapsed,
        "completed_at_utc": utc_now(),
    }
    atomic_json(output_dir / "final_summary.json", summary)
    del model, tokenizer, groups, global_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated DP-LoRA paper reconstruction (no SlaClip)"
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", nargs="+", choices=["bert", "gpt2"], default=["bert", "gpt2"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
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
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-validation-records", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--acknowledge-non-dp-diagnostics", action="store_true")
    return parser.parse_args(argv)


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
        print(json.dumps({"status": "verified", **summary}, indent=2, default=json_default))
        return
    if not args.acknowledge_non_dp_diagnostics:
        raise SystemExit(
            "--acknowledge-non-dp-diagnostics is required because exact client "
            "losses and gradient norms are written"
        )
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --check-inputs is used")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.mkdir(mode=0o700)
    config = make_effective_config(args)
    validate_config(config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if len(set(args.models)) != len(args.models):
        raise SystemExit("--models must not contain duplicates")
    repo = repository_state()
    if repo.get("dirty") is True:
        raise SystemExit("repository is dirty; refusing a formalized run")

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
        "The public MedDialog mirror has 257,469 records versus 257,332 reported in the paper; all public train/validation/test splits are united for training because the paper calls the full corpus its training dataset and does not publish a split.",
        "The public validation split is reused only for a fixed-mask LM-loss diagnostic and overlaps that all-split training corpus; it is not a paper downstream benchmark score.",
        "No independent epsilon is reported because the paper does not provide the constants/calibration needed to map sigma=2 to its epsilon sweep.",
    ]
    run_config = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "privacy_label": PRIVACY_LABEL,
        "method": "federated_dp_lora_algorithm_1_reconstruction",
        "contains_slaclip": False,
        "repository": repo,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "input": summary,
        "training_corpus": {
            "source_splits": ["train", "validation", "test"],
            "rows": manifest["formal_dataset"]["total_rows"],
            "reason": "paper labels the full 257,332-dialogue corpus as training data and publishes no split",
        },
        "internal_validation_diagnostic": {
            "source_split": "validation",
            "overlaps_training_corpus": True,
            "paper_benchmark_metric": False,
            "fixed_examples_and_bert_masks_across_rounds": True,
        },
        "requested_arguments": vars(args),
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
    atomic_json(output_dir / "run_config.json", run_config)
    (output_dir / "PRIVACY-NOTICE.txt").write_text(
        "NON_DP_PRIVATE_DIAGNOSTIC\n"
        "Exact client batch losses and gradient norms must not be published as DP outputs.\n",
        encoding="utf-8",
    )

    train, validation = load_tables(manifest, config.smoke)
    results: dict[str, Any] = {}
    started = time.monotonic()
    try:
        for model_kind in args.models:
            results[model_kind] = train_one_model(
                model_kind=model_kind,
                manifest=manifest,
                train=train,
                validation=validation,
                output_dir=output_dir / model_kind,
                config=config,
                device=device,
            )
        final = {
            "schema_version": 1,
            "status": "COMPLETED",
            "privacy_label": PRIVACY_LABEL,
            "contains_slaclip": False,
            "method": "federated_dp_lora_algorithm_1_reconstruction",
            "models": results,
            "elapsed_seconds": time.monotonic() - started,
            "completed_at_utc": utc_now(),
        }
        atomic_json(output_dir / "final_summary.json", final)
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "FAILED",
            "privacy_label": PRIVACY_LABEL,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at_utc": utc_now(),
        }
        with contextlib.suppress(Exception):
            atomic_json(output_dir / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
