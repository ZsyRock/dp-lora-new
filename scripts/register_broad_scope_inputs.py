#!/usr/bin/env python3
"""Register already-staged broad-scope corpora and model snapshots.

This command never invents a paper dataset and never downloads a gated model.
It converts a directory of standardized ``src``/``tgt`` Parquet splits plus
one or more pinned Hugging Face snapshots into the immutable input-manifest
contract consumed by ``paper_repro/train_federated.py``.

Expected dataset layout::

    DATASET_ROOT/train/*.parquet
    DATASET_ROOT/validation/*.parquet
    DATASET_ROOT/test/*.parquet

For SlimPajama, put document text in ``src`` and an empty string in ``tgt``.
For the finance reconstruction, put the input sentence in ``src`` and the
sentiment label in ``tgt``.  The manifest records that the paper's original
9,540-example finance training corpus was not released.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from paper_repro.reproducibility import canonical_json_fingerprint
    from paper_repro.train_federated import EXPECTED_DATASETS, EXPECTED_MODELS
except ModuleNotFoundError:  # direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paper_repro.reproducibility import canonical_json_fingerprint
    from paper_repro.train_federated import EXPECTED_DATASETS, EXPECTED_MODELS


TEXT_FORMATS = {
    "meddialog": "medical_dialogue_pair",
    "slimpajama": "plain_text",
    "finance": "financial_sentiment_pair",
}
PAPER_EXACTNESS = {
    "meddialog": "public_reconstruction",
    "slimpajama": "public_subset_reconstruction",
    "finance": "paper_source_not_released",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _inside(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{description} is outside {root}: {path}") from error
    return path.absolute()


def parse_model_snapshot(value: str) -> tuple[str, Path]:
    model, separator, raw_path = value.partition("=")
    if not separator or model not in EXPECTED_MODELS or not raw_path:
        raise argparse.ArgumentTypeError(
            "--model-snapshot must be MODEL=/absolute/snapshot/path"
        )
    return model, Path(raw_path)


def parquet_entry(path: Path, split: str, dataset_root: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to register broad inputs") from error
    parquet = pq.ParquetFile(path)
    schema = {field.name: str(field.type) for field in parquet.schema_arrow}
    if not {"src", "tgt"}.issubset(schema):
        raise ValueError(f"Parquet file lacks src/tgt columns: {path}")
    rows = int(parquet.metadata.num_rows)
    if rows <= 0:
        raise ValueError(f"Parquet file is empty: {path}")
    return {
        "split": split,
        "relative_path": str(path.relative_to(dataset_root)),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
        "schema": schema,
    }


def dataset_contract(
    *,
    profile: str,
    dataset_root: Path,
    repo_id: str,
    revision: str,
    source_provenance: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if profile in EXPECTED_DATASETS:
        expected = EXPECTED_DATASETS[profile]
        if repo_id != expected["repo_id"] or revision != expected["revision"]:
            raise ValueError(
                f"{profile} must remain pinned to "
                f"{expected['repo_id']}@{expected['revision']}"
            )
    elif profile == "finance":
        if not repo_id or len(revision) < 7:
            raise ValueError("finance reconstruction requires a source and revision")
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(profile)
    combined: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        paths = sorted((dataset_root / split).glob("*.parquet"))
        if not paths:
            raise ValueError(f"dataset split has no Parquet files: {split}")
        entries = [parquet_entry(path, split, dataset_root) for path in paths]
        combined[split] = {
            "rows": sum(entry["rows"] for entry in entries),
            "files": [entry["path"] for entry in entries],
        }
        inventory.extend({"role": "formal_dataset", **entry} for entry in entries)
    contract = {
        "profile": profile,
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_path": str(dataset_root),
        "text_format": TEXT_FORMATS[profile],
        "paper_exactness": PAPER_EXACTNESS[profile],
        "combined_splits": combined,
        "total_rows": sum(value["rows"] for value in combined.values()),
        "selection_contract": (
            "all registered rows; deterministic subset construction must be "
            "documented upstream and is bound by the file inventory"
        ),
    }
    if source_provenance is not None:
        contract["source_provenance"] = source_provenance
    return contract, inventory


def model_contract(
    model: str, snapshot: Path, hf_home: Path
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    spec = EXPECTED_MODELS[model]
    _inside(snapshot, hf_home, f"{model} snapshot")
    if snapshot.name != spec["revision"]:
        raise ValueError(
            f"{model} snapshot directory must name revision {spec['revision']}"
        )
    config_path = snapshot / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid model configuration: {config_path}") from error
    if config.get("model_type") != spec["model_type"]:
        raise ValueError(f"{model} model_type does not match the pinned contract")
    paths = sorted(path for path in snapshot.iterdir() if path.is_file())
    if config_path not in paths:
        raise ValueError(f"{model} snapshot has no config.json")
    weight_names = {path.name for path in paths}
    if not any(
        name == "model.safetensors"
        or name == "model.safetensors.index.json"
        or name == "pytorch_model.bin"
        or name == "pytorch_model.bin.index.json"
        or (name.startswith("model-") and name.endswith(".safetensors"))
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
        for name in weight_names
    ):
        raise ValueError(f"{model} snapshot has no recognized model weights")
    files = [
        {
            "relative_path": path.name,
            "path": str(path.absolute()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    model_key = str(spec["manifest_key"])
    contract = {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "snapshot_path": str(snapshot.absolute()),
        "model_type": spec["model_type"],
        "objective": spec["objective"],
        "target_modules": list(spec["target_modules"]),
        "trust_remote_code": bool(spec["trust_remote_code"]),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    inventory = [
        {"role": "model", "model": model_key, **item} for item in files
    ]
    return model_key, contract, inventory


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = _inside(args.dataset_root, args.data_root, "dataset root")
    formal_dataset, inventory = dataset_contract(
        profile=args.profile,
        dataset_root=dataset_root,
        repo_id=args.dataset_repo_id,
        revision=args.dataset_revision,
        source_provenance=(
            None
            if getattr(args, "source_repo_id", None) is None
            else {
                "repo_id": args.source_repo_id,
                "revision": args.source_revision,
                "selection_contract": args.source_selection_contract,
            }
        ),
    )
    models: dict[str, Any] = {}
    seen_models: set[str] = set()
    for model, raw_snapshot in args.model_snapshot:
        if model in seen_models:
            raise ValueError(f"duplicate model snapshot: {model}")
        seen_models.add(model)
        key, contract, model_inventory = model_contract(
            model, raw_snapshot, args.hf_home
        )
        models[key] = contract
        inventory.extend(model_inventory)
    if not models:
        raise ValueError("at least one --model-snapshot is required")
    inventory.sort(key=lambda item: (str(item["role"]), str(item["path"])))
    return {
        "schema_version": 2,
        "status": "STAGING_COMPLETE_VERIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_home": str(args.hf_home),
        "data_root": str(args.data_root),
        "formal_dataset": formal_dataset,
        "models": models,
        "inventory_files": len(inventory),
        "inventory_bytes": sum(int(item["bytes"]) for item in inventory),
        "inventory_sha256": canonical_json_fingerprint(inventory),
        "inventory": inventory,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(TEXT_FORMATS), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-repo-id")
    parser.add_argument("--source-revision")
    parser.add_argument("--source-selection-contract")
    parser.add_argument("--model-snapshot", action="append", type=parse_model_snapshot)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--marker", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    args.hf_home = args.hf_home.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.model_snapshot = args.model_snapshot or []
    source_values = (
        args.source_repo_id,
        args.source_revision,
        args.source_selection_contract,
    )
    if any(value is not None for value in source_values) and not all(
        isinstance(value, str) and value for value in source_values
    ):
        raise SystemExit(
            "--source-repo-id, --source-revision, and "
            "--source-selection-contract must be supplied together"
        )
    args.model_snapshot = [
        (model, path.expanduser().absolute()) for model, path in args.model_snapshot
    ]
    if args.manifest.parent != args.data_root:
        raise SystemExit("--manifest must be directly below --data-root")
    marker = (
        args.marker.expanduser().resolve()
        if args.marker is not None
        else args.data_root / ".complete"
    )
    if marker.parent != args.data_root:
        raise SystemExit("--marker must be directly below --data-root")
    manifest = build_manifest(args)
    atomic_json(args.manifest, manifest)
    atomic_json(
        marker,
        {
            "schema_version": 2,
            "status": "STAGING_COMPLETE_VERIFIED",
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "inventory_sha256": manifest["inventory_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "status": "registered",
                "profile": args.profile,
                "manifest": str(args.manifest),
                "models": sorted(manifest["models"]),
                "rows": manifest["formal_dataset"]["combined_splits"],
                "inventory_sha256": manifest["inventory_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
