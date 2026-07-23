#!/usr/bin/env python3
"""Stage and verify the immutable inputs used by the DP-LoRA paper runner.

The formal MedDialog input is the union of the two configurations in the
``lighteval/med_dialog`` Parquet dataset.  The much smaller legacy
``UCSD26/medical_dialog`` ``processed.en`` JSON files are intentionally not
part of this manifest and must only be used for smoke tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "bert-base-uncased": {
        "repo_id": "google-bert/bert-base-uncased",
        "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "model_type": "bert",
        "required_files": (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ),
        "minimum_weight_bytes": 400_000_000,
    },
    "gpt2": {
        "repo_id": "openai-community/gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "model_type": "gpt2",
        "required_files": (
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
        "minimum_weight_bytes": 500_000_000,
    },
}

DATASET_REPO_ID = "lighteval/med_dialog"
DATASET_REVISION = "ce8a234c92ea9a37743ad8154253ba897a4a70a5"
DATASET_EXPECTED: dict[str, dict[str, int]] = {
    "healthcaremagic": {
        "train": 181_122,
        "validation": 22_641,
        "test": 22_642,
    },
    "icliniq": {
        "train": 24_851,
        "validation": 3_105,
        "test": 3_108,
    },
}
PARQUET_SCHEMA = {"tgt": "string", "src": "string", "id": "int64"}


def user_and_scratch() -> tuple[str, Path]:
    user_name = pwd.getpwuid(os.getuid()).pw_name
    scratch = Path(
        os.environ.get("DPLORA_PAPER_SCRATCH_ROOT")
        or os.environ.get("SCRATCH")
        or f"/scratch/{user_name}"
    )
    return user_name, scratch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_symlink(link: Path, target: Path) -> None:
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink dataset path: {link}")
    temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.symlink_to(target, target_is_directory=True)
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)


def require_inside(path: Path, root: Path, description: str) -> None:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(f"{description} is not inside {root}: {path}") from error


def validate_model_snapshot(
    name: str, spec: dict[str, Any], snapshot: Path, hf_home: Path
) -> dict[str, Any]:
    if snapshot.name != spec["revision"]:
        raise RuntimeError(
            f"{name} snapshot revision mismatch: {snapshot.name} != {spec['revision']}"
        )
    require_inside(snapshot, hf_home, f"{name} snapshot")
    config_path = snapshot / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {name} config: {config_path}") from error
    if config.get("model_type") != spec["model_type"]:
        raise RuntimeError(
            f"{name} model_type mismatch: {config.get('model_type')!r}"
        )

    files = []
    for relative_name in spec["required_files"]:
        path = snapshot / relative_name
        if not path.is_file():
            raise RuntimeError(f"required {name} file is missing: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"required {name} file is empty: {path}")
        files.append(
            {
                "relative_path": relative_name,
                "path": str(path),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    weight_size = (snapshot / "model.safetensors").stat().st_size
    if weight_size < spec["minimum_weight_bytes"]:
        raise RuntimeError(
            f"{name} model.safetensors is unexpectedly small: {weight_size} bytes"
        )
    return {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "snapshot_path": str(snapshot),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }


def parquet_metadata(path: Path) -> tuple[int, dict[str, str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to validate the formal MedDialog Parquet files"
        ) from error
    try:
        parquet_file = parquet.ParquetFile(path)
        rows = int(parquet_file.metadata.num_rows)
        schema = {
            field.name: str(field.type) for field in parquet_file.schema_arrow
        }
    except Exception as error:
        raise RuntimeError(f"invalid Parquet file: {path}") from error
    return rows, schema


def validate_dataset_snapshot(
    snapshot: Path, dataset_link: Path, hf_home: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if snapshot.name != DATASET_REVISION:
        raise RuntimeError(
            f"dataset snapshot revision mismatch: {snapshot.name} != {DATASET_REVISION}"
        )
    require_inside(snapshot, hf_home, "MedDialog snapshot")
    if not dataset_link.is_symlink():
        raise RuntimeError(f"formal dataset link is missing: {dataset_link}")
    if dataset_link.resolve(strict=True) != snapshot.resolve(strict=True):
        raise RuntimeError(
            f"formal dataset link resolves to the wrong snapshot: {dataset_link}"
        )

    configurations: dict[str, Any] = {}
    combined_splits: dict[str, dict[str, Any]] = {
        split: {"rows": 0, "files": []}
        for split in ("train", "validation", "test")
    }
    inventory: list[dict[str, Any]] = []
    for configuration, expected_splits in DATASET_EXPECTED.items():
        configurations[configuration] = {}
        for split, expected_rows in expected_splits.items():
            relative_path = (
                f"{configuration}/{split}-00000-of-00001.parquet"
            )
            path = dataset_link / relative_path
            if not path.is_file():
                raise RuntimeError(f"formal MedDialog file is missing: {path}")
            rows, schema = parquet_metadata(path)
            if rows != expected_rows:
                raise RuntimeError(
                    f"{configuration}/{split} row-count mismatch: "
                    f"{rows} != {expected_rows}"
                )
            if schema != PARQUET_SCHEMA:
                raise RuntimeError(
                    f"{configuration}/{split} schema mismatch: "
                    f"{schema} != {PARQUET_SCHEMA}"
                )
            item = {
                "configuration": configuration,
                "split": split,
                "relative_path": relative_path,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": rows,
                "schema": schema,
            }
            configurations[configuration][split] = item
            combined_splits[split]["rows"] += rows
            combined_splits[split]["files"].append(str(path))
            inventory.append({"role": "formal_dataset", **item})

    return (
        {
            "repo_id": DATASET_REPO_ID,
            "revision": DATASET_REVISION,
            "snapshot_path": str(snapshot),
            "formal_dataset_root": str(dataset_link),
            "schema": PARQUET_SCHEMA,
            "configurations": configurations,
            "combined_splits": combined_splits,
            "total_rows": sum(item["rows"] for item in combined_splits.values()),
            "total_bytes": sum(item["bytes"] for item in inventory),
        },
        inventory,
    )


def build_manifest(
    hf_home: Path,
    data_root: Path,
    model_snapshots: dict[str, Path],
    dataset_snapshot: Path,
    dataset_link: Path,
    *,
    created_at_utc: str,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    inventory: list[dict[str, Any]] = []
    for name, spec in MODEL_SPECS.items():
        model = validate_model_snapshot(name, spec, model_snapshots[name], hf_home)
        models[name] = model
        for item in model["files"]:
            inventory.append({"role": "model", "model": name, **item})

    dataset, dataset_inventory = validate_dataset_snapshot(
        dataset_snapshot, dataset_link, hf_home
    )
    inventory.extend(dataset_inventory)
    inventory.sort(key=lambda item: (item["role"], item["path"]))
    return {
        "schema_version": 1,
        "status": "STAGING_COMPLETE_VERIFIED",
        "created_at_utc": created_at_utc,
        "hf_home": str(hf_home),
        "data_root": str(data_root),
        "models": models,
        "formal_dataset": dataset,
        "inventory_files": len(inventory),
        "inventory_bytes": sum(item["bytes"] for item in inventory),
        "inventory_sha256": canonical_digest(inventory),
        "inventory": inventory,
    }


def read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must contain a JSON object: {path}")
    return value


def snapshot_paths_from_manifest(
    manifest: dict[str, Any], hf_home: Path
) -> tuple[dict[str, Path], Path]:
    models = manifest.get("models")
    dataset = manifest.get("formal_dataset")
    if not isinstance(models, dict) or not isinstance(dataset, dict):
        raise RuntimeError("input manifest is missing model or dataset metadata")
    model_paths: dict[str, Path] = {}
    for name, spec in MODEL_SPECS.items():
        recorded = models.get(name)
        if not isinstance(recorded, dict):
            raise RuntimeError(f"input manifest is missing model: {name}")
        if recorded.get("repo_id") != spec["repo_id"]:
            raise RuntimeError(f"input manifest {name} repo_id mismatch")
        if recorded.get("revision") != spec["revision"]:
            raise RuntimeError(f"input manifest {name} revision mismatch")
        model_paths[name] = Path(str(recorded.get("snapshot_path")))
        require_inside(model_paths[name], hf_home, f"recorded {name} snapshot")
    if dataset.get("repo_id") != DATASET_REPO_ID:
        raise RuntimeError("input manifest dataset repo_id mismatch")
    if dataset.get("revision") != DATASET_REVISION:
        raise RuntimeError("input manifest dataset revision mismatch")
    dataset_path = Path(str(dataset.get("snapshot_path")))
    require_inside(dataset_path, hf_home, "recorded dataset snapshot")
    return model_paths, dataset_path


def download_snapshots(hf_home: Path, max_workers: int) -> tuple[dict[str, Path], Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required to stage the pinned public snapshots"
        ) from error

    model_paths: dict[str, Path] = {}
    for name, spec in MODEL_SPECS.items():
        path = snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            cache_dir=hf_home,
            allow_patterns=list(spec["required_files"]),
            max_workers=max_workers,
            token=False,
        )
        model_paths[name] = Path(path)

    dataset_patterns = ["README.md"] + [
        f"{configuration}/{split}-00000-of-00001.parquet"
        for configuration, splits in DATASET_EXPECTED.items()
        for split in splits
    ]
    dataset_path = Path(
        snapshot_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=hf_home,
            allow_patterns=dataset_patterns,
            max_workers=max_workers,
            token=False,
        )
    )
    return model_paths, dataset_path


def check_existing(
    hf_home: Path, data_root: Path, manifest_path: Path, marker_path: Path
) -> dict[str, Any]:
    manifest = read_json(manifest_path, "input manifest")
    marker = read_json(marker_path, "completion marker")
    if marker.get("status") != "STAGING_COMPLETE_VERIFIED":
        raise RuntimeError("completion marker status is not verified")
    manifest_hash = sha256_file(manifest_path)
    if marker.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("completion marker does not match the input manifest")
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported input manifest schema version")
    if manifest.get("hf_home") != str(hf_home):
        raise RuntimeError("input manifest HF_HOME mismatch")
    if manifest.get("data_root") != str(data_root):
        raise RuntimeError("input manifest data-root mismatch")

    model_paths, dataset_path = snapshot_paths_from_manifest(manifest, hf_home)
    dataset_link = data_root / "meddialog-parquet"
    verified = build_manifest(
        hf_home,
        data_root,
        model_paths,
        dataset_path,
        dataset_link,
        created_at_utc=str(manifest.get("created_at_utc")),
    )
    for key in (
        "models",
        "formal_dataset",
        "inventory_files",
        "inventory_bytes",
        "inventory_sha256",
        "inventory",
    ):
        if manifest.get(key) != verified.get(key):
            raise RuntimeError(f"input manifest no longer matches staged {key}")
    if marker.get("inventory_sha256") != manifest.get("inventory_sha256"):
        raise RuntimeError("completion marker inventory digest mismatch")
    return manifest


def parse_args() -> argparse.Namespace:
    _, scratch = user_and_scratch()
    hf_home = Path(
        os.environ.get("DPLORA_PAPER_HF_HOME")
        or scratch / "cache/dp-lora-paper/huggingface"
    )
    data_root = Path(
        os.environ.get("DPLORA_PAPER_DATA_ROOT")
        or scratch / "datasets/dp-lora-paper"
    )
    manifest = Path(
        os.environ.get("DPLORA_PAPER_INPUT_MANIFEST")
        or data_root / "input-manifest.json"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--hf-home", type=Path, default=hf_home)
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--manifest", type=Path, default=manifest)
    parser.add_argument("--marker", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    os.umask(0o077)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    args = parse_args()
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    hf_home = args.hf_home.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    marker_path = (
        args.marker.expanduser().resolve()
        if args.marker is not None
        else data_root / ".complete"
    )
    if manifest_path.parent != data_root:
        raise SystemExit("--manifest must be located directly below --data-root")
    if marker_path.parent != data_root:
        raise SystemExit("--marker must be located directly below --data-root")

    if args.check_only:
        manifest = check_existing(
            hf_home, data_root, manifest_path, marker_path
        )
    else:
        hf_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        data_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(hf_home, 0o700)
        os.chmod(data_root, 0o700)
        model_paths, dataset_path = download_snapshots(
            hf_home, args.max_workers
        )
        dataset_link = data_root / "meddialog-parquet"
        atomic_symlink(dataset_link, dataset_path)
        manifest = build_manifest(
            hf_home,
            data_root,
            model_paths,
            dataset_path,
            dataset_link,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        atomic_write_json(manifest_path, manifest)
        marker = {
            "schema_version": 1,
            "status": "STAGING_COMPLETE_VERIFIED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "inventory_sha256": manifest["inventory_sha256"],
            "dataset_revision": DATASET_REVISION,
            "model_revisions": {
                name: spec["revision"] for name, spec in MODEL_SPECS.items()
            },
        }
        atomic_write_json(marker_path, marker)
        manifest = check_existing(
            hf_home, data_root, manifest_path, marker_path
        )

    print(
        json.dumps(
            {
                "status": "verified",
                "manifest": str(manifest_path),
                "marker": str(marker_path),
                "inventory_files": manifest["inventory_files"],
                "inventory_bytes": manifest["inventory_bytes"],
                "inventory_sha256": manifest["inventory_sha256"],
                "formal_dataset_rows": manifest["formal_dataset"][
                    "combined_splits"
                ],
                "model_revisions": {
                    name: model["revision"]
                    for name, model in manifest["models"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
