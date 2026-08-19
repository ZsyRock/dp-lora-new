from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from paper_repro.train_federated import validate_input_manifest
from scripts.register_broad_scope_inputs import build_manifest


def _write_dataset(root: Path) -> Path:
    dataset = root / "datasets" / "slimpajama"
    for split in ("train", "validation", "test"):
        directory = dataset / split
        directory.mkdir(parents=True)
        pq.write_table(
            pa.table({"src": [f"{split} text"], "tgt": [""]}),
            directory / "part.parquet",
        )
    return dataset


def _write_model(root: Path) -> tuple[Path, Path]:
    hf_home = root / "hf"
    revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    snapshot = hf_home / "hub" / "models--gpt2" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "gpt2"}), encoding="utf-8"
    )
    (snapshot / "model.safetensors").write_bytes(b"fake-weights")
    return hf_home, snapshot


def test_registered_slimpajama_manifest_is_accepted_by_trainer_validator(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path)
    hf_home, snapshot = _write_model(tmp_path)
    data_root = tmp_path / "datasets"
    args = argparse.Namespace(
        profile="slimpajama",
        dataset_root=dataset,
        dataset_repo_id="cerebras/SlimPajama-627B",
        dataset_revision="417f7eebaec467f82121948075e8b98d33ffb58a",
        model_snapshot=[("gpt2", snapshot)],
        hf_home=hf_home,
        data_root=data_root,
    )
    manifest = build_manifest(args)
    path = data_root / "input-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    validated = validate_input_manifest(path)
    assert validated["formal_dataset"]["profile"] == "slimpajama"
    assert set(validated["models"]) == {"gpt2"}


def test_finance_must_be_labelled_as_unreleased_reconstruction(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path)
    hf_home, snapshot = _write_model(tmp_path)
    args = argparse.Namespace(
        profile="finance",
        dataset_root=dataset,
        dataset_repo_id="example/finance-reconstruction",
        dataset_revision="abcdef1234567",
        model_snapshot=[("gpt2", snapshot)],
        hf_home=hf_home,
        data_root=tmp_path / "datasets",
    )
    manifest = build_manifest(args)
    assert (
        manifest["formal_dataset"]["paper_exactness"]
        == "paper_source_not_released"
    )
