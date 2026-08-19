#!/usr/bin/env python3
"""Stage deterministic public reconstructions for default baseline training.

The paper's SlimPajama selection and 9,540-row finance corpus were not
released.  This script therefore records source revisions and deterministic
selection rules rather than silently presenting these files as paper-exact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

try:
    from scripts.register_broad_scope_inputs import atomic_json, sha256_file
except ModuleNotFoundError:  # direct-script execution
    from register_broad_scope_inputs import atomic_json, sha256_file  # type: ignore[no-redef]


SLIM_REPO = "iankur/SlimPajama-1B"
SLIM_REVISION = "60cbb9c02f5db40156e2f220bd5256853abbffd5"
FINANCE_REPO = "FinGPT/fingpt-sentiment-train"
FINANCE_REVISION = "a2701f2155c1371ba38b92a2c41e8b0c0fca7614"
FINANCE_ROWS = 9_540


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _download_parquets(repo: str, revision: str, cache: Path) -> list[Path]:
    info = HfApi().dataset_info(repo, revision=revision)
    names = sorted(
        sibling.rfilename
        for sibling in info.siblings or []
        if sibling.rfilename.endswith(".parquet")
    )
    if not names:
        raise RuntimeError(f"source dataset has no Parquet files: {repo}")
    return [
        Path(hf_hub_download(
            repo_id=repo, repo_type="dataset", filename=name,
            revision=revision, cache_dir=cache,
        ))
        for name in names
    ]


def _standard_table(src: Iterable[str], tgt: Iterable[str]) -> pa.Table:
    return pa.table({
        "src": pa.array(list(src), type=pa.string()),
        "tgt": pa.array(list(tgt), type=pa.string()),
    })


def stage_slimpajama(root: Path, cache: Path) -> dict[str, Any]:
    paths = _download_parquets(SLIM_REPO, SLIM_REVISION, cache)
    outputs: list[dict[str, Any]] = []
    counters = {"train": 0, "validation": 0, "test": 0}
    for source in paths:
        name = source.name
        split = "validation" if "validation" in name else "test" if "test" in name else "train"
        table = pq.read_table(source, columns=["text"])
        text = pc.cast(table["text"], pa.string())
        standardized = pa.table({
            "src": text,
            "tgt": pa.array([""] * len(table), type=pa.string()),
        })
        index = counters[split]
        counters[split] += 1
        target = root / split / f"part-{index:05d}.parquet"
        _atomic_parquet(target, standardized)
        outputs.append({
            "split": split, "path": str(target), "rows": len(standardized),
            "bytes": target.stat().st_size, "sha256": sha256_file(target),
            "source_file": name,
        })
    if any(counters[split] == 0 for split in counters):
        raise RuntimeError("SlimPajama reconstruction lacks a required split")
    return {
        "profile": "slimpajama", "paper_exactness": "public_subset_reconstruction",
        "official_source": "cerebras/SlimPajama-627B",
        "mirror_repo_id": SLIM_REPO, "mirror_revision": SLIM_REVISION,
        "selection": "all rows in the pinned public 1B-token SlimPajama mirror",
        "files": outputs,
    }


def _finance_label(value: str) -> str | None:
    normalized = value.strip().lower()
    if "neutral" in normalized:
        return "Neutral"
    if "positive" in normalized or normalized in {"bullish", "strongly bullish"}:
        return "Bullish"
    if "negative" in normalized or normalized in {"bearish", "strongly bearish"}:
        return "Bearish"
    return None


def stage_finance(root: Path, cache: Path) -> dict[str, Any]:
    paths = _download_parquets(FINANCE_REPO, FINANCE_REVISION, cache)
    table = pa.concat_tables([
        pq.read_table(path, columns=["input", "output"]) for path in paths
    ]).combine_chunks()
    candidates: list[tuple[str, str, str]] = []
    for index in range(len(table)):
        source = str(table["input"][index].as_py() or "").strip()
        label = _finance_label(str(table["output"][index].as_py() or ""))
        if not source or label is None:
            continue
        key = hashlib.sha256(
            f"{source}\0{label}\0{index}".encode("utf-8")
        ).hexdigest()
        candidates.append((key, source, label))
    candidates.sort()
    selected = candidates[:FINANCE_ROWS]
    if len(selected) != FINANCE_ROWS:
        raise RuntimeError("finance source has fewer than 9,540 usable rows")
    split_sizes = {"train": 7_632, "validation": 954, "test": 954}
    outputs = []
    offset = 0
    for split, size in split_sizes.items():
        rows = selected[offset:offset + size]
        offset += size
        target = root / split / "part-00000.parquet"
        standardized = _standard_table(
            (row[1] for row in rows), (row[2] for row in rows)
        )
        _atomic_parquet(target, standardized)
        outputs.append({
            "split": split, "path": str(target), "rows": len(standardized),
            "bytes": target.stat().st_size, "sha256": sha256_file(target),
        })
    counts = {label: sum(row[2] == label for row in selected) for label in ("Bearish", "Bullish", "Neutral")}
    return {
        "profile": "finance", "paper_exactness": "paper_source_not_released",
        "source_repo_id": FINANCE_REPO, "source_revision": FINANCE_REVISION,
        "selection": "lowest SHA-256 keys after three-class normalization; exactly 9,540 rows",
        "downstream_overlap_warning": "FinGPT aggregates public sentiment datasets; FPB/FiQA-SA/TFNS evaluation requires an explicit overlap audit.",
        "label_counts": counts, "files": outputs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["slimpajama", "finance"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    root = args.output_root.resolve()
    cache = args.hf_home.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = stage_slimpajama(root, cache) if args.profile == "slimpajama" else stage_finance(root, cache)
    record["schema_version"] = 1
    record["status"] = "STAGING_COMPLETE_VERIFIED"
    atomic_json(root / "source-provenance.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
