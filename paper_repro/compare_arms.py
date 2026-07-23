#!/usr/bin/env python3
"""Validate and summarize a matched three-arm DP-LoRA experiment.

This tool never interprets an internal MedDialog language-model loss as a
paper benchmark.  It proves only that the no-DP, clip-only, and paper-literal
arms used the same non-mechanism contract and stochastic data schedule.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from paper_repro.reproducibility import (
        METHOD_SPECS,
        canonical_json_fingerprint,
    )
except ModuleNotFoundError:  # Support direct execution.
    from reproducibility import (  # type: ignore[no-redef]
        METHOD_SPECS,
        canonical_json_fingerprint,
    )


EXPECTED_METHODS = (
    "no_dp_lora_control",
    "clip_only_control",
    "paper_dp_lora",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_contract(contract: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(contract)
    normalized.pop("method", None)
    effective = normalized.get("effective_config")
    if not isinstance(effective, dict):
        raise RuntimeError("scientific contract has no effective_config object")
    effective.pop("method", None)
    return normalized


def evaluation_delta(model_summary: dict[str, Any]) -> dict[str, Any]:
    evaluations = model_summary.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) < 2:
        raise RuntimeError("model summary has no initial/final evaluation pair")
    initial = evaluations[0]
    final = evaluations[-1]
    if initial.get("round") != 0:
        raise RuntimeError("model evaluation series does not start at round zero")
    initial_loss = float(initial["loss"])
    final_loss = float(final["loss"])
    return {
        "metric": "internal_disjoint_meddialog_lm_loss_not_paper_benchmark",
        "initial": initial_loss,
        "final": final_loss,
        "final_minus_initial": final_loss - initial_loss,
        "improved": final_loss < initial_loss,
    }


def build_comparison(arm_directories: dict[str, Path]) -> dict[str, Any]:
    if tuple(arm_directories) != EXPECTED_METHODS:
        raise ValueError(f"exactly these ordered methods are required: {EXPECTED_METHODS}")
    loaded: dict[str, dict[str, Any]] = {}
    normalized_fingerprints: dict[str, str] = {}
    for method, directory in arm_directories.items():
        run_config_path = directory / "run_config.json"
        final_summary_path = directory / "final_summary.json"
        run_config = load_object(run_config_path, f"{method} run config")
        final_summary = load_object(final_summary_path, f"{method} final summary")
        if run_config.get("method") != method or final_summary.get("method") != method:
            raise RuntimeError(f"method identity mismatch for {method}")
        if final_summary.get("status") != "COMPLETED":
            raise RuntimeError(f"arm is not complete: {method}")
        if final_summary.get("paper_result_reproduced") is not False:
            raise RuntimeError(f"arm has an unsafe paper reproduction claim: {method}")
        contract = run_config.get("scientific_contract")
        if not isinstance(contract, dict):
            raise RuntimeError(f"arm has no scientific contract: {method}")
        if canonical_json_fingerprint(contract) != run_config.get(
            "run_config_fingerprint"
        ):
            raise RuntimeError(f"scientific contract fingerprint mismatch: {method}")
        if final_summary.get("run_config_fingerprint") != run_config.get(
            "run_config_fingerprint"
        ):
            raise RuntimeError(f"root summary/config fingerprint mismatch: {method}")
        normalized_fingerprints[method] = canonical_json_fingerprint(
            normalized_contract(contract)
        )
        loaded[method] = {
            "directory": directory,
            "run_config": run_config,
            "final_summary": final_summary,
            "run_config_sha256": sha256_file(run_config_path),
            "final_summary_sha256": sha256_file(final_summary_path),
        }
    if len(set(normalized_fingerprints.values())) != 1:
        raise RuntimeError(
            "arms differ outside the explicitly allowed method contract"
        )

    reference_models = list(
        loaded[EXPECTED_METHODS[0]]["final_summary"].get("models", {})
    )
    if not reference_models:
        raise RuntimeError("comparison contains no models")
    model_comparisons: dict[str, Any] = {}
    for model in reference_models:
        schedules: dict[str, str] = {}
        supervision_schedules: dict[str, str] = {}
        partitions: dict[str, list[str]] = {}
        per_method: dict[str, Any] = {}
        for method in EXPECTED_METHODS:
            root_summary = loaded[method]["final_summary"]
            models = root_summary.get("models")
            if not isinstance(models, dict) or list(models) != reference_models:
                raise RuntimeError(f"model set/order mismatch: {method}")
            model_summary = models.get(model)
            if not isinstance(model_summary, dict):
                raise RuntimeError(f"missing {model} summary: {method}")
            model_summary_path = (
                loaded[method]["directory"] / model / "final_summary.json"
            )
            model_summary_on_disk = load_object(
                model_summary_path, f"{method}/{model} final summary"
            )
            if model_summary_on_disk != model_summary:
                raise RuntimeError(
                    f"root and model summaries do not match: {method}/{model}"
                )
            if model_summary.get("status") != "COMPLETED":
                raise RuntimeError(f"incomplete {model} summary: {method}")
            if model_summary.get("method") != method:
                raise RuntimeError(f"model/method identity mismatch: {method}/{model}")
            integrity = model_summary.get("adapter_integrity")
            if not isinstance(integrity, dict) or not integrity.get("all_finite"):
                raise RuntimeError(f"adapter integrity gate failed: {method}/{model}")
            adapter_path = (
                loaded[method]["directory"]
                / model
                / "final_adapter"
                / "adapter_model.safetensors"
            )
            if sha256_file(adapter_path) != model_summary.get("adapter_sha256"):
                raise RuntimeError(f"adapter checksum mismatch: {method}/{model}")
            adapter_config_path = adapter_path.with_name("adapter_config.json")
            if sha256_file(adapter_config_path) != model_summary.get(
                "adapter_config_sha256"
            ):
                raise RuntimeError(
                    f"adapter configuration checksum mismatch: {method}/{model}"
                )
            behavior = model_summary.get("behavior_summary")
            if not isinstance(behavior, dict):
                raise RuntimeError(f"behavior summary is missing: {method}/{model}")
            schedules[method] = str(behavior.get("sample_schedule_sha256"))
            supervision_schedules[method] = str(
                behavior.get("supervision_schedule_sha256")
            )
            partition_digests = model_summary.get("client_partition_sha256")
            if not isinstance(partition_digests, list):
                raise RuntimeError(f"partition digests are missing: {method}/{model}")
            partitions[method] = [str(value) for value in partition_digests]
            per_method[method] = {
                "release_class": METHOD_SPECS[method].release_class,
                "adapter_sha256": model_summary.get("adapter_sha256"),
                "client_steps": model_summary.get("client_steps"),
                "clipping": model_summary.get("clipping"),
                "behavior_summary": behavior,
                "internal_holdout": evaluation_delta(model_summary),
                "privacy_accounting": model_summary.get("privacy_accounting"),
            }
        if len(set(schedules.values())) != 1:
            raise RuntimeError(f"sample schedules do not match for {model}: {schedules}")
        if len(set(supervision_schedules.values())) != 1:
            raise RuntimeError(
                f"supervision schedules do not match for {model}: {supervision_schedules}"
            )
        if len({tuple(value) for value in partitions.values()}) != 1:
            raise RuntimeError(f"client partitions do not match for {model}")
        if per_method["no_dp_lora_control"]["clipping"]["any_group"]["count"] != 0:
            raise RuntimeError(f"no-DP arm unexpectedly applied clipping for {model}")
        model_comparisons[model] = {
            "matched_sample_schedule_sha256": next(iter(schedules.values())),
            "matched_supervision_schedule_sha256": next(
                iter(supervision_schedules.values())
            ),
            "matched_client_partition_sha256": next(iter(partitions.values())),
            "arms": per_method,
        }

    evidence = {
        method: {
            "directory": str(loaded[method]["directory"]),
            "run_config_sha256": loaded[method]["run_config_sha256"],
            "final_summary_sha256": loaded[method]["final_summary_sha256"],
            "run_config_fingerprint": loaded[method]["run_config"].get(
                "run_config_fingerprint"
            ),
        }
        for method in EXPECTED_METHODS
    }
    comparison_core = {
        "schema_version": 1,
        "status": "VALID_MATCHED_THREE_ARM_LEVEL1_COMPARISON",
        "claim_level": 1,
        "paper_result_reproduced": False,
        "paper_benchmarks_evaluated": False,
        "contains_slaclip": False,
        "base_model_only_arm_included": False,
        "normalized_scientific_contract_sha256": next(
            iter(normalized_fingerprints.values())
        ),
        "arm_evidence": evidence,
        "models": model_comparisons,
        "privacy_notice": {
            "controls_are_non_private": True,
            "exact_diagnostics_are_non_private": True,
            "paper_dp_arm_independently_certified": False,
            "epsilon": None,
            "sigma_is_not_epsilon": True,
        },
    }
    comparison_core["comparison_fingerprint"] = canonical_json_fingerprint(
        comparison_core
    )
    return comparison_core


def atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_arm(value: str) -> tuple[str, Path]:
    method, separator, raw_path = value.partition("=")
    if not separator or method not in EXPECTED_METHODS or not raw_path:
        raise argparse.ArgumentTypeError("arm must be EXPECTED_METHOD=PATH")
    return method, Path(raw_path).expanduser().resolve()


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    arm_directories: dict[str, Path] = {}
    for method, directory in args.arm:
        if method in arm_directories:
            raise SystemExit(f"duplicate --arm method: {method}")
        arm_directories[method] = directory
    ordered = {
        method: arm_directories[method]
        for method in EXPECTED_METHODS
        if method in arm_directories
    }
    if tuple(ordered) != EXPECTED_METHODS:
        raise SystemExit(f"all three methods are required: {EXPECTED_METHODS}")
    comparison = build_comparison(ordered)
    output = args.output.expanduser().resolve()
    if args.verify_existing:
        existing = load_object(output, "existing pair comparison")
        if existing.get("comparison_fingerprint") != comparison.get(
            "comparison_fingerprint"
        ):
            raise SystemExit("existing pair comparison fingerprint mismatch")
        print(json.dumps({"status": "verified", "output": str(output)}, indent=2))
        return
    comparison["created_at_utc"] = utc_now()
    try:
        atomic_create_json(output, comparison)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing comparison: {output}") from error
    print(
        json.dumps(
            {
                "status": comparison["status"],
                "comparison_fingerprint": comparison["comparison_fingerprint"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
