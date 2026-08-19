#!/usr/bin/env python3
"""Run and collect paper-default DP-LoRA baseline trajectories.

This campaign intentionally contains fixed-C DP-LoRA only.  Exact slack/CDF
endpoints are reconstructed after training from NON_DP_PRIVATE_DIAGNOSTIC raw
norms so later Full-SlaClip targets can be selected without changing the
baseline mechanism.  SlaClip and SlaClip-Q are not executable methods here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from paper_repro import broad_scope_campaign as broad
    from paper_repro.slaclip import (
        build_slack_vector,
        normalize_noisy_slack,
        stationary_beta_from_exact_endpoints,
    )
except ModuleNotFoundError:  # direct-script execution
    import broad_scope_campaign as broad  # type: ignore[no-redef]
    from slaclip import (  # type: ignore[no-redef]
        build_slack_vector,
        normalize_noisy_slack,
        stationary_beta_from_exact_endpoints,
    )


SCHEMA_VERSION = 1
PAPER_MODELS = frozenset({"bert", "gpt2", "chatglm2", "llama2"})
STAGEABLE_MODELS = frozenset({"bert", "gpt2", "chatglm2"})
DOMAINS = frozenset({"meddialog", "slimpajama", "finance"})
GROUPS = ("A", "B")


def load(path: Path, label: str) -> dict[str, Any]:
    return broad.load_object(path, label)


def validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(spec)
    supplied_fingerprint = raw.pop("spec_sha256", None)
    broad._require_exact_keys(
        raw,
        {
            "schema_version", "campaign_name", "description", "domains",
            "models", "blocked_paper_models", "paper_default", "seeds",
            "evaluation", "scientific_boundary",
        },
        "default baseline specification",
    )
    spec = raw
    if spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported default-baseline schema")
    domains = spec["domains"]
    if not isinstance(domains, list) or {
        item.get("id") for item in domains if isinstance(item, dict)
    } != DOMAINS:
        raise ValueError("all three paper training domains are required")
    for item in domains:
        broad._require_exact_keys(item, {"id", "paper_exactness"}, "domain")
    models = spec["models"]
    if not isinstance(models, list) or set(models) != STAGEABLE_MODELS:
        raise ValueError("stageable matrix must contain BERT, GPT-2, and ChatGLM2")
    blocked = spec["blocked_paper_models"]
    if not isinstance(blocked, dict) or set(blocked) != PAPER_MODELS - STAGEABLE_MODELS:
        raise ValueError("every omitted paper model needs an explicit blocker")
    defaults = spec["paper_default"]
    expected_defaults = {
        "method": "paper_dp_lora", "num_clients": 5, "rounds": 50,
        "batch_size": 8, "noise_multiplier": 2.0,
        "learning_rate": 5e-4, "clip_norm": 10.0,
        "rank": 512, "max_seq_length": 128, "delta": 1e-5,
        "slaclip_num_slots_for_offline_diagnostics": 5,
    }
    if defaults != expected_defaults:
        raise ValueError("paper-default hyperparameters differ from the frozen contract")
    seeds = spec["seeds"]
    if (
        not isinstance(seeds, list) or len(seeds) < 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("at least three unique positive seeds are required")
    boundary = spec["scientific_boundary"]
    if (
        boundary.get("full_slaclip_run") is not False
        or boundary.get("slaclip_q_run") is not False
        or boundary.get("exact_paper_reproduction") is not False
    ):
        raise ValueError("baseline claim boundaries are invalid")
    value = json.loads(json.dumps(spec))
    value["spec_sha256"] = broad.fingerprint(spec)
    if supplied_fingerprint is not None and supplied_fingerprint != value["spec_sha256"]:
        raise ValueError("supplied default-baseline fingerprint is invalid")
    return value


def expand_arms(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = validate_spec(spec)
    defaults = value["paper_default"]
    arms: list[dict[str, Any]] = []
    for domain in value["domains"]:
        for model in value["models"]:
            for seed in value["seeds"]:
                arm = {
                    "stage": "paper_default_baseline",
                    "domain": domain["id"],
                    "model": model,
                    "seed": seed,
                    "method": defaults["method"],
                    "clip_norm": defaults["clip_norm"],
                    **{
                        key: defaults[key]
                        for key in (
                            "num_clients", "rounds", "batch_size",
                            "noise_multiplier", "learning_rate", "rank",
                            "max_seq_length", "delta",
                        )
                    },
                }
                arm["rng_domain"] = f"paper-default-{domain['id']}-{model}-s{seed}"
                arm["arm_id"] = arm["rng_domain"] + "-fixed-c10"
                arms.append(arm)
    return arms


def validate_input_index(
    spec: Mapping[str, Any], index: Mapping[str, Any]
) -> dict[str, Path]:
    value = validate_spec(spec)
    broad._require_exact_keys(index, {"schema_version", "domains"}, "input index")
    if index["schema_version"] != 1 or not isinstance(index["domains"], dict):
        raise ValueError("invalid input index")
    expected = {item["id"] for item in value["domains"]}
    if set(index["domains"]) != expected:
        raise ValueError("input index does not cover all training domains")
    paths: dict[str, Path] = {}
    for domain, raw in index["domains"].items():
        path = Path(str(raw))
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"missing absolute input manifest for {domain}: {path}")
        paths[domain] = path.resolve()
    return paths


def _summary_complete(path: Path, arm: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        value = load(path, "root final summary")
    except ValueError:
        return False
    models = value.get("models")
    return bool(
        value.get("status") == "COMPLETED"
        and value.get("method") == "paper_dp_lora"
        and isinstance(models, dict)
        and set(models) == {arm["model"]}
        and isinstance(models[arm["model"]], dict)
        and models[arm["model"]].get("status") == "COMPLETED"
    )


def _run_arms(
    spec: Mapping[str, Any], *, arms: Sequence[Mapping[str, Any]],
    input_index: Mapping[str, Any], campaign_root: Path, repository: Path,
    python_bin: Path, private_key: Path, smoke: bool,
) -> dict[str, Any]:
    value = validate_spec(spec)
    manifests = validate_input_index(spec, input_index)
    root = campaign_root.resolve()
    section = "preflight-smokes" if smoke else "arms"
    for name in (section, "arm-logs", "arm-status"):
        (root / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    complete = 0
    for index, arm in enumerate(arms):
        output = root / section / str(arm["arm_id"])
        summary = output / "final_summary.json"
        if _summary_complete(summary, arm):
            complete += 1
            continue
        command = broad._arm_command(
            arm, repository=repository.resolve(), python_bin=python_bin.resolve(),
            input_manifest=manifests[str(arm["domain"])], output_dir=output,
            private_key=private_key.resolve(),
        )
        if smoke:
            command.append("--smoke")
        status_path = root / "arm-status" / f"{arm['arm_id']}{'-smoke' if smoke else ''}.json"
        broad._write_json(status_path, {
            "status": "RUNNING", "smoke": smoke, "index": index,
            "arm_id": arm["arm_id"], "arm_sha256": broad.fingerprint(arm),
            "spec_sha256": value["spec_sha256"],
        })
        suffix = "smoke" if smoke else "train"
        with (root / "arm-logs" / f"{arm['arm_id']}-{suffix}.out").open("ab") as out, (
            root / "arm-logs" / f"{arm['arm_id']}-{suffix}.err"
        ).open("ab") as err:
            result = subprocess.run(
                command, cwd=repository.resolve(), stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, check=False,
            )
        if result.returncode != 0 or not _summary_complete(summary, arm):
            broad._write_json(status_path, {
                "status": "FAILED_OR_CHECKPOINTED", "smoke": smoke,
                "index": index, "arm_id": arm["arm_id"],
                "return_code": result.returncode,
                "arm_sha256": broad.fingerprint(arm),
                "spec_sha256": value["spec_sha256"],
            })
            raise RuntimeError(f"baseline arm failed: {arm['arm_id']}")
        complete += 1
        broad._write_json(status_path, {
            "status": "COMPLETED", "smoke": smoke, "index": index,
            "arm_id": arm["arm_id"], "arm_sha256": broad.fingerprint(arm),
            "spec_sha256": value["spec_sha256"],
        })
    return {
        "status": "COMPLETED", "smoke": smoke, "arm_count": len(arms),
        "completed_arm_count": complete, "spec_sha256": value["spec_sha256"],
    }


def run_smokes(spec: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    all_arms = expand_arms(spec)
    selected = []
    for model in validate_spec(spec)["models"]:
        selected.append(next(
            arm for arm in all_arms
            if arm["domain"] == "meddialog" and arm["model"] == model
        ))
    return _run_arms(spec, arms=selected, smoke=True, **kwargs)


def run_campaign(spec: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _run_arms(spec, arms=expand_arms(spec), smoke=False, **kwargs)


def _mean(records: Sequence[Mapping[str, Any]], group: str, key: str) -> float:
    return statistics.fmean(float(record["gradient_groups"][group][key]) for record in records)


def collect(
    spec: Mapping[str, Any], campaign_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    value = validate_spec(spec)
    root = campaign_root.resolve()
    slots = int(value["paper_default"]["slaclip_num_slots_for_offline_diagnostics"])
    round_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    for arm in expand_arms(spec):
        model_root = root / "arms" / arm["arm_id"] / arm["model"]
        final = load(model_root / "final_summary.json", "model final summary")
        if final.get("status") != "COMPLETED":
            raise ValueError(f"incomplete baseline arm: {arm['arm_id']}")
        for evaluation in final.get("evaluations", []):
            evaluation_rows.append({
                "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
                "domain": arm["domain"], "model": arm["model"],
                "seed": arm["seed"], "round": evaluation["round"],
                "loss": evaluation["loss"], "exp_loss": evaluation["exp_loss"],
                "records": evaluation["records"], "objective": evaluation["objective"],
            })
        for round_index in range(1, int(arm["rounds"]) + 1):
            shard = load(
                model_root / "private_diagnostics" / "rounds" / f"round-{round_index:05d}.json",
                "round diagnostic shard",
            )
            records = shard.get("client_records")
            summary = shard.get("round_summary")
            if not isinstance(records, list) or not records or not isinstance(summary, dict):
                raise ValueError(f"invalid diagnostic shard: {arm['arm_id']}/{round_index}")
            endpoint_noise_std = float(arm["noise_multiplier"]) * math.sqrt(
                slots / float(arm["num_clients"])
            )
            for group in GROUPS:
                signals = [
                    build_slack_vector(
                        float(record["gradient_groups"][group]["raw_norm"]),
                        float(arm["clip_norm"]), slots,
                    )
                    for record in records
                ]
                exact = normalize_noisy_slack(
                    [math.fsum(signal[k] for signal in signals) for k in range(slots)],
                    float(arm["clip_norm"]), slots, len(records),
                )
                stationary = stationary_beta_from_exact_endpoints(
                    float(arm["clip_norm"]), exact[0], exact[-1], epsilon=1e-6
                )
                group_summary = summary[group]
                federated = summary["federated_update"][group]
                round_rows.append({
                    "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
                    "domain": arm["domain"], "model": arm["model"],
                    "seed": arm["seed"], "round": round_index, "group": group,
                    "method": arm["method"], "clip_norm_initial": arm["clip_norm"],
                    "clip_norm_current": arm["clip_norm"], "clip_norm_change": 0.0,
                    "actual_clipped_count": group_summary["clipped_count"],
                    "actual_clipped_fraction": group_summary["clipped_fraction"],
                    "would_clip_fraction": group_summary["would_clip_fraction"],
                    "mean_raw_norm": group_summary["mean_raw_norm"],
                    "max_raw_norm": group_summary["max_raw_norm"],
                    "mean_clip_factor": group_summary["mean_clip_factor"],
                    "mean_removed_gradient_l2": _mean(records, group, "removed_gradient_l2"),
                    "mean_retained_energy_fraction": _mean(records, group, "retained_energy_fraction"),
                    "mean_noise_l2_norm": group_summary["mean_noise_l2_norm"],
                    "mean_private_gradient_l2_norm": group_summary["mean_private_gradient_l2_norm"],
                    "mean_relative_local_update": group_summary["mean_relative_local_update"],
                    "aggregate_signal_gradient_l2": federated["aggregate_signal_gradient_l2"],
                    "aggregate_noise_gradient_l2": federated["aggregate_noise_gradient_l2"],
                    "signal_to_noise_l2_ratio": federated["signal_to_noise_l2_ratio"],
                    "signal_noise_cosine": federated["signal_noise_cosine"],
                    "actual_global_update_l2": federated["actual_global_update_l2"],
                    "relative_global_update": federated["relative_global_update"],
                    "mean_training_loss": summary["mean_training_loss"],
                    "supervised_tokens": summary["supervised_tokens"],
                    "exact_cdf_near_threshold_q": exact[0],
                    "exact_cdf_near_zero_r": exact[-1],
                    "near_zero_adjusted_fraction": stationary["near_zero_adjusted"],
                    "stationary_full_slaclip_beta": stationary["stationary_beta"],
                    "predicted_normalized_endpoint_noise_std": endpoint_noise_std,
                    "q_endpoint_signal_to_noise": exact[0] / endpoint_noise_std,
                    "r_endpoint_signal_to_noise": exact[-1] / endpoint_noise_std,
                })
                for record in records:
                    metrics = record["gradient_groups"][group]
                    client_rows.append({
                        "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
                        "domain": arm["domain"], "model": arm["model"],
                        "seed": arm["seed"], "round": round_index,
                        "client": record["client"], "group": group,
                        "clip_norm": metrics["clip_threshold"],
                        "raw_norm": metrics["raw_norm"],
                        "raw_to_threshold_ratio": metrics["raw_to_threshold_ratio"],
                        "clipped": metrics["clipped"],
                        "clip_factor": metrics["clip_factor"],
                        "clipped_norm": metrics["clipped_norm"],
                        "removed_gradient_l2": metrics["removed_gradient_l2"],
                        "retained_energy_fraction": metrics["retained_energy_fraction"],
                        "noise_l2_norm": metrics["noise_l2_norm"],
                        "signal_to_noise_l2_ratio": metrics["signal_to_noise_l2_ratio"],
                        "signal_noise_cosine": metrics["signal_noise_cosine"],
                        "relative_local_update": metrics["relative_local_update"],
                        "loss": record["loss"],
                    })
    return round_rows, client_rows, evaluation_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    broad._write_csv(path.resolve(), rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-spec")
    validate.add_argument("--spec", type=Path, required=True)
    expand = commands.add_parser("expand")
    expand.add_argument("--spec", type=Path, required=True)
    for name in ("run-smokes", "run-campaign"):
        command = commands.add_parser(name)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--input-index", type=Path, required=True)
        command.add_argument("--campaign-root", type=Path, required=True)
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--python-bin", type=Path, required=True)
        command.add_argument("--private-key", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--spec", type=Path, required=True)
    collect_parser.add_argument("--campaign-root", type=Path, required=True)
    collect_parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    spec = load(args.spec.resolve(), "default baseline specification")
    if args.command == "validate-spec":
        value = validate_spec(spec)
        print(json.dumps({
            "status": "VALID", "spec_sha256": value["spec_sha256"],
            "arm_count": len(expand_arms(spec)), "full_slaclip": False,
            "slaclip_q": False, "blocked_paper_models": value["blocked_paper_models"],
        }, indent=2, sort_keys=True))
    elif args.command == "expand":
        print(json.dumps({"arms": expand_arms(spec)}, indent=2, sort_keys=True))
    elif args.command in {"run-smokes", "run-campaign"}:
        kwargs = {
            "input_index": load(args.input_index.resolve(), "input index"),
            "campaign_root": args.campaign_root,
            "repository": args.repository,
            "python_bin": args.python_bin,
            "private_key": args.private_key,
        }
        result = run_smokes(spec, **kwargs) if args.command == "run-smokes" else run_campaign(spec, **kwargs)
        broad._write_json(args.output.resolve(), result)
    elif args.command == "collect":
        rounds, clients, evaluations = collect(spec, args.campaign_root)
        output = args.output_directory.resolve()
        write_csv(output / "baseline_round_telemetry.csv", rounds)
        write_csv(output / "baseline_client_telemetry.csv", clients)
        write_csv(output / "baseline_evaluation_telemetry.csv", evaluations)
        broad._write_json(output / "baseline_telemetry_manifest.json", {
            "status": "COMPLETE", "spec_sha256": validate_spec(spec)["spec_sha256"],
            "round_rows": len(rounds), "client_rows": len(clients),
            "evaluation_rows": len(evaluations),
            "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
        })


if __name__ == "__main__":
    main()
