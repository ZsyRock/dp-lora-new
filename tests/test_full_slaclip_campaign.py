from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import paper_repro.full_slaclip_campaign as campaign_module
from paper_repro.full_slaclip_campaign import (
    FULL_SLACLIP_METHOD,
    _arm_command,
    aggregate_campaign,
    atomic_json,
    build_runtime_manifest,
    expand_spec,
    load_spec,
    validate_or_create_key,
    validate_runtime_manifest,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = REPOSITORY / "hpc" / "full-slaclip-campaign-spec.json"
SLURM_WORKER = REPOSITORY / "hpc" / "full_slaclip_campaign.sbatch"


def test_slurm_worker_exports_required_cuda_determinism_contract() -> None:
    worker = SLURM_WORKER.read_text(encoding="utf-8")
    required_export = "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
    assert worker.count(required_export) == 1
    assert worker.index(required_export) < worker.index("run_step()")


def test_checked_in_matrix_is_exact_and_contains_no_slaclip_q() -> None:
    arms = expand_spec(load_spec(SPEC))
    assert len(arms) == 60
    assert Counter(arm["family"] for arm in arms) == {
        "primary": 30,
        "sensitivity": 24,
        "control": 6,
    }
    assert Counter(arm["method"] for arm in arms) == {
        "paper_dp_lora": 15,
        "slaclip_dp_lora": 39,
        "no_dp_lora_control": 3,
        "clip_only_control": 3,
    }
    assert all("slaclip_q" not in arm["method"].lower() for arm in arms)
    assert [(arm["index"], arm["wave"], arm["lane"]) for arm in arms] == [
        (index, index // 2, index % 2) for index in range(60)
    ]

    primary_full = [
        arm
        for arm in arms
        if arm["family"] == "primary" and arm["method"] == FULL_SLACLIP_METHOD
    ]
    assert {arm["initial_clip_norm"] for arm in primary_full} == {
        0.1,
        1.0,
        5.0,
        10.0,
        20.0,
    }
    assert {arm["seed"] for arm in primary_full} == {42, 43, 44}
    assert {arm["slaclip_eta"] for arm in primary_full} == {0.2}
    assert {arm["slaclip_beta"] for arm in primary_full} == {0.5}
    assert all(arm["slaclip_num_slots"] == 15 for arm in primary_full)
    assert all(arm["slaclip_c_min"] == 0.1 for arm in primary_full)
    assert all(arm["slaclip_c_max"] == 50.0 for arm in primary_full)

    paper_style = [arm for arm in arms if arm["analysis_role"] == "paper_style_primary"]
    assert len(paper_style) == 6
    assert all(arm["initial_clip_norm"] == 10.0 for arm in paper_style)

    sensitivity = [arm for arm in arms if arm["family"] == "sensitivity"]
    assert len(sensitivity) == 24
    assert not any(
        arm["slaclip_eta"] == 0.2 and arm["slaclip_beta"] == 0.5
        for arm in sensitivity
    )
    assert all(arm["reference_arm_id"] for arm in sensitivity)


def test_runtime_manifest_is_self_hashing(tmp_path: Path) -> None:
    input_manifest = tmp_path / "input-manifest.json"
    input_manifest.write_text('{"pinned":true}\n', encoding="utf-8")
    runtime = build_runtime_manifest(
        SPEC,
        repository_sha="a" * 40,
        input_manifest=input_manifest,
        created_at_utc="2026-07-29T00:00:00+00:00",
    )
    validate_runtime_manifest(runtime)
    assert runtime["expected_arm_count"] == 60
    mutated = json.loads(json.dumps(runtime))
    mutated["arms"][0]["seed"] = 999
    try:
        validate_runtime_manifest(mutated)
    except RuntimeError as error:
        assert "fingerprint mismatch" in str(error)
    else:
        raise AssertionError("mutated runtime manifest was accepted")


def test_campaign_private_key_is_created_fail_closed(tmp_path: Path) -> None:
    key = tmp_path / "private" / "campaign.key"
    validate_or_create_key(key, create=True)
    assert len(key.read_bytes()) == 32
    assert key.stat().st_mode & 0o777 == 0o600
    alias = tmp_path / "private" / "alias.key"
    alias.symlink_to(key)
    try:
        validate_or_create_key(alias, create=False)
    except RuntimeError as error:
        assert "user-owned, 32 bytes" in str(error)
    else:
        raise AssertionError("symlinked campaign key was accepted")


def test_arm_command_uses_full_controller_endpoints_without_q_inputs(tmp_path: Path) -> None:
    arms = expand_spec(load_spec(SPEC))
    adaptive = next(arm for arm in arms if arm["method"] == FULL_SLACLIP_METHOD)
    fixed = next(arm for arm in arms if arm["method"] == "paper_dp_lora")
    common = {
        "repository": tmp_path / "repo",
        "python_bin": tmp_path / "env" / "bin" / "python",
        "input_manifest": tmp_path / "input.json",
        "private_key": tmp_path / "key",
        "stop_file": tmp_path / "stop",
    }
    adaptive_command = _arm_command(
        adaptive,
        output_dir=tmp_path / "adaptive",
        **common,
    )
    assert "--slaclip-beta" in adaptive_command
    assert "--slaclip-eta" in adaptive_command
    assert adaptive_command[adaptive_command.index("--slaclip-num-slots") + 1] == "15"
    assert adaptive_command[adaptive_command.index("--slaclip-c-max") + 1] == "50.0"
    assert "--slaclip-target" not in adaptive_command
    assert "--slaclip-calibration" not in adaptive_command
    assert all("slaclip_q" not in value.lower() for value in adaptive_command)

    fixed_command = _arm_command(
        fixed,
        output_dir=tmp_path / "fixed",
        **common,
    )
    assert "--slaclip-beta" not in fixed_command
    assert "--slaclip-num-slots" not in fixed_command


def test_completed_arm_fast_path_avoids_python_and_key_reload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_manifest = tmp_path / "input-manifest.json"
    input_manifest.write_text('{"pinned":true}\n', encoding="utf-8")
    runtime = build_runtime_manifest(
        SPEC,
        repository_sha="c" * 40,
        input_manifest=input_manifest,
        created_at_utc="2026-07-29T00:00:00+00:00",
    )
    root = tmp_path / "campaign"
    runtime_path = root / "runtime-manifest.json"
    atomic_json(runtime_path, runtime)
    arm = runtime["arms"][0]
    final_path = root / "arms" / arm["arm_id"] / "final_summary.json"
    atomic_json(
        final_path,
        {"status": "COMPLETED", "method": arm["method"]},
    )
    atomic_json(
        root / "arm-status" / f"{arm['arm_id']}.json",
        {
            "status": "COMPLETED",
            "arm_id": arm["arm_id"],
            "index": arm["index"],
            "method": arm["method"],
            "final_summary_sha256": campaign_module.sha256_file(final_path),
        },
    )
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(campaign_module, "repository_sha", lambda _path: "c" * 40)
    monkeypatch.setattr(campaign_module, "repository_dirty", lambda _path: False)
    result = campaign_module.run_arm(
        argparse.Namespace(
            manifest=runtime_path,
            arm_index=0,
            repository=repository,
            python_bin=tmp_path / "missing-python",
            private_key=tmp_path / "missing-key",
        )
    )
    assert result == 0


def test_comparisons_are_created_incrementally_and_all_reverified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_manifest = tmp_path / "input-manifest.json"
    input_manifest.write_text('{"pinned":true}\n', encoding="utf-8")
    runtime = build_runtime_manifest(
        SPEC,
        repository_sha="d" * 40,
        input_manifest=input_manifest,
        created_at_utc="2026-07-29T00:00:00+00:00",
    )
    root = tmp_path / "campaign"
    calls: list[tuple[str, bool]] = []

    def fake_comparator(
        *,
        baseline_dir: Path,
        adaptive_dir: Path,
        output: Path,
        verify_existing: bool,
    ) -> None:
        del baseline_dir
        calls.append((adaptive_dir.name, verify_existing))
        if verify_existing:
            assert output.is_file()
            return
        atomic_json(
            output,
            {
                "status": campaign_module.FULL_COMPARISON_STATUS,
                "comparison_fingerprint": "e" * 64,
            },
        )

    monkeypatch.setattr(campaign_module, "_run_full_comparator", fake_comparator)
    first_pair = {runtime["arms"][0]["arm_id"], runtime["arms"][1]["arm_id"]}
    records = campaign_module.ensure_full_comparisons(
        runtime,
        root,
        completed_arm_ids=first_pair,
        require_complete=False,
    )
    assert len(records) == 1
    assert records[0]["validation"] == "CREATED_AND_VERIFIED_THIS_PASS"
    assert calls == [(runtime["arms"][1]["arm_id"], False)]

    calls.clear()
    records = campaign_module.ensure_full_comparisons(
        runtime,
        root,
        completed_arm_ids={arm["arm_id"] for arm in runtime["arms"]},
        require_complete=False,
    )
    assert len(records) == 39
    assert len(calls) == 38
    assert all(verify is False for _, verify in calls)

    calls.clear()
    records = campaign_module.ensure_full_comparisons(
        runtime,
        root,
        completed_arm_ids={arm["arm_id"] for arm in runtime["arms"]},
        require_complete=True,
    )
    assert len(records) == 39
    assert len(calls) == 39
    assert all(verify is True for _, verify in calls)
    assert all(record["validation"] == "REVERIFIED_THIS_PASS" for record in records)


def _fake_model_summary(*, adaptive: bool, final_loss: float) -> dict[str, object]:
    group = {
        "near_threshold_proxy": {"quantiles": {"0.5": 0.7}},
        "near_zero_proxy": {"quantiles": {"0.5": 0.2}},
        "near_zero_adjusted": {"quantiles": {"0.5": 0.02}},
        "dynamic_target_unclipped": {"quantiles": {"0.5": 0.6}},
        "dynamic_target_clipped": {"quantiles": {"0.5": 0.4}},
        "controller_error": {"quantiles": {"0.5": -0.1}},
        "near_threshold_proxy_error": {"quantiles": {"0.5": 0.03}},
        "near_zero_proxy_error": {"quantiles": {"0.5": -0.04}},
        "raw_log_step": {"quantiles": {"0.5": -0.02}},
        "clip_threshold_used": {"quantiles": {"0.5": 9.0}},
        "next_clip_threshold": {"quantiles": {"0.5": 8.0}},
        "final_next_clip_threshold": 7.0,
        "gamma_clamped_low_count": 1,
        "gamma_clamped_high_count": 2,
        "log_step_bounded_count": 3,
        "lower_bound_hits": 4,
        "upper_bound_hits": 5,
        "noisy_adjacent_monotonicity_violations": 6,
        "exact_adjacent_monotonicity_violations": 0,
    }
    behavior_group = {
        "actual_clipped_fraction": 0.1,
        "would_clip_fraction": 0.1,
        **{
            name: {"quantiles": {"0.5": float(index)}}
            for index, name in enumerate(
                (
                    "raw_gradient_l2",
                    "clipped_signal_gradient_l2",
                    "noise_gradient_l2",
                    "signal_to_noise_l2_ratio",
                    "signal_noise_cosine",
                    "global_update_l2",
                ),
                start=1,
            )
        },
    }
    return {
        "evaluations": [{"round": 0, "loss": 4.0}, {"round": 50, "loss": final_loss}],
        "clipping": {"any_group": {"fraction": 0.1, "would_fraction": 0.1}},
        "behavior_summary": {"groups": {"A": behavior_group, "B": behavior_group}},
        "elapsed_seconds": 12.0,
        "slaclip": {"controller_summary": {"groups": {"A": group, "B": group}}}
        if adaptive
        else None,
    }


def test_incremental_aggregator_writes_paired_machine_readable_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_manifest = tmp_path / "input-manifest.json"
    input_manifest.write_text('{"pinned":true}\n', encoding="utf-8")
    runtime = build_runtime_manifest(
        SPEC,
        repository_sha="b" * 40,
        input_manifest=input_manifest,
        created_at_utc="2026-07-29T00:00:00+00:00",
    )
    runtime_path = tmp_path / "campaign" / "runtime-manifest.json"
    atomic_json(runtime_path, runtime)
    for arm in runtime["arms"][:2]:
        adaptive = arm["method"] == FULL_SLACLIP_METHOD
        summary = {
            "status": "COMPLETED",
            "method": arm["method"],
            "models": {
                "bert": _fake_model_summary(adaptive=adaptive, final_loss=3.0 if adaptive else 3.2),
                "gpt2": _fake_model_summary(adaptive=adaptive, final_loss=3.5 if adaptive else 3.7),
            },
        }
        atomic_json(tmp_path / "campaign" / "arms" / arm["arm_id"] / "final_summary.json", summary)
        atomic_json(
            tmp_path / "campaign" / "arm-status" / f"{arm['arm_id']}.json",
            {"status": "COMPLETED"},
        )

    monkeypatch.setattr(
        campaign_module,
        "ensure_full_comparisons",
        lambda *_args, **_kwargs: [
            {
                "arm_id": runtime["arms"][1]["arm_id"],
                "path": "comparison.json",
                "sha256": "f" * 64,
                "comparison_fingerprint": "e" * 64,
                "validation": "CREATED_AND_VERIFIED_THIS_PASS",
            }
        ],
    )
    complete = aggregate_campaign(
        argparse.Namespace(manifest=runtime_path, require_complete=False)
    )
    assert complete is False
    campaign_summary = json.loads(
        (tmp_path / "campaign" / "campaign_summary.json").read_text(encoding="utf-8")
    )
    assert campaign_summary["completed_arm_count"] == 2
    assert campaign_summary["metric_row_count"] == 4
    assert campaign_summary["paired_metric_row_count"] == 2
    assert campaign_summary["paired_aggregate_row_count"] == 2
    assert campaign_summary["comparison_evidence_count"] == 1
    assert campaign_summary["comparisons_verified_this_pass"] == 1
    with (tmp_path / "campaign" / "paired_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 2
    assert {
        round(float(row["final_loss_difference_slaclip_minus_fixed"]), 12)
        for row in paired
    } == {-0.2}
    with (tmp_path / "campaign" / "campaign_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metrics = list(csv.DictReader(handle))
    adaptive_rows = [row for row in metrics if row["method"] == FULL_SLACLIP_METHOD]
    assert {float(row["cdf_near_threshold_median_A"]) for row in adaptive_rows} == {
        0.7
    }
    assert {float(row["raw_gradient_l2_median_A"]) for row in adaptive_rows} == {
        1.0
    }
    assert {
        int(row["noisy_adjacent_monotonicity_violations_A"])
        for row in adaptive_rows
    } == {6}
    with (tmp_path / "campaign" / "paired_aggregate_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        paired_aggregates = list(csv.DictReader(handle))
    assert len(paired_aggregates) == 2
    assert {
        round(
            float(row["final_loss_difference_slaclip_minus_fixed_mean"]),
            12,
        )
        for row in paired_aggregates
    } == {-0.2}
