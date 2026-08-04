from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import paper_repro.full_slaclip_campaign as campaign_module
from paper_repro.full_slaclip_campaign import (
    FULL_SLACLIP_METHOD,
    _arm_command,
    _model_metrics,
    aggregate_campaign,
    atomic_json,
    build_runtime_manifest,
    expand_spec,
    load_spec,
    paired_inference,
    validated_step_environment,
    validate_or_create_key,
    validate_runtime_manifest,
    write_development_beta_selection,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = REPOSITORY / "hpc" / "full-slaclip-campaign-spec.json"
BETA5_SPEC = REPOSITORY / "hpc" / "full-slaclip-beta5-screen-spec.json"
K5_RANGE_SPEC = REPOSITORY / "hpc" / "full-slaclip-k5-baseline-range-spec.json"
SLURM_WORKER = REPOSITORY / "hpc" / "full_slaclip_campaign.sbatch"
BETA5_SUBMITTER = REPOSITORY / "hpc" / "submit_full_slaclip_beta5_screen.sh"
K5_RANGE_SUBMITTER = (
    REPOSITORY / "hpc" / "submit_full_slaclip_k5_baseline_range.sh"
)


def test_slurm_worker_exports_required_cuda_determinism_contract() -> None:
    worker = SLURM_WORKER.read_text(encoding="utf-8")
    required_export = "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
    assert worker.count(required_export) == 1
    assert worker.index(required_export) < worker.index("run_step()")
    assert worker.count("export SLURM_EXPORT_ENV=ALL") == 1
    run_step = worker[worker.index("run_step()") : worker.index('echo "job_id=')]
    assert run_step.count("--export=ALL") == 1
    assert run_step.count("< /dev/null") == 1
    assert "consume future wave rows" in run_step


def test_compute_step_environment_is_validated_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    values = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "SLURM_EXPORT_ENV": "ALL",
        "HF_HOME": str(tmp_path / "hf"),
        "TMPDIR": str(tmp_path / "tmp"),
        "OMP_NUM_THREADS": "12",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    assert validated_step_environment()["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    try:
        validated_step_environment()
    except RuntimeError as error:
        assert "CUBLAS_WORKSPACE_CONFIG" in str(error)
    else:
        raise AssertionError("missing cuBLAS contract was accepted")


def test_checked_in_matrix_is_exact_and_contains_no_slaclip_q() -> None:
    arms = expand_spec(load_spec(SPEC))
    assert len(arms) == 108
    assert Counter(arm["family"] for arm in arms) == {
        "primary": 20,
        "threshold_robustness": 24,
        "sensitivity": 24,
        "noise_sensitivity": 18,
        "client_sensitivity": 12,
        "control": 10,
    }
    assert Counter(arm["method"] for arm in arms) == {
        "paper_dp_lora": 37,
        "slaclip_dp_lora": 61,
        "no_dp_lora_control": 5,
        "clip_only_control": 5,
    }
    assert all("slaclip_q" not in arm["method"].lower() for arm in arms)
    assert [(arm["index"], arm["wave"], arm["lane"]) for arm in arms] == [
        (index, index // 2, index % 2) for index in range(108)
    ]

    primary_full = [
        arm
        for arm in arms
        if arm["family"] == "primary" and arm["method"] == FULL_SLACLIP_METHOD
    ]
    assert {arm["initial_clip_norm"] for arm in primary_full} == {10.0}
    assert {arm["seed"] for arm in primary_full} == set(range(42, 52))
    assert {arm["slaclip_eta"] for arm in primary_full} == {0.2}
    assert {arm["slaclip_beta"] for arm in primary_full} == {0.5}

    paper_style = [
        arm
        for arm in arms
        if arm["analysis_role"] == "paper_setting_confirmatory_seed_replication"
    ]
    assert len(paper_style) == 20
    assert all(arm["initial_clip_norm"] == 10.0 for arm in paper_style)

    robustness = [arm for arm in arms if arm["family"] == "threshold_robustness"]
    assert len(robustness) == 24
    assert {arm["initial_clip_norm"] for arm in robustness} == {
        0.1,
        1.0,
        5.0,
        20.0,
    }
    assert {arm["seed"] for arm in robustness} == {42, 43, 44}

    sensitivity = [arm for arm in arms if arm["family"] == "sensitivity"]
    assert len(sensitivity) == 24
    assert not any(
        arm["slaclip_eta"] == 0.2 and arm["slaclip_beta"] == 0.5
        for arm in sensitivity
    )
    assert all(arm["reference_arm_id"] for arm in sensitivity)

    noise = [arm for arm in arms if arm["family"] == "noise_sensitivity"]
    assert len(noise) == 18
    assert {arm["noise_multiplier"] for arm in noise} == {0.5, 1.0, 4.0}
    assert {arm["seed"] for arm in noise} == {42, 43, 44}

    clients = [arm for arm in arms if arm["family"] == "client_sensitivity"]
    assert len(clients) == 12
    assert {arm["num_clients"] for arm in clients} == {20, 80}
    assert {arm["seed"] for arm in clients} == {42, 43, 44}

    controls = [arm for arm in arms if arm["family"] == "control"]
    assert len(controls) == 10
    assert {arm["seed"] for arm in controls} == {42, 43, 44, 45, 46}

    adaptive = [arm for arm in arms if arm["method"] == FULL_SLACLIP_METHOD]
    assert len(adaptive) == 61
    assert all(arm["batch_size"] < 128 for arm in adaptive)
    assert all(arm["slaclip_num_slots"] == 15 for arm in adaptive)
    assert all(arm["slaclip_c_min"] == 0.1 for arm in adaptive)
    assert all(arm["slaclip_c_max"] == 50.0 for arm in adaptive)
    assert all(arm["data_split_seed"] == 1729 for arm in arms)
    assert all(arm["evaluation_seed"] == 2718 for arm in arms)


def test_beta5_screen_is_full_slaclip_only_and_has_five_base_targets() -> None:
    spec = load_spec(BETA5_SPEC)
    arms = expand_spec(spec)
    assert len(arms) == 30
    assert Counter(arm["family"] for arm in arms) == {
        "primary": 10,
        "sensitivity": 20,
    }
    assert Counter(arm["method"] for arm in arms) == {
        "paper_dp_lora": 5,
        "slaclip_dp_lora": 25,
    }
    assert {arm["seed"] for arm in arms} == {52, 53, 54, 55, 56}
    adaptive = [arm for arm in arms if arm["method"] == FULL_SLACLIP_METHOD]
    candidates = {0.01, 0.03, 0.066, 0.146, 0.5}
    assert {arm["slaclip_base_target_clipped_fraction"] for arm in adaptive} == candidates
    assert {arm["slaclip_beta"] for arm in adaptive} == candidates
    assert all(
        arm["slaclip_base_target_clipped_fraction"] == arm["slaclip_beta"]
        for arm in adaptive
    )
    assert all(arm["slaclip_eta"] == 0.2 for arm in adaptive)
    assert all(arm["initial_clip_norm"] == 10.0 for arm in arms)
    assert all(arm["slaclip_num_slots"] == 15 for arm in adaptive)
    assert all("slaclip_q" not in arm["method"].lower() for arm in arms)
    boundary = spec["scientific_boundary"]
    assert boundary["analysis_role"] == (
        "development_hyperparameter_screen_not_confirmatory_test"
    )
    assert boundary["paper_benchmarks_evaluated"] is False
    assert "SlaClip-Q" in boundary["excluded_method_family"]
    assert [(arm["index"], arm["wave"], arm["lane"]) for arm in arms] == [
        (index, index // 2, index % 2) for index in range(30)
    ]


def test_beta5_submitter_uses_two_l4_lanes_and_development_spec() -> None:
    submitter = BETA5_SUBMITTER.read_text(encoding="utf-8")
    assert "DPLORA_FULL_SPEC_RELATIVE=hpc/full-slaclip-beta5-screen-spec.json" in submitter
    assert "DPLORA_FULL_GPU_GRES=gpu:l4:2" in submitter
    assert "DPLORA_FULL_PARTITION=l4" in submitter
    assert "DPLORA_FULL_EXPECTED_GPU=L4" in submitter
    assert "DPLORA_FULL_WALLTIME=1-12:00:00" in submitter
    assert "slaclip_q" not in submitter.lower()


def test_k5_baseline_range_screen_is_full_slaclip_only_and_pre_registered() -> None:
    spec = load_spec(K5_RANGE_SPEC)
    arms = expand_spec(spec)
    assert len(arms) == 30
    assert Counter(arm["family"] for arm in arms) == {
        "primary": 10,
        "sensitivity": 20,
    }
    assert Counter(arm["method"] for arm in arms) == {
        "paper_dp_lora": 5,
        "slaclip_dp_lora": 25,
    }
    adaptive = [arm for arm in arms if arm["method"] == FULL_SLACLIP_METHOD]
    candidates = {0.0, 0.19, 0.38, 0.57, 0.76}
    assert {
        arm["slaclip_base_target_clipped_fraction"] for arm in adaptive
    } == candidates
    assert {arm["slaclip_beta"] for arm in adaptive} == candidates
    assert all(arm["slaclip_num_slots"] == 5 for arm in adaptive)
    assert all(arm["num_clients"] == 5 for arm in arms)
    assert all(arm["noise_multiplier"] == 2.0 for arm in arms)
    assert all("slaclip_q" not in arm["method"].lower() for arm in arms)

    boundary = spec["scientific_boundary"]
    calibration = boundary["baseline_calibration"]
    assert boundary["candidate_base_target_clipped_fractions"] == [
        0.0,
        0.19,
        0.38,
        0.57,
        0.76,
    ]
    assert "remaining non-small-gradient mass" in boundary["beta_semantics"]
    assert calibration["metric"] == "any_group_clipped_fraction"
    assert calibration["bert_roundwise_mean_min"] == 0.0
    assert calibration["bert_roundwise_mean_max"] == 0.76
    assert len(calibration["source_round_summaries_sha256"]) == 10
    assert all(
        len(value) == 64
        for value in calibration["source_round_summaries_sha256"].values()
    )
    assert calibration["gpt2_degenerate_interval"] is True
    assert calibration["gpt2_grid_role"] == (
        "cross_model_exploration_only_not_within_model_calibration"
    )
    assert boundary["expected_normalized_cdf_endpoint_noise_std"] == 2.0


def test_k5_range_submitter_uses_two_l4_lanes_and_right_sized_resources() -> None:
    submitter = K5_RANGE_SUBMITTER.read_text(encoding="utf-8")
    assert (
        "DPLORA_FULL_SPEC_RELATIVE=hpc/full-slaclip-k5-baseline-range-spec.json"
        in submitter
    )
    assert "DPLORA_FULL_GPU_GRES=gpu:l4:2" in submitter
    assert "DPLORA_FULL_PARTITION=l4" in submitter
    assert "DPLORA_FULL_HOST_MEMORY=24G" in submitter
    assert "DPLORA_FULL_LANE_MEMORY=12G" in submitter
    assert "DPLORA_FULL_WALLTIME=01:00:00" in submitter
    assert "slaclip_q" not in submitter.lower()


def test_beta5_development_selection_is_per_model_and_explicitly_not_test(
    tmp_path: Path,
) -> None:
    candidates = [0.0, 0.03, 0.066, 0.146, 0.5]
    seeds = [52, 53]
    runtime = {
        "campaign_name": "test-beta5",
        "manifest_sha256": "a" * 64,
        "scientific_boundary": {
            "analysis_role": "development_hyperparameter_screen_not_confirmatory_test",
            "candidate_base_target_clipped_fractions": candidates,
            "development_seeds": seeds,
        },
    }
    metric_rows = []
    paired_rows = []
    winners = {"bert": 0.03, "gpt2": 0.146}
    for model, winner in winners.items():
        for beta in candidates:
            for seed in seeds:
                paired_rows.append(
                    {
                        "model": model,
                        "method": FULL_SLACLIP_METHOD,
                        "seed": seed,
                        "slaclip_beta": beta,
                        "final_loss_difference_slaclip_minus_fixed": (
                            abs(beta - winner)
                        ),
                        "normalized_loss_auc_difference_slaclip_minus_fixed": (
                            abs(beta - winner) + 0.01
                        ),
                    }
                )
                metric_rows.append(
                    {
                        "model": model,
                        "method": FULL_SLACLIP_METHOD,
                        "seed": seed,
                        "slaclip_beta": beta,
                        **{
                            f"{name}_{group}": 0
                            for group in ("A", "B")
                            for name in (
                                "gamma_clamped_low_count",
                                "gamma_clamped_high_count",
                                "lower_bound_hits",
                                "upper_bound_hits",
                            )
                        },
                    }
                )
    path = write_development_beta_selection(
        runtime,
        tmp_path,
        metric_rows=metric_rows,
        paired_rows=paired_rows,
    )
    assert path == tmp_path / "development_beta_selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    assert selection["status"] == "DEVELOPMENT_SELECTION_ONLY_NOT_TEST_EVIDENCE"
    assert {
        model: value["selected_base_target_clipped_fraction_for_future_confirmation"]
        for model, value in selection["models"].items()
    } == winners
    assert len(selection["models"]["bert"]["ordered_candidates"]) == 5


def test_paired_arms_share_randomness_and_fixed_evaluation_protocol() -> None:
    arms = expand_spec(load_spec(SPEC))
    by_id = {arm["arm_id"]: arm for arm in arms}

    # The campaign deliberately applies common random numbers across method,
    # initial-C, and sigma variations for each training seed.
    for seed in {arm["seed"] for arm in arms}:
        seeded = [arm for arm in arms if arm["seed"] == seed]
        assert {arm["rng_domain"] for arm in seeded} == {
            f"full-slaclip-cdf:s{seed}"
        }
        assert {arm["data_split_seed"] for arm in seeded} == {1729}
        assert {arm["evaluation_seed"] for arm in seeded} == {2718}

    seed_42 = [arm for arm in arms if arm["seed"] == 42]
    assert len({arm["initial_clip_norm"] for arm in seed_42}) > 1
    assert len({arm["noise_multiplier"] for arm in seed_42}) > 1
    assert {arm["method"] for arm in seed_42} >= {
        "paper_dp_lora",
        FULL_SLACLIP_METHOD,
    }

    for arm in arms:
        reference_id = arm["reference_arm_id"]
        if reference_id is None:
            continue
        reference = by_id[reference_id]
        assert arm["seed"] == reference["seed"]
        assert arm["rng_domain"] == reference["rng_domain"]
        assert arm["data_split_seed"] == reference["data_split_seed"] == 1729
        assert arm["evaluation_seed"] == reference["evaluation_seed"] == 2718
        if arm["method"] == FULL_SLACLIP_METHOD:
            assert arm["slaclip_num_slots"] == 15


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
    assert runtime["expected_arm_count"] == 108
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
    assert "--slaclip-base-target-clipped-fraction" in adaptive_command
    assert "--slaclip-beta" not in adaptive_command
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
    assert "--slaclip-base-target-clipped-fraction" not in fixed_command
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
            "runtime_manifest_sha256": runtime["manifest_sha256"],
            "repository_sha": runtime["repository_sha"],
            "arm_spec_sha256": campaign_module.sha256_bytes(
                campaign_module.canonical_bytes(arm)
            ),
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
    assert len(records) == 61
    assert len(calls) == 60
    assert all(verify is False for _, verify in calls)

    calls.clear()
    records = campaign_module.ensure_full_comparisons(
        runtime,
        root,
        completed_arm_ids={arm["arm_id"] for arm in runtime["arms"]},
        require_complete=True,
    )
    assert len(records) == 61
    assert len(calls) == 61
    assert all(verify is True for _, verify in calls)
    assert all(record["validation"] == "REVERIFIED_THIS_PASS" for record in records)


def test_model_metrics_reports_trapezoid_auc_and_best_checkpoint() -> None:
    arm = expand_spec(load_spec(SPEC))[0]
    summary = _fake_model_summary(adaptive=False, final_loss=3.0)
    summary["evaluations"] = [
        {"round": 0, "loss": 4.0},
        {"round": 10, "loss": 2.0},
        {"round": 50, "loss": 3.0},
    ]
    metrics = _model_metrics(arm, "bert", summary)
    assert metrics["best_loss"] == 2.0
    assert metrics["best_round"] == 10
    assert metrics["final_minus_best"] == 1.0
    assert math.isclose(metrics["normalized_loss_auc"], 2.6)


def test_paired_inference_reports_effect_size_interval_and_exact_sign_flip() -> None:
    result = paired_inference([-0.3, -0.2, -0.1])
    assert result["n"] == 3
    assert math.isclose(float(result["mean"]), -0.2)
    assert math.isclose(float(result["median"]), -0.2)
    assert math.isclose(float(result["sample_std"]), 0.1)
    assert math.isclose(float(result["standard_error"]), 0.1 / math.sqrt(3))
    assert float(result["ci95_low"]) < -0.2 < float(result["ci95_high"])
    assert math.isclose(float(result["cohens_dz"]), -2.0)
    assert result["negative_fraction"] == 1.0
    assert result["zero_fraction"] == 0.0
    assert result["exact_sign_flip_p"] == 0.25

    empty = paired_inference([])
    assert empty["n"] == 0
    assert empty["mean"] is None
    assert empty["exact_sign_flip_p"] is None


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
