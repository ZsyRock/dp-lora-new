from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import pytest

import paper_repro.full_slaclip_campaign as campaign_module
from paper_repro.full_slaclip_campaign import (
    FULL_SLACLIP_METHOD,
    ORACLE_SLACLIP_METHOD,
    _arm_command,
    _model_metrics,
    aggregate_campaign,
    atomic_json,
    build_runtime_manifest,
    expand_spec,
    load_spec,
    paired_inference,
    resolve_oracle_noisy_arm_pairs,
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
EXIT_POLICY = REPOSITORY / "hpc" / "full_slaclip_exit_policy.sh"
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
    assert 'parent_lanes="${SLURM_NTASKS:-0}"' in worker
    assert '[[ "$parent_lanes" -eq 2 ]]' in worker
    assert "starting_sequential_wave=" in worker


def test_shell_exit_policy_makes_hard_failures_dominate_checkpoint_stops() -> None:
    cases = {
        (0,): "SUCCESS",
        (0, 0): "SUCCESS",
        (75,): "CHECKPOINTED_STOP",
        (0, 75): "CHECKPOINTED_STOP",
        (1,): "HARD_FAILURE",
        (1, 75): "HARD_FAILURE",
        (75, 1): "HARD_FAILURE",
    }
    for return_codes, expected in cases.items():
        completed = subprocess.run(
            [str(EXIT_POLICY), *[str(value) for value in return_codes]],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip() == expected


def test_shell_wait_policy_retries_a_trap_interrupted_wait() -> None:
    script = r'''
set -u
source "$1"
full_slaclip_stop_generation=0
trap 'full_slaclip_stop_generation=$((full_slaclip_stop_generation + 1))' USR1
(sleep 0.05; kill -USR1 "$$") &
notifier_pid=$!
(sleep 0.20; exit 75) &
child_pid=$!
set +e
wait_for_full_slaclip_child "$child_pid"
child_rc=$?
wait "$notifier_pid"
notifier_rc=$?
set -e
printf 'child_rc=%s notifier_rc=%s generation=%s\n' \
    "$child_rc" "$notifier_rc" "$full_slaclip_stop_generation"
[[ "$child_rc" -eq 75 && "$notifier_rc" -eq 0 && "$full_slaclip_stop_generation" -eq 1 ]]
'''
    completed = subprocess.run(
        ["bash", "-c", script, "bash", str(EXIT_POLICY)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "child_rc=75 notifier_rc=0 generation=1"


def test_slurm_worker_uses_signal_safe_wait_for_every_background_child() -> None:
    worker = SLURM_WORKER.read_text(encoding="utf-8")
    assert '\nwait "$' not in worker
    assert worker.count("wait_for_full_slaclip_child") == 9
    assert "--method oracle_slaclip_control" in worker
    assert "NON-DP" in worker


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


def test_oracle_control_matrix_uses_exact_endpoints_and_primary_references(
    tmp_path: Path,
) -> None:
    spec = copy.deepcopy(load_spec(SPEC))
    spec["expected_arm_count"] = 110
    spec["oracle_controls"] = {
        "initial_clip_norm": 10.0,
        "seeds": [42, 43],
        "etas": [0.05],
        "betas": [0.76],
        "method": ORACLE_SLACLIP_METHOD,
    }
    spec_path = tmp_path / "oracle-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    arms = expand_spec(load_spec(spec_path))
    oracle = [arm for arm in arms if arm["method"] == ORACLE_SLACLIP_METHOD]
    assert len(oracle) == 2
    assert {arm["family"] for arm in oracle} == {"oracle_control"}
    assert {arm["analysis_role"] for arm in oracle} == {
        "non_dp_exact_endpoint_oracle_controller_diagnostic"
    }
    assert {arm["controller_input"] for arm in oracle} == {"exact_endpoints"}
    assert {arm["slaclip_eta"] for arm in oracle} == {0.05}
    assert {arm["slaclip_beta"] for arm in oracle} == {0.76}
    assert {arm["noise_multiplier"] for arm in oracle} == {2.0}
    assert {arm["slaclip_num_slots"] for arm in oracle} == {15}
    by_id = {arm["arm_id"]: arm for arm in arms}
    for arm in oracle:
        reference = by_id[arm["reference_arm_id"]]
        assert reference["method"] == "paper_dp_lora"
        assert reference["seed"] == arm["seed"]
        assert reference["initial_clip_norm"] == arm["initial_clip_norm"]
        command = _arm_command(
            arm,
            repository=tmp_path / "repo",
            python_bin=tmp_path / "env" / "bin" / "python",
            input_manifest=tmp_path / "input.json",
            output_dir=tmp_path / arm["arm_id"],
            private_key=tmp_path / "key",
            stop_file=tmp_path / "stop",
        )
        assert command[command.index("--method") + 1] == ORACLE_SLACLIP_METHOD
        assert "--slaclip-num-slots" in command


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


def test_k5_range_submitter_uses_one_l4_lane_and_right_sized_resources() -> None:
    submitter = K5_RANGE_SUBMITTER.read_text(encoding="utf-8")
    assert (
        "DPLORA_FULL_SPEC_RELATIVE=hpc/full-slaclip-k5-baseline-range-spec.json"
        in submitter
    )
    assert "DPLORA_FULL_GPU_GRES=gpu:l4swarm:1" in submitter
    assert "DPLORA_FULL_PARTITION=scavenger_l4" in submitter
    assert "DPLORA_FULL_HOST_MEMORY=12G" in submitter
    assert "DPLORA_FULL_LANE_MEMORY=12G" in submitter
    assert "DPLORA_FULL_WALLTIME=02:00:00" in submitter
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


def test_groupwise_full_controller_arm_is_validated_and_uses_canonical_ab_cli(
    tmp_path: Path,
) -> None:
    adaptive = copy.deepcopy(
        next(
            arm
            for arm in expand_spec(load_spec(SPEC))
            if arm["method"] == FULL_SLACLIP_METHOD
        )
    )
    adaptive["slaclip_base_target_clipped_fraction"] = None
    adaptive["slaclip_beta"] = None
    adaptive["slaclip_base_target_clipped_fraction_by_group"] = {
        "A": 0.125,
        "B": 0.875,
    }
    adaptive["slaclip_beta_by_group"] = {"A": 0.125, "B": 0.875}
    adaptive["slaclip_baseline_calibration_lock_sha256"] = "c" * 64
    adaptive["acknowledge_slaclip_baseline_calibration_is_non_dp"] = True
    campaign_module._validate_runtime_arm(adaptive, index=adaptive["index"])

    command = _arm_command(
        adaptive,
        repository=tmp_path / "repo",
        python_bin=tmp_path / "env" / "bin" / "python",
        input_manifest=tmp_path / "input.json",
        output_dir=tmp_path / "adaptive",
        private_key=tmp_path / "key",
        stop_file=tmp_path / "stop",
    )
    assert "--slaclip-base-target-clipped-fraction" not in command
    assert command[command.index("--slaclip-base-target-clipped-fraction-a") + 1] == "0.125"
    assert command[command.index("--slaclip-base-target-clipped-fraction-b") + 1] == "0.875"
    assert "--slaclip-beta" not in command
    assert command[
        command.index("--slaclip-baseline-calibration-lock-sha256") + 1
    ] == "c" * 64
    assert "--acknowledge-slaclip-baseline-calibration-is-non-dp" in command

    invalid = copy.deepcopy(adaptive)
    invalid["learning_rate"] = "0.0005"
    with pytest.raises(RuntimeError, match="learning_rate must be numeric"):
        campaign_module._validate_runtime_arm(invalid, index=invalid["index"])
    invalid_ack = copy.deepcopy(adaptive)
    invalid_ack["acknowledge_slaclip_baseline_calibration_is_non_dp"] = 1
    with pytest.raises(RuntimeError, match="not acknowledged"):
        campaign_module._validate_runtime_arm(
            invalid_ack, index=invalid_ack["index"]
        )


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
        {
            "round": 0, "loss": 4.0, "token_accuracy": 0.2,
            "supervised_tokens": 100, "correct_tokens": 20,
            "token_accuracy_definition": "supervised_token_top1_micro_accuracy",
        },
        {
            "round": 10, "loss": 2.0, "token_accuracy": 0.5,
            "supervised_tokens": 100, "correct_tokens": 50,
            "token_accuracy_definition": "supervised_token_top1_micro_accuracy",
        },
        {
            "round": 50, "loss": 3.0, "token_accuracy": 0.4,
            "supervised_tokens": 100, "correct_tokens": 40,
            "token_accuracy_definition": "supervised_token_top1_micro_accuracy",
        },
    ]
    metrics = _model_metrics(arm, "bert", summary)
    assert metrics["best_loss"] == 2.0
    assert metrics["best_round"] == 10
    assert metrics["final_minus_best"] == 1.0
    assert math.isclose(metrics["normalized_loss_auc"], 2.6)
    assert math.isclose(metrics["loss_total_variation"], 3.0)
    assert math.isclose(metrics["loss_excess_total_variation"], 2.0)
    assert math.isclose(metrics["initial_token_accuracy"], 0.2)
    assert math.isclose(metrics["final_token_accuracy"], 0.4)
    assert math.isclose(metrics["best_token_accuracy"], 0.5)
    assert math.isclose(metrics["normalized_token_accuracy_auc"], 0.43)
    assert math.isclose(metrics["token_accuracy_total_variation"], 0.4)
    assert metrics["final_supervised_tokens"] == 100
    assert metrics["final_correct_tokens"] == 40
    assert (
        metrics["token_accuracy_definition"]
        == "supervised_token_top1_micro_accuracy"
    )
    for name in (
        "loss_total_variation",
        "loss_excess_total_variation",
        "final_token_accuracy",
        "normalized_token_accuracy_auc",
        "final_supervised_tokens",
        "final_correct_tokens",
    ):
        assert name in campaign_module.BASE_METRIC_COLUMNS
        assert name in campaign_module.AGGREGATED_METRICS


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
        "log_threshold_total_variation": 0.8,
    }
    behavior_group = {
        "actual_clipped_fraction": 0.1,
        "would_clip_fraction": 0.1,
        "fully_clipped_round_count": 0,
        "fully_clipped_round_fraction": 0.0,
        **{
            name: {"quantiles": {"0.5": float(index)}}
            for index, name in enumerate(
                (
                    "raw_gradient_l2",
                    "raw_to_threshold_ratio",
                    "removed_gradient_l2",
                    "retained_energy_fraction",
                    "clipped_signal_gradient_l2",
                    "noise_gradient_l2",
                    "signal_to_noise_l2_ratio",
                    "signal_noise_cosine",
                    "global_update_l2",
                    "aggregate_signal_gradient_l2",
                    "aggregate_noise_gradient_l2",
                    "aggregate_signal_to_noise_l2_ratio",
                    "relative_global_update",
                ),
                start=1,
            )
        },
    }
    return {
        "evaluations": [
            {
                "round": 0,
                "loss": 4.0,
                "supervised_tokens": 100,
                "correct_tokens": 20,
                "token_accuracy": 0.2,
                "token_accuracy_definition": (
                    "supervised_token_top1_micro_accuracy"
                ),
            },
            {
                "round": 50,
                "loss": final_loss,
                "supervised_tokens": 100,
                "correct_tokens": 40,
                "token_accuracy": 0.4,
                "token_accuracy_definition": (
                    "supervised_token_top1_micro_accuracy"
                ),
            },
        ],
        "clipping": {"any_group": {"fraction": 0.1, "would_fraction": 0.1}},
        "behavior_summary": {
            "sample_schedule_sha256": "a" * 64,
            "supervision_schedule_sha256": "b" * 64,
            "groups": {"A": behavior_group, "B": behavior_group},
        },
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


def test_oracle_and_non_dp_controls_use_separate_descriptive_pairing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = copy.deepcopy(load_spec(K5_RANGE_SPEC))
    spec["expected_arm_count"] = 8
    spec["primary"]["seeds"] = [52, 53]
    spec["sensitivity"]["seeds"] = []
    spec["controls"]["seeds"] = [52]
    spec["oracle_controls"] = {
        "initial_clip_norm": 10.0,
        "seeds": [52, 53],
        "etas": [0.2],
        "betas": [0.38],
        "method": ORACLE_SLACLIP_METHOD,
    }
    spec_path = tmp_path / "diagnostic-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    input_manifest = tmp_path / "input-manifest.json"
    input_manifest.write_text('{"pinned":true}\n', encoding="utf-8")
    runtime = build_runtime_manifest(
        spec_path,
        repository_sha="c" * 40,
        input_manifest=input_manifest,
        created_at_utc="2026-08-04T00:00:00+00:00",
    )
    runtime_path = tmp_path / "campaign" / "runtime-manifest.json"
    atomic_json(runtime_path, runtime)
    completed_methods = {
        "paper_dp_lora",
        "no_dp_lora_control",
        "clip_only_control",
        FULL_SLACLIP_METHOD,
        ORACLE_SLACLIP_METHOD,
    }
    losses = {
        "paper_dp_lora": 3.4,
        "no_dp_lora_control": 2.9,
        "clip_only_control": 3.2,
        FULL_SLACLIP_METHOD: 3.1,
        ORACLE_SLACLIP_METHOD: 3.0,
    }
    for arm in runtime["arms"]:
        if arm["method"] not in completed_methods:
            continue
        adaptive = arm["method"] in {
            FULL_SLACLIP_METHOD,
            ORACLE_SLACLIP_METHOD,
        }
        summary = {
            "status": "COMPLETED",
            "method": arm["method"],
            "models": {
                model: _fake_model_summary(
                    adaptive=adaptive,
                    final_loss=losses[arm["method"]],
                )
                for model in ("bert", "gpt2")
            },
        }
        atomic_json(
            tmp_path / "campaign" / "arms" / arm["arm_id"] / "final_summary.json",
            summary,
        )
        atomic_json(
            tmp_path / "campaign" / "arm-status" / f"{arm['arm_id']}.json",
            {"status": "COMPLETED"},
        )
    monkeypatch.setattr(
        campaign_module,
        "ensure_full_comparisons",
        lambda *_args, **_kwargs: [],
    )
    assert aggregate_campaign(
        argparse.Namespace(manifest=runtime_path, require_complete=False)
    ) is False

    with (tmp_path / "campaign" / "campaign_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metrics = list(csv.DictReader(handle))
    control_rows = [
        row for row in metrics if row["method"] in {"no_dp_lora_control", "clip_only_control"}
    ]
    oracle_rows = [
        row for row in metrics if row["method"] == ORACLE_SLACLIP_METHOD
    ]
    assert {float(row["noise_multiplier"]) for row in control_rows} == {2.0}
    assert {
        float(row["effective_gradient_noise_multiplier"]) for row in control_rows
    } == {0.0}
    assert {float(row["noise_multiplier"]) for row in oracle_rows} == {2.0}
    assert {
        float(row["effective_gradient_noise_multiplier"]) for row in oracle_rows
    } == {2.0}
    assert {row["controller_input"] for row in oracle_rows} == {
        "exact_endpoints"
    }

    with (tmp_path / "campaign" / "paired_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        efficacy_pairs = list(csv.DictReader(handle))
    assert len(efficacy_pairs) == 4
    assert {row["method"] for row in efficacy_pairs} == {FULL_SLACLIP_METHOD}
    with (tmp_path / "campaign" / "diagnostic_paired_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        diagnostic = list(csv.DictReader(handle))
    assert len(diagnostic) == 8
    assert {row["method"] for row in diagnostic} == {
        "no_dp_lora_control",
        "clip_only_control",
        ORACLE_SLACLIP_METHOD,
    }
    assert {row["comparison_role"] for row in diagnostic} == {
        "NON_DP_EXACT_ENDPOINT_ORACLE_DIAGNOSTIC",
        "NON_DP_MECHANISM_CONTROL_EXPLORATORY",
    }
    assert all(
        float(row["final_loss_difference_candidate_minus_fixed"]) < 0.0
        for row in diagnostic
    )
    with (
        tmp_path / "campaign" / "diagnostic_paired_aggregate_metrics.csv"
    ).open(encoding="utf-8", newline="") as handle:
        diagnostic_aggregate = list(csv.DictReader(handle))
    assert len(diagnostic_aggregate) == 6
    assert all(
        "holm" not in key.lower()
        for row in diagnostic_aggregate
        for key in row
    )
    campaign_summary = json.loads(
        (tmp_path / "campaign" / "campaign_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert campaign_summary["diagnostic_paired_metric_row_count"] == 8
    assert campaign_summary["diagnostic_paired_aggregate_row_count"] == 6
    assert "excluded from SlaClip efficacy selection" in campaign_summary[
        "diagnostic_pairing_policy"
    ]

    with (tmp_path / "campaign" / "oracle_vs_noisy_paired_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        oracle_vs_noisy = list(csv.DictReader(handle))
    assert len(oracle_vs_noisy) == 4
    assert {row["candidate_method"] for row in oracle_vs_noisy} == {
        ORACLE_SLACLIP_METHOD
    }
    assert {row["noisy_method"] for row in oracle_vs_noisy} == {
        FULL_SLACLIP_METHOD
    }
    assert {
        round(float(row["final_loss_difference_candidate_minus_noisy"]), 12)
        for row in oracle_vs_noisy
    } == {-0.1}
    assert {float(row["candidate_final_threshold_A"]) for row in oracle_vs_noisy} == {
        7.0
    }
    assert {
        float(row["candidate_retained_energy_fraction_median_A"])
        for row in oracle_vs_noisy
    } == {4.0}
    assert {
        float(row["candidate_aggregate_signal_to_noise_l2_ratio_median_B"])
        for row in oracle_vs_noisy
    } == {12.0}
    with (
        tmp_path / "campaign" / "oracle_vs_noisy_paired_aggregate_metrics.csv"
    ).open(encoding="utf-8", newline="") as handle:
        oracle_vs_noisy_aggregate = list(csv.DictReader(handle))
    assert len(oracle_vs_noisy_aggregate) == 2
    assert all(
        "holm" not in key.lower() and "p_value" not in key.lower()
        for row in oracle_vs_noisy_aggregate
        for key in row
    )
    assert campaign_summary[
        "expected_oracle_vs_noisy_paired_metric_row_count"
    ] == 4
    assert campaign_summary["oracle_vs_noisy_paired_metric_row_count"] == 4
    assert campaign_summary["oracle_vs_noisy_paired_aggregate_row_count"] == 2
    assert "excluded from efficacy claims" in campaign_summary[
        "oracle_vs_noisy_pairing_policy"
    ]
    assert campaign_summary["development_beta_selection"] is None


def test_oracle_noisy_pair_resolution_is_exact_and_fails_closed() -> None:
    assert resolve_oracle_noisy_arm_pairs(
        {"arms": expand_spec(load_spec(K5_RANGE_SPEC))}
    ) == []

    spec = copy.deepcopy(load_spec(K5_RANGE_SPEC))
    spec["expected_arm_count"] = 6
    spec["primary"]["seeds"] = [52, 53]
    spec["sensitivity"]["seeds"] = []
    spec["controls"]["seeds"] = []
    spec["oracle_controls"] = {
        "initial_clip_norm": 10.0,
        "seeds": [52, 53],
        "etas": [0.2],
        "betas": [0.38],
        "method": ORACLE_SLACLIP_METHOD,
    }
    arms = expand_spec(spec)
    simple_runtime = {"arms": arms}
    pairs = resolve_oracle_noisy_arm_pairs(simple_runtime)
    assert len(pairs) == 2
    assert pairs[0][0]["method"] == ORACLE_SLACLIP_METHOD
    assert pairs[0][1]["method"] == FULL_SLACLIP_METHOD

    provenance_bound = copy.deepcopy(simple_runtime)
    for arm in provenance_bound["arms"]:
        if arm["method"] in {FULL_SLACLIP_METHOD, ORACLE_SLACLIP_METHOD}:
            arm["slaclip_baseline_calibration_lock_sha256"] = "a" * 64
            arm["acknowledge_slaclip_baseline_calibration_is_non_dp"] = True
            arm["slaclip_calibration_provenance"] = (
                "exact_NON_DP_fixed_trajectory_diagnostics_frozen_before_fresh_seeds"
            )
    assert len(resolve_oracle_noisy_arm_pairs(provenance_bound)) == 2
    mismatched_provenance = copy.deepcopy(provenance_bound)
    next(
        arm
        for arm in mismatched_provenance["arms"]
        if arm["method"] == ORACLE_SLACLIP_METHOD
    )["slaclip_baseline_calibration_lock_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="must be unique"):
        resolve_oracle_noisy_arm_pairs(mismatched_provenance)

    missing = copy.deepcopy(simple_runtime)
    oracle = next(
        arm for arm in missing["arms"] if arm["method"] == ORACLE_SLACLIP_METHOD
    )
    oracle["slaclip_eta"] = 0.1
    try:
        resolve_oracle_noisy_arm_pairs(missing)
    except RuntimeError as error:
        assert "must be unique" in str(error)
    else:
        raise AssertionError("missing noisy oracle counterpart was accepted")

    ambiguous = copy.deepcopy(simple_runtime)
    noisy = next(
        arm for arm in ambiguous["arms"] if arm["method"] == FULL_SLACLIP_METHOD
    )
    duplicate = copy.deepcopy(noisy)
    duplicate["arm_id"] = "duplicate-noisy-arm"
    ambiguous["arms"].append(duplicate)
    try:
        resolve_oracle_noisy_arm_pairs(ambiguous)
    except RuntimeError as error:
        assert "must be unique" in str(error)
    else:
        raise AssertionError("ambiguous noisy oracle counterpart was accepted")
