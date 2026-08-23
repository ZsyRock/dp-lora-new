from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pytest

from paper_repro import full_slaclip_campaign as full
from paper_repro import identified_slaclip_n128_campaign as campaign
from paper_repro import staged_slaclip_campaign as staged


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "hpc" / "identified-slaclip-n128-spec.json"
WORKER = ROOT / "hpc" / "identified_slaclip_n128.sbatch"
SUBMITTER = ROOT / "hpc" / "submit_identified_slaclip_n128.sh"


def _eligible_rows() -> list[dict[str, float | int | str]]:
    levels = (0.1, 0.3, 0.5, 0.7, 0.8)
    rows = []
    for seed in (1300, 1301):
        for round_index in range(2, 51):
            hard = levels[(round_index - 2) % len(levels)]
            for group in campaign.GROUPS:
                rows.append({
                    "seed": seed, "round": round_index, "group": group,
                    "actual_clipped_fraction": hard,
                    "remaining_non_small_gradient_fraction": 0.8,
                    "stationary_surrogate_target_clipped": hard * 0.5,
                })
    return rows


def _profiles(spec: dict) -> list[dict]:
    result = campaign.assess_fixed_eligibility(_eligible_rows(), [1300, 1301], spec)
    assert result["eligible"] is True
    return result["profiles"]


def _fixed_lock(spec: dict, fixed_arms: list[dict]) -> dict:
    rank_one = [arm for arm in fixed_arms if arm["candidate_rank"] == 1]
    return campaign._lock_payload({
        "strongest_screened_fixed_C_star_by_group": {"A": 0.1, "B": 1.0},
        "strongest_screened_fixed_C_star_seed_evidence": [
            {"seed": arm["seed"], "arm_id": arm["arm_id"]} for arm in rank_one
        ],
        "selected_calibration_C0_by_group": {"A": 0.2, "B": 2.0},
        "calibrated_profiles": _profiles(spec),
    })


def test_strict_spec_mechanism_lock_and_exact_arm_counts() -> None:
    spec = campaign.load_spec(SPEC)
    candidates = [
        {"upstream_rank": index, "C_A": index / 10.0, "C_B": float(index)}
        for index in range(1, 11)
    ]
    fixed = campaign.fixed_screen_arms(spec, "finance", candidates)
    assert len(fixed) == 20
    assert all(arm["method"] == campaign.FIXED_METHOD for arm in fixed)
    lock = _fixed_lock(spec, fixed)
    development = campaign.development_arms(spec, "finance", lock)
    assert len(development) == 10
    assert all(arm["method"] == campaign.SLACLIP_METHOD for arm in development)
    selected = {"selected_profile": lock["calibrated_profiles"][2]}
    confirmation = campaign.confirmation_arms(spec, "finance", lock, selected)
    assert len(confirmation) == 12
    assert sum(arm["method"] == campaign.FIXED_METHOD for arm in confirmation) == 6
    assert sum(arm["method"] == campaign.SLACLIP_METHOD for arm in confirmation) == 6
    assert len(fixed) + len(development) + len(confirmation) == 42
    assert spec["common"]["automatic_num_slots_upper_bound"] == 5
    assert spec["common"]["theoretical_normalized_endpoint_noise_std"] == pytest.approx(
        0.39528470752104744
    )


def test_nested_spec_extra_key_and_rule_tamper_fail_closed(tmp_path: Path) -> None:
    value = json.loads(SPEC.read_text(encoding="utf-8"))
    value["eligibility"]["unregistered_switch"] = True
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="eligibility preregistration keys differ"):
        campaign.load_spec(extra)

    value = json.loads(SPEC.read_text(encoding="utf-8"))
    value["profile_development"]["selection_rule"] = list(reversed(
        value["profile_development"]["selection_rule"]
    ))
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="profile-development preregistration differs"):
        campaign.load_spec(tampered)

    value = json.loads(SPEC.read_text(encoding="utf-8"))
    value["scientific_boundary"]["end_to_end_dp_certified"] = True
    boundary = tmp_path / "boundary.json"
    boundary.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="scientific boundary differs"):
        campaign.load_spec(boundary)


def test_fixed_screen_fails_closed_on_deduplicated_ten_candidates() -> None:
    spec = campaign.load_spec(SPEC)
    candidates = [
        {"upstream_rank": index, "C_A": 1.0, "C_B": 2.0}
        for index in range(1, 11)
    ]
    with pytest.raises(RuntimeError, match="deduplicated"):
        campaign.fixed_screen_arms(spec, "finance", candidates)


def test_per_group_eligibility_and_beta_semantics_are_distinct() -> None:
    spec = campaign.load_spec(SPEC)
    result = campaign.assess_fixed_eligibility(_eligible_rows(), [1300, 1301], spec)
    assert result["eligible"] is True
    assert result["any_group_clipped_fraction_consumed_for_beta"] is False
    assert len(result["profiles"]) == 5
    for group in campaign.GROUPS:
        evidence = result["groups"][group]
        assert evidence["robust_width"] >= 0.2
        assert evidence["distinct_pooled_hard_rate_strata"] >= 5
        assert evidence["fully_clipped_round_fraction"] < 0.2
        for profile in result["profiles"]:
            fit = profile["groups"][group]
            assert fit["beta_identifiable"] is True
            assert fit["box_constraint_feasible"] is True
            assert 0.0 < fit["beta"] < 1.0
            assert fit["desired_hard_clipped_fraction"] == pytest.approx(
                fit["weighted_hard_clipped_fraction"]
            )
            # A hard-rate outcome label is not directly passed as beta.
            assert fit["desired_hard_clipped_fraction"] != pytest.approx(fit["beta"])


def test_eligibility_rejects_fully_clipped_or_no_cross_seed_overlap() -> None:
    spec = campaign.load_spec(SPEC)
    rows = _eligible_rows()
    for row in rows:
        if row["group"] == "B":
            row["actual_clipped_fraction"] = 1.0
    result = campaign.assess_fixed_eligibility(rows, [1300, 1301], spec)
    assert result["eligible"] is False
    assert result["groups"]["B"]["gates"]["fully_clipped_round_fraction_below_0p20"] is False


def test_calibration_trajectory_excludes_gpt2_round_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    requested: list[int] = []

    def fake_load(path: Path, _label: str) -> dict:
        round_index = int(path.stem.split("-")[-1])
        requested.append(round_index)
        records = [
            {
                "gradient_groups": {
                    group: {
                        "raw_norm": 0.5,
                        "clip_threshold": 1.0,
                        "noise_std_per_coordinate": 2.0,
                    }
                    for group in campaign.GROUPS
                }
            }
            for _ in range(2)
        ]
        return {
            "round": round_index, "model": "gpt2",
            "method": campaign.FIXED_METHOD, "client_records": records,
            "round_summary": {
                group: {"clipped_fraction": 0.0} for group in campaign.GROUPS
            },
        }

    monkeypatch.setattr(full, "load_object", fake_load)
    arm = {
        "arm_id": "source", "model": "gpt2", "models": ["gpt2"],
        "method": campaign.FIXED_METHOD, "seed": 1300, "rounds": 50,
        "num_clients": 2, "noise_multiplier": 2.0,
        "initial_clip_norm_by_group": {"A": 1.0, "B": 1.0},
    }
    rows = campaign._fixed_trajectory_rows(
        tmp_path, [arm], slots=5, epsilon=1e-6, round_min=2, round_max=50
    )
    assert len(rows) == 49 * 2
    assert min(requested) == 2 and max(requested) == 50
    assert 1 not in requested


def test_development_hard_target_observation_excludes_round_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested: list[int] = []

    def fake_load(path: Path, _label: str) -> dict:
        round_index = int(path.stem.split("-")[-1])
        requested.append(round_index)
        return {
            "round_summary": {
                "round": round_index,
                "A": {"clipped_fraction": 0.25},
                "B": {"clipped_fraction": 0.75},
            }
        }

    monkeypatch.setattr(full, "load_object", fake_load)
    arm = {"arm_id": "adaptive", "model": "gpt2", "models": ["gpt2"]}
    means = campaign._round_2_to_50_hard_rate_means(tmp_path, arm)
    assert means == {"A": 0.25, "B": 0.75}
    assert requested == list(range(2, 51))


def test_six_seed_gate_uses_exact_magnitude_preserving_sign_flip() -> None:
    spec = campaign.load_spec(SPEC)
    final = full.paired_inference([-0.1] * 6)
    auc = full.paired_inference([-0.01] * 6)
    accuracy = full.paired_inference([0.01] * 6)
    gates = campaign.confirmation_gates(final, auc, accuracy, 6, spec)
    assert final["exact_sign_flip_p"] == pytest.approx(0.03125)
    assert all(gates.values())
    gates_five_wins = campaign.confirmation_gates(final, auc, accuracy, 5, spec)
    assert gates_five_wins["six_of_six_final_loss_wins"] is False
    worse_accuracy = full.paired_inference([-0.01] * 6)
    gates_worse_accuracy = campaign.confirmation_gates(
        final, auc, worse_accuracy, 6, spec
    )
    assert gates_worse_accuracy["mean_final_token_accuracy_delta_not_negative"] is False


def test_round_trajectory_persists_internal_validation_learning_curve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_load(path: Path, _label: str) -> dict:
        round_index = int(path.stem.split("-")[-1])
        validation = (
            {
                "loss": 1.25,
                "token_accuracy": 0.75,
                "supervised_tokens": 128,
            }
            if round_index == 2
            else None
        )
        return {
            "round_summary": {
                "round": round_index,
                "mean_training_loss": 2.5,
                "validation": validation,
                "A": {"clipped_fraction": 0.25},
                "B": {"clipped_fraction": 0.5},
            }
        }

    monkeypatch.setattr(full, "load_object", fake_load)
    arm = {
        "stage": campaign.FIXED_STAGE,
        "domain": "finance",
        "arm_id": "fixed",
        "method": campaign.FIXED_METHOD,
        "seed": 1400,
        "rounds": 2,
        "initial_clip_norm_by_group": {"A": 1.0, "B": 2.0},
    }
    rows = campaign._round_trajectory_rows(tmp_path, [{"arms": [arm]}])
    assert len(rows) == 4
    round_one = [row for row in rows if row["round"] == 1]
    round_two = [row for row in rows if row["round"] == 2]
    for field in (
        "internal_validation_loss",
        "internal_validation_supervised_token_accuracy",
        "internal_validation_supervised_tokens",
    ):
        assert {row[field] for row in round_one} == {None}
    assert {row["internal_validation_loss"] for row in round_two} == {1.25}
    assert {
        row["internal_validation_supervised_token_accuracy"] for row in round_two
    } == {0.75}
    assert {row["internal_validation_supervised_tokens"] for row in round_two} == {128}


def test_development_target_gate_cannot_cancel_opposite_seed_errors() -> None:
    assessment = campaign.development_target_feasibility(
        [
            {"A": 0.3, "B": 0.3},
            {"A": 0.7, "B": 0.7},
        ],
        {"A": 0.5, "B": 0.5},
        0.15,
    )
    assert assessment["mean_error_by_group"] == pytest.approx({"A": 0.0, "B": 0.0})
    assert assessment["max_abs_seed_error_by_group"] == pytest.approx({"A": 0.2, "B": 0.2})
    assert assessment["feasible"] is False


def test_confirmation_mechanism_gate_is_per_seed_and_anti_cancellation() -> None:
    spec = campaign.load_spec(SPEC)
    passing = [
        {
            "round_2_to_50_hard_target_error_A": 0.1,
            "round_2_to_50_hard_target_error_B": -0.1,
        }
        for _ in range(6)
    ]
    result = campaign.confirmation_mechanism_validity(passing, spec)
    assert result["passed"] is True
    assert result["groups"]["A"]["passing_seed_count"] == 6

    cancelling = [
        {
            "round_2_to_50_hard_target_error_A": -0.2 if index % 2 else 0.2,
            "round_2_to_50_hard_target_error_B": 0.0,
        }
        for index in range(6)
    ]
    assert sum(row["round_2_to_50_hard_target_error_A"] for row in cancelling) == pytest.approx(0.0)
    result = campaign.confirmation_mechanism_validity(cancelling, spec)
    assert result["groups"]["A"]["mean_absolute_error"] == pytest.approx(0.2)
    assert result["groups"]["A"]["passed"] is False
    assert result["passed"] is False


def test_journal_trajectory_declares_desired_and_dynamic_target_errors() -> None:
    source = Path(campaign.__file__).read_text(encoding="utf-8")
    for column in (
        '"desired_hard_clipped_fraction"',
        '"hard_minus_desired_target"',
        '"actual_minus_dynamic_target"',
        '"exact_oracle_fields_privacy_label"',
    ):
        assert column in source
    spec = campaign.load_spec(SPEC)
    assert spec["scientific_boundary"]["cdf_reconstruction_evidence_role"] == (
        "descriptive_NON_DP_exact_vs_noisy_telemetry_without_accuracy_gate"
    )
    assert "target_CDF_tracking_attribution_supported" not in source
    assert "target_tracking_attribution_supported" not in source
    assert '"mean_hard_rate_target_achievement_supported"' in source
    assert (
        '"within_seed_temporal_tracking_stability_evidence_is_descriptive"'
        in source
    )


def test_rng_pairing_reuses_sample_and_gradient_noise_domains() -> None:
    spec = campaign.load_spec(SPEC)
    candidates = [
        {"upstream_rank": index, "C_A": index / 10.0, "C_B": float(index)}
        for index in range(1, 11)
    ]
    fixed = campaign.fixed_screen_arms(spec, "finance", candidates)
    lock = _fixed_lock(spec, fixed)
    development = campaign.development_arms(spec, "finance", lock)
    fixed_by_id = {arm["arm_id"]: arm for arm in fixed}
    for arm in development:
        reference = fixed_by_id[arm["reference_arm_id"]]
        assert arm["seed"] == reference["seed"]
        assert arm["rng_domain"] == reference["rng_domain"]
    confirmation = campaign.confirmation_arms(
        spec, "finance", lock, {"selected_profile": lock["calibrated_profiles"][0]}
    )
    for fixed_arm, adaptive_arm in zip(confirmation[::2], confirmation[1::2]):
        assert adaptive_arm["reference_arm_id"] == fixed_arm["arm_id"]
        assert adaptive_arm["rng_domain"] == fixed_arm["rng_domain"]


def test_commands_have_no_adaptive_arguments_on_fixed_and_no_excluded_variant(tmp_path: Path) -> None:
    spec = campaign.load_spec(SPEC)
    fixed = campaign.fixed_screen_arms(
        spec, "finance",
        [{"upstream_rank": i, "C_A": i / 10.0, "C_B": float(i)} for i in range(1, 11)],
    )
    lock = _fixed_lock(spec, fixed)
    adaptive = campaign.development_arms(spec, "finance", lock)[0]
    common = dict(
        repository=ROOT, python_bin=Path(sys.executable),
        input_manifest=tmp_path / "input.json", output_dir=tmp_path / "out",
        private_key=tmp_path / "key", stop_file=tmp_path / "stop",
    )
    fixed_command = full._arm_command(fixed[0], **common)
    adaptive_command = full._arm_command(adaptive, **common)
    assert not any(part.startswith("--slaclip") for part in fixed_command)
    assert "--slaclip-base-target-clipped-fraction-a" in adaptive_command
    for command in (fixed_command, adaptive_command):
        joined = " ".join(command).lower()
        assert "slaclip-q" not in joined
        assert "quantile" not in joined
        assert "--pair-noise-across-methods" in command


def test_run_arm_rechecks_input_manifest_and_inventory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = campaign.load_spec(SPEC)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"inventory_sha256": "a" * 64}), encoding="utf-8")
    key_path = tmp_path / "key"
    key_path.write_bytes(b"k" * 32)
    arms = campaign.fixed_screen_arms(
        spec, "finance",
        [{"upstream_rank": i, "C_A": i / 10.0, "C_B": float(i)} for i in range(1, 11)],
    )
    manifest = campaign._runtime_manifest(
        spec=spec, spec_path=SPEC, repository_sha="b" * 40,
        input_record={"path": str(input_path), "sha256": full.sha256_file(input_path), "inventory_sha256": "a" * 64},
        stage=campaign.FIXED_STAGE, arms=arms, created_at_utc="2026-08-23T00:00:00Z",
        parent_lock_sha256="c" * 64,
        private_key_commitment=full.sha256_file(key_path),
    )
    manifest_path = tmp_path / "runtime.json"
    full.atomic_json(manifest_path, manifest)
    called = []
    monkeypatch.setattr(full, "run_arm", lambda args: called.append(args) or 0)
    args = argparse.Namespace(
        manifest=manifest_path, repository=ROOT, python_bin=Path(sys.executable),
        private_key=key_path, arm_index=0,
    )
    assert campaign.run_arm(args) == 0 and called
    input_path.write_text(json.dumps({"inventory_sha256": "d" * 64}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        campaign.run_arm(args)


def test_run_arm_rejects_replaced_private_rng_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = campaign.load_spec(SPEC)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"inventory_sha256": "a" * 64}), encoding="utf-8")
    key_path = tmp_path / "key"
    key_path.write_bytes(b"k" * 32)
    arms = campaign.fixed_screen_arms(
        spec, "finance",
        [{"upstream_rank": i, "C_A": i / 10.0, "C_B": float(i)} for i in range(1, 11)],
    )
    manifest = campaign._runtime_manifest(
        spec=spec, spec_path=SPEC, repository_sha="b" * 40,
        input_record={"path": str(input_path), "sha256": full.sha256_file(input_path), "inventory_sha256": "a" * 64},
        stage=campaign.FIXED_STAGE, arms=arms, created_at_utc="2026-08-23T00:00:00Z",
        parent_lock_sha256="c" * 64,
        private_key_commitment=full.sha256_file(key_path),
    )
    manifest_path = tmp_path / "runtime.json"
    full.atomic_json(manifest_path, manifest)
    monkeypatch.setattr(full, "run_arm", lambda _args: 0)
    key_path.write_bytes(b"x" * 32)
    with pytest.raises(RuntimeError, match="private RNG key commitment"):
        campaign.run_arm(argparse.Namespace(
            manifest=manifest_path, repository=ROOT, python_bin=Path(sys.executable),
            private_key=key_path, arm_index=0,
        ))


def test_resume_and_tamper_are_fail_closed(tmp_path: Path) -> None:
    candidate = campaign._lock_payload({"status": "LOCKED", "value": 1})
    path = tmp_path / "lock.json"
    campaign._write_or_verify(path, candidate, "test lock")
    campaign._write_or_verify(path, candidate, "test lock")
    tampered = copy.deepcopy(candidate)
    tampered["value"] = 2
    full.atomic_json(path, tampered)
    with pytest.raises(RuntimeError, match="differs"):
        campaign._write_or_verify(path, candidate, "test lock")
    with pytest.raises(RuntimeError, match="self-hash"):
        campaign._load_lock(path, "test lock")


def test_resume_state_behavior_rejects_key_replacement(tmp_path: Path) -> None:
    key = tmp_path / "private.key"
    key.write_bytes(b"a" * 32)
    plan = campaign._lock_payload({
        "status": "IDENTIFIED_N128_CAMPAIGN_PREREGISTERED",
        "repository_sha": "b" * 40,
        "spec_sha256": "c" * 64,
        "private_key_commitment": full.sha256_file(key),
    })
    plan_path = tmp_path / "plan.json"
    full.atomic_json(plan_path, plan)
    assert campaign.validate_resume_state(plan_path, key, "b" * 40, "c" * 64) == plan
    key.write_bytes(b"x" * 32)
    with pytest.raises(RuntimeError, match="private RNG key commitment"):
        campaign.validate_resume_state(plan_path, key, "b" * 40, "c" * 64)


def test_single_allocation_wrapper_resources_dependency_and_test_only_boundary() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")
    executable_worker = "\n".join(
        line for line in worker.splitlines() if not line.lstrip().startswith("#")
    )
    assert "sbatch " not in executable_worker.lower()
    assert "--array" not in worker
    assert "--signal=B:USR1@300" in submitter
    assert '--dependency="afterok:$dependency_job_id"' in submitter
    assert 'gres="${DPLORA_IDENTIFIED_GRES:-gpu:a100:1}"' in submitter
    assert 'host_memory="${DPLORA_IDENTIFIED_HOST_MEMORY:-80G}"' in submitter
    assert 'walltime="${DPLORA_IDENTIFIED_WALLTIME:-24:00:00}"' in submitter
    assert "Prospective paths only" in submitter
    assert "do not create a campaign directory" in submitter
    assert submitter.count('sbatch "${sbatch_args[@]}" "$worker"') == 1
    assert "fixed_screen_20_arms" in worker
    assert "five_profile_development_10_arms" in worker
    assert "independent_confirmation_12_arms" in worker
    assert 'repository="${DPLORA_IDENTIFIED_REPO_DIR:-$script_repository}"' in submitter
    assert '--resume requires explicit DPLORA_IDENTIFIED_RUN_ID' in submitter
    assert 'resume requires the existing locked plan and RNG key' in worker
    assert 'validate-resume-state' in worker
    assert 'rm -f -- "$global_stop"' in worker
    assert worker.index('rm -f -- "$global_stop"') < worker.index('campaign_stage="runtime_preflight"')
