from pathlib import Path
from argparse import Namespace

import pytest

from paper_repro import full_slaclip_campaign as full
from paper_repro import groupwise_slaclip_campaign as campaign
from paper_repro.slaclip import full_slaclip_update


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "hpc" / "groupwise-slaclip-campaign-spec.json"


def fixed_lock():
    return {
        "lock_sha256": "a" * 64,
        "models": {
            "bert": {
                "selected_fixed_C": 3.0,
                "beta_A": 0.0,
                "beta_B_grid": [0.1, 0.2, 0.3, 0.4, 0.5],
            },
            "gpt2": {
                "selected_fixed_C": 1.0,
                "beta_A": 0.05,
                "beta_B_grid": [0.2, 0.3, 0.4, 0.5, 0.6],
            },
        },
    }


def selection_lock():
    return {
        "lock_sha256": "b" * 64,
        "models": {
            "bert": {
                "selected_initial_C": 2.5,
                "selected_beta_A": 0.0,
                "selected_beta_B": 0.3,
                "selected_eta": 0.005,
            },
            "gpt2": {
                "selected_initial_C": 0.75,
                "selected_beta_A": 0.05,
                "selected_beta_B": 0.4,
                "selected_eta": 0.0025,
            },
        },
    }


def test_spec_and_exact_arm_counts():
    spec = campaign.load_spec(SPEC)
    assert len(campaign.fixed_stage_arms(spec)) == 6
    stage2 = campaign.stage2_arms(spec, fixed_lock())
    stage3 = campaign.stage3_arms(spec, fixed_lock(), selection_lock())
    assert len(stage2) == 180
    assert len(stage3) == 120
    assert {arm["method"] for arm in stage3} == {
        full.FIXED_DP_METHOD,
        full.FULL_SLACLIP_METHOD,
        full.ORACLE_SLACLIP_METHOD,
    }
    assert sum(arm["method"] == full.FIXED_DP_METHOD for arm in stage3) == 40
    assert sum(arm["method"] == full.FULL_SLACLIP_METHOD for arm in stage3) == 40
    assert sum(arm["method"] == full.ORACLE_SLACLIP_METHOD for arm in stage3) == 40


def test_groupwise_fields_and_frozen_non_dp_calibration_provenance():
    spec = campaign.load_spec(SPEC)
    fixed = campaign.fixed_stage_arms(spec)[0]
    for key in (
        "slaclip_base_target_clipped_fraction",
        "slaclip_beta",
        "slaclip_base_target_clipped_fraction_by_group",
        "slaclip_beta_by_group",
        "slaclip_baseline_calibration_lock_sha256",
        "slaclip_calibration_provenance",
    ):
        assert fixed[key] is None
    assert fixed["acknowledge_slaclip_baseline_calibration_is_non_dp"] is False

    adaptive = campaign.stage2_arms(spec, fixed_lock())[0]
    assert adaptive["slaclip_base_target_clipped_fraction"] is None
    assert adaptive["slaclip_beta"] is None
    assert adaptive["slaclip_base_target_clipped_fraction_by_group"] == adaptive["slaclip_beta_by_group"]
    assert set(adaptive["slaclip_beta_by_group"]) == {"A", "B"}
    assert adaptive["slaclip_baseline_calibration_lock_sha256"] == "a" * 64
    assert adaptive["acknowledge_slaclip_baseline_calibration_is_non_dp"] is True
    assert adaptive["slaclip_calibration_provenance"] == (
        "exact_NON_DP_fixed_trajectory_diagnostics_frozen_before_fresh_seeds"
    )


def test_stationary_beta_back_substitutes_into_full_controller():
    clip_norm = 3.0
    q_value = 0.8
    r_value = 0.3
    beta, z_value = campaign.stationary_beta(q_value, r_value, clip_norm, 1e-6)
    update = full_slaclip_update(
        clip_norm,
        q_value,
        r_value,
        eta=0.01,
        base_target_clipped_fraction=beta,
        min_clip_norm=0.1,
        max_clip_norm=50.0,
    )
    assert z_value == pytest.approx(r_value / (clip_norm + 1e-6))
    assert update["controller_error"] == pytest.approx(0.0, abs=1e-12)
    assert update["next_clip_norm"] == pytest.approx(clip_norm)
    with pytest.raises(RuntimeError, match="cannot define"):
        campaign.stationary_beta(0.2, 0.3, clip_norm, 1e-6)


def test_oracle_noisy_pairs_include_group_betas_and_calibration_lock():
    spec = campaign.load_spec(SPEC)
    arms = campaign.stage3_arms(spec, fixed_lock(), selection_lock())
    manifest = {"arms": arms}
    pairs = full.resolve_oracle_noisy_arm_pairs(manifest)
    assert len(pairs) == 40
    for oracle, noisy in pairs:
        assert oracle["slaclip_beta_by_group"] == noisy["slaclip_beta_by_group"]
        assert oracle["slaclip_baseline_calibration_lock_sha256"] == noisy["slaclip_baseline_calibration_lock_sha256"]
        assert oracle["slaclip_calibration_provenance"] == noisy["slaclip_calibration_provenance"]


def test_groupwise_arm_command_uses_only_ab_targets_and_non_dp_ack(tmp_path):
    spec = campaign.load_spec(SPEC)
    arm = campaign.stage2_arms(spec, fixed_lock())[0]
    command = full._arm_command(
        arm,
        repository=ROOT,
        python_bin=Path("/python"),
        input_manifest=tmp_path / "inputs.json",
        output_dir=tmp_path / "output",
        private_key=tmp_path / "key",
        stop_file=tmp_path / "stop",
    )
    assert "--slaclip-base-target-clipped-fraction-a" in command
    assert "--slaclip-base-target-clipped-fraction-b" in command
    assert "--slaclip-base-target-clipped-fraction" not in command
    assert "--slaclip-beta" not in command
    assert "--slaclip-baseline-calibration-lock-sha256" in command
    assert "--acknowledge-slaclip-baseline-calibration-is-non-dp" in command


def test_preflight_does_not_claim_baseline_derived_calibration(tmp_path):
    spec = campaign.load_spec(SPEC)
    master = {
        "campaign_name": spec["campaign_name"],
        "created_at_utc": "2026-08-07T00:00:00+00:00",
        "repository_sha": "c" * 40,
        "spec_sha256": "d" * 64,
        "input_manifest_path": str(tmp_path / "inputs.json"),
        "input_manifest_sha256": "e" * 64,
    }
    value = campaign._preflight(master, spec)
    adaptive = next(arm for arm in value["arms"] if arm["method"] == full.FULL_SLACLIP_METHOD)
    assert adaptive["slaclip_baseline_calibration_lock_sha256"] is None
    assert adaptive["acknowledge_slaclip_baseline_calibration_is_non_dp"] is False


def test_spec_explicitly_excludes_lr_scan_and_slaclip_q():
    spec = campaign.load_spec(SPEC)
    boundary = spec["scientific_boundary"]
    assert boundary["learning_rate_policy"].startswith("fixed_at_5e-4")
    assert boundary["excluded_method_family"] == "SlaClip-Q"
    assert spec["common"]["learning_rate"] == 5e-4


def test_locked_source_evidence_and_calibration_csv_fail_closed(tmp_path, monkeypatch):
    spec = campaign.load_spec(SPEC)
    master = {
        "manifest_sha256": "1" * 64,
        "repository_sha": "2" * 40,
        "input_manifest_path": str(tmp_path / "inputs.json"),
        "created_at_utc": "2026-08-07T00:00:00+00:00",
    }
    lock = {"lock_sha256": "3" * 64, "source_evidence": [], "calibration_csv_sha256": "4" * 64}
    calibration = tmp_path / "fixed_groupwise_beta_calibration.csv"
    calibration.write_text("immutable\n", encoding="utf-8")
    monkeypatch.setattr(campaign, "_validate_fixed_lock", lambda *_args: None)

    def evidence_changed(*_args):
        raise RuntimeError("locked source evidence changed")

    monkeypatch.setattr(campaign.staged, "_verify_locked_evidence", evidence_changed)
    with pytest.raises(RuntimeError, match="source evidence"):
        campaign._ensure_stage2(tmp_path, SPEC, spec, master, lock)

    monkeypatch.setattr(campaign.staged, "_verify_locked_evidence", lambda *_args: None)
    with pytest.raises(RuntimeError, match="calibration CSV"):
        campaign._ensure_stage2(tmp_path, SPEC, spec, master, lock)


def test_selection_rule_is_lexicographic_and_stability_breaks_metric_ties():
    base = {
        "mean_paired_final_loss_delta": -0.01,
        "mean_paired_normalized_loss_auc_delta": -0.02,
        "controller_instability_event_count": 3,
        "eta": 0.005,
        "beta_B": 0.4,
        "initial_C": 3.0,
    }
    candidates = [
        {**base, "controller_instability_event_count": 4, "eta": 0.0025},
        {**base, "controller_instability_event_count": 3, "eta": 0.01},
        {**base, "mean_paired_final_loss_delta": -0.011, "controller_instability_event_count": 99},
    ]
    ordered = sorted(candidates, key=campaign._selection_sort_key)
    assert ordered[0]["mean_paired_final_loss_delta"] == -0.011
    assert ordered[1]["controller_instability_event_count"] == 3


def test_strict_aggregate_rejects_incomplete_campaign(tmp_path):
    spec = campaign.load_spec(SPEC)
    inputs = tmp_path / "input-manifest.json"
    inputs.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "campaign"
    root.mkdir()
    manifest = campaign._runtime(
        spec, SPEC, "a" * 40, inputs, "2026-08-07T00:00:00+00:00",
        campaign.fixed_stage_arms(spec), campaign.FIXED_STAGE, None,
    )
    full.atomic_json(root / campaign.MASTER_RUNTIME_NAME, manifest)
    (root / "arm-status").mkdir()
    with pytest.raises(RuntimeError, match="requires both immutable selection locks"):
        campaign.aggregate_campaign(Namespace(
            campaign_root=root, spec=SPEC, require_complete=True
        ))


def test_worker_is_one_allocation_sequential_and_signal_safe():
    worker = (ROOT / "hpc" / "groupwise_slaclip_campaign.sbatch").read_text(encoding="utf-8")
    assert "--ntasks=1" in worker
    assert "--array" not in worker
    executable_lines = "\n".join(
        line for line in worker.splitlines() if not line.lstrip().startswith("#")
    )
    assert "sbatch " not in executable_lines.lower()
    assert "run-smoke" in worker
    assert "run-arm" in worker
    assert "source \"$exit_policy\"" in worker
    assert "wait_for_full_slaclip_child" in worker
    assert "run_step_signal_safe" in worker
