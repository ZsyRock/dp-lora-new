from __future__ import annotations

import math
from pathlib import Path

import pytest

from paper_repro import full_slaclip_campaign as full
from paper_repro import staged_slaclip_campaign as staged
from paper_repro import target_profile_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "hpc" / "target-profile-campaign-spec.json"
WORKER = ROOT / "hpc" / "target_profile_campaign.sbatch"
SUBMITTER = ROOT / "hpc" / "submit_target_profile_campaign.sh"
SHARED_SUBMITTER = ROOT / "hpc" / "submit_staged_slaclip_tuned_fixed.sh"


def _calibration_lock(spec: dict) -> dict:
    models = {}
    for model in campaign.MODELS:
        profiles = []
        for index, position in enumerate((0.0, 0.25, 0.5, 0.75, 1.0), start=1):
            profiles.append({
                "profile_index": index,
                "interpolation_position": position,
                "groups": {
                    "A": {
                        "desired_hard_clipped_fraction": 0.0,
                        "beta": 0.0,
                    },
                    "B": {
                        "desired_hard_clipped_fraction": 0.2 + position * 0.4,
                        "beta": 0.3 + position * 0.4,
                    },
                },
            })
        models[model] = profiles
    return staged._lock_payload({
        "schema_version": 1,
        "models": models,
    })


def _upstream_fixed_lock() -> dict:
    return {
        "models": {
            "bert": {"selected_fixed_C_by_group": {"A": 1.0, "B": 3.0}},
            "gpt2": {"selected_fixed_C_by_group": {"A": 0.1, "B": 1.0}},
        }
    }


def test_strict_spec_and_arm_counts() -> None:
    spec = campaign.load_spec(SPEC)
    calibration = _calibration_lock(spec)
    upstream = _upstream_fixed_lock()
    development = campaign.development_arms(spec, calibration, upstream)
    assert len(development) == 156
    assert sum(arm["method"] == campaign.FIXED_METHOD for arm in development) == 6
    assert sum(arm["method"] == campaign.NOISY_METHOD for arm in development) == 150
    assert all(arm["slaclip_num_slots"] == 5 for arm in development if arm["method"] == campaign.NOISY_METHOD)

    selections = {
        "models": {
            model: [
                {
                    "profile_index": index,
                    "selected_eta": 0.001,
                }
                for index in range(1, 6)
            ]
            for model in campaign.MODELS
        }
    }
    confirmation = campaign.confirmation_arms(
        spec, calibration, upstream, selections
    )
    assert len(confirmation) == 240
    assert sum(arm["method"] == campaign.FIXED_METHOD for arm in confirmation) == 40
    assert sum(arm["method"] == campaign.NOISY_METHOD for arm in confirmation) == 200


def test_hard_rate_is_bracketed_then_beta_fits_stationary_surrogate() -> None:
    rows = [
        {"actual_clipped_fraction": 0.2,
         "remaining_non_small_gradient_fraction": 0.5,
         "stationary_surrogate_target_clipped": 0.1},
        {"actual_clipped_fraction": 0.2,
         "remaining_non_small_gradient_fraction": 0.75,
         "stationary_surrogate_target_clipped": 0.2},
        {"actual_clipped_fraction": 0.6,
         "remaining_non_small_gradient_fraction": 0.5,
         "stationary_surrogate_target_clipped": 0.3},
        {"actual_clipped_fraction": 0.6,
         "remaining_non_small_gradient_fraction": 0.75,
         "stationary_surrogate_target_clipped": 0.4},
    ]
    result = campaign.calibrate_beta_for_hard_rate(rows, 0.4)
    # Each stratum has weight 1/2 and each point within it has weight 1/4.
    expected = (
        0.25 * (0.5 * 0.1 + 0.75 * 0.2 + 0.5 * 0.3 + 0.75 * 0.4)
        / (0.25 * (0.5**2 + 0.75**2 + 0.5**2 + 0.75**2))
    )
    assert result["beta"] == pytest.approx(expected)
    assert result["bracketing_hard_rate_lower"] == 0.2
    assert result["bracketing_hard_rate_upper"] == 0.6
    assert result["bracketing_lower_weight"] == pytest.approx(0.5)
    assert result["bracketing_upper_weight"] == pytest.approx(0.5)
    assert result["weighted_hard_clipped_fraction"] == pytest.approx(0.4)
    assert result["hard_target_reconstruction_error"] == pytest.approx(0.0)
    assert result["weighted_stationary_surrogate_target_mean"] == pytest.approx(0.25)
    assert result["surrogate_fit_rmse"] >= 0.0
    assert result["box_constraint_feasible"] is True


def test_beta_fit_reports_upper_bound_infeasibility() -> None:
    rows = [
        {"actual_clipped_fraction": hard,
         "remaining_non_small_gradient_fraction": 0.1,
         "stationary_surrogate_target_clipped": 0.9}
        for hard in (0.2, 0.6)
    ]
    result = campaign.calibrate_beta_for_hard_rate(rows, 0.4)
    assert result["unconstrained_beta"] > 1.0
    assert result["beta"] == 1.0
    assert result["box_constraint_feasible"] is False
    assert result["surrogate_target_pointwise_feasible_weight"] == 0.0


def test_profiles_are_equal_spacing_over_p10_p90_and_allow_degenerate_A() -> None:
    spec = campaign.load_spec(SPEC)
    rows = []
    for model in campaign.MODELS:
        for group in campaign.GROUPS:
            for index in range(250):
                actual = 0.0 if group == "A" else index / 249.0
                rows.append({
                    "model": model,
                    "group": group,
                    "actual_clipped_fraction": actual,
                    "remaining_non_small_gradient_fraction": 0.8,
                    "stationary_surrogate_target_clipped": actual * 0.5,
                })
    profiles = campaign.derive_target_profiles(rows, spec)
    for model in campaign.MODELS:
        a_targets = [
            profile["groups"]["A"]["desired_hard_clipped_fraction"]
            for profile in profiles[model]
        ]
        b_targets = [
            profile["groups"]["B"]["desired_hard_clipped_fraction"]
            for profile in profiles[model]
        ]
        assert a_targets == [0.0] * 5
        gaps = [right - left for left, right in zip(b_targets, b_targets[1:])]
        assert max(gaps) - min(gaps) < 1e-15
        assert b_targets[0] == pytest.approx(staged._linear_quantile(
            [index / 249.0 for index in range(250)], 0.1
        ))
        assert b_targets[-1] == pytest.approx(staged._linear_quantile(
            [index / 249.0 for index in range(250)], 0.9
        ))


def test_full_slaclip_only_and_no_threshold_stability_claim() -> None:
    spec = campaign.load_spec(SPEC)
    assert spec["scientific_boundary"]["excluded_method_family"] == "SlaClip-Q"
    assert spec["scientific_boundary"]["utility_stability_not_threshold_stability"] is True
    assert spec["confirmation"]["hypothesis_count"] == 10
    assert math.isclose(
        spec["confirmation"]["bonferroni_alpha_per_hypothesis"], 0.005
    )
    assert spec["common"]["eval_every"] == 5
    assert spec["development"]["etas"][-1] == 0.01
    assert spec["confirmation"]["target_achievement_absolute_tolerance"] == 0.075
    assert spec["confirmation"]["full_clipped_round_fraction_cap"] == 0.2


def test_development_selection_ranks_feasibility_before_utility() -> None:
    infeasible_but_lower_loss = {
        "target_feasible": False,
        "mean_paired_final_loss_delta": -10.0,
        "mean_paired_normalized_loss_auc_delta": -10.0,
        "controller_instability_event_count": 0,
        "eta": 0.0005,
    }
    feasible = {
        "target_feasible": True,
        "mean_paired_final_loss_delta": 1.0,
        "mean_paired_normalized_loss_auc_delta": 1.0,
        "controller_instability_event_count": 10,
        "eta": 0.01,
    }
    ordered = sorted(
        [infeasible_but_lower_loss, feasible], key=campaign._candidate_sort_key
    )
    assert ordered[0] is feasible


def test_development_lock_rejects_selected_eta_not_ranked_first() -> None:
    spec = campaign.load_spec(SPEC)
    calibration = _calibration_lock(spec)
    candidates = [
        {
            "eta": eta,
            "target_feasible": True,
            "mean_paired_final_loss_delta": float(index),
            "mean_paired_normalized_loss_auc_delta": float(index),
            "controller_instability_event_count": index,
        }
        for index, eta in enumerate(spec["development"]["etas"])
    ]
    models = {}
    for model in campaign.MODELS:
        models[model] = []
        for profile in calibration["models"][model]:
            models[model].append({
                "profile_index": profile["profile_index"],
                "selected_eta": candidates[1]["eta"],
                "desired_hard_clipped_fraction_by_group": {
                    group: profile["groups"][group][
                        "desired_hard_clipped_fraction"
                    ]
                    for group in campaign.GROUPS
                },
                "beta_by_group": {
                    group: profile["groups"][group]["beta"]
                    for group in campaign.GROUPS
                },
                "selected_candidate_feasible": candidates[0][
                    "target_feasible"
                ],
                "no_feasible_candidate": False,
                "ordered_eta_candidates": candidates,
            })
    lock = staged._lock_payload({
        "schema_version": campaign.LOCK_SCHEMA_VERSION,
        "status": "TARGET_PROFILE_DEVELOPMENT_SELECTION_LOCKED",
        "master_runtime_manifest_sha256": "master",
        "calibration_lock_sha256": calibration["lock_sha256"],
        "selection_rule": spec["development"]["selection_rule"],
        "development_seeds": spec["development"]["seeds"],
        "confirmation_data_accessed": False,
        "models": models,
    })
    with pytest.raises(
        RuntimeError, match="development lock selected profile differs"
    ):
        campaign._validate_development_lock(
            lock, {"manifest_sha256": "master"}, calibration, spec
        )


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf])
def test_confirmation_metric_rejects_non_finite_or_bool(value: object) -> None:
    with pytest.raises(RuntimeError, match="missing/non-finite"):
        campaign._require_confirmation_metric(
            {"metric": value}, "metric", "synthetic"
        )


def _synthetic_confirmation_pairs(spec: dict) -> list[dict]:
    rows = []
    for model in campaign.MODELS:
        for profile_index in range(1, 6):
            for seed in spec["confirmation"]["seeds"]:
                fixed_loss = 2.0 + (int(seed) % 3) * 1e-5
                rows.append({
                    "model": model,
                    "profile_index": profile_index,
                    "target_profile_position": (profile_index - 1) / 4,
                    "seed": seed,
                    "eta": 0.001,
                    "beta_A": 0.5,
                    "beta_B": 0.5,
                    "desired_hard_clipped_fraction_A": 0.8,
                    "desired_hard_clipped_fraction_B": 0.8,
                    "final_loss_delta_slaclip_minus_fixed": -0.001,
                    "normalized_loss_auc_delta_slaclip_minus_fixed": -0.001,
                    "final_token_accuracy_delta_slaclip_minus_fixed": 0.01,
                    "loss_excess_total_variation_fixed": 0.2,
                    "loss_excess_total_variation_slaclip": 0.1,
                    "final_minus_best_fixed": 0.2,
                    "final_minus_best_slaclip": 0.1,
                    "mean_actual_clipped_fraction_A_observation": 0.8,
                    "mean_actual_clipped_fraction_B_observation": 0.8,
                    "actual_target_absolute_error_median_A": 0.05,
                    "actual_target_absolute_error_median_B": 0.05,
                    "fully_clipped_round_fraction_A": 0.0,
                    "fully_clipped_round_fraction_B": 0.0,
                    "final_loss_fixed": fixed_loss,
                    "final_loss_slaclip": fixed_loss - 0.001,
                })
    return rows


def test_confirmation_gate_uses_all_twenty_pairs_and_supports_joint_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = campaign.load_spec(SPEC)

    def fake_inference(values: list[float]) -> dict:
        mean = sum(values) / len(values)
        width = max(abs(mean) * 0.1, 1e-8)
        return {
            "n": len(values), "mean": mean, "median": mean,
            "sample_std": 0.0, "standard_error": 0.0,
            "ci95_low": mean - width, "ci95_high": mean + width,
            "cohens_dz": None,
            "negative_fraction": sum(value < 0 for value in values) / len(values),
            "zero_fraction": sum(value == 0 for value in values) / len(values),
            "exact_sign_flip_p": 0.001,
        }

    monkeypatch.setattr(campaign.full, "paired_inference", fake_inference)
    records = campaign._gate_records(
        spec, _synthetic_confirmation_pairs(spec)
    )
    assert len(records) == 10
    assert all(record["seed_count"] == 20 for record in records)
    assert all(record["joint_claim_supported"] for record in records)


def test_confirmation_gate_rejects_inference_that_drops_a_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = campaign.load_spec(SPEC)

    def incomplete_inference(values: list[float]) -> dict:
        return {
            "n": len(values) - 1, "mean": 0.0, "ci95_low": -1.0,
            "ci95_high": 1.0, "exact_sign_flip_p": 1.0,
        }

    monkeypatch.setattr(campaign.full, "paired_inference", incomplete_inference)
    with pytest.raises(RuntimeError, match="dropped paired seeds"):
        campaign._gate_records(spec, _synthetic_confirmation_pairs(spec))


def test_single_a100_worker_and_upstream_dependency_are_fail_closed() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")
    shared = SHARED_SUBMITTER.read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=1" in worker
    assert '[[ ! "$step_gres" =~ ^gpu:[A-Za-z0-9_-]+:1$ ]]' in worker
    assert "--array" not in worker
    assert "sbatch " not in "\n".join(
        line for line in worker.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert 'gpu:a100:1' in submitter
    assert 'afterok:1367079' in submitter
    assert 'oracle-ceiling-2c3d2c6-20260808a' in submitter
    for name in (
        "DPLORA_STAGED_UPSTREAM_CAMPAIGN_ROOT",
        "DPLORA_STAGED_UPSTREAM_REPOSITORY",
        "DPLORA_STAGED_UPSTREAM_EXPECTED_SHA",
        "DPLORA_STAGED_UPSTREAM_SPEC",
        "DPLORA_STAGED_UPSTREAM_INPUT_MANIFEST",
    ):
        assert name in shared
    assert '"${upstream_worker_args[@]}"' in shared


def test_target_control_plane_is_in_small_artifact_archive(
    tmp_path: Path,
) -> None:
    root = tmp_path / "target-profile-campaign"
    root.mkdir()
    for name in (
        "stage2-confirmation-runtime-manifest.json",
        "target-profile-calibration.lock.json",
        "target-profile-development-selection.lock.json",
        "target-profile-confirmation-gate.lock.json",
        "target_profile_calibration.csv",
        "target_profile_source_trajectory.csv",
        "confirmation_hypothesis_metrics.csv",
    ):
        path = root / name
        path.write_text("immutable\n", encoding="utf-8")
        assert full._archive_candidate(path, root), name
