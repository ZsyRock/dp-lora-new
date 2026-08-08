import copy
from collections import Counter
from pathlib import Path

import pytest

from paper_repro import full_slaclip_campaign as full
from paper_repro import oracle_ceiling_campaign as campaign
from paper_repro import staged_slaclip_campaign as staged


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "hpc" / "oracle-ceiling-campaign-spec.json"
WORKER = ROOT / "hpc" / "oracle_ceiling_campaign.sbatch"
SUBMITTER = ROOT / "hpc" / "submit_oracle_ceiling_campaign.sh"
SHARED_SUBMITTER = ROOT / "hpc" / "submit_staged_slaclip_tuned_fixed.sh"


def _coarse_lock() -> dict:
    return {
        "models": {
            "bert": {
                "top_candidates": [
                    {"C_A": 0.75, "C_B": 2.5},
                    {"C_A": 1.0, "C_B": 3.0},
                    {"C_A": 1.25, "C_B": 3.5},
                ]
            },
            "gpt2": {
                "top_candidates": [
                    {"C_A": 0.1, "C_B": 0.85},
                    {"C_A": 0.125, "C_B": 1.0},
                    {"C_A": 0.15, "C_B": 1.15},
                ]
            },
        }
    }


def _fixed_lock() -> dict:
    return {
        "lock_sha256": "a" * 64,
        "models": {
            "bert": {
                "selected_fixed_C_by_group": {"A": 1.0, "B": 3.0},
                "beta_A_grid": [0.1, 0.5, 0.9],
                "beta_B_grid": [0.2, 0.6, 1.0],
            },
            "gpt2": {
                "selected_fixed_C_by_group": {"A": 0.125, "B": 1.0},
                "beta_A_grid": [0.0, 0.4, 0.8],
                "beta_B_grid": [0.15, 0.55, 0.95],
            },
        },
    }


def _oracle_lock() -> dict:
    return {
        "models": {
            "bert": {
                "selected_beta_A": 0.5,
                "selected_beta_B": 0.6,
                "selected_eta": 0.05,
            },
            "gpt2": {
                "selected_beta_A": 0.4,
                "selected_beta_B": 0.55,
                "selected_eta": 0.2,
            },
        }
    }


def test_spec_and_all_stage_arm_counts_and_methods() -> None:
    spec = campaign.load_spec(SPEC)
    coarse = campaign.fixed_coarse_arms(spec)
    refinement = campaign.fixed_refinement_arms(spec, _coarse_lock())
    oracle = campaign.oracle_development_arms(spec, _fixed_lock())
    confirmation = campaign.confirmation_arms(
        spec, _fixed_lock(), _oracle_lock()
    )

    assert campaign.EXPECTED_COUNTS == {
        campaign.FIXED_COARSE_STAGE: 60,
        campaign.FIXED_REFINEMENT_STAGE: 18,
        campaign.ORACLE_DEVELOPMENT_STAGE: 168,
        campaign.CONFIRMATION_STAGE: 80,
        "total": 326,
    }
    assert [len(coarse), len(refinement), len(oracle), len(confirmation)] == [
        60,
        18,
        168,
        80,
    ]
    assert Counter(arm["method"] for arm in coarse) == {
        campaign.FIXED_METHOD: 60
    }
    assert Counter(arm["method"] for arm in refinement) == {
        campaign.FIXED_METHOD: 18
    }
    assert Counter(arm["method"] for arm in oracle) == {
        campaign.FIXED_METHOD: 6,
        campaign.ORACLE_METHOD: 162,
    }
    assert Counter(arm["method"] for arm in confirmation) == {
        campaign.FIXED_METHOD: 40,
        campaign.ORACLE_METHOD: 40,
    }
    assert all("slaclip_q" not in arm["method"].lower() for arm in oracle)

    # Exercise the executable runtime-arm validator, not only the campaign's
    # combinatorics. This catches schema drift in the generic runner.
    for arms in (coarse, refinement, oracle, confirmation):
        for index, arm in enumerate(arms):
            full._validate_runtime_arm(arm, index=index)


def test_groupwise_initial_thresholds_are_authoritative_and_paired() -> None:
    spec = campaign.load_spec(SPEC)
    coarse = campaign.fixed_coarse_arms(spec)
    expected_grids = spec["fixed_coarse"]["clip_norm_grid_by_model"]

    for model in campaign.MODELS:
        subset = [arm for arm in coarse if arm["models"] == [model]]
        assert {
            arm["initial_clip_norm_by_group"]["A"] for arm in subset
        } == set(expected_grids[model]["A"])
        assert {
            arm["initial_clip_norm_by_group"]["B"] for arm in subset
        } == set(expected_grids[model]["B"])
        assert all(
            arm["initial_clip_norm"]
            == arm["initial_clip_norm_by_group"]["B"]
            for arm in subset
        )

    oracle = campaign.oracle_development_arms(spec, _fixed_lock())
    by_id = {arm["arm_id"]: arm for arm in oracle}
    for arm in oracle:
        if arm["method"] != campaign.ORACLE_METHOD:
            continue
        reference = by_id[arm["reference_arm_id"]]
        assert reference["method"] == campaign.FIXED_METHOD
        assert arm["initial_clip_norm_by_group"] == reference[
            "initial_clip_norm_by_group"
        ]
        assert arm["seed"] == reference["seed"]
        assert arm["rng_domain"] == reference["rng_domain"]
        assert arm["controller_input"] == full.EXACT_CONTROLLER_INPUT
        assert arm["slaclip_beta_by_group"] == arm[
            "slaclip_base_target_clipped_fraction_by_group"
        ]
        assert arm["slaclip_baseline_calibration_lock_sha256"] == "a" * 64
        assert arm["acknowledge_slaclip_baseline_calibration_is_non_dp"] is True

    mismatched = copy.deepcopy(coarse[0])
    mismatched["initial_clip_norm_by_group"]["B"] += 1.0
    with pytest.raises(RuntimeError, match="legacy clip norm"):
        full._validate_runtime_arm(mismatched, index=0)


def test_stage_seeds_are_disjoint_and_arm_seed_sets_match_the_spec() -> None:
    spec = campaign.load_spec(SPEC)
    declared = {
        campaign.FIXED_COARSE_STAGE: set(spec["fixed_coarse"]["seeds"]),
        campaign.FIXED_REFINEMENT_STAGE: set(
            spec["fixed_refinement"]["seeds"]
        ),
        campaign.ORACLE_DEVELOPMENT_STAGE: set(
            spec["oracle_development"]["seeds"]
        ),
        campaign.CONFIRMATION_STAGE: set(spec["confirmation"]["seeds"]),
    }
    values = list(declared.values())
    assert all(
        not left & right
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )

    actual = {
        campaign.FIXED_COARSE_STAGE: campaign.fixed_coarse_arms(spec),
        campaign.FIXED_REFINEMENT_STAGE: campaign.fixed_refinement_arms(
            spec, _coarse_lock()
        ),
        campaign.ORACLE_DEVELOPMENT_STAGE: campaign.oracle_development_arms(
            spec, _fixed_lock()
        ),
        campaign.CONFIRMATION_STAGE: campaign.confirmation_arms(
            spec, _fixed_lock(), _oracle_lock()
        ),
    }
    for stage, arms in actual.items():
        assert {arm["seed"] for arm in arms} == declared[stage]
        assert all(
            arm["rng_domain"] == f"oracle-ceiling:s{arm['seed']}"
            for arm in arms
        )


def test_three_point_beta_grid_uses_quantiles_and_deterministic_fallback() -> None:
    grid, fallback = campaign._three_point_beta_grid([0.0, 0.5, 1.0])
    assert grid == pytest.approx([0.1, 0.5, 0.9])
    assert fallback is False

    duplicate_grid, duplicate_fallback = campaign._three_point_beta_grid(
        [0.5] * 25
    )
    assert duplicate_grid == [0.0, 0.5, 1.0]
    assert duplicate_fallback is True
    assert campaign._three_point_beta_grid([0.5] * 25) == (
        duplicate_grid,
        duplicate_fallback,
    )

    with pytest.raises(RuntimeError, match="stationary-beta"):
        campaign._three_point_beta_grid([])
    with pytest.raises(RuntimeError, match="stationary-beta"):
        campaign._three_point_beta_grid([0.2, 1.01])


def test_synthetic_confirmation_gate_has_pass_and_no_go_models() -> None:
    spec = campaign.load_spec(SPEC)
    paired = []
    for model in campaign.MODELS:
        passing = model == "bert"
        for seed in spec["confirmation"]["seeds"]:
            paired.append(
                {
                    "model": model,
                    "seed": seed,
                    "final_loss_delta_oracle_minus_fixed": (
                        -0.00075 if passing else -0.00001
                    ),
                    "normalized_loss_auc_delta_oracle_minus_fixed": (
                        -0.0001 if passing else 0.00001
                    ),
                    "controller_instability_events": 0 if passing else 11,
                }
            )

    records = campaign._gate_records(spec, paired)
    by_model = {record["model"]: record for record in records}

    bert = by_model["bert"]
    assert bert["seed_count"] == 20
    assert bert["oracle_ceiling_gate_passed"] is True
    assert all(bert["criteria"].values())
    assert bert["final_loss_delta_mean"] == pytest.approx(-0.00075)
    assert bert["final_loss_delta_ci95_high"] < 0.0
    assert bert["final_loss_delta_exact_sign_flip_p"] < 0.025

    gpt2 = by_model["gpt2"]
    assert gpt2["oracle_ceiling_gate_passed"] is False
    assert gpt2["criteria"]["mean_below_negative_MPRD"] is False
    assert gpt2["criteria"]["normalized_loss_auc_mean_not_positive"] is False
    assert (
        gpt2["criteria"]["max_controller_instability_events_at_most_10"]
        is False
    )


def test_gate_rejects_incomplete_confirmation_pairing() -> None:
    spec = campaign.load_spec(SPEC)
    with pytest.raises(RuntimeError, match="confirmation pairing is incomplete"):
        campaign._gate_records(
            spec,
            [
                {
                    "model": "bert",
                    "final_loss_delta_oracle_minus_fixed": -1.0,
                    "normalized_loss_auc_delta_oracle_minus_fixed": -1.0,
                    "controller_instability_events": 0,
                }
            ],
        )


def test_gate_lock_is_recomputed_from_frozen_confirmation_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = campaign.load_spec(SPEC)
    paired = [
        {
            "model": model,
            "seed": seed,
            "final_loss_delta_oracle_minus_fixed": -0.001,
            "normalized_loss_auc_delta_oracle_minus_fixed": -0.001,
            "controller_instability_events": 0,
        }
        for model in campaign.MODELS
        for seed in spec["confirmation"]["seeds"]
    ]
    records = campaign._gate_records(spec, paired)
    passed = [record["model"] for record in records]
    master = {"manifest_sha256": "1" * 64}
    fixed = {"lock_sha256": "2" * 64}
    oracle = {"lock_sha256": "3" * 64}
    stage4 = {"manifest_sha256": "4" * 64}
    lock = {
        "status": "ORACLE_CEILING_GATE_PASSED",
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "fixed_selection_lock_sha256": fixed["lock_sha256"],
        "oracle_selection_lock_sha256": oracle["lock_sha256"],
        "stage4_runtime_manifest_sha256": stage4["manifest_sha256"],
        "confirmation_seeds": spec["confirmation"]["seeds"],
        "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
        "passed_models": passed,
        "models": records,
        "source_evidence": [],
    }
    monkeypatch.setattr(staged, "_verify_locked_evidence", lambda *_args: None)
    monkeypatch.setattr(
        campaign, "_confirmation_rows", lambda *_args: ([], paired)
    )
    campaign._validate_gate_lock(
        tmp_path, lock, master, fixed, oracle, stage4, spec
    )

    tampered = copy.deepcopy(lock)
    tampered["models"][0]["final_loss_delta_mean"] = 0.0
    with pytest.raises(RuntimeError, match="frozen evidence"):
        campaign._validate_gate_lock(
            tmp_path, tampered, master, fixed, oracle, stage4, spec
        )


def test_oracle_selection_lock_rejects_candidate_or_selected_tampering() -> None:
    spec = campaign.load_spec(SPEC)
    fixed = _fixed_lock()
    master = {"manifest_sha256": "1" * 64}
    stage3 = {"manifest_sha256": "2" * 64}
    models = {}
    for model in campaign.MODELS:
        fixed_record = fixed["models"][model]
        candidates = []
        for beta_a in fixed_record["beta_A_grid"]:
            for beta_b in fixed_record["beta_B_grid"]:
                for eta in spec["oracle_development"]["etas"]:
                    candidates.append(
                        {
                            "beta_A": beta_a,
                            "beta_B": beta_b,
                            "eta": eta,
                            "seed_count": 3,
                            "mean_paired_final_loss_delta": beta_a + beta_b + eta,
                            "mean_paired_normalized_loss_auc_delta": 0.0,
                            "controller_instability_event_count": 0,
                        }
                    )
        candidates.sort(key=campaign._oracle_sort_key)
        selected = candidates[0]
        models[model] = {
            "selected_beta_A": selected["beta_A"],
            "selected_beta_B": selected["beta_B"],
            "selected_eta": selected["eta"],
            "selected_initial_C_by_group": fixed_record[
                "selected_fixed_C_by_group"
            ],
            "selected_num_slots": 5,
            "ordered_candidates": candidates,
        }
    lock = {
        "status": "EXACT_ORACLE_DEVELOPMENT_SELECTION_LOCKED",
        "master_runtime_manifest_sha256": master["manifest_sha256"],
        "fixed_selection_lock_sha256": fixed["lock_sha256"],
        "stage3_runtime_manifest_sha256": stage3["manifest_sha256"],
        "selection_rule": spec["oracle_development"]["selection_rule"],
        "development_seeds": spec["oracle_development"]["seeds"],
        "confirmation_data_accessed": False,
        "oracle_privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
        "models": models,
    }
    campaign._validate_oracle_lock(lock, master, fixed, stage3, spec)
    tampered = copy.deepcopy(lock)
    tampered["models"]["bert"]["selected_eta"] = 0.2
    with pytest.raises(RuntimeError, match="frozen candidate"):
        campaign._validate_oracle_lock(tampered, master, fixed, stage3, spec)


def test_preflight_templates_match_run_smoke_c10_resolution_contract() -> None:
    spec = campaign.load_spec(SPEC)
    master = {
        "campaign_name": spec["campaign_name"],
        "created_at_utc": "2026-08-08T00:00:00+00:00",
        "repository_sha": "a" * 40,
        "spec_sha256": "b" * 64,
        "input_manifest_path": "/immutable/input-manifest.json",
        "input_manifest_sha256": "c" * 64,
    }
    runtime = campaign._preflight(master, spec)

    # Mirror full.run_preflight_smoke's deliberate oracle-to-noisy-template
    # mapping and its paper-style C=10 selector. Both requested smoke methods
    # must resolve to exactly one template at the earliest seed.
    for desired_method in (campaign.FIXED_METHOD, campaign.ORACLE_METHOD):
        template_method = (
            campaign.NOISY_METHOD
            if desired_method == campaign.ORACLE_METHOD
            else desired_method
        )
        candidates = [
            arm
            for arm in runtime["arms"]
            if arm["family"] == "primary"
            and arm["method"] == template_method
            and arm["initial_clip_norm"] == 10.0
        ]
        assert candidates, f"no C=10 smoke template for {desired_method}"
        assert all(
            arm["initial_clip_norm_by_group"] == {"A": 1.0, "B": 10.0}
            for arm in candidates
        )
        first_seed = min(int(arm["seed"]) for arm in candidates)
        assert sum(int(arm["seed"]) == first_seed for arm in candidates) == 1


def test_arm_command_passes_both_groupwise_clip_norm_flags(tmp_path: Path) -> None:
    spec = campaign.load_spec(SPEC)
    fixed = next(
        arm
        for arm in campaign.fixed_coarse_arms(spec)
        if arm["models"] == ["bert"]
        and arm["initial_clip_norm_by_group"] == {"A": 0.75, "B": 2.5}
    )
    oracle = next(
        arm
        for arm in campaign.oracle_development_arms(spec, _fixed_lock())
        if arm["method"] == campaign.ORACLE_METHOD
        and arm["models"] == ["gpt2"]
    )

    for arm in (fixed, oracle):
        command = full._arm_command(
            arm,
            repository=ROOT,
            python_bin=Path("/runtime/bin/python"),
            input_manifest=tmp_path / "input-manifest.json",
            output_dir=tmp_path / "output",
            private_key=tmp_path / "private.key",
            stop_file=tmp_path / "stop.request",
        )
        assert command.count("--clip-norm-a") == 1
        assert command.count("--clip-norm-b") == 1
        assert command[command.index("--clip-norm-a") + 1] == str(
            arm["initial_clip_norm_by_group"]["A"]
        )
        assert command[command.index("--clip-norm-b") + 1] == str(
            arm["initial_clip_norm_by_group"]["B"]
        )


def test_worker_is_one_allocation_and_runs_stages_and_locks_in_order() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    assert "#SBATCH --nodes=1" in worker
    assert "#SBATCH --ntasks=1" in worker
    assert "#SBATCH --array" not in worker
    assert '[[ ! "$step_gres" =~ ^gpu:[A-Za-z0-9_-]+:1$ ]]' in worker

    executable = "\n".join(
        line for line in worker.splitlines() if not line.lstrip().startswith("#")
    )
    assert "sbatch " not in executable.lower()
    assert "--array" not in executable

    ordered_markers = (
        'campaign_stage="groupwise_fixed_coarse"',
        'lock-coarse "${identity[@]}"',
        'campaign_stage="groupwise_fixed_refinement"',
        'lock-fixed "${identity[@]}"',
        'campaign_stage="exact_oracle_development"',
        'lock-oracle "${identity[@]}"',
        'campaign_stage="fresh_seed_confirmation"',
        'campaign_stage="oracle_ceiling_gate"',
        'lock-gate "${identity[@]}"',
        'campaign_stage="final_revalidation"',
    )
    positions = [worker.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    assert worker.count('run_manifest_stage "$campaign_stage"') == 4


def test_archive_whitelist_covers_oracle_ceiling_control_plane(
    tmp_path: Path,
) -> None:
    expected_small_files = {
        "stage2-fixed-refinement-runtime-manifest.json",
        "stage3-oracle-development-runtime-manifest.json",
        "stage4-confirmation-runtime-manifest.json",
        "groupwise-fixed-coarse-selection.lock.json",
        "strong-groupwise-fixed-selection.lock.json",
        "oracle-ceiling-selection.lock.json",
        "oracle-ceiling-gate.lock.json",
        "strong_groupwise_fixed_beta_calibration.csv",
    }
    root = tmp_path / "campaign"
    root.mkdir()
    for name in expected_small_files:
        path = root / name
        path.write_text("immutable\n", encoding="utf-8")
        assert full._archive_candidate(path, root), name


def test_submit_wrapper_safely_appends_optional_slurm_dependency() -> None:
    wrapper = SUBMITTER.read_text(encoding="utf-8")
    shared = SHARED_SUBMITTER.read_text(encoding="utf-8")

    assert 'if [[ -n "${DPLORA_ORACLE_DEPENDENCY:-}" ]]' in wrapper
    assert (
        'export DPLORA_STAGED_DEPENDENCY="$DPLORA_ORACLE_DEPENDENCY"'
        in wrapper
    )
    assert (
        'if [[ -n "$dependency" && ! "$dependency" =~ '
        '^afterany:[1-9][0-9]*$'
        in shared
    )
    assert 'sbatch_args+=("--dependency=$dependency")' in shared
    assert '"$sbatch_bin" "${sbatch_args[@]}" "$worker"' in shared
    executable = "\n".join(
        line for line in shared.splitlines() if not line.lstrip().startswith("#")
    )
    assert "eval " not in executable


def test_fixed_trajectory_uses_group_specific_initial_thresholds(
    tmp_path: Path,
) -> None:
    arm = {
        "stage": campaign.FIXED_COARSE_STAGE,
        "arm_id": "groupwise-fixed",
        "method": campaign.FIXED_METHOD,
        "models": ["bert"],
        "seed": 480,
        "rounds": 1,
        "initial_clip_norm": 3.0,
        "initial_clip_norm_by_group": {"A": 0.75, "B": 3.0},
        "slaclip_beta": None,
        "slaclip_base_target_clipped_fraction_by_group": None,
    }
    shard = {
        "client_records": [
            {
                "gradient_groups": {
                    group: {
                        "raw_norm": threshold / 2.0,
                        "raw_to_threshold_ratio": 0.5,
                        "removed_gradient_l2": 0.0,
                        "retained_energy_fraction": 1.0,
                        "noise_l2_norm": threshold,
                    }
                    for group, threshold in {"A": 0.75, "B": 3.0}.items()
                }
            }
        ],
        "round_summary": {
            group: {
                "clipped_count": 0,
                "clipped_fraction": 0.0,
                "would_clip_count": 0,
                "would_clip_fraction": 0.0,
            }
            for group in ("A", "B")
        },
    }
    shard["round_summary"]["federated_update"] = {
        group: {} for group in ("A", "B")
    }
    shard_path = staged._round_shard_path(tmp_path, arm, 1)
    full.atomic_json(shard_path, shard)

    rows = staged._round_trajectory_rows(tmp_path, arm)
    by_group = {row["group"]: row for row in rows}
    assert by_group["A"]["initial_C"] == 0.75
    assert by_group["A"]["clip_threshold_used"] == 0.75
    assert by_group["A"]["next_clip_threshold"] == 0.75
    assert by_group["B"]["initial_C"] == 3.0
    assert by_group["B"]["clip_threshold_used"] == 3.0
    assert by_group["B"]["next_clip_threshold"] == 3.0


def test_fixed_groupwise_threshold_and_noise_evidence_is_fail_closed(
    tmp_path: Path,
) -> None:
    arm = {
        "arm_id": "fixed-evidence",
        "method": campaign.FIXED_METHOD,
        "models": ["bert"],
        "rounds": 1,
        "num_clients": 1,
        "noise_multiplier": 2.0,
        "initial_clip_norm_by_group": {"A": 0.75, "B": 3.0},
    }
    arm_root = tmp_path / "arms" / arm["arm_id"]
    full.atomic_json(
        arm_root / "run_config.json",
        {
            "effective_config": {"clip_norm_A": 0.75, "clip_norm_B": 3.0},
            "scientific_contract": {
                "algorithm_contract": {
                    "initial_clip_threshold_by_group": {"A": 0.75, "B": 3.0},
                    "groupwise_fixed_thresholds": True,
                }
            },
        },
    )
    shard_path = campaign._round_path(tmp_path, arm, 1)
    shard = {
        "round": 1,
        "model": "bert",
        "method": campaign.FIXED_METHOD,
        "client_records": [
            {
                "gradient_groups": {
                    "A": {"clip_threshold": 0.75, "noise_std_per_coordinate": 1.5},
                    "B": {"clip_threshold": 3.0, "noise_std_per_coordinate": 6.0},
                }
            }
        ],
    }
    full.atomic_json(shard_path, shard)
    campaign._validate_fixed_groupwise_threshold_evidence(tmp_path, arm)

    shard["client_records"][0]["gradient_groups"]["A"][
        "noise_std_per_coordinate"
    ] = 6.0
    full.atomic_json(shard_path, shard)
    with pytest.raises(RuntimeError, match="threshold/noise evidence"):
        campaign._validate_fixed_groupwise_threshold_evidence(tmp_path, arm)


def test_calibration_treats_first_endpoint_as_proxy_not_unclipped_identity(
    tmp_path: Path,
) -> None:
    arm = {
        "arm_id": "proxy-calibration",
        "models": ["bert"],
        "seed": 480,
    }
    thresholds = {"A": 1.0, "B": 3.0}
    shard = {
        "round": 1,
        "model": "bert",
        "method": campaign.FIXED_METHOD,
        "client_records": [
            {
                "gradient_groups": {
                    group: {
                        "raw_norm": 0.9 * threshold,
                        "clip_threshold": threshold,
                    }
                    for group, threshold in thresholds.items()
                }
            }
        ],
        "round_summary": {
            group: {"clipped_fraction": 0.0}
            for group in campaign.GROUPS
        },
    }
    full.atomic_json(campaign._round_path(tmp_path, arm, 1), shard)
    values, rows = campaign._calibrate_groupwise(
        tmp_path,
        [arm],
        thresholds=thresholds,
        num_slots=5,
        epsilon=1e-6,
        rounds=1,
    )
    assert set(values) == {"A", "B"}
    assert len(rows) == 2
    assert all(
        row["actual_minus_one_minus_near_threshold_proxy"]
        == pytest.approx(-0.5)
        for row in rows
    )
