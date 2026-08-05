from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from paper_repro import full_slaclip_campaign as full
from paper_repro import staged_slaclip_campaign as staged


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "hpc" / "staged-slaclip-tuned-fixed-spec.json"
WORKER_PATH = ROOT / "hpc" / "staged_slaclip_tuned_fixed.sbatch"
SUBMIT_PATH = ROOT / "hpc" / "submit_staged_slaclip_tuned_fixed.sh"


def official_spec() -> dict:
    return staged.load_spec(SPEC_PATH)


def fixed_lock_fixture(spec: dict) -> dict:
    grid = spec["fixed_development"]["clip_norm_grid"]
    models = {}
    for model, selected in (("bert", 3.0), ("gpt2", 1.0)):
        candidates, boundary = staged._c0_candidates(grid, selected)
        models[model] = {
            "selected_fixed_C": selected,
            "fixed_grid_boundary_hit": boundary,
            "slaclip_initial_C_candidates": candidates,
            "derived_beta_grid": [0.4, 0.5, 0.6, 0.7, 0.8],
            "beta_calibration": {},
            "ordered_fixed_C_candidates": [],
        }
    return {
        "models": models,
        "lock_sha256": "a" * 64,
    }


def slaclip_lock_fixture(fixed_lock: dict) -> dict:
    return {
        "models": {
            "bert": {
                "selected_fixed_C": 3.0,
                "selected_initial_C": 3.0,
                "selected_beta": 0.7,
            },
            "gpt2": {
                "selected_fixed_C": 1.0,
                "selected_initial_C": 1.0,
                "selected_beta": 0.6,
            },
        },
        "lock_sha256": "b" * 64,
    }


def test_official_spec_and_all_stage_arm_counts() -> None:
    spec = official_spec()
    stage1 = staged.fixed_stage_arms(spec)
    assert len(stage1) == 130
    assert len({arm["arm_id"] for arm in stage1}) == 130
    assert {tuple(arm["models"]) for arm in stage1} == {("bert",), ("gpt2",)}
    assert {arm["method"] for arm in stage1} == {full.FIXED_DP_METHOD}
    assert 0.1 in {arm["initial_clip_norm"] for arm in stage1}
    assert 10.0 in {arm["initial_clip_norm"] for arm in stage1}

    fixed_lock = fixed_lock_fixture(spec)
    stage2 = staged.stage2_arms(spec, fixed_lock)
    assert len(stage2) == 150
    assert len({arm["arm_id"] for arm in stage2}) == 150
    assert {arm["slaclip_num_slots"] for arm in stage2} == {5}
    assert {arm["slaclip_eta"] for arm in stage2} == {0.2}
    assert {arm["method"] for arm in stage2} == {full.FULL_SLACLIP_METHOD}
    assert all("slaclip_q" not in arm["arm_id"] for arm in stage2)
    assert all(
        arm["reference_arm_id"].startswith(f"dev-fixed-{arm['models'][0]}-")
        for arm in stage2
    )

    stage3 = staged.stage3_arms(spec, fixed_lock, slaclip_lock_fixture(fixed_lock))
    assert len(stage3) == 80
    assert len({arm["arm_id"] for arm in stage3}) == 80
    assert {arm["seed"] for arm in stage3} == set(range(200, 220))
    assert {arm["seed"] for arm in stage3}.isdisjoint(
        set(spec["fixed_development"]["seeds"])
    )
    for index in range(0, len(stage3), 2):
        fixed, adaptive = stage3[index : index + 2]
        assert fixed["method"] == full.FIXED_DP_METHOD
        assert adaptive["method"] == full.FULL_SLACLIP_METHOD
        assert adaptive["reference_arm_id"] == fixed["arm_id"]
        assert adaptive["seed"] == fixed["seed"]
        assert adaptive["models"] == fixed["models"]
        assert adaptive["rng_domain"] == fixed["rng_domain"]


def test_beta_q10_q90_grid_and_degenerate_fail_closed() -> None:
    values = [index / 100.0 for index in range(101)]
    grid, low, high = staged.derive_beta_grid(values)
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.9)
    assert grid == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])
    with pytest.raises(RuntimeError, match="degenerate"):
        staged.derive_beta_grid([0.6] * 250)
    with pytest.raises(RuntimeError, match="invalid"):
        staged.derive_beta_grid([0.1, 1.1])


def test_fixed_trajectory_maps_to_conditional_full_slaclip_beta(
    tmp_path: Path,
) -> None:
    arm = {"arm_id": "calibration-arm", "models": ["bert"], "seed": 110}
    shard_path = staged._round_shard_path(tmp_path, arm, 1)
    shard_path.parent.mkdir(parents=True, mode=0o700)
    raw_norms = [0.0, 1.0, 2.0, 12.0]
    full.atomic_json(
        shard_path,
        {
            "round": 1,
            "model": "bert",
            "method": full.FIXED_DP_METHOD,
            "client_records": [
                {"gradient_groups": {"B": {"raw_norm": norm}}}
                for norm in raw_norms
            ],
            "round_summary": {
                "B": {"clipped_fraction": 0.25},
                "any_group_clipped_fraction": 0.25,
            },
        },
    )
    values, rows = staged._fixed_beta_calibration(
        tmp_path,
        [arm],
        clip_norm=10.0,
        num_slots=5,
        epsilon=1e-6,
        rounds=1,
    )
    # At K=5, the final exact CDF endpoint is 1, 1/2, 0, 0 across
    # these four records, hence 0.375 after public-count normalization.
    expected_z = 0.375 / (10.0 + 1e-6)
    expected_beta = 0.25 / (1.0 - expected_z)
    assert values == pytest.approx([expected_beta])
    assert rows[0]["exact_normalized_slack_endpoint_K"] == pytest.approx(0.375)
    assert rows[0]["near_zero_adjusted_z"] == pytest.approx(expected_z)
    assert rows[0]["remaining_non_small_gradient_fraction"] == pytest.approx(
        1.0 - expected_z
    )
    assert rows[0]["conditional_beta"] == pytest.approx(expected_beta)


def test_initial_C_boundary_policy_uses_same_direction_neighbours() -> None:
    grid = official_spec()["fixed_development"]["clip_norm_grid"]
    assert staged._c0_candidates(grid, 0.1) == ([0.1, 0.3, 0.5], True)
    assert staged._c0_candidates(grid, 10.0) == ([5.0, 7.0, 10.0], True)
    assert staged._c0_candidates(grid, 3.0) == ([2.5, 3.0, 4.0], False)


def test_dynamic_manifests_are_hash_bound_to_their_parent_locks(
    tmp_path: Path,
) -> None:
    spec = official_spec()
    input_manifest = tmp_path / "input-manifest.json"
    input_manifest.write_text("{}\n", encoding="utf-8")
    fixed_lock = fixed_lock_fixture(spec)
    slaclip_lock = slaclip_lock_fixture(fixed_lock)
    shared = {
        "spec": spec,
        "spec_path": SPEC_PATH,
        "repository_sha": "1" * 40,
        "input_manifest": input_manifest,
        "created_at_utc": "2026-08-05T00:00:00+00:00",
    }
    stage2 = staged._runtime_manifest(
        **shared,
        arms=staged.stage2_arms(spec, fixed_lock),
        stage=staged.SLACLIP_STAGE,
        parent_lock_sha256=fixed_lock["lock_sha256"],
    )
    stage3 = staged._runtime_manifest(
        **shared,
        arms=staged.stage3_arms(spec, fixed_lock, slaclip_lock),
        stage=staged.CONFIRMATION_STAGE,
        parent_lock_sha256=slaclip_lock["lock_sha256"],
    )
    assert stage2["parent_selection_lock_sha256"] == "a" * 64
    assert stage3["parent_selection_lock_sha256"] == "b" * 64
    assert stage2["manifest_sha256"] != stage3["manifest_sha256"]

    # A rewritten manifest can be internally self-consistent, but it must not
    # replace the deterministic candidate derived from the immutable lock.
    tampered = json.loads(json.dumps(stage3))
    tampered["arms"][0]["initial_clip_norm"] = 9.0
    tampered_without_hash = dict(tampered)
    tampered_without_hash.pop("manifest_sha256")
    tampered["manifest_sha256"] = full.sha256_bytes(
        full.canonical_bytes(tampered_without_hash)
    )
    path = tmp_path / staged.STAGE3_RUNTIME_NAME
    full.atomic_json(path, tampered)
    full.load_runtime(path)
    with pytest.raises(RuntimeError, match="differs"):
        staged._write_or_verify(path, stage3, "Stage 3 runtime manifest")


def test_preregistered_success_rule_and_diagnostic_schema_are_fixed(
    tmp_path: Path,
) -> None:
    value = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    value["confirmation"]["success_rule"]["minimum_seed_win_fraction"] = 0.6
    changed = tmp_path / "changed-spec.json"
    changed.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="success rule"):
        staged.load_spec(changed)
    assert {
        "privacy_label",
        "controller_input",
        "actual_clipped_fraction",
        "dynamic_target_clipped",
        "noisy_dynamic_target_clipped",
        "oracle_dynamic_target_clipped",
        "cdf_error_mae",
        "noisy_cdf_out_of_range_count",
        "noisy_adjacent_monotonicity_violations",
        "round_total_elapsed_seconds",
        "cuda_max_memory_allocated_bytes",
    } <= set(staged.TRAJECTORY_COLUMNS)


def test_lock_fingerprint_and_existing_file_are_immutable(tmp_path: Path) -> None:
    lock = staged._lock_payload(
        {"schema_version": staged.LOCK_SCHEMA_VERSION, "status": "LOCKED"}
    )
    staged._validate_lock(lock, "test lock")
    changed = dict(lock)
    changed["status"] = "CHANGED"
    with pytest.raises(RuntimeError, match="fingerprint"):
        staged._validate_lock(changed, "test lock")
    path = tmp_path / "selection.lock.json"
    staged._write_or_verify(path, lock, "test lock")
    staged._write_or_verify(path, lock, "test lock")
    with pytest.raises(RuntimeError, match="differs"):
        staged._write_or_verify(path, staged._lock_payload({"x": 1}), "test lock")


def test_resume_with_existing_fixed_lock_does_not_reselect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = official_spec()
    input_manifest = tmp_path / "input.json"
    input_manifest.write_text("{}\n", encoding="utf-8")
    campaign = tmp_path / "campaign"
    campaign.mkdir(mode=0o700)
    expected_sha = "1" * 40
    master = staged._build_master_runtime(
        spec,
        SPEC_PATH,
        repository_sha=expected_sha,
        input_manifest=input_manifest,
        created_at_utc="2026-08-05T00:00:00+00:00",
    )
    full.atomic_json(campaign / staged.MASTER_RUNTIME_NAME, master)
    full.atomic_json(
        campaign / staged.PREFLIGHT_RUNTIME_NAME,
        staged._preflight_manifest(master, spec),
    )
    lock_models = fixed_lock_fixture(spec)["models"]
    lock = staged._lock_payload(
        {
            "schema_version": staged.LOCK_SCHEMA_VERSION,
            "status": "FIXED_DEVELOPMENT_SELECTION_LOCKED",
            "campaign_name": master["campaign_name"],
            "master_runtime_manifest_sha256": master["manifest_sha256"],
            "stage1_runtime_manifest_sha256": master["manifest_sha256"],
            "spec_sha256": master["spec_sha256"],
            "selection_rule": spec["fixed_development"]["selection_rule"],
            "development_seeds": spec["fixed_development"]["seeds"],
            "confirmation_data_accessed": False,
            "models": lock_models,
            "source_evidence": [],
            "beta_calibration_csv_sha256": "c" * 64,
            "created_at_utc": "2026-08-05T00:00:01+00:00",
        }
    )
    full.atomic_json(campaign / staged.FIXED_LOCK_NAME, lock)
    for name in ("arms", "arm-status", "arm-logs", "control", "tmp", "preflight", "selection"):
        (campaign / name).mkdir(mode=0o700, exist_ok=True)

    monkeypatch.setattr(full, "repository_sha", lambda _path: expected_sha)
    monkeypatch.setattr(full, "repository_dirty", lambda _path: False)
    monkeypatch.setattr(full, "validate_or_create_key", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(staged, "_verify_locked_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        staged, "_verify_beta_calibration_artifact", lambda *_args, **_kwargs: None
    )
    materialized: list[str] = []
    monkeypatch.setattr(
        staged,
        "_ensure_stage2_manifest",
        lambda *_args, **_kwargs: materialized.append("stage2") or campaign / staged.STAGE2_RUNTIME_NAME,
    )
    monkeypatch.setattr(
        staged,
        "derive_beta_grid",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume reselected beta")),
    )
    staged.prepare_campaign(
        argparse.Namespace(
            campaign_root=campaign,
            repository=ROOT,
            spec=SPEC_PATH,
            input_manifest=input_manifest,
            expected_code_sha=expected_sha,
            private_key=tmp_path / "key",
            resume=True,
        )
    )
    assert materialized == ["stage2"]


def test_confirmation_pairing_holm_and_success_rule() -> None:
    spec = official_spec()
    fixed_lock = fixed_lock_fixture(spec)
    confirmation = {
        "arms": staged.stage3_arms(spec, fixed_lock, slaclip_lock_fixture(fixed_lock))
    }
    metric_rows = []
    for arm in confirmation["arms"]:
        adaptive = arm["method"] == full.FULL_SLACLIP_METHOD
        metric_rows.append(
            {
                "arm_id": arm["arm_id"],
                "model": arm["models"][0],
                "seed": arm["seed"],
                "initial_clip_norm": arm["initial_clip_norm"],
                "slaclip_beta": arm["slaclip_beta"],
                "final_loss": 0.9 if adaptive else 1.0,
                "best_loss": 0.8 if adaptive else 0.9,
                "normalized_loss_auc": 0.85 if adaptive else 0.95,
                "actual_clipped_fraction": 0.6 if adaptive else 0.5,
                "elapsed_seconds": 2.0 if adaptive else 1.0,
                "sample_schedule_sha256": f"{arm['models'][0]}-{arm['seed']}-samples",
                "supervision_schedule_sha256": f"{arm['models'][0]}-{arm['seed']}-masks",
                "pair_noise_across_methods": True,
                "rng_domain": arm["rng_domain"],
                "private_key_commitment": "c" * 64,
            }
        )
    paired = staged._paired_confirmation_rows(metric_rows, confirmation)
    assert len(paired) == 40
    aggregate = staged._confirmation_aggregate(paired, spec)
    assert len(aggregate) == 2
    for row in aggregate:
        assert row["seed_count"] == 20
        assert row["final_loss_difference_slaclip_minus_fixed_mean"] == pytest.approx(-0.1)
        assert row["final_loss_difference_slaclip_minus_fixed_holm_p"] < 0.05
        assert row["primary_success"] is True

    mismatched = [dict(row) for row in metric_rows]
    adaptive_row = next(
        row for row in mismatched if row["arm_id"].startswith("confirm-slaclip-")
    )
    adaptive_row["private_key_commitment"] = "d" * 64
    with pytest.raises(RuntimeError, match="private_key_commitment"):
        staged._paired_confirmation_rows(mismatched, confirmation)


def test_staged_archive_inventory_contains_control_plane(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "archive"
    source.mkdir(mode=0o700)
    names = {
        "preflight-runtime-manifest.json",
        "fixed-selection.lock.json",
        "slaclip-selection.lock.json",
        "stage2-runtime-manifest.json",
        "stage3-runtime-manifest.json",
        "fixed_beta_calibration.csv",
        "fixed_trajectory.csv",
        "slaclip_trajectory.csv",
        "confirmation_paired_metrics.csv",
        "confirmation_aggregate_metrics.csv",
    }
    for name in names:
        (source / name).write_text("{}\n", encoding="utf-8")
    full.archive_small(
        argparse.Namespace(campaign_root=source, archive_root=destination)
    )
    inventory = json.loads((destination / "archive-inventory.json").read_text())
    archived = {item["path"] for item in inventory["files"]}
    assert names <= archived


def test_worker_is_one_sbatch_without_arrays_or_child_submissions() -> None:
    worker = WORKER_PATH.read_text(encoding="utf-8")
    submit = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "#SBATCH --array" not in worker
    assert "sbatch " not in worker
    assert "--array" not in submit
    assert submit.count('"$sbatch_bin" "${sbatch_args[@]}" "$worker"') == 1
    assert "DPLORA_STAGED_PARTITION:-scavenger_4a100" in submit
    assert "DPLORA_STAGED_GPU_GRES:-gpu:a100swarm:2" in submit
    assert "DPLORA_STAGED_WALLTIME:-12:00:00" in submit
    assert 'a100|a100swarm) default_expected_gpu="A100"; default_min_vram_gib=39' in submit
    subprocess.run(["bash", "-n", str(WORKER_PATH), str(SUBMIT_PATH)], check=True)
