from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from paper_repro import baseline_followup_campaign as campaign
from paper_repro import full_slaclip_campaign as full


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "hpc" / "baseline-followup-fixed-spec.json"


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _upstream(tmp_path: Path) -> tuple[Path, Path]:
    upstream = tmp_path / "upstream"
    telemetry = upstream / "telemetry"
    manifests: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    for domain in campaign.DOMAINS:
        manifest = tmp_path / "inputs" / domain / "input-manifest.json"
        inventory_sha = (str(campaign.DOMAINS.index(domain) + 3) * 64)[:64]
        _json(manifest, {
            "schema_version": 2, "domain": domain,
            "inventory_sha256": inventory_sha,
        })
        manifests[domain] = str(manifest)
        manifest_hashes[domain] = full.sha256_file(manifest)
    input_index = tmp_path / "inputs" / "input-index.json"
    _json(input_index, {"schema_version": 1, "domains": manifests})

    spec_sha = "1" * 64
    integrity_arms = []
    for domain in campaign.DOMAINS:
        for model in campaign.MODELS:
            for seed in campaign.SOURCE_SEEDS:
                integrity_arms.append({
                    "domain": domain,
                    "model": model,
                    "seed": seed,
                    "arm_id": f"paper-default-{domain}-{model}-s{seed}-fixed-c10",
                    "rng_domain": f"paper-default-{domain}-{model}-s{seed}",
                    "run_config_fingerprint": (
                        f"{campaign.DOMAINS.index(domain) + 3:x}"
                        f"{campaign.MODELS.index(model) + 6:x}"
                        f"{seed:x}" * 32
                    )[:64],
                    "method": campaign.FIXED_METHOD,
                    "status": "COMPLETED",
                    "client_steps": 250,
                    "repository_sha": "2" * 40,
                    "input_manifest_sha256": manifest_hashes[domain],
                    "input_inventory_sha256": (
                        str(campaign.DOMAINS.index(domain) + 3) * 64
                    )[:64],
                })
    arm_integrity = telemetry / "baseline_arm_integrity.json"
    _json(arm_integrity, {
        "status": "COMPLETE", "spec_sha256": spec_sha, "arms": integrity_arms,
    })

    client_rows: list[dict[str, object]] = []
    for domain_index, domain in enumerate(campaign.DOMAINS):
        for model_index, model in enumerate(campaign.MODELS):
            for seed in campaign.SOURCE_SEEDS:
                for round_index in range(1, 51):
                    for client in range(5):
                        for group_index, group in enumerate(campaign.GROUPS):
                            # Large late-round values make an accidental use of
                            # rounds 11--50 immediately visible in quantiles.
                            raw = (
                                1000.0 + round_index
                                if round_index > 10
                                else domain_index * 100 + model_index * 20
                                + group_index * 5 + round_index + client / 10
                                + (seed - 1200) / 100
                            )
                            client_rows.append({
                                "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
                                "arm_id": f"paper-default-{domain}-{model}-s{seed}-fixed-c10",
                                "rng_domain": f"paper-default-{domain}-{model}-s{seed}",
                                "domain": domain, "model": model, "seed": seed,
                                "method": campaign.FIXED_METHOD,
                                "run_config_fingerprint": (
                                    f"{campaign.DOMAINS.index(domain) + 3:x}"
                                    f"{campaign.MODELS.index(model) + 6:x}"
                                    f"{seed:x}" * 32
                                )[:64],
                                "repository_sha": "2" * 40,
                                "input_manifest_sha256": manifest_hashes[domain],
                                "input_inventory_sha256": (
                                    str(campaign.DOMAINS.index(domain) + 3) * 64
                                )[:64],
                                "round": round_index, "client": client,
                                "group": group, "clip_norm": 10.0,
                                "raw_norm": raw,
                            })
    _csv(
        telemetry / "baseline_client_telemetry.csv",
        [
            "privacy_label", "arm_id", "rng_domain", "domain", "model",
            "seed", "method", "run_config_fingerprint", "repository_sha",
            "input_manifest_sha256", "input_inventory_sha256", "round",
            "client", "group", "clip_norm", "raw_norm",
        ],
        client_rows,
    )
    round_rows = [
        {"row": index} for index in range(2700)
    ]
    evaluation_rows = [
        {"row": index} for index in range(162)
    ]
    _csv(telemetry / "baseline_round_telemetry.csv", ["row"], round_rows)
    _csv(telemetry / "baseline_evaluation_telemetry.csv", ["row"], evaluation_rows)
    _json(upstream / "campaign-summary.json", {
        "arm_count": 27,
        "completed_arm_count": 27,
        "smoke": False,
        "spec_sha256": spec_sha,
        "status": "COMPLETED",
    })
    _json(telemetry / "baseline_telemetry_manifest.json", {
        "status": "COMPLETE",
        "spec_sha256": spec_sha,
        "round_rows": 2700,
        "client_rows": 13500,
        "evaluation_rows": 162,
        "arm_integrity_rows": 27,
        "arm_integrity_sha256": full.sha256_file(arm_integrity),
        "privacy_label": "NON_DP_PRIVATE_DIAGNOSTIC",
        "initialization_transient_policy": "round 1 excluded",
    })
    return upstream, input_index


def test_spec_is_fixed_only_and_has_exact_resource_sized_matrix() -> None:
    spec = campaign.load_spec(SPEC)
    assert spec["expected_arm_count"] == 180
    assert spec["source_calibration"]["round_min"] == 2
    assert spec["source_calibration"]["round_max"] == 10
    assert spec["development"]["seeds"] == [1300, 1301]
    assert spec["scientific_boundary"]["full_slaclip_run"] is False
    assert spec["scientific_boundary"]["slaclip_q_run"] is False


def test_plan_locks_all_sources_and_uses_only_rounds_2_to_10(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream, input_index = _upstream(tmp_path)
    key = tmp_path / "private.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    monkeypatch.setattr(full, "repository_sha", lambda _path: "a" * 40)
    monkeypatch.setattr(full, "repository_dirty", lambda _path: False)
    plan = campaign.build_plan(
        spec_path=SPEC,
        repository=ROOT,
        expected_code_sha="a" * 40,
        upstream_root=upstream,
        input_index=input_index,
        private_key=key,
        created_at_utc="2026-08-23T00:00:00+00:00",
    )
    assert len(plan["arms"]) == 180
    assert len(plan["settings"]) == 9
    assert set(plan["upstream"]["source_files"]) == set(campaign.SOURCE_FILES)
    assert all(len(item["source_files"]["client_telemetry"]["sha256"]) == 64 for item in [plan["upstream"]])
    first = plan["settings"][0]
    assert first["candidate_count"] == 10
    assert first["candidates"][-1]["clip_norm_by_group"] == {"A": 10.0, "B": 10.0}
    # The selected-window source values are below 11; rounds 11--50 were >1000.
    assert max(first["source_threshold_quantiles_by_group"]["A"].values()) < 11
    grouped = {}
    for arm in plan["arms"]:
        grouped.setdefault((arm["domain"], arm["model"], arm["seed"]), set()).add(
            arm["rng_domain"]
        )
        assert arm["method"] == "paper_dp_lora"
        assert not any("slaclip" in key.lower() for key in arm)
    assert all(len(values) == 1 for values in grouped.values())


def test_locked_plan_detects_any_upstream_csv_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream, input_index = _upstream(tmp_path)
    key = tmp_path / "private.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    monkeypatch.setattr(full, "repository_sha", lambda _path: "a" * 40)
    monkeypatch.setattr(full, "repository_dirty", lambda _path: False)
    plan = campaign.build_plan(
        spec_path=SPEC, repository=ROOT, expected_code_sha="a" * 40,
        upstream_root=upstream, input_index=input_index, private_key=key,
        created_at_utc="2026-08-23T00:00:00+00:00",
    )
    plan_path = tmp_path / "runtime-manifest.json"
    _json(plan_path, plan)
    round_csv = upstream / "telemetry" / "baseline_round_telemetry.csv"
    round_csv.write_text(round_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable inputs"):
        campaign._plan_identity(
            plan_path=plan_path, spec_path=SPEC, repository=ROOT,
            expected_code_sha="a" * 40, upstream_root=upstream,
            input_index=input_index, private_key=key,
        )


def test_fixed_arm_command_has_groupwise_thresholds_and_no_adaptive_option(
    tmp_path: Path,
) -> None:
    arm = {
        "model": "bert", "method": "paper_dp_lora", "num_clients": 5,
        "rounds": 50, "batch_size": 8, "noise_multiplier": 2.0,
        "learning_rate": 5e-4, "initial_clip_norm": 3.0,
        "initial_clip_norm_by_group": {"A": 1.0, "B": 3.0},
        "rank": 512, "max_seq_length": 128, "max_validation_records": 512,
        "seed": 1300, "data_split_seed": 1729, "evaluation_seed": 2718,
        "delta": 1e-5, "eval_every": 10, "checkpoint_every": 10,
        "rng_domain": "baseline-followup:meddialog:bert:s1300",
    }
    command = campaign._arm_command(
        arm, repository=ROOT, python_bin=Path("/usr/bin/python3"),
        input_manifest=tmp_path / "input.json", output_dir=tmp_path / "output",
        private_key=tmp_path / "key", stop_file=tmp_path / "stop", smoke=False,
    )
    assert command[command.index("--clip-norm-a") + 1] == "1.0"
    assert command[command.index("--clip-norm-b") + 1] == "3.0"
    assert "--pair-noise-across-methods" in command
    assert "--checkpoint-every" in command
    assert command[command.index("--checkpoint-every") + 1] == "10"
    assert not any("slaclip" in item.lower() for item in command)


def test_candidate_ranking_freezes_all_candidates_and_top_three() -> None:
    rows = []
    for candidate in range(10):
        for seed in (1300, 1301):
            rows.append({
                "C_A": float(candidate + 1), "C_B": float(candidate + 2),
                "seed": seed, "final_loss": float(candidate) + seed * 1e-6,
                "normalized_loss_auc": float(candidate) + 0.5,
                "actual_clipped_fraction": 0.5,
            })
    rankings = campaign._candidate_rankings(
        rows, seeds=[1300, 1301], noise_multiplier=2.0
    )
    assert len(rankings) == 10
    assert [(item["C_A"], item["C_B"]) for item in rankings[:3]] == [
        (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)
    ]
    assert all(len(item["seed_evidence"]) == 2 for item in rankings)


def test_plan_validation_rejects_an_adaptive_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream, input_index = _upstream(tmp_path)
    key = tmp_path / "private.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    monkeypatch.setattr(full, "repository_sha", lambda _path: "a" * 40)
    monkeypatch.setattr(full, "repository_dirty", lambda _path: False)
    plan = campaign.build_plan(
        spec_path=SPEC, repository=ROOT, expected_code_sha="a" * 40,
        upstream_root=upstream, input_index=input_index, private_key=key,
        created_at_utc="2026-08-23T00:00:00+00:00",
    )
    arm = plan["arms"][0]
    arm["slaclip_eta"] = 0.1
    payload = {name: value for name, value in arm.items() if name != "arm_sha256"}
    arm["arm_sha256"] = full.sha256_bytes(full.canonical_bytes(payload))
    plan_payload = {name: value for name, value in plan.items() if name != "manifest_sha256"}
    plan["manifest_sha256"] = full.sha256_bytes(full.canonical_bytes(plan_payload))
    with pytest.raises(RuntimeError, match="fixed-only"):
        campaign.validate_plan(plan, campaign.load_spec(SPEC))


def test_locked_input_manifest_is_rechecked_between_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream, input_index = _upstream(tmp_path)
    key = tmp_path / "private.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    monkeypatch.setattr(full, "repository_sha", lambda _path: "a" * 40)
    monkeypatch.setattr(full, "repository_dirty", lambda _path: False)
    plan = campaign.build_plan(
        spec_path=SPEC, repository=ROOT, expected_code_sha="a" * 40,
        upstream_root=upstream, input_index=input_index, private_key=key,
        created_at_utc="2026-08-23T00:00:00+00:00",
    )
    arm = plan["arms"][0]
    manifest = campaign._validate_locked_input_manifest(plan, arm)
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked input manifest changed"):
        campaign._validate_locked_input_manifest(plan, arm)


def test_completed_reuse_requires_matching_status_and_summary_hash(
    tmp_path: Path,
) -> None:
    arm = {
        "arm_id": "arm-1", "arm_sha256": "a" * 64, "model": "bert",
    }
    plan = {"manifest_sha256": "b" * 64}
    summary_path = tmp_path / "final_summary.json"
    status_path = tmp_path / "status.json"
    _json(summary_path, {
        "status": "COMPLETED", "method": campaign.FIXED_METHOD,
        "run_config_fingerprint": "c" * 64,
        "models": {"bert": {
            "status": "COMPLETED", "method": campaign.FIXED_METHOD,
            "run_config_fingerprint": "c" * 64,
        }},
    })
    _json(status_path, {
        "schema_version": 1, "status": "RUNNING", "smoke": False,
        "arm_id": "arm-1", "arm_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "final_summary_sha256": full.sha256_file(summary_path),
    })
    assert campaign._verified_completed_status(
        status_path, summary_path, arm, plan, smoke=False
    ) is None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "COMPLETED"
    _json(status_path, status)
    assert campaign._verified_completed_status(
        status_path, summary_path, arm, plan, smoke=False
    ) is not None
    summary_path.write_text(
        summary_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert campaign._verified_completed_status(
        status_path, summary_path, arm, plan, smoke=False
    ) is None


def test_metrics_file_is_cryptographically_bound_to_selection_lock(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / campaign.METRICS_NAME
    metrics.write_text("arm_id,final_loss\na,1.0\n", encoding="utf-8")
    lock = campaign._bind_metrics_to_lock(
        {"schema_version": 1, "status": "LOCKED"}, metrics, 1
    )
    path = tmp_path / campaign.SELECTION_LOCK_NAME
    _json(path, lock)
    assert campaign._validate_selection_artifacts(path, metrics, 1) == lock
    metrics.write_text("arm_id,final_loss\na,2.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="metrics identity"):
        campaign._validate_selection_artifacts(path, metrics, 1)


def test_test_only_submission_uses_ephemeral_sandbox() -> None:
    wrapper = (ROOT / "hpc" / "submit_baseline_followup_fixed.sh").read_text(
        encoding="utf-8"
    )
    assert "--test-only and --resume cannot be combined" in wrapper
    assert 'mktemp -d "$scratch_root/tmp/dp-lora-followup-test.XXXXXXXX"' in wrapper
    assert 'rm -rf -- "$test_sandbox"' in wrapper
    assert wrapper.index('test_sandbox=""') < wrapper.index(
        'mkdir -p "$run_root" "$slurm_root" "$private_key_root"'
    )
