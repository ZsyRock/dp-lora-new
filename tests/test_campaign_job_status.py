from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from paper_repro.full_slaclip_campaign import (
    JOB_STATUS_NAME,
    _archive_candidate,
    atomic_json,
    canonical_bytes,
    mark_job_status,
    sha256_bytes,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SLURM_WORKER = REPOSITORY / "hpc" / "full_slaclip_campaign.sbatch"
REPOSITORY_SHA = "a" * 40


def _prepared_campaign(tmp_path: Path) -> tuple[Path, Path]:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    runtime = {
        "schema_version": 1,
        "campaign_name": "job-status-test",
        "created_at_utc": "2026-07-30T00:00:00+00:00",
        "repository_sha": REPOSITORY_SHA,
        "spec_sha256": "b" * 64,
        "input_manifest_path": "/immutable/input-manifest.json",
        "input_manifest_sha256": "c" * 64,
        "expected_arm_count": 0,
        "scientific_boundary": {"claim_level": 1},
        "arms": [],
    }
    runtime["manifest_sha256"] = sha256_bytes(canonical_bytes(runtime))
    runtime_path = campaign_root / "runtime-manifest.json"
    atomic_json(runtime_path, runtime)
    return campaign_root, runtime_path


def _mark(
    campaign_root: Path,
    runtime_path: Path,
    *,
    status: str,
    job_id: str,
    reason: str,
    exit_code: int | None = None,
) -> None:
    mark_job_status(
        argparse.Namespace(
            campaign_root=campaign_root,
            runtime_manifest=runtime_path,
            status=status,
            slurm_job_id=job_id,
            repository_sha=REPOSITORY_SHA,
            reason=reason,
            exit_code=exit_code,
        )
    )


def test_job_status_records_running_and_completed_atomically(tmp_path: Path) -> None:
    campaign_root, runtime_path = _prepared_campaign(tmp_path)
    _mark(
        campaign_root,
        runtime_path,
        status="RUNNING",
        job_id="12345",
        reason="allocation_started",
    )
    running = json.loads((campaign_root / JOB_STATUS_NAME).read_text(encoding="utf-8"))
    assert running["status"] == "RUNNING"
    assert running["attempt_number"] == 1
    assert running["exit_code"] is None
    assert running["terminal_at_utc"] is None

    _mark(
        campaign_root,
        runtime_path,
        status="COMPLETED",
        job_id="12345",
        reason="all_arms_revalidated_and_archived",
        exit_code=0,
    )
    completed = json.loads(
        (campaign_root / JOB_STATUS_NAME).read_text(encoding="utf-8")
    )
    assert completed["status"] == "COMPLETED"
    assert completed["attempt_number"] == 1
    assert completed["started_at_utc"] == running["started_at_utc"]
    assert completed["exit_code"] == 0
    assert completed["resumable"] is False


def test_checkpoint_failure_is_resumable_and_resume_increments_attempt(
    tmp_path: Path,
) -> None:
    campaign_root, runtime_path = _prepared_campaign(tmp_path)
    _mark(
        campaign_root,
        runtime_path,
        status="RUNNING",
        job_id="100",
        reason="allocation_started",
    )
    _mark(
        campaign_root,
        runtime_path,
        status="FAILED",
        job_id="100",
        reason="checkpointed_stop_at_formal_matrix",
        exit_code=75,
    )
    failed = json.loads((campaign_root / JOB_STATUS_NAME).read_text(encoding="utf-8"))
    assert failed["status"] == "FAILED"
    assert failed["resumable"] is True

    _mark(
        campaign_root,
        runtime_path,
        status="RUNNING",
        job_id="101",
        reason="allocation_started",
    )
    resumed = json.loads((campaign_root / JOB_STATUS_NAME).read_text(encoding="utf-8"))
    assert resumed["status"] == "RUNNING"
    assert resumed["attempt_number"] == 2
    assert resumed["slurm_job_id"] == "101"


def test_stale_allocation_cannot_overwrite_new_owner(tmp_path: Path) -> None:
    campaign_root, runtime_path = _prepared_campaign(tmp_path)
    _mark(
        campaign_root,
        runtime_path,
        status="RUNNING",
        job_id="200",
        reason="allocation_started",
    )
    with pytest.raises(RuntimeError, match="owning RUNNING allocation"):
        _mark(
            campaign_root,
            runtime_path,
            status="FAILED",
            job_id="199",
            reason="late_exit",
            exit_code=1,
        )


def test_job_status_is_included_in_incremental_archive(tmp_path: Path) -> None:
    campaign_root, _ = _prepared_campaign(tmp_path)
    status_path = campaign_root / JOB_STATUS_NAME
    status_path.write_text("{}\n", encoding="utf-8")
    assert _archive_candidate(status_path, campaign_root)


def test_slurm_exit_trap_preserves_failure_and_marks_terminal_state() -> None:
    worker = SLURM_WORKER.read_text(encoding="utf-8")
    assert "local original_rc=$?" in worker
    assert 'trap finalize_campaign EXIT' in worker
    assert '--status RUNNING' in worker
    assert '--status "$terminal_status"' in worker
    assert '--exit-code "$final_rc"' in worker
    assert 'exit "$final_rc"' in worker
    assert worker.index('trap finalize_campaign EXIT') < worker.index(
        'campaign_stage="runtime_and_input_preflight"'
    )
