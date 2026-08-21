from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_repro.default_baseline_campaign import (
    RUNTIME_PACKAGES,
    evaluation_telemetry_row,
    expand_arms,
    parse_runtime_lock,
    validate_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "hpc" / "default-baseline-reproduction-spec.json"
RUNTIME_LOCK = ROOT / "environment" / "paper-repro-runtime.lock"


def value() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def test_default_matrix_has_three_domains_three_stageable_models_three_seeds() -> None:
    spec = validate_spec(value())
    arms = expand_arms(spec)
    assert len(arms) == 27
    assert {arm["domain"] for arm in arms} == {
        "meddialog", "slimpajama", "finance"
    }
    assert {arm["model"] for arm in arms} == {"bert", "gpt2", "chatglm2"}
    assert {arm["method"] for arm in arms} == {"paper_dp_lora"}
    assert {arm["clip_norm"] for arm in arms} == {10.0}
    assert all(arm["num_clients"] == 5 and arm["rounds"] == 50 for arm in arms)


def test_llama_blocker_and_no_slaclip_are_explicit() -> None:
    spec = validate_spec(value())
    assert set(spec["blocked_paper_models"]) == {"llama2"}
    assert spec["scientific_boundary"]["full_slaclip_run"] is False
    assert spec["scientific_boundary"]["slaclip_q_run"] is False


def test_default_hyperparameter_drift_fails_closed() -> None:
    spec = value()
    spec["paper_default"]["clip_norm"] = 1.0
    with pytest.raises(ValueError, match="paper-default"):
        validate_spec(spec)


def test_runtime_lock_covers_chatglm_dynamic_dependency() -> None:
    versions = parse_runtime_lock(RUNTIME_LOCK)
    assert set(versions) == RUNTIME_PACKAGES
    assert versions["sentencepiece"] == "0.2.1"


def test_utility_telemetry_keeps_token_accuracy_counts() -> None:
    row = evaluation_telemetry_row(
        {"arm_id": "example", "model": "gpt2"},
        {
            "round": 5,
            "loss": 2.0,
            "exp_loss": 7.389,
            "records": 4,
            "batches": 1,
            "objective": "causal_lm",
            "supervised_tokens": 100,
            "correct_tokens": 25,
            "token_accuracy": 0.25,
            "token_accuracy_definition": "supervised_token_top1_micro_accuracy",
        },
    )
    assert row["arm_id"] == "example"
    assert row["token_accuracy"] == 0.25
    assert row["correct_tokens"] == 25
    assert row["supervised_tokens"] == 100


def test_submission_validates_dynamic_runtime_before_queueing() -> None:
    submit = (ROOT / "hpc" / "submit_default_baseline_reproduction.sh").read_text()
    worker = (ROOT / "hpc" / "default_baseline_reproduction.sbatch").read_text()
    assert "validate-runtime" in submit
    assert "validate-runtime" in worker
    assert "paper-repro-runtime.lock" in submit
