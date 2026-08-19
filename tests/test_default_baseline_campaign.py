from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_repro.default_baseline_campaign import expand_arms, validate_spec


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "hpc" / "default-baseline-reproduction-spec.json"


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
