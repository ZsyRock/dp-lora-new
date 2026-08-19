from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from paper_repro.broad_scope_campaign import (
    analyse_clipping_regimes,
    derive_adaptive_plan,
    expand_fixed_development,
    validate_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "hpc" / "broad-scope-full-slaclip-spec.json"


def spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_broad_scope_covers_all_paper_domains_and_models_without_slaclip_q() -> None:
    value = validate_spec(spec())
    assert {item["id"] for item in value["domains"]} == {
        "meddialog",
        "slimpajama",
        "finance",
    }
    assert {item["id"] for item in value["models"]} == {
        "bert",
        "gpt2",
        "chatglm2",
        "llama2",
    }
    assert "slaclip_q" not in json.dumps(value).lower()
    arms = expand_fixed_development(value)
    assert len(arms) == 3 * 4 * 5 * 3
    assert {arm["method"] for arm in arms} == {"paper_dp_lora"}


def test_confirmation_seeds_must_be_disjoint() -> None:
    value = spec()
    value["confirmation"]["seeds"][0] = value["fixed_development"]["seeds"][0]
    with pytest.raises(ValueError, match="disjoint"):
        validate_spec(value)


def _baseline_rows(value: dict) -> list[dict[str, str]]:
    rows = []
    for domain in value["domains"]:
        for model in value["models"]:
            for clip_norm in value["fixed_development"]["clip_norms"]:
                for seed in value["fixed_development"]["seeds"]:
                    for round_index in (1, 2, 3):
                        # C=1 is deliberately the best utility setting.
                        loss = abs(float(clip_norm) - 1.0) + seed * 1e-8
                        rows.append(
                            {
                                "domain": domain["id"],
                                "model": model["id"],
                                "method": "paper_dp_lora",
                                "clip_norm": str(clip_norm),
                                "seed": str(seed),
                                "round": str(round_index),
                                "final_loss": str(loss),
                                "actual_clipped_fraction_A": str(
                                    [0.2, 0.4, 0.6][round_index - 1]
                                ),
                                "actual_clipped_fraction_B": str(
                                    [0.3, 0.5, 0.7][round_index - 1]
                                ),
                                "near_zero_adjusted_fraction_A": "0.2",
                                "near_zero_adjusted_fraction_B": "0.25",
                            }
                        )
    return rows


def test_adaptive_targets_are_derived_after_near_zero_mass_is_removed() -> None:
    value = spec()
    plan = derive_adaptive_plan(value, _baseline_rows(value))
    assert plan["selection_count"] == 12
    assert plan["adaptive_arm_count"] == 12 * 5 * 3 * 3
    selection = plan["selections"][0]
    assert selection["selected_fixed_clip_norm"] == 1.0
    middle_a = selection["group_targets"]["A"][2]
    assert middle_a["total_clipped_fraction_target"] == pytest.approx(0.4)
    assert middle_a["base_target_clipped_fraction_beta"] == pytest.approx(0.5)


def test_adaptive_target_keeps_near_zero_endpoint_paired_with_clip_quantile() -> None:
    value = spec()
    rows = _baseline_rows(value)
    for row in rows:
        if float(row["clip_norm"]) == 1.0:
            clipped = float(row["actual_clipped_fraction_A"])
            row["near_zero_adjusted_fraction_A"] = str(clipped / 2)
    plan = derive_adaptive_plan(value, rows)
    middle = plan["selections"][0]["group_targets"]["A"][2]
    assert middle["total_clipped_fraction_target"] == pytest.approx(0.4)
    assert middle["near_zero_adjusted_fraction_at_target"] == pytest.approx(0.2)
    assert middle["base_target_clipped_fraction_beta"] == pytest.approx(0.5)


def test_adaptive_derivation_fails_without_near_zero_telemetry() -> None:
    value = spec()
    rows = _baseline_rows(value)
    del rows[0]["near_zero_adjusted_fraction_A"]
    with pytest.raises(ValueError, match="near_zero_adjusted_fraction_A"):
        derive_adaptive_plan(value, rows)


def test_regime_bins_require_multiple_settings_and_fresh_seed_pairs() -> None:
    value = spec()
    rows = []
    for domain, model in (("meddialog", "bert"), ("slimpajama", "gpt2")):
        for seed in value["confirmation"]["seeds"]:
            rows.append(
                {
                    "domain": domain,
                    "model": model,
                    "seed": str(seed),
                    "baseline_actual_clipped_fraction": "0.4",
                    "final_loss_difference_slaclip_minus_fixed": "-0.01",
                }
            )
    bins = analyse_clipping_regimes(value, rows)
    target = next(item for item in bins if item.lower == 0.3)
    assert target.pair_count == 20
    assert target.setting_count == 2
    assert target.win_fraction == 1.0
    assert target.evidence_gate_passed is True
