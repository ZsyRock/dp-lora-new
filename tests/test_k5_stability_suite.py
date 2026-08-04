from __future__ import annotations

from collections import Counter
from pathlib import Path

from paper_repro.full_slaclip_campaign import (
    expand_spec,
    load_spec,
    resolve_oracle_noisy_arm_pairs,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = REPOSITORY / "hpc" / "full-slaclip-k5-stability-suite-spec.json"
SUBMITTER = REPOSITORY / "hpc" / "submit_full_slaclip_k5_stability_suite.sh"


def test_k5_stability_suite_is_one_pre_registered_80_arm_matrix() -> None:
    spec = load_spec(SPEC)
    arms = expand_spec(spec)
    assert len(arms) == spec["expected_arm_count"] == 80
    assert Counter(arm["family"] for arm in arms) == {
        "primary": 20,
        "threshold_robustness": 24,
        "sensitivity": 12,
        "client_sensitivity": 6,
        "control": 6,
        "oracle_control": 12,
    }
    assert Counter(arm["method"] for arm in arms) == {
        "paper_dp_lora": 25,
        "slaclip_dp_lora": 37,
        "oracle_slaclip_control": 12,
        "no_dp_lora_control": 3,
        "clip_only_control": 3,
    }
    assert [arm["index"] for arm in arms] == list(range(80))
    assert {arm["wave"] for arm in arms} == set(range(40))


def test_k5_stability_suite_freezes_primary_before_development_ablations() -> None:
    spec = load_spec(SPEC)
    arms = expand_spec(spec)
    boundary = spec["scientific_boundary"]
    assert boundary["confirmation_seeds_disjoint_from_selection_seeds"] is True
    frozen = boundary["frozen_primary_configuration"]
    assert frozen["num_slots"] == 5
    assert frozen["eta"] == 0.2
    assert frozen["base_target_clipped_fraction"] == 0.76
    primary = [arm for arm in arms if arm["family"] == "primary"]
    assert {arm["seed"] for arm in primary} == set(range(100, 110))
    assert {arm["initial_clip_norm"] for arm in primary} == {10.0}
    adaptive_primary = [arm for arm in primary if arm["method"] == "slaclip_dp_lora"]
    assert {arm["slaclip_num_slots"] for arm in adaptive_primary} == {5}
    assert {arm["slaclip_eta"] for arm in adaptive_primary} == {0.2}
    assert {
        arm["slaclip_base_target_clipped_fraction"] for arm in adaptive_primary
    } == {0.76}
    assert {arm["controller_input"] for arm in adaptive_primary} == {
        "noisy_endpoints"
    }


def test_oracle_controls_are_non_private_diagnostics_with_fixed_references() -> None:
    arms = expand_spec(load_spec(SPEC))
    ids = {arm["arm_id"] for arm in arms}
    oracle = [arm for arm in arms if arm["method"] == "oracle_slaclip_control"]
    assert len(oracle) == 12
    assert {arm["seed"] for arm in oracle} == {100, 101, 102}
    assert {arm["slaclip_eta"] for arm in oracle} == {0.025, 0.05, 0.1, 0.2}
    assert {arm["controller_input"] for arm in oracle} == {"exact_endpoints"}
    assert {arm["analysis_role"] for arm in oracle} == {
        "non_dp_exact_endpoint_oracle_controller_diagnostic"
    }
    assert all(arm["reference_arm_id"] in ids for arm in oracle)
    assert all(
        arm["reference_arm_id"]
        == f"primary-c10-s{arm['seed']}-fixed"
        for arm in oracle
    )


def test_oracle_controls_have_12_exact_noisy_pairs_and_8_model_eta_groups() -> None:
    arms = expand_spec(load_spec(SPEC))
    pairs = resolve_oracle_noisy_arm_pairs({"arms": arms})
    assert len(pairs) == 12
    assert all(oracle["seed"] == noisy["seed"] for oracle, noisy in pairs)
    assert all(
        oracle["initial_clip_norm"] == noisy["initial_clip_norm"]
        and oracle["num_clients"] == noisy["num_clients"]
        and oracle["noise_multiplier"] == noisy["noise_multiplier"]
        and oracle["slaclip_beta"] == noisy["slaclip_beta"]
        and oracle["slaclip_eta"] == noisy["slaclip_eta"]
        and oracle["slaclip_num_slots"] == noisy["slaclip_num_slots"]
        for oracle, noisy in pairs
    )
    assert len(pairs) * 2 == 24
    assert {
        (oracle["slaclip_eta"], model)
        for oracle, _noisy in pairs
        for model in ("bert", "gpt2")
    } == {
        (eta, model)
        for eta in (0.025, 0.05, 0.1, 0.2)
        for model in ("bert", "gpt2")
    }


def test_stability_submitter_is_one_single_l4_allocation() -> None:
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "full-slaclip-k5-stability-suite-spec.json" in submitter
    assert "DPLORA_FULL_PARTITION=scavenger_l4" in submitter
    assert "DPLORA_FULL_GPU_GRES=gpu:l4swarm:1" in submitter
    assert "DPLORA_FULL_WALLTIME=12:00:00" in submitter
    assert "--array" not in submitter
    assert submitter.count("submit_full_slaclip_campaign.sh") == 1
