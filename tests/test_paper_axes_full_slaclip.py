from collections import Counter
from pathlib import Path

from paper_repro.full_slaclip_campaign import expand_spec, load_spec


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "hpc" / "paper-axes-full-slaclip-spec.json"
SUBMITTER = ROOT / "hpc" / "submit_paper_axes_full_slaclip.sh"


def test_paper_axes_matrix_is_paired_and_excludes_slaclip_q() -> None:
    spec = load_spec(SPEC)
    arms = expand_spec(spec)
    assert len(arms) == spec["expected_arm_count"] == 70
    assert Counter(arm["family"] for arm in arms) == {
        "primary": 10,
        "noise_sensitivity": 18,
        "rank_sensitivity": 36,
        "control": 6,
    }
    assert Counter(arm["method"] for arm in arms) == {
        "paper_dp_lora": 32,
        "slaclip_dp_lora": 32,
        "no_dp_lora_control": 3,
        "clip_only_control": 3,
    }
    assert all("slaclip_q" not in arm["method"].lower() for arm in arms)


def test_rank_sweep_matches_the_paper_axis_and_has_exact_pairs() -> None:
    arms = expand_spec(load_spec(SPEC))
    rank_arms = [arm for arm in arms if arm["family"] == "rank_sensitivity"]
    assert {arm["rank"] for arm in rank_arms} == {4, 16, 64, 128, 256, 1024}
    by_id = {arm["arm_id"]: arm for arm in arms}
    for adaptive in (
        arm for arm in rank_arms if arm["method"] == "slaclip_dp_lora"
    ):
        fixed = by_id[adaptive["reference_arm_id"]]
        assert fixed["method"] == "paper_dp_lora"
        assert fixed["rank"] == adaptive["rank"]
        assert fixed["seed"] == adaptive["seed"]
        assert fixed["rng_domain"] == adaptive["rng_domain"]
        assert adaptive["slaclip_num_slots"] == 5


def test_submitter_requests_one_a100_in_one_allocation() -> None:
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "paper-axes-full-slaclip-spec.json" in submitter
    assert "DPLORA_FULL_PARTITION" in submitter and "a100" in submitter
    assert "gpu:a100:1" in submitter
    assert "--array" not in submitter
    assert submitter.count("submit_full_slaclip_campaign.sh") == 1
