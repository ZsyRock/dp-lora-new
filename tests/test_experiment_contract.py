"""CLI contract for the baseline and main-SlaClip experiment runner."""

from examples.sst2_roberta import parse_args


def test_default_experiment_is_fixed_c_baseline():
    args = parse_args([])
    assert args.clipping == "fixed"
    assert args.initial_clip_norm == 1.0
    assert not args.observe_private_gradients


def test_main_slaclip_accepts_custom_initial_threshold_and_controller():
    args = parse_args(
        [
            "--clipping",
            "slaclip",
            "--initial-clip-norm",
            "2.5",
            "--slaclip-eta",
            "0.1",
            "--slaclip-beta",
            "0.7",
            "--slaclip-c-min",
            "0.2",
            "--slaclip-c-max",
            "8.0",
            "--observe-private-gradients",
            "--acknowledge-non-dp-diagnostics",
        ]
    )
    assert args.clipping == "slaclip"
    assert args.initial_clip_norm == 2.5
    assert args.slaclip_eta == 0.1
    assert args.slaclip_beta == 0.7
    assert args.slaclip_c_min == 0.2
    assert args.slaclip_c_max == 8.0
    assert args.observe_private_gradients


def test_legacy_max_grad_norm_alias_maps_to_shared_initial_threshold():
    args = parse_args(["--max-grad-norm", "3.0"])
    assert args.initial_clip_norm == 3.0
