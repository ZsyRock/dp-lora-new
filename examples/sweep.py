"""Run only matched fixed-C/SlaClip pairs across epsilon or LoRA rank."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="sweep", required=True)
    for name in ("epsilon", "rank"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--lora-method", choices=["ffa", "vanilla"], default="ffa")
        sub.add_argument("--seed", type=int, nargs="+", default=[42, 43, 44])
        sub.add_argument(
            "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
        )
        sub.add_argument("--output-dir", default=f"results/sst2_{name}_sweep")
        sub.add_argument("--epochs", type=int, default=3)
        sub.add_argument("--logical-batch-size", type=int, default=256)
        sub.add_argument("--physical-batch-size", type=int, default=32)
        sub.add_argument("--initial-clip-norm", type=float, default=1.0)
        sub.add_argument("--slaclip-eta", type=float, default=0.2)
        sub.add_argument("--slaclip-beta", type=float, default=0.5)
        sub.add_argument("--slaclip-c-min", type=float, default=0.1)
        sub.add_argument("--slaclip-c-max", type=float, default=50.0)
        sub.add_argument("--observe-private-gradients", action="store_true")
        sub.add_argument("--acknowledge-non-dp-diagnostics", action="store_true")
        sub.add_argument("--store-per-sample-norms", action="store_true")
    subparsers.choices["epsilon"].add_argument(
        "--values", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0]
    )
    subparsers.choices["epsilon"].add_argument("--rank", type=int, default=8)
    subparsers.choices["rank"].add_argument(
        "--values", type=int, nargs="+", default=[2, 4, 8, 16]
    )
    subparsers.choices["rank"].add_argument("--epsilon", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.observe_private_gradients and not args.acknowledge_non_dp_diagnostics:
        raise SystemExit(
            "--observe-private-gradients requires " "--acknowledge-non-dp-diagnostics"
        )
    if args.store_per_sample_norms and not args.observe_private_gradients:
        raise SystemExit(
            "--store-per-sample-norms requires --observe-private-gradients"
        )
    runner = Path(__file__).with_name("sst2_roberta.py")
    for seed in args.seed:
        for value in args.values:
            run_dir = Path(args.output_dir) / f"{args.sweep}_{value}_seed{seed}"
            command = [
                sys.executable,
                str(runner),
                "--clipping",
                "both",
                "--lora-method",
                args.lora_method,
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--logical-batch-size",
                str(args.logical_batch_size),
                "--physical-batch-size",
                str(args.physical_batch_size),
                "--initial-clip-norm",
                str(args.initial_clip_norm),
                "--slaclip-eta",
                str(args.slaclip_eta),
                "--slaclip-beta",
                str(args.slaclip_beta),
                "--slaclip-c-min",
                str(args.slaclip_c_min),
                "--slaclip-c-max",
                str(args.slaclip_c_max),
                "--output-dir",
                str(run_dir),
                "--run-name",
                "paired",
            ]
            if args.observe_private_gradients:
                command.extend(
                    [
                        "--observe-private-gradients",
                        "--acknowledge-non-dp-diagnostics",
                    ]
                )
            if args.store_per_sample_norms:
                command.append("--store-per-sample-norms")
            if args.sweep == "epsilon":
                command.extend(["--epsilon", str(value), "--rank", str(args.rank)])
            else:
                command.extend(["--epsilon", str(args.epsilon), "--rank", str(value)])
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
