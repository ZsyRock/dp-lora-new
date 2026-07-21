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
                "--output-dir",
                str(run_dir),
            ]
            if args.sweep == "epsilon":
                command.extend(["--epsilon", str(value), "--rank", str(args.rank)])
            else:
                command.extend(["--epsilon", str(args.epsilon), "--rank", str(value)])
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
