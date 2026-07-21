"""Paired fixed-C versus main-SlaClip DP-LoRA experiment on SST-2."""

from __future__ import annotations

import argparse
import json
import platform
import random
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dp_lora import (
    DPLoRAEngine,
    GradientObservationConfig,
    SlaClipConfig,
)
from dp_lora.data.virtual_batch import VirtualBatchManager


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched fixed-C baseline and main-SlaClip DP-LoRA on SST-2"
    )
    parser.add_argument(
        "--clipping",
        choices=["fixed", "slaclip", "both"],
        default="fixed",
        help="fixed is the default baseline; both runs the matched comparison",
    )
    parser.add_argument("--lora-method", choices=["ffa", "vanilla"], default="ffa")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--logical-batch-size", type=int, default=256)
    parser.add_argument("--physical-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--initial-clip-norm",
        "--max-grad-norm",
        dest="initial_clip_norm",
        type=float,
        default=1.0,
        help="fixed baseline C and SlaClip initial threshold C0",
    )
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional smoke-test subset; omit for the full training split",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="Optional smoke-test subset; omit for the full validation split",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument(
        "--K", type=int, default=None, help="SlaClip slots; omit for paper rule"
    )
    parser.add_argument(
        "--slaclip-eta", "--eta", dest="slaclip_eta", type=float, default=0.2
    )
    parser.add_argument(
        "--slaclip-beta", "--beta", dest="slaclip_beta", type=float, default=0.5
    )
    parser.add_argument(
        "--slaclip-c-min", "--c-min", dest="slaclip_c_min", type=float, default=0.1
    )
    parser.add_argument(
        "--slaclip-c-max", "--c-max", dest="slaclip_c_max", type=float, default=50.0
    )
    parser.add_argument("--output-dir", default="results/sst2")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Unique output directory name; defaults to UTC timestamp plus seed",
    )
    parser.add_argument(
        "--observe-private-gradients",
        action="store_true",
        help="Write exact gradient norms/histograms for trusted analysis (NOT DP)",
    )
    parser.add_argument(
        "--acknowledge-non-dp-diagnostics",
        action="store_true",
        help="Required second switch for exact private diagnostics",
    )
    parser.add_argument("--diagnostic-bins", type=int, default=20)
    parser.add_argument(
        "--no-parameter-statistics",
        action="store_false",
        dest="parameter_statistics",
        help="Disable exact per-parameter norm summaries to reduce analysis overhead",
    )
    parser.set_defaults(parameter_statistics=True)
    parser.add_argument("--store-per-sample-norms", action="store_true")
    return parser.parse_args(argv)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_noise_generator(device: torch.device, seed: int):
    """Use independent DP RNG streams without perturbing model/dropout RNG."""
    if device.type not in ("cpu", "cuda"):
        return None
    return torch.Generator(device=device).manual_seed(seed)


def installed_versions() -> dict[str, str]:
    packages = (
        "dp-lora",
        "torch",
        "transformers",
        "peft",
        "opacus",
        "dp-accounting",
        "datasets",
        "scikit-learn",
        "numpy",
    )
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def repository_commit() -> str | None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def prepare_data(
    tokenizer,
    max_length: int,
    revision: str | None,
    max_train_samples: int | None,
    max_validation_samples: int | None,
):
    dataset = load_dataset("glue", "sst2", revision=revision)
    if max_train_samples is not None:
        dataset["train"] = dataset["train"].select(
            range(min(max_train_samples, len(dataset["train"])))
        )
    if max_validation_samples is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(max_validation_samples, len(dataset["validation"])))
        )

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    prepared = dataset.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    prepared = prepared.rename_column("label", "labels")
    prepared.set_format("torch")
    train = prepared["train"]
    validation = prepared["validation"]
    metadata = {
        "name": "glue/sst2",
        "requested_revision": revision,
        "version": str(train.info.version),
        "train_fingerprint": train._fingerprint,
        "validation_fingerprint": validation._fingerprint,
        "train_records": len(train),
        "validation_records": len(validation),
    }
    return train, validation, metadata


def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    labels, predictions, probabilities = [], [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        labels.append(batch["labels"].cpu())
        predictions.append(logits.argmax(dim=-1).cpu())
        probabilities.append(logits.softmax(dim=-1).cpu())

    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(predictions).numpy()
    y_prob = torch.cat(probabilities).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    if y_prob.shape[1] == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "macro_ovr_roc_auc": float(auc),
    }


def train_epoch(
    model,
    loader,
    optimizer,
    engine,
    device,
    physical_batch_size,
    epoch,
):
    model.train()
    epoch_loss_sum = 0.0
    epoch_sample_count = 0
    logical_loss_sum = 0.0
    logical_sample_count = 0
    with VirtualBatchManager(
        data_loader=loader,
        max_physical_batch_size=physical_batch_size,
        optimizer=optimizer,
    ) as physical_loader:
        for microbatch_index, batch in enumerate(
            tqdm(physical_loader, leave=False), start=1
        ):
            batch = {key: value.to(device) for key, value in batch.items()}
            sample_count = int(batch["labels"].shape[0])
            optimizer.zero_grad()
            if sample_count == 0:
                loss_value = None
                per_sample_grads = engine.grad_sample_module.empty_per_sample_grads()
            else:
                loss = model(**batch).loss
                loss.backward()
                loss_value = float(loss.item())
                per_sample_grads = engine.grad_sample_module.get_per_sample_grads()

            step_stats = optimizer.step(per_sample_grads)
            engine.grad_sample_module.clear_per_sample_grads()
            if loss_value is not None:
                weighted_loss = loss_value * sample_count
                epoch_loss_sum += weighted_loss
                epoch_sample_count += sample_count
                logical_loss_sum += weighted_loss
                logical_sample_count += sample_count

            if not step_stats["skipped"]:
                if engine.observer is not None:
                    engine.observer.record_training_step(
                        {
                            "epoch": int(epoch),
                            "microbatch_index": int(microbatch_index),
                            "logical_step": int(step_stats["logical_step"]),
                            "mean_training_loss": (
                                logical_loss_sum / logical_sample_count
                                if logical_sample_count
                                else None
                            ),
                            "epsilon": float(engine.get_epsilon()),
                            "learning_rates": [
                                float(group["lr"]) for group in optimizer.param_groups
                            ],
                            "optimizer_statistics": step_stats,
                        }
                    )
                logical_loss_sum = 0.0
                logical_sample_count = 0

    return epoch_loss_sum / epoch_sample_count if epoch_sample_count else None


def run_one(
    args,
    clipping_mode,
    train_dataset,
    validation_dataset,
    dataset_metadata,
    device,
    run_root,
    sampling_seed,
    gradient_noise_seed,
    slaclip_noise_seed,
):
    seed_everything(args.seed)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, revision=args.model_revision
    )
    resolved_model_revision = getattr(base_model.config, "_commit_hash", None)
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            target_modules=["query", "value"],
            lora_dropout=0.0,
            bias="none",
            task_type="SEQ_CLS",
        ),
    )
    model.to(device)
    loader_generator = torch.Generator().manual_seed(sampling_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.logical_batch_size,
        shuffle=True,
        generator=loader_generator,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.physical_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    delta = len(train_dataset) ** -1.1
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    run_dir = run_root / clipping_mode
    run_dir.mkdir(parents=True, exist_ok=False)
    observation = GradientObservationConfig(
        enabled=args.observe_private_gradients,
        acknowledge_non_dp=args.acknowledge_non_dp_diagnostics,
        output_dir=str(run_dir / "private_diagnostics"),
        histogram_bins=args.diagnostic_bins,
        parameter_statistics=args.parameter_statistics,
        store_per_sample_norms=args.store_per_sample_norms,
    )
    slaclip = SlaClipConfig(
        num_slots=args.K,
        eta=args.slaclip_eta,
        beta=args.slaclip_beta,
        c_min=args.slaclip_c_min,
        c_max=args.slaclip_c_max,
    )
    gradient_generator = make_noise_generator(device, gradient_noise_seed)
    slaclip_generator = make_noise_generator(device, slaclip_noise_seed)
    engine = DPLoRAEngine()
    model, optimizer, train_loader = engine.make_private_with_epsilon(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=args.epsilon,
        target_delta=delta,
        epochs=args.epochs,
        max_grad_norm=args.initial_clip_norm,
        method=args.lora_method,
        clipping_mode=clipping_mode,
        slaclip=slaclip,
        generator=gradient_generator,
        slaclip_generator=slaclip_generator,
        gradient_observation=observation,
    )

    device_name = None
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    run_config = {
        "schema_version": 1,
        "privacy_status": "CONFIGURATION_ONLY_NO_PRIVATE_DATA",
        "run_name": args.run_name,
        "clipping": clipping_mode,
        "clip_semantics": {
            "fixed": "C_t is constant at initial_clip_norm",
            "slaclip": "initial_clip_norm is C0; main SlaClip adapts C_t",
        }[clipping_mode],
        "arguments": vars(args),
        "dataset": dataset_metadata,
        "model": {
            "name": args.model,
            "requested_revision": args.model_revision,
            "resolved_revision": resolved_model_revision,
        },
        "privacy": {
            "target_epsilon": args.epsilon,
            "target_delta": delta,
            "noise_multiplier": optimizer.noise_multiplier,
            "sample_rate": engine.sample_rate,
            "expected_batch_size": optimizer.expected_batch_size,
            "poisson_sampling": True,
        },
        "clipping_configuration": {
            "initial_clip_norm": args.initial_clip_norm,
            "slaclip_num_slots": (
                optimizer.num_slots if clipping_mode == "slaclip" else None
            ),
            "slaclip_eta": args.slaclip_eta,
            "slaclip_beta": args.slaclip_beta,
            "slaclip_c_min": args.slaclip_c_min,
            "slaclip_c_max": args.slaclip_c_max,
        },
        "diagnostics": {
            "enabled": args.observe_private_gradients,
            "privacy_status": (
                "NON_DP_PRIVATE_DIAGNOSTIC"
                if args.observe_private_gradients
                else "DISABLED"
            ),
            "parameter_statistics": args.parameter_statistics,
            "store_per_sample_norms": args.store_per_sample_norms,
        },
        "randomness": {
            "public_model_initialization_seed": args.seed,
            "poisson_sampling_seed": "SECRET_NOT_LOGGED",
            "gradient_noise_seed": "SECRET_NOT_LOGGED",
            "slaclip_auxiliary_noise_seed": "SECRET_NOT_LOGGED",
            "paired_sampling_and_gradient_streams": args.clipping == "both",
            "separate_noise_streams": gradient_generator is not None,
        },
        "runtime": {
            "git_commit": repository_commit(),
            "command": [sys.executable, *sys.argv],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "device_name": device_name,
            "packages": installed_versions(),
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    epochs = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        private_training_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            engine,
            device,
            args.physical_batch_size,
            epoch,
        )
        metrics = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "epsilon": engine.get_epsilon(),
            "seconds": time.time() - epoch_start,
            **metrics,
        }
        epochs.append(record)
        print(json.dumps({"clipping": clipping_mode, **record}, sort_keys=True))
        if engine.observer is not None:
            engine.observer.record_epoch_summary(
                {
                    "epoch": epoch,
                    "mean_training_loss": private_training_loss,
                    "epsilon": record["epsilon"],
                    "validation_metrics": metrics,
                    "clip_norm_after_epoch": float(optimizer.max_grad_norm),
                }
            )

    result = {
        "clipping": clipping_mode,
        "lora_method": args.lora_method,
        "seed": args.seed,
        "rank": args.rank,
        "initial_clip_norm": args.initial_clip_norm,
        "target_epsilon": args.epsilon,
        "final_epsilon": engine.get_epsilon(),
        "delta": delta,
        "noise_multiplier": optimizer.noise_multiplier,
        "logical_batch_size": optimizer.expected_batch_size,
        "sample_rate": engine.sample_rate,
        "epochs": epochs,
        "total_seconds": time.time() - start,
        "exact_diagnostics_enabled": args.observe_private_gradients,
        "private_training_loss_excluded": True,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    if args.observe_private_gradients and not args.acknowledge_non_dp_diagnostics:
        raise SystemExit(
            "--observe-private-gradients requires --acknowledge-non-dp-diagnostics"
        )
    if args.store_per_sample_norms and not args.observe_private_gradients:
        raise SystemExit(
            "--store-per-sample-norms requires --observe-private-gradients"
        )
    if args.initial_clip_norm <= 0:
        raise SystemExit("--initial-clip-norm must be positive")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise SystemExit("--max-train-samples must be positive")
    if args.max_validation_samples is not None and args.max_validation_samples <= 0:
        raise SystemExit("--max-validation-samples must be positive")
    if args.clipping in ("slaclip", "both") and not (
        args.slaclip_c_min <= args.initial_clip_norm <= args.slaclip_c_max
    ):
        raise SystemExit("SlaClip requires c_min <= initial_clip_norm <= c_max")
    if args.run_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_name = f"{timestamp}_seed{args.seed}"
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise SystemExit("--run-name must be a single safe directory name")
    run_root = Path(args.output_dir) / args.run_name
    if run_root.exists():
        raise SystemExit(
            f"Refusing to mix or overwrite experiment outputs: {run_root} already exists"
        )

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    train_dataset, validation_dataset, dataset_metadata = prepare_data(
        tokenizer,
        args.max_seq_length,
        args.dataset_revision,
        args.max_train_samples,
        args.max_validation_samples,
    )
    sampling_seed = secrets.randbits(63)
    gradient_noise_seed = secrets.randbits(63)
    slaclip_noise_seed = secrets.randbits(63)
    modes = ["fixed", "slaclip"] if args.clipping == "both" else [args.clipping]
    results = [
        run_one(
            args,
            mode,
            train_dataset,
            validation_dataset,
            dataset_metadata,
            device,
            run_root,
            sampling_seed,
            gradient_noise_seed,
            slaclip_noise_seed,
        )
        for mode in modes
    ]
    (run_root / "experiment_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
