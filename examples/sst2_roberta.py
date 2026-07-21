"""Paired fixed-C versus main-SlaClip DP-LoRA experiment on SST-2."""

from __future__ import annotations

import argparse
import json
import random
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched fixed-C baseline and main-SlaClip DP-LoRA on SST-2"
    )
    parser.add_argument(
        "--clipping",
        choices=["fixed", "slaclip", "both"],
        default="both",
        help="'both' runs the required paired comparison with identical shared settings",
    )
    parser.add_argument("--lora-method", choices=["ffa", "vanilla"], default="ffa")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--logical-batch-size", type=int, default=256)
    parser.add_argument("--physical-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument(
        "--K", type=int, default=None, help="SlaClip slots; omit for paper rule"
    )
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--c-min", type=float, default=0.1)
    parser.add_argument("--c-max", type=float, default=50.0)
    parser.add_argument("--output-dir", default="results/sst2")
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
    parser.add_argument("--store-per-sample-norms", action="store_true")
    return parser.parse_args()


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


def prepare_data(tokenizer, max_length: int):
    dataset = load_dataset("glue", "sst2")

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
    return prepared["train"], prepared["validation"]


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


def train_epoch(model, loader, optimizer, engine, device, physical_batch_size):
    model.train()
    loss_sum = 0.0
    microbatches = 0
    with VirtualBatchManager(
        data_loader=loader,
        max_physical_batch_size=physical_batch_size,
        optimizer=optimizer,
    ) as physical_loader:
        for batch in tqdm(physical_loader, leave=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            per_sample_grads = engine.grad_sample_module.get_per_sample_grads()
            optimizer.step(per_sample_grads)
            engine.grad_sample_module.clear_per_sample_grads()
            loss_sum += float(loss.item())
            microbatches += 1
    return loss_sum / max(1, microbatches)


def run_one(args, clipping_mode, train_dataset, validation_dataset, device):
    seed_everything(args.seed)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2
    )
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
    loader_generator = torch.Generator().manual_seed(args.seed)
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
    run_dir = Path(args.output_dir) / f"{clipping_mode}_seed{args.seed}"
    observation = GradientObservationConfig(
        enabled=args.observe_private_gradients,
        acknowledge_non_dp=args.acknowledge_non_dp_diagnostics,
        output_dir=str(run_dir / "private_diagnostics"),
        histogram_bins=args.diagnostic_bins,
        store_per_sample_norms=args.store_per_sample_norms,
    )
    slaclip = SlaClipConfig(
        num_slots=args.K,
        eta=args.eta,
        beta=args.beta,
        c_min=args.c_min,
        c_max=args.c_max,
    )
    engine = DPLoRAEngine()
    model, optimizer, train_loader = engine.make_private_with_epsilon(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=args.epsilon,
        target_delta=delta,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        method=args.lora_method,
        clipping_mode=clipping_mode,
        slaclip=slaclip,
        gradient_observation=observation,
    )
    model.to(device)

    epochs = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            engine,
            device,
            args.physical_batch_size,
        )
        metrics = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "loss": loss,
            "epsilon": engine.get_epsilon(),
            "seconds": time.time() - epoch_start,
            **metrics,
        }
        epochs.append(record)
        print(json.dumps({"clipping": clipping_mode, **record}, sort_keys=True))

    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "clipping": clipping_mode,
        "lora_method": args.lora_method,
        "seed": args.seed,
        "rank": args.rank,
        "target_epsilon": args.epsilon,
        "final_epsilon": engine.get_epsilon(),
        "delta": delta,
        "noise_multiplier": optimizer.noise_multiplier,
        "logical_batch_size": optimizer.expected_batch_size,
        "sample_rate": engine.sample_rate,
        "epochs": epochs,
        "total_seconds": time.time() - start,
        "exact_diagnostics_enabled": args.observe_private_gradients,
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
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset, validation_dataset = prepare_data(tokenizer, args.max_seq_length)
    modes = ["fixed", "slaclip"] if args.clipping == "both" else [args.clipping]
    results = [
        run_one(args, mode, train_dataset, validation_dataset, device) for mode in modes
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"paired_seed{args.seed}.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
