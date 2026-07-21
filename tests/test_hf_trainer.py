"""One-step Hugging Face Trainer integration smoke test."""

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from dp_lora import DPLoRAConfig
from dp_lora.integrations import DPLoRATrainer


class TinyDataset(Dataset):
    def __init__(self):
        self.inputs = torch.randn(8, 3)
        self.labels = torch.randint(0, 2, (8,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"input_ids": self.inputs[index], "labels": self.labels[index]}


class TinyTrainerLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 3, bias=False)})
        self.classifier = nn.Linear(3, 2)

    def forward(self, input_ids, labels=None, **kwargs):
        hidden = input_ids + self.lora_B["default"](self.lora_A["default"](input_ids))
        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def test_trainer_runs_one_private_step(tmp_path):
    trainer = DPLoRATrainer(
        model=TinyTrainerLoRA(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            max_steps=1,
            report_to=[],
            disable_tqdm=True,
            save_strategy="no",
            logging_strategy="no",
        ),
        train_dataset=TinyDataset(),
        dp_config=DPLoRAConfig(
            noise_multiplier=1.0,
            method="vanilla",
            clipping_mode="fixed",
        ),
    )
    trainer.train()
    assert trainer.dp_engine.accountant.steps == 1
