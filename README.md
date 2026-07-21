# DP-LoRA: fixed clipping and SlaClip experiments

This repository is a research implementation of record-level, flat DP-SGD for
LoRA fine-tuning. Its experiment contract is intentionally narrow:

- `fixed`: the fixed-`C` DP-LoRA baseline;
- `slaclip`: the same baseline with only the main SlaClip threshold controller.

There is no SlaClip-Q experiment. A paired run keeps the model, LoRA layout,
dataset split, seed, logical/physical batch sizes, optimizer, learning rate,
sampling rate, noise multiplier, number of logical steps and privacy accountant
identical. The method-specific difference is whether `C_t` stays fixed or is
updated from SlaClip's noised slack release.

## Privacy contract

The protected unit is one training record. For every record `i`, all trainable
LoRA and classification-head gradients are treated as one flat vector:

```text
g_i = concat(g_i^A1, g_i^B1, ..., g_i^head)
g_i_bar = g_i * min(1, C_t / ||g_i||_2)
```

Clipped vectors are summed, Gaussian noise with standard deviation `sigma*C_t`
is added, and the result is divided by the expected Poisson batch size. The RDP
accountant uses exactly the same Poisson sample rate and logical-step count as
the runtime data loader.

The implementation fails closed:

- a trainable parameter without a validated per-sample gradient raises an error;
- ordinary gradients can never silently update an uncovered classification head;
- mean-reduced losses are restored to unreduced per-record gradient scale before
  clipping;
- epsilon accounting is rejected for a non-Poisson loader;
- incomplete two-pass ghost clipping is explicitly disabled.

For SlaClip, the clipped gradient and the `K` slack coordinates form one joint
sensitivity-`C_t` Gaussian release. The controller uses only the noised slack
coordinates, so it does not add a second accountant event. Fixed clipping and
SlaClip therefore have the same epsilon when their shared settings match.

## Exact “god-view” diagnostics

It is possible to keep the DP optimizer active while recording exact private
gradient statistics for scientific analysis. This does **not** make those log
files differentially private. The trained model can retain its stated DP
guarantee if the exact diagnostic channel remains access-controlled, but the
complete set of outputs `(model, exact logs)` must not be described as having
the model's `(epsilon, delta)` guarantee.

The observer is off by default and requires two explicit switches:

```bash
python examples/sst2_roberta.py \
  --clipping both \
  --observe-private-gradients \
  --acknowledge-non-dp-diagnostics
```

It writes one JSONL record per logical optimizer step containing:

- exact per-record gradient-norm histogram on `[0, C_t]` plus overflow;
- exact clipped count and fraction;
- exact norm mean and quantiles;
- `C_before` and `C_after`;
- optionally every per-record norm;
- SlaClip's already-noised slack/controller values when applicable.

Files are written below a `private_diagnostics/` directory, carry the label
`NON_DP_PRIVATE_DIAGNOSTIC`, include a warning file, and are ignored by Git.
The package provides no remote-logger integration for this data.

Python configuration is equally explicit:

```python
from dp_lora import GradientObservationConfig

observation = GradientObservationConfig(
    enabled=True,
    acknowledge_non_dp=True,
    output_dir="private_diagnostics/pilot_001",
    histogram_bins=20,
    store_per_sample_norms=False,
)
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install \
  --constraint requirements/constraints-tested.txt \
  --editable ".[dev,examples]"
python -m pytest -q
```

The constraints file records the user-space versions verified on Ubuntu. It
does not pin PyTorch because a GPU build must match the target machine's driver
and CUDA environment.

The repository's automated suite covers clipping/noise, loss scaling,
classification-head coverage, Poisson accounting, virtual batches, the main
SlaClip equations, exact-observer labelling, and checkpoint state restoration.

## HPC bootstrap

Clone the repository, check out an accepted commit or release tag, inspect the
cluster's CUDA/driver modules, and then install the matching official PyTorch
build. The bootstrap script creates or updates the Conda environment, installs
the remaining tested dependencies and runs the unit suite:

```bash
git clone https://github.com/ZsyRock/dp-lora-new.git
cd dp-lora-new

# Example only: select the index/version supported by the target cluster.
DP_LORA_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
DP_LORA_TORCH_VERSION=2.10.0 \
bash scripts/bootstrap_hpc.sh
```

Record the exact code revision before each run:

```bash
git rev-parse HEAD
mkdir -p results/sst2
sbatch hpc/slurm/sst2_pair.sbatch
```

The Slurm file is a portable starting point. Cluster-specific account,
partition, module and scratch-storage directives should be added after inspecting
the target HPC documentation. Exact private diagnostic files must stay in
access-controlled storage and are ignored by Git.

## Paired SST-2 pilot

Run the fixed baseline followed by SlaClip with the same shared settings:

```bash
python examples/sst2_roberta.py \
  --clipping both \
  --lora-method ffa \
  --model roberta-base \
  --rank 8 \
  --epsilon 8 \
  --epochs 3 \
  --logical-batch-size 256 \
  --physical-batch-size 32 \
  --learning-rate 5e-4 \
  --max-grad-norm 1 \
  --seed 42 \
  --device cuda
```

`--clipping both` resets the seed, model initialization and data-loader RNG
before each side of the pair. Results are kept separate under
`results/sst2/{fixed,slaclip}_seed42/`.

The classification report contains the five prespecified metrics:

- accuracy;
- macro precision;
- macro recall;
- macro F1;
- macro one-vs-rest ROC-AUC computed from continuous probabilities.

For at least three paired seeds:

```bash
python examples/sweep.py epsilon \
  --values 1 2 4 8 \
  --rank 8 \
  --seed 42 43 44 \
  --lora-method ffa \
  --device cuda
```

The sweep runner always invokes `--clipping both`; it cannot create an unpaired
fixed/SlaClip comparison.

## Library API

```python
import torch
from dp_lora import DPLoRAEngine, SlaClipConfig

engine = DPLoRAEngine()
model, private_optimizer, private_loader = engine.make_private_with_epsilon(
    model=peft_model,
    optimizer=torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=5e-4
    ),
    data_loader=train_loader,
    target_epsilon=8.0,
    target_delta=1e-5,
    epochs=3,
    max_grad_norm=1.0,
    method="ffa",                 # LoRA parameterization
    clipping_mode="slaclip",     # or "fixed"
    slaclip=SlaClipConfig(num_slots=None, eta=0.2, beta=0.5),
)

for batch in private_loader:
    private_optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    per_sample_grads = engine.grad_sample_module.get_per_sample_grads()
    private_optimizer.step(per_sample_grads)
    engine.grad_sample_module.clear_per_sample_grads()

print(engine.get_epsilon())
```

The automatic `K` rule uses the SlaClip paper's tabulated values at `sigma=1`
and its formula otherwise. Set `num_slots` to an explicit positive integer to
override it.

## Logical and physical batches

`VirtualBatchManager` splits one logical Poisson batch into physical
microbatches. Each microbatch is individually materialized and clipped, but
clipped sums are accumulated; Gaussian noise, the parameter update and the
accountant step occur once, at the logical-batch boundary.

```python
from dp_lora.data.virtual_batch import VirtualBatchManager

with VirtualBatchManager(
    data_loader=private_loader,
    max_physical_batch_size=32,
    optimizer=private_optimizer,
) as physical_loader:
    ...
```

## Checkpointing

Checkpoint only after a completed logical step. `engine.state_dict()` includes
the wrapped optimizer, privacy accountant, DP random generator (when supplied),
logical-step counter and SlaClip controller state:

```python
torch.save(
    {
        "model": model.state_dict(),
        "engine": engine.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
    },
    checkpoint_path,
)
```

After reconstructing the identical model, optimizer, loader and engine, restore
the model and scheduler and call `engine.load_state_dict(checkpoint["engine"])`.
Configuration mismatches are rejected instead of silently changing the privacy
mechanism.

## Current scope

- Supported trainable modules: LoRA Linear layers and trainable Linear
  classification heads. Unsupported trainable parameter types fail closed.
- Supported formal accountant: Poisson-subsampled Gaussian RDP.
- Supported comparisons: fixed-C baseline and main SlaClip only.
- Ghost clipping remains unavailable until its two-pass update has the same
  coverage and resume guarantees as the materialized path.
- Exact logs are research-only private artifacts and must never be committed or
  published as DP outputs.

## Version policy

Development happens on `codex/<topic>` branches and enters `main` through a
tested pull request. Package release candidates use versions such as
`0.2.0rc1`; accepted commits receive immutable Git tags such as `v0.2.0` only
after CI and the target-HPC pilot pass. See [VERSIONING.md](VERSIONING.md) and
[CHANGELOG.md](CHANGELOG.md) for the full policy and release notes.

## License

MIT. See [LICENSE](LICENSE).
