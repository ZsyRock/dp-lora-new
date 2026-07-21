# Experiment contracts

The repository supports exactly two experiment arms: the fixed-C DP-LoRA
baseline and the same baseline with the main SlaClip threshold controller. All
shared arguments should remain identical when the two arms are compared.

## 1. Default baseline with trusted diagnostics

The runner defaults to fixed clipping, FFA-LoRA, RoBERTa-base, rank 8, target
epsilon 8, three epochs, logical batch size 256, physical batch size 32,
learning rate `5e-4`, `C=1` and seed 42:

```bash
python examples/sst2_roberta.py \
  --run-name baseline_default_seed42 \
  --observe-private-gradients \
  --acknowledge-non-dp-diagnostics
```

The DP model update remains active. The diagnostic channel is exact and is not
a DP output; keep its directory and the Slurm log access-controlled.

## 2. Main SlaClip on the same baseline

Main SlaClip starts at a configurable `C0` and then adapts `C_t`. It does not
have a fixed final target threshold. Configure `C0`, the controller, and its
bounds explicitly:

```bash
python examples/sst2_roberta.py \
  --clipping slaclip \
  --run-name slaclip_c0_2_seed42 \
  --initial-clip-norm 2.0 \
  --slaclip-eta 0.2 \
  --slaclip-beta 0.5 \
  --slaclip-c-min 0.1 \
  --slaclip-c-max 20 \
  --observe-private-gradients \
  --acknowledge-non-dp-diagnostics
```

For a controlled comparison, prefer one paired command so both arms receive the
same `C0`, model/data initialization, sampling seed and gradient-noise seed:

```bash
python examples/sst2_roberta.py \
  --clipping both \
  --run-name paired_c0_1_seed42 \
  --initial-clip-norm 1.0 \
  --observe-private-gradients \
  --acknowledge-non-dp-diagnostics
```

SlaClip uses a separate auxiliary-noise generator. This prevents its K slack
coordinates from shifting the gradient-noise stream relative to the fixed-C
arm while retaining independent Gaussian coordinates in the joint release.
Poisson-sampling and DP-noise seeds come from operating-system entropy, are
shared where required inside a paired process, and are never logged. Publishing
those internal seeds could undermine the privacy mechanism or its amplification;
the user-facing `--seed` controls reproducible model initialization only.

## Output schema

Each run has a unique, non-overwriting root:

```text
results/sst2/<run-name>/
├── experiment_results.json
├── fixed/ or slaclip/
│   ├── run_config.json
│   ├── metrics.json
│   └── private_diagnostics/
│       ├── NON_DP_PRIVATE_DATA.txt
│       ├── gradient_observations.jsonl
│       ├── training_steps.jsonl
│       └── epoch_summaries.jsonl
```

`run_config.json` records the Git SHA, complete arguments, dataset fingerprints,
resolved model revision when available, package versions, device, realized
noise multiplier/sample rate and the randomness policy. `metrics.json` contains
epsilon and the five validation metrics but excludes exact private training
loss. Internal sampling and DP-noise seeds are represented only as
`SECRET_NOT_LOGGED`.

The private JSONL files record, per logical step:

- exact global norm histogram, mean, standard deviation and quantiles;
- exact per-parameter norm summaries;
- exact clipped count/fraction and mean clip factor;
- exact clipped-sum, sampled-noise and applied private-gradient norms;
- `C_before` and `C_after`;
- SlaClip `q_hat`, `r_hat`, `gamma_t` and slack indicators;
- exact training loss, learning rate, epoch and epsilon;
- optional raw per-record norms with `--store-per-sample-norms`.

## HPC submission

The Slurm template enables trusted diagnostics. Select an arm and controller
without editing source code:

```bash
DP_LORA_CLIPPING=fixed \
DP_LORA_RUN_NAME=baseline_default_seed42 \
sbatch hpc/slurm/sst2_pair.sbatch

DP_LORA_CLIPPING=slaclip \
DP_LORA_INITIAL_CLIP_NORM=2.0 \
DP_LORA_SLACLIP_C_MAX=20 \
DP_LORA_RUN_NAME=slaclip_c0_2_seed42 \
sbatch hpc/slurm/sst2_pair.sbatch
```

## Ubuntu/HPC boundary

Ubuntu validation covers the optimizer equations, DP coverage, privacy
accounting, virtual batches, empty Poisson draws, diagnostics, checkpoint state,
CLI contracts, package build and CPU integration tests. The target HPC is still
needed to validate its actual NVIDIA driver/CUDA/PyTorch combination, scheduler
directives, memory ceiling, model/dataset cache access and full-scale runtime.
Those are deployment measurements rather than missing experiment logic, though
any issue exposed by the pilot should still be fixed and versioned normally.

For a fast CPU integration check before submitting a full run, select a tiny
RoBERTa-compatible model and bounded splits while keeping the entire DP path:

```bash
python examples/sst2_roberta.py \
  --clipping both \
  --model hf-internal-testing/tiny-random-roberta \
  --epochs 1 \
  --logical-batch-size 8 \
  --physical-batch-size 4 \
  --max-seq-length 32 \
  --max-train-samples 32 \
  --max-validation-samples 32 \
  --device cpu \
  --run-name ubuntu_smoke \
  --observe-private-gradients \
  --acknowledge-non-dp-diagnostics
```
