# Validation record

The `0.2.0rc1` candidate was validated on Ubuntu on 2026-07-21 before its Draft
PR was opened.

## Automated checks

- 61 unit/integration tests passed.
- The Python wheel built successfully as `dp_lora-0.2.0rc1`.
- Package dependency checks, Python formatting, shell syntax, YAML parsing and
  Git whitespace checks passed.
- The suite covers flat clipping/noise, classification-head coverage,
  mean-loss scaling, Poisson accounting, virtual batches, empty Poisson draws,
  SlaClip equations/controller state, exact diagnostics and a Hugging Face
  Trainer step.

## Real CPU smoke

The full paired command in [EXPERIMENTS.md](EXPERIMENTS.md) was run with
`hf-internal-testing/tiny-random-roberta`, 32 SST-2 training records, 32
validation records, one epoch and trusted diagnostics.

The structural assertions passed:

- fixed and SlaClip each completed four logical DP steps;
- both arms reported the same epsilon under the matched schedule;
- the fixed threshold stayed constant and the SlaClip threshold adapted;
- every gradient record had aligned training-step context and per-parameter
  summaries;
- ordinary metrics excluded exact private training loss;
- Poisson-sampling and DP-noise seeds were not logged.

The smoke output remains under the Git-ignored `results/` tree. Exact diagnostic
values are intentionally not copied into this tracked validation record.

## Still target-HPC-specific

The Ubuntu run cannot validate the target cluster's NVIDIA driver, supported
CUDA/PyTorch build, GPU memory ceiling, scheduler/account/module directives,
filesystem performance or external model/data cache policy. Run the same smoke
on an allocated GPU before launching the full experiment matrix.
