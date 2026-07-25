# Executable DP-LoRA paper reconstruction

The upstream `main` branch is an incomplete prototype.  The
`hpc/paper-repro` branch adds a fail-closed, single-GPU simulation of the five
logical clients in Algorithm 1 of
“Differentially Private Low-Rank Adaptation of Large Language Model Using
Federated Learning” (arXiv:2312.17493).

The formal runner uses BERT-base and GPT-2 small sequentially on the English
MedDialog HealthcareMagic+iCliniq data.  It fixes the values explicitly stated
by the paper: `K=5`, `T=50`, `B=8`, `sigma=2`, `lr=5e-4`, `C=10`, and LoRA
rank `512`.  It updates both LoRA A and B, clips their aggregate batch-gradient
groups separately, adds Gaussian noise, and equally averages the five local
adapter states after every round.  An extension arm, `slaclip_q_dp_lora`, adds
the fixed-target SlaClip-Q controller without changing that federated update
unit.

The paper omits model revisions, exact LoRA targets/alpha/dropout, optimizer,
sequence length, seed, and enough privacy-accounting constants to reproduce its
epsilon claims.  This branch records the chosen assumptions in every
`run_config.json`; it is an algorithm-level reconstruction, not a claim of
bit-for-bit reproduction.  Exact client batch losses and gradient statistics
are labelled `NON_DP_PRIVATE_DIAGNOSTIC`.

The executable is `paper_repro/train_federated.py`; immutable input staging is
implemented in `scripts/stage_paper_inputs.py`.  Cluster-specific launch files
are intentionally kept outside this Git worktree under
`$HOME/hpc/projects/dp-lora-paper/`.

## What a completed run proves

A completed run is **Level 1: algorithm-execution reconstruction**.  It proves
that the pinned implementation completed its declared client/round schedule
and produced internally consistent adapter and diagnostic artifacts.  It does
not prove reproduction of a paper table: the six downstream benchmark
protocols are not sufficiently specified in the paper and are not staged by
this repository.  See `docs/REPRODUCIBILITY.md` for the Level 1/2/3 gates.

The runner exposes four auditable training arms:

- `no_dp_lora_control`: neither clipping nor Gaussian noise; still records
  whether each A/B aggregate gradient *would* exceed `C`;
- `clip_only_control`: clipping without Gaussian noise; and
- `paper_dp_lora`: the literal reconstructed clipping-plus-noise mechanism.
- `slaclip_q_dp_lora`: a federated SlaClip-Q adaptation with independent A/B
  thresholds.  Its required target file is derived from a completed matched
  `paper_dp_lora` baseline by taking, for each model and A/B group, the median
  of its per-round actual clipping fractions.

The fixed-target arm is SlaClip-Q, not the camera-ready full SlaClip controller
whose target changes dynamically.  Each client/group jointly releases its
clipped gradient and slack coordinates with the same `sigma*C_t` coordinate
scale; all clients in a round use the same `C_t`, and `C_{t+1}` takes effect
only in the following round.  The target calibration is derived from exact
private diagnostics, so this exploratory path remains
`NON_DP_PRIVATE_DIAGNOSTIC`, with `epsilon=null` and no end-to-end DP claim.

The same private HMAC run key and `--rng-domain` give all arms identical client
partitions, sampled examples, and BERT supervision masks without writing the
raw key or derived seeds to logs.  Control adapters and all exact diagnostics
remain non-private artifacts.

## Analysis and recovery artifacts

Each model writes one atomic diagnostic shard per completed round, periodic
pickle-free `safetensors` checkpoints, attempt/progress state, consolidated
JSONL records, adapter integrity checks, and a final behavior summary.  The
statistics include actual and counterfactual clipping rates, raw/clipped/noisy
gradient norms, signal/noise ratios and cosine, parameter/update norms,
effective `B @ A` norms, sample coverage/repetition, token counts, per-phase
timings, peak host/GPU memory, and a FedAvg reconstruction residual.  A
configuration fingerprint prevents a checkpoint from being resumed under a
different code/data/method or numerical-backend contract.  The runner hashes
every staged input byte and requires all dataset/model references to close
exactly over that inventory.  It also enforces deterministic Torch algorithms,
disables TF32/cuDNN benchmarking, and records the CUDA driver, CPU, BLAS and
thread settings.

`--resume` is the only mode that accepts an existing output directory.
`--stop-file PATH` requests a checkpointed stop at the next completed round;
the runner exits with status 75 after recording `CHECKPOINTED_STOP`.  Private
key files must be exactly 32 bytes, mode `0600`, in a user-owned mode `0700`
directory.  `sigma` is never relabelled as `epsilon`; accounting output is
explicitly `NOT_CERTIFIED` until the missing sensitivity/composition contract
is independently resolved.  Resume and completed-run reuse fail closed unless
the private round shards, checkpoint prefix, trainer state, consolidated logs,
final adapter and final summaries all reconcile.

## Iridis Slurm entry points

The cluster wrapper is
`$HOME/hpc/projects/dp-lora-paper/submit.sh`.  It pins the formal paper-literal
settings (`K=5`, `T=50`, `B=8`, `sigma=2`, `lr=5e-4`, `C=10`, `r=512`) and
supports:

```text
submit.sh smoke [--test-only]
submit.sh formal [--test-only]
submit.sh paired-smoke [--test-only]
submit.sh paired-formal [--test-only]
submit.sh slaclip-smoke [--test-only]
submit.sh slaclip-formal [--test-only]
```

`paired-formal` is the recommended behavior-analysis run: one allocation
executes no-DP, clip-only, and paper DP-LoRA in fixed order and validates their
sample, supervision, and client-partition schedules before creating
`pair_summary.json`.  Set `DPLORA_PAPER_SEED` for an independent repetition;
`slaclip-formal` first runs the matched fixed-threshold baseline, atomically
derives its four median targets, and then runs SlaClip-Q in the same allocation
before producing a target-bound comparison.  It also retains the no-DP and
clip-only controls for mechanism diagnosis.
all other formal mechanism parameters remain fixed.  A partial or
completed-but-unarchived run is resumed only with both an explicit existing
`DPLORA_PAPER_RUN_ID` and `--resume`.  Completed arms are deeply revalidated
without retraining and their authoritative summaries are preserved byte for
byte; an already archived output remains immutable and is refused.
`--test-only` performs byte/runtime preflight and asks Slurm to validate the
request without creating a job.

See `docs/CURRENT_RESULT_ASSESSMENT.md` for the evidence-based interpretation
of legacy job `1298681` and its 7-minute-31-second wall time.

# Upstream project description

The financial industry has experienced significant strides in Natural Language Processing (NLP) facilitated by Language Model (LM) technologies. However, the escalating concerns regarding data privacy present a formidable barrier to the ongoing enhancement of these models. A notable challenge involves potential adversarial attackers exploiting the weight of Language Models trained by individual banks, thereby jeopardizing user data confidentiality.

This project seeks to address the critical issue of data privacy in Language Model training within the financial sector by proposing and implementing a Privacy-Preserving Federated Learning Protocol.
The primary goal is to establish a collaborative framework that empowers multiple banks to collectively train a Language Model without the necessity to share or access each other's private and sensitive data. Departing from the conventional practice of sharing precise model weights, this innovative framework facilitates the exchange of "biased weights." This approach thwarts third-party attempts to infer training data, thereby safeguarding the confidentiality of sensitive information.

The core principle of this federated learning approach is to ensure that the Language Model remains robust, accurate, and reflective of the diverse financial data landscape. Simultaneously, it addresses the privacy concerns inherent to individual financial institutions. By mitigating data privacy risks, this project strives to foster an environment where advancements in NLP can continue to flourish in the financial sector, promoting collaborative innovation while upholding the highest standards of data security.

![image](https://github.com/Michonster/FinLLM-DP-Lora/assets/83566627/700b1274-2bec-41c8-a717-57d1b7036165)

Process and planning for this project is based on this paper: https://arxiv.org/abs/2312.17493
