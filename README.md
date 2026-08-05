# Executable DP-LoRA paper reconstruction

The upstream `main` branch is an incomplete prototype.  This reconstruction
adds a fail-closed simulation of the five logical clients in Algorithm 1 of
“Differentially Private Low-Rank Adaptation of Large Language Model Using
Federated Learning” (arXiv:2312.17493).

The formal runner uses BERT-base and GPT-2 small sequentially on the English
MedDialog HealthcareMagic+iCliniq data.  It fixes the values explicitly stated
by the paper: `K=5`, `T=50`, `B=8`, `sigma=2`, `lr=5e-4`, `C=10`, and LoRA
rank `512`.  It updates both LoRA A and B, clips their aggregate batch-gradient
groups separately, adds Gaussian noise, and equally averages the five local
adapter states after every round.  The active adaptive extension is
`slaclip_dp_lora`: full SlaClip using the two noisy endpoints of its recovered
CDF.  Its base target clipped fraction is configurable; it does not use the
fixed-target SlaClip-Q controller or a baseline-derived calibration.

The paper omits model revisions, exact LoRA targets/alpha/dropout, optimizer,
sequence length, seed, and enough privacy-accounting constants to reproduce its
epsilon claims.  This branch records the chosen assumptions in every
`run_config.json`; it is an algorithm-level reconstruction, not a claim of
bit-for-bit reproduction.  Exact client batch losses and gradient statistics
are labelled `NON_DP_PRIVATE_DIAGNOSTIC`.

The executable is `paper_repro/train_federated.py`; immutable input staging is
implemented in `scripts/stage_paper_inputs.py`.  The tuned-fixed confirmation
study is launched by `hpc/submit_staged_slaclip_tuned_fixed.sh`; the older
general campaign remains available through `hpc/submit_full_slaclip_campaign.sh`.

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
- `clip_only_control`: clipping without Gaussian noise;
- `paper_dp_lora`: the literal reconstructed clipping-plus-noise mechanism;
  and
- `slaclip_dp_lora`: the full two-endpoint SlaClip controller with independent
  A/B thresholds.

For full SlaClip, let `s_hat[0]` be the noisy CDF endpoint near the current
threshold and `s_hat[K-1]` the noisy endpoint near zero.  The controller sets
`gamma_t = clip(1 - beta * (1 - s_hat[K-1] / (C_t + 1e-6)), 0, 1)` and then
`C_{t+1} = clip(C_t * exp(eta * (gamma_t - s_hat[0])), C_min, C_max)`.
Equivalently, the dynamic target clipped fraction is
`p_target,t = clip(beta * (1 - s_hat[K-1] / (C_t + 1e-6)), 0, 1)`.
Thus `beta` is the base target after discounting the estimated small-gradient
mass, not a fixed realized clipping rate.  The paper's factor `1/2` is the
default `beta=0.5`; other data/model settings may use another value through
`--slaclip-base-target-clipped-fraction`.  The old `--slaclip-beta` spelling is
accepted only as a backwards-compatible alias, and conflicting values fail
closed.
Only these noisy endpoints drive the update.  Exact CDF values and actual
clipping events are retained solely as `NON_DP_PRIVATE_DIAGNOSTIC` telemetry.
There is no `q_target`, target file, or calibration pass in the active method.

Each client/group jointly releases its clipped gradient and slack coordinates
with the same `sigma*C_t` coordinate scale; all clients in a round use the
same `C_t`, and `C_{t+1}` takes effect only in the following round.  This is an
exploratory adaptation of the sample-level SlaClip construction to the
repository's five client aggregate-gradient records.  Its composition and
sensitivity have not been independently certified, so it retains
`epsilon=null` and no end-to-end DP claim.  Older fixed-target artifacts are
historical only and are neither an active nor a recommended launch method.

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
timings, peak host/GPU memory, and a FedAvg reconstruction residual.  Full
SlaClip additionally records the complete noisy and exact diagnostic CDFs,
both controller endpoints, `gamma_t`, threshold update, bound hits, and the A/B
threshold trajectories.  A configuration fingerprint prevents a checkpoint
from being resumed under a
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

The current fair-baseline study is declared by
`hpc/staged-slaclip-tuned-fixed-spec.json` and coordinated by
`paper_repro/staged_slaclip_campaign.py`.  One Slurm allocation executes three
ordered stages without an array or child job:

1. 13 fixed thresholds, five development seeds, and both models (130
   model-level arms), with complete per-round clipping trajectories;
2. three model-specific initial thresholds crossed with five conditional
   full-SlaClip base targets and the same development seeds (150 arms); and
3. the locked best fixed and full-SlaClip configurations on twenty disjoint
   confirmation seeds per model (80 arms).

The five base targets are derived separately for each model from the selected
fixed-C trajectory.  For every development round, the calibration computes
`z_t = s_exact[K-1] / (C + 1e-6)` and
`beta_t = p_clipped,t / (1 - z_t)`, then takes five equally spaced values over
the empirical q10--q90 interval.  These exact calibration values are
`NON_DP_PRIVATE_DIAGNOSTIC`; they are a development procedure, not a claim of
end-to-end private hyperparameter tuning.  Both selections are hash-bound
before the next stage is materialized, and confirmation uses seeds `200..219`
that are absent from development.

The staged wrapper defaults to one 12-hour `scavenger_4a100` allocation with
two A100 lanes.  Completed arms, round checkpoints, selection locks, and the
compact result archive are reusable with the same `DPLORA_STAGED_RUN_ID` and
`--resume` if the preemptible partition cancels the allocation.

`hpc/submit_full_slaclip_campaign.sh` submits
`hpc/full_slaclip_campaign.sbatch` as one Slurm job on one node.  Inside that
single allocation, two independent GPU lanes consume a resumable 108-arm,
54-wave manifest; no array or nested child job is submitted.  Every arm runs
both BERT-base and GPT-2 small, so the manifest represents 216 model-level
training executions:

- 20 confirmatory primary arms: fixed DP-LoRA and full SlaClip at the paper's
  `C_0=10`, using ten paired seeds `42..51`;
- 24 initial-threshold robustness arms: both methods at
  `C_0 in {0.1, 1, 5, 20}`, seeds `42..44`;
- 24 full-SlaClip controller-sensitivity arms at `C_0=10`, seeds `42..44`,
  crossing `eta in {0.05, 0.1, 0.2}` with
  `beta in {0.5, 0.9, 0.99}` and excluding the already-covered
  `(eta=0.2, beta=0.5)` default;
- 18 noise-sensitivity arms: both methods at
  `sigma in {0.5, 1, 4}`, seeds `42..44`;
- 12 protected-record-count arms: both methods at
  `K_clients in {20, 80}`, seeds `42..44`; and
- 10 mechanism controls at `C_0=10`: no-DP and clip-only, seeds `42..46`.

The wrapper accepts `--test-only` for scheduler validation without submission
and `--resume` for an existing `DPLORA_FULL_RUN_ID`; the two flags may be
combined.

The development-only five-point base-target screen is declared in
`hpc/full-slaclip-beta5-screen-spec.json` and launched with
`hpc/submit_full_slaclip_beta5_screen.sh`.  It fixes `eta=0.2` and compares
`beta in {0.01, 0.03, 0.066, 0.146, 0.5}` over five paired seeds on the
currently staged MedDialog × {BERT-base, GPT-2 small} reconstruction.  Every
round logs the configured base fraction, the noisy near-zero mass adjustment,
the remaining non-small-gradient fraction, and the resulting dynamic target.
This screen is for per-model hyperparameter selection; it is not an
independent test-set result or a reproduction of the paper's downstream
benchmarks.  SlaClip-Q is not an arm in this campaign.

The confirmatory setting retains `K_clients=5`, `T=50`, `B=8`, `sigma=2`,
`lr=5e-4`, and LoRA rank `512`; only explicitly labelled sensitivity arms vary
`K_clients` or `sigma`.  Full SlaClip uses `K_slots=15`, bounds `[0.1, 50]`,
and defaults `eta=0.2`, `beta=0.5`.  The requested `K_slots=15` small-batch
policy exceeds the automatic bound implied by only five released client
records at the primary setting; its noise consequence is recorded.

Evaluation fixes a content-disjoint split (`data_split_seed=1729`), supervision
masks (`evaluation_seed=2718`), and a 512-record holdout across training seeds
and methods.  New outputs include full-CDF error and out-of-range slots,
non-private exact-oracle update error/direction agreement, threshold stability,
removed-gradient norm and retained energy, FedAvg signal/noise, best/final/AUC
loss, and paired confidence, effect-size, exact sign-flip, and Holm statistics.

Periodic checkpoints, per-arm status, incremental compact archives, strict
revalidation, and an atomic `job-status.json`
(`RUNNING`/`COMPLETED`/`FAILED`) make the campaign resumable without confusing
an incremental `IN_PROGRESS` summary with a live allocation.  The campaign
remains Level 1: it lacks the paper's downstream benchmarks, 6B/7B protocols,
and an end-to-end privacy certificate, so it is not by itself a
journal-complete result package.

See `docs/CURRENT_RESULT_ASSESSMENT.md` for the evidence-based interpretation
of legacy job `1298681` and its 7-minute-31-second wall time.

# Upstream project description

The financial industry has experienced significant strides in Natural Language Processing (NLP) facilitated by Language Model (LM) technologies. However, the escalating concerns regarding data privacy present a formidable barrier to the ongoing enhancement of these models. A notable challenge involves potential adversarial attackers exploiting the weight of Language Models trained by individual banks, thereby jeopardizing user data confidentiality.

This project seeks to address the critical issue of data privacy in Language Model training within the financial sector by proposing and implementing a Privacy-Preserving Federated Learning Protocol.
The primary goal is to establish a collaborative framework that empowers multiple banks to collectively train a Language Model without the necessity to share or access each other's private and sensitive data. Departing from the conventional practice of sharing precise model weights, this innovative framework facilitates the exchange of "biased weights." This approach thwarts third-party attempts to infer training data, thereby safeguarding the confidentiality of sensitive information.

The core principle of this federated learning approach is to ensure that the Language Model remains robust, accurate, and reflective of the diverse financial data landscape. Simultaneously, it addresses the privacy concerns inherent to individual financial institutions. By mitigating data privacy risks, this project strives to foster an environment where advancements in NLP can continue to flourish in the financial sector, promoting collaborative innovation while upholding the highest standards of data security.

![image](https://github.com/Michonster/FinLLM-DP-Lora/assets/83566627/700b1274-2bec-41c8-a717-57d1b7036165)

Process and planning for this project is based on this paper: https://arxiv.org/abs/2312.17493
