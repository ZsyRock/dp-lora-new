# Reproducibility claims and evaluation gates

This document defines what may be claimed from this repository's DP-LoRA
experiments and what evidence is still required to reproduce the numerical
results in arXiv:2312.17493v2. It deliberately separates successful execution
of a reconstructed mechanism from reproduction of the paper's benchmark
scores.

## Claim levels

### Level 1: mechanism execution

This level establishes that the declared reconstruction ran as implemented. A
run may be described as a **mechanism execution** when all of the following are
true:

- the code, base-model revisions, input manifest, environment, effective
  hyperparameters, and output namespace are immutable and recorded;
- every consumed dataset/model path is covered by the byte-verified inventory,
  and the driver, deterministic-algorithm, CPU/BLAS and thread contract is
  recorded in the scientific fingerprint;
- all requested client/round updates completed, with no missing or duplicate
  `(round, client)` records;
- saved LoRA adapters contain exactly the expected A/B tensors and no
  non-finite values;
- the normally zero-initialized LoRA B matrices became nonzero, demonstrating
  that optimizer updates reached the saved adapter;
- clipping/noise diagnostics reconcile from client records through round and
  final summaries, and checkpoint/round-shard digests survive a strict resume
  or completed-run revalidation; and
- any exact losses, gradient norms, and clipping events are labelled as
  non-DP private diagnostics rather than DP releases.

Passing this level does **not** establish downstream utility, an independently
verified privacy budget, or agreement with a number in the paper. Acceptable
wording is, for example, "we successfully executed our reconstruction of the
federated DP-LoRA mechanism." It is not acceptable to shorten this to "we
reproduced the paper's results."

### Level 2: benchmark reconstruction under public assumptions

This level adds a complete, versioned evaluation protocol chosen from public
information where the paper is silent. It may be claimed only after every data,
protocol, model-specific, and control gate below passes. The resulting scores
must be described as a **benchmark reconstruction under explicitly documented
public assumptions**.

Level 2 permits comparisons between runs made by this repository under the
same evaluator. It does not establish that the authors used the same dataset
release, prompt, task head, decoder, normalization, or scoring rule. Paper
table values may be shown as contextual references, but must not be treated as
strict pass/fail targets.

### Level 3: strict paper result reproduction

This level requires evidence that the experiment and evaluator match the
authors' original protocol, not merely a plausible reconstruction. In addition
to all Level 1 and Level 2 gates, it requires authoritative specification of:

- exact benchmark releases, configurations, examples, splits, and labels;
- exact preprocessing, prompt templates, demonstrations, answer ordering,
  task heads, tokenization, truncation, and context length;
- exact forced-choice scoring or generation settings, prediction
  normalization, metric implementation, and aggregation;
- exact base checkpoints, LoRA targets, initialization, optimizer, local-step
  schedule, client partitioning, and all other training choices omitted by the
  paper;
- the privacy adjacency relation, sampling mechanism, sensitivity and
  clipping semantics, aggregation/composition rule, accountant, and the
  mapping from noise to each reported `(epsilon, delta)` pair; and
- the authors' seed/repetition policy or enough independent repetitions to
  make a statistically defensible comparison.

The public paper and reference repository do not currently provide this
information. Until it is obtained from an authoritative artifact or the
authors, Level 3 is blocked and the repository must not claim strict numerical
reproduction.

## Benchmark data gates

None of the following six downstream benchmarks was present in the local
DP-LoRA scratch/cache inventory at the time of the reproducibility audit:

| Domain | Benchmark | Required staging evidence |
| --- | --- | --- |
| General | BoolQ | Exact release/configuration, labelled evaluation split, revision or source archive hash, schema/count, and licence |
| General | PIQA | Exact release/configuration, labelled evaluation split, revision or source archive hash, schema/count, and licence |
| General | WinoGrande | Exact release/configuration (including size variant), labelled evaluation split, revision or source archive hash, schema/count, and licence |
| Medical | MedQuAD | Exact upstream release, included subsets, split construction, archive/file hashes, schema/count, references, and licence |
| Medical | LiveQA Test | Exact task year/release, question/reference files, split, archive/file hashes, schema/count, and licence |
| Medical | MEDIQA-Ans | Exact task year/release, question/reference files, split, archive/file hashes, schema/count, and licence |

Before evaluation, each benchmark must have an immutable manifest containing
the information above. The manifest must also document differences from the
paper's approximate counts. Medical evaluation records must be checked for
exact and near-duplicate overlap with the staged MedDialog training union; the
overlap report is part of the result, not an optional cleanup step. Evaluation
must stop rather than silently substitute another dataset or unlabelled test
split.

## Evaluation protocol gates

The paper reports generic "performance" values but does not specify accuracy,
exact match, token F1, ROUGE, BLEU, perplexity, or another metric. It also does
not publish prompts, few-shot examples, choice scoring, decoding settings,
normalization, or a scorer. Consequently, the evaluator must freeze and record
at least:

1. dataset field mapping and exact example identifiers;
2. the byte-for-byte prompt or model input template;
3. zero/few-shot policy and any demonstrations;
4. tokenizer revision, context/truncation side, maximum input length, and
   answer-choice ordering;
5. forced-choice scoring formula, token-length normalization, tie handling,
   and aggregation;
6. generation method, maximum new tokens, stopping strings/token IDs,
   temperature, sampling, beam parameters, and random seed;
7. answer normalization and the pinned implementation/version of every
   metric; and
8. per-example inputs, predictions, raw choice scores, normalized scores, and
   aggregate metrics in machine-readable outputs.

Evaluation must be deterministic when configured as deterministic, and a
second execution over the same artifacts must reproduce the same per-example
scores within a declared numerical tolerance. The current fixed-mask
MedDialog MLM/CLM holdout is removed from the united training index pool, but
the gate is content-based rather than index-only: the selected holdout uses
unique normalized `(src, tgt)` contents and every matching duplicate is also
removed from the train/validation/test union. Its loss remains only an internal
training diagnostic; it is not one of the paper's benchmark metrics.

## Model-specific limitations

### GPT-2

The causal-LM runner can support a declared reconstructed protocol:

- BoolQ, PIQA, and WinoGrande may use conditional continuation
  log-likelihood. Both unnormalized and token-length-normalized results should
  be retained so that the choice is visible.
- Medical QA may use deterministic generation, preferably greedy decoding for
  the primary reconstruction. Exact decoding and stopping rules must be
  versioned.
- Because the paper does not name its medical scorer, report a transparent
  suite such as normalized exact match, token F1, and ROUGE-L, plus any pinned
  official task scorer. No member of that suite may be relabelled as the
  paper's unnamed metric.

### BERT

The current BERT artifact is a masked-language-model adapter. It does not
natively generate variable-length answers for MedQuAD, LiveQA, or MEDIQA-Ans.
Therefore:

- general multiple-choice tasks may be evaluated with a fully specified
  pseudo-log-likelihood or masked-choice reconstruction, but the result must be
  labelled BERT-specific and must not be presented as the paper's unknown
  scoring protocol;
- ranking the provided medical reference answers by pseudo-likelihood is not
  an open-ended QA evaluation and would expose the candidate answers to the
  model; it must not be used as a substitute for the paper's medical scores;
- open-ended medical evaluation remains `UNSUPPORTED` until an authoritative
  task head, candidate construction, or evaluation protocol is available, or
  until a new protocol is introduced and clearly labelled as a repository
  extension.

GPT-2 conditional likelihood and BERT pseudo-likelihood are different model
interfaces. Matching metric names alone does not make their protocols
equivalent to each other or to the unpublished paper protocol.

## Required controls and repetitions

Every benchmark study must evaluate these four controls through the identical
data and evaluation pipeline:

| Arm | Training mechanism |
| --- | --- |
| `base` | Pinned pretrained model with no federated adaptation |
| `no_dp` | Same federated LoRA schedule with neither clipping nor Gaussian noise |
| `clip_only` | Same schedule and declared clipping bound, with zero Gaussian noise |
| `dp` | Same schedule, declared clipping bound, and declared Gaussian noise multiplier |

Apart from the mechanism difference in the table, model revision, LoRA
configuration, client partitions, batch selection policy, rounds, local steps,
learning rate, context length, and evaluator must remain fixed. Any unavoidable
difference must be named in the result.

A claim about the adaptive extension must additionally evaluate
`slaclip_dp_lora` against its matched `paper_dp_lora` arm with the same initial
threshold, seed, examples, supervision masks, and standardized Gaussian-noise
draws.  The active extension is full SlaClip; an older fixed-target run is not
an acceptable substitute.

Use at least three independent training seeds for every stochastic trained arm
and report each seed plus mean, standard deviation, and sample count. More
seeds may be required when variance is large. The deterministic `base` arm may
be evaluated once only if determinism is verified. DP randomness must remain
independent between repetitions; privacy-sensitive seed material need not be
published, but run provenance and independence must be auditable.

## Privacy-label gate

`sigma = 2` is a Gaussian noise multiplier/scale. It is **not** evidence that
`epsilon = 2`. Results from the current fixed-sigma experiment must be labelled
with `sigma=2` and must not be placed in, or named after, the paper's
`epsilon=2` row.

An epsilon label is allowed only after a reviewed accountant records the
adjacency definition, sampling probability and sampling model, number and
composition of mechanisms, client participation, A/B grouping and clipping
sensitivity, aggregation weights, delta, accountant implementation/version,
and all numerical inputs. If the public paper does not provide enough
information to establish equivalence, the computed budget must be labelled as
the repository's reconstructed privacy accounting rather than the authors'
budget.

### Full SlaClip extension boundary

The active adaptive arm is `slaclip_dp_lora`.  For each model and LoRA A/B
group, its K-slot noisy slack release provides two CDF proxies:
`q_hat = s_hat[0]` near the current threshold and
`r_hat = s_hat[K-1]` near zero.  Following the pinned full-SlaClip controller,
it computes

```text
z_hat   = r_hat / (C_t + 1e-6)
p_target,t = clip(beta * (1 - z_hat), 0, 1)
gamma_t = 1 - p_target,t
C_{t+1} = clip(C_t * exp(eta * (gamma_t - q_hat)), C_min, C_max)
```

Here `beta` is the configurable base target clipped fraction after removing
the noisy near-zero/small-gradient mass.  It is not the realized or fixed
clipping fraction: `p_target,t` changes every round with `z_hat`.  The
reference formula's factor `1/2` is the default `beta=0.5`, exposed as
`--slaclip-base-target-clipped-fraction`; the legacy `--slaclip-beta` spelling
is only a checked compatibility alias.

All five clients in a round use the same `C_t`; the new threshold takes effect
only in the next round.  Only the *noisy* endpoints drive the update.  The
exact CDF proxy, raw gradient norms, and actual clipping fractions are written
only as `NON_DP_PRIVATE_DIAGNOSTIC` telemetry and must not be presented as DP
releases.  Full SlaClip has no SlaClip-Q fixed target, target file, or
runtime calibration input.  A development campaign may pre-register several
scalar `beta` hyperparameters from earlier non-DP private baseline diagnostics;
each candidate still drives the full two-endpoint dynamic controller above.

The slack coordinates and clipped gradient form one bounded vector per client
and group and receive independent Gaussian coordinates at scale `sigma*C_t`.
This retains the joint-release construction inside the declared federated
adaptation, but it does not establish the missing end-to-end accountant.  In
particular, the original construction is adapted here to five client
aggregate-gradient records rather than a standard per-sample DP-SGD batch.
The slot count is explicit in each campaign specification.  With five released
records at `sigma=2`, both `K=15` and the later `K=5` development screen exceed
the automatic rule's `K=1`; their theoretical normalized endpoint-noise
standard deviations are 3.464 and 2.0, respectively.  Consequently all current full
SlaClip results must retain `epsilon=null`,
`end_to_end_dp_certified=false`, and the Level-1 claim boundary.

Baseline-derived *fixed-target controllers* may still be mentioned by old
artifacts so that historical runs remain interpretable.  That SlaClip-Q-style
ablation is not an active method.  It is distinct from selecting a scalar
full-SlaClip `beta` before training: the latter still computes the per-round
target as `beta * (1-z_hat)`.

## K=5 baseline-range development screen

`hpc/full-slaclip-k5-baseline-range-spec.json` pre-registers one 30-arm,
15-wave campaign with matched fixed-`C=10` baselines and full SlaClip at
`K=5`.  It retains the previous MedDialog, BERT/GPT-2, five-client, 50-round,
batch-8, `sigma=2`, learning-rate, LoRA-rank, evaluation, and seed settings.

The candidate base target clipped fractions are
`{0, 0.19, 0.38, 0.57, 0.76}`.  They are the five equally spaced endpoints of
the BERT baseline's roundwise-five-seed-mean any-group clipping interval
`[0, 0.76]`; this follows the requested range-grid example and is not an
empirical quantile estimator.  The specification pins the source campaign and
all ten source trajectory hashes.  GPT-2's matched baseline interval is
exactly `[0,0]`, so the same five values are explicitly labelled cross-model
exploration for GPT-2 rather than falsely described as a within-model grid.
The resulting ranking remains development evidence only; a selected value
must be frozen before an independent confirmation run.

## Current single-job Level-1 campaign

The checked-in campaign specification contains 108 resumable runner arms in
one Slurm allocation on one node.  Two GPU lanes consume 54 waves without job
arrays or nested submissions.  Each runner arm trains both BERT-base and GPT-2
small, producing 216 model-level training executions:

| Family | Runner-arm matrix | Count |
| --- | --- | ---: |
| Confirmatory primary | `paper_dp_lora` and `slaclip_dp_lora`; paper setting `C_0=10`; paired seeds `42..51` | 20 |
| Initial-threshold robustness | Both methods; `C_0 in {0.1, 1, 5, 20}`; seeds `42..44` | 24 |
| Full-SlaClip sensitivity | `C_0=10`; seeds `42..44`; `eta in {0.05, 0.1, 0.2}` x `beta in {0.5, 0.9, 0.99}`, excluding the primary default `(0.2, 0.5)` | 24 |
| Noise sensitivity | Both methods; `C_0=10`; `sigma in {0.5, 1, 4}`; seeds `42..44` | 18 |
| Protected-record-count sensitivity | Both methods; `C_0=10`; `K_clients in {20, 80}`; seeds `42..44` | 12 |
| Mechanism controls | `no_dp_lora_control` and `clip_only_control`; `C_0=10`; seeds `42..46` | 10 |

The confirmatory setting remains five clients, 50 rounds, batch size 8,
`sigma=2`, learning rate `5e-4`, and LoRA rank 512; only explicitly labelled
sensitivity arms vary client count or sigma.  Full SlaClip uses `K_slots=15`,
`C_min=0.1`, and `C_max=50`.  The content-disjoint validation split, evaluation
seed, and holdout size are fixed across methods and training seeds.
Checkpointed per-arm state, incremental compact archives, strict completed-arm
validation, and an atomic allocation-level `job-status.json` make the same
immutable campaign resumable after a wall-time stop.

This matrix measures execution behavior, initialization/controller/noise/
client-count sensitivity, paired internal-loss/clipping trajectories, full-CDF
error and controller-oracle diagnostics, threshold stability, gradient
retention, FedAvg signal/noise, and paired inferential statistics.  Exact CDF
and oracle values remain `NON_DP_PRIVATE_DIAGNOSTIC`.  It does **not** implement
the paper's BoolQ, PIQA, WinoGrande, MedQuAD, LiveQA, or MEDIQA-Ans evaluators;
it does not add ChatGLM2-6B or Llama2-7B; and it does not resolve the
privacy-accounting omissions.  Until those gates are implemented, even a clean
campaign completion remains Level 1 rather than a paper benchmark
reproduction, a journal-complete result package, or evidence of an end-to-end
certified privacy guarantee.

## Minimum result package

A Level 2 result package must contain immutable references to the code and
environment, model and adapter manifests, benchmark manifests and overlap
reports, evaluator configuration, all four control arms, per-example outputs,
per-seed metrics, aggregate mean/standard deviation, and the exact claim level.
Missing gates must be reported as missing; they must not be inferred from a
successful Slurm exit code or from the existence of a nonzero adapter.
