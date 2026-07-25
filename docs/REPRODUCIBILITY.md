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

Every benchmark study must evaluate these four arms through the identical
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

### SlaClip-Q extension boundary

The repository's fixed-target adaptive arm is `slaclip_q_dp_lora`. It is the
SlaClip-Q ablation, not full SlaClip: for every model and LoRA A/B group it
uses `gamma = 1 - q_target` and updates the clipping threshold only after all
clients in a round have completed. Its target is the median of the matched
fixed-threshold baseline's per-round actual clipping fractions for that same
model/group. The calibration manifest binds the baseline configuration,
summaries, complete round-shard prefix, reducer, targets, and hashes.

The slack coordinates and clipped gradient form one bounded vector per client
and group and receive independent Gaussian coordinates at scale `sigma*C_t`.
This preserves the intended joint-release construction inside the declared
federated adaptation, but it does not supply the missing end-to-end accountant.
Moreover, selecting `q_target` from exact baseline diagnostics is itself a
data-dependent non-DP calibration step. Therefore the current SlaClip-Q study
must retain `NON_DP_PRIVATE_DIAGNOSTIC`, `epsilon=null`, and
`end_to_end_dp_certified=false`. A privacy claim would require a public or
independent calibration set, or a separately private target release and full
composition analysis.

## Minimum result package

A Level 2 result package must contain immutable references to the code and
environment, model and adapter manifests, benchmark manifests and overlap
reports, evaluator configuration, all four control arms, per-example outputs,
per-seed metrics, aggregate mean/standard deviation, and the exact claim level.
Missing gates must be reported as missing; they must not be inferred from a
successful Slurm exit code or from the existence of a nonzero adapter.
