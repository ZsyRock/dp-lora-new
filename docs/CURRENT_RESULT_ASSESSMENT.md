# Assessment of the completed legacy run and current evidence gap

This note records why Slurm job `1298681` is not evidence of numerical paper
reproduction, while also explaining why its `00:07:31` wall time is plausible.
It applies only to the legacy schema-1 run produced by commit `2dc4c7b...`; the
new runner deliberately gives it a different configuration fingerprint.

## Execution evidence

Slurm recorded job `1298681` as `COMPLETED`, exit code `0:0`, on one GPU.  Its
wall time decomposed into a 9-second CUDA check, a 2-minute-26-second BERT/GPT-2
preflight smoke step, and a 3-minute-40-second formal runner step, plus input
checks and archival overhead.  The root formal runner reported 188.1 seconds
of model work.

That duration is not intrinsically suspicious.  Algorithm 1 uses 50 rounds,
five logical clients, and one batch per client/round: only 250 client batches
per model, not an epoch over all 257,469 staged records.  BERT-base and GPT-2
small were run sequentially on an A100.

The saved run reported:

| Model | Client steps | Internal loss, round 0 | Internal loss, round 50 | Any-group clipping |
| --- | ---: | ---: | ---: | ---: |
| BERT-base | 250 | 2.9298 | 4.0240 | 40/250 (16.0%) |
| GPT-2 small | 250 | 4.0999 | 4.3575 | 0/250 (0.0%) |

At round 50, BERT clipped A for 4/5 clients and B for 5/5 clients.  The logged
Gaussian gradient norms were about 75,245 for each BERT A/B group, whereas raw
gradient norms were about 11.1 and 13.2.  GPT-2 raw group norms remained below
`C=10`, while logged noise norms were about 43,442 for A and 75,241 for B.
The internal diagnostic worsened for both models rather than showing utility
improvement.

## Claim decision

The job is evidence that the earlier reconstructed loop really executed and
saved nonzero adapters.  It is **not a successful reproduction of the paper's
reported results**, for four independent reasons:

1. none of BoolQ, PIQA, WinoGrande, MedQuAD, LiveQA, or MEDIQA-Ans was evaluated;
2. the paper's missing prompts, task interfaces, scorers, and privacy-accounting
   contract were not recovered;
3. there was no matched no-DP or clip-only control, so degradation could not be
   attributed to clipping versus noise; and
4. the legacy internal validation selection was not content-disjoint from the
   training union and therefore was not a valid held-out utility measure.

The correct label is **legacy Level-1 algorithm execution, with an invalid
internal holdout for utility inference**.  Its statistics motivated the new
matched-arm, content-isolated, checkpoint-bound instrumentation; they must not
be mixed with results from the new schema.

For descriptive context only, a retrospective recomputation over the 50 legacy
round summaries found per-round clipping-fraction medians of zero for BERT A,
BERT B, GPT-2 A, and GPT-2 B.  BERT's across-all-client means were still 6.4%
for A and 16.0% for B, illustrating how strongly a clipping summary depends on
its reducer.  These values are neither targets nor calibration inputs for the
active method.  Full SlaClip derives its dynamic target each round from the
two endpoints of the *noisy* CDF proxy; it does not consume a fixed target or
a baseline-derived median.  Any older fixed-target artifact is retained only
for historical audit and is not evidence for full SlaClip.

## What the current full-SlaClip campaign can establish

The current plan is a single resumable Slurm allocation containing 60 runner
arms, with every arm training both BERT-base and GPT-2 small.  Its primary
matched comparison crosses fixed DP-LoRA and full `slaclip_dp_lora` over the
paper's initialization sweep `C_0 in {0.1, 1, 5, 10, 20}` and seeds
`42, 43, 44` (30 arms).  It adds 24 full-SlaClip
controller-sensitivity arms at `C_0=10` over seeds `42..44`,
`eta in {0.05, 0.1, 0.2}`, and `beta in {0.5, 0.9, 0.99}`, excluding the
already-covered default pair.  Six no-DP/clip-only arms at `C_0=10` and seeds
`42..44` complete the mechanism controls.  Two GPU lanes execute this manifest
inside the same job; checkpointed state and compact incremental archives make
it resumable without splitting the work into separately queued jobs.

This design is materially stronger than one seed at the paper's default
`C=10`: it can test whether an apparent gain is paired across seeds, robust to
the initial threshold, and stable under controller settings while separating
clipping from Gaussian-noise effects.  The most useful outputs are the
per-round internal loss, actual clipping fraction, raw/clipped/noisy norms,
signal-to-noise and cosine statistics, both noisy CDF endpoints, exact CDF
diagnostics, dynamic `gamma_t`, threshold trajectories, bound hits, paired
per-seed deltas, and aggregate mean/standard deviation.

No effectiveness result follows from the specification alone.  Such a claim
requires all scheduled arms to complete (or be explicitly reported missing),
their artifacts to pass strict revalidation, and the paired summaries to show
a repeatable utility improvement rather than only a lower noise norm.  Even a
successful campaign would remain a **Level-1 mechanism study**: it does not
evaluate BoolQ, PIQA, WinoGrande, MedQuAD, LiveQA, or MEDIQA-Ans; it does not
implement the paper's ChatGLM2-6B or Llama2-7B protocols; and its adaptation of
full SlaClip to five client aggregate-gradient records has no independently
certified end-to-end privacy accountant.  Those are the next gates before a
paper benchmark reproduction or privacy claim is justified.
