# Assessment of the completed legacy run

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
