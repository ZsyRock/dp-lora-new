# Broad-scope Full SlaClip study

This extension is a staged hypothesis test, not a claim that the original
DP-LoRA results have already been reproduced.  It expands the executable
training matrix from MedDialog × {BERT-base, GPT-2 small} to the three training
domains and four model families discussed by the DP-LoRA paper:

| Training profile | Models | Reconstruction boundary |
|---|---|---|
| MedDialog | BERT, GPT-2, ChatGLM2-6B, Llama2-7B | public reconstruction |
| SlimPajama | BERT, GPT-2, ChatGLM2-6B, Llama2-7B | deterministic public subset reconstruction |
| Finance | BERT, GPT-2, ChatGLM2-6B, Llama2-7B | paper source was not uniquely released |

Llama2 access requires the user to accept Meta's license.  The staging command
does not bypass that gate.  The finance profile must identify and hash the
chosen public reconstruction.  Neither profile may be labelled an exact paper
reproduction.

## Scientific design

1. Scan fixed `C = {0.1, 0.3, 1, 3, 10}` on three development seeds for every
   dataset/model setting.
2. Select fixed C using development loss only.  From its complete round
   trajectory, take five clipping-fraction quantiles and recover the exact
   near-zero endpoint from private diagnostic shards.
3. Convert each desired total clipping fraction to generalized Full SlaClip
   beta as `beta = total_target / (1 - near_zero_adjusted)`.  The implementation
   fails closed when near-zero telemetry is absent.  This calibration is
   explicitly `NON_DP_PRIVATE_DIAGNOSTIC`.
4. Screen three controller rates on disjoint development seeds, lock one
   setting, and compare it with the selected fixed C on at least ten fresh
   paired seeds.
5. Aggregate paired effects into baseline clipping-rate bins.  A bin is not
   eligible for a regime claim until it contains at least ten seed pairs from
   each of at least two dataset/model settings.

This can support statements such as “Full SlaClip was beneficial in this
observed clipping regime.”  It cannot establish a universal failure boundary
from clipping rate alone: endpoint signal-to-noise, client count, model,
dataset, learning rate, rank, and controller rate remain possible moderators.

SlaClip-Q is absent from the specification, planner, runner, and analysis.

## Input registration

Place standardized Parquet files under `train/`, `validation/`, and `test/`;
each file needs `src` and `tgt` columns.  Register a pinned domain plus one or
more already-downloaded model snapshots:

```bash
ENV_PREFIX=/absolute/versioned/environment
HF_HOME=/absolute/scratch/huggingface
"$ENV_PREFIX/bin/python" scripts/register_broad_scope_inputs.py \
  --profile slimpajama \
  --data-root /absolute/scratch/data \
  --dataset-root /absolute/scratch/data/slimpajama-subset \
  --dataset-repo-id cerebras/SlimPajama-627B \
  --dataset-revision 417f7eebaec467f82121948075e8b98d33ffb58a \
  --hf-home "$HF_HOME" \
  --model-snapshot bert=/absolute/snapshot/revision \
  --output /absolute/scratch/inputs/slimpajama/input-manifest.json
```

Create a private input index outside Git:

```json
{
  "schema_version": 1,
  "domains": {
    "meddialog": "/absolute/path/meddialog/input-manifest.json",
    "slimpajama": "/absolute/path/slimpajama/input-manifest.json",
    "finance": "/absolute/path/finance/input-manifest.json"
  }
}
```

Every manifest used for a given domain must contain the model snapshot needed
by each requested arm.

## Resumable Slurm stages

`hpc/broad_scope_stage.sbatch` runs arms sequentially inside one GPU allocation.
It accepts scheduler account, QoS/partition, typed GRES, memory, walltime,
signal, and logs through `sbatch` options, so those values can be chosen from a
fresh cluster query.  Reusing the same campaign root resumes a partial arm from
its checkpoint and skips validated completed arms.

The fixed stage writes `fixed-private-telemetry.csv` and a hash-locked
`adaptive-development-plan.lock.json`.  Submit the adaptive stage only with
that lock.  A final confirmation stage should be generated only after adaptive
development selection is frozen; do not reuse development seeds.

The current trainer reports internal held-out LM loss and detailed clipping,
CDF, threshold, noise, and update telemetry.  The nine downstream paper tasks
(BoolQ, PIQA, WinoGrande, FPB, FiQA-SA, TFNS, MedQuAD, LiveQA, MEDIQA-Ans) are
still an external-evaluation gate.  Journal claims about task accuracy require
that evaluator and cannot substitute internal loss.

## Paper-default baseline trajectory campaign

Before any wider fixed-C or Full SlaClip search, the repository can run a
paper-default baseline layer with `K_clients=5`, `T=50`, `B=8`, `sigma=2`,
`learning_rate=5e-4`, `C=10`, and rank `512`.  The frozen campaign
specification is `hpc/default-baseline-reproduction-spec.json` and its single
allocation worker is `hpc/default_baseline_reproduction.sbatch`.

The currently stageable matrix is MedDialog, a pinned public SlimPajama subset
reconstruction, and a pinned 9,540-row FinGPT finance reconstruction crossed
with BERT-base, GPT-2 small, and ChatGLM2-6B over three seeds (27 arms).  The
paper's Llama2-7B arm is recorded as blocked until the user accepts the model
license and configures authentication; the code does not bypass that gate.

This baseline-only campaign writes three plot-ready long tables:

- `baseline_round_telemetry.csv`: groupwise C, clipping, norm, retained-energy,
  update, loss, exact endpoint, predicted endpoint-noise, and endpoint-SNR
  trajectories;
- `baseline_client_telemetry.csv`: per-client/per-group norm, clip factor,
  removed gradient, retained energy, noise, update, and loss values; and
- `baseline_evaluation_telemetry.csv`: fixed-holdout loss by evaluation round.

Exact endpoint columns are `NON_DP_PRIVATE_DIAGNOSTIC` development telemetry.
They are suitable for selecting a later target-clipping candidate before fresh
confirmation seeds, but are not a privacy-preserving public release.
