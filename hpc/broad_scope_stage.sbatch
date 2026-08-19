#!/usr/bin/env bash

#SBATCH --job-name=dp-lora-broad-scope
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Account, QoS/partition, typed GPU GRES, memory, walltime, signal and log
# paths are supplied at submission time after querying the live scheduler.
set -euo pipefail
umask 077

repository="${1:?missing immutable repository snapshot}"
expected_sha="${2:?missing expected repository SHA}"
env_prefix="${3:?missing environment prefix}"
scratch_root="${4:?missing scratch root}"
hf_home="${5:?missing Hugging Face cache root}"
input_index="${6:?missing broad input index}"
campaign_root="${7:?missing campaign root}"
private_key="${8:?missing persistent private RNG key}"
spec="${9:?missing broad-scope specification}"
expected_spec_sha256="${10:?missing specification file SHA-256}"
stage="${11:?missing fixed or adaptive stage}"
adaptive_plan="${12:--}"

python_bin="$env_prefix/bin/python"
campaign="$repository/paper_repro/broad_scope_campaign.py"
trainer="$repository/paper_repro/train_federated.py"
step_cpus="${SLURM_CPUS_PER_TASK:-0}"

if [[ "${SLURM_JOB_NUM_NODES:-0}" != 1 || "${SLURM_NTASKS:-0}" != 1 ]]; then
    echo "ERROR: broad campaign requires one node and one sequential task" >&2
    exit 2
fi
if [[ ! "$step_cpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid CPU allocation: $step_cpus" >&2
    exit 2
fi
if [[ "$stage" != fixed && "$stage" != adaptive ]]; then
    echo "ERROR: stage must be fixed or adaptive" >&2
    exit 2
fi
for required in "$python_bin" "$campaign" "$trainer" "$input_index" "$spec" "$private_key"; do
    [[ -e "$required" ]] || { echo "ERROR: missing runtime input: $required" >&2; exit 2; }
done
if [[ "$stage" == adaptive && ! -f "$adaptive_plan" ]]; then
    echo "ERROR: adaptive stage requires a locked adaptive plan" >&2
    exit 2
fi
if [[ "$(git -C "$repository" rev-parse HEAD)" != "$expected_sha" || -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: immutable repository snapshot identity failed" >&2
    exit 2
fi
if [[ "$(sha256sum "$spec" | awk '{print $1}')" != "$expected_spec_sha256" ]]; then
    echo "ERROR: broad specification changed after submission" >&2
    exit 2
fi

export HF_HOME="$hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$step_cpus"
export TMPDIR="${SLURM_TMPDIR:-$scratch_root/tmp/dp-lora-broad-$SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$campaign_root"
chmod 700 "$TMPDIR" "$campaign_root"

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "repository_sha=$expected_sha"
echo "stage=$stage"
echo "campaign_root=$campaign_root"
nvidia-smi

srun --exact --nodes=1 --ntasks=1 "--cpus-per-task=$step_cpus" \
    "$python_bin" -m pip check
srun --exact --nodes=1 --ntasks=1 "--cpus-per-task=$step_cpus" \
    "$python_bin" "$campaign" validate-spec --spec "$spec"

run_args=(
    run-stage --spec "$spec" --stage "$stage"
    --input-index "$input_index" --campaign-root "$campaign_root"
    --repository "$repository" --python-bin "$python_bin"
    --private-key "$private_key"
    --output "$campaign_root/$stage-stage-summary.json"
)
if [[ "$stage" == adaptive ]]; then
    run_args+=(--adaptive-plan "$adaptive_plan")
fi
srun --exact --nodes=1 --ntasks=1 "--cpus-per-task=$step_cpus" \
    "$python_bin" "$campaign" "${run_args[@]}"

if [[ "$stage" == fixed ]]; then
    "$python_bin" "$campaign" collect-fixed --spec "$spec" \
        --campaign-root "$campaign_root" \
        --output "$campaign_root/fixed-private-telemetry.csv"
    "$python_bin" "$campaign" derive-adaptive --spec "$spec" \
        --baseline-telemetry "$campaign_root/fixed-private-telemetry.csv" \
        --output "$campaign_root/adaptive-development-plan.lock.json"
fi

echo "stage_status=COMPLETED"
