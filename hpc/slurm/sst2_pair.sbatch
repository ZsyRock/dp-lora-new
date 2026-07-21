#!/usr/bin/env bash
#SBATCH --job-name=dp-lora-pair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_dir="${DP_LORA_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
env_name="${DP_LORA_CONDA_ENV:-dp-lora}"
clipping="${DP_LORA_CLIPPING:-both}"
initial_clip_norm="${DP_LORA_INITIAL_CLIP_NORM:-1.0}"
slaclip_eta="${DP_LORA_SLACLIP_ETA:-0.2}"
slaclip_beta="${DP_LORA_SLACLIP_BETA:-0.5}"
slaclip_c_min="${DP_LORA_SLACLIP_C_MIN:-0.1}"
slaclip_c_max="${DP_LORA_SLACLIP_C_MAX:-50.0}"
run_name="${DP_LORA_RUN_NAME:-slurm_${SLURM_JOB_ID:-manual}}"
conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${env_name}"
cd "${repo_dir}"

echo "commit=$(git rev-parse HEAD)"
echo "host=$(hostname)"
nvidia-smi || true

srun python examples/sst2_roberta.py \
    --clipping "${clipping}" \
    --lora-method ffa \
    --model roberta-base \
    --rank 8 \
    --epsilon 8 \
    --epochs 3 \
    --logical-batch-size 256 \
    --physical-batch-size 32 \
    --learning-rate 5e-4 \
    --initial-clip-norm "${initial_clip_norm}" \
    --slaclip-eta "${slaclip_eta}" \
    --slaclip-beta "${slaclip_beta}" \
    --slaclip-c-min "${slaclip_c_min}" \
    --slaclip-c-max "${slaclip_c_max}" \
    --seed 42 \
    --device cuda \
    --run-name "${run_name}" \
    --observe-private-gradients \
    --acknowledge-non-dp-diagnostics
