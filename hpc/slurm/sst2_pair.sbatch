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
conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${env_name}"
cd "${repo_dir}"

echo "commit=$(git rev-parse HEAD)"
echo "host=$(hostname)"
nvidia-smi || true

srun python examples/sst2_roberta.py \
    --clipping both \
    --lora-method ffa \
    --model roberta-base \
    --rank 8 \
    --epsilon 8 \
    --epochs 3 \
    --logical-batch-size 256 \
    --physical-batch-size 32 \
    --learning-rate 5e-4 \
    --max-grad-norm 1 \
    --seed 42 \
    --device cuda
