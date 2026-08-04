#!/usr/bin/env bash
set -euo pipefail

# MedDialog × {BERT-base, GPT-2 small}, K=5 full two-endpoint SlaClip.
# beta is the base target clipped fraction of the remaining non-small mass;
# the per-round target is beta * (1 - s_hat[K-1]/C_t), not a fixed quantile.
export DPLORA_FULL_SPEC_RELATIVE=hpc/full-slaclip-k5-baseline-range-spec.json
export DPLORA_FULL_PARTITION=l4
export DPLORA_FULL_GPU_GRES=gpu:l4:2
export DPLORA_FULL_EXPECTED_GPU=L4
export DPLORA_FULL_MIN_VRAM_GIB=20
export DPLORA_FULL_CPUS_PER_TASK=4
export DPLORA_FULL_HOST_MEMORY=24G
export DPLORA_FULL_LANE_MEMORY=12G
export DPLORA_FULL_WALLTIME=01:00:00
export DPLORA_FULL_JOB_NAME=dp-lora-slaclip-k5-range5

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_directory/submit_full_slaclip_campaign.sh" "$@"
