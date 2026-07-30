#!/usr/bin/env bash
set -euo pipefail

# MedDialog × {BERT-base, GPT-2 small} development screen.  This remains the
# full two-endpoint SlaClip controller: each listed beta is the base target
# clipped fraction and the round-t target is beta * (1 - s_hat[K-1]/C_t).
export DPLORA_FULL_SPEC_RELATIVE=hpc/full-slaclip-beta5-screen-spec.json
export DPLORA_FULL_PARTITION=l4
export DPLORA_FULL_GPU_GRES=gpu:l4:2
export DPLORA_FULL_EXPECTED_GPU=L4
export DPLORA_FULL_MIN_VRAM_GIB=20
export DPLORA_FULL_CPUS_PER_TASK=4
export DPLORA_FULL_HOST_MEMORY=64G
export DPLORA_FULL_LANE_MEMORY=32G
export DPLORA_FULL_WALLTIME=1-12:00:00
export DPLORA_FULL_JOB_NAME=dp-lora-slaclip-beta5

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_directory/submit_full_slaclip_campaign.sh" "$@"
