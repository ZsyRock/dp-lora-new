#!/usr/bin/env bash
set -euo pipefail

# One resumable allocation for the frozen K=5 confirmation and all planned
# stability/fair-baseline diagnostics.  Live site policy permits only one L4
# per scavenger_l4 job, so the generic worker executes deterministic waves
# sequentially without an array or child sbatch submissions.
export DPLORA_FULL_SPEC_RELATIVE=hpc/full-slaclip-k5-stability-suite-spec.json
export DPLORA_FULL_PARTITION=scavenger_l4
export DPLORA_FULL_GPU_GRES=gpu:l4swarm:1
export DPLORA_FULL_EXPECTED_GPU=L4
export DPLORA_FULL_MIN_VRAM_GIB=20
export DPLORA_FULL_CPUS_PER_TASK=4
export DPLORA_FULL_HOST_MEMORY=12G
export DPLORA_FULL_LANE_MEMORY=12G
export DPLORA_FULL_WALLTIME=12:00:00
export DPLORA_FULL_JOB_NAME=dp-lora-slaclip-k5-stability

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_directory/submit_full_slaclip_campaign.sh" "$@"
