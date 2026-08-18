#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DPLORA_FULL_SPEC_RELATIVE="hpc/paper-axes-full-slaclip-spec.json"
export DPLORA_FULL_ACCOUNT="${DPLORA_PAPER_AXES_ACCOUNT:-normal}"
export DPLORA_FULL_PARTITION="${DPLORA_PAPER_AXES_PARTITION:-a100}"
export DPLORA_FULL_GPU_GRES="${DPLORA_PAPER_AXES_GPU_GRES:-gpu:a100:1}"
export DPLORA_FULL_CPUS_PER_TASK="${DPLORA_PAPER_AXES_CPUS_PER_TASK:-4}"
export DPLORA_FULL_HOST_MEMORY="${DPLORA_PAPER_AXES_HOST_MEMORY:-32G}"
export DPLORA_FULL_LANE_MEMORY="${DPLORA_PAPER_AXES_LANE_MEMORY:-32G}"
export DPLORA_FULL_WALLTIME="${DPLORA_PAPER_AXES_WALLTIME:-1-12:00:00}"
export DPLORA_FULL_JOB_NAME="${DPLORA_PAPER_AXES_JOB_NAME:-dp-lora-paper-axes}"

user_name="$(id -un)"
user_home="$(getent passwd "$user_name" | cut -d: -f6)"
scratch_root="${DPLORA_PAPER_AXES_SCRATCH_ROOT:-${SCRATCH:-/scratch/$user_name}}"
export DPLORA_FULL_SCRATCH_ROOT="$scratch_root"
export DPLORA_FULL_RUN_ROOT="${DPLORA_PAPER_AXES_RUN_ROOT:-$scratch_root/runs/dp-lora-paper/paper-axes-full-slaclip-campaigns}"
export DPLORA_FULL_ARCHIVE_ROOT="${DPLORA_PAPER_AXES_ARCHIVE_ROOT:-$user_home/hpc/projects/dp-lora-paper/completed-runs/paper-axes-full-slaclip}"
export DPLORA_FULL_PRIVATE_KEY_ROOT="${DPLORA_PAPER_AXES_PRIVATE_KEY_ROOT:-$user_home/hpc/projects/dp-lora-paper/private-rng/paper-axes-full-slaclip}"

exec "$script_dir/submit_full_slaclip_campaign.sh" "$@"
