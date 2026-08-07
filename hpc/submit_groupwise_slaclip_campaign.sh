#!/usr/bin/env bash
set -euo pipefail

# Reuse the mature snapshot, environment, offline-input, scheduler and
# test-only gates while selecting the independent groupwise worker/spec/module.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DPLORA_STAGED_WORKER_RELATIVE="hpc/groupwise_slaclip_campaign.sbatch"
export DPLORA_STAGED_SPEC_RELATIVE="hpc/groupwise-slaclip-campaign-spec.json"
export DPLORA_STAGED_ORCHESTRATOR_RELATIVE="paper_repro/groupwise_slaclip_campaign.py"

export DPLORA_STAGED_ACCOUNT="${DPLORA_GROUPWISE_ACCOUNT:-normal}"
export DPLORA_STAGED_PARTITION="${DPLORA_GROUPWISE_PARTITION:-scavenger_l4}"
export DPLORA_STAGED_GPU_GRES="${DPLORA_GROUPWISE_GPU_GRES:-gpu:l4swarm:1}"
export DPLORA_STAGED_CPUS_PER_TASK="${DPLORA_GROUPWISE_CPUS_PER_TASK:-4}"
export DPLORA_STAGED_HOST_MEMORY="${DPLORA_GROUPWISE_HOST_MEMORY:-16G}"
export DPLORA_STAGED_LANE_MEMORY="${DPLORA_GROUPWISE_STEP_MEMORY:-16G}"
export DPLORA_STAGED_WALLTIME="${DPLORA_GROUPWISE_WALLTIME:-12:00:00}"
export DPLORA_STAGED_JOB_NAME="${DPLORA_GROUPWISE_JOB_NAME:-dp-lora-groupwise-slaclip}"
export DPLORA_STAGED_RUN_PREFIX="groupwise-slaclip"

if [[ -n "${DPLORA_GROUPWISE_RUN_ID:-}" ]]; then
    export DPLORA_STAGED_RUN_ID="$DPLORA_GROUPWISE_RUN_ID"
fi
if [[ -n "${DPLORA_GROUPWISE_REPO_DIR:-}" ]]; then
    export DPLORA_STAGED_REPO_DIR="$DPLORA_GROUPWISE_REPO_DIR"
fi
if [[ -n "${DPLORA_GROUPWISE_EXPECTED_SHA:-}" ]]; then
    export DPLORA_STAGED_EXPECTED_SHA="$DPLORA_GROUPWISE_EXPECTED_SHA"
fi
if [[ -n "${DPLORA_GROUPWISE_ENV_PREFIX:-}" ]]; then
    export DPLORA_STAGED_ENV_PREFIX="$DPLORA_GROUPWISE_ENV_PREFIX"
fi
if [[ -n "${DPLORA_GROUPWISE_SCRATCH_ROOT:-}" ]]; then
    export DPLORA_STAGED_SCRATCH_ROOT="$DPLORA_GROUPWISE_SCRATCH_ROOT"
fi

user_name="$(id -un)"
user_home="$(getent passwd "$user_name" | cut -d: -f6)"
scratch_root="${DPLORA_GROUPWISE_SCRATCH_ROOT:-${SCRATCH:-/scratch/$user_name}}"
export DPLORA_STAGED_RUN_ROOT="${DPLORA_GROUPWISE_RUN_ROOT:-$scratch_root/runs/dp-lora-paper/groupwise-slaclip-campaigns}"
export DPLORA_STAGED_ARCHIVE_ROOT="${DPLORA_GROUPWISE_ARCHIVE_ROOT:-$user_home/hpc/projects/dp-lora-paper/completed-runs/groupwise-slaclip}"
export DPLORA_STAGED_PRIVATE_KEY_ROOT="${DPLORA_GROUPWISE_PRIVATE_KEY_ROOT:-$user_home/hpc/projects/dp-lora-paper/private-rng/groupwise-slaclip}"

exec "$script_dir/submit_staged_slaclip_tuned_fixed.sh" "$@"
