#!/usr/bin/env bash
set -euo pipefail

# Reuse the immutable-snapshot, offline-input and scheduler gates while
# selecting the strong groupwise-fixed versus exact-oracle campaign.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DPLORA_STAGED_WORKER_RELATIVE="hpc/oracle_ceiling_campaign.sbatch"
export DPLORA_STAGED_SPEC_RELATIVE="hpc/oracle-ceiling-campaign-spec.json"
export DPLORA_STAGED_ORCHESTRATOR_RELATIVE="paper_repro/oracle_ceiling_campaign.py"

export DPLORA_STAGED_ACCOUNT="${DPLORA_ORACLE_ACCOUNT:-normal}"
export DPLORA_STAGED_PARTITION="${DPLORA_ORACLE_PARTITION:-a100}"
export DPLORA_STAGED_GPU_GRES="${DPLORA_ORACLE_GPU_GRES:-gpu:a100:1}"
export DPLORA_STAGED_CPUS_PER_TASK="${DPLORA_ORACLE_CPUS_PER_TASK:-4}"
export DPLORA_STAGED_HOST_MEMORY="${DPLORA_ORACLE_HOST_MEMORY:-16G}"
export DPLORA_STAGED_LANE_MEMORY="${DPLORA_ORACLE_STEP_MEMORY:-16G}"
export DPLORA_STAGED_WALLTIME="${DPLORA_ORACLE_WALLTIME:-1-12:00:00}"
export DPLORA_STAGED_JOB_NAME="${DPLORA_ORACLE_JOB_NAME:-dp-lora-oracle-ceiling}"
export DPLORA_STAGED_RUN_PREFIX="oracle-ceiling"

if [[ -n "${DPLORA_ORACLE_DEPENDENCY:-}" ]]; then
    export DPLORA_STAGED_DEPENDENCY="$DPLORA_ORACLE_DEPENDENCY"
fi
if [[ -n "${DPLORA_ORACLE_RUN_ID:-}" ]]; then
    export DPLORA_STAGED_RUN_ID="$DPLORA_ORACLE_RUN_ID"
fi
if [[ -n "${DPLORA_ORACLE_REPO_DIR:-}" ]]; then
    export DPLORA_STAGED_REPO_DIR="$DPLORA_ORACLE_REPO_DIR"
fi
if [[ -n "${DPLORA_ORACLE_EXPECTED_SHA:-}" ]]; then
    export DPLORA_STAGED_EXPECTED_SHA="$DPLORA_ORACLE_EXPECTED_SHA"
fi
if [[ -n "${DPLORA_ORACLE_ENV_PREFIX:-}" ]]; then
    export DPLORA_STAGED_ENV_PREFIX="$DPLORA_ORACLE_ENV_PREFIX"
fi
if [[ -n "${DPLORA_ORACLE_SCRATCH_ROOT:-}" ]]; then
    export DPLORA_STAGED_SCRATCH_ROOT="$DPLORA_ORACLE_SCRATCH_ROOT"
fi

user_name="$(id -un)"
user_home="$(getent passwd "$user_name" | cut -d: -f6)"
scratch_root="${DPLORA_ORACLE_SCRATCH_ROOT:-${SCRATCH:-/scratch/$user_name}}"
export DPLORA_STAGED_RUN_ROOT="${DPLORA_ORACLE_RUN_ROOT:-$scratch_root/runs/dp-lora-paper/oracle-ceiling-campaigns}"
export DPLORA_STAGED_ARCHIVE_ROOT="${DPLORA_ORACLE_ARCHIVE_ROOT:-$user_home/hpc/projects/dp-lora-paper/completed-runs/oracle-ceiling}"
export DPLORA_STAGED_PRIVATE_KEY_ROOT="${DPLORA_ORACLE_PRIVATE_KEY_ROOT:-$user_home/hpc/projects/dp-lora-paper/private-rng/oracle-ceiling}"

exec "$script_dir/submit_staged_slaclip_tuned_fixed.sh" "$@"
