#!/usr/bin/env bash
set -euo pipefail

# Submit the five-profile, actual-clipping-target Full-SlaClip confirmation as
# one sequential A100 allocation.  The upstream oracle-ceiling campaign
# supplies only a frozen strong fixed-C development lock; all formal target
# comparisons use new seeds in this job.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DPLORA_STAGED_WORKER_RELATIVE="hpc/target_profile_campaign.sbatch"
export DPLORA_STAGED_SPEC_RELATIVE="hpc/target-profile-campaign-spec.json"
export DPLORA_STAGED_ORCHESTRATOR_RELATIVE="paper_repro/target_profile_campaign.py"

export DPLORA_STAGED_ACCOUNT="${DPLORA_TARGET_ACCOUNT:-normal}"
export DPLORA_STAGED_PARTITION="${DPLORA_TARGET_PARTITION:-a100}"
export DPLORA_STAGED_GPU_GRES="${DPLORA_TARGET_GPU_GRES:-gpu:a100:1}"
export DPLORA_STAGED_CPUS_PER_TASK="${DPLORA_TARGET_CPUS_PER_TASK:-4}"
export DPLORA_STAGED_HOST_MEMORY="${DPLORA_TARGET_HOST_MEMORY:-16G}"
export DPLORA_STAGED_LANE_MEMORY="${DPLORA_TARGET_STEP_MEMORY:-16G}"
export DPLORA_STAGED_WALLTIME="${DPLORA_TARGET_WALLTIME:-1-12:00:00}"
export DPLORA_STAGED_JOB_NAME="${DPLORA_TARGET_JOB_NAME:-dp-lora-target-profile}"
export DPLORA_STAGED_RUN_PREFIX="target-profile"

# Wait for the already-queued strong-fixed/oracle campaign without consuming
# another A100 concurrently.  afterok also prevents target calibration from
# running against an incomplete upstream lock.
export DPLORA_STAGED_DEPENDENCY="${DPLORA_TARGET_DEPENDENCY:-afterok:1367079}"

if [[ -n "${DPLORA_TARGET_RUN_ID:-}" ]]; then
    export DPLORA_STAGED_RUN_ID="$DPLORA_TARGET_RUN_ID"
fi
if [[ -n "${DPLORA_TARGET_REPO_DIR:-}" ]]; then
    export DPLORA_STAGED_REPO_DIR="$DPLORA_TARGET_REPO_DIR"
fi
if [[ -n "${DPLORA_TARGET_EXPECTED_SHA:-}" ]]; then
    export DPLORA_STAGED_EXPECTED_SHA="$DPLORA_TARGET_EXPECTED_SHA"
fi
if [[ -n "${DPLORA_TARGET_ENV_PREFIX:-}" ]]; then
    export DPLORA_STAGED_ENV_PREFIX="$DPLORA_TARGET_ENV_PREFIX"
fi
if [[ -n "${DPLORA_TARGET_SCRATCH_ROOT:-}" ]]; then
    export DPLORA_STAGED_SCRATCH_ROOT="$DPLORA_TARGET_SCRATCH_ROOT"
fi

user_name="$(id -un)"
user_home="$(getent passwd "$user_name" | cut -d: -f6)"
scratch_root="${DPLORA_TARGET_SCRATCH_ROOT:-${SCRATCH:-/scratch/$user_name}}"
export DPLORA_STAGED_RUN_ROOT="${DPLORA_TARGET_RUN_ROOT:-$scratch_root/runs/dp-lora-paper/target-profile-campaigns}"
export DPLORA_STAGED_ARCHIVE_ROOT="${DPLORA_TARGET_ARCHIVE_ROOT:-$user_home/hpc/projects/dp-lora-paper/completed-runs/target-profile}"
export DPLORA_STAGED_PRIVATE_KEY_ROOT="${DPLORA_TARGET_PRIVATE_KEY_ROOT:-$user_home/hpc/projects/dp-lora-paper/private-rng/target-profile}"

upstream_sha="2c3d2c6b5911a6b87812ac1c806fc67a85926554"
upstream_repository="${DPLORA_TARGET_UPSTREAM_REPOSITORY:-$user_home/src/DP-LoRA-paper-snapshots/$upstream_sha}"
export DPLORA_STAGED_UPSTREAM_CAMPAIGN_ROOT="${DPLORA_TARGET_UPSTREAM_CAMPAIGN_ROOT:-$scratch_root/runs/dp-lora-paper/oracle-ceiling-campaigns/oracle-ceiling-2c3d2c6-20260808a}"
export DPLORA_STAGED_UPSTREAM_REPOSITORY="$upstream_repository"
export DPLORA_STAGED_UPSTREAM_EXPECTED_SHA="${DPLORA_TARGET_UPSTREAM_EXPECTED_SHA:-$upstream_sha}"
export DPLORA_STAGED_UPSTREAM_SPEC="${DPLORA_TARGET_UPSTREAM_SPEC:-$upstream_repository/hpc/oracle-ceiling-campaign-spec.json}"
export DPLORA_STAGED_UPSTREAM_INPUT_MANIFEST="${DPLORA_TARGET_UPSTREAM_INPUT_MANIFEST:-$scratch_root/datasets/dp-lora-paper/input-manifest.json}"

exec "$script_dir/submit_staged_slaclip_tuned_fixed.sh" "$@"
