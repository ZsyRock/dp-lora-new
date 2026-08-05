#!/usr/bin/env bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

usage() {
    cat >&2 <<'USAGE'
Usage: hpc/submit_staged_slaclip_tuned_fixed.sh [--resume] [--test-only]

Submit the full three-stage tuned-fixed versus full-SlaClip campaign as one
Slurm allocation.  No arrays or child sbatch jobs are created.

--resume requires DPLORA_STAGED_RUN_ID to name the partial campaign.
--test-only runs every login-node and scheduler gate without queuing the job.

Important overrides:
  DPLORA_STAGED_RUN_ID
  DPLORA_STAGED_REPO_DIR
  DPLORA_STAGED_EXPECTED_SHA
  DPLORA_STAGED_ENV_PREFIX
  DPLORA_STAGED_SCRATCH_ROOT
  DPLORA_STAGED_ARCHIVE_ROOT
  DPLORA_STAGED_ACCOUNT             default: normal
  DPLORA_STAGED_PARTITION           default: scavenger_l4
  DPLORA_STAGED_GPU_GRES            default: gpu:l4swarm:1
  DPLORA_STAGED_CPUS_PER_TASK       default: 4
  DPLORA_STAGED_HOST_MEMORY         default: 12G
  DPLORA_STAGED_LANE_MEMORY         default: 12G
  DPLORA_STAGED_WALLTIME            default: 12:00:00

The default scavenger partition can cancel the allocation for higher-priority
work.  Resume the same DPLORA_STAGED_RUN_ID with --resume; completed arms and
both immutable selection locks are revalidated and reused.
USAGE
}

resume=0
test_only=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            [[ "$resume" -eq 0 ]] || { echo "ERROR: duplicate --resume" >&2; exit 2; }
            resume=1
            ;;
        --test-only)
            [[ "$test_only" -eq 0 ]] || { echo "ERROR: duplicate --test-only" >&2; exit 2; }
            test_only=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
    shift
done

user_name="$(id -un)"
user_home="$(getent passwd "$user_name" | cut -d: -f6)"
if [[ -z "$user_home" || ! -d "$user_home" ]]; then
    echo "ERROR: could not resolve persistent home" >&2
    exit 1
fi
scratch_root="${DPLORA_STAGED_SCRATCH_ROOT:-${SCRATCH:-/scratch/$user_name}}"
repo_dir="${DPLORA_STAGED_REPO_DIR:-$user_home/src/DP-LoRA-paper-repro}"
repo_dir="$(readlink -f "$repo_dir")"
git_bin="$(command -v git || true)"
if [[ -z "$git_bin" || "$($git_bin -C "$repo_dir" rev-parse --is-inside-work-tree 2>/dev/null || true)" != true ]]; then
    echo "ERROR: staged repository is not a Git worktree: $repo_dir" >&2
    exit 1
fi
if [[ -n "$($git_bin -C "$repo_dir" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: commit or remove every worktree change before submission" >&2
    exit 1
fi
actual_sha="$($git_bin -C "$repo_dir" rev-parse HEAD)"
expected_sha="${DPLORA_STAGED_EXPECTED_SHA:-$actual_sha}"
if [[ ! "$expected_sha" =~ ^[0-9a-f]{40}$ || "$actual_sha" != "$expected_sha" ]]; then
    echo "ERROR: requested full SHA does not match the clean worktree HEAD" >&2
    exit 1
fi
short_sha="${expected_sha:0:7}"

snapshot_root="${DPLORA_STAGED_SNAPSHOT_ROOT:-$user_home/src/DP-LoRA-paper-snapshots}"
snapshot_repo="$snapshot_root/$expected_sha"
mkdir -p "$snapshot_root"
chmod 700 "$snapshot_root"
if [[ ! -e "$snapshot_repo" ]]; then
    "$git_bin" -C "$repo_dir" worktree add --detach "$snapshot_repo" "$expected_sha"
fi
if [[ "$($git_bin -C "$snapshot_repo" rev-parse HEAD 2>/dev/null || true)" != "$expected_sha" ]]; then
    echo "ERROR: immutable snapshot has the wrong SHA" >&2
    exit 1
fi
if [[ -n "$($git_bin -C "$snapshot_repo" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: immutable snapshot is dirty" >&2
    exit 1
fi

env_prefix="${DPLORA_STAGED_ENV_PREFIX:-$scratch_root/envs/dp-lora-paper-$short_sha}"
env_prefix="$(readlink -f "$env_prefix")"
python_bin="$env_prefix/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: versioned environment interpreter is missing: $python_bin" >&2
    echo "Set DPLORA_STAGED_ENV_PREFIX to a dependency-identical tested environment." >&2
    exit 1
fi

worker="$snapshot_repo/hpc/staged_slaclip_tuned_fixed.sbatch"
submit_snapshot="$snapshot_repo/hpc/submit_staged_slaclip_tuned_fixed.sh"
exit_policy="$snapshot_repo/hpc/full_slaclip_exit_policy.sh"
spec="$snapshot_repo/hpc/staged-slaclip-tuned-fixed-spec.json"
staged="$snapshot_repo/paper_repro/staged_slaclip_campaign.py"
full="$snapshot_repo/paper_repro/full_slaclip_campaign.py"
trainer="$snapshot_repo/paper_repro/train_federated.py"
stage_script="$snapshot_repo/scripts/stage_paper_inputs.sh"
runtime_lock="$snapshot_repo/environment/paper-repro-runtime.lock"
for required in "$worker" "$submit_snapshot" "$exit_policy" "$spec" "$staged" "$full" "$trainer" "$stage_script" "$runtime_lock"; do
    [[ -f "$required" ]] || { echo "ERROR: snapshot is missing $required" >&2; exit 1; }
done
bash -n "$worker" "$submit_snapshot" "$exit_policy"
"$python_bin" - "$staged" "$full" <<'PY'
import sys
from pathlib import Path
for name in sys.argv[1:]:
    source = Path(name).read_bytes()
    compile(source, name, "exec")
    print(f"python_source_compiled={name}")
PY
"$python_bin" -m pip check
"$python_bin" - "$runtime_lock" <<'PY'
import importlib.metadata
import platform
import sys
from pathlib import Path
expected = {}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.count("==") != 1:
        raise SystemExit(f"invalid runtime-lock line: {raw!r}")
    name, version = line.split("==", 1)
    if not name or not version or name in expected:
        raise SystemExit(f"invalid/duplicate runtime-lock entry: {raw!r}")
    expected[name] = version
if expected.pop("python", None) != platform.python_version():
    raise SystemExit("Python version does not match runtime lock")
for name, version in expected.items():
    if importlib.metadata.version(name) != version:
        raise SystemExit(f"runtime version mismatch for {name}")
print(f"runtime_lock_verified={sys.argv[1]}")
PY
"$python_bin" "$staged" validate-spec --spec "$spec"
PYTHONPATH="$snapshot_repo" "$python_bin" -c \
    'from paper_repro.reproducibility import METHOD_SPECS; assert "slaclip_dp_lora" in METHOD_SPECS; assert "slaclip_q_dp_lora" not in METHOD_SPECS'

hf_home="${DPLORA_STAGED_HF_HOME:-$scratch_root/cache/dp-lora-paper/huggingface}"
data_root="${DPLORA_STAGED_DATA_ROOT:-$scratch_root/datasets/dp-lora-paper}"
input_manifest="${DPLORA_STAGED_INPUT_MANIFEST:-$data_root/input-manifest.json}"
if [[ ! -f "$input_manifest" ]]; then
    echo "ERROR: immutable input manifest is missing: $input_manifest" >&2
    exit 1
fi
DPLORA_PAPER_PYTHON="$python_bin" \
DPLORA_PAPER_SCRATCH_ROOT="$scratch_root" \
DPLORA_PAPER_HF_HOME="$hf_home" \
DPLORA_PAPER_DATA_ROOT="$data_root" \
DPLORA_PAPER_INPUT_MANIFEST="$input_manifest" \
"$stage_script" --check-only
"$python_bin" "$trainer" --check-inputs --input-manifest "$input_manifest"

if [[ "$resume" -eq 1 && -z "${DPLORA_STAGED_RUN_ID:-}" ]]; then
    echo "ERROR: --resume requires DPLORA_STAGED_RUN_ID" >&2
    exit 2
fi
if [[ -n "${DPLORA_STAGED_RUN_ID:-}" ]]; then
    run_id="$DPLORA_STAGED_RUN_ID"
else
    run_id="staged-slaclip-${short_sha}-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: unsafe run ID: $run_id" >&2
    exit 1
fi

run_root="${DPLORA_STAGED_RUN_ROOT:-$scratch_root/runs/dp-lora-paper/staged-slaclip-campaigns}"
campaign_root="$run_root/$run_id"
archive_base="${DPLORA_STAGED_ARCHIVE_ROOT:-$user_home/hpc/projects/dp-lora-paper/completed-runs/staged-slaclip}"
archive_root="$archive_base/$run_id"
private_key_root="${DPLORA_STAGED_PRIVATE_KEY_ROOT:-$user_home/hpc/projects/dp-lora-paper/private-rng/staged-slaclip}"
private_key="$private_key_root/$run_id.key"
if [[ "$resume" -eq 1 ]]; then
    [[ -d "$campaign_root" ]] || { echo "ERROR: resume campaign is missing" >&2; exit 1; }
    [[ -f "$private_key" ]] || { echo "ERROR: resume RNG key is missing" >&2; exit 1; }
    master_manifest="$campaign_root/runtime-manifest.json"
    [[ -f "$master_manifest" ]] || { echo "ERROR: master manifest is missing" >&2; exit 1; }
    stale_args=(
        recover-stale-job-status
        --campaign-root "$campaign_root"
        --runtime-manifest "$master_manifest"
        --repository-sha "$expected_sha"
    )
    if [[ "$test_only" -eq 1 ]]; then
        stale_args+=(--check-only)
    fi
    "$python_bin" "$full" "${stale_args[@]}"
else
    [[ ! -e "$campaign_root" ]] || { echo "ERROR: refusing to overwrite campaign" >&2; exit 1; }
    [[ ! -e "$archive_root" ]] || { echo "ERROR: refusing to overwrite archive" >&2; exit 1; }
    [[ ! -e "$private_key" ]] || { echo "ERROR: refusing to reuse RNG key" >&2; exit 1; }
fi
mkdir -p "$run_root/slurm" "$archive_base" "$private_key_root"
chmod 700 "$run_root" "$run_root/slurm" "$archive_base" "$private_key_root"

account="${DPLORA_STAGED_ACCOUNT:-normal}"
partition="${DPLORA_STAGED_PARTITION:-scavenger_l4}"
gpu_gres="${DPLORA_STAGED_GPU_GRES:-gpu:l4swarm:1}"
cpus_per_task="${DPLORA_STAGED_CPUS_PER_TASK:-4}"
host_memory="${DPLORA_STAGED_HOST_MEMORY:-12G}"
lane_memory="${DPLORA_STAGED_LANE_MEMORY:-12G}"
walltime="${DPLORA_STAGED_WALLTIME:-12:00:00}"
job_name="${DPLORA_STAGED_JOB_NAME:-dp-lora-staged-slaclip}"
if [[ "$gpu_gres" =~ ^gpu:([A-Za-z0-9_-]+):([12])$ ]]; then
    gpu_type="${BASH_REMATCH[1]}"
    gpu_lanes="${BASH_REMATCH[2]}"
else
    echo "ERROR: typed allocation must contain one or two GPUs" >&2
    exit 1
fi
lane_gres="gpu:$gpu_type:1"
case "${gpu_type,,}" in
    l4|l4swarm) default_expected_gpu="L4"; default_min_vram_gib=20 ;;
    a100|a100swarm) default_expected_gpu="A100"; default_min_vram_gib=39 ;;
    h200) default_expected_gpu="H200"; default_min_vram_gib=100 ;;
    *) default_expected_gpu="$gpu_type"; default_min_vram_gib=1 ;;
esac
expected_gpu="${DPLORA_STAGED_EXPECTED_GPU:-$default_expected_gpu}"
min_vram_gib="${DPLORA_STAGED_MIN_VRAM_GIB:-$default_min_vram_gib}"
if [[ ! "$cpus_per_task" =~ ^[1-9][0-9]*$ || "$cpus_per_task" -gt 12 ]]; then
    echo "ERROR: CPUs per lane must be 1..12" >&2
    exit 1
fi
if [[ ! "$walltime" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "ERROR: invalid walltime syntax" >&2
    exit 1
fi
if [[ ! "$job_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: unsafe job name" >&2
    exit 1
fi

sbatch_bin="$(command -v sbatch || true)"
[[ -n "$sbatch_bin" ]] || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
spec_sha256="$(sha256sum "$spec" | awk '{print $1}')"
sbatch_args=(
    --parsable
    "--account=$account"
    "--partition=$partition"
    "--gres=$gpu_gres"
    --nodes=1
    "--ntasks=$gpu_lanes"
    "--cpus-per-task=$cpus_per_task"
    "--mem=$host_memory"
    "--time=$walltime"
    --signal=B:USR1@600
    "--job-name=$job_name"
    "--output=$run_root/slurm/%x-%j.out"
    "--error=$run_root/slurm/%x-%j.err"
    --export=NONE
)
if [[ "$test_only" -eq 1 ]]; then
    sbatch_args+=(--test-only)
fi

echo "Campaign:           $run_id"
echo "Resume:             $resume"
echo "Source snapshot:    $snapshot_repo"
echo "Pinned SHA:         $expected_sha"
echo "Environment:        $env_prefix"
echo "Input manifest:     $input_manifest"
echo "Scratch output:     $campaign_root"
echo "Persistent archive: $archive_root"
echo "Resources: account=$account partition=$partition gres=$gpu_gres nodes=1 lanes=$gpu_lanes cpus/lane=$cpus_per_task mem=$host_memory lane_mem=$lane_memory time=$walltime"
echo "GPU gate: name_contains=$expected_gpu min_vram_gib=$min_vram_gib"
if [[ "$partition" == scavenger_* ]]; then
    echo "Preemption note: this partition may CANCEL the job; resume the same run ID with --resume."
fi

"$sbatch_bin" "${sbatch_args[@]}" "$worker" \
    "$snapshot_repo" \
    "$expected_sha" \
    "$env_prefix" \
    "$scratch_root" \
    "$hf_home" \
    "$input_manifest" \
    "$campaign_root" \
    "$archive_root" \
    "$private_key" \
    "$spec" \
    "$spec_sha256" \
    "$resume" \
    "$lane_memory" \
    "$lane_gres" \
    "$expected_gpu" \
    "$min_vram_gib"
