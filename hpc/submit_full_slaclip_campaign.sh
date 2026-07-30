#!/usr/bin/env bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

usage() {
    cat >&2 <<'USAGE'
Usage: hpc/submit_full_slaclip_campaign.sh [--resume] [--test-only]

Submit the 108-arm, 54-wave full-SlaClip campaign as exactly one Slurm job.
The allocation performs its two H200 CUDA smokes before starting the matrix.

--resume requires DPLORA_FULL_RUN_ID to name an existing partial campaign.
--test-only performs all login-node gates and Slurm scheduler validation but
does not create a queued job.

Important overrides:
  DPLORA_FULL_RUN_ID          stable campaign identifier (required to resume)
  DPLORA_FULL_REPO_DIR        development Git worktree
  DPLORA_FULL_EXPECTED_SHA    full committed SHA (default: worktree HEAD)
  DPLORA_FULL_ENV_PREFIX      tested Python environment for this code SHA
  DPLORA_FULL_SCRATCH_ROOT    scratch root
  DPLORA_FULL_ARCHIVE_ROOT    persistent incremental-small-artifact archive
  DPLORA_FULL_ACCOUNT         Slurm account (default: normal)
  DPLORA_FULL_PARTITION       partitions (default: quad_h200,dual_h200)
  DPLORA_FULL_GPU_GRES        fixed typed allocation (default: gpu:h200:2)
  DPLORA_FULL_CPUS_PER_TASK   CPU cores per GPU lane (default: 12; max: 12)
  DPLORA_FULL_HOST_MEMORY     parent host RAM (default: 384G)
  DPLORA_FULL_LANE_MEMORY     host RAM per concurrent lane (default: 192G)
  DPLORA_FULL_WALLTIME        walltime (default/max here: 2-12:00:00)

The confirmatory paper setting fixes K_clients=5, T=50, B=8, sigma=2,
lr=5e-4, rank=512, K_slots=15, and full SlaClip C bounds [0.1, 50].
Pre-registered sensitivity arms vary sigma and client count while retaining
K_slots=15. Exact gradient/CDF diagnostics are non-DP private artifacts.
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
    echo "ERROR: could not resolve the persistent home directory" >&2
    exit 1
fi
scratch_root="${DPLORA_FULL_SCRATCH_ROOT:-${SCRATCH:-/scratch/$user_name}}"
repo_dir="${DPLORA_FULL_REPO_DIR:-$user_home/src/DP-LoRA-paper-repro}"
repo_dir="$(readlink -f "$repo_dir")"
git_bin="$(command -v git || true)"
if [[ -z "$git_bin" || "$($git_bin -C "$repo_dir" rev-parse --is-inside-work-tree 2>/dev/null || true)" != true ]]; then
    echo "ERROR: DPLORA_FULL_REPO_DIR is not a Git worktree: $repo_dir" >&2
    exit 1
fi
if [[ -n "$($git_bin -C "$repo_dir" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: commit or remove every worktree change before submission" >&2
    exit 1
fi
actual_sha="$($git_bin -C "$repo_dir" rev-parse HEAD)"
expected_sha="${DPLORA_FULL_EXPECTED_SHA:-$actual_sha}"
if [[ ! "$expected_sha" =~ ^[0-9a-f]{40}$ || "$actual_sha" != "$expected_sha" ]]; then
    echo "ERROR: the requested full SHA does not match the clean worktree HEAD" >&2
    exit 1
fi
short_sha="${expected_sha:0:7}"

snapshot_root="${DPLORA_FULL_SNAPSHOT_ROOT:-$user_home/src/DP-LoRA-paper-snapshots}"
snapshot_repo="$snapshot_root/$expected_sha"
mkdir -p "$snapshot_root"
chmod 700 "$snapshot_root"
if [[ ! -e "$snapshot_repo" ]]; then
    "$git_bin" -C "$repo_dir" worktree add --detach "$snapshot_repo" "$expected_sha"
fi
if [[ "$($git_bin -C "$snapshot_repo" rev-parse HEAD 2>/dev/null || true)" != "$expected_sha" ]]; then
    echo "ERROR: immutable source snapshot has the wrong SHA: $snapshot_repo" >&2
    exit 1
fi
if [[ -n "$($git_bin -C "$snapshot_repo" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: immutable source snapshot is dirty: $snapshot_repo" >&2
    exit 1
fi

env_prefix="${DPLORA_FULL_ENV_PREFIX:-$scratch_root/envs/dp-lora-paper-$short_sha}"
env_prefix="$(readlink -f "$env_prefix")"
python_bin="$env_prefix/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: versioned environment interpreter is missing: $python_bin" >&2
    echo "Set DPLORA_FULL_ENV_PREFIX to the tested dependency-identical environment." >&2
    exit 1
fi

worker="$snapshot_repo/hpc/full_slaclip_campaign.sbatch"
submit_snapshot="$snapshot_repo/hpc/submit_full_slaclip_campaign.sh"
spec="$snapshot_repo/hpc/full-slaclip-campaign-spec.json"
coordinator="$snapshot_repo/paper_repro/full_slaclip_campaign.py"
trainer="$snapshot_repo/paper_repro/train_federated.py"
stage_script="$snapshot_repo/scripts/stage_paper_inputs.sh"
runtime_lock="$snapshot_repo/environment/paper-repro-runtime.lock"
for required in "$worker" "$submit_snapshot" "$spec" "$coordinator" "$trainer" "$stage_script" "$runtime_lock"; do
    [[ -f "$required" ]] || { echo "ERROR: pinned snapshot is missing $required" >&2; exit 1; }
done
bash -n "$submit_snapshot" "$worker"
"$python_bin" - "$coordinator" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_bytes()
compile(source, sys.argv[1], "exec")
print(f"python_source_compiled={sys.argv[1]}")
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
    raise SystemExit("Python version does not match the runtime lock")
for name, version in expected.items():
    if importlib.metadata.version(name) != version:
        raise SystemExit(f"runtime version mismatch for {name}")
print(f"runtime_lock_verified={sys.argv[1]}")
PY
"$python_bin" "$coordinator" validate-spec --spec "$spec"
PYTHONPATH="$snapshot_repo" "$python_bin" -c \
    'from paper_repro.reproducibility import METHOD_SPECS; assert "slaclip_dp_lora" in METHOD_SPECS; assert "slaclip_q_dp_lora" not in METHOD_SPECS'

hf_home="${DPLORA_FULL_HF_HOME:-$scratch_root/cache/dp-lora-paper/huggingface}"
data_root="${DPLORA_FULL_DATA_ROOT:-$scratch_root/datasets/dp-lora-paper}"
input_manifest="${DPLORA_FULL_INPUT_MANIFEST:-$data_root/input-manifest.json}"
if [[ ! -f "$input_manifest" ]]; then
    echo "ERROR: staged immutable input manifest is missing: $input_manifest" >&2
    exit 1
fi
DPLORA_PAPER_PYTHON="$python_bin" \
DPLORA_PAPER_SCRATCH_ROOT="$scratch_root" \
DPLORA_PAPER_HF_HOME="$hf_home" \
DPLORA_PAPER_DATA_ROOT="$data_root" \
DPLORA_PAPER_INPUT_MANIFEST="$input_manifest" \
"$stage_script" --check-only
"$python_bin" "$trainer" --check-inputs --input-manifest "$input_manifest"

if [[ "$resume" -eq 1 && -z "${DPLORA_FULL_RUN_ID:-}" ]]; then
    echo "ERROR: --resume requires DPLORA_FULL_RUN_ID" >&2
    exit 2
fi
if [[ -n "${DPLORA_FULL_RUN_ID:-}" ]]; then
    run_id="$DPLORA_FULL_RUN_ID"
else
    run_id="full-slaclip-${short_sha}-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: unsafe DPLORA_FULL_RUN_ID: $run_id" >&2
    exit 1
fi

run_root="${DPLORA_FULL_RUN_ROOT:-$scratch_root/runs/dp-lora-paper/full-slaclip-campaigns}"
campaign_root="$run_root/$run_id"
archive_base="${DPLORA_FULL_ARCHIVE_ROOT:-$user_home/hpc/projects/dp-lora-paper/completed-runs/full-slaclip}"
archive_root="$archive_base/$run_id"
private_key_root="${DPLORA_FULL_PRIVATE_KEY_ROOT:-$user_home/hpc/projects/dp-lora-paper/private-rng/full-slaclip}"
private_key="$private_key_root/$run_id.key"
if [[ "$resume" -eq 1 ]]; then
    [[ -d "$campaign_root" ]] || { echo "ERROR: resume campaign is missing: $campaign_root" >&2; exit 1; }
    [[ -f "$private_key" ]] || { echo "ERROR: persistent resume RNG key is missing: $private_key" >&2; exit 1; }
else
    [[ ! -e "$campaign_root" ]] || { echo "ERROR: refusing to overwrite campaign: $campaign_root" >&2; exit 1; }
    [[ ! -e "$archive_root" ]] || { echo "ERROR: refusing to overwrite archive: $archive_root" >&2; exit 1; }
    [[ ! -e "$private_key" ]] || { echo "ERROR: refusing to reuse a campaign RNG key: $private_key" >&2; exit 1; }
fi
mkdir -p "$run_root/slurm" "$archive_base" "$private_key_root"
chmod 700 "$run_root" "$run_root/slurm" "$archive_base" "$private_key_root"

account="${DPLORA_FULL_ACCOUNT:-normal}"
partition="${DPLORA_FULL_PARTITION:-quad_h200,dual_h200}"
gpu_gres="${DPLORA_FULL_GPU_GRES:-gpu:h200:2}"
cpus_per_task="${DPLORA_FULL_CPUS_PER_TASK:-12}"
host_memory="${DPLORA_FULL_HOST_MEMORY:-384G}"
lane_memory="${DPLORA_FULL_LANE_MEMORY:-192G}"
walltime="${DPLORA_FULL_WALLTIME:-2-12:00:00}"
if [[ "$gpu_gres" != "gpu:h200:2" ]]; then
    echo "ERROR: this two-lane formal campaign requires exactly gpu:h200:2" >&2
    exit 1
fi
if [[ ! "$cpus_per_task" =~ ^[1-9][0-9]*$ || "$cpus_per_task" -gt 12 ]]; then
    echo "ERROR: CPU cores per lane must be 1..12 (24 total QoS maximum)" >&2
    exit 1
fi
if [[ "$walltime" != "2-12:00:00" ]]; then
    echo "ERROR: this formal contract fixes the discovered 60-hour maximum walltime" >&2
    exit 1
fi

sbatch_bin="$(command -v sbatch || true)"
if [[ -z "$sbatch_bin" ]]; then
    echo "ERROR: sbatch is unavailable" >&2
    exit 1
fi
spec_sha256="$(sha256sum "$spec" | awk '{print $1}')"
sbatch_args=(
    --parsable
    "--account=$account"
    "--partition=$partition"
    "--gres=$gpu_gres"
    --nodes=1
    --ntasks=2
    "--cpus-per-task=$cpus_per_task"
    "--mem=$host_memory"
    "--time=$walltime"
    --signal=B:USR1@600
    --job-name=dp-lora-full-slaclip
    "--output=$run_root/slurm/%x-%j.out"
    "--error=$run_root/slurm/%x-%j.err"
    --export=NONE
)
if [[ "$test_only" -eq 1 ]]; then
    sbatch_args+=(--test-only)
fi

echo "Campaign:          $run_id"
echo "Resume:            $resume"
echo "Source snapshot:   $snapshot_repo"
echo "Pinned SHA:        $expected_sha"
echo "Environment:       $env_prefix"
echo "Input manifest:    $input_manifest"
echo "Scratch output:    $campaign_root"
echo "Persistent archive:$archive_root"
echo "Resources:         account=$account partition=$partition gres=$gpu_gres nodes=1 lanes=2 cpus/lane=$cpus_per_task mem=$host_memory lane_mem=$lane_memory time=$walltime"

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
    "$lane_memory"
