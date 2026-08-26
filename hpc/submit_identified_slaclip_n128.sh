#!/usr/bin/env bash
set -euo pipefail
umask 077

test_only=0
resume=0
dependency_job_id="1425084"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-only) test_only=1; shift ;;
        --resume) resume=1; shift ;;
        --dependency-job-id)
            [[ $# -ge 2 ]] || { echo "ERROR: --dependency-job-id needs a value" >&2; exit 2; }
            dependency_job_id="$2"; shift 2 ;;
        *) echo "usage: $0 [--test-only] [--resume (requires DPLORA_IDENTIFIED_RUN_ID)] [--dependency-job-id 1425084 (verified complete, not reattached)]" >&2; exit 2 ;;
    esac
done
if [[ "$test_only" -eq 1 && "$resume" -eq 1 ]]; then
    echo "ERROR: --test-only and --resume cannot be combined" >&2
    exit 2
fi
if [[ ! "$dependency_job_id" =~ ^[1-9][0-9]*$ || "$dependency_job_id" != "1425084" ]]; then
    echo "ERROR: this preregistration requires completed upstream job 1425084" >&2
    exit 2
fi
upstream_job_record="$(
    sacct -X -n -P -j "$dependency_job_id" --format=JobIDRaw,State,ExitCode |
        awk -F'|' -v expected="$dependency_job_id" '$1 == expected {print $2 "|" $3; exit}'
)"
if [[ "$upstream_job_record" != "COMPLETED|0:0" ]]; then
    echo "ERROR: upstream job $dependency_job_id is not COMPLETED with exit 0:0: ${upstream_job_record:-missing}" >&2
    exit 2
fi

user_name="$(id -un)"
user_home="$(getent passwd "$user_name" | cut -d: -f6)"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
script_repository="$(cd -- "$script_dir/.." && pwd -P)"
scratch_root="${DPLORA_IDENTIFIED_SCRATCH_ROOT:-${SCRATCH:-/scratch/$user_name}}"
repository="${DPLORA_IDENTIFIED_REPO_DIR:-$script_repository}"
expected_sha="${DPLORA_IDENTIFIED_EXPECTED_SHA:-$(git -C "$repository" rev-parse HEAD)}"
short_sha="${expected_sha:0:7}"
env_prefix="${DPLORA_IDENTIFIED_ENV_PREFIX:-$scratch_root/envs/dp-lora-fcfbc49}"
hf_home="${DPLORA_IDENTIFIED_HF_HOME:-$scratch_root/cache/dp-lora-paper/huggingface}"
upstream_root="${DPLORA_IDENTIFIED_UPSTREAM_ROOT:-$scratch_root/runs/dp-lora-paper/baseline-followup-fixed/baseline-followup-fixed-fcfbc49-20260823}"
upstream_repository="${DPLORA_IDENTIFIED_UPSTREAM_REPO:-$user_home/hpc/projects/dp-lora-paper/worktrees/fcfbc49}"
upstream_spec="${DPLORA_IDENTIFIED_UPSTREAM_SPEC:-$upstream_repository/hpc/baseline-followup-fixed-spec.json}"
worker="$repository/hpc/identified_slaclip_n128.sbatch"
spec="$repository/hpc/identified-slaclip-n128-spec.json"
campaign="$repository/paper_repro/identified_slaclip_n128_campaign.py"
default_campaign="$repository/paper_repro/default_baseline_campaign.py"
default_spec="$repository/hpc/default-baseline-reproduction-spec.json"
runtime_lock="$repository/environment/paper-repro-runtime.lock"

run_root="${DPLORA_IDENTIFIED_RUN_ROOT:-$scratch_root/runs/dp-lora-paper/identified-slaclip-n128}"
if [[ "$resume" -eq 1 && -z "${DPLORA_IDENTIFIED_RUN_ID:-}" ]]; then
    echo "ERROR: --resume requires explicit DPLORA_IDENTIFIED_RUN_ID for an existing campaign" >&2
    exit 2
fi
run_id="${DPLORA_IDENTIFIED_RUN_ID:-identified-slaclip-n128-$short_sha-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: unsafe run ID: $run_id" >&2
    exit 2
fi
campaign_root="$run_root/$run_id"
private_key_root="${DPLORA_IDENTIFIED_PRIVATE_KEY_ROOT:-$user_home/hpc/projects/dp-lora-paper/private-rng/identified-slaclip-n128}"
private_key="$private_key_root/$run_id.key"
slurm_root="$run_root/slurm"
backup_base="${DPLORA_IDENTIFIED_BACKUP_ROOT:-$user_home/hpc/projects/dp-lora-paper/completed-runs/identified-slaclip-n128}"
backup_root="$backup_base/$run_id"

account="${DPLORA_IDENTIFIED_ACCOUNT:-normal}"
partition="${DPLORA_IDENTIFIED_PARTITION:-a100}"
qos="${DPLORA_IDENTIFIED_QOS:-normal}"
gres="${DPLORA_IDENTIFIED_GRES:-gpu:a100:1}"
cpus="${DPLORA_IDENTIFIED_CPUS:-8}"
host_memory="${DPLORA_IDENTIFIED_HOST_MEMORY:-80G}"
walltime="${DPLORA_IDENTIFIED_WALLTIME:-24:00:00}"
minimum_vram_gib="${DPLORA_IDENTIFIED_MINIMUM_VRAM_GIB:-75}"
exclude_nodes="${DPLORA_IDENTIFIED_EXCLUDE_NODES:-}"

for path in \
    "$repository" "$env_prefix/bin/python" "$hf_home" "$upstream_root" \
    "$upstream_repository" "$upstream_spec" "$worker" "$spec" "$campaign" \
    "$default_campaign" "$default_spec" "$runtime_lock"; do
    [[ -e "$path" ]] || { echo "ERROR: missing submission input: $path" >&2; exit 2; }
done
if [[ ! "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || \
   [[ "$(git -C "$repository" rev-parse HEAD)" != "$expected_sha" ]] || \
   [[ -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: repository must be a clean immutable snapshot at $expected_sha" >&2
    exit 2
fi
if [[ "$(git -C "$upstream_repository" rev-parse HEAD)" != "fcfbc490f664c64e4463c501cc0631a599f2cb25" ]] || \
   [[ -n "$(git -C "$upstream_repository" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: upstream repository snapshot differs from the preregistration" >&2
    exit 2
fi
if [[ "$account" != "normal" || "$partition" != "a100" || "$qos" != "normal" || \
      "$gres" != "gpu:a100:1" || "$host_memory" != "80G" || "$walltime" != "24:00:00" ]]; then
    echo "ERROR: campaign is fixed to normal/a100, one A100, 80G RAM, 24 hours" >&2
    exit 2
fi
if [[ ! "$cpus" =~ ^[1-9][0-9]*$ || "$cpus" -gt 12 ]]; then
    echo "ERROR: CPUs must be between 1 and 12" >&2
    exit 2
fi
if [[ -n "$exclude_nodes" && ! "$exclude_nodes" =~ ^[A-Za-z0-9][A-Za-z0-9,._-]*$ ]]; then
    echo "ERROR: unsafe Slurm node exclusion list: $exclude_nodes" >&2
    exit 2
fi

python_bin="$env_prefix/bin/python"
"$python_bin" "$campaign" validate-spec --spec "$spec" >/dev/null
"$python_bin" -m pip check >/dev/null
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$python_bin" "$default_campaign" validate-runtime \
    --spec "$default_spec" --runtime-lock "$runtime_lock" --hf-home "$hf_home" \
    >/dev/null

spec_sha="$(sha256sum "$spec" | awk '{print $1}')"
resume_flag="$resume"
if [[ "$test_only" -eq 1 ]]; then
    # Prospective paths only: do not create a campaign directory, RNG key,
    # backup directory, or Slurm-log directory during scheduler validation.
    prospective="$scratch_root/tmp/dp-lora-id-n128-test-only-$run_id"
    campaign_root="$prospective/campaign"
    private_key="$prospective/private.key"
    backup_root="$prospective/backup"
    output_path="/dev/null"
    error_path="/dev/null"
else
    mkdir -p "$run_root" "$slurm_root" "$private_key_root" "$backup_base"
    chmod 700 "$run_root" "$slurm_root" "$private_key_root" "$backup_base"
    if [[ "$resume" -eq 0 ]]; then
        [[ ! -e "$campaign_root" ]] || { echo "ERROR: refusing to overwrite $campaign_root" >&2; exit 2; }
        [[ ! -e "$private_key" ]] || { echo "ERROR: refusing to reuse $private_key" >&2; exit 2; }
        "$python_bin" - "$private_key" <<'PY'
import os, secrets, sys
path=sys.argv[1]
fd=os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as handle:
    handle.write(secrets.token_bytes(32))
PY
    else
        [[ -f "$campaign_root/identified-campaign-plan.lock.json" ]] || { echo "ERROR: resume plan missing" >&2; exit 2; }
        [[ -f "$private_key" ]] || { echo "ERROR: resume RNG key missing" >&2; exit 2; }
    fi
    [[ "$(stat -c '%a' "$private_key")" == 600 ]] || { echo "ERROR: RNG key mode differs" >&2; exit 2; }
    output_path="$slurm_root/%x-%j.out"
    error_path="$slurm_root/%x-%j.err"
fi

sbatch_args=(
    --parsable --account="$account" --partition="$partition" --qos="$qos"
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus"
    --gres="$gres" --mem="$host_memory" --time="$walltime"
    --signal=B:USR1@300
    --job-name=dp-lora-id-n128 --output="$output_path" --error="$error_path"
    --export=NONE
)
if [[ "$test_only" -eq 1 ]]; then sbatch_args+=(--test-only); fi
if [[ -n "$exclude_nodes" ]]; then sbatch_args+=(--exclude="$exclude_nodes"); fi

echo "repository=$repository"
echo "repository_sha=$expected_sha"
echo "upstream_campaign=$upstream_root"
echo "upstream_job=verified_completed:$dependency_job_id:0:0"
echo "campaign_root=$campaign_root"
echo "critical_results_backup=$backup_root"
echo "resources=account:$account qos:$qos partition:$partition gres:$gres cpus:$cpus mem:$host_memory time:$walltime signal:USR1@300 exclude:${exclude_nodes:-none}"

sbatch "${sbatch_args[@]}" "$worker" \
    "$repository" "$expected_sha" "$env_prefix" "$scratch_root" "$hf_home" \
    "$upstream_root" "$upstream_repository" "$upstream_spec" \
    "$campaign_root" "$private_key" "$spec" "$spec_sha" \
    "$minimum_vram_gib" "$host_memory" "$gres" "$backup_root" "$resume_flag"
