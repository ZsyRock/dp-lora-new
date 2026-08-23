#!/usr/bin/env bash
set -euo pipefail
umask 077

test_only=0
resume=0
for argument in "$@"; do
    case "$argument" in
        --test-only) test_only=1 ;;
        --resume) resume=1 ;;
        *) echo "usage: $0 [--test-only] [--resume]" >&2; exit 2 ;;
    esac
done
if [[ "$test_only" -eq 1 && "$resume" -eq 1 ]]; then
    echo "ERROR: --test-only and --resume cannot be combined" >&2
    exit 2
fi

user_name="$(id -un)"
user_home="$(getent passwd "$user_name" | cut -d: -f6)"
scratch_root="${DPLORA_FOLLOWUP_SCRATCH_ROOT:-${SCRATCH:-/scratch/$user_name}}"
repository="${DPLORA_FOLLOWUP_REPO_DIR:-$user_home/src/DP-LoRA-paper-repro}"
expected_sha="${DPLORA_FOLLOWUP_EXPECTED_SHA:-$(git -C "$repository" rev-parse HEAD)}"
short_sha="${expected_sha:0:7}"
env_prefix="${DPLORA_FOLLOWUP_ENV_PREFIX:-$scratch_root/envs/dp-lora-$short_sha}"
hf_home="${DPLORA_FOLLOWUP_HF_HOME:-$scratch_root/cache/dp-lora-paper/huggingface}"
input_root="${DPLORA_FOLLOWUP_INPUT_ROOT:-$scratch_root/datasets/dp-lora-default-baselines}"
input_index="${DPLORA_FOLLOWUP_INPUT_INDEX:-$input_root/input-index.json}"
upstream_root="${DPLORA_FOLLOWUP_UPSTREAM_ROOT:-$user_home/hpc/projects/dp-lora-paper/completed-runs/default-baselines/default-baselines-9a0e0f4-20260822}"
worker="$repository/hpc/baseline_followup_fixed.sbatch"
spec="$repository/hpc/baseline-followup-fixed-spec.json"
campaign="$repository/paper_repro/baseline_followup_campaign.py"
default_campaign="$repository/paper_repro/default_baseline_campaign.py"
default_spec="$repository/hpc/default-baseline-reproduction-spec.json"
runtime_lock="$repository/environment/paper-repro-runtime.lock"

run_root="${DPLORA_FOLLOWUP_RUN_ROOT:-$scratch_root/runs/dp-lora-paper/baseline-followup-fixed}"
run_id="${DPLORA_FOLLOWUP_RUN_ID:-baseline-followup-fixed-$short_sha-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: unsafe run ID: $run_id" >&2
    exit 2
fi
formal_campaign_root="$run_root/$run_id"
private_key_root="${DPLORA_FOLLOWUP_PRIVATE_KEY_ROOT:-$user_home/hpc/projects/dp-lora-paper/private-rng/baseline-followup-fixed}"
formal_private_key_root="$private_key_root"
formal_private_key="$private_key_root/$run_id.key"
slurm_root="$run_root/slurm"
formal_slurm_root="$slurm_root"
backup_base="${DPLORA_FOLLOWUP_BACKUP_ROOT:-$user_home/hpc/projects/dp-lora-paper/completed-runs/baseline-followup-fixed}"
formal_backup_root="$backup_base/$run_id"

account="${DPLORA_FOLLOWUP_ACCOUNT:-normal}"
partition="${DPLORA_FOLLOWUP_PARTITION:-a100}"
qos="${DPLORA_FOLLOWUP_QOS:-normal}"
gres="${DPLORA_FOLLOWUP_GRES:-gpu:a100:1}"
cpus="${DPLORA_FOLLOWUP_CPUS:-8}"
host_memory="${DPLORA_FOLLOWUP_HOST_MEMORY:-80G}"
walltime="${DPLORA_FOLLOWUP_WALLTIME:-24:00:00}"
minimum_vram_gib="${DPLORA_FOLLOWUP_MINIMUM_VRAM_GIB:-75}"

for path in \
    "$repository" "$env_prefix/bin/python" "$hf_home" "$input_index" \
    "$upstream_root" "$worker" "$spec" "$campaign" "$default_campaign" \
    "$default_spec" "$runtime_lock"; do
    [[ -e "$path" ]] || { echo "ERROR: missing submission input: $path" >&2; exit 2; }
done
if [[ ! "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || \
   [[ "$(git -C "$repository" rev-parse HEAD)" != "$expected_sha" ]] || \
   [[ -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: repository must be a clean immutable snapshot at $expected_sha" >&2
    exit 2
fi
if [[ "$account" != "normal" || "$partition" != "a100" || "$qos" != "normal" || \
      "$gres" != "gpu:a100:1" || "$host_memory" != "80G" || "$walltime" != "24:00:00" ]]; then
    echo "ERROR: this campaign is fixed to normal/a100, one A100, 80G host RAM, and 24 hours" >&2
    exit 2
fi
if [[ ! "$cpus" =~ ^[1-9][0-9]*$ || "$cpus" -gt 12 ]]; then
    echo "ERROR: CPUs must be between 1 and 12" >&2
    exit 2
fi

python_bin="$env_prefix/bin/python"
"$python_bin" "$campaign" validate-spec --spec "$spec"
"$python_bin" -m pip check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$python_bin" "$default_campaign" validate-runtime \
    --spec "$default_spec" --runtime-lock "$runtime_lock" --hf-home "$hf_home" \
    >/dev/null

test_sandbox=""
cleanup_test_sandbox() {
    if [[ -n "$test_sandbox" && "$test_sandbox" == "$scratch_root"/tmp/dp-lora-followup-test.* ]]; then
        rm -rf -- "$test_sandbox"
    fi
}
if [[ "$test_only" -eq 1 ]]; then
    mkdir -p "$scratch_root/tmp"
    chmod 700 "$scratch_root/tmp"
    test_sandbox="$(mktemp -d "$scratch_root/tmp/dp-lora-followup-test.XXXXXXXX")"
    chmod 700 "$test_sandbox"
    trap cleanup_test_sandbox EXIT
    run_root="$test_sandbox/runs"
    campaign_root="$run_root/$run_id"
    private_key_root="$test_sandbox/private-rng"
    private_key="$private_key_root/$run_id.key"
    slurm_root="$test_sandbox/slurm"
    backup_root="$test_sandbox/critical-results"
else
    campaign_root="$formal_campaign_root"
    private_key_root="$formal_private_key_root"
    private_key="$formal_private_key"
    slurm_root="$formal_slurm_root"
    backup_root="$formal_backup_root"
fi

mkdir -p "$run_root" "$slurm_root" "$private_key_root"
chmod 700 "$run_root" "$slurm_root" "$private_key_root"
if [[ "$resume" -eq 0 ]]; then
    [[ ! -e "$campaign_root" ]] || {
        echo "ERROR: refusing to overwrite campaign root: $campaign_root" >&2
        exit 2
    }
    [[ ! -e "$private_key" ]] || {
        echo "ERROR: refusing to reuse private RNG key: $private_key" >&2
        exit 2
    }
    "$python_bin" - "$private_key" <<'PY'
import os, secrets, sys
path = sys.argv[1]
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(secrets.token_bytes(32))
PY
else
    [[ -f "$campaign_root/runtime-manifest.json" ]] || {
        echo "ERROR: resume candidate plan is missing" >&2
        exit 2
    }
    [[ -f "$private_key" ]] || {
        echo "ERROR: resume private RNG key is missing" >&2
        exit 2
    }
fi
[[ "$(stat -c '%a' "$private_key")" == 600 ]] || {
    echo "ERROR: private RNG key mode must be 600" >&2
    exit 2
}

prepare_args=(
    prepare --spec "$spec" --repository "$repository"
    --expected-code-sha "$expected_sha" --upstream-root "$upstream_root"
    --input-index "$input_index" --campaign-root "$campaign_root"
    --private-key "$private_key"
)
if [[ "$resume" -eq 1 ]]; then prepare_args+=(--resume); fi
"$python_bin" "$campaign" "${prepare_args[@]}"

plan="$campaign_root/runtime-manifest.json"
plan_sha="$($python_bin - "$plan" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["manifest_sha256"])
PY
)"
spec_sha="$(sha256sum "$spec" | awk '{print $1}')"

sbatch_args=(
    --parsable --account="$account" --partition="$partition" --qos="$qos"
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus"
    --gres="$gres" --mem="$host_memory" --time="$walltime"
    --signal=B:USR1@300 --job-name=dp-lora-fixed-followup
    --output="$slurm_root/%x-%j.out" --error="$slurm_root/%x-%j.err"
    --export=NONE
)
if [[ "$test_only" -eq 1 ]]; then sbatch_args+=(--test-only); fi

echo "repository=$repository"
echo "repository_sha=$expected_sha"
echo "environment=$env_prefix"
echo "upstream_baseline=$upstream_root"
echo "input_index=$input_index"
echo "campaign_root=$campaign_root"
echo "critical_results_backup=$backup_root"
echo "candidate_plan_sha256=$plan_sha"
echo "resources=account:$account qos:$qos partition:$partition gres:$gres cpus:$cpus mem:$host_memory time:$walltime"

sbatch "${sbatch_args[@]}" "$worker" \
    "$repository" "$expected_sha" "$env_prefix" "$scratch_root" "$hf_home" \
    "$upstream_root" "$input_index" "$campaign_root" "$private_key" "$spec" \
    "$spec_sha" "$plan_sha" "$minimum_vram_gib" "$host_memory" "$gres" \
    "$backup_root"
