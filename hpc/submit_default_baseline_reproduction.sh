#!/usr/bin/env bash
set -euo pipefail
umask 077

test_only=0
if [[ "${1:-}" == "--test-only" ]]; then
    test_only=1
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--test-only]" >&2
    exit 2
fi

user_name="$(id -un)"
user_home="$(getent passwd "$user_name" | cut -d: -f6)"
scratch_root="${DPLORA_DEFAULT_SCRATCH_ROOT:-${SCRATCH:-/scratch/$user_name}}"
repository="${DPLORA_DEFAULT_REPO_DIR:-$user_home/src/DP-LoRA-paper-repro}"
expected_sha="${DPLORA_DEFAULT_EXPECTED_SHA:-$(git -C "$repository" rev-parse HEAD)}"
short_sha="${expected_sha:0:7}"
env_prefix="${DPLORA_DEFAULT_ENV_PREFIX:-$scratch_root/envs/dp-lora-$short_sha}"
hf_home="${DPLORA_DEFAULT_HF_HOME:-$scratch_root/cache/dp-lora-paper/huggingface}"
input_root="${DPLORA_DEFAULT_INPUT_ROOT:-$scratch_root/datasets/dp-lora-default-baselines}"
input_index="${DPLORA_DEFAULT_INPUT_INDEX:-$input_root/input-index.json}"
worker="$repository/hpc/default_baseline_reproduction.sbatch"
spec="$repository/hpc/default-baseline-reproduction-spec.json"
run_root="${DPLORA_DEFAULT_RUN_ROOT:-$scratch_root/runs/dp-lora-paper/default-baseline-campaigns}"
run_id="${DPLORA_DEFAULT_RUN_ID:-default-baselines-$short_sha-$(date -u +%Y%m%dT%H%M%SZ)}"
campaign_root="$run_root/$run_id"
private_key_root="${DPLORA_DEFAULT_PRIVATE_KEY_ROOT:-$user_home/hpc/projects/dp-lora-paper/private-rng/default-baselines}"
private_key="$private_key_root/$run_id.key"
slurm_root="$run_root/slurm"

account="${DPLORA_DEFAULT_ACCOUNT:-normal}"
partition="${DPLORA_DEFAULT_PARTITION:-a100}"
gres="${DPLORA_DEFAULT_GRES:-gpu:a100:1}"
cpus="${DPLORA_DEFAULT_CPUS:-8}"
host_memory="${DPLORA_DEFAULT_HOST_MEMORY:-200G}"
walltime="${DPLORA_DEFAULT_WALLTIME:-2-12:00:00}"
minimum_vram_gib="${DPLORA_DEFAULT_MINIMUM_VRAM_GIB:-75}"

for path in "$repository" "$env_prefix/bin/python" "$hf_home" "$input_index" "$worker" "$spec"; do
    [[ -e "$path" ]] || { echo "ERROR: missing submission input: $path" >&2; exit 2; }
done
if [[ "$(git -C "$repository" rev-parse HEAD)" != "$expected_sha" || -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: repository must be a clean immutable snapshot at $expected_sha" >&2
    exit 2
fi
"$env_prefix/bin/python" "$repository/paper_repro/default_baseline_campaign.py" \
    validate-spec --spec "$spec" >/dev/null

mkdir -p "$campaign_root" "$private_key_root" "$slurm_root"
chmod 700 "$campaign_root" "$private_key_root" "$slurm_root"
if [[ ! -f "$private_key" ]]; then
    "$env_prefix/bin/python" - "$private_key" <<'PY'
import os, secrets, sys
path = sys.argv[1]
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as handle:
    handle.write(secrets.token_bytes(32))
PY
fi
[[ "$(stat -c '%a' "$private_key")" == 600 ]] || { echo "ERROR: private key mode is not 600" >&2; exit 2; }

spec_file_sha="$(sha256sum "$spec" | awk '{print $1}')"
sbatch_args=(
    --account="$account" --partition="$partition" --nodes=1 --ntasks=1
    --cpus-per-task="$cpus" --gres="$gres" --mem="$host_memory"
    --time="$walltime" --signal=B:USR1@300
    --job-name=dp-lora-default-baselines
    --output="$slurm_root/%x-%j.out" --error="$slurm_root/%x-%j.err"
)
if [[ "$test_only" -eq 1 ]]; then sbatch_args+=(--test-only); fi

echo "repository=$repository"
echo "repository_sha=$expected_sha"
echo "environment=$env_prefix"
echo "input_index=$input_index"
echo "campaign_root=$campaign_root"
echo "resources=account:$account partition:$partition gres:$gres cpus:$cpus mem:$host_memory time:$walltime"

sbatch "${sbatch_args[@]}" "$worker" \
    "$repository" "$expected_sha" "$env_prefix" "$scratch_root" "$hf_home" \
    "$input_index" "$campaign_root" "$private_key" "$spec" \
    "$spec_file_sha" "$minimum_vram_gib"
