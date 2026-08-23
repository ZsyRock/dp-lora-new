#!/usr/bin/env bash

#SBATCH --job-name=dp-lora-fixed-followup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Account, typed A100 GRES, 80 GiB host memory, 24-hour walltime, signal and
# log paths are supplied by the submission wrapper after live validation.
set -euo pipefail
umask 077

repository="${1:?missing immutable repository snapshot}"
expected_sha="${2:?missing expected repository SHA}"
env_prefix="${3:?missing environment prefix}"
scratch_root="${4:?missing scratch root}"
hf_home="${5:?missing Hugging Face cache root}"
upstream_root="${6:?missing completed default-baseline root}"
input_index="${7:?missing baseline input index}"
campaign_root="${8:?missing follow-up campaign root}"
private_key="${9:?missing private RNG key}"
spec="${10:?missing follow-up specification}"
expected_spec_sha="${11:?missing specification SHA-256}"
expected_plan_sha="${12:?missing candidate-plan SHA-256}"
minimum_vram_gib="${13:?missing minimum GPU VRAM GiB}"
step_memory="${14:?missing step host memory}"
step_gres="${15:?missing typed step GPU GRES}"
backup_root="${16:?missing persistent critical-results backup root}"

python_bin="$env_prefix/bin/python"
campaign="$repository/paper_repro/baseline_followup_campaign.py"
trainer="$repository/paper_repro/train_federated.py"
default_campaign="$repository/paper_repro/default_baseline_campaign.py"
default_spec="$repository/hpc/default-baseline-reproduction-spec.json"
runtime_lock="$repository/environment/paper-repro-runtime.lock"
exit_policy="$repository/hpc/full_slaclip_exit_policy.sh"
plan="$campaign_root/runtime-manifest.json"
global_stop="$campaign_root/control/stop.request"
step_cpus="${SLURM_CPUS_PER_TASK:-0}"

if [[ "${SLURM_JOB_NUM_NODES:-0}" != 1 || "${SLURM_NTASKS:-0}" != 1 ]]; then
    echo "ERROR: fixed follow-up requires one node and one sequential task" >&2
    exit 2
fi
if [[ ! "$step_cpus" =~ ^[1-9][0-9]*$ || "$step_cpus" -gt 12 ]]; then
    echo "ERROR: invalid CPU allocation: $step_cpus" >&2
    exit 2
fi
if [[ ! "$step_gres" =~ ^gpu:[A-Za-z0-9_-]+:1$ ]]; then
    echo "ERROR: fixed follow-up requires exactly one typed GPU" >&2
    exit 2
fi
if [[ "$backup_root" != /* || "$backup_root" == / ]]; then
    echo "ERROR: unsafe persistent backup root: $backup_root" >&2
    exit 2
fi
for required in \
    "$python_bin" "$campaign" "$trainer" "$default_campaign" \
    "$default_spec" "$runtime_lock" "$exit_policy" "$upstream_root" \
    "$input_index" "$private_key" "$spec" "$plan"; do
    [[ -e "$required" ]] || {
        echo "ERROR: missing runtime input: $required" >&2
        exit 2
    }
done
if [[ "$(git -C "$repository" rev-parse HEAD)" != "$expected_sha" || -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: immutable repository snapshot identity failed" >&2
    exit 2
fi
if [[ "$(sha256sum "$spec" | awk '{print $1}')" != "$expected_spec_sha" ]]; then
    echo "ERROR: follow-up specification changed after submission" >&2
    exit 2
fi
if [[ "$($python_bin - "$plan" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["manifest_sha256"])
PY
)" != "$expected_plan_sha" ]]; then
    echo "ERROR: candidate plan identity changed after submission" >&2
    exit 2
fi

export HF_HOME="$hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS="$step_cpus"
export TMPDIR="${SLURM_TMPDIR:-$scratch_root/tmp/dp-lora-fixed-followup-$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
source "$exit_policy"
full_slaclip_stop_generation=0

campaign_stage="allocation_started"
campaign_completed=0
backup_critical_results() {
    mkdir -p "$backup_root/arm-status"
    chmod 700 "$backup_root" "$backup_root/arm-status"
    local candidate
    for candidate in \
        "$plan" "$spec" "$campaign_root/campaign_summary.json" \
        "$campaign_root/campaign_metrics.csv" \
        "$campaign_root/strong-groupwise-fixed-selection.lock.json"; do
        if [[ -f "$candidate" ]]; then
            install -m 600 "$candidate" "$backup_root/$(basename "$candidate")"
        fi
    done
    if [[ -d "$campaign_root/arm-status" ]]; then
        cp -a "$campaign_root/arm-status/." "$backup_root/arm-status/"
    fi
    echo "critical_results_backup=$backup_root"
}
finalize_campaign() {
    local original_rc=$?
    local final_rc="$original_rc"
    trap - EXIT
    set +e
    "$python_bin" "$campaign" aggregate \
        --plan "$plan" --spec "$spec" --campaign-root "$campaign_root" || true
    backup_critical_results || true
    if [[ "$original_rc" -eq 0 && "$campaign_completed" -ne 1 ]]; then
        final_rc=1
    fi
    echo "campaign_exit_stage=$campaign_stage"
    exit "$final_rc"
}
trap finalize_campaign EXIT

request_stop() {
    full_slaclip_stop_generation=$((full_slaclip_stop_generation + 1))
    mkdir -p "$(dirname "$global_stop")"
    touch "$global_stop"
    echo "checkpoint_stop_requested_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap request_stop USR1 TERM INT

run_step() {
    srun --export=ALL --exact --exclusive --nodes=1 --ntasks=1 \
        "--cpus-per-task=$step_cpus" "--mem=$step_memory" \
        "--gres=$step_gres" "$@" < /dev/null
}

run_step_signal_safe() {
    local return_code=0
    local classification
    local restore_errexit=0
    [[ "$-" == *e* ]] && restore_errexit=1
    run_step "$@" &
    local child_pid=$!
    set +e
    wait_for_full_slaclip_child "$child_pid"
    return_code=$?
    if [[ "$restore_errexit" -eq 1 ]]; then set -e; fi
    classification="$(classify_full_slaclip_exit_codes "$return_code")"
    if [[ "$classification" == "HARD_FAILURE" ]]; then return "$return_code"; fi
    if [[ "$classification" == "CHECKPOINTED_STOP" || -e "$global_stop" ]]; then
        return 75
    fi
    return 0
}

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "repository_sha=$expected_sha"
echo "campaign_root=$campaign_root"
echo "critical_results_backup=$backup_root"
nvidia-smi

campaign_stage="runtime_gpu_and_input_preflight"
run_step_signal_safe "$python_bin" -c \
    'import sys, torch; available=torch.cuda.is_available(); print({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "available": available}); available or sys.exit("CUDA is unavailable"); p=torch.cuda.get_device_properties(0); actual=p.total_memory/1024**3; print({"gpu": p.name, "vram_gib": actual}); actual >= float(sys.argv[1]) or sys.exit(f"GPU VRAM {actual:.1f} GiB is below the requirement")' \
    "$minimum_vram_gib"
run_step_signal_safe "$python_bin" -m pip check
run_step_signal_safe "$python_bin" "$default_campaign" validate-runtime \
    --spec "$default_spec" --runtime-lock "$runtime_lock" --hf-home "$hf_home"
run_step_signal_safe "$python_bin" "$campaign" validate-locked-plan \
    --plan "$plan" --spec "$spec" --repository "$repository" \
    --expected-code-sha "$expected_sha" --upstream-root "$upstream_root" \
    --input-index "$input_index" --private-key "$private_key"
while IFS= read -r input_manifest; do
    run_step_signal_safe "$python_bin" "$trainer" \
        --check-inputs --input-manifest "$input_manifest"
done < <("$python_bin" "$campaign" print-input-manifests --plan "$plan")

campaign_stage="real_model_smokes"
run_step_signal_safe "$python_bin" "$campaign" run-smokes \
    --plan "$plan" --spec "$spec" --campaign-root "$campaign_root" \
    --repository "$repository" --python-bin "$python_bin" \
    --private-key "$private_key"

campaign_stage="groupwise_fixed_development"
while IFS= read -r arm_index; do
    echo "fixed_development_arm_index=$arm_index"
    run_step_signal_safe "$python_bin" "$campaign" run-arm \
        --plan "$plan" --spec "$spec" --campaign-root "$campaign_root" \
        --repository "$repository" --python-bin "$python_bin" \
        --private-key "$private_key" --arm-index "$arm_index"
done < <("$python_bin" "$campaign" print-arm-indices --plan "$plan")

campaign_stage="fixed_selection_lock"
"$python_bin" "$campaign" lock-selection \
    --plan "$plan" --spec "$spec" --campaign-root "$campaign_root"

campaign_stage="final_revalidation"
"$python_bin" "$campaign" aggregate \
    --plan "$plan" --spec "$spec" --campaign-root "$campaign_root" \
    --require-complete

campaign_completed=1
campaign_stage="completed"
echo "campaign_status=COMPLETED"
echo "candidate_plan=$plan"
echo "selection_lock=$campaign_root/strong-groupwise-fixed-selection.lock.json"
echo "metrics=$campaign_root/campaign_metrics.csv"
