#!/usr/bin/env bash

#SBATCH --job-name=dp-lora-id-n128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Account, A100 GRES, 80 GiB RAM, 24h walltime, USR1 signal, and log paths are
# supplied by the submission wrapper after validation.
set -euo pipefail
umask 077

repository="${1:?missing immutable repository snapshot}"
expected_sha="${2:?missing expected repository SHA}"
env_prefix="${3:?missing environment prefix}"
scratch_root="${4:?missing scratch root}"
hf_home="${5:?missing Hugging Face cache root}"
upstream_root="${6:?missing completed job1425084 campaign root}"
upstream_repository="${7:?missing immutable upstream repository snapshot}"
upstream_spec="${8:?missing upstream specification}"
campaign_root="${9:?missing campaign root}"
private_key="${10:?missing private RNG key}"
spec="${11:?missing identified campaign specification}"
expected_spec_sha="${12:?missing specification SHA-256}"
minimum_vram_gib="${13:?missing minimum GPU VRAM GiB}"
step_memory="${14:?missing step host memory}"
step_gres="${15:?missing typed step GPU GRES}"
backup_root="${16:?missing persistent critical backup root}"
resume="${17:?missing resume flag}"

python_bin="$env_prefix/bin/python"
campaign="$repository/paper_repro/identified_slaclip_n128_campaign.py"
trainer="$repository/paper_repro/train_federated.py"
default_campaign="$repository/paper_repro/default_baseline_campaign.py"
default_spec="$repository/hpc/default-baseline-reproduction-spec.json"
runtime_lock="$repository/environment/paper-repro-runtime.lock"
exit_policy="$repository/hpc/full_slaclip_exit_policy.sh"
fixed_manifest="$campaign_root/stage1-fixed-screen-runtime-manifest.json"
development_manifest="$campaign_root/stage2-profile-development-runtime-manifest.json"
confirmation_manifest="$campaign_root/stage3-confirmation-runtime-manifest.json"
global_stop="$campaign_root/control/stop.request"
step_cpus="${SLURM_CPUS_PER_TASK:-0}"

if [[ "${SLURM_JOB_NUM_NODES:-0}" != 1 || "${SLURM_NTASKS:-0}" != 1 ]]; then
    echo "ERROR: identified campaign requires one node and one sequential task" >&2
    exit 2
fi
if [[ ! "$step_cpus" =~ ^[1-9][0-9]*$ || "$step_cpus" -gt 12 ]]; then
    echo "ERROR: invalid CPU allocation: $step_cpus" >&2
    exit 2
fi
if [[ "$step_gres" != "gpu:a100:1" || "$step_memory" != "80G" ]]; then
    echo "ERROR: runtime contract requires one typed A100 and 80G host RAM" >&2
    exit 2
fi
if [[ "$backup_root" != /* || "$backup_root" == / ]]; then
    echo "ERROR: unsafe persistent backup root: $backup_root" >&2
    exit 2
fi
if [[ "$resume" != 0 && "$resume" != 1 ]]; then
    echo "ERROR: resume flag must be 0 or 1" >&2
    exit 2
fi
for required in \
    "$python_bin" "$campaign" "$trainer" "$default_campaign" \
    "$default_spec" "$runtime_lock" "$exit_policy" "$upstream_root" \
    "$upstream_repository" "$upstream_spec" "$private_key" "$spec"; do
    [[ -e "$required" ]] || { echo "ERROR: missing runtime input: $required" >&2; exit 2; }
done
if [[ "$(git -C "$repository" rev-parse HEAD)" != "$expected_sha" || -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: immutable repository snapshot identity failed" >&2
    exit 2
fi
if [[ "$(sha256sum "$spec" | awk '{print $1}')" != "$expected_spec_sha" ]]; then
    echo "ERROR: identified specification changed after submission" >&2
    exit 2
fi
if [[ "$campaign_root" != /* || "$campaign_root" == / ]]; then
    echo "ERROR: unsafe campaign root: $campaign_root" >&2
    exit 2
fi
if [[ "$resume" -eq 1 ]]; then
    plan="$campaign_root/identified-campaign-plan.lock.json"
    [[ -f "$plan" && -f "$private_key" ]] || {
        echo "ERROR: resume requires the existing locked plan and RNG key" >&2
        exit 2
    }
    PYTHONDONTWRITEBYTECODE=1 "$python_bin" "$campaign" validate-resume-state \
        --plan "$plan" --private-key "$private_key" \
        --expected-code-sha "$expected_sha" --expected-spec-sha "$expected_spec_sha"
    # The path is exact and root-validated above.  A prior USR1 stop request
    # must be cleared before the first preflight srun, otherwise resume loops.
    if [[ -e "$global_stop" ]]; then rm -f -- "$global_stop"; fi
elif [[ -e "$global_stop" ]]; then
    echo "ERROR: new campaign unexpectedly contains a global stop request" >&2
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
export TMPDIR="${SLURM_TMPDIR:-$scratch_root/tmp/dp-lora-id-n128-$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
source "$exit_policy"
full_slaclip_stop_generation=0

campaign_stage="allocation_started"
campaign_completed=0
backup_critical_results() {
    mkdir -p "$backup_root/arm-status" "$backup_root/arm-logs" "$backup_root/arm-summaries"
    chmod 700 "$backup_root" "$backup_root/arm-status" "$backup_root/arm-logs" "$backup_root/arm-summaries"
    local candidate
    for candidate in "$spec" "$campaign_root"/*.json "$campaign_root"/*.csv; do
        if [[ -f "$candidate" ]]; then
            install -m 600 "$candidate" "$backup_root/$(basename "$candidate")"
        fi
    done
    if [[ -d "$campaign_root/arm-status" ]]; then
        cp -a "$campaign_root/arm-status/." "$backup_root/arm-status/"
    fi
    if [[ -d "$campaign_root/arm-logs" ]]; then
        cp -a "$campaign_root/arm-logs/." "$backup_root/arm-logs/"
    fi
    if [[ -d "$campaign_root/arms" ]]; then
        while IFS= read -r -d '' candidate; do
            local relative="${candidate#"$campaign_root/arms/"}"
            local destination="$backup_root/arm-summaries/$(dirname "$relative")"
            mkdir -p "$destination"
            install -m 600 "$candidate" "$destination/$(basename "$candidate")"
        done < <(find "$campaign_root/arms" -type f \
            \( -name final_summary.json -o -name run_config.json \) -print0)
    fi
    echo "critical_results_backup=$backup_root"
}
finalize_campaign() {
    local original_rc=$?
    local final_rc="$original_rc"
    trap - EXIT
    set +e
    # USR1 arrives only five minutes before termination.  Persist the latest
    # status/locks first; aggregation can re-read many round shards.
    backup_critical_results || true
    if [[ -f "$campaign_root/identified-campaign-plan.lock.json" ]]; then
        "$python_bin" "$campaign" aggregate \
            --campaign-root "$campaign_root" --spec "$spec" \
            --repository "$repository" --expected-code-sha "$expected_sha" || true
    fi
    backup_critical_results || true
    if [[ "$original_rc" -eq 0 && "$campaign_completed" -ne 1 ]]; then final_rc=1; fi
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
    if [[ "$classification" == "CHECKPOINTED_STOP" || -e "$global_stop" ]]; then return 75; fi
    return 0
}

echo "job_id=$SLURM_JOB_ID"
echo "dependency_parent_job=1425084"
echo "repository_sha=$expected_sha"
echo "campaign_root=$campaign_root"
nvidia-smi

campaign_stage="runtime_preflight"
run_step_signal_safe "$python_bin" -c \
    'import sys, torch; ok=torch.cuda.is_available(); print({"torch":torch.__version__,"cuda":torch.version.cuda,"available":ok}); ok or sys.exit("CUDA unavailable"); p=torch.cuda.get_device_properties(0); gib=p.total_memory/1024**3; print({"gpu":p.name,"vram_gib":gib}); gib >= float(sys.argv[1]) or sys.exit("insufficient VRAM")' \
    "$minimum_vram_gib"
run_step_signal_safe "$python_bin" -m pip check
run_step_signal_safe "$python_bin" "$default_campaign" validate-runtime \
    --spec "$default_spec" --runtime-lock "$runtime_lock" --hf-home "$hf_home"
campaign_stage="prepare_and_deep_upstream_validation"
prepare_args=(
    prepare --spec "$spec" --repository "$repository"
    --expected-code-sha "$expected_sha" --upstream-root "$upstream_root"
    --upstream-repository "$upstream_repository" --upstream-spec "$upstream_spec"
    --campaign-root "$campaign_root" --private-key "$private_key"
)
if [[ "$resume" -eq 1 ]]; then prepare_args+=(--resume); fi
run_step_signal_safe "$python_bin" "$campaign" "${prepare_args[@]}"
selected_input_manifest="$("$python_bin" - "$fixed_manifest" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["input_manifest_path"])
PY
)"
run_step_signal_safe "$python_bin" "$trainer" --check-inputs \
    --input-manifest "$selected_input_manifest"

# Exactly two lightweight real-model code-path smokes (the trainer reduces
# smoke dimensions internally): one fixed and one adaptive.  N=128/K=5 is
# validated by the locked manifests, mechanism gate, and every formal arm.
campaign_stage="fixed_smoke"
run_step_signal_safe "$python_bin" "$campaign" run-smoke \
    --manifest "$fixed_manifest" --campaign-root "$campaign_root" \
    --repository "$repository" --python-bin "$python_bin" \
    --private-key "$private_key" --arm-index 0 --label fixed-smoke

campaign_stage="fixed_screen_20_arms"
while IFS= read -r arm_index; do
    run_step_signal_safe "$python_bin" "$campaign" run-arm \
        --manifest "$fixed_manifest" --repository "$repository" \
        --python-bin "$python_bin" --private-key "$private_key" \
        --arm-index "$arm_index"
done < <("$python_bin" "$campaign" print-arm-indices --manifest "$fixed_manifest")

campaign_stage="lock_strongest_screened_fixed_and_calibration_C0"
run_step_signal_safe "$python_bin" "$campaign" lock-fixed \
    --campaign-root "$campaign_root" --spec "$spec" \
    --repository "$repository" --expected-code-sha "$expected_sha"

campaign_stage="adaptive_smoke"
run_step_signal_safe "$python_bin" "$campaign" run-smoke \
    --manifest "$development_manifest" --campaign-root "$campaign_root" \
    --repository "$repository" --python-bin "$python_bin" \
    --private-key "$private_key" --arm-index 0 --label adaptive-smoke

campaign_stage="five_profile_development_10_arms"
while IFS= read -r arm_index; do
    run_step_signal_safe "$python_bin" "$campaign" run-arm \
        --manifest "$development_manifest" --repository "$repository" \
        --python-bin "$python_bin" --private-key "$private_key" \
        --arm-index "$arm_index"
done < <("$python_bin" "$campaign" print-arm-indices --manifest "$development_manifest")

campaign_stage="lock_best_feasible_profile"
run_step_signal_safe "$python_bin" "$campaign" lock-development \
    --campaign-root "$campaign_root" --spec "$spec" \
    --repository "$repository" --expected-code-sha "$expected_sha"

campaign_stage="independent_confirmation_12_arms"
while IFS= read -r arm_index; do
    run_step_signal_safe "$python_bin" "$campaign" run-arm \
        --manifest "$confirmation_manifest" --repository "$repository" \
        --python-bin "$python_bin" --private-key "$private_key" \
        --arm-index "$arm_index"
done < <("$python_bin" "$campaign" print-arm-indices --manifest "$confirmation_manifest")

campaign_stage="strict_aggregate_and_gate"
run_step_signal_safe "$python_bin" "$campaign" aggregate \
    --campaign-root "$campaign_root" --spec "$spec" \
    --repository "$repository" --expected-code-sha "$expected_sha" \
    --require-complete

campaign_completed=1
campaign_stage="completed"
echo "campaign_status=COMPLETED"
