#!/usr/bin/env bash

#SBATCH --job-name=dp-lora-oracle-ceiling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# The submit wrapper supplies account, A100 GRES, memory, walltime, signals,
# dependency and log paths after validating the live cluster configuration.
set -euo pipefail
umask 077

repository="${1:?missing immutable repository snapshot}"
expected_sha="${2:?missing expected code SHA}"
env_prefix="${3:?missing environment prefix}"
scratch_root="${4:?missing scratch root}"
hf_home="${5:?missing Hugging Face cache root}"
input_manifest="${6:?missing immutable input manifest}"
campaign_root="${7:?missing campaign root}"
archive_root="${8:?missing persistent archive root}"
private_key="${9:?missing persistent campaign RNG key}"
spec="${10:?missing oracle-ceiling campaign specification}"
expected_spec_sha256="${11:?missing specification SHA-256}"
resume="${12:?missing resume flag}"
step_memory="${13:?missing step memory}"
step_gres="${14:?missing typed step GPU GRES}"
expected_gpu="${15:?missing expected CUDA device substring}"
min_vram_gib="${16:?missing minimum VRAM GiB}"

python_bin="$env_prefix/bin/python"
campaign="$repository/paper_repro/oracle_ceiling_campaign.py"
full="$repository/paper_repro/full_slaclip_campaign.py"
trainer="$repository/paper_repro/train_federated.py"
exit_policy="$repository/hpc/full_slaclip_exit_policy.sh"
master_manifest="$campaign_root/runtime-manifest.json"
preflight_manifest="$campaign_root/preflight-runtime-manifest.json"
stage2_manifest="$campaign_root/stage2-fixed-refinement-runtime-manifest.json"
stage3_manifest="$campaign_root/stage3-oracle-development-runtime-manifest.json"
stage4_manifest="$campaign_root/stage4-confirmation-runtime-manifest.json"
global_stop="$campaign_root/control/stop.request"
step_cpus="${SLURM_CPUS_PER_TASK:-0}"

if [[ "${SLURM_JOB_NUM_NODES:-0}" != 1 || "${SLURM_NTASKS:-0}" != 1 ]]; then
    echo "ERROR: oracle-ceiling campaign requires one node and one sequential task" >&2
    exit 2
fi
if [[ ! "$step_cpus" =~ ^[1-9][0-9]*$ || "$step_cpus" -gt 12 ]]; then
    echo "ERROR: invalid CPU allocation: $step_cpus" >&2
    exit 2
fi
if [[ ! "$step_gres" =~ ^gpu:[A-Za-z0-9_-]+:1$ ]]; then
    echo "ERROR: campaign requires exactly one typed GPU" >&2
    exit 2
fi
for required in "$python_bin" "$campaign" "$full" "$trainer" "$exit_policy" "$spec" "$input_manifest"; do
    [[ -e "$required" ]] || { echo "ERROR: missing runtime input: $required" >&2; exit 2; }
done
if [[ "$(git -C "$repository" rev-parse HEAD)" != "$expected_sha" || -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: immutable repository snapshot identity failed" >&2
    exit 2
fi
if [[ "$(sha256sum "$spec" | awk '{print $1}')" != "$expected_spec_sha256" ]]; then
    echo "ERROR: campaign specification changed after submission" >&2
    exit 2
fi
if [[ "$resume" != 0 && "$resume" != 1 ]]; then
    echo "ERROR: invalid resume flag" >&2
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
export TMPDIR="${SLURM_TMPDIR:-$scratch_root/tmp/dp-lora-oracle-ceiling-$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
source "$exit_policy"
full_slaclip_stop_generation=0

prepare_args=(
    prepare --campaign-root "$campaign_root" --spec "$spec"
    --repository "$repository" --expected-code-sha "$expected_sha"
    --input-manifest "$input_manifest" --private-key "$private_key"
)
if [[ "$resume" -eq 1 ]]; then prepare_args+=(--resume); fi
"$python_bin" "$campaign" "${prepare_args[@]}"

campaign_stage="prepared"
campaign_completed=0
finalize_campaign() {
    local original_rc=$?
    local final_rc="$original_rc"
    local terminal="FAILED"
    local reason="batch_exit_at_${campaign_stage}"
    trap - EXIT
    set +e
    if [[ "$original_rc" -eq 0 && "$campaign_completed" -eq 1 ]]; then
        terminal="COMPLETED"
        reason="all_326_arms_and_locks_revalidated"
    elif [[ "$original_rc" -eq 0 ]]; then
        final_rc=1
        reason="batch_exited_without_completion_at_${campaign_stage}"
    elif [[ "$original_rc" -eq 75 ]]; then
        reason="checkpointed_stop_at_${campaign_stage}"
    fi
    if [[ -f "$master_manifest" ]]; then
        "$python_bin" "$campaign" aggregate --campaign-root "$campaign_root" --spec "$spec" || true
        "$python_bin" "$full" mark-job-status \
            --campaign-root "$campaign_root" --runtime-manifest "$master_manifest" \
            --status "$terminal" --slurm-job-id "$SLURM_JOB_ID" \
            --repository-sha "$expected_sha" --reason "$reason" --exit-code "$final_rc" || true
        "$python_bin" "$full" archive-small --campaign-root "$campaign_root" --archive-root "$archive_root" || true
    fi
    exit "$final_rc"
}
trap finalize_campaign EXIT

"$python_bin" "$full" mark-job-status \
    --campaign-root "$campaign_root" --runtime-manifest "$master_manifest" \
    --status RUNNING --slurm-job-id "$SLURM_JOB_ID" \
    --repository-sha "$expected_sha" --reason allocation_started

request_stop() {
    full_slaclip_stop_generation=$((full_slaclip_stop_generation + 1))
    mkdir -p "$(dirname "$global_stop")"
    touch "$global_stop"
    echo "checkpoint_stop_requested_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap request_stop USR1 TERM INT

run_step() {
    srun --export=ALL --exact --exclusive --nodes=1 --ntasks=1 \
        "--cpus-per-task=$step_cpus" "--mem=$step_memory" "--gres=$step_gres" \
        "$@" < /dev/null
}

run_step_signal_safe() {
    local rc=0
    local classification
    local restore_errexit=0
    [[ "$-" == *e* ]] && restore_errexit=1
    run_step "$@" &
    local child_pid=$!
    set +e
    wait_for_full_slaclip_child "$child_pid"
    rc=$?
    if [[ "$restore_errexit" -eq 1 ]]; then set -e; fi
    classification="$(classify_full_slaclip_exit_codes "$rc")"
    if [[ "$classification" == "HARD_FAILURE" ]]; then return "$rc"; fi
    if [[ "$classification" == "CHECKPOINTED_STOP" || -e "$global_stop" ]]; then
        return 75
    fi
    return 0
}

run_arm() {
    local manifest="$1"
    local index="$2"
    local rc=0
    set +e
    run_step_signal_safe "$python_bin" "$full" run-arm \
        --manifest "$manifest" --arm-index "$index" --repository "$repository" \
        --python-bin "$python_bin" --private-key "$private_key"
    rc=$?
    set -e
    if [[ "$rc" -eq 75 || -e "$global_stop" ]]; then return 75; fi
    if [[ "$rc" -ne 0 ]]; then
        echo "ERROR: arm failed: manifest=$manifest index=$index rc=$rc" >&2
        return "$rc"
    fi
}

run_manifest_stage() {
    local stage_name="$1"
    local manifest="$2"
    local left right
    [[ -f "$manifest" ]] || { echo "ERROR: missing $stage_name manifest" >&2; return 2; }
    while IFS=$'\t' read -r left right; do
        echo "stage=$stage_name arm_index=$left"
        run_arm "$manifest" "$left" || return $?
        echo "stage=$stage_name arm_index=$right"
        run_arm "$manifest" "$right" || return $?
    done < <("$python_bin" "$campaign" waves --manifest "$manifest")
}

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "repository_sha=$expected_sha"
echo "campaign_root=$campaign_root"
nvidia-smi

campaign_stage="runtime_and_input_preflight"
run_step_signal_safe "$python_bin" -m pip check
run_step_signal_safe "$python_bin" "$trainer" --check-inputs --input-manifest "$input_manifest"
run_step_signal_safe "$python_bin" "$full" cuda-smoke --lane 0 \
    --output "$campaign_root/preflight/cuda-lane-0.json" \
    --expected-device-name "$expected_gpu" --min-vram-gib "$min_vram_gib"
run_step_signal_safe "$python_bin" "$full" run-smoke --manifest "$preflight_manifest" \
    --lane 0 --method paper_dp_lora --repository "$repository" \
    --python-bin "$python_bin" --private-key "$private_key"
run_step_signal_safe "$python_bin" "$full" run-smoke --manifest "$preflight_manifest" \
    --lane 0 --method oracle_slaclip_control --repository "$repository" \
    --python-bin "$python_bin" --private-key "$private_key"

identity=(
    --campaign-root "$campaign_root" --spec "$spec" --repository "$repository"
    --expected-code-sha "$expected_sha" --input-manifest "$input_manifest"
)

campaign_stage="groupwise_fixed_coarse"
run_manifest_stage "$campaign_stage" "$master_manifest"
"$python_bin" "$campaign" lock-coarse "${identity[@]}"
"$python_bin" "$campaign" aggregate --campaign-root "$campaign_root" --spec "$spec"
"$python_bin" "$full" archive-small --campaign-root "$campaign_root" --archive-root "$archive_root"

campaign_stage="groupwise_fixed_refinement"
run_manifest_stage "$campaign_stage" "$stage2_manifest"
"$python_bin" "$campaign" lock-fixed "${identity[@]}"
"$python_bin" "$campaign" aggregate --campaign-root "$campaign_root" --spec "$spec"
"$python_bin" "$full" archive-small --campaign-root "$campaign_root" --archive-root "$archive_root"

campaign_stage="exact_oracle_development"
run_manifest_stage "$campaign_stage" "$stage3_manifest"
"$python_bin" "$campaign" lock-oracle "${identity[@]}"
"$python_bin" "$campaign" aggregate --campaign-root "$campaign_root" --spec "$spec"
"$python_bin" "$full" archive-small --campaign-root "$campaign_root" --archive-root "$archive_root"

campaign_stage="fresh_seed_confirmation"
run_manifest_stage "$campaign_stage" "$stage4_manifest"

campaign_stage="oracle_ceiling_gate"
"$python_bin" "$campaign" lock-gate "${identity[@]}"

campaign_stage="final_revalidation"
"$python_bin" "$campaign" aggregate --campaign-root "$campaign_root" --spec "$spec" --require-complete
"$python_bin" "$full" archive-small --campaign-root "$campaign_root" --archive-root "$archive_root"
campaign_completed=1
campaign_stage="completed"
echo "campaign_status=COMPLETED"
echo "campaign_summary=$campaign_root/campaign_summary.json"
echo "persistent_small_artifact_archive=$archive_root"
