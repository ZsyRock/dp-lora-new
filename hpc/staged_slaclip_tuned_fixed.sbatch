#!/usr/bin/env bash

#SBATCH --job-name=dp-lora-staged-slaclip
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4

# Account, partition, typed GRES, memory, walltime, signal, and log paths are
# supplied by submit_staged_slaclip_tuned_fixed.sh after live scheduler gates.
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
spec="${10:?missing staged campaign specification}"
expected_spec_sha256="${11:?missing specification SHA-256}"
resume="${12:?missing resume flag}"
lane_memory="${13:?missing per-lane host memory}"
lane_gres="${14:?missing typed per-lane GPU GRES}"
expected_gpu="${15:?missing expected CUDA device-name substring}"
min_vram_gib="${16:?missing minimum per-lane VRAM GiB}"

python_bin="$env_prefix/bin/python"
staged="$repository/paper_repro/staged_slaclip_campaign.py"
full="$repository/paper_repro/full_slaclip_campaign.py"
trainer="$repository/paper_repro/train_federated.py"
exit_policy="$repository/hpc/full_slaclip_exit_policy.sh"
master_manifest="$campaign_root/runtime-manifest.json"
preflight_manifest="$campaign_root/preflight-runtime-manifest.json"
stage2_manifest="$campaign_root/stage2-runtime-manifest.json"
stage3_manifest="$campaign_root/stage3-runtime-manifest.json"
global_stop="$campaign_root/control/stop.request"
lane_cpus="${SLURM_CPUS_PER_TASK:-0}"
parent_lanes="${SLURM_NTASKS:-0}"

if [[ "${SLURM_JOB_NUM_NODES:-0}" != 1 || ! "$parent_lanes" =~ ^[12]$ ]]; then
    echo "ERROR: staged campaign requires one node and one or two GPU lanes" >&2
    exit 2
fi
if [[ ! "$lane_cpus" =~ ^[1-9][0-9]*$ || "$lane_cpus" -gt 12 ]]; then
    echo "ERROR: invalid per-lane CPU allocation: $lane_cpus" >&2
    exit 2
fi
for required in "$python_bin" "$staged" "$full" "$trainer" "$exit_policy"; do
    [[ -e "$required" ]] || { echo "ERROR: missing runtime input: $required" >&2; exit 2; }
done
if [[ "$(git -C "$repository" rev-parse HEAD)" != "$expected_sha" ]]; then
    echo "ERROR: immutable repository SHA changed after submission" >&2
    exit 2
fi
if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: immutable repository snapshot became dirty" >&2
    exit 2
fi
if [[ "$(sha256sum "$spec" | awk '{print $1}')" != "$expected_spec_sha256" ]]; then
    echo "ERROR: staged campaign specification changed after submission" >&2
    exit 2
fi
if [[ "$resume" != 0 && "$resume" != 1 ]]; then
    echo "ERROR: invalid resume flag" >&2
    exit 2
fi
if [[ ! "$lane_gres" =~ ^gpu:[A-Za-z0-9_-]+:1$ ]]; then
    echo "ERROR: each lane must request exactly one typed GPU" >&2
    exit 2
fi
if [[ -z "$expected_gpu" || ! "$min_vram_gib" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid CUDA device gate" >&2
    exit 2
fi

source "$exit_policy"
full_slaclip_stop_generation=0
export HF_HOME="$hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS="$lane_cpus"
export TMPDIR="${SLURM_TMPDIR:-$scratch_root/tmp/dp-lora-staged-$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

prepare_args=(
    prepare
    --campaign-root "$campaign_root"
    --spec "$spec"
    --repository "$repository"
    --expected-code-sha "$expected_sha"
    --input-manifest "$input_manifest"
    --private-key "$private_key"
)
if [[ "$resume" -eq 1 ]]; then
    prepare_args+=(--resume)
fi
"$python_bin" "$staged" "${prepare_args[@]}"

campaign_stage="prepared"
campaign_completed=0
finalize_campaign() {
    local original_rc=$?
    local final_rc="$original_rc"
    local terminal_status="FAILED"
    local terminal_reason="batch_exit_at_${campaign_stage}"
    trap - EXIT
    set +e
    if [[ "$original_rc" -eq 0 && "$campaign_completed" -eq 1 ]]; then
        terminal_status="COMPLETED"
        terminal_reason="all_stages_locked_revalidated_and_archived"
    elif [[ "$original_rc" -eq 0 ]]; then
        final_rc=1
        terminal_reason="batch_exited_without_completion_at_${campaign_stage}"
    elif [[ "$original_rc" -eq 75 ]]; then
        terminal_reason="checkpointed_stop_at_${campaign_stage}"
    fi
    if [[ -f "$master_manifest" ]]; then
        "$python_bin" "$staged" aggregate \
            --campaign-root "$campaign_root" --spec "$spec" || true
        "$python_bin" "$full" mark-job-status \
            --campaign-root "$campaign_root" \
            --runtime-manifest "$master_manifest" \
            --status "$terminal_status" \
            --slurm-job-id "$SLURM_JOB_ID" \
            --repository-sha "$expected_sha" \
            --reason "$terminal_reason" \
            --exit-code "$final_rc" || true
        "$python_bin" "$full" archive-small \
            --campaign-root "$campaign_root" --archive-root "$archive_root" || true
    fi
    exit "$final_rc"
}
trap finalize_campaign EXIT

"$python_bin" "$full" mark-job-status \
    --campaign-root "$campaign_root" \
    --runtime-manifest "$master_manifest" \
    --status RUNNING \
    --slurm-job-id "$SLURM_JOB_ID" \
    --repository-sha "$expected_sha" \
    --reason allocation_started

request_stop() {
    full_slaclip_stop_generation=$((full_slaclip_stop_generation + 1))
    mkdir -p "$(dirname "$global_stop")"
    touch "$global_stop"
    echo "checkpoint_stop_requested_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap request_stop USR1 TERM INT

run_step() {
    srun \
        --export=ALL \
        --exact \
        --exclusive \
        --nodes=1 \
        --ntasks=1 \
        "--cpus-per-task=$lane_cpus" \
        "--mem=$lane_memory" \
        "--gres=$lane_gres" \
        "$@" < /dev/null
}

wait_pair_or_stop() {
    local first_pid="$1"
    local second_pid="${2:-}"
    local first_rc=0
    local second_rc=0
    set +e
    wait_for_full_slaclip_child "$first_pid"
    first_rc=$?
    if [[ "$first_rc" -ne 0 ]]; then
        request_stop
    fi
    if [[ -n "$second_pid" ]]; then
        wait_for_full_slaclip_child "$second_pid"
        second_rc=$?
    fi
    set -e
    local classification
    classification="$(classify_full_slaclip_exit_codes "$first_rc" "$second_rc")"
    if [[ "$classification" == "HARD_FAILURE" ]]; then
        echo "ERROR: staged wave failed: lane0=$first_rc lane1=$second_rc" >&2
        return 1
    fi
    if [[ "$classification" == "CHECKPOINTED_STOP" || -e "$global_stop" ]]; then
        echo "Staged campaign stopped at a checkpoint boundary."
        return 75
    fi
    return 0
}

run_manifest_stage() {
    local stage_name="$1"
    local manifest="$2"
    [[ -f "$manifest" ]] || { echo "ERROR: missing $stage_name manifest" >&2; return 2; }
    local wave=0
    while IFS=$'\t' read -r arm_index_0 arm_index_1; do
        echo "stage=$stage_name wave=$wave arm_indices=$arm_index_0,$arm_index_1"
        if [[ "$parent_lanes" -eq 2 ]]; then
            run_step "$python_bin" "$full" run-arm \
                --manifest "$manifest" \
                --arm-index "$arm_index_0" \
                --repository "$repository" \
                --python-bin "$python_bin" \
                --private-key "$private_key" &
            local first_pid=$!
            run_step "$python_bin" "$full" run-arm \
                --manifest "$manifest" \
                --arm-index "$arm_index_1" \
                --repository "$repository" \
                --python-bin "$python_bin" \
                --private-key "$private_key" &
            local second_pid=$!
            wait_pair_or_stop "$first_pid" "$second_pid" || return $?
        else
            local arm_index
            for arm_index in "$arm_index_0" "$arm_index_1"; do
                run_step "$python_bin" "$full" run-arm \
                    --manifest "$manifest" \
                    --arm-index "$arm_index" \
                    --repository "$repository" \
                    --python-bin "$python_bin" \
                    --private-key "$private_key" &
                local only_pid=$!
                wait_pair_or_stop "$only_pid" || return $?
            done
        fi
        wave=$((wave + 1))
    done < <("$python_bin" "$staged" waves --manifest "$manifest")
}

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "repository_sha=$expected_sha"
echo "campaign_root=$campaign_root"
nvidia-smi

campaign_stage="runtime_and_input_preflight"
run_step "$python_bin" -m pip check
run_step "$python_bin" "$trainer" --check-inputs --input-manifest "$input_manifest"

campaign_stage="cuda_preflight"
run_step "$python_bin" "$full" cuda-smoke \
    --lane 0 \
    --output "$campaign_root/preflight/cuda-lane-0.json" \
    --expected-device-name "$expected_gpu" \
    --min-vram-gib "$min_vram_gib" &
cuda_pid_0=$!
cuda_pid_1=""
if [[ "$parent_lanes" -eq 2 ]]; then
    run_step "$python_bin" "$full" cuda-smoke \
        --lane 1 \
        --output "$campaign_root/preflight/cuda-lane-1.json" \
        --expected-device-name "$expected_gpu" \
        --min-vram-gib "$min_vram_gib" &
    cuda_pid_1=$!
fi
wait_pair_or_stop "$cuda_pid_0" "$cuda_pid_1"

campaign_stage="real_model_preflight"
run_step "$python_bin" "$full" run-smoke \
    --manifest "$preflight_manifest" \
    --lane 0 \
    --method paper_dp_lora \
    --repository "$repository" \
    --python-bin "$python_bin" \
    --private-key "$private_key" &
smoke_pid_0=$!
smoke_pid_1=""
if [[ "$parent_lanes" -eq 2 ]]; then
    run_step "$python_bin" "$full" run-smoke \
        --manifest "$preflight_manifest" \
        --lane 1 \
        --method slaclip_dp_lora \
        --repository "$repository" \
        --python-bin "$python_bin" \
        --private-key "$private_key" &
    smoke_pid_1=$!
fi
wait_pair_or_stop "$smoke_pid_0" "$smoke_pid_1"
if [[ "$parent_lanes" -eq 1 ]]; then
    run_step "$python_bin" "$full" run-smoke \
        --manifest "$preflight_manifest" \
        --lane 1 \
        --method slaclip_dp_lora \
        --repository "$repository" \
        --python-bin "$python_bin" \
        --private-key "$private_key" &
    smoke_pid_only=$!
    wait_pair_or_stop "$smoke_pid_only"
fi

identity_args=(
    --campaign-root "$campaign_root"
    --spec "$spec"
    --repository "$repository"
    --expected-code-sha "$expected_sha"
    --input-manifest "$input_manifest"
)

campaign_stage="fixed_development"
run_manifest_stage "$campaign_stage" "$master_manifest"
"$python_bin" "$staged" lock-fixed "${identity_args[@]}"
"$python_bin" "$staged" aggregate --campaign-root "$campaign_root" --spec "$spec"
"$python_bin" "$full" archive-small \
    --campaign-root "$campaign_root" --archive-root "$archive_root"

campaign_stage="slaclip_development"
run_manifest_stage "$campaign_stage" "$stage2_manifest"
"$python_bin" "$staged" lock-slaclip "${identity_args[@]}"
"$python_bin" "$staged" aggregate --campaign-root "$campaign_root" --spec "$spec"
"$python_bin" "$full" archive-small \
    --campaign-root "$campaign_root" --archive-root "$archive_root"

campaign_stage="confirmation"
run_manifest_stage "$campaign_stage" "$stage3_manifest"

campaign_stage="final_revalidation"
"$python_bin" "$staged" aggregate \
    --campaign-root "$campaign_root" --spec "$spec" --require-complete
"$python_bin" "$full" archive-small \
    --campaign-root "$campaign_root" --archive-root "$archive_root"
campaign_completed=1
campaign_stage="completed"
echo "campaign_status=COMPLETED"
echo "campaign_summary=$campaign_root/campaign_summary.json"
echo "persistent_small_artifact_archive=$archive_root"
