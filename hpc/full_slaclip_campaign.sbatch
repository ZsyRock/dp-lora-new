#!/usr/bin/env bash

#SBATCH --job-name=dp-lora-full-slaclip
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12

# Account, partition, typed GRES, host RAM, walltime, signal, and log paths are
# supplied by submit_full_slaclip_campaign.sh after live cluster inspection.
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
spec="${10:?missing campaign specification}"
expected_spec_sha256="${11:?missing specification SHA-256}"
resume="${12:?missing resume flag}"
lane_memory="${13:?missing per-lane host memory}"
lane_gres="${14:?missing typed per-lane GPU GRES}"
expected_gpu="${15:?missing expected CUDA device-name substring}"
min_vram_gib="${16:?missing minimum per-lane VRAM GiB}"

python_bin="$env_prefix/bin/python"
coordinator="$repository/paper_repro/full_slaclip_campaign.py"
trainer="$repository/paper_repro/train_federated.py"
exit_policy="$repository/hpc/full_slaclip_exit_policy.sh"
runtime_manifest="$campaign_root/runtime-manifest.json"
global_stop="$campaign_root/control/stop.request"
lane_cpus="${SLURM_CPUS_PER_TASK:-0}"
parent_lanes="${SLURM_NTASKS:-0}"

if [[ "${SLURM_JOB_NUM_NODES:-0}" != 1 || ! "$parent_lanes" =~ ^[12]$ ]]; then
    echo "ERROR: full SlaClip campaign requires one node and one or two GPU lanes" >&2
    exit 2
fi
if [[ ! "$lane_cpus" =~ ^[1-9][0-9]*$ || "$lane_cpus" -gt 12 ]]; then
    echo "ERROR: invalid per-lane CPU allocation: $lane_cpus" >&2
    exit 2
fi
if [[ ! -x "$python_bin" || ! -f "$coordinator" || ! -f "$trainer" || ! -f "$exit_policy" ]]; then
    echo "ERROR: pinned runtime or repository entry point is missing" >&2
    exit 2
fi
if [[ "$(git -C "$repository" rev-parse HEAD)" != "$expected_sha" ]]; then
    echo "ERROR: immutable repository SHA changed after submission" >&2
    exit 2
fi
if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: immutable repository snapshot became dirty" >&2
    exit 2
fi
if [[ "$(sha256sum "$spec" | awk '{print $1}')" != "$expected_spec_sha256" ]]; then
    echo "ERROR: checked-in campaign specification changed after submission" >&2
    exit 2
fi
if [[ "$resume" != 0 && "$resume" != 1 ]]; then
    echo "ERROR: invalid resume flag" >&2
    exit 2
fi
if [[ ! "$lane_gres" =~ ^gpu:[A-Za-z0-9_-]+:1$ ]]; then
    echo "ERROR: each campaign lane must request exactly one typed GPU" >&2
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
# sbatch --export=NONE deliberately sanitizes the submission environment.
# Override its inherited step policy only after defining the exact environment
# above so every exclusive srun lane receives the reproducibility contract.
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS="$lane_cpus"
export TMPDIR="${SLURM_TMPDIR:-$scratch_root/tmp/dp-lora-full-slaclip-$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

prepare_args=(
    prepare
    --spec "$spec"
    --repository "$repository"
    --expected-code-sha "$expected_sha"
    --input-manifest "$input_manifest"
    --campaign-root "$campaign_root"
    --private-key "$private_key"
)
if [[ "$resume" -eq 1 ]]; then
    prepare_args+=(--resume)
fi
"$python_bin" "$coordinator" "${prepare_args[@]}"

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
        terminal_reason="all_arms_revalidated_and_archived"
    elif [[ "$original_rc" -eq 0 ]]; then
        final_rc=1
        terminal_reason="batch_exited_without_completion_at_${campaign_stage}"
    elif [[ "$original_rc" -eq 75 ]]; then
        terminal_reason="checkpointed_stop_at_${campaign_stage}"
    fi
    "$python_bin" "$coordinator" mark-job-status \
        --campaign-root "$campaign_root" \
        --runtime-manifest "$runtime_manifest" \
        --status "$terminal_status" \
        --slurm-job-id "$SLURM_JOB_ID" \
        --repository-sha "$expected_sha" \
        --reason "$terminal_reason" \
        --exit-code "$final_rc"
    marker_rc=$?
    if [[ "$marker_rc" -ne 0 ]]; then
        echo "ERROR: failed to write terminal campaign job status (rc=$marker_rc)" >&2
        if [[ "$final_rc" -eq 0 ]]; then
            final_rc=1
        fi
    fi
    "$python_bin" "$coordinator" archive-small \
        --campaign-root "$campaign_root" --archive-root "$archive_root" || true
    exit "$final_rc"
}
trap finalize_campaign EXIT
"$python_bin" "$coordinator" mark-job-status \
    --campaign-root "$campaign_root" \
    --runtime-manifest "$runtime_manifest" \
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
    # The formal-wave loop reads its manifest from a process-substitution
    # pipe.  Background srun processes must never inherit that descriptor or
    # they can consume future wave rows and make the loop end early.
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

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "repository_sha=$expected_sha"
echo "campaign_root=$campaign_root"
echo "stop_file=$global_stop"
nvidia-smi

# A single-task step prevents a multi-task parent allocation from confusing
# single-process input/runtime checks.
campaign_stage="runtime_and_input_preflight"
run_step "$python_bin" -m pip check
run_step "$python_bin" "$trainer" --check-inputs --input-manifest "$input_manifest"

# Verify every independently scheduled typed-GPU lane before the formal matrix.
campaign_stage="cuda_preflight"
smoke_rc_0=0
smoke_rc_1=0
run_step "$python_bin" "$coordinator" cuda-smoke \
    --lane 0 \
    --output "$campaign_root/preflight/cuda-lane-0.json" \
    --expected-device-name "$expected_gpu" \
    --min-vram-gib "$min_vram_gib" &
smoke_pid_0=$!
if [[ "$parent_lanes" -eq 2 ]]; then
    run_step "$python_bin" "$coordinator" cuda-smoke \
        --lane 1 \
        --output "$campaign_root/preflight/cuda-lane-1.json" \
        --expected-device-name "$expected_gpu" \
        --min-vram-gib "$min_vram_gib" &
    smoke_pid_1=$!
fi
set +e
wait_for_full_slaclip_child "$smoke_pid_0"
smoke_rc_0=$?
if [[ "$smoke_rc_0" -ne 0 ]]; then
    request_stop
fi
if [[ "$parent_lanes" -eq 2 ]]; then
    wait_for_full_slaclip_child "$smoke_pid_1"
    smoke_rc_1=$?
fi
set -e
if [[ "$smoke_rc_0" -ne 0 || "$smoke_rc_1" -ne 0 ]]; then
    "$python_bin" "$coordinator" aggregate --manifest "$runtime_manifest" || true
    "$python_bin" "$coordinator" archive-small \
        --campaign-root "$campaign_root" --archive-root "$archive_root" || true
    echo "ERROR: typed-GPU CUDA preflight failed: lane0=$smoke_rc_0 lane1=$smoke_rc_1" >&2
    exit 1
fi

# Exercise the actual pinned models, LoRA wiring, paper baseline mechanism,
# full noisy CDF-endpoint controller, exact-endpoint NON-DP oracle control,
# checkpoint writer, and final validation.  The baseline/full pair shares its
# private RNG domain; the oracle then runs on one verified lane.
# On resume, train_federated.py --resume deeply revalidates completed smokes.
campaign_stage="real_model_preflight"
run_step "$python_bin" "$coordinator" run-smoke \
    --manifest "$runtime_manifest" \
    --lane 0 \
    --method paper_dp_lora \
    --repository "$repository" \
    --python-bin "$python_bin" \
    --private-key "$private_key" &
model_smoke_pid_0=$!
model_smoke_rc_0=0
model_smoke_rc_1=0
if [[ "$parent_lanes" -eq 2 ]]; then
    run_step "$python_bin" "$coordinator" run-smoke \
        --manifest "$runtime_manifest" \
        --lane 1 \
        --method slaclip_dp_lora \
        --repository "$repository" \
        --python-bin "$python_bin" \
        --private-key "$private_key" &
    model_smoke_pid_1=$!
fi
set +e
wait_for_full_slaclip_child "$model_smoke_pid_0"
model_smoke_rc_0=$?
if [[ "$model_smoke_rc_0" -ne 0 ]]; then
    request_stop
fi
if [[ "$parent_lanes" -eq 2 ]]; then
    wait_for_full_slaclip_child "$model_smoke_pid_1"
    model_smoke_rc_1=$?
elif [[ "$model_smoke_rc_0" -eq 0 ]]; then
    run_step "$python_bin" "$coordinator" run-smoke \
        --manifest "$runtime_manifest" \
        --lane 1 \
        --method slaclip_dp_lora \
        --repository "$repository" \
        --python-bin "$python_bin" \
        --private-key "$private_key" &
    model_smoke_pid_1=$!
    wait_for_full_slaclip_child "$model_smoke_pid_1"
    model_smoke_rc_1=$?
else
    model_smoke_rc_1=0
fi
set -e
if [[ "$model_smoke_rc_0" -ne 0 || "$model_smoke_rc_1" -ne 0 ]]; then
    "$python_bin" "$coordinator" aggregate --manifest "$runtime_manifest" || true
    "$python_bin" "$coordinator" archive-small \
        --campaign-root "$campaign_root" --archive-root "$archive_root" || true
    model_smoke_classification="$(
        classify_full_slaclip_exit_codes "$model_smoke_rc_0" "$model_smoke_rc_1"
    )"
    if [[ "$model_smoke_classification" == "HARD_FAILURE" ]]; then
        echo "ERROR: paired real-model preflight failed: baseline=$model_smoke_rc_0 full_slaclip=$model_smoke_rc_1" >&2
        exit 1
    fi
    if [[ "$model_smoke_classification" == "CHECKPOINTED_STOP" ]]; then
        echo "Real-model preflight stopped at a checkpoint boundary; resume the same run ID."
        exit 75
    fi
    echo "ERROR: real-model preflight reached an unclassified failure state" >&2
    exit 1
fi

# The oracle path deliberately retains gradient/slack noise but consumes exact
# endpoint signals for the next threshold.  It is an explicitly NON-DP
# mechanism diagnostic and must pass a real-model smoke before formal arms.
model_smoke_rc_oracle=0
run_step "$python_bin" "$coordinator" run-smoke \
    --manifest "$runtime_manifest" \
    --lane 0 \
    --method oracle_slaclip_control \
    --repository "$repository" \
    --python-bin "$python_bin" \
    --private-key "$private_key" &
model_smoke_pid_oracle=$!
set +e
wait_for_full_slaclip_child "$model_smoke_pid_oracle"
model_smoke_rc_oracle=$?
set -e
if [[ "$model_smoke_rc_oracle" -ne 0 ]]; then
    "$python_bin" "$coordinator" aggregate --manifest "$runtime_manifest" || true
    "$python_bin" "$coordinator" archive-small \
        --campaign-root "$campaign_root" --archive-root "$archive_root" || true
    model_smoke_oracle_classification="$(
        classify_full_slaclip_exit_codes "$model_smoke_rc_oracle"
    )"
    if [[ "$model_smoke_oracle_classification" == "CHECKPOINTED_STOP" ]]; then
        echo "Oracle real-model preflight stopped at a checkpoint boundary; resume the same run ID."
        exit 75
    fi
    echo "ERROR: oracle real-model preflight failed: oracle=$model_smoke_rc_oracle" >&2
    exit 1
fi

"$python_bin" "$coordinator" aggregate --manifest "$runtime_manifest"
"$python_bin" "$coordinator" archive-small \
    --campaign-root "$campaign_root" --archive-root "$archive_root"

if [[ -e "$global_stop" ]]; then
    echo "Checkpoint stop was requested before the first campaign wave."
    exit 75
fi

campaign_stage="formal_matrix"
if [[ "$parent_lanes" -eq 2 ]]; then
    while IFS=$'\t' read -r arm_index_0 arm_index_1; do
        echo "starting_wave=$((arm_index_0 / 2)) arm_indices=$arm_index_0,$arm_index_1"
        run_step "$python_bin" "$coordinator" run-arm \
            --manifest "$runtime_manifest" \
            --arm-index "$arm_index_0" \
            --repository "$repository" \
            --python-bin "$python_bin" \
            --private-key "$private_key" &
        arm_pid_0=$!
        run_step "$python_bin" "$coordinator" run-arm \
            --manifest "$runtime_manifest" \
            --arm-index "$arm_index_1" \
            --repository "$repository" \
            --python-bin "$python_bin" \
            --private-key "$private_key" &
        arm_pid_1=$!

        set +e
        wait_for_full_slaclip_child "$arm_pid_0"
        arm_rc_0=$?
        if [[ "$arm_rc_0" -ne 0 ]]; then
            request_stop
        fi
        wait_for_full_slaclip_child "$arm_pid_1"
        arm_rc_1=$?
        set -e

        "$python_bin" "$coordinator" aggregate --manifest "$runtime_manifest"
        "$python_bin" "$coordinator" archive-small \
            --campaign-root "$campaign_root" --archive-root "$archive_root"

        arm_classification="$(classify_full_slaclip_exit_codes "$arm_rc_0" "$arm_rc_1")"
        if [[ "$arm_classification" == "HARD_FAILURE" ]]; then
            echo "ERROR: campaign wave failed: lane0=$arm_rc_0 lane1=$arm_rc_1" >&2
            exit 1
        fi
        if [[ "$arm_classification" == "CHECKPOINTED_STOP" || -e "$global_stop" ]]; then
            echo "Campaign stopped at a checkpoint boundary; resume the same run ID."
            exit 75
        fi
    done < <("$python_bin" "$coordinator" waves --manifest "$runtime_manifest")
else
    while IFS=$'\t' read -r arm_index_0 arm_index_1; do
        echo "starting_sequential_wave=$((arm_index_0 / 2)) arm_indices=$arm_index_0,$arm_index_1"
        for arm_index in "$arm_index_0" "$arm_index_1"; do
            run_step "$python_bin" "$coordinator" run-arm \
                --manifest "$runtime_manifest" \
                --arm-index "$arm_index" \
                --repository "$repository" \
                --python-bin "$python_bin" \
                --private-key "$private_key" &
            arm_pid=$!
            set +e
            wait_for_full_slaclip_child "$arm_pid"
            arm_rc=$?
            set -e
            "$python_bin" "$coordinator" aggregate --manifest "$runtime_manifest"
            "$python_bin" "$coordinator" archive-small \
                --campaign-root "$campaign_root" --archive-root "$archive_root"

            arm_classification="$(classify_full_slaclip_exit_codes "$arm_rc")"
            if [[ "$arm_classification" == "HARD_FAILURE" ]]; then
                echo "ERROR: sequential campaign arm failed: index=$arm_index rc=$arm_rc" >&2
                exit 1
            fi
            if [[ "$arm_classification" == "CHECKPOINTED_STOP" || -e "$global_stop" ]]; then
                echo "Campaign stopped at a checkpoint boundary; resume the same run ID."
                exit 75
            fi
        done
    done < <("$python_bin" "$coordinator" waves --manifest "$runtime_manifest")
fi

campaign_stage="final_revalidation"
"$python_bin" "$coordinator" aggregate \
    --manifest "$runtime_manifest" --require-complete
"$python_bin" "$coordinator" archive-small \
    --campaign-root "$campaign_root" --archive-root "$archive_root"
campaign_completed=1
campaign_stage="completed"
echo "campaign_status=COMPLETED"
echo "campaign_summary=$campaign_root/campaign_summary.json"
echo "persistent_small_artifact_archive=$archive_root"
