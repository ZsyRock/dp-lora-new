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

python_bin="$env_prefix/bin/python"
coordinator="$repository/paper_repro/full_slaclip_campaign.py"
trainer="$repository/paper_repro/train_federated.py"
runtime_manifest="$campaign_root/runtime-manifest.json"
global_stop="$campaign_root/control/stop.request"
lane_cpus="${SLURM_CPUS_PER_TASK:-0}"

if [[ "${SLURM_JOB_NUM_NODES:-0}" != 1 || "${SLURM_NTASKS:-0}" != 2 ]]; then
    echo "ERROR: full SlaClip campaign requires one node and two parent tasks" >&2
    exit 2
fi
if [[ ! "$lane_cpus" =~ ^[1-9][0-9]*$ || "$lane_cpus" -gt 12 ]]; then
    echo "ERROR: invalid per-lane CPU allocation: $lane_cpus" >&2
    exit 2
fi
if [[ ! -x "$python_bin" || ! -f "$coordinator" || ! -f "$trainer" ]]; then
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

export HF_HOME="$hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
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

request_stop() {
    mkdir -p "$(dirname "$global_stop")"
    touch "$global_stop"
    echo "checkpoint_stop_requested_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap request_stop USR1 TERM INT

run_step() {
    srun \
        --exact \
        --exclusive \
        --nodes=1 \
        --ntasks=1 \
        "--cpus-per-task=$lane_cpus" \
        "--mem=$lane_memory" \
        --gres=gpu:h200:1 \
        "$@"
}

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "repository_sha=$expected_sha"
echo "campaign_root=$campaign_root"
echo "stop_file=$global_stop"
nvidia-smi

# A single-task step prevents the two-task parent allocation from confusing
# single-process input/runtime checks.
run_step "$python_bin" -m pip check
run_step "$python_bin" "$trainer" --check-inputs --input-manifest "$input_manifest"

# Verify both independently scheduled H200 lanes before the formal matrix.
run_step "$python_bin" "$coordinator" cuda-smoke \
    --lane 0 --output "$campaign_root/preflight/cuda-lane-0.json" &
smoke_pid_0=$!
run_step "$python_bin" "$coordinator" cuda-smoke \
    --lane 1 --output "$campaign_root/preflight/cuda-lane-1.json" &
smoke_pid_1=$!
set +e
wait "$smoke_pid_0"
smoke_rc_0=$?
if [[ "$smoke_rc_0" -ne 0 ]]; then
    request_stop
fi
wait "$smoke_pid_1"
smoke_rc_1=$?
set -e
if [[ "$smoke_rc_0" -ne 0 || "$smoke_rc_1" -ne 0 ]]; then
    "$python_bin" "$coordinator" aggregate --manifest "$runtime_manifest" || true
    "$python_bin" "$coordinator" archive-small \
        --campaign-root "$campaign_root" --archive-root "$archive_root" || true
    echo "ERROR: two-lane H200 CUDA preflight failed: lane0=$smoke_rc_0 lane1=$smoke_rc_1" >&2
    exit 1
fi

# Exercise the actual pinned models, LoRA wiring, paper baseline mechanism,
# full CDF-endpoint controller, checkpoint writer, and final validation.  Both
# smokes share their private RNG domain and run concurrently on the two lanes.
# On resume, train_federated.py --resume deeply revalidates completed smokes.
run_step "$python_bin" "$coordinator" run-smoke \
    --manifest "$runtime_manifest" \
    --lane 0 \
    --repository "$repository" \
    --python-bin "$python_bin" \
    --private-key "$private_key" &
model_smoke_pid_0=$!
run_step "$python_bin" "$coordinator" run-smoke \
    --manifest "$runtime_manifest" \
    --lane 1 \
    --repository "$repository" \
    --python-bin "$python_bin" \
    --private-key "$private_key" &
model_smoke_pid_1=$!
set +e
wait "$model_smoke_pid_0"
model_smoke_rc_0=$?
if [[ "$model_smoke_rc_0" -ne 0 ]]; then
    request_stop
fi
wait "$model_smoke_pid_1"
model_smoke_rc_1=$?
set -e
if [[ "$model_smoke_rc_0" -ne 0 || "$model_smoke_rc_1" -ne 0 ]]; then
    "$python_bin" "$coordinator" aggregate --manifest "$runtime_manifest" || true
    "$python_bin" "$coordinator" archive-small \
        --campaign-root "$campaign_root" --archive-root "$archive_root" || true
    if [[ "$model_smoke_rc_0" -eq 75 || "$model_smoke_rc_1" -eq 75 ]]; then
        echo "Real-model preflight stopped at a checkpoint boundary; resume the same run ID."
        exit 75
    fi
    echo "ERROR: paired real-model preflight failed: baseline=$model_smoke_rc_0 full_slaclip=$model_smoke_rc_1" >&2
    exit 1
fi

"$python_bin" "$coordinator" aggregate --manifest "$runtime_manifest"
"$python_bin" "$coordinator" archive-small \
    --campaign-root "$campaign_root" --archive-root "$archive_root"

if [[ -e "$global_stop" ]]; then
    echo "Checkpoint stop was requested before the first campaign wave."
    exit 75
fi

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
    wait "$arm_pid_0"
    arm_rc_0=$?
    if [[ "$arm_rc_0" -ne 0 ]]; then
        request_stop
    fi
    wait "$arm_pid_1"
    arm_rc_1=$?
    set -e

    "$python_bin" "$coordinator" aggregate --manifest "$runtime_manifest"
    "$python_bin" "$coordinator" archive-small \
        --campaign-root "$campaign_root" --archive-root "$archive_root"

    if [[ "$arm_rc_0" -eq 75 || "$arm_rc_1" -eq 75 || -e "$global_stop" ]]; then
        echo "Campaign stopped at a checkpoint boundary; resume the same run ID."
        exit 75
    fi
    if [[ "$arm_rc_0" -ne 0 || "$arm_rc_1" -ne 0 ]]; then
        echo "ERROR: campaign wave failed: lane0=$arm_rc_0 lane1=$arm_rc_1" >&2
        exit 1
    fi
done < <("$python_bin" "$coordinator" waves --manifest "$runtime_manifest")

"$python_bin" "$coordinator" aggregate \
    --manifest "$runtime_manifest" --require-complete
"$python_bin" "$coordinator" archive-small \
    --campaign-root "$campaign_root" --archive-root "$archive_root"
echo "campaign_status=COMPLETED"
echo "campaign_summary=$campaign_root/campaign_summary.json"
echo "persistent_small_artifact_archive=$archive_root"
