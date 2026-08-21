#!/usr/bin/env bash

#SBATCH --job-name=dp-lora-default-baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Account, partition, typed GPU GRES, memory, walltime, signal, and logs are
# supplied by the submission command after live scheduler validation.
set -euo pipefail
umask 077

repository="${1:?missing immutable repository snapshot}"
expected_sha="${2:?missing expected repository SHA}"
env_prefix="${3:?missing environment prefix}"
scratch_root="${4:?missing scratch root}"
hf_home="${5:?missing Hugging Face cache root}"
input_index="${6:?missing baseline input index}"
campaign_root="${7:?missing campaign root}"
private_key="${8:?missing private RNG key}"
spec="${9:?missing baseline specification}"
expected_spec_file_sha="${10:?missing baseline spec file SHA-256}"
minimum_vram_gib="${11:?missing minimum GPU VRAM GiB}"

python_bin="$env_prefix/bin/python"
campaign="$repository/paper_repro/default_baseline_campaign.py"
runtime_lock="$repository/environment/paper-repro-runtime.lock"
step_cpus="${SLURM_CPUS_PER_TASK:-0}"

if [[ "${SLURM_JOB_NUM_NODES:-0}" != 1 || "${SLURM_NTASKS:-0}" != 1 ]]; then
    echo "ERROR: campaign requires one node and one sequential task" >&2
    exit 2
fi
if [[ ! "$step_cpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid CPU allocation: $step_cpus" >&2
    exit 2
fi
for path in "$python_bin" "$campaign" "$runtime_lock" "$input_index" "$private_key" "$spec"; do
    [[ -e "$path" ]] || { echo "ERROR: missing runtime input: $path" >&2; exit 2; }
done
if [[ "$(git -C "$repository" rev-parse HEAD)" != "$expected_sha" || -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: immutable repository snapshot identity failed" >&2
    exit 2
fi
if [[ "$(sha256sum "$spec" | awk '{print $1}')" != "$expected_spec_file_sha" ]]; then
    echo "ERROR: baseline specification changed after submission" >&2
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
export OMP_NUM_THREADS="$step_cpus"
export TMPDIR="${SLURM_TMPDIR:-$scratch_root/tmp/dp-lora-default-$SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$campaign_root"
chmod 700 "$TMPDIR" "$campaign_root"

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "repository_sha=$expected_sha"
echo "campaign_root=$campaign_root"
nvidia-smi

srun --exact --nodes=1 --ntasks=1 "--cpus-per-task=$step_cpus" \
    "$python_bin" - "$minimum_vram_gib" <<'PY'
import sys, torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
p = torch.cuda.get_device_properties(0)
minimum = float(sys.argv[1])
actual = p.total_memory / 1024**3
print({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
       "gpu": p.name, "vram_gib": actual})
if actual < minimum:
    raise SystemExit(f"GPU VRAM {actual:.1f} GiB is below {minimum:.1f} GiB")
PY
srun --exact --nodes=1 --ntasks=1 "--cpus-per-task=$step_cpus" \
    "$python_bin" -m pip check
srun --exact --nodes=1 --ntasks=1 "--cpus-per-task=$step_cpus" \
    "$python_bin" "$campaign" validate-runtime --spec "$spec" \
    --runtime-lock "$runtime_lock" --hf-home "$hf_home"
srun --exact --nodes=1 --ntasks=1 "--cpus-per-task=$step_cpus" \
    "$python_bin" "$campaign" validate-spec --spec "$spec"

# One real-model one-round smoke per model is part of this same allocation.
# A failure aborts before the 27 formal arms consume substantial GPU time.
srun --exact --nodes=1 --ntasks=1 "--cpus-per-task=$step_cpus" \
    "$python_bin" "$campaign" run-smokes --spec "$spec" \
    --input-index "$input_index" --campaign-root "$campaign_root" \
    --repository "$repository" --python-bin "$python_bin" \
    --private-key "$private_key" \
    --output "$campaign_root/smoke-summary.json"

srun --exact --nodes=1 --ntasks=1 "--cpus-per-task=$step_cpus" \
    "$python_bin" "$campaign" run-campaign --spec "$spec" \
    --input-index "$input_index" --campaign-root "$campaign_root" \
    --repository "$repository" --python-bin "$python_bin" \
    --private-key "$private_key" \
    --output "$campaign_root/campaign-summary.json"

"$python_bin" "$campaign" collect --spec "$spec" \
    --campaign-root "$campaign_root" --output-directory "$campaign_root/telemetry"

echo "campaign_status=COMPLETED"
echo "round_telemetry=$campaign_root/telemetry/baseline_round_telemetry.csv"
echo "client_telemetry=$campaign_root/telemetry/baseline_client_telemetry.csv"
echo "evaluation_telemetry=$campaign_root/telemetry/baseline_evaluation_telemetry.csv"
