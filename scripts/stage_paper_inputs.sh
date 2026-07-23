#!/usr/bin/env bash
set -euo pipefail
umask 077

user_name="$(id -un)"
scratch_root="${DPLORA_PAPER_SCRATCH_ROOT:-${SCRATCH:-/scratch/$user_name}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
hf_home="${DPLORA_PAPER_HF_HOME:-$scratch_root/cache/dp-lora-paper/huggingface}"
data_root="${DPLORA_PAPER_DATA_ROOT:-$scratch_root/datasets/dp-lora-paper}"
manifest="${DPLORA_PAPER_INPUT_MANIFEST:-$data_root/input-manifest.json}"

if [[ -n "${DPLORA_PAPER_PYTHON:-}" ]]; then
    python_bin="$DPLORA_PAPER_PYTHON"
elif [[ -x "$scratch_root/envs/dp-lora-paper/bin/python" ]]; then
    python_bin="$scratch_root/envs/dp-lora-paper/bin/python"
elif [[ -x "$scratch_root/envs/dp-lora-aa0f0be/bin/python" ]]; then
    python_bin="$scratch_root/envs/dp-lora-aa0f0be/bin/python"
else
    python_bin="$(command -v python3)"
fi

if [[ ! -x "$python_bin" ]]; then
    echo "Staging interpreter is missing or not executable: $python_bin" >&2
    exit 1
fi

export HF_HOME="$hf_home"
export HF_HUB_DISABLE_TELEMETRY=1

exec "$python_bin" "$script_dir/stage_paper_inputs.py" \
    --hf-home "$hf_home" \
    --data-root "$data_root" \
    --manifest "$manifest" \
    "$@"
