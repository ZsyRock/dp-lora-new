#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="${DP_LORA_CONDA_ENV:-dp-lora}"
torch_index_url="${DP_LORA_TORCH_INDEX_URL:-}"
torch_version="${DP_LORA_TORCH_VERSION:-}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Load the cluster's Conda/Miniforge module first." >&2
    exit 1
fi

conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${env_name}"; then
    conda env update --name "${env_name}" --file "${repo_dir}/environment-hpc.yml"
else
    conda env create --name "${env_name}" --file "${repo_dir}/environment-hpc.yml"
fi
conda activate "${env_name}"

if ! python -c 'import torch' >/dev/null 2>&1; then
    if [[ -z "${torch_index_url}" ]]; then
        echo "PyTorch is not installed in ${env_name}." >&2
        echo "Ask the HPC administrator for the supported CUDA build, then rerun with:" >&2
        echo "  DP_LORA_TORCH_INDEX_URL=<official-pytorch-index> scripts/bootstrap_hpc.sh" >&2
        exit 2
    fi
    if [[ -n "${torch_version}" ]]; then
        python -m pip install "torch==${torch_version}" --index-url "${torch_index_url}"
    else
        python -m pip install torch --index-url "${torch_index_url}"
    fi
fi

python -m pip install --upgrade pip
python -m pip install \
    --constraint "${repo_dir}/requirements/constraints-tested.txt" \
    --editable "${repo_dir}[dev,examples]"
python -m pytest -q "${repo_dir}/tests"

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA runtime: {torch.version.cuda}")
PY
