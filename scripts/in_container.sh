#!/usr/bin/env bash
# Run a command inside the shared Singularity image on the assigned physical GPU.
# The image is shared and read-only; every project package lives in this repo's
# envs/* venvs, which are created INSIDE the container so they use its python.
#
#   scripts/in_container.sh envs/infer/bin/python -c "import torch; print(torch.__version__)"
#
# GPU policy (CLAUDE_EXECUTION_ORDER §1.2): physical 6 only. Inside the container
# torch therefore sees exactly one device at logical cuda:0. Physical != logical.
set -euo pipefail
SIF="${MUNN_SIF:-/dataset/singularity_images/pytorch271-cu128-devel.sif}"
GPU="${MUNN_GPU:-6}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$SIF" ] || { echo "singularity image not found: $SIF" >&2; exit 1; }
case "$GPU" in *,*) echo "MUNN_GPU must name exactly one physical GPU" >&2; exit 1;; esac
[ $# -gt 0 ] || { echo "usage: $0 <command> [args...]" >&2; exit 1; }
exec env \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU" \
  SINGULARITYENV_CUDA_DEVICE_ORDER=PCI_BUS_ID \
  SINGULARITYENV_PYTHONNOUSERSITE=1 \
  SINGULARITYENV_PYTHONPATH="" \
  SINGULARITYENV_HF_HOME="$ROOT/hf_cache" \
  SINGULARITYENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
  SINGULARITYENV_MUNN_PHYSICAL_GPU="$GPU" \
  singularity exec --nv --bind /dataset:/dataset --pwd "$ROOT" "$SIF" "$@"
