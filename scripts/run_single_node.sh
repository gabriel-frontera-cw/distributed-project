#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash scripts/run_single_node.sh 1 configs/synthetic_resnet50.yaml
#   bash scripts/run_single_node.sh 4 configs/cifar100_resnet50.yaml --training.precision=bf16

NGPUS=${1:-1}
CONFIG=${2:-configs/synthetic_resnet50.yaml}
shift || true
shift || true

EXTRA_ARGS=("$@")

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=${NGPUS} \
  -m src.main --config "${CONFIG}" "${EXTRA_ARGS[@]}"
