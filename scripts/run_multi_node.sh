#!/usr/bin/env bash
set -euo pipefail

# Template for multi-node runs
# Set environment appropriately before invoking.

# Required environment (example values):
#   export MASTER_ADDR=10.0.0.1
#   export MASTER_PORT=29500
#   export NODE_RANK=0            # 0..NNODES-1
#   export NNODES=2
#   export NPROC_PER_NODE=8

# Optional NCCL tips:
#   export NCCL_DEBUG=INFO
#   export NCCL_SOCKET_IFNAME=eth0   # or ib0 for InfiniBand
#   export NCCL_IB_DISABLE=0         # enable IB if available
#   export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Example hostfile usage (OpenMPI-style; for reference only):
#   export MASTER_ADDR=$(head -n1 hostfile)
#   export NODE_RANK=$(grep -n $(hostname) hostfile | cut -d: -f1 | awk '{print $1-1}')

CONFIG=${1:-configs/synthetic_resnet50.yaml}
shift || true

EXTRA_ARGS=("$@")

if [[ -z "${MASTER_ADDR:-}" || -z "${MASTER_PORT:-}" || -z "${NODE_RANK:-}" || -z "${NNODES:-}" || -z "${NPROC_PER_NODE:-}" ]]; then
  echo "Please set MASTER_ADDR, MASTER_PORT, NODE_RANK, NNODES, NPROC_PER_NODE" >&2
  exit 1
fi

torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  -m src.main --config "${CONFIG}" "${EXTRA_ARGS[@]}"

# Slurm snippet (commented):
# srun --ntasks=$((NNODES*NPROC_PER_NODE)) --gpus-per-task=1 --nodes=${NNODES} \
#   bash -c 'RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS \
#   MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
#   python -m src.main --config ${CONFIG} "$@"'
