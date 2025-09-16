GPU Training Benchmark Suite
===========================

Production-grade suite for benchmarking deep learning training throughput and step-time across single-GPU, multi-GPU single-node, and multi-node configurations using PyTorch.

Features
- Config-driven (YAML) with CLI overrides (`--key.subkey=value`).
- Native `torch.distributed` via `torchrun` (single/multi-node).
- Precision: fp32, AMP (fp16), bf16, channels_last.
- Benchmark methodology: warmup discard, fixed steps/time window.
- Metrics: per-epoch JSONL, final JSON + CSV summary (rank 0 only).
- Env capture: host, GPUs, CUDA/driver, PyTorch, NCCL env.
- Optional `torch.profiler` trace (rank 0).
- Real (CIFAR100) or synthetic data mode.
- Config validation via Pydantic.

Project Structure
- `configs/` example configs for CIFAR100 and synthetic.
- `src/` modular code: data_loader, model_loader, trainer, metrics, profiling, utils, main.
- `scripts/` helpers for single- and multi-node runs.
- `notebooks/analysis.ipynb` reporting and plots.
- `results/` outputs per run: JSONL, JSON, CSV, traces.

Quick Start
1) Install dependencies (ideally in a fresh virtualenv):
   - `pip install -r requirements.txt`
2) Single-node run with 1 GPU (synthetic):
   - `bash scripts/run_single_node.sh 1 configs/synthetic_resnet50.yaml`
3) Single-node scaling with 4 GPUs (CIFAR100 + bf16):
   - `bash scripts/run_single_node.sh 4 configs/cifar100_resnet50.yaml --training.precision=bf16`
4) Multi-node (template):
   - Fill env vars and run `bash scripts/run_multi_node.sh configs/synthetic_resnet50.yaml`.

Config
- See `configs/cifar100_resnet50.yaml` and `configs/synthetic_resnet50.yaml`.
- Override any nested key via CLI: `--training.epochs=5 --data.batch_size=512`.

Dataset Notes
- CIFAR100 is auto-downloaded to `data/` by default; images are resized to 224 for ResNet-50 throughput comparability.
- Synthetic mode isolates compute from I/O.

Precision & Memory Formats
- `training.precision`: `fp32`, `amp` (fp16 with `GradScaler`), or `bf16`.
- `model.channels_last`: `true` to use NHWC memory layout (CUDA only).

Benchmarking Methodology
- `benchmark.warmup_steps` discarded from stats.
- Measure by fixed steps (`benchmark.measure_steps`) or time (`benchmark.measure_seconds`).
- `benchmark.log_step_times=true` records per-step times to compute mean/p50/p95.
- Throughput (samples/sec) computed per epoch and summarized per run.
- `benchmark.sync_cuda=true` for strict timing (slower).

Distributed Training
- Uses `torchrun` environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`).
- Backend defaults to `nccl` if CUDA available, else `gloo`.
- Only rank 0 writes logs.

Profiler (optional)
- Enable via config section:
  ```yaml
  profiling:
    enabled: true
  ```
- Trace is saved for rank 0 at `results/<exp>/<run>/trace_rank0.json`.

Repro Checklist
- Pin package versions (requirements.txt).
- Record metadata (auto-captured in run summary).
- Fix seeds (`training.seed`).
- Enable `training.cudnn_benchmark=true` for throughput benchmarking.

Interpreting Results
- Per-epoch JSONL: `results/<exp>/<run>/metrics_epoch.jsonl`.
- Run summary JSON: `results/<exp>/<run>/run_summary.json` (also CSV `summary.csv`).
- Use `notebooks/analysis.ipynb` to plot throughput vs. GPU count, scaling efficiency, total time, and step-time distributions.

Troubleshooting
- NCCL hangs/timeouts: set `NCCL_DEBUG=INFO`, adjust `NCCL_SOCKET_IFNAME`, ensure firewall allows `MASTER_PORT`.
- No GPUs: backend falls back to `gloo`; throughput numbers will not be representative.
