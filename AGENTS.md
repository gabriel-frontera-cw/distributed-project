# AGENTS.md — GPU Training Benchmark Suite

Scope: This file applies to the entire repository.

Purpose: Guidance for agents contributing changes. Priorities are repeatability, comparability, and ease of extension.

Structure
- Keep the modular layout under `src/`:
  - `data_loader.py`, `model_loader.py`, `trainer.py`, `metrics.py`, `profiling.py`, `utils.py`, `main.py`.
- Do not rename these modules or move entrypoints without explicit instruction.
- Configs live in `configs/`; scripts in `scripts/`; results in `results/` (gitignored except for `.gitkeep`).

Distributed & IO
- Use native `torch.distributed` and `torchrun` env (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`).
- Initialize DDP via `utils.setup_ddp()`; only rank 0 should write files (`utils.is_rank_zero()`).
- All per-run outputs go under `results/<experiment>/<run_id>/`.

Configuration
- Configs are YAML and validated via Pydantic in `src/main.py`.
- Support CLI overrides using dot-keys (e.g., `--training.epochs=5`). Use `utils.parse_unknown_overrides()` and `utils.apply_overrides()`.
- Avoid adding new top-level config sections unless necessary; extend existing sections when possible.

Precision & Performance
- Support `fp32`, `amp` (fp16 + `GradScaler`), and `bf16`. Respect `channels_last`.
- For benchmarking, prefer throughput-first settings and minimize extraneous work in the hot path.

Logging & Metrics
- Use `metrics.MetricLogger` for per-epoch JSONL and final JSON/CSV summary. Do not duplicate logging logic in other modules.
- Include environment metadata using `utils.capture_env_metadata()` in the run summary (rank 0 only).

Profiling
- Gate `torch.profiler` via config and write traces only for rank 0.

Dependencies & Style
- Keep dependencies minimal (see `requirements.txt`).
- Match existing code style; prioritize clarity and small, focused changes.
- Avoid adding global side effects; prefer explicit function parameters.

Testing & Validation
- If adding features that affect training flow or metrics, add lightweight validation or doc examples rather than heavy tests.

Notes
- Do not commit large binary artifacts or dataset files. Results are gitignored by default.
