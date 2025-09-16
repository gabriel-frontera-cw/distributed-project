import argparse
import os
import sys
from typing import Any, Dict

import torch
import yaml
from pydantic import BaseModel, Field, ValidationError, validator

from .trainer import Trainer
from .utils import (
    apply_overrides,
    get_commit_hash,
    parse_unknown_overrides,
    set_cudnn_benchmark,
    set_seed,
    setup_ddp,
)


class CompileCfg(BaseModel):
    enabled: bool = False
    mode: str = Field("default", regex=r"^(default|reduce-overhead|max-autotune)$")


class DataCfg(BaseModel):
    name: str
    path: str | None = None
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True


class ModelCfg(BaseModel):
    name: str
    num_classes: int = 1000
    channels_last: bool = False


class TrainingCfg(BaseModel):
    epochs: int = 1
    precision: str = Field("fp32", regex=r"^(fp32|amp|bf16)$")
    optimizer: str = Field("Adam", regex=r"^(Adam|SGD|adam|sgd)$")
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    momentum: float | None = None
    grad_clip_norm: float | None = None
    compile: CompileCfg = CompileCfg()
    seed: int = 1337
    cudnn_benchmark: bool = False


class BenchmarkCfg(BaseModel):
    enabled: bool = True
    warmup_steps: int = 0
    measure_by: str = Field("steps", regex=r"^(steps|time)$")
    measure_steps: int = 0
    measure_seconds: float | None = None
    log_step_times: bool = True
    sync_cuda: bool = False
    evaluate_after: bool = False


class DistributedCfg(BaseModel):
    backend: str = "nccl"
    gradient_accumulation_steps: int = 1
    world_size: int | None = None
    local_rank: int | None = None
    node_rank: int | None = None


class RunCfg(BaseModel):
    experiment_name: str
    run_id: str | None = None
    save_dir: str = "./results"
    capture_env: bool = True
    commit_hash: str | None = "auto"
    notes: str | None = None


class Config(BaseModel):
    data: DataCfg
    model: ModelCfg
    training: TrainingCfg
    benchmark: BenchmarkCfg
    distributed: DistributedCfg
    run: RunCfg
    profiling: dict | None = None

    @validator("run")
    def validate_commit_hash(cls, v: RunCfg):
        if v.commit_hash == "auto":
            v.commit_hash = get_commit_hash()
        return v


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="GPU Training Benchmark Suite")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args, unknown = parser.parse_known_args(argv)

    cfg_dict = load_config(args.config)
    overrides = parse_unknown_overrides(unknown)
    if overrides:
        cfg_dict = apply_overrides(cfg_dict, overrides)

    try:
        cfg = Config(**cfg_dict)
    except ValidationError as e:
        print("Config validation error:", e, file=sys.stderr)
        sys.exit(2)

    # Initialize DDP
    rank, local_rank, world_size = setup_ddp(cfg.distributed.backend)

    # Seeds and perf knobs
    set_seed(cfg.training.seed)
    set_cudnn_benchmark(cfg.training.cudnn_benchmark)

    if rank == 0:
        print("Validated config:")
        print(cfg.json(indent=2))

    trainer = Trainer(cfg.dict(), rank, local_rank, world_size)
    summary = trainer.train()

    # Teardown
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    if rank == 0:
        print("Run summary written to:")
        print(os.path.join(trainer.out_dir, "run_summary.json"))


if __name__ == "__main__":
    main()
