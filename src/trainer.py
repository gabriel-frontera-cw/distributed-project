import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD

from .data_loader import get_dataloader
from .metrics import MetricLogger
from .model_loader import get_model
from .profiling import maybe_profile
from .utils import Timer, capture_env_metadata, ensure_dir, is_rank_zero


PRECISION_MAP = {
    "fp32": (torch.float32, False),
    "amp": (torch.float16, True),
    "bf16": (torch.bfloat16, False),
}


class Trainer:
    def __init__(self, config: Dict, rank: int, local_rank: int, world_size: int) -> None:
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # Data
        self.dataloader, self.sampler = get_dataloader(config, world_size > 1)

        # Model
        self.model = get_model(config).to(self.device)
        if config["model"].get("channels_last", False) and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        if world_size > 1:
            self.model = DDP(self.model.to(self.device), device_ids=[local_rank] if self.device.type == "cuda" else None)

        # Optimizer
        tcfg = config["training"]
        if tcfg["optimizer"].lower() == "sgd":
            self.optimizer = SGD(
                self.model.parameters(),
                lr=float(tcfg["learning_rate"]),
                momentum=float(tcfg.get("momentum", 0.9)),
                weight_decay=float(tcfg.get("weight_decay", 0.0)),
            )
        else:
            self.optimizer = Adam(
                self.model.parameters(),
                lr=float(tcfg["learning_rate"]),
                weight_decay=float(tcfg.get("weight_decay", 0.0)),
            )

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Precision
        precision = tcfg.get("precision", "fp32").lower()
        if precision not in PRECISION_MAP:
            raise ValueError(f"Invalid precision: {precision}")
        self.autocast_dtype, use_grad_scaler = PRECISION_MAP[precision]
        self.use_amp = precision == "amp"
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=(self.use_amp and self.device.type == "cuda"))

        # Benchmarking
        self.bcfg = config["benchmark"]

        # Logging directories
        rcfg = config["run"]
        run_id = rcfg.get("run_id") or rcfg.get("runId") or None
        if not run_id:
            from .utils import default_run_id

            run_id = default_run_id()
            self.config["run"]["run_id"] = run_id
        self.out_dir = os.path.join(rcfg["save_dir"], rcfg["experiment_name"], run_id)
        if is_rank_zero():
            ensure_dir(self.out_dir)
        self.metric_logger = MetricLogger(self.out_dir, self.rank)

        # Profiler
        self.profile_enabled = bool(config.get("profiling", {}).get("enabled", False))

        # Gradient accumulation
        self.grad_accum_steps = int(config["distributed"].get("gradient_accumulation_steps", 1))

    def _step(self, batch, autocast_dtype, sync_cuda: bool, do_step: bool) -> Tuple[int, float]:
        inputs, targets = batch
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        if self.config["model"].get("channels_last", False) and self.device.type == "cuda":
            inputs = inputs.to(memory_format=torch.channels_last)

        batch_size = inputs.size(0)

        # Use CUDA events for accurate timing when available
        use_cuda_timing = self.device.type == "cuda"
        if use_cuda_timing:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
        else:
            timer = Timer()
            timer.__enter__()

        with torch.autocast(
            device_type=self.device.type,
            dtype=autocast_dtype,
            enabled=(self.use_amp or autocast_dtype in (torch.bfloat16,)),
        ):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets) / self.grad_accum_steps

        if self.use_amp:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if do_step:
            # Gradient clipping (supports AMP via unscale_)
            max_norm = self.config["training"].get("grad_clip_norm")
            if max_norm is not None:
                if self.use_amp:
                    self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_norm))

            if self.use_amp:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        if use_cuda_timing:
            end_evt.record()
            if sync_cuda:
                torch.cuda.synchronize()
            step_ms = start_evt.elapsed_time(end_evt)
        else:
            timer.__exit__(None, None, None)
            step_ms = timer.ms

        return batch_size, step_ms

        # unreachable

    def train(self) -> Dict:
        epochs = int(self.config["training"]["epochs"])
        sync_cuda = bool(self.bcfg.get("sync_cuda", False)) and self.device.type == "cuda"

        env_meta = capture_env_metadata() if self.config["run"].get("capture_env", True) and is_rank_zero() else {}

        overall_start = time.perf_counter()
        total_steps_all = 0
        total_samples_all = 0
        total_measured_time_agg = 0.0

        # Optional profiler (rank 0 only writes)
        trace_path = os.path.join(self.out_dir, "trace_rank0.json") if (self.profile_enabled and self.rank == 0) else None

        for epoch in range(1, epochs + 1):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

            epoch_start = time.perf_counter()
            measured_steps = 0
            measured_samples = 0
            total_steps = 0

            warmup_steps = int(self.bcfg.get("warmup_steps", 0))
            measure_by = self.bcfg.get("measure_by", "steps").lower()
            measure_steps = int(self.bcfg.get("measure_steps", 0))
            measure_seconds = float(self.bcfg.get("measure_seconds", 0.0) or 0.0)

            measuring = self.bcfg.get("enabled", True)
            measure_time_start: Optional[float] = None

            prof_mgr = maybe_profile(self.profile_enabled, trace_path=trace_path)
            with prof_mgr as prof:
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                for step, batch in enumerate(self.dataloader, start=1):
                    total_steps += 1

                    # Warmup section (not logged)
                    if measuring and warmup_steps > 0:
                        if self.use_amp:
                            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                                outputs = self.model(batch[0].to(self.device))
                                loss = self.criterion(outputs, batch[1].to(self.device)) / self.grad_accum_steps
                                self.grad_scaler.scale(loss).backward()
                        else:
                            outputs = self.model(batch[0].to(self.device))
                            loss = self.criterion(outputs, batch[1].to(self.device)) / self.grad_accum_steps
                            loss.backward()

                        if total_steps % self.grad_accum_steps == 0:
                            if self.use_amp:
                                self.grad_scaler.step(self.optimizer)
                                self.grad_scaler.update()
                            else:
                                self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)

                        warmup_steps -= 1
                        if prof is not None:
                            prof.step()
                        continue

                    # Start measurement window
                    if measuring and measure_by == "time" and measure_time_start is None:
                        measure_time_start = time.perf_counter()

                    do_step = (total_steps % self.grad_accum_steps == 0)
                    batch_size, step_ms = self._step(batch, self.autocast_dtype, sync_cuda, do_step)

                    total_steps_all += 1
                    total_samples_all += batch_size

                    if measuring and self.bcfg.get("log_step_times", True):
                        self.metric_logger.log_step(step_ms)

                    if measuring:
                        measured_steps += 1
                        measured_samples += batch_size

                    # Profiler tick
                    if prof is not None:
                        prof.step()

                    # Exit measurement window if criteria met
                    if measuring:
                        if measure_by == "steps" and measured_steps >= measure_steps and measure_steps > 0:
                            break
                        if measure_by == "time" and measure_time_start is not None:
                            if (time.perf_counter() - measure_time_start) >= measure_seconds > 0.0:
                                break

            epoch_time_s = time.perf_counter() - epoch_start
            measured_time_s = None
            if measuring:
                if measure_by == "steps":
                    # Approx measured time from step stats if synced; else use epoch time
                    stats = self.metric_logger.summarize_steps()
                    mean_ms = stats.get("mean_step_ms", 0.0)
                    measured_time_s = (mean_ms / 1000.0) * measured_steps if mean_ms > 0 else epoch_time_s
                else:
                    measured_time_s = min(measure_seconds, epoch_time_s) if measure_seconds else epoch_time_s
                # Aggregate across ranks for throughput comparability
                if self.world_size > 1 and dist.is_initialized():
                    tdev = self.device if self.device.type == "cuda" else torch.device("cpu")
                    msamp_t = torch.tensor([measured_samples], device=tdev, dtype=torch.long)
                    mtime_t = torch.tensor([measured_time_s], device=tdev, dtype=torch.float64)
                    msteps_t = torch.tensor([measured_steps], device=tdev, dtype=torch.long)
                    tsteps_t = torch.tensor([total_steps], device=tdev, dtype=torch.long)
                    etime_t = torch.tensor([epoch_time_s], device=tdev, dtype=torch.float64)
                    dist.all_reduce(msamp_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(mtime_t, op=dist.ReduceOp.MAX)
                    dist.all_reduce(msteps_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(tsteps_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(etime_t, op=dist.ReduceOp.MAX)
                    measured_samples_g = int(msamp_t.item())
                    measured_time_s_g = float(mtime_t.item())
                    measured_steps_g = int(msteps_t.item())
                    total_steps_g = int(tsteps_t.item())
                    epoch_time_s_g = float(etime_t.item())
                else:
                    measured_samples_g = measured_samples
                    measured_time_s_g = measured_time_s
                    measured_steps_g = measured_steps
                    total_steps_g = total_steps
                    epoch_time_s_g = epoch_time_s
                total_measured_time_agg += measured_time_s_g
            else:
                measured_time_s = epoch_time_s
                # No aggregation; single-process values
                measured_samples_g = measured_samples
                measured_time_s_g = measured_time_s
                measured_steps_g = measured_steps
                total_steps_g = total_steps
                epoch_time_s_g = epoch_time_s

            throughput = (measured_samples_g / measured_time_s_g) if measured_time_s_g and measured_time_s_g > 0 else 0.0

            # Log per-epoch metrics (rank 0 only writes in MetricLogger)
            self.metric_logger.log_epoch(
                epoch=epoch,
                throughput_samples_per_s=throughput,
                epoch_time_s=epoch_time_s_g,
                total_steps=total_steps_g,
                measured_steps=measured_steps_g,
                measured_samples=measured_samples_g,
            )

        wall_clock_s = time.perf_counter() - overall_start
        # Aggregate final totals across ranks for summary
        if self.world_size > 1 and dist.is_initialized():
            tdev = self.device if self.device.type == "cuda" else torch.device("cpu")
            tot_steps_t = torch.tensor([total_steps_all], device=tdev, dtype=torch.long)
            tot_samples_t = torch.tensor([total_samples_all], device=tdev, dtype=torch.long)
            wall_clock_t = torch.tensor([wall_clock_s], device=tdev, dtype=torch.float64)
            dist.all_reduce(tot_steps_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(tot_samples_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(wall_clock_t, op=dist.ReduceOp.MAX)
            total_steps_all_g = int(tot_steps_t.item())
            total_samples_all_g = int(tot_samples_t.item())
            wall_clock_s_g = float(wall_clock_t.item())
        else:
            total_steps_all_g = total_steps_all
            total_samples_all_g = total_samples_all
            wall_clock_s_g = wall_clock_s

        # Final summary (rank 0 writes)
        run_summary = {
            "experiment_name": self.config["run"]["experiment_name"],
            "run_id": self.config["run"]["run_id"],
            "world_size": self.world_size,
            "precision": self.config["training"]["precision"],
            "channels_last": self.config["model"].get("channels_last", False),
            "batch_size": self.config["data"]["batch_size"],
            "global_batch_size": int(self.config["data"]["batch_size"]) * int(self.world_size),
            "effective_batch_size": int(self.config["data"]["batch_size"]) * int(self.world_size) * int(self.grad_accum_steps),
            "total_steps": total_steps_all_g,
            "total_samples": total_samples_all_g,
            "total_measured_time_s": total_measured_time_agg,
            "total_wall_clock_s": wall_clock_s_g,
            "commit_hash": self.config["run"].get("commit_hash"),
            "env": env_meta,
            "notes": self.config["run"].get("notes", ""),
        }

        self.metric_logger.write_run_summary(run_summary)
        return run_summary
