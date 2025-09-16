import csv
import json
import os
import statistics
from typing import Any, Dict, List, Optional

from .utils import ensure_dir, safe_write_json


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


class MetricLogger:
    def __init__(self, out_dir: str, rank: int, write_csv: bool = True) -> None:
        self.out_dir = out_dir
        ensure_dir(self.out_dir)
        self.rank = rank
        self.step_times_ms: List[float] = []
        self.epoch_jsonl_path = os.path.join(self.out_dir, "metrics_epoch.jsonl")
        self.summary_json_path = os.path.join(self.out_dir, "run_summary.json")
        self.csv_path = os.path.join(self.out_dir, "summary.csv") if write_csv else None
        if self.csv_path and self.rank == 0 and not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "throughput_samples_per_s",
                        "epoch_time_s",
                        "mean_step_ms",
                        "p50_step_ms",
                        "p95_step_ms",
                        "total_steps",
                        "measured_steps",
                        "measured_samples",
                    ],
                )
                writer.writeheader()

    def log_step(self, step_time_ms: float) -> None:
        self.step_times_ms.append(step_time_ms)

    def summarize_steps(self) -> Dict[str, float]:
        st = self.step_times_ms
        mean_ms = statistics.fmean(st) if st else 0.0
        p50_ms = percentile(st, 50)
        p95_ms = percentile(st, 95)
        return {
            "mean_step_ms": mean_ms,
            "p50_step_ms": p50_ms,
            "p95_step_ms": p95_ms,
        }

    def log_epoch(
        self,
        epoch: int,
        throughput_samples_per_s: float,
        epoch_time_s: float,
        total_steps: int,
        measured_steps: int,
        measured_samples: int,
    ) -> Dict[str, Any]:
        stats = self.summarize_steps()
        record = {
            "epoch": epoch,
            "throughput_samples_per_s": throughput_samples_per_s,
            "epoch_time_s": epoch_time_s,
            "total_steps": total_steps,
            "measured_steps": measured_steps,
            "measured_samples": measured_samples,
            **stats,
        }
        if self.rank == 0:
            with open(self.epoch_jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            if self.csv_path:
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                    writer.writerow(record)
        # reset step stats for next epoch
        self.step_times_ms.clear()
        return record

    def write_run_summary(self, summary: Dict[str, Any]) -> None:
        if self.rank == 0:
            safe_write_json(self.summary_json_path, summary)
