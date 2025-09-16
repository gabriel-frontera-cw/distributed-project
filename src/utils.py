import json
import os
import socket
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist


def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_ddp(backend: Optional[str] = None) -> Tuple[int, int, int]:
    """Initialize torch.distributed from torchrun env vars if present.

    Returns (rank, local_rank, world_size). Sets CUDA device to local_rank if available.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("LOCAL-ID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    use_distributed = world_size > 1
    has_cuda = torch.cuda.is_available()

    if use_distributed and not dist.is_initialized():
        selected_backend = backend or ("nccl" if has_cuda else "gloo")
        os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))
        dist.init_process_group(backend=selected_backend, rank=rank, world_size=world_size)

    if has_cuda:
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def set_cudnn_benchmark(enabled: bool) -> None:
    try:
        torch.backends.cudnn.benchmark = enabled
    except Exception:
        pass


def now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def default_run_id() -> str:
    return f"run_{now_ts()}_{os.getpid()}"


def get_commit_hash() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def capture_env_metadata() -> Dict[str, Any]:
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )

    nccl_env = {k: os.environ[k] for k in os.environ if k.startswith("NCCL_")}

    return {
        "hostname": socket.gethostname(),
        "os": os.name,
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "pytorch_version": torch.__version__,
        "nccl_version": getattr(torch.cuda, "nccl", None),
        "gpus": gpu_info,
        "env": {"NCCL": nccl_env},
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply dot-delimited overrides into nested dict."""
    def set_in(d, keys, value):
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    out = dict(base)
    for k, v in overrides.items():
        parts = k.split(".")
        # try to coerce JSON-like values
        if isinstance(v, str):
            if v.lower() in {"true", "false"}:
                v = v.lower() == "true"
            elif v.lower() == "null":
                v = None
            else:
                try:
                    if v.isdigit():
                        v = int(v)
                    else:
                        v = float(v)
                except Exception:
                    pass
        set_in(out, parts, v)
    return out


def parse_unknown_overrides(argv) -> Dict[str, Any]:
    """Parse --a.b.c=value or --a.b.c value from unknown args."""
    overrides: Dict[str, Any] = {}
    skip = False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok.startswith("--"):
            key = tok[2:]
            value = None
            if "=" in key:
                key, value = key.split("=", 1)
            else:
                # lookahead for value
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    value = argv[i + 1]
                    skip = True
                else:
                    value = "true"
            overrides[key] = value
    return overrides


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.perf_counter()
        self.ms = (self.end - self.start) * 1000.0
        self.s = (self.end - self.start)
        return False
