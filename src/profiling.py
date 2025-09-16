from contextlib import contextmanager
from typing import Optional

import torch


@contextmanager
def maybe_profile(enabled: bool, on_trace_ready=None, activities=None, record_shapes=True, profile_memory=True, with_stack=True, schedule=None, trace_path: Optional[str] = None):
    if not enabled:
        yield None
        return

    if activities is None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

    if schedule is None:
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        yield prof
        if trace_path is not None:
            prof.export_chrome_trace(trace_path)
