from typing import Optional

import torch


def get_model(config: dict) -> torch.nn.Module:
    mcfg = config["model"]
    name = mcfg["name"].lower()
    num_classes = int(mcfg.get("num_classes", 1000))

    import torchvision.models as models

    if not hasattr(models, name):
        raise ValueError(f"Unknown torchvision model: {mcfg['name']}")

    ctor = getattr(models, name)
    # Some models use num_classes, others use weights; handle common case
    try:
        model = ctor(num_classes=num_classes)
    except TypeError:
        model = ctor()
        # best-effort: replace final layer if attribute exists
        if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
            in_f = model.fc.in_features
            model.fc = torch.nn.Linear(in_f, num_classes)

    if mcfg.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)

    # torch.compile if available and enabled
    tcfg = config["training"].get("compile", {"enabled": False})
    if tcfg.get("enabled", False) and hasattr(torch, "compile"):
        mode = tcfg.get("mode", "default")
        try:
            model = torch.compile(model, mode=mode)
        except Exception:
            pass

    return model
