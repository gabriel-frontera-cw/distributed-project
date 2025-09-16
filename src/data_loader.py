from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class SyntheticImageDataset(Dataset):
    def __init__(self, num_classes: int = 1000, size: int = 224, length: int = 10000):
        self.num_classes = num_classes
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(3, self.size, self.size)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


def get_cifar100_dataset(root: str, train: bool = True):
    import torchvision
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return torchvision.datasets.CIFAR100(root=root, train=train, transform=transform, download=True)


def get_dataloader(config: dict, is_distributed: bool) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dcfg = config["data"]
    name = dcfg["name"].lower()

    if name == "cifar100":
        dataset = get_cifar100_dataset(dcfg["path"], train=True)
        num_classes = 100
    elif name == "synthetic":
        num_classes = int(config["model"].get("num_classes", 1000))
        dataset = SyntheticImageDataset(num_classes=num_classes, size=224)
    else:
        raise ValueError(f"Unsupported dataset: {dcfg['name']}")

    sampler = DistributedSampler(dataset) if is_distributed else None

    num_workers = int(dcfg.get("num_workers", 8))
    pin_memory = bool(dcfg.get("pin_memory", True))

    dl_kwargs = dict(
        batch_size=dcfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # Guard params invalid when num_workers == 0
    if num_workers > 0:
        dl_kwargs.update(
            prefetch_factor=dcfg.get("prefetch_factor", 2),
            persistent_workers=bool(dcfg.get("persistent_workers", True)),
        )

    dl = DataLoader(
        dataset,
        **dl_kwargs,
    )

    return dl, sampler
