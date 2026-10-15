from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


def resolve_device(config: dict[str, Any]) -> torch.device:
    name = str(config.get("device", "auto"))
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device {name!r} is unavailable; use device='cpu' explicitly")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_config(config: dict[str, Any]) -> tuple[float, int, int, str, str]:
    training = dict(config.get("training", {}))
    learning_rate = float(
        training.get(
            "lr",
            config.get("learning_rate", config.get("learning_rate_init", 1e-3)),
        )
    )
    epochs = max(1, int(training.get("epochs", config.get("epochs", 10))))
    batch_size = max(1, int(training.get("batch_size", config.get("batch_size", 256))))
    optimizer = str(training.get("optimizer", "adam")).lower()
    loss = str(training.get("loss", "mse")).lower()
    return learning_rate, epochs, batch_size, optimizer, loss


def make_optimizer(
    name: str,
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float,
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    raise ValueError(f"unsupported ORBIT optimizer: {name}")


def batches(count: int, batch_size: int, device: torch.device) -> Iterable[torch.Tensor]:
    yield from torch.randperm(count, device=device).split(batch_size)
