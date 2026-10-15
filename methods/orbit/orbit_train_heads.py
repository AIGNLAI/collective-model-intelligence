from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from methods.base import RoutingTargets
from methods.orbit.orbit_neural import OrbitEmbedLLMNet, OrbitMLP
from methods.orbit.orbit_torch_estimator import OrbitTorchEstimator
from methods.orbit.orbit_utils import batches, make_optimizer, training_config


def fit_mlp(
    features: NDArray[np.float32],
    targets: RoutingTargets,
    config: dict[str, Any],
    device: torch.device,
) -> OrbitTorchEstimator:
    learning_rate, epochs, batch_size, optimizer_name, loss_name = training_config(config)
    if loss_name != "mse":
        raise ValueError(f"ORBIT MLP requires MSE, got {loss_name}")
    models = [OrbitMLP(features.shape[1], targets.num_models, config).to(device) for _ in range(3)]
    feature_tensor = torch.from_numpy(features).to(device)
    score_matrices = (
        targets.performance_scores,
        targets.cost_scores,
        targets.latency_scores,
    )
    for model, values in zip(models, score_matrices, strict=True):
        target = torch.from_numpy(values).to(device)
        optimizer = make_optimizer(optimizer_name, model.parameters(), learning_rate)
        for _ in range(epochs):
            for indices in batches(len(features), batch_size, device):
                loss = F.mse_loss(model(feature_tensor[indices]), target[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    return OrbitTorchEstimator("mlp", tuple(model.cpu().eval() for model in models))


def fit_embedllm(
    features: NDArray[np.float32],
    targets: RoutingTargets,
    config: dict[str, Any],
    device: torch.device,
) -> OrbitTorchEstimator:
    learning_rate, epochs, batch_size, optimizer_name, _ = training_config(config)
    model = OrbitEmbedLLMNet(
        features.shape[1],
        targets.num_models,
        int(config.get("embed_dim", 256)),
    ).to(device)
    optimizer = make_optimizer(optimizer_name, model.parameters(), learning_rate)
    feature_tensor = torch.from_numpy(features).to(device)
    performance = torch.from_numpy(targets.performance_scores).to(device)
    cost = torch.from_numpy(targets.cost_scores).to(device)
    latency = torch.from_numpy(targets.latency_scores).to(device)
    noise = float(config.get("alpha", 0.05))
    training = dict(config.get("training", {}))
    cost_loss_weight = float(training.get("cost_loss_weight", 1.0))
    latency_loss_weight = float(training.get("latency_loss_weight", 1.0))
    for _ in range(epochs):
        for indices in batches(len(features), batch_size, device):
            performance_logits, predicted_cost, predicted_latency = model(
                feature_tensor[indices], noise
            )
            loss = F.binary_cross_entropy_with_logits(performance_logits, performance[indices])
            loss += cost_loss_weight * F.mse_loss(predicted_cost, cost[indices])
            loss += latency_loss_weight * F.mse_loss(predicted_latency, latency[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return OrbitTorchEstimator("embedllm_routejudge", (model.cpu().eval(),))
