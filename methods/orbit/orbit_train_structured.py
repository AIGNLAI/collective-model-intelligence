from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from methods.base import RoutingTargets
from methods.orbit.orbit_neural import OrbitEdgeGNN, OrbitMIRTNet
from methods.orbit.orbit_torch_estimator import OrbitTorchEstimator
from methods.orbit.orbit_train_classical import capability_prototypes
from methods.orbit.orbit_utils import batches, make_optimizer, training_config


def fit_mirt(
    features: NDArray[np.float32],
    targets: RoutingTargets,
    config: dict[str, Any],
    device: torch.device,
) -> OrbitTorchEstimator:
    learning_rate, epochs, batch_size, optimizer_name, loss_name = training_config(config)
    if loss_name not in {"mse", "bce"}:
        raise ValueError(f"ORBIT MIRT requires MSE or BCE, got {loss_name}")
    latent_dim = int(dict(config.get("model", {})).get("latent_dim", config.get("latent_dim", 25)))
    descriptors = capability_prototypes(features, targets.performance_scores)
    descriptor_tensor = torch.from_numpy(descriptors).to(device)
    feature_tensor = torch.from_numpy(features).to(device)
    models = [OrbitMIRTNet(features.shape[1], latent_dim).to(device) for _ in range(3)]
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
                losses = []
                for model_index in range(targets.num_models):
                    descriptor = (
                        descriptor_tensor[model_index].unsqueeze(0).expand(len(indices), -1)
                    )
                    prediction = model(descriptor, feature_tensor[indices])
                    selected_target = target[indices, model_index]
                    loss_function = F.binary_cross_entropy if loss_name == "bce" else F.mse_loss
                    losses.append(loss_function(prediction, selected_target))
                loss = torch.stack(losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    return OrbitTorchEstimator("mirt", tuple(model.cpu().eval() for model in models), descriptors)


def fit_graphrouter(
    features: NDArray[np.float32],
    targets: RoutingTargets,
    config: dict[str, Any],
    device: torch.device,
) -> OrbitTorchEstimator:
    learning_rate, epochs, batch_size, optimizer_name, loss_name = training_config(config)
    model_config = dict(config.get("model", {}))
    model = OrbitEdgeGNN(
        features.shape[1],
        int(model_config.get("hidden_dim", 256)),
        int(model_config.get("num_layers", 2)),
    ).to(device)
    descriptors = capability_prototypes(features, targets.performance_scores)
    descriptor_tensor = torch.from_numpy(descriptors).to(device)
    feature_tensor = torch.from_numpy(features).to(device)
    target = torch.from_numpy(
        np.stack(
            [
                targets.performance_scores,
                targets.cost_scores,
                targets.latency_scores,
            ],
            axis=2,
        )
    ).to(device)
    optimizer = make_optimizer(optimizer_name, model.parameters(), learning_rate)
    for _ in range(epochs):
        for indices in batches(len(features), batch_size, device):
            prediction = model(feature_tensor[indices], descriptor_tensor)
            if loss_name == "mse":
                loss = F.mse_loss(prediction, target[indices])
            elif loss_name == "bce":
                loss = F.binary_cross_entropy_with_logits(prediction, target[indices])
            else:
                raise ValueError(f"unsupported ORBIT GraphRouter loss: {loss_name}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return OrbitTorchEstimator("graphrouter", (model.cpu().eval(),), descriptors)
