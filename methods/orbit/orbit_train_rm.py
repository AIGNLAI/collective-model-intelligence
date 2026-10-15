from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from methods.base import RoutingTargets
from methods.orbit.orbit_neural import OrbitRMScoreNet
from methods.orbit.orbit_torch_estimator import OrbitTorchEstimator
from methods.orbit.orbit_utils import batches, make_optimizer, training_config


def fit_rm_softmax(
    features: NDArray[np.float32],
    targets: RoutingTargets,
    config: dict[str, Any],
    device: torch.device,
) -> OrbitTorchEstimator:
    """Fit regret minimization plus calibration for the three score heads.

    Original ORBIT RMSoftmax returns only selection logits. The calibration term
    is the minimal extension required to expose performance/cost/latency scores.
    """

    learning_rate, epochs, batch_size, optimizer_name, _ = training_config(config)
    hidden = list(map(int, config.get("router_hidden", [512, 256])))
    model = OrbitRMScoreNet(features.shape[1], hidden, targets.num_models).to(device)
    optimizer = make_optimizer(optimizer_name, model.parameters(), learning_rate)
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

    count = max(1, int(config.get("num_lambdas", config.get("lambda_count", 30))))
    cost_lambdas = torch.linspace(
        float(config.get("cost_lambda_min", config.get("lambda_min", 0.0))),
        float(config.get("cost_lambda_max", config.get("lambda_max", 1.0))),
        count,
        device=device,
    )
    latency_lambdas = torch.linspace(
        float(config.get("latency_lambda_min", 0.0)),
        float(config.get("latency_lambda_max", 1.0)),
        count,
        device=device,
    ).roll(1)
    training = dict(config.get("training", {}))
    temperature = float(training.get("temperature", 1.0))
    score_loss_weight = float(training.get("score_loss_weight", 1.0))

    for _ in range(epochs):
        for flat_indices in batches(len(features) * count, batch_size, device):
            lambda_indices = torch.div(flat_indices, len(features), rounding_mode="floor")
            sample_indices = flat_indices.remainder(len(features))
            cost_weight = cost_lambdas[lambda_indices]
            latency_weight = latency_lambdas[lambda_indices]
            true_scores = target[sample_indices]
            predicted_scores = model(feature_tensor[sample_indices])
            true_utility = (
                true_scores[:, :, 0]
                - cost_weight[:, None] * true_scores[:, :, 1]
                - latency_weight[:, None] * true_scores[:, :, 2]
            )
            predicted_utility = (
                predicted_scores[:, :, 0]
                - cost_weight[:, None] * predicted_scores[:, :, 1]
                - latency_weight[:, None] * predicted_scores[:, :, 2]
            )
            regret = (
                (
                    F.softmax(predicted_utility / temperature, dim=1)
                    * (true_utility.max(dim=1, keepdim=True).values - true_utility)
                )
                .sum(dim=1)
                .mean()
            )
            calibration = F.mse_loss(predicted_scores, true_scores)
            loss = regret + score_loss_weight * calibration
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return OrbitTorchEstimator("causal_rm_softmax", (model.cpu().eval(),))
