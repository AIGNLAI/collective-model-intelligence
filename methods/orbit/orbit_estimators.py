from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from methods.base import RoutingTargets

ScoreTriplet = tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]


class ScoreEstimator(Protocol):
    def predict_components(self, query_embeddings: NDArray[np.float32]) -> ScoreTriplet: ...


def _merge(outputs: list[list[np.ndarray]]) -> ScoreTriplet:
    merged = tuple(np.concatenate(output).astype(np.float32) for output in outputs)
    return merged  # type: ignore[return-value]


@dataclass
class OrbitClusterEstimator:
    centers: NDArray[np.float32]
    performance: NDArray[np.float32]
    cost: NDArray[np.float32]
    latency: NDArray[np.float32]
    top_p: int

    def predict_components(self, query_embeddings: NDArray[np.float32]) -> ScoreTriplet:
        centers = torch.from_numpy(self.centers)
        outputs: list[list[np.ndarray]] = [[], [], []]
        values = (self.performance, self.cost, self.latency)
        for start in range(0, len(query_embeddings), 1024):
            query = torch.from_numpy(query_embeddings[start : start + 1024])
            nearest = torch.topk(
                torch.cdist(query, centers), self.top_p, largest=False, dim=1
            ).indices.numpy()
            for output, component in zip(outputs, values, strict=True):
                output.append(component[nearest].mean(axis=1))
        return _merge(outputs)


@dataclass
class OrbitKNNEstimator:
    training_embeddings: NDArray[np.float32]
    targets: RoutingTargets
    neighbors: int
    epsilon: float

    def __post_init__(self) -> None:
        if not 1 <= self.neighbors <= len(self.training_embeddings):
            raise ValueError("kNN neighbor count must be between 1 and the training row count")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("kNN epsilon must be positive and finite")

    def predict_components(self, query_embeddings: NDArray[np.float32]) -> ScoreTriplet:
        training = torch.from_numpy(self.training_embeddings)
        target_tensors = tuple(
            torch.from_numpy(values)
            for values in (
                self.targets.performance_scores,
                self.targets.cost_scores,
                self.targets.latency_scores,
            )
        )
        outputs: list[list[np.ndarray]] = [[], [], []]
        for start in range(0, len(query_embeddings), 256):
            query = torch.from_numpy(query_embeddings[start : start + 256])
            distances, indices = torch.topk(
                torch.cdist(query, training),
                self.neighbors,
                largest=False,
                sorted=True,
            )
            zeros = distances <= 1e-12
            exact = zeros.any(dim=1, keepdim=True).expand_as(zeros)
            weights = torch.where(exact, zeros.float(), 1.0 / (distances + self.epsilon))
            weights /= weights.sum(dim=1, keepdim=True) + 1e-12
            for output, target in zip(outputs, target_tensors, strict=True):
                prediction = (weights.unsqueeze(-1) * target[indices]).sum(dim=1)
                output.append(prediction.numpy())
        return _merge(outputs)
