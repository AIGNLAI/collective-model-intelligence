# Copyright 2026 SinapisAI
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for trainable model-selection methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]
IntVector = NDArray[np.int64]


def as_float_matrix(name: str, value: object) -> FloatMatrix:
    """Validate and normalize a finite two-dimensional float matrix."""

    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


# Private compatibility alias used by the ORBIT adapters.
_as_float_matrix = as_float_matrix


def _validate_model_ids(model_ids: tuple[str, ...], width: int) -> None:
    if len(model_ids) != width:
        raise ValueError(f"model_ids has length {len(model_ids)}, expected score width {width}")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must be unique")
    if any(not isinstance(model_id, str) or not model_id.strip() for model_id in model_ids):
        raise ValueError("model_ids must contain non-empty strings")


class SelectionMethod(Protocol):
    """Minimal contract for non-trainable or request-native selectors."""

    def select(self, request: Mapping[str, object], candidates: Sequence[str]) -> str:
        """Return one identifier from a non-empty candidate sequence."""
        ...


@dataclass(frozen=True)
class RoutingTargets:
    """Training labels with shape [num_queries, num_models]."""

    model_ids: tuple[str, ...]
    performance_scores: FloatMatrix
    cost_scores: FloatMatrix
    latency_scores: FloatMatrix

    def __post_init__(self) -> None:
        performance = as_float_matrix("performance_scores", self.performance_scores)
        cost = as_float_matrix("cost_scores", self.cost_scores)
        latency = as_float_matrix("latency_scores", self.latency_scores)
        if performance.shape != cost.shape or performance.shape != latency.shape:
            raise ValueError("all target score matrices must have the same shape")
        if np.any(cost < 0) or np.any(latency < 0):
            raise ValueError("cost and latency targets must be non-negative")
        _validate_model_ids(self.model_ids, performance.shape[1])
        object.__setattr__(self, "performance_scores", performance)
        object.__setattr__(self, "cost_scores", cost)
        object.__setattr__(self, "latency_scores", latency)

    @property
    def num_queries(self) -> int:
        return self.performance_scores.shape[0]

    @property
    def num_models(self) -> int:
        return self.performance_scores.shape[1]


@dataclass(frozen=True)
class RoutingScores:
    """Predicted per-model performance, cost, and latency matrices."""

    model_ids: tuple[str, ...]
    performance_scores: FloatMatrix
    cost_scores: FloatMatrix
    latency_scores: FloatMatrix

    def __post_init__(self) -> None:
        performance = as_float_matrix("performance_scores", self.performance_scores)
        cost = as_float_matrix("cost_scores", self.cost_scores)
        latency = as_float_matrix("latency_scores", self.latency_scores)
        if performance.shape != cost.shape or performance.shape != latency.shape:
            raise ValueError("all predicted score matrices must have the same shape")
        if np.any(cost < 0) or np.any(latency < 0):
            raise ValueError("predicted cost and latency must be non-negative")
        _validate_model_ids(self.model_ids, performance.shape[1])
        object.__setattr__(self, "performance_scores", performance)
        object.__setattr__(self, "cost_scores", cost)
        object.__setattr__(self, "latency_scores", latency)

    @property
    def shape(self) -> tuple[int, int]:
        return self.performance_scores.shape


@dataclass(frozen=True)
class SelectionPolicy:
    """Convert predicted score matrices into one model per query."""

    performance_weight: float = 1.0
    cost_weight: float = 0.0
    latency_weight: float = 0.0
    max_cost: float | None = None
    max_latency: float | None = None

    def __post_init__(self) -> None:
        values = [self.performance_weight, self.cost_weight, self.latency_weight]
        values.extend(limit for limit in (self.max_cost, self.max_latency) if limit is not None)
        if not np.isfinite(values).all():
            raise ValueError("policy weights and limits must be finite")
        if self.performance_weight < 0:
            raise ValueError("performance_weight must be non-negative")
        if self.cost_weight < 0 or self.latency_weight < 0:
            raise ValueError("cost_weight and latency_weight must be non-negative")
        if self.max_cost is not None and self.max_cost < 0:
            raise ValueError("max_cost must be non-negative")
        if self.max_latency is not None and self.max_latency < 0:
            raise ValueError("max_latency must be non-negative")

    def utility(self, scores: RoutingScores) -> FloatMatrix:
        return np.asarray(
            self.performance_weight * scores.performance_scores
            - self.cost_weight * scores.cost_scores
            - self.latency_weight * scores.latency_scores,
            dtype=np.float32,
        )

    def select(self, scores: RoutingScores) -> tuple[IntVector, FloatMatrix]:
        utilities = self.utility(scores)
        feasible = np.ones(scores.shape, dtype=bool)
        if self.max_cost is not None:
            feasible &= scores.cost_scores <= self.max_cost
        if self.max_latency is not None:
            feasible &= scores.latency_scores <= self.max_latency
        invalid = np.flatnonzero(~feasible.any(axis=1))
        if len(invalid):
            raise ValueError(
                f"no candidate model satisfies the policy constraints for rows {invalid.tolist()}"
            )
        masked = np.where(feasible, utilities, -np.inf)
        return np.argmax(masked, axis=1).astype(np.int64), utilities


@dataclass(frozen=True)
class DeploymentOutput:
    """Selected models plus the predictions used to make each decision."""

    selected_model_ids: tuple[str, ...]
    selected_indices: IntVector
    scores: RoutingScores
    utility_scores: FloatMatrix


class ScoringRouter(ABC):
    """Deployment contract for embedding-based scoring routers."""

    model_ids: tuple[str, ...]
    selection_policy: SelectionPolicy

    @abstractmethod
    def predict_scores(self, query_embeddings: object) -> RoutingScores:
        raise NotImplementedError

    def route_embeddings(self, query_embeddings: object) -> DeploymentOutput:
        scores = self.predict_scores(query_embeddings)
        indices, utilities = self.selection_policy.select(scores)
        selected = tuple(scores.model_ids[index] for index in indices)
        return DeploymentOutput(selected, indices, scores, utilities)

    def predict(self, query_embeddings: object) -> IntVector:
        return self.route_embeddings(query_embeddings).selected_indices
