from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]


def _matrix(name: str, value: object) -> FloatMatrix:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


@dataclass(frozen=True)
class EvaluationOutcomes:
    """Held-out outcome matrices, never passed to the deployed router."""

    model_ids: tuple[str, ...]
    performance: FloatMatrix
    cost: FloatMatrix
    latency: FloatMatrix

    def __post_init__(self) -> None:
        performance = _matrix("performance", self.performance)
        cost = _matrix("cost", self.cost)
        latency = _matrix("latency", self.latency)
        if performance.shape != cost.shape or performance.shape != latency.shape:
            raise ValueError("all outcome matrices must have the same shape")
        if len(self.model_ids) != performance.shape[1]:
            raise ValueError("model_ids length must equal the outcome matrix width")
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("model_ids must be unique")
        if (cost < 0).any() or (latency < 0).any():
            raise ValueError("actual cost and latency must be non-negative")
        object.__setattr__(self, "performance", performance)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "latency", latency)

    @property
    def shape(self) -> tuple[int, int]:
        return self.performance.shape


@dataclass(frozen=True)
class BudgetPoint:
    budget: float
    cost: float
    performance: float
    latency: float


@dataclass(frozen=True)
class ParetoPoint:
    cost: float
    performance: float


@dataclass(frozen=True)
class TargetCostResult:
    fraction: float
    target_performance: float
    reached: bool
    minimum_cost: float | None


@dataclass(frozen=True)
class BudgetCurveResult:
    peak_performance: float
    best_single_model: str
    best_single_performance: float
    oracle_performance: float
    cheapest_model: str
    cheapest_model_cost: float
    most_expensive_model: str
    most_expensive_model_cost: float
    auc: float
    normalized_auc: float
    targets: tuple[TargetCostResult, ...]
    budget_points: tuple[BudgetPoint, ...]
    pareto_frontier: tuple[ParetoPoint, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
