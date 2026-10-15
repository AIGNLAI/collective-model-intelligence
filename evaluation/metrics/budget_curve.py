from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from evaluation.metrics.pareto import pareto_frontier, trapezoid_auc
from evaluation.metrics.schema import (
    BudgetCurveResult,
    BudgetPoint,
    EvaluationOutcomes,
    ParetoPoint,
    TargetCostResult,
)
from methods.base import RoutingScores, ScoringRouter, SelectionPolicy


class BudgetCurveEvaluator:
    """Evaluate a scoring router without exposing ground truth to the router.

    A budget is applied to each query using the router's predicted cost scores.
    The selected models are then evaluated using the held-out actual outcomes.
    """

    def __init__(
        self,
        num_budget_points: int = 100,
        target_fractions: Sequence[float] = (0.9, 0.95, 1.0),
    ) -> None:
        if num_budget_points < 2:
            raise ValueError("num_budget_points must be at least 2")
        fractions = tuple(float(value) for value in target_fractions)
        if not fractions:
            raise ValueError("target_fractions must not be empty")
        if not np.isfinite(fractions).all() or any(value <= 0 or value > 1 for value in fractions):
            raise ValueError("target_fractions must be finite values in (0, 1]")
        self.num_budget_points = num_budget_points
        self.target_fractions = fractions

    def evaluate(
        self,
        router: ScoringRouter,
        query_embeddings: object,
        outcomes: EvaluationOutcomes,
    ) -> BudgetCurveResult:
        """Predict once, then evaluate the router over the full budget grid."""

        scores = router.predict_scores(query_embeddings)
        return self.evaluate_scores(scores, router.selection_policy, outcomes)

    def evaluate_scores(
        self,
        scores: RoutingScores,
        selection_policy: SelectionPolicy,
        outcomes: EvaluationOutcomes,
    ) -> BudgetCurveResult:
        """Evaluate already-computed deployment scores over the budget grid."""

        aligned = _align_outcomes(scores, outcomes)
        if scores.shape[0] != aligned.shape[0]:
            raise ValueError("predicted scores and actual outcomes must have the same row count")

        mean_performance = aligned.performance.mean(axis=0, dtype=np.float64)
        mean_cost = aligned.cost.mean(axis=0, dtype=np.float64)
        best_index = int(np.argmax(mean_performance))
        cheapest_index = int(np.argmin(mean_cost))
        expensive_index = int(np.argmax(mean_cost))
        cost_start = float(mean_cost[cheapest_index])
        cost_end = float(mean_cost[expensive_index])
        if cost_end <= cost_start:
            raise ValueError("AUC requires distinct cheapest and most-expensive model costs")

        budgets = np.linspace(
            cost_start,
            cost_end,
            num=self.num_budget_points,
            dtype=np.float64,
        )
        utilities = selection_policy.utility(scores)
        budget_points = tuple(
            _evaluate_budget(
                budget=float(budget),
                scores=scores,
                utilities=utilities,
                policy=selection_policy,
                outcomes=aligned,
            )
            for budget in budgets
        )

        # The cheapest fixed model defines the benchmark's left endpoint. It is
        # a reference point, not an extra budget sample.
        candidates = [
            (cost_start, float(mean_performance[cheapest_index])),
            *((point.cost, point.performance) for point in budget_points),
        ]
        frontier = pareto_frontier(candidates, cost_start, cost_end)
        auc, normalized_auc = trapezoid_auc(frontier)
        best_single_performance = float(mean_performance[best_index])

        targets = tuple(
            _target_cost(fraction, best_single_performance, frontier)
            for fraction in self.target_fractions
        )
        return BudgetCurveResult(
            peak_performance=max(point.performance for point in budget_points),
            best_single_model=scores.model_ids[best_index],
            best_single_performance=best_single_performance,
            oracle_performance=float(aligned.performance.max(axis=1).mean(dtype=np.float64)),
            cheapest_model=scores.model_ids[cheapest_index],
            cheapest_model_cost=cost_start,
            most_expensive_model=scores.model_ids[expensive_index],
            most_expensive_model_cost=cost_end,
            auc=auc,
            normalized_auc=normalized_auc,
            targets=targets,
            budget_points=budget_points,
            pareto_frontier=frontier,
        )


def _align_outcomes(
    scores: RoutingScores,
    outcomes: EvaluationOutcomes,
) -> EvaluationOutcomes:
    if set(scores.model_ids) != set(outcomes.model_ids):
        missing = sorted(set(scores.model_ids) - set(outcomes.model_ids))
        extra = sorted(set(outcomes.model_ids) - set(scores.model_ids))
        raise ValueError(
            "predicted and actual model IDs must match; "
            f"missing outcomes={missing}, extra outcomes={extra}"
        )
    positions = {model_id: index for index, model_id in enumerate(outcomes.model_ids)}
    order = [positions[model_id] for model_id in scores.model_ids]
    return EvaluationOutcomes(
        model_ids=scores.model_ids,
        performance=outcomes.performance[:, order],
        cost=outcomes.cost[:, order],
        latency=outcomes.latency[:, order],
    )


def _evaluate_budget(
    budget: float,
    scores: RoutingScores,
    utilities: NDArray[np.float32],
    policy: SelectionPolicy,
    outcomes: EvaluationOutcomes,
) -> BudgetPoint:
    effective_budget = budget
    if policy.max_cost is not None:
        effective_budget = min(effective_budget, policy.max_cost)

    latency_feasible = np.ones(scores.shape, dtype=bool)
    if policy.max_latency is not None:
        latency_feasible &= scores.latency_scores <= policy.max_latency
    feasible = latency_feasible & (scores.cost_scores <= effective_budget)

    # Predicted costs may be miscalibrated, so a row can have no feasible model.
    # Falling back to its cheapest predicted model keeps every query evaluable
    # and never consults held-out actual outcomes.
    for row in np.flatnonzero(~feasible.any(axis=1)):
        candidates = np.flatnonzero(latency_feasible[row])
        if candidates.size == 0:
            candidates = np.arange(scores.shape[1])
        local = int(np.argmin(scores.cost_scores[row, candidates]))
        feasible[row, candidates[local]] = True

    selected = np.argmax(np.where(feasible, utilities, -np.inf), axis=1)
    rows = np.arange(scores.shape[0])
    return BudgetPoint(
        budget=budget,
        cost=float(outcomes.cost[rows, selected].mean(dtype=np.float64)),
        performance=float(outcomes.performance[rows, selected].mean(dtype=np.float64)),
        latency=float(outcomes.latency[rows, selected].mean(dtype=np.float64)),
    )


def _target_cost(
    fraction: float,
    best_single_performance: float,
    frontier: tuple[ParetoPoint, ...],
) -> TargetCostResult:
    target = fraction * best_single_performance
    reached_at = next(
        (point.cost for point in frontier if point.performance >= target),
        None,
    )
    return TargetCostResult(
        fraction=fraction,
        target_performance=target,
        reached=reached_at is not None,
        minimum_cost=reached_at,
    )
