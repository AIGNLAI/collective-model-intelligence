"""The report's nominal-budget metric, separate from legacy Pareto AUC."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from evaluation.metrics.budget_curve import BudgetCurveEvaluator
from evaluation.metrics.pareto import pareto_frontier
from evaluation.metrics.schema import ParetoPoint, TargetCostResult


def nominal_budget_auc(budgets: object, qualities: object) -> tuple[float, float]:
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(qualities, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or y.shape != x.shape:
        raise ValueError("At least two aligned budget and quality values are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Budgets and qualities must be finite")
    if np.any(np.diff(x) <= 0) or np.any(y < 0) or np.any(y > 1):
        raise ValueError("Budgets must increase strictly and qualities must lie in [0, 1]")
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = float(integrate(np.maximum.accumulate(y), x))
    normalized = area / float(x[-1] - x[0])
    if normalized > float(y.max()) + 1e-12:
        raise AssertionError("Nominal-budget nAUC cannot exceed selector peak quality")
    return area, normalized


class ReportBudgetCurveEvaluator(BudgetCurveEvaluator):
    """Report §6.3: nominal-budget cumulative envelope; no fixed-model points.

    Selection and the predicted-cheapest fallback are unchanged. Actual model
    outcomes construct a common budget grid on the reporting population, and
    enter scoring only. Target costs use actual selector operating points.
    """

    def evaluate_scores(self, scores, selection_policy, outcomes):
        legacy = super().evaluate_scores(scores, selection_policy, outcomes)
        points = legacy.budget_points
        area, normalized = nominal_budget_auc(
            [point.budget for point in points],
            [point.performance for point in points],
        )
        targets = []
        for fraction in self.target_fractions:
            threshold = fraction * legacy.best_single_performance
            cost = min((p.cost for p in points if p.performance >= threshold), default=None)
            targets.append(TargetCostResult(fraction, threshold, cost is not None, cost))
        left, right = min(p.cost for p in points), max(p.cost for p in points)
        frontier = (
            pareto_frontier(((p.cost, p.performance) for p in points), left, right)
            if right > left
            else (ParetoPoint(left, legacy.peak_performance),)
        )
        return replace(
            legacy,
            auc=area,
            normalized_auc=normalized,
            targets=tuple(targets),
            pareto_frontier=frontier,
        )
