"""Cost/performance evaluation for deployed scoring routers."""

from evaluation.metrics.budget_curve import BudgetCurveEvaluator
from evaluation.metrics.pareto import pareto_frontier, trapezoid_auc
from evaluation.metrics.report import ReportBudgetCurveEvaluator, nominal_budget_auc
from evaluation.metrics.schema import (
    BudgetCurveResult,
    BudgetPoint,
    EvaluationOutcomes,
    ParetoPoint,
    TargetCostResult,
)

__all__ = [
    "BudgetCurveEvaluator",
    "BudgetCurveResult",
    "BudgetPoint",
    "EvaluationOutcomes",
    "ParetoPoint",
    "ReportBudgetCurveEvaluator",
    "TargetCostResult",
    "nominal_budget_auc",
    "pareto_frontier",
    "trapezoid_auc",
]
