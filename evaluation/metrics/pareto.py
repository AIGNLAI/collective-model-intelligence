from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from evaluation.metrics.schema import ParetoPoint


def pareto_frontier(
    points: Iterable[tuple[float, float]],
    cost_start: float,
    cost_end: float,
) -> tuple[ParetoPoint, ...]:
    """Return a bounded, monotone cost/performance Pareto frontier."""

    if not np.isfinite([cost_start, cost_end]).all() or cost_end <= cost_start:
        raise ValueError("cost_end must be finite and greater than cost_start")
    values = [(float(cost), float(performance)) for cost, performance in points]
    if not values:
        raise ValueError("at least one cost/performance point is required")
    if not np.isfinite(values).all():
        raise ValueError("Pareto points must contain only finite values")

    # At equal cost retain the strongest performance, then remove every point
    # that is weakly dominated by a cheaper point.
    by_cost: dict[float, float] = {}
    for cost, performance in values:
        by_cost[cost] = max(performance, by_cost.get(cost, -np.inf))
    ordered = sorted(by_cost.items())

    left_performance = max(
        (performance for cost, performance in ordered if cost <= cost_start),
        default=ordered[0][1],
    )
    frontier = [ParetoPoint(cost_start, left_performance)]
    best = left_performance
    for cost, performance in ordered:
        if cost <= cost_start or cost > cost_end or performance <= best:
            continue
        frontier.append(ParetoPoint(cost, performance))
        best = performance

    # User-specified right completion: continue horizontally when the router's
    # last realized cost does not reach the most-expensive-model endpoint.
    if frontier[-1].cost < cost_end:
        frontier.append(ParetoPoint(cost_end, frontier[-1].performance))
    return tuple(frontier)


def trapezoid_auc(frontier: tuple[ParetoPoint, ...]) -> tuple[float, float]:
    if len(frontier) < 2:
        raise ValueError("AUC requires at least two frontier points")
    costs = np.asarray([point.cost for point in frontier], dtype=np.float64)
    performance = np.asarray([point.performance for point in frontier], dtype=np.float64)
    # NumPy 2.x provides trapezoid; keep support for the declared NumPy 1.26 minimum.
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(integrate(performance, costs))
    width = float(costs[-1] - costs[0])
    if width <= 0:
        raise ValueError("AUC frontier must have a positive cost range")
    return auc, auc / width
