"""Report curve-first multi-run estimates and exact run-bootstrap uncertainty."""

from __future__ import annotations

from itertools import product

import numpy as np

from evaluation.metrics.report import nominal_budget_auc


def _curve_metrics(curve: np.ndarray, strongest: float) -> dict:
    # Columns are nominal budget, realized mean cost, quality, stored latency.
    peak = int(np.argmax(curve[:, 2]))
    result = {
        "peak_performance": float(curve[peak, 2]),
        "peak_cost": float(curve[peak, 1]),
        "normalized_auc": nominal_budget_auc(curve[:, 0], curve[:, 2])[1],
    }
    for fraction in (90, 95, 100):
        feasible = curve[:, 2] >= fraction / 100 * strongest
        result[f"target_{fraction}"] = float(curve[feasible, 1].min()) if feasible.any() else None
    return result


def aggregate_report_runs(cards: list[dict]) -> dict:
    """Average operating points first, then calculate every curve-level metric.

    For the report's three fitted runs, enumerate all 3**3 ordered bootstrap
    resamples. Their population standard deviation is the exact bootstrap SE.
    Test requests, fixed-model references, and nominal grids must be shared.
    """
    if len(cards) not in (1, 3):
        raise ValueError("The report protocol requires one or three fitted runs")
    if any(card["query_count"] != cards[0]["query_count"] for card in cards):
        raise ValueError("Fitted runs must use identical evaluation populations")
    fields = ("budget", "cost", "performance", "latency")
    curves = np.asarray(
        [[[point[key] for key in fields] for point in card["budget_points"]] for card in cards],
        dtype=np.float64,
    )
    if curves.ndim != 3 or curves.shape[1] < 2 or not np.isfinite(curves).all():
        raise ValueError("Fitted runs must provide aligned finite budget curves")
    if np.any(curves[:, :, 2] < 0) or np.any(curves[:, :, 2] > 1):
        raise ValueError("Every fitted run's quality must lie in [0, 1]")
    if np.any(curves[:, :, [0, 1, 3]] < 0):
        raise ValueError("Budget, realized cost, and latency must be non-negative")
    if not np.allclose(curves[:, :, 0], curves[0, :, 0], atol=1e-14, rtol=0):
        raise ValueError("Fitted runs do not share the same nominal-budget grid")
    for key in ("best_single_performance", "oracle_performance"):
        values = [card[key] for card in cards]
        if not np.allclose(values, values[0], atol=1e-12, rtol=0):
            raise ValueError("Fitted runs do not share their fixed-model reference population")
    strongest = float(cards[0]["best_single_performance"])
    mean = curves.mean(axis=0)
    estimate = _curve_metrics(mean, strongest)
    bootstrap = [
        _curve_metrics(curves[list(sample)].mean(axis=0), strongest)
        for sample in product(range(len(cards)), repeat=len(cards))
    ]
    result = {"query_count": cards[0]["query_count"], "runs": len(cards)}
    for key in ("peak_performance", "normalized_auc", "peak_cost"):
        result[key] = {
            "mean": estimate[key],
            "std": float(np.std([b[key] for b in bootstrap], ddof=0)),
        }
    for key in ("best_single_performance", "oracle_performance"):
        result[key] = {"mean": float(cards[0][key]), "std": 0.0}
    result["targets"] = {}
    for fraction in (90, 95, 100):
        key = f"target_{fraction}"
        finite = [sample[key] for sample in bootstrap if sample[key] is not None]
        mean_cost = estimate[key]
        reached = sum(
            _curve_metrics(curves[index], strongest)[key] is not None for index in range(len(cards))
        )
        result["targets"][str(fraction)] = {
            "reached_runs": reached,
            "total_runs": len(cards),
            "minimum_cost": {
                "mean": mean_cost,
                "std": float(np.std(finite, ddof=0)),
            }
            if mean_cost is not None
            else None,
        }
    result["budget_points"] = [dict(zip(fields, row.tolist(), strict=True)) for row in mean]
    return result
