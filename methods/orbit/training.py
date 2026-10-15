from __future__ import annotations

from typing import Any

from methods.base import RoutingTargets, SelectionPolicy, _as_float_matrix
from methods.orbit.orbit_estimators import OrbitKNNEstimator, ScoreEstimator
from methods.orbit.orbit_extended import (
    fit_global_local,
    fit_low_rank,
    fit_ridge,
    fit_svm,
)
from methods.orbit.orbit_router import OrbitRouter
from methods.orbit.orbit_train_classical import fit_avengers
from methods.orbit.orbit_train_heads import fit_embedllm, fit_mlp
from methods.orbit.orbit_train_rm import fit_rm_softmax
from methods.orbit.orbit_train_structured import fit_graphrouter, fit_mirt
from methods.orbit.orbit_utils import resolve_device, seed_everything

ORBIT_REPOSITORY = "https://github.com/LAMDA-Model-Reuse/ORBIT"
ORBIT_COMMIT = "494810de2605f69737e72b55baf6e60c95c6dec0"

SUPPORTED_ORBIT_METHODS = (
    "avengers",
    "avengerspro",
    "causal_rm_softmax",
    "eagle",
    "embedllm_routejudge",
    "graphrouter",
    "hybridllm",
    "knn",
    "mlp",
    "mirt",
    "routellm_mf",
    "svm_router",
)

# These labels describe integration with ORBIT, not exact paper reproduction.
# "native" denotes a port of the corresponding ORBIT model family; compatible
# adaptations can replace architectures, objectives, or unavailable input types.
METHOD_IMPLEMENTATION_LEVEL = {
    method: (
        "native"
        if method
        in {
            "avengerspro",
            "causal_rm_softmax",
            "embedllm_routejudge",
            "graphrouter",
            "knn",
            "mlp",
            "mirt",
        }
        else "embedding_compatible"
    )
    for method in SUPPORTED_ORBIT_METHODS
}


def _policy(config: dict[str, Any]) -> SelectionPolicy:
    values = dict(config.get("selection", {}))
    return SelectionPolicy(
        performance_weight=float(values.get("performance_weight", 1.0)),
        cost_weight=float(values.get("cost_weight", config.get("cost_weight", 0.0))),
        latency_weight=float(values.get("latency_weight", config.get("latency_weight", 0.0))),
        max_cost=float(values["max_cost"]) if values.get("max_cost") is not None else None,
        max_latency=(
            float(values["max_latency"]) if values.get("max_latency") is not None else None
        ),
    )


def _normalize(method: str) -> str:
    aliases = {
        "avengers_pro": "avengerspro",
        "rmsoftmax": "causal_rm_softmax",
        "rm_softmax": "causal_rm_softmax",
        "embedllm": "embedllm_routejudge",
        "routejudge": "embedllm_routejudge",
        "graph_router": "graphrouter",
        "mlp_router": "mlp",
        "svm": "svm_router",
        "route_llm_mf": "routellm_mf",
    }
    value = method.strip().lower().replace("-", "_")
    return aliases.get(value, value)


def train_orbit_router(
    method: str,
    query_embeddings: object,
    performance_scores: object,
    cost_scores: object,
    latency_scores: object,
    model_ids: list[str] | tuple[str, ...],
    config: dict[str, Any] | None = None,
) -> OrbitRouter:
    config = dict(config or {})
    method = _normalize(method)
    if method not in SUPPORTED_ORBIT_METHODS:
        raise ValueError(f"unsupported ORBIT method {method!r}")
    features = _as_float_matrix("query_embeddings", query_embeddings)
    targets = RoutingTargets(tuple(model_ids), performance_scores, cost_scores, latency_scores)
    if len(features) != targets.num_queries:
        raise ValueError("query_embeddings and scores must have the same row count")

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(config)
    estimator: ScoreEstimator

    if method in {"avengers", "avengerspro"}:
        estimator = fit_avengers(features, targets, config, seed)
    elif method == "knn":
        estimator = OrbitKNNEstimator(
            features,
            targets,
            min(int(config.get("k", config.get("n_neighbors", 300))), len(features)),
            float(config.get("eps", 1e-8)),
        )
    elif method == "eagle":
        estimator = fit_global_local(features, targets, config)
    elif method == "hybridllm":
        estimator = fit_ridge(features, targets, config)
    elif method == "svm_router":
        estimator = fit_svm(features, targets, config)
    elif method == "routellm_mf":
        estimator = fit_low_rank(features, targets, config)
    elif method == "causal_rm_softmax":
        estimator = fit_rm_softmax(features, targets, config, device)
    elif method == "embedllm_routejudge":
        estimator = fit_embedllm(features, targets, config, device)
    elif method == "graphrouter":
        estimator = fit_graphrouter(features, targets, config, device)
    elif method == "mirt":
        estimator = fit_mirt(features, targets, config, device)
    else:
        estimator = fit_mlp(features, targets, config, device)

    return OrbitRouter(method, estimator, targets.model_ids, features.shape[1], _policy(config))
