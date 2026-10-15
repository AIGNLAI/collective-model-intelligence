from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR

from methods.base import RoutingTargets
from methods.orbit.orbit_estimators import OrbitKNNEstimator, ScoreTriplet


@dataclass
class SklearnTripleEstimator:
    """Three independent sklearn multi-output predictors."""

    models: tuple[Any, Any, Any]

    def predict_components(self, query_embeddings: NDArray[np.float32]) -> ScoreTriplet:
        return tuple(
            np.asarray(model.predict(query_embeddings), dtype=np.float32) for model in self.models
        )  # type: ignore[return-value]


@dataclass
class LowRankTripleEstimator:
    """RouteLLM-MF-style low-rank query/model interaction predictor."""

    regressors: tuple[Ridge, Ridge, Ridge]
    components: tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
    means: tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]

    def predict_components(self, query_embeddings: NDArray[np.float32]) -> ScoreTriplet:
        outputs = []
        for regressor, component, mean in zip(
            self.regressors, self.components, self.means, strict=True
        ):
            latent = regressor.predict(query_embeddings)
            outputs.append(np.asarray(latent @ component + mean, dtype=np.float32))
        return tuple(outputs)  # type: ignore[return-value]


@dataclass
class GlobalLocalEstimator:
    """Eagle/SWRanking-compatible blend of global and local capability scores."""

    local: OrbitKNNEstimator
    global_scores: ScoreTriplet
    local_weight: float

    def predict_components(self, query_embeddings: NDArray[np.float32]) -> ScoreTriplet:
        local = self.local.predict_components(query_embeddings)
        return tuple(
            np.asarray(
                self.local_weight * local_score + (1.0 - self.local_weight) * global_score[None, :],
                dtype=np.float32,
            )
            for local_score, global_score in zip(local, self.global_scores, strict=True)
        )  # type: ignore[return-value]


def fit_ridge(
    features: NDArray[np.float32], targets: RoutingTargets, config: dict[str, Any]
) -> SklearnTripleEstimator:
    alpha = float(config.get("alpha", 1.0))
    matrices = (
        targets.performance_scores,
        targets.cost_scores,
        targets.latency_scores,
    )
    return SklearnTripleEstimator(
        tuple(Ridge(alpha=alpha).fit(features, values) for values in matrices)
    )


def fit_svm(
    features: NDArray[np.float32], targets: RoutingTargets, config: dict[str, Any]
) -> SklearnTripleEstimator:
    c_value = float(config.get("C", 1.0))
    max_iter = int(config.get("max_iter", 2000))
    matrices = (
        targets.performance_scores,
        targets.cost_scores,
        targets.latency_scores,
    )
    models = tuple(
        MultiOutputRegressor(
            LinearSVR(C=c_value, max_iter=max_iter, random_state=int(config.get("seed", 42)))
        ).fit(features, values)
        for values in matrices
    )
    return SklearnTripleEstimator(models)  # type: ignore[arg-type]


def fit_low_rank(
    features: NDArray[np.float32], targets: RoutingTargets, config: dict[str, Any]
) -> LowRankTripleEstimator:
    rank = max(1, min(int(config.get("rank", 8)), targets.num_models))
    regressors = []
    components = []
    means = []
    for values in (
        targets.performance_scores,
        targets.cost_scores,
        targets.latency_scores,
    ):
        mean = values.mean(axis=0)
        _, _, right = np.linalg.svd(values - mean, full_matrices=False)
        component = right[:rank]
        latent = (values - mean) @ component.T
        regressors.append(Ridge(alpha=float(config.get("alpha", 1.0))).fit(features, latent))
        components.append(component.astype(np.float32))
        means.append(mean.astype(np.float32))
    return LowRankTripleEstimator(
        tuple(regressors),
        tuple(components),
        tuple(means),  # type: ignore[arg-type]
    )


def fit_global_local(
    features: NDArray[np.float32], targets: RoutingTargets, config: dict[str, Any]
) -> GlobalLocalEstimator:
    local = OrbitKNNEstimator(
        features,
        targets,
        min(int(config.get("k", config.get("n_neighbors", 15))), len(features)),
        float(config.get("eps", 1e-8)),
    )
    global_scores = tuple(
        values.mean(axis=0).astype(np.float32)
        for values in (
            targets.performance_scores,
            targets.cost_scores,
            targets.latency_scores,
        )
    )
    return GlobalLocalEstimator(
        local,
        global_scores,  # type: ignore[arg-type]
        float(config.get("local_weight", 0.7)),
    )
