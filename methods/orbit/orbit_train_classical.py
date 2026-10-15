from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from methods.base import RoutingTargets
from methods.orbit.orbit_estimators import OrbitClusterEstimator


def capability_prototypes(
    features: NDArray[np.float32], performance: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Build model descriptors using training labels only."""

    descriptors = []
    fallback = features.mean(axis=0)
    for column in range(performance.shape[1]):
        weights = np.maximum(performance[:, column], 0).astype(np.float64)
        vector = np.average(features, axis=0, weights=weights) if weights.sum() else fallback
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        descriptors.append(vector.astype(np.float32))
    return np.stack(descriptors)


def fit_avengers(
    features: NDArray[np.float32],
    targets: RoutingTargets,
    config: dict[str, Any],
    seed: int,
) -> OrbitClusterEstimator:
    model_config = dict(config.get("model", {}))
    clusters = min(
        int(model_config.get("clusters", config.get("n_clusters", 200))),
        len(features),
    )
    kmeans = KMeans(
        n_clusters=clusters,
        n_init=model_config.get("n_init", "auto"),
        max_iter=int(model_config.get("max_iter", config.get("kmeans_iters", 1000))),
        algorithm=str(model_config.get("algorithm", "elkan")),
        random_state=seed,
    ).fit(features)
    cluster_scores = [np.zeros((clusters, targets.num_models), dtype=np.float32) for _ in range(3)]
    target_scores = (
        targets.performance_scores,
        targets.cost_scores,
        targets.latency_scores,
    )
    for cluster in range(clusters):
        mask = kmeans.labels_ == cluster
        for destination, source in zip(cluster_scores, target_scores, strict=True):
            destination[cluster] = source[mask].mean(0) if mask.any() else source.mean(0)
    return OrbitClusterEstimator(
        centers=kmeans.cluster_centers_.astype(np.float32),
        performance=cluster_scores[0],
        cost=cluster_scores[1],
        latency=cluster_scores[2],
        top_p=min(int(config.get("multi_cluster", 1)), clusters),
    )
