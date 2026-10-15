from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from methods.base import (
    RoutingScores,
    ScoringRouter,
    SelectionPolicy,
    _as_float_matrix,
)
from methods.orbit.orbit_estimators import ScoreEstimator


@dataclass
class OrbitRouter(ScoringRouter):
    """Trained ORBIT method ready for score prediction and model selection."""

    method: str
    estimator: ScoreEstimator
    model_ids: tuple[str, ...]
    embedding_dim: int
    selection_policy: SelectionPolicy

    def predict_scores(self, query_embeddings: object) -> RoutingScores:
        features = _as_float_matrix("query_embeddings", query_embeddings)
        if features.shape[1] != self.embedding_dim:
            raise ValueError(
                f"query embedding width is {features.shape[1]}, expected {self.embedding_dim}"
            )
        performance, cost, latency = self.estimator.predict_components(features)
        return RoutingScores(
            self.model_ids,
            performance,
            np.maximum(cost, 0.0),
            np.maximum(latency, 0.0),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> OrbitRouter:
        """Load a trusted local artifact created by :meth:`save`."""

        router = joblib.load(Path(path))
        if not isinstance(router, cls):
            raise TypeError(f"artifact does not contain {cls.__name__}")
        return router
