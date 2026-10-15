from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from methods.orbit.orbit_estimators import ScoreTriplet, _merge


@dataclass
class OrbitTorchEstimator:
    kind: str
    models: tuple[torch.nn.Module, ...]
    auxiliary: NDArray[np.float32] | None = None

    def predict_components(self, query_embeddings: NDArray[np.float32]) -> ScoreTriplet:
        for model in self.models:
            model.eval()
        outputs: list[list[np.ndarray]] = [[], [], []]
        with torch.no_grad():
            for start in range(0, len(query_embeddings), 512):
                query = torch.from_numpy(query_embeddings[start : start + 512])
                components = self._predict_batch(query)
                for output, component in zip(outputs, components, strict=True):
                    output.append(np.asarray(component, dtype=np.float32))
        return _merge(outputs)

    def _predict_batch(self, query: torch.Tensor) -> tuple[np.ndarray, ...]:
        if self.kind == "mlp":
            return tuple(model(query).numpy() for model in self.models)
        if self.kind == "embedllm_routejudge":
            performance_logits, cost, latency = self.models[0](query)
            return (
                torch.sigmoid(performance_logits).numpy(),
                cost.numpy(),
                latency.numpy(),
            )
        if self.kind == "mirt":
            assert self.auxiliary is not None
            descriptors = torch.from_numpy(self.auxiliary)
            return tuple(
                np.column_stack(
                    [
                        model(descriptor.unsqueeze(0).expand(len(query), -1), query).numpy()
                        for descriptor in descriptors
                    ]
                )
                for model in self.models
            )
        if self.kind == "graphrouter":
            assert self.auxiliary is not None
            prediction = self.models[0](query, torch.from_numpy(self.auxiliary)).numpy()
            return tuple(prediction[:, :, index] for index in range(3))
        if self.kind == "causal_rm_softmax":
            prediction = self.models[0](query).numpy()
            return tuple(prediction[:, :, index] for index in range(3))
        raise ValueError(f"unknown torch estimator kind: {self.kind}")
