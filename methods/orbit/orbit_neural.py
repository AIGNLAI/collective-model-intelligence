from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class OrbitMLP(torch.nn.Module):
    """ORBIT MLP with its optional matrix-factorization bottleneck."""

    def __init__(self, input_dim: int, output_dim: int, config: dict[str, Any]):
        super().__init__()
        model_config = dict(config.get("model", {}))
        hidden = model_config.get("hidden_sizes", config.get("hidden_layer_sizes", [256, 128]))
        hidden = [hidden] if isinstance(hidden, int) else list(hidden)
        activations = {
            "relu": torch.nn.ReLU,
            "leakyrelu": torch.nn.LeakyReLU,
            "leaky_relu": torch.nn.LeakyReLU,
            "leaky": torch.nn.LeakyReLU,
            "tanh": torch.nn.Tanh,
            "sigmoid": torch.nn.Sigmoid,
            "elu": torch.nn.ELU,
        }
        activation = activations.get(
            str(model_config.get("activation", "relu")).lower(), torch.nn.ReLU
        )
        dropout = float(model_config.get("dropout", config.get("dropout", 0.0)))
        batch_norm = bool(model_config.get("use_batchnorm", False))
        mf_config = dict(model_config.get("mf", {}))
        self.use_mf = bool(mf_config.get("use_mf", False))
        if self.use_mf:
            compressed_dim = max(1, input_dim // 4)
            self.mf_encoder = torch.nn.Linear(input_dim, compressed_dim)
            self.mf_decoder = torch.nn.Linear(compressed_dim, input_dim)
        else:
            self.mf_encoder = None
            self.mf_decoder = None

        layers: list[torch.nn.Module] = []
        width = input_dim
        for hidden_width in map(int, hidden):
            layers.append(torch.nn.Linear(width, hidden_width))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_width))
            layers.append(activation())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            width = hidden_width
        self.body = torch.nn.Sequential(*layers) if layers else torch.nn.Identity()
        self.output = torch.nn.Linear(width, output_dim)

        init_std = float(mf_config.get("init_std", 0.02))
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, 0.0, init_std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.use_mf:
            assert self.mf_encoder is not None and self.mf_decoder is not None
            features = self.mf_decoder(self.mf_encoder(features))
        return self.output(self.body(features))


class OrbitEmbedLLMNet(torch.nn.Module):
    """Query/model embedding scorer with three prediction heads."""

    def __init__(self, input_dim: int, num_models: int, embedding_dim: int):
        super().__init__()
        self.query_projection = torch.nn.Linear(input_dim, embedding_dim)
        self.model_embeddings = torch.nn.Embedding(num_models, embedding_dim)
        self.model_bias = torch.nn.Parameter(torch.zeros(num_models))
        self.cost_head = torch.nn.Linear(embedding_dim, 1)
        self.latency_head = torch.nn.Linear(embedding_dim, 1)
        torch.nn.init.normal_(self.model_embeddings.weight, 0.0, 0.02)

    def forward(self, features: torch.Tensor, noise: float = 0.0):
        queries = self.query_projection(features)
        models = self.model_embeddings.weight
        if noise > 0:
            queries = queries + noise * torch.randn_like(queries)
            models = models + noise * torch.randn_like(models)
        performance = queries @ models.T + self.model_bias.unsqueeze(0)
        cost = self.cost_head(models).squeeze(1).unsqueeze(0).expand(len(features), -1)
        latency = self.latency_head(models).squeeze(1).unsqueeze(0).expand(len(features), -1)
        return performance, cost, latency


class OrbitMIRTNet(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.theta = torch.nn.Linear(input_dim, latent_dim, bias=False)
        self.discrimination = torch.nn.Linear(input_dim, latent_dim, bias=False)
        self.difficulty = torch.nn.Linear(input_dim, 1, bias=False)

    def forward(self, model_descriptor: torch.Tensor, query: torch.Tensor):
        theta = self.theta(model_descriptor)
        discrimination = F.softplus(self.discrimination(query))
        return torch.sigmoid((theta * discrimination).sum(-1) - self.difficulty(query).squeeze(-1))


class OrbitRMScoreNet(torch.nn.Module):
    """Three-head RMSoftmax extension exposing calibrated components."""

    def __init__(self, input_dim: int, hidden: list[int], num_models: int):
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = input_dim
        for hidden_width in hidden:
            layers.extend([torch.nn.Linear(width, hidden_width), torch.nn.ReLU()])
            width = hidden_width
        layers.append(torch.nn.Linear(width, num_models * 3))
        self.network = torch.nn.Sequential(*layers)
        self.num_models = num_models

    def forward(self, features: torch.Tensor):
        return self.network(features).reshape(len(features), self.num_models, 3)


class OrbitGCNLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, nodes: torch.Tensor):
        transformed = self.linear(nodes)
        query, models = transformed[:, :1], transformed[:, 1:]
        return torch.cat(
            [query + self.bias, query / np.sqrt(2.0) + models / 2.0 + self.bias],
            dim=1,
        )


class OrbitEdgeGNN(torch.nn.Module):
    """ORBIT EdgeGNN with performance, cost, and latency edge outputs."""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        widths = [input_dim] + [hidden_dim] * max(1, layers)
        self.convolutions = torch.nn.ModuleList(
            OrbitGCNLayer(source, target) for source, target in itertools.pairwise(widths)
        )
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 3),
        )

    def forward(self, queries: torch.Tensor, model_features: torch.Tensor):
        models = model_features.unsqueeze(0).expand(len(queries), -1, -1)
        nodes = torch.cat([queries.unsqueeze(1), models], dim=1)
        for convolution in self.convolutions:
            nodes = F.relu(convolution(nodes))
        query_nodes = nodes[:, :1].expand(-1, len(model_features), -1)
        return self.edge_mlp(torch.cat([query_nodes, nodes[:, 1:]], dim=-1))
