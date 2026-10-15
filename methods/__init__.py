"""Public model-selection interfaces and method registry."""

from typing import Any

from methods.base import (
    DeploymentOutput,
    RoutingScores,
    RoutingTargets,
    ScoringRouter,
    SelectionMethod,
    SelectionPolicy,
)

_ORBIT_EXPORTS = {
    "METHOD_IMPLEMENTATION_LEVEL",
    "ORBIT_COMMIT",
    "ORBIT_REPOSITORY",
    "SUPPORTED_ORBIT_METHODS",
    "OrbitRouter",
    "train_orbit_router",
}


def __getattr__(name: str) -> Any:
    """Load optional ORBIT dependencies only when their exports are requested."""

    if name in _ORBIT_EXPORTS:
        from methods import orbit

        return getattr(orbit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DeploymentOutput",
    "METHOD_IMPLEMENTATION_LEVEL",
    "ORBIT_COMMIT",
    "ORBIT_REPOSITORY",
    "SUPPORTED_ORBIT_METHODS",
    "OrbitRouter",
    "RoutingScores",
    "RoutingTargets",
    "ScoringRouter",
    "SelectionMethod",
    "SelectionPolicy",
    "train_orbit_router",
]
