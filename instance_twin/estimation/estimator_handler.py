"""
State estimation: one filter per agent, fed by whichever state sensors the agent declares.
Sensors hand over named components; the filter knows how to predict each one from its own state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time

import numpy as np

from instance_twin.irsim_borrowed.kinematics.kinematics_handler import KinematicsHandler

# Everything a state sensor may measure: pose in the world, twist in the body frame
COMPONENTS = frozenset({"x", "y", "theta", "vx", "vy", "wz"})
ANGLE_COMPONENTS = frozenset({"theta"})

_estimator_registry: dict[str, type["StateEstimator"]] = {}


def register_estimator(name: str):
    """Decorator to register a StateEstimator subclass."""

    normalized_name = name.lower()

    def decorator(cls):
        existing = _estimator_registry.get(normalized_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Estimator '{normalized_name}' is already registered "
                f"for class {existing.__name__}"
            )
        _estimator_registry[normalized_name] = cls
        return cls

    return decorator


@dataclass
class Measurement:
    """Filter-agnostic sensor reading: measured value and noise std per component."""
    values: dict[str, float]
    noise: dict[str, float]
    source: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def absolute(self) -> bool:
        """A position fix, which is what an estimate needs before it can be trusted."""
        return "x" in self.values and "y" in self.values


class StateEstimator(ABC):
    """State = kinematics state, velocity = kinematics action; both sized by the kinematics."""

    def __init__(self, kinematics: KinematicsHandler):
        self.kinematics = kinematics
        self.initialized = False
        self.t: float | None = None

    @property
    @abstractmethod
    def state(self) -> np.ndarray:
        """Estimated kinematics state, (state_dim, 1)."""

    @property
    @abstractmethod
    def velocity(self) -> np.ndarray:
        """Estimated kinematics action, (action_dim, 1)."""

    @abstractmethod
    def predict(self, dt: float, command: np.ndarray | None) -> None:
        """Propagates dt seconds under the held command."""

    @abstractmethod
    def update(self, measurement: Measurement) -> bool:
        """Corrects with one measurement, False when rejected."""

    def predict_to(self, t: float, command: np.ndarray | None) -> None:
        """Propagates up to time t; older times are left as they are."""
        if self.t is None:
            self.t = t
            return
        if t > self.t:
            self.predict(t - self.t, command)
            self.t = t

    def observe(self, state: np.ndarray, velocity: np.ndarray) -> dict[str, float]:
        """Every component in COMPONENTS, as a perfect sensor would read it."""
        linear, angular = self.kinematics.body_twist(state, velocity)
        return {
            "x": float(state[0, 0]),
            "y": float(state[1, 0]),
            "theta": float(state[2, 0]) if state.shape[0] > 2 else 0.0,
            "vx": linear[0],
            "vy": linear[1],
            "wz": angular[2],
        }


class EstimatorFactory:

    @staticmethod
    def create_estimator(name: str = "ekf", **kwargs) -> StateEstimator:
        name = name.lower()
        cls = _estimator_registry.get(name) if name else None
        if cls is None:
            raise ValueError(f"Unknown estimator type: {name}")
        return cls(**kwargs)

    @staticmethod
    def get_handler_class(name: str) -> type[StateEstimator] | None:
        """Look up a registered estimator class by name without instantiation."""
        return _estimator_registry.get(name.lower() if name else "")

    @staticmethod
    def available_estimators() -> dict[str, type[StateEstimator]]:
        """Read-only view of the registry: ``{name: estimator_class}``."""
        return dict(_estimator_registry)


# avoiding a circular-import error guarantees the registry is populated.
from instance_twin.estimation import ekf  # noqa: E402,F401
