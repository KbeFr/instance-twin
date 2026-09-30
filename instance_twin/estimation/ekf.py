"""
Extended Kalman filter over [kinematics state; velocity].
Jacobians are taken numerically, so any registered kinematics or sensor works unchanged.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from instance_twin.estimation.estimator_handler import (
    ANGLE_COMPONENTS, Measurement, StateEstimator, register_estimator,
)
from instance_twin.irsim_borrowed.kinematics.kinematics_handler import KinematicsHandler

_EPS = 1e-6

# Absolute components map straight onto state rows; variance they get when forgotten
_POSE_ROWS = {"x": 0, "y": 1, "theta": 2}
_UNKNOWN_VAR = {"x": 10.0 ** 2, "y": 10.0 ** 2, "theta": math.pi ** 2}


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def _broadcast(value: float | list[float], size: int) -> np.ndarray:
    """Scalar or per-component list -> vector of `size`."""
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    return np.full(size, arr[0]) if arr.size == 1 else arr[:size]


@register_estimator("ekf")
class ExtendedKalmanFilter(StateEstimator):
    """Motion from the kinematics model; velocity follows the command with a first-order lag."""

    def __init__(self,
                 kinematics: KinematicsHandler,
                 response_time: float = 0.2,
                 state_noise: float | list[float] = 0.01,
                 velocity_noise: float | list[float] = 0.3,
                 initial_pose: list[float] | None = None,
                 initial_std: float | list[float] = 0.1,
                 gate: float | None = None,
                 gate_patience: int = 5,
                 max_step: float = 0.05,
                 **params,
    ):
        super().__init__(kinematics)
        self.n = kinematics.state_dim
        self.m = kinematics.action_dim
        self.response_time = float(response_time)
        self.max_step = float(max_step)
        self.gate = float(gate) if gate else None      # chi-square bound, e.g. 13.8 = 99.9 % for 2 dof
        self.gate_patience = int(gate_patience)
        self._angles = [i for i in kinematics.angle_indices if i < self.n]

        # A start pose is a guess until the first fix confirms it; misses count per sensor
        self._anchored = False
        self._misses: dict[str, int] = {}

        # Process noise std per sqrt(s): pose components, then velocity components
        self._q = np.concatenate([_broadcast(state_noise, self.n), _broadcast(velocity_noise, self.m)]) ** 2

        self.x = np.zeros((self.n + self.m, 1))
        p0 = np.ones(self.n + self.m)
        p0[:2] = 10.0 ** 2                             # position unknown until a fix arrives
        if self.n > 2:
            p0[2] = math.pi ** 2
        p0[self.n:] = 0.1 ** 2                         # robots start standing still

        if initial_pose is not None:
            k = min(len(initial_pose), self.n)
            self.x[:k, 0] = initial_pose[:k]
            p0[:k] = _broadcast(initial_std, k) ** 2
            self.initialized = True
        self.P = np.diag(p0)

    @property
    def state(self) -> np.ndarray:
        return self.x[:self.n].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.n:].copy()

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()

    # --- predict ---

    def predict(self, dt: float, command: np.ndarray | None) -> None:
        """Sub-stepped, so a long gap between messages does not integrate as one straight line."""
        u = None if command is None else np.asarray(command, dtype=float).reshape(self.m, 1)
        steps = max(1, math.ceil(dt / self.max_step))
        h = dt / steps
        for _ in range(steps):
            f = lambda x: self._transition(x, u, h)
            F = self._jacobian(f, self.x, self._angles)
            self.x = f(self.x)
            self.x[self._angles] = _wrap(self.x[self._angles])
            self.P = F @ self.P @ F.T + np.diag(self._q) * h

    def _transition(self, x: np.ndarray, u: np.ndarray | None, dt: float) -> np.ndarray:
        state, vel = x[:self.n], x[self.n:]
        nxt = self.kinematics.step(state, vel, dt)
        if u is not None:
            alpha = 1.0 if self.response_time <= 0 else 1.0 - math.exp(-dt / self.response_time)
            vel = vel + alpha * (u - vel)
        return np.vstack([nxt, vel])

    # --- update ---

    def update(self, measurement: Measurement) -> bool:
        keys = list(measurement.values)
        z = np.array([[measurement.values[k]] for k in keys], dtype=float)
        R = np.diag([measurement.noise[k] ** 2 for k in keys])
        angles = [i for i, k in enumerate(keys) if k in ANGLE_COMPONENTS]

        # Start pose or lost estimate: this fix takes over whatever it measures directly
        if measurement.absolute and not self._anchored:
            self._forget(keys)

        h = lambda x: self._expected(x, keys)
        H = self._jacobian(h, self.x, angles)
        y = z - h(self.x)
        y[angles] = _wrap(y[angles])

        S = H @ self.P @ H.T + R
        if not self._accept(measurement, y, S):
            return False

        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        self.x[self._angles] = _wrap(self.x[self._angles])

        # Joseph form keeps P symmetric positive definite under round-off
        I_KH = np.eye(self.P.shape[0]) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        if measurement.absolute:
            self.initialized = self._anchored = True
        return True

    def _accept(self, measurement: Measurement, y: np.ndarray, S: np.ndarray) -> bool:
        """Chi-square gate; after gate_patience misses in a row the estimate is the suspect, not the sensor."""
        source = measurement.source
        if self.gate is None or (y.T @ np.linalg.solve(S, y)).item() <= self.gate:
            self._misses[source] = 0
            return True

        self._misses[source] = self._misses.get(source, 0) + 1
        if self._misses[source] < self.gate_patience:
            return False

        self._misses[source] = 0
        if measurement.absolute:
            self._anchored = False      # next fix re-seeds, e.g. after the robot was carried away
            return False
        return True                     # a rate that keeps disagreeing: the velocity estimate is off

    def _forget(self, keys: list[str]) -> None:
        """Measured pose rows become unknown, uncoupled from the rest, which keeps P positive definite."""
        for key in keys:
            row = _POSE_ROWS.get(key)
            if row is None or row >= self.n:
                continue
            self.P[row, :] = 0.0
            self.P[:, row] = 0.0
            self.P[row, row] = _UNKNOWN_VAR[key]

    def _expected(self, x: np.ndarray, keys: list[str]) -> np.ndarray:
        seen = self.observe(x[:self.n], x[self.n:])
        return np.array([[seen[k]] for k in keys], dtype=float)

    @staticmethod
    def _jacobian(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, out_angles: list[int]) -> np.ndarray:
        """Central differences; angle outputs wrapped so +-pi does not blow up a column."""
        cols = []
        for i in range(x.shape[0]):
            d = np.zeros_like(x)
            d[i, 0] = _EPS
            diff = fn(x + d) - fn(x - d)
            diff[out_angles] = _wrap(diff[out_angles])
            cols.append(diff[:, 0] / (2 * _EPS))
        return np.stack(cols, axis=1)
