"""
State sensors: each turns one kind of agent message into named components for the estimator.
Which components get fused, and how much they are trusted, comes from the agent spec's noise.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from core_msgs.instance_agent.sensor_payloads import PoseMessage, GPSMessage
from core_msgs.topic_contract import MessageType
from instance_twin.sensors.base_sensors import StateSensor

logger = logging.getLogger(__name__)

_state_registry = {}


def register_state_sensor(name: str):
    """Decorator to register a StateSensor subclass"""

    normalized_name = name.lower()

    def decorator(cls):
        existing = _state_registry.get(normalized_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"State sensor '{normalized_name}' is already registered "
                f"for class {existing.__name__}"
            )
        _state_registry[normalized_name] = cls
        return cls

    return decorator


@register_state_sensor("pose")
class PoseSensor(StateSensor):
    """
    Pose sensor from either simulated agent or other agent detection trough aggregate
    Should always be attached to agent
    """
    components = frozenset({"x", "y", "theta", "vx" , "wz"})
    default_topic = MessageType.POSE

    def read(self, payload: PoseMessage) -> Dict[str, float | None]:
        return {
            "x": getattr(payload, "x", None),
            "y": getattr(payload, "y", None),
            "theta": getattr(payload, "theta", None),
            "vx": getattr(payload, "linear_velocity", None),
            "wz": getattr(payload, "angular_velocity", None),
        }

@register_state_sensor("gps")
class GpsSensor(StateSensor):
    """Gps sensor attached to agent, higher noise than pose.
    Its vx, vy are world frame while the estimator's are body frame, so only position is read."""
    components = frozenset({"x", "y"})
    default_topic = MessageType.GPS

    def read(self, payload: GPSMessage) -> Dict[str, float | None]:
        return {
            "x": getattr(payload, "x", None),
            "y": getattr(payload, "y", None),
        }


@register_state_sensor("odom")
class OdometrySensor(StateSensor):
    """Velocities reported with an odometry pose, fused as rates so odom drift never enters the estimate."""
    components = frozenset({"vx", "wz"})
    default_topic = MessageType.ODOM

    def read(self, payload: Any) -> Dict[str, float | None]:
        return {
            "vx": getattr(payload, "linear_velocity", None),
            "wz": getattr(payload, "angular_velocity", None),
        }


@register_state_sensor("wheel_odom")
class WheelOdometrySensor(StateSensor):
    """Left/right wheel speeds in m/s -> body velocities; the track width is this sensor's model."""
    components = frozenset({"vx", "wz"})
    default_topic = MessageType.WHEEL_ODOM

    def __init__(self, name: str, track_width: float, **kwargs: Any):
        self.track_width = float(track_width)
        super().__init__(name, **kwargs)

    def read(self, payload: Any) -> Dict[str, float | None] | None:
        left, right = getattr(payload, "left", None), getattr(payload, "right", None)
        if left is None or right is None:
            return None
        return {"vx": (left + right) / 2.0, "wz": (right - left) / self.track_width}


@register_state_sensor("imu")
class ImuSensor(StateSensor):
    """Gyro yaw rate, plus yaw when the IMU reports an orientation in the world frame."""
    components = frozenset({"wz", "theta"})
    default_topic = MessageType.IMU

    def read(self, payload: Any) -> Dict[str, float | None]:
        gyro = getattr(payload, "angular_velocity", None)
        return {
            "wz": gyro[2] if gyro is not None and len(gyro) > 2 else None,
            "theta": getattr(payload, "orientation_yaw", None),
        }


class StateSensorFactory:
    """
    Factory class to create StateSensors.
    """

    @staticmethod
    def create_sensor(name: str, **kwargs) -> StateSensor:
        name = name.lower()
        cls = _state_registry.get(name) if name else None
        if cls is None:
            raise ValueError(f"Unknown state sensor type: {name}")
        return cls(name, **kwargs)

    @staticmethod
    def get_handler_class(name: str) -> type[StateSensor] | None:
        """Look up a registered state sensor class by name without instantiation """
        return _state_registry.get(name.lower() if name else "")

    @staticmethod
    def available_sensors() -> dict[str, type[StateSensor]]:
        """Read-only view of the registry: ``{name: sensor_class}``."""
        return dict(_state_registry)
