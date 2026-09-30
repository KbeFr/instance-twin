"""
ROS message <-> core_msgs converters, registered per ROS type.
A new robot on known ROS types needs no new converter, only sensors on topics the mapping covers.
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

from core_msgs.instance_agent.controll_payloads import MotionCommand
from core_msgs.instance_agent.sensor_payloads import (
    ArucoDetection, BatteryMessage, DetectedObjectSim2D, DetectionMessage, ImuMessage, Pose, PoseMessage,
)
from core_msgs.utils.math import Quaternion, Vector3, quaternion_to_yaw

TypeLookup = Callable[[str], type]

_decoder_registry: dict[str, type["RosDecoder"]] = {}
_encoder_registry: dict[str, type["RosEncoder"]] = {}


def _register(registry: dict, ros_type: str):
    def decorator(cls):
        existing = registry.get(ros_type)
        if existing is not None and existing is not cls:
            raise ValueError(f"'{ros_type}' is already registered for class {existing.__name__}")
        registry[ros_type] = cls
        return cls
    return decorator


def register_ros_decoder(ros_type: str):
    """Decorator to register a robot -> twin converter for a ROS type."""
    return _register(_decoder_registry, ros_type)


def register_ros_encoder(ros_type: str):
    """Decorator to register a twin -> robot converter for a ROS type."""
    return _register(_encoder_registry, ros_type)


class RosDecoder(ABC):
    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def decode(self, ros_msg: Any) -> Any | None:
        """ROS message -> core_msgs message, None to drop."""


class RosEncoder(ABC):
    accepts: ClassVar[type] = MotionCommand

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def encode(self, payload: Any, types: TypeLookup) -> Any:
        """Canonical message of type `accepts` -> ROS message."""


# --- helpers ---

def _vec3(types: TypeLookup, v: list[float]) -> Any:
    x, y, z = (list(v) + [0.0, 0.0, 0.0])[:3]
    return types("geometry_msgs/msg/Vector3")(x=float(x), y=float(y), z=float(z))


def _twist(cmd: MotionCommand, types: TypeLookup) -> Any:
    return types("geometry_msgs/msg/Twist")(linear=_vec3(types, cmd.linear), angular=_vec3(types, cmd.angular))


# --- robot -> twin ---

@register_ros_decoder("nav_msgs/msg/Odometry")
class OdometryDecoder(RosDecoder):
    """Odometry -> planar pose, shifted from the odom frame into the world by `origin: [x, y, theta]`."""

    def __init__(self, origin: list[float] | None = None, **params: Any):
        super().__init__(**params)
        self.ox, self.oy, self.otheta = (origin or [0.0, 0.0, 0.0])

    def decode(self, ros_msg: Any) -> PoseMessage:
        p = ros_msg.pose.pose
        c, s = math.cos(self.otheta), math.sin(self.otheta)
        quat = p.orientation
        yaw = quaternion_to_yaw(quat.x, quat.y, quat.z, quat.w) + self.otheta
        return PoseMessage(
            x=self.ox + c * p.position.x - s * p.position.y,
            y=self.oy + s * p.position.x + c * p.position.y,
            theta=math.atan2(math.sin(yaw), math.cos(yaw)),
            linear_velocity=ros_msg.twist.twist.linear.x,
            angular_velocity=ros_msg.twist.twist.angular.z,
            frame_id="world",
        )


@register_ros_decoder("sensor_msgs/msg/Imu")
class ImuDecoder(RosDecoder):
    """Orientation is dropped when the driver flags it unknown (covariance[0] == -1)."""

    def decode(self, ros_msg: Any) -> ImuMessage:
        a, w, q = ros_msg.linear_acceleration, ros_msg.angular_velocity, ros_msg.orientation
        has_orientation = ros_msg.orientation_covariance[0] != -1.0
        return ImuMessage(
            linear_acceleration=[a.x, a.y, a.z],
            angular_velocity=[w.x, w.y, w.z],
            orientation_yaw=quaternion_to_yaw(q.x, q.y, q.z, q.w) if has_orientation else None,
            orientation_quaternion=[q.x, q.y, q.z, q.w] if has_orientation else None,
        )


@register_ros_decoder("ros2_aruco_interfaces/msg/ArucoMarkers")
class ArucoMarkersDecoder(RosDecoder):
    """Marker poses in the camera's optical frame, as ros2_aruco publishes them."""

    def __init__(self, sensor_type: str = "aruco", **params: Any):
        super().__init__(**params)
        self.sensor_type = sensor_type

    def decode(self, ros_msg: Any) -> ArucoDetection:
        poses = [
            Pose(position=Vector3(x=p.position.x, y=p.position.y, z=p.position.z),
                 orientation=Quaternion(x=p.orientation.x, y=p.orientation.y,
                                        z=p.orientation.z, w=p.orientation.w))
            for p in ros_msg.poses
        ]
        return ArucoDetection(marker_ids=[int(i) for i in ros_msg.marker_ids], poses=poses)


@register_ros_decoder("sensor_msgs/msg/BatteryState")
class BatteryStateDecoder(RosDecoder):
    _CHARGING = 1  # POWER_SUPPLY_STATUS_CHARGING

    def decode(self, ros_msg: Any) -> BatteryMessage | None:
        if math.isnan(ros_msg.percentage):
            return None
        return BatteryMessage(
            percentage=100.0 * float(ros_msg.percentage),
            voltage=float(ros_msg.voltage),
            charging=ros_msg.power_supply_status == self._CHARGING,
        )


@register_ros_decoder("sensor_msgs/msg/LaserScan")
class LaserScanDecoder(RosDecoder):
    """Clusters a scan into blobs, so the twin gets a handful of detections, not 1000 ranges."""

    def __init__(self, sensor_type: str = "lidar",
                 max_range: float = 3.0,
                 cluster_gap: float = 0.15,
                 min_points: int = 3,
                 **params: Any):

        super().__init__(**params)
        self.sensor_type = sensor_type
        self.max_range = float(max_range)
        self.cluster_gap = float(cluster_gap)
        self.min_points = int(min_points)

    def decode(self, ros_msg: Any) -> DetectionMessage:
        limit = min(self.max_range, float(ros_msg.range_max))
        clusters: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        prev: tuple[float, float] | None = None

        for i, r in enumerate(ros_msg.ranges):
            r = float(r)
            if not (ros_msg.range_min < r < limit):
                prev = None
                continue
            a = ros_msg.angle_min + i * ros_msg.angle_increment
            point = (r * math.cos(a), r * math.sin(a))
            if prev is None or math.dist(point, prev) > self.cluster_gap:
                if current:
                    clusters.append(current)
                current = []
            current.append(point)
            prev = point
        if current:
            clusters.append(current)

        detections = [self._to_detection(c) for c in clusters if len(c) >= self.min_points]
        return DetectionMessage(sensor_type=self.sensor_type, payload=detections)

    @staticmethod
    def _to_detection(points: list[tuple[float, float]]) -> DetectedObjectSim2D:
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        return DetectedObjectSim2D(
            distance=math.hypot(cx, cy),
            bearing=math.atan2(cy, cx),
            radius=max(math.dist((cx, cy), p) for p in points),
        )


# --- twin -> robot ---

@register_ros_encoder("geometry_msgs/msg/Twist")
class TwistEncoder(RosEncoder):
    def encode(self, payload: MotionCommand, types: TypeLookup) -> Any:
        return _twist(payload, types)


@register_ros_encoder("geometry_msgs/msg/TwistStamped")
class TwistStampedEncoder(RosEncoder):
    """Jazzy turtlebot4 / Create3 expects a stamped cmd_vel."""

    def encode(self, payload: MotionCommand, types: TypeLookup) -> Any:
        now = time.time()
        stamp = types("builtin_interfaces/msg/Time")(sec=int(now), nanosec=int((now % 1) * 1e9))
        header = types("std_msgs/msg/Header")(stamp=stamp, frame_id=self.params.get("frame_id", "base_link"))
        return types("geometry_msgs/msg/TwistStamped")(header=header, twist=_twist(payload, types))


class RosConverterFactory:

    @staticmethod
    def create_decoder(ros_type: str, **params: Any) -> RosDecoder:
        cls = _decoder_registry.get(ros_type)
        if cls is None:
            raise ValueError(f"No ROS decoder registered for {ros_type}")
        return cls(**params)

    @staticmethod
    def create_encoder(ros_type: str, **params: Any) -> RosEncoder:
        cls = _encoder_registry.get(ros_type)
        if cls is None:
            raise ValueError(f"No ROS encoder registered for {ros_type}")
        return cls(**params)

    @staticmethod
    def available() -> dict[str, list[str]]:
        """Read-only view: ``{"decoders": [...], "encoders": [...]}``."""
        return {"decoders": list(_decoder_registry), "encoders": list(_encoder_registry)}
