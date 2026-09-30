from __future__ import annotations

import logging
import math
from typing import Any, Dict

from core_msgs.topic_contract import MessageType
from instance_twin.sensors.base_sensors import PerceptionSensor
from core_msgs.utils.frames import Frame
from core_msgs.instance_agent.sensor_payloads import ArucoDetection, DetectedObjectSim2D
from core_msgs.instance_aggregate.payloads import ObstacleObservation


logger = logging.getLogger(__name__)

_perception_registry = {}


def register_perception_sensor(name: str):
    """Decorator to register a SensorHandler subclass"""

    normalized_name = name.lower()

    def decorator(cls):
        existing = _perception_registry.get(normalized_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Perception sensor '{normalized_name}' is already registered "
                f"for class {existing.__name__}"
            )
        _perception_registry[normalized_name] = cls
        return cls

    return decorator


@register_perception_sensor("aruco")
class ArucoDetectionSensor(PerceptionSensor):
    """Marker poses relative to a camera -> world obstacles."""

    default_topic = MessageType.ARUCO_DETECTIONS

    def __init__(self,
                 name: str,
                 offset: Dict[str, Any] | None = None,
                 radius: float = 0.8,
                 topic: str | None = None,
                 **params: Any):
        self.radius = float(radius)
        super().__init__(name, topic ,offset)

    def get_obstacle_observations(self, payload: ArucoDetection, robot_pose) -> list[ObstacleObservation]:
        if not isinstance(payload, ArucoDetection):
            logger.warning("[%s] expected ArucoDetection, got %s", self.name, type(payload).__name__)
            return []

        sensor = self.sensor_frame(*robot_pose)
        out: list[ObstacleObservation] = []

        for marker_id, pose in zip(payload.marker_ids or [], payload.poses or []):
            marker = Frame(pose.position, pose.orientation)
            world = sensor.compose(marker)

            out.append(ObstacleObservation(
                # A marker id is globally unique and stable, so two cameras
                # seeing marker 7 report one obstacle.
                id=f"aruco:{marker_id}",
                x=world.x,
                y=world.y,
                theta=world.yaw,
                radius=self.radius,
            ))

        return out


@register_perception_sensor("sim2d_object")
class Sim2DDetectionHandler(PerceptionSensor):
    """Relative object positions -> world obstacles."""

    default_topic = MessageType.SIM2D_DETECTIONS

    def __init__(self,
                 name: str,
                 offset: Dict[str, Any] | None = None,
                 radius: float = 0.8,
                 topic: str | None = None,
                 **params: Any):
        self.radius = float(radius)
        super().__init__(name, topic, offset)

    def get_obstacle_observations(self, payload: list[DetectedObjectSim2D], robot_pose) -> list[ObstacleObservation]:
        if isinstance(payload, DetectedObjectSim2D):
            payload = [payload]
        if not isinstance(payload, (list, tuple)):
            logger.warning("[%s] expected a list of DetectedObjectSim2D, got %s",
                           self.name, type(payload).__name__)
            return []

        sensor = self.sensor_frame(*robot_pose)
        out: list[ObstacleObservation] = []

        for det in payload:
            if det.distance is None or det.bearing is None:
                continue
            # polar (distance, bearing) in the sensor's own frame -> local xy
            local_x = det.distance * math.cos(det.bearing)
            local_y = det.distance * math.sin(det.bearing)
            wx, wy, _ = sensor.apply(local_x, local_y)

            out.append(ObstacleObservation(
                # No identity: a plain camera cannot claim its blob #3 is
                # the same object another camera called #3. The sim's ids
                # happen to be stable, so they are namespaced per sensor.
                id=f"{self.name}:{det.id}",
                x=wx,
                y=wy,
                # A position-only detection carries no heading.
                theta=0.0,
                radius=self.radius,
                confidence=det.confidence if det.confidence is not None else 1.0,
            ))

        return out


@register_perception_sensor("lidar")
class LidarClusterSensor(PerceptionSensor):
    """Scan clusters -> world obstacles, keyed by grid cell since clusters carry no identity."""

    default_topic = MessageType.LIDAR


    def __init__(self,
                 name: str, offset: Dict[str, Any] | None = None,
                 radius: float = 0.1,
                 cell: float = 0.3,
                 topic: str | None = None,
                 **params: Any):
        self.radius = float(radius)
        self.cell = float(cell)
        super().__init__(name, topic, offset)

    def get_obstacle_observations(self, payload: list[DetectedObjectSim2D], robot_pose) -> list[ObstacleObservation]:
        if not isinstance(payload, (list, tuple)):
            logger.warning("[%s] expected a list of DetectedObjectSim2D, got %s",
                           self.name, type(payload).__name__)
            return []

        sensor = self.sensor_frame(*robot_pose)
        out: list[ObstacleObservation] = []

        for det in payload:
            if det.distance is None or det.bearing is None:
                continue
            wx, wy, _ = sensor.apply(det.distance * math.cos(det.bearing),
                                     det.distance * math.sin(det.bearing))
            out.append(ObstacleObservation(
                # Same cell -> same id, so a static wall refreshes instead of piling up.
                id=f"{self.name}:{round(wx / self.cell)}:{round(wy / self.cell)}",
                x=wx,
                y=wy,
                theta=0.0,
                radius=det.radius if det.radius is not None else self.radius,
                confidence=det.confidence if det.confidence is not None else 1.0,
            ))

        return out


class PerceptionSensorFactory:
    """
    Factory class to create DetectionHandlers.
    """

    @staticmethod
    def create_handler(name: str, **kwargs) -> PerceptionSensor:
        name = name.lower()
        cls = _perception_registry.get(name) if name else None
        if cls is None:
            raise ValueError(f"Unknown detection type: {name}")
        return cls(name, **kwargs)

    @staticmethod
    def get_handler_class(name: str) -> type[PerceptionSensor] | None:
        """Look up a registered handler class by name without instantiation """
        return _perception_registry.get(name.lower() if name else "")

    @staticmethod
    def available_sensors() -> dict[str, type[PerceptionSensor]]:
        """Read-only view of the registry: ``{name: handler_class}``."""
        return dict(_perception_registry)