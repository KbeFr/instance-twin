import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from core_msgs.utils.frames import Frame
from core_msgs.utils.math import Vector3, Quaternion

logger = logging.getLogger(__name__)


@dataclass
class Offset:
    """A sensor's mount pose, relative to the robot's own body frame.

    Every sensor - aruco, sim2d_object, anything added later - reports the
    same shape on the wire: a full 3D position plus a full orientation.
    """
    position: Vector3
    orientation: Quaternion


# --- Shared Base ---
class BaseSensor(ABC):
    def __init__(self, name: str, offset: Dict[str, Any] | None):
        self.name = name
        self.offset = self._parse_offset(offset or {})

    def _parse_offset(self, offset: Dict[str, Any]) -> Offset:
        """Position + orientation"""
        pos = offset.get("position")
        ori = offset.get("orientation")

        if pos is None:
            logger.warning("[%s] no position given; defaulting to 0,0,0.", self.name)
        if ori is None:
            logger.warning("[%s] no orientation given; defaulting to identity.", self.name)

        return Offset(
            position=Vector3(**pos) if pos else Vector3(),
            orientation=Quaternion(**ori) if ori else Quaternion(),
        )


# --- Functional Branches ---
class PerceptionSensor(BaseSensor):
    def _build_mount(self) -> Frame:
        """Offset -> mount frame, relative to the robot body"""
        return Frame(self.offset.position, self.offset.orientation)

    def sensor_frame(self, x: float, y: float, theta: float) -> Frame:
        """The sensor's own frame in the world, based on agent position and orientation (2d)"""
        return Frame.from_2d(x, y, theta).compose(self._build_mount())

    @abstractmethod
    def get_obstacle_observations(self, payload, robot_pose) -> list:
        pass


class StateSensor(BaseSensor):
    @abstractmethod
    def get_state_update(self, payload) -> dict:
        pass