import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict

from core_msgs.topic_contract import MessageType
from core_msgs.utils.frames import Frame
from core_msgs.utils.math import Vector3, Quaternion
from instance_twin.estimation.estimator_handler import Measurement

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

    default_topic: ClassVar[MessageType]

    # True: published by the aggregate on {ns}/{instance}/..., core_msgs as-is (no agent interface)
    instance_scoped: ClassVar[bool] = False

    def __init__(self, name: str,
                 topic: str | None = None,
                 offset: Dict[str, Any] | None = None
    ):
        self.name = name
        self.offset = self._parse_offset(offset or {})
        self.topic = MessageType(topic) if topic else self.default_topic


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

    def _build_mount(self) -> Frame:
        """Offset -> mount frame, relative to the robot body"""
        return Frame(self.offset.position, self.offset.orientation)

    def sensor_frame(self, x: float, y: float, theta: float) -> Frame:
        """The sensor's own frame in the world, based on agent position and orientation (2d)"""
        return Frame.from_2d(x, y, theta).compose(self._build_mount())


# --- Functional Branches ---
class PerceptionSensor(BaseSensor):

    @abstractmethod
    def get_obstacle_observations(self, payload, robot_pose) -> list:
        pass


class StateSensor(BaseSensor):
    """Payload -> Measurement. The components given a noise std are the ones fused."""

    components: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, name: str,
                 noise: Dict[str, float],
                 topic: str | None = None,
                 offset: Dict[str, Any] | None = None
    ):
        # State sensors sit at the body origin unless told otherwise
        super().__init__(name, topic, offset or {"position": {}, "orientation": {}})

        self.noise = {k: float(v) for k, v in (noise or {}).items()}

        unknown = set(self.noise) - self.components
        if unknown or not self.noise:
            raise ValueError(f"[{name}] noise must name components from {sorted(self.components)}, "
                             f"got {sorted(self.noise)}")

    @abstractmethod
    def read(self, payload) -> Dict[str, float | None] | None:
        """Payload -> raw component values, None to drop."""

    def get_state_update(self, payload) -> Measurement | None:
        """Only configured components with a value make it into the measurement."""
        values = self.read(payload)
        if not values:
            return None
        picked = {k: float(values[k]) for k in self.noise if values.get(k) is not None}
        if not picked:
            return None
        # The sender's sigma when it gives one; the configured noise is the floor we never trust past
        reported = getattr(payload, "std", None) or {}
        noise = {k: max(self.noise[k], float(reported.get(k) or 0.0)) for k in picked}
        return Measurement(values=picked, noise=noise, source=self.name)