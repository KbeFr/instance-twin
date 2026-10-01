"""
The shadow model of one agent.
"""
import logging
from collections import defaultdict
from typing import Any

import numpy as np

from instance_twin.battery.battery_handler import BatteryFactory
from instance_twin.interfaces import Channel
from instance_twin.irsim_borrowed.controller.controller_handler import ControllerFactory
from instance_twin.irsim_borrowed.geometry.geometryhandler import GeometryFactory
from instance_twin.irsim_borrowed.kinematics.kinematics_handler import KinematicsFactory
from instance_twin.interfaces.interface_handler import AgentInterface, AgentInterfaceFactory
from instance_twin.estimation.estimator_handler import EstimatorFactory, StateEstimator
from instance_twin.sensors.base_sensors import PerceptionSensor, StateSensor
from instance_twin.sensors.perception_sensors import PerceptionSensorFactory
from instance_twin.sensors.state_sensors import StateSensorFactory
from core_msgs.global_msgs.global_payloads import AgentDiscoveryMessage
from core_msgs.instance_agent.controll_payloads import MotionCommand
from core_msgs.instance_agent.sensor_payloads import DetectionMessage
from core_msgs.instance_aggregate.mission_handshake import MissionBidding
from core_msgs.instance_aggregate.payloads import ObstacleObservation
from core_msgs.topic_contract import MessageType, Direction

logger = logging.getLogger(__name__)

# The only default: there is a single estimator implementation
DEFAULT_ESTIMATOR = {"name": "ekf"}


class DigitalTwinConfigError(ValueError):
    """Raised when a digital twin cannot be faithfully constructed."""


class AgentProfile:
    def __init__(self, agent_name: str, disc: AgentDiscoveryMessage):
        self.name = agent_name

        def get(key: str, default: Any = None) -> Any:
            val = getattr(disc, key, None)
            return default if val is None else val

        def need(key: str) -> Any:
            val = getattr(disc, key, None)
            if not val and val != 0:
                raise DigitalTwinConfigError(f"agent '{agent_name}': discovery is missing '{key}'")
            return val

        # Identity
        self.id = need("agent_id")
        self.kind = need("kind")
        self.agent_type = need("agent_type")

        # Specs
        self.radius = need("radius")
        self.max_speed = need("max_speed")
        self.mass = get("mass")
        self.friction = get("friction")
        self.avg_speed = get("avg_speed")
        self.shape_config = need("shape")

        # Only what no sensor or model covers, like {"action": "out"}
        self.topic_dict = get("topics", {})

        # Models
        kinematics_cfg = need("kinematics")
        self.geometry = GeometryFactory.create_geometry(**self.shape_config)
        self.kinematics = KinematicsFactory.create_kinematics(**kinematics_cfg)

        # Estimator predicts with a noise-free copy: its own covariance is the noise model
        estimator_cfg = dict(get("estimator", DEFAULT_ESTIMATOR))
        self.estimator: StateEstimator = EstimatorFactory.create_estimator(
            kinematics=KinematicsFactory.create_kinematics(**{**kinematics_cfg, "noise": False}),
            **estimator_cfg,
        )

        battery_cfg = get("battery")
        self.battery = BatteryFactory.create_battery(**battery_cfg) if battery_cfg else None
        if self.battery and None in (self.mass, self.friction, self.avg_speed):
            raise DigitalTwinConfigError(f"agent '{agent_name}': battery needs mass, friction and avg_speed")

        controller_cfg = get("controller")
        self.controller = ControllerFactory.create_controller(**controller_cfg) if controller_cfg else None

        # Sensors: at least one state sensor, or the estimator never anchors. A static agent
        # (e.g. a fixed camera) may instead be placed once through estimator.initial_pose.
        state_cfgs = get("state_sensors", [])

        if not state_cfgs and estimator_cfg.get("initial_pose") is None:
            raise DigitalTwinConfigError(
                f"agent '{agent_name}': needs state_sensors, or estimator.initial_pose for a static agent")

        self.state_sensors: dict[str, StateSensor] = {
            s.name: s for s in (StateSensorFactory.create_sensor(**c) for c in state_cfgs)}
        self.perception_sensors: dict[str, PerceptionSensor] = {
            s.name: s for s in (PerceptionSensorFactory.create_handler(**c) for c in get("perception_sensors", []))}

        # Topic -> sensors listening on it
        self._state_routes: dict[MessageType, list[StateSensor]] = defaultdict(list)
        for s in self.state_sensors.values():
            self._state_routes[s.topic].append(s)
        self._perception_routes: dict[MessageType, list[PerceptionSensor]] = defaultdict(list)
        for s in self.perception_sensors.values():
            self._perception_routes[s.topic].append(s)

        # Instance-scoped topics come from the aggregate as core_msgs; only the rest is the agent's
        sensors = [*self.state_sensors.values(), *self.perception_sensors.values()]
        self.instance_topics: list[MessageType] = list(dict.fromkeys(
            s.topic for s in sensors if s.instance_scoped))

        # Interface: everything announced must be carried, or the twin would silently miss it
        topics = [t for t in (*self._state_routes, *self._perception_routes) if t not in self.instance_topics]
        if self.battery:
            topics.append(self.battery.topic)
        channels = [Channel(t, Direction.IN) for t in topics] + Channel.from_dict(self.topic_dict)

        interface_kwargs = need("interface")
        try:
            self.interface: AgentInterface = AgentInterfaceFactory.create_interface(
                 topic_channels=channels, **interface_kwargs)
        except ValueError as ex:
            raise DigitalTwinConfigError(f"Cannot interface agent '{agent_name}': {ex}") from ex

        missing = list(self.interface.unsupported())
        if missing:
            raise DigitalTwinConfigError(
                f"agent type '{self.agent_type}' has no translation for "
                f"{[ch.to_dict() for ch in missing]} of agent '{agent_name}'")

        # Runtime state, sized by the kinematics: diff [x,y,th]/[v,w], acker adds steer, a uav more
        self.state = np.zeros((self.kinematics.state_dim, 1))
        self.velocity = np.zeros((self.kinematics.action_dim, 1))
        self.goal: np.ndarray | None = None
        self.linked: bool = False
        self.battery_depleted: bool = False
        self.goal_threshold: float = 0.25

    def ingest(self, topic: MessageType, msg: Any, received: float) -> list[ObstacleObservation]:
        """Hands one agent message to everything listening on its topic; returns the obstacles it revealed.
        Any message on the agent's own channels proves the link; a relayed fix does not."""
        if topic not in self.instance_topics:
            self.linked = True

        for sensor in self._state_routes.get(topic, ()):
            measurement = sensor.get_state_update(msg)
            if measurement is None:
                continue
            self.estimator.predict_to(received, self.velocity)
            if not self.estimator.update(measurement):
                logger.debug("[%s] %s measurement rejected by the gate", self.name, sensor.name)
        self.state = self.estimator.state

        if self.battery and topic == self.battery.topic:
            self.battery.measure(msg)

        # Projecting from an unanchored pose would plant obstacles in the wrong place
        perception = self._perception_routes.get(topic, ())
        if not perception or not self.estimator.initialized:
            return []
        payload = msg.payload if isinstance(msg, DetectionMessage) else msg
        observations: list[ObstacleObservation] = []
        for sensor in perception:
            observations.extend(sensor.get_obstacle_observations(payload, self.pose))
        return observations

    def step_physics(self, dt: float, pending_obstacles: list, now: float, halt: bool = False) -> None:
        """Brings the estimate up to now under the last command, then picks the next command from it."""
        self.estimator.predict_to(now, self.velocity)
        self.state = self.estimator.state

        # No command until the estimate is anchored: driving blind from (0, 0) is worse than waiting
        if halt or self.goal is None or self.controller is None or not self.estimator.initialized:
            self.velocity = np.zeros_like(self.velocity)
        else:
            action = self.controller.get_action(self, pending_obstacles)
            self.velocity = np.asarray(action, dtype=float).reshape(self.velocity.shape)

        self.geometry.step(self.state)

        if self.battery:
            v = float(self.velocity[0, 0])
            w = float(self.velocity[1, 0]) if self.velocity.shape[0] > 1 else 0.0
            self.battery.step(dt, v, w, self.mass, self.friction)
            self.battery_depleted = self.battery.status <= 0.0

    def estimate_traversal(self, distance: float) -> MissionBidding | None:
        """Time + battery cost for this agent to cover distance metres."""
        if not self.battery:
            return None

        duration = distance / max(float(self.avg_speed), 1e-6)
        joules = self.battery.predict_joules(distance, duration, self.mass, self.friction)
        drain = self.battery.to_percent(joules)

        return MissionBidding(
            time_bidding=duration,
            battery_bidding=drain,
            battery_margin=self.battery.status - drain,
        )

    def motion_command(self) -> MotionCommand:
        """Current action in every form an interface may need: raw action plus body twist."""
        linear, angular = self.kinematics.body_twist(self.state, self.velocity)
        return MotionCommand(
            kinematics=self.kinematics.name,
            action=self.velocity.reshape(-1).tolist(),
            linear=linear,
            angular=angular,
        )

    @property
    def position(self) -> np.ndarray:
        return self.state[0:2]

    @property
    def pose(self) -> tuple[float, float, float]:
        """(x, y, theta) -- what perception projects from."""
        return (
            float(self.state[0, 0]),
            float(self.state[1, 0]),
            float(self.state[2, 0]) if self.state.shape[0] > 2 else 0.0,
        )

    @property
    def velocity_xy(self) -> np.ndarray:
        return self.kinematics.velocity_to_xy(self.state, self.velocity)

    @property
    def vel_min(self) -> np.ndarray:
        return np.c_[self.kinematics.vel_min]

    @property
    def vel_max(self) -> np.ndarray:
        return np.c_[self.kinematics.vel_max]

    @property
    def shape(self) -> str:
        return self.shape_config.get("name", "circle")