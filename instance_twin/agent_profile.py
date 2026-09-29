"""
The shadow model of one agent.
"""
import logging
from importlib.resources import files
from typing import Any

import numpy as np

from instance_twin.battery.battery_handler import BatteryFactory
from instance_twin.irsim_borrowed.controller.controller_handler import ControllerFactory
from instance_twin.irsim_borrowed.geometry.geometryhandler import GeometryFactory
from instance_twin.irsim_borrowed.kinematics.kinematics_handler import KinematicsFactory
from instance_twin.interfaces.interface_handler import AgentInterface, AgentInterfaceFactory
from instance_twin.sensors.base_sensors import PerceptionSensor
from instance_twin.sensors.perception_sensors import PerceptionSensorFactory
from core_msgs.global_msgs.global_payloads import AgentDiscoveryMessage
from core_msgs.instance_agent.controll_payloads import MotionCommand
from core_msgs.instance_aggregate.mission_handshake import MissionBidding
from core_msgs.utils.utils import load_config

logger = logging.getLogger(__name__)

DEFAULT_AGENT_CONFIG_PATH =str( files("instance_twin").joinpath("config", "default_agent_config.yaml"))

class DigitalTwinConfigError(ValueError):
    """Raised when a digital twin cannot be faithfully constructed."""


class AgentProfile:
    def __init__(self, agent_name: str, disc: AgentDiscoveryMessage):

        #self.default_config :dict = load_config(DEFAULT_AGENT_CONFIG_PATH)

        def _get(key: str) -> Any:
            """
            Safely fetches an attribute. Handles missing attributes and explicit Nones.
            TODO: needs be stricter with real deployment
            """
            val = getattr(disc, key, None)
            if val is None :
                    #raise DigitalTwinConfigError(
                    #    f"Cannot initialize agent '{agent_name}': The parameter '{key}' "
                    #    f"is missing from both the discovery message and the fallback config."
                    #)
                logger.warning("AgentConfig parameter missing from discovery %s"
                               "using default from config", key)
            return val

        # Identity
        self.name = agent_name
        self.id = _get("agent_id")
        self.kind = _get("kind")
        self.agent_type = _get("agent_type")

        # specs
        self.radius = _get("radius")
        self.mass = _get("mass")
        self.friction = _get("friction")
        self.avg_speed = _get("avg_speed")
        self.max_speed = _get("max_speed")

        self.shape_config = _get("shape")
        self.topic_dict = _get("topics")

        # Wire format, picked by agent type: simulated core_msgs, a ROS2 bridge robot, ...
        try:
            self.interface: AgentInterface = AgentInterfaceFactory.create_interface(
                self.agent_type, agent_name, topics=self.topic_dict, **(_get("interface") or {}))
        except ValueError as ex:
            raise DigitalTwinConfigError(f"Cannot interface agent '{agent_name}': {ex}") from ex

        # Sub-systems
        self.geometry = GeometryFactory.create_geometry(**self.shape_config)
        self.kinematics = KinematicsFactory.create_kinematics(**_get("kinematics"))

        battery_cfg = _get("battery")
        self.battery = BatteryFactory.create_battery(**battery_cfg) if battery_cfg else None

        controller_cfg = _get("controller")
        self.controller = ControllerFactory.create_controller(**controller_cfg) if controller_cfg else None

        #  Perception Sensors
        self.perception_sensors: dict[str, PerceptionSensor] = {}
        for sensor_config in _get("perception_sensors"):
            sensor = PerceptionSensorFactory.create_handler(**sensor_config)
            self.perception_sensors[sensor.name] = sensor

        # Runtime State, sized by the kinematics: diff [x,y,th]/[v,w], acker adds steer, a uav more
        self.state = np.zeros((self.kinematics.state_dim, 1))
        self.velocity = np.zeros((self.kinematics.action_dim, 1))
        self.goal: np.ndarray | None = None
        self.linked: bool = False
        self.battery_depleted: bool = False
        self.goal_threshold: float = 0.25

    def step_physics(self, dt: float, pending_obstacles: list) -> None:
        """Advances the agent's physical simulation by one time step."""
        if self.goal is None or self.controller is None:
            self.velocity = np.zeros_like(self.velocity)
        else:
            action = self.controller.get_action(self, pending_obstacles)
            self.velocity = np.asarray(action, dtype=float).reshape(self.velocity.shape)

        if self.kinematics is not None:
            self.state = self.kinematics.step(self.state, self.velocity, dt)

        if self.geometry is not None:
            self.geometry.step(self.state)

        if self.battery is not None:
            v = float(self.velocity[0, 0])
            w = float(self.velocity[1, 0]) if self.velocity.shape[0] > 1 else 0.0
            self.battery.step(dt, v, w, self.mass, self.friction)
            self.battery_depleted = self.battery.status <= 0.0

    def estimate_traversal(self, distance: float) -> MissionBidding | None:
        """Time + battery cost for this agent to cover distance metres."""
        if not self.battery:
            return None

        v = max(float(self.avg_speed), 1e-6)
        duration = distance / v

        joules = self.battery.predict_joules(distance, duration, self.mass, self.friction)
        drain = self.battery.to_percent(joules)
        soc_after = self.battery.status - drain

        return MissionBidding(
            time_bidding=duration,
            battery_bidding=drain,
            battery_margin=soc_after,
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
        return self.kinematics.velocity_to_xy(self.state, self.velocity) if self.kinematics else np.zeros((2, 1))

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
    def vel_min(self) -> np.ndarray:
        return np.c_[self.kinematics.vel_min] if self.kinematics else np.zeros((2, 1))

    @property
    def vel_max(self) -> np.ndarray:
        return np.c_[self.kinematics.vel_max] if self.kinematics else np.ones((2, 1))

    @property
    def shape(self) -> str:
        return self.shape_config.get("name", "circle")