"""
Robots behind the flexia ROS2 bridge. Each robot type only declares its channels;
the codec and the per-ROS-type converters do the translation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

from core_msgs.topic_contract import Direction, MessageType

from instance_twin.interfaces.interface_handler import AgentInterface, Channel, register_agent_interface
from instance_twin.interfaces.ros_codec import RosCodec, RosSerializedMessage
from instance_twin.interfaces.ros_converters import RosConverterFactory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RosChannel:
    ros_type: str
    params: dict = field(default_factory=dict)


class RosBridgeInterface(AgentInterface):
    """Contract channels carrying CDR in _payload; the bridge routes them to its ROS topics."""

    ros_distro: ClassVar[str] = "humble"
    inbound: ClassVar[dict[MessageType, RosChannel]] = {}
    outbound: ClassVar[dict[MessageType, RosChannel]] = {}

    def __init__(self,
                 agent_name: str,
                 ros_distro: str | None = None,
                 params: dict[str, dict] | None = None,
                 extra_msgs: dict[str, str] | None = None,
                 **kwargs: Any
    ):
        super().__init__(agent_name)

        self._codec = RosCodec(ros_distro or self.ros_distro, extra_msgs)

        # Per-deployment overrides keyed by message type, e.g. {"pose": {"origin": [1, 2, 0]}}
        overrides = params or {}
        self._decoders = {
            msg_type: RosConverterFactory.create_decoder(
                channel.ros_type,
                **{**channel.params, **overrides.get(msg_type.value, {})})
            for msg_type, channel in self.inbound.items()
        }
        self._encoders = {
            msg_type: RosConverterFactory.create_encoder(
                channel.ros_type,
                **{**channel.params, **overrides.get(msg_type.value, {})})
            for msg_type, channel in self.outbound.items()
        }

    def channels(self) -> list[Channel]:
        channels = [Channel(msg_type, Direction.IN) for msg_type in self.inbound]
        channels.extend(Channel(msg_type, Direction.OUT) for msg_type in self.outbound)
        return channels

    def decode(self, msg_type: MessageType, payload: Any) -> Any | None:
        channel = self.inbound.get(msg_type)
        if channel is None:
            return None
        return self._decoders[msg_type].decode(self._codec.decode(payload, channel.ros_type))

    def encode(self, msg_type: MessageType, payload: Any) -> RosSerializedMessage | None:
        """Only what the channel's encoder accepts; a link message has no ROS form and is dropped."""
        channel = self.outbound.get(msg_type)
        encoder = self._encoders.get(msg_type)
        if channel is None or not isinstance(payload, encoder.accepts):
            return None
        return self._codec.encode(encoder.encode(payload, self._codec.type), channel.ros_type)


@register_agent_interface("turtlebot4")
class Turtlebot4Interface(RosBridgeInterface):
    """TurtleBot4 on Humble, streams without a link handshake, so its first pose is the link ack."""
    inbound = {
        MessageType.POSE: RosChannel("nav_msgs/msg/Odometry"),
        #MessageType.DETECTIONS: RosChannel("sensor_msgs/msg/LaserScan", {"sensor_type": "lidar"}),
        MessageType.BATTERY: RosChannel("sensor_msgs/msg/BatteryState"),
    }
    outbound = {MessageType.ACTION: RosChannel("geometry_msgs/msg/Twist")}


@register_agent_interface("turtlebot4_jazzy")
class Turtlebot4JazzyInterface(Turtlebot4Interface):
    """Jazzy switched cmd_vel to TwistStamped."""
    ros_distro = "jazzy"
    outbound = {MessageType.ACTION: RosChannel("geometry_msgs/msg/TwistStamped")}
