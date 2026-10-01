"""
Robots behind the flexia ROS2 bridge. The agent's sensors decide the channels; a channel travels
as CDR when the topic has a ROS translation, as core_msgs when the system publishes it.
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar

import jsonpickle

from core_msgs.topic_contract import Direction, MessageType

from instance_twin.interfaces.interface_handler import AgentInterface, Channel, register_agent_interface
from instance_twin.interfaces.ros.ros_codec import RosBridgeEnvelope, RosCodec
from instance_twin.interfaces.ros.ros_converters import RosConverterFactory, RosDecoder, RosEncoder
from instance_twin.interfaces.ros.ros_topic_mapping import EXTRA_MSGS, ROS_TOPIC_MAPPING, RosChannel

logger = logging.getLogger(__name__)


@register_agent_interface("ros2")
class RosBridgeInterface(AgentInterface):
    """Any ROS2 robot on the stock mapping; robot types subclass it to override what differs."""

    ros_distro: ClassVar[str] = "humble"
    ros_types: ClassVar[dict[MessageType, RosChannel]] = {}

    def __init__(self,
                 name : str,
                 topic_channels: list[Channel],
                 ros_distro: str | None = None,
                 ros_types: dict[str, str] | None = None,
                 params: dict[str, dict] | None = None,
                 extra_msgs: dict[str, str] | None = None,
                 **kwargs: Any
    ):
        super().__init__(name, topic_channels)

        self._codec = RosCodec(ros_distro or self.ros_distro, {**EXTRA_MSGS, **(extra_msgs or {})})

        # Stock table < robot class < deployment config, like ros_types: {action: geometry_msgs/msg/TwistStamped}
        mapping = {**ROS_TOPIC_MAPPING,
                   **{MessageType(k): RosChannel(v) for k, v in (ros_types or {}).items()}}
        ## could integrate ros_channel and channel I think for prettier code

        # Per-deployment converter params keyed by topic like {"lidar": {"max_range": 2.0}}
        overrides = params or {}
        self._ros: dict[MessageType, RosChannel] = {}
        self._decoders: dict[MessageType, RosDecoder] = {}
        self._encoders: dict[MessageType, RosEncoder] = {}

        for channel in self.topic_channels:
            ros = mapping.get(channel.msg_type)
            if ros is None or not self._codec.knows(ros.ros_type):
                continue
            converter_params = {**ros.params, **overrides.get(channel.msg_type.value, {})}
            try:
                if channel.direction in (Direction.IN, Direction.INOUT):
                    self._decoders[channel.msg_type] = RosConverterFactory.create_decoder(ros.ros_type, **converter_params)
                if channel.direction in (Direction.OUT, Direction.INOUT):
                    self._encoders[channel.msg_type] = RosConverterFactory.create_encoder(ros.ros_type, **converter_params)
            except ValueError:
                continue    # no converter for this ROS type: reported through unsupported()
            self._ros[channel.msg_type] = ros

    @property
    def ros_topics(self) -> dict[str, str]:
        """Contract topic -> ROS type for every channel built, e.g. to configure the bridge from."""
        return {msg_type.value: ros.ros_type for msg_type, ros in self._ros.items()}

    def supports(self, channel: Channel) -> bool:
        """A built translation, or an inbound topic the system publishes in core_msgs."""
        if channel.direction in (Direction.IN, Direction.INOUT) and channel.msg_type in self._decoders:
            return True
        if channel.direction in (Direction.OUT, Direction.INOUT) and channel.msg_type in self._encoders:
            return True
        return False

    def decode(self, msg_type: MessageType, payload: Any) -> Any | None:
        decoder = self._decoders.get(msg_type)
        if decoder is None:
            logger.warning("decoder none for message_type %s", msg_type.value )
            return None
        return decoder.decode(self._codec.decode(payload, self._ros[msg_type].ros_type))

    def encode(self, msg_type: MessageType, payload: Any) -> RosBridgeEnvelope | None:
        """Only what the channel's encoder accepts; a link message has no ROS form and is dropped."""
        encoder = self._encoders.get(msg_type)
        if encoder is None or not isinstance(payload, encoder.accepts):
            return None
        return self._codec.encode(encoder.encode(payload, self._codec.type), self._ros[msg_type].ros_type)


@register_agent_interface("turtlebot4")
class Turtlebot4Interface(RosBridgeInterface):
    """TurtleBot4 on Humble, streams without a link handshake, so its first pose is the link ack."""
    inbound = {
        MessageType.ODOM: RosChannel("nav_msgs/msg/Odometry"),
        MessageType.BATTERY: RosChannel("sensor_msgs/msg/BatteryState"),
        MessageType.IMU: RosChannel("sensor_msgs/msg/Imu"),

    }
    outbound = {MessageType.ACTION: RosChannel("geometry_msgs/msg/Twist")}


@register_agent_interface("turtlebot4_jazzy")
class Turtlebot4JazzyInterface(Turtlebot4Interface):
    """Jazzy switched cmd_vel to TwistStamped."""
    ros_distro = "jazzy"

    outbound = {MessageType.ACTION: RosChannel("geometry_msgs/msg/TwistStamped")}
