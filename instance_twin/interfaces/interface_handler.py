"""
How the twin talks to one agent type: its channels and wire format,
translated to and from the core_msgs the twin works with.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

from core_msgs.topic_contract import Direction, MessageType

logger = logging.getLogger(__name__)

_interface_registry: dict[str, type["AgentInterface"]] = {}


def register_agent_interface(agent_type: str):
    """Decorator to register an AgentInterface subclass for an agent type."""

    normalized_name = agent_type.lower()

    def decorator(cls):
        existing = _interface_registry.get(normalized_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Agent interface '{normalized_name}' is already registered "
                f"for class {existing.__name__}"
            )
        _interface_registry[normalized_name] = cls
        return cls

    return decorator


@dataclass(frozen=True)
class Channel:
    """One agent channel, named by the topic contract: {namespace}/{agent}/{msg_type}."""
    msg_type: MessageType
    direction: Direction

    def to_dict(self) -> dict[str, str]:
        return {self.msg_type.value: self.direction.value}

    @classmethod
    def from_dict(cls, topic_dict: dict[str, str] | None) -> List[Channel]:
        """Unknown entries are skipped, the same way register_node_topics treats them."""
        channels: List[Channel] = []
        for _type, _dir in (topic_dict or {}).items():
            try:
                channels.append(Channel(msg_type=MessageType(_type), direction=Direction(_dir)))
            except ValueError:
                logger.warning("skipping unrecognized topic entry: %s=%s", _type, _dir)
        return channels


class AgentInterface(ABC):
    """Wire <-> core_msgs for one agent type. Never sees the twin, only canonical messages."""

    def __init__(self, agent_name: str, topic_channels: List[Channel], **params: Any):
        self.agent_name = agent_name

        # Everything the agent's sensors and topics ask for, duplicates folded
        self.topic_channels: List[Channel] = list(dict.fromkeys(topic_channels))

    @abstractmethod
    def supports(self, channel: Channel) -> bool:
        """Whether this wire format can carry the channel, e.g. a ROS translation exists."""

    @abstractmethod
    def decode(self, msg_type: MessageType, payload: Any) -> Any | None:
        """Wire payload -> core_msgs message, None to drop."""

    @abstractmethod
    def encode(self, msg_type: MessageType, payload: Any) -> Any | None:
        """Canonical message (MotionCommand, InitialCommandMessage, ...) -> wire payload, None to drop."""

    def channels(self) -> List[Channel]:
        """Requested channels this interface can carry, seen from the instance side."""
        return [ch for ch in self.topic_channels if self.supports(ch)]

    def unsupported(self) -> List[Channel]:
        """Requested channels this interface has no translation for."""
        return [ch for ch in self.topic_channels if not self.supports(ch)]

    def topic_dict(self) -> dict[str, str]:
        """Channels as the {name: dir} dict register_node_topics takes."""
        topics: dict[str, str] = {}
        for channel in self.channels():
            topics.update(channel.to_dict())
        return topics


class AgentInterfaceFactory:

    @staticmethod
    def create_interface(agent_type: str, agent_name: str, **kwargs) -> AgentInterface:
        cls = _interface_registry.get(agent_type.lower()) if agent_type else None
        if cls is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return cls(agent_name, **kwargs)

    @staticmethod
    def get_handler_class(agent_type: str) -> type[AgentInterface] | None:
        """Look up a registered interface class by agent type without instantiation."""
        return _interface_registry.get(agent_type.lower() if agent_type else "")

    @staticmethod
    def available_interfaces() -> dict[str, type[AgentInterface]]:
        """Read-only view of the registry: ``{agent_type: interface_class}``."""
        return dict(_interface_registry)


# avoiding a circular-import error guarantees the registry is populated.
from instance_twin.interfaces import sim_interface, ros_interfaces  # noqa: E402,F401
