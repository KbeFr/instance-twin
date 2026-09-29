"""
Agents that already speak core_msgs over flex topics, like the IR-SIM agent nodes.
"""
from __future__ import annotations

import logging
from typing import Any

import jsonpickle

from core_msgs.instance_agent.controll_payloads import MotionCommand, VelocityCommandMessage
from core_msgs.topic_contract import Direction, MessageType

from instance_twin.interfaces.interface_handler import AgentInterface, Channel, register_agent_interface

logger = logging.getLogger(__name__)


@register_agent_interface("simulated")
class SimulatedInterface(AgentInterface):
    """Channels from the agent's discovered topic dict, jsonpickled core_msgs on the wire."""

    def __init__(self, agent_name: str, topics: dict | None = None, **params: Any):
        super().__init__(agent_name, **params)
        self.topics = topics or {}

    def channels(self) -> list[Channel]:
        out: list[Channel] = []
        for name, direction in self.topics.items():
            try:
                out.append(Channel(MessageType(name), Direction(direction)))
            except ValueError:
                logger.warning("[%s] skipping unrecognized topic %s=%s", self.agent_name, name, direction)
        return out

    def decode(self, msg_type: MessageType, payload: Any) -> Any | None:
        return jsonpickle.decode(payload)

    def encode(self, msg_type: MessageType, payload: Any) -> Any | None:
        """The sim drives on the raw kinematics action; anything else is already core_msgs.
        set_data serializes, so objects are returned as-is."""
        if isinstance(payload, MotionCommand):
            return VelocityCommandMessage(kinematics=payload.kinematics, cmd=list(payload.action))
        return payload
