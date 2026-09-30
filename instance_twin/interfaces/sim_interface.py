"""
Agents that already speak core_msgs over flex topics, like the IR-SIM agent nodes.
"""
from __future__ import annotations

import logging
from typing import Any

import jsonpickle

from core_msgs.instance_agent.controll_payloads import MotionCommand, VelocityCommandMessage
from core_msgs.topic_contract import MessageType, TOPIC_SPECS

from instance_twin.interfaces.interface_handler import AgentInterface, Channel, register_agent_interface

logger = logging.getLogger(__name__)


@register_agent_interface("simulated")
class SimulatedInterface(AgentInterface):
    """Jsonpickled core_msgs on the contract topics, so every contract topic is carried as-is."""

    def supports(self, channel: Channel) -> bool:
        return channel.msg_type in TOPIC_SPECS

    def decode(self, msg_type: MessageType, payload: Any) -> Any | None:
        return jsonpickle.decode(payload)

    def encode(self, msg_type: MessageType, payload: Any) -> Any | None:
        """The sim drives on the raw kinematics action; anything else is already core_msgs.
        set_data serializes, so objects are returned as-is."""
        if isinstance(payload, MotionCommand):
            return VelocityCommandMessage(kinematics=payload.kinematics, cmd=list(payload.action))
        return payload
