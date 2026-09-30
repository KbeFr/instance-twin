"""
instance_comms.py
"""
from __future__ import annotations

import time
from functools import partial

import jsonpickle

from flexCommunicator.clientLibraries.flcpy.flexNode import flexNode
from flexCommunicator.clientLibraries.flcpy.utils.constants import APPLICATION_STATUS

from core_msgs.global_msgs.global_payloads import (
    HeartBeatMessage, InstanceDiscoveryMessage, RegisteredMessage,
)
from core_msgs.topic_contract import Direction, MessageType, register_node_topics, get_data_name

from instance_twin.interfaces.interface_handler import AgentInterface


INSTANCE_DISCOVERY = True


class InstanceNetworkNode(flexNode):
    """FlexNode interface to the instance twin.

    Liveness, by phase:
        unbound, unknown    -> DISCOVERY (only when INSTANCE_DISCOVERY)
        unbound, registered -> HEARTBEAT to the aggregate
        bound               -> neither: TWIN_STATE is the heartbeat, and a bound
                               instance that keeps beating looks like a free one
    """

    def __init__(self,
                 flex_config:dict,
                 comm_matrix_path:str,
                 global_topic_dict: dict,
                 aggregate_topic_dict: dict,
                 twin,
                 loop_freq: float,
    ):

        super().__init__(
            config=flex_config["Instance_Twin_Config"],
            loaded_config=True,
            communicationMatrix=comm_matrix_path,
            verbose=True,
        )

        self.twin = twin
        self.namespace = twin.namespace
        self.node_name = twin.name

        self.instance_discovery = INSTANCE_DISCOVERY
        self._instantiate_topic = (MessageType.ACTIVATE if INSTANCE_DISCOVERY
                                   else MessageType.INSTANTIATE)

        self.decode_errors = 0

        self._aggregate_topic_dict = aggregate_topic_dict

        self._agg_topics: dict = {}
        self._link_topics: dict = {}                            # agent msg types we publish on
        self._interface: AgentInterface | None = None
        self._agent_name: str | None = None
        self._warned: set[str] = set()

        self._registered_in: dict | None = None   # inbound callbacks actually wired
        self._aggregate_name: str | None = None
        self._heartbeat_paused = False
        self.discovery_timer = None

        # Check if we use discovery for instance
        if INSTANCE_DISCOVERY:
            # Create the timer for discovery and run it
            _discovery_freq = loop_freq/2
            self.discovery_timer = self.create_timer(timer_period=1 / _discovery_freq,
                                            callback=self.send_discovery, autostart=True)

            # Discovery is only global topic
            topic_dict = {"activate" : "inout" , "discovery" : "out", "heartbeat": "out"}
            self._registered_in = {MessageType.ACTIVATE: self._ingest_instantiate}
            register_node_topics(
                node=self, topic_dict=topic_dict,
                namespace=self.namespace, node_id=self.node_name,
                in_callbacks=self._registered_in,
            )
        else:
            # Only global instantiate topic needs to be set at init
            topic_dict = {"instantiate" : "inout"}
            self._registered_in = {MessageType.INSTANTIATE: self._ingest_instantiate}
            register_node_topics(
                node=self, topic_dict=topic_dict,
                namespace=self.namespace, node_id=self.node_name,
                in_callbacks=self._registered_in,
            )


        # Not autostart: the twin only steps once it owns an agent
        self.step_timer = self.create_timer(timer_period=1 / loop_freq,
                                            callback=twin.step, autostart=False)


        self.application_status.set(value=APPLICATION_STATUS.INITIALIZED)

    # --- inbound: decode, then queue -------------

    def _decode(self, kind: str, payload: str):
        try:
            return jsonpickle.decode(payload)
        except Exception:
            self.decode_errors += 1
            self.logger.exception("decode failed kind=%s", kind)
            return None

    def _ingest(self, kind: str, payload: str) -> None:
        msg = self._decode(kind, payload)
        if msg is not None:
            self.twin.inbox.put((kind, msg, time.monotonic()))

    def _ingest_instantiate(self, payload: str) -> None:
        """Handled inline, not queued: the step timer isn't running when unbound."""
        msg = self._decode(MessageType.INSTANTIATE, payload)
        if msg is None:
            return

        #For the discovery flow
        if isinstance(msg, RegisteredMessage):
            if msg.node_name == self.node_name:
                self.handle_registered(msg.aggregate_name)
            return

        try:
            self.twin.handle_instantiate(msg)
        except Exception:
            self.logger.exception("instantiate handler failed")

    # --- outbound -----------------------------------------------------------

    def publish_to_aggregate(self, msg_type: MessageType, payload) -> None:
        """Scoped to this instance: the aggregate subscribes per instance."""
        if payload is None:
            return
        self._publish(get_data_name(self.node_name, msg_type), payload)

    def publish_to_agent(self, msg_type: MessageType, payload) -> None:
        """Canonical message -> the agent's wire format through its interface, then publish."""
        interface = self._interface
        if interface is None or payload is None or msg_type not in self._link_topics:
            return
        try:
            wire = interface.encode(msg_type, payload)
        except Exception as ex:
            key = f"encode:{msg_type.value}"
            if key not in self._warned:
                self._warned.add(key)
                self.logger.warning("encode failed on %s: %s (suppressing repeats)", msg_type.value, ex)
            return
        if wire is not None:
            self._publish(get_data_name(self._agent_name, msg_type), wire)

    def _publish(self, data_name: str, payload) -> None:
        try:
            self.set_data(data_name, payload)
        except Exception:
            key = str(data_name)
            if key not in self._warned:
                self._warned.add(key)
                self.logger.exception("publish failed on %s (suppressing repeats)", key)

    # --- subscription lifecycle ---------------------------------------------

    def subscribe_agent(self, agent_name: str, interface: AgentInterface) -> None:
        """Wire up both channels: aggregate <-> instance and instance <-> agent."""
        self._agent_name = agent_name
        self._interface = interface

        self._agg_topics = register_node_topics(
            node=self, topic_dict=self._aggregate_topic_dict,
            namespace=self.namespace, node_id=self.node_name,
            in_callbacks={
                MessageType.MISSION:  lambda p: self._ingest(MessageType.MISSION, p),
            },
        )

        # The interface only carries what the agent's sensors asked for, so every inbound channel is wired
        inbound = [ch.msg_type for ch in interface.channels() if ch.direction in (Direction.IN, Direction.INOUT)]
        self._link_topics = register_node_topics(
            node=self, topic_dict=interface.topic_dict(),
            namespace=self.namespace, node_id=agent_name,
            in_callbacks={msg_type: partial(self._ingest_agent, msg_type) for msg_type in inbound},
        )
        self.logger.debug("Subscribed to agent=%s via %s", agent_name, type(interface).__name__)

    def _ingest_agent(self, msg_type: MessageType, payload) -> None:
        """Agent wire format -> core_msgs through its interface, then queue."""
        interface = self._interface
        if interface is None:
            return
        try:
            msg = interface.decode(msg_type, payload)
        except Exception as ex:
            self.decode_errors += 1
            key = f"decode:{msg_type.value}"
            if key not in self._warned:
                self._warned.add(key)
                self.logger.warning("decode failed on %s: %s (suppressing repeats)", msg_type.value, ex)
            return
        if msg is not None:
            # Stamped on arrival: agent clocks are not synced with ours, so their stamps can't time the filter
            self.twin.inbox.put((msg_type, msg, time.monotonic()))

    def unsubscribe_agent(self) -> None:
        """
        *Still needs the logic in the flex node to actually be able to unsubscribe.
        Until then the interface is dropped, so late agent traffic is ignored here.
        """
        self._agg_topics = {}
        self._link_topics = {}
        self._interface = None
        self._agent_name = None

    # --- stepping -----------------------------------------------------------

    def start_stepping(self) -> None:
        if hasattr(self.step_timer, "start"):
            self.step_timer.start()
            self.logger.info("twin stepping started (loop timer .start() called)")
        else:
            self.logger.error(
                "twin timer has no 'start' method (%s) - stepping was never "
                "started, so the twin will never drain its inbox or seed "
                "from pose", type(self.step_timer).__name__,
            )

    def stop_stepping(self) -> None:
        self.step_timer.stop()


    # --- Discovery ------------------------------

    def send_discovery(self):
        """The comm layer will handle the discovery sending, answers will be routed to base"""
        #TODO add version to discovery, to see if compatible with agent?
        message = InstanceDiscoveryMessage(self.node_name)

        # handles the global scope automatically
        self.publish_to_aggregate(MessageType.DISCOVERY, message)

    def stop_discovery(self) -> None:
        if self.discovery_timer is not None:
            self.discovery_timer.stop()

    def start_discovery(self) -> None:
        if self.discovery_timer is not None:
            self.discovery_timer.start()


    # --- Registered on aggregate with discovery

    def handle_registered(self , aggregate_name : str) -> None:
        """Discovery has arrived, stop discovery and start heartbeat"""
        self.stop_discovery()

        self._aggregate_name = aggregate_name
        self.twin.aggregate_name = aggregate_name

        # attach custom heartbeat to flexNodes heartbeat (configured in flexNode config)
        if self.heartbeat is None:
            self.logger.error("No heartbeat registered for instance twin %s, we be seen as stale in aggregate" ,self.node_name)
            return

        self.heartbeat.attach(self._send_heartbeat)

    def _send_heartbeat(self) -> None:
        if self._heartbeat_paused or self._aggregate_name is None:
            return
        self._publish(get_data_name(self.node_name, MessageType.HEARTBEAT),HeartBeatMessage())

    # --- Liveness lifecycle, called by the twin at bind / unbind ------------

    def pause_heartbeat(self) -> None:
        self._heartbeat_paused = True
        self.stop_discovery()

    def resume_heartbeat(self) -> None:
        self._heartbeat_paused = False
        if self._aggregate_name is None:
            # Never registered, or the aggregate restarted and forgot us.
            self.start_discovery()