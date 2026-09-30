"""
shadow twin for a single agent
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Any

import numpy as np

from instance_twin.agent_profile import AgentProfile, DigitalTwinConfigError
from core_msgs.utils.dispatch import handles, MessageDispatcher
from core_msgs.global_msgs.global_payloads import AgentDiscoveryMessage
from core_msgs.topic_contract import MessageType
from core_msgs.instance_aggregate import payloads
from core_msgs.instance_aggregate.handshake import (
    HandshakeAction, HandshakeEnvelope, HandshakeResponder,
)
from core_msgs.instance_aggregate.mission import Mission, MissionType, MissionStatus
from core_msgs.instance_aggregate.mission_handshake import MissionBidding, MissionPlanHint, MissionResponder
from core_msgs.instance_agent.controll_payloads import InitialCommandMessage
from instance_twin.instance_comms import INSTANCE_DISCOVERY


class InstanceTwin(MessageDispatcher):
    """One container = one twin = at most one agent."""

    def __init__(
        self,
        name: str = "InstanceTwin",
        namespace: str = "default_ns",
        loop_freq: float = 10.0,
        drive_agent: bool = False,
        goal_threshold: float = 0.25,
        obstacle_ttl: float = 5.0,
    ) -> None:

        self.transport = None
        self.logger = logging.getLogger(name)

        self.name = name
        self.namespace = namespace
        self.aggregate_name = None # name of aggregate twin of network

        self.drive_agent = drive_agent

        self.loop_freq = loop_freq
        self.dt = 1.0 / loop_freq

        self.goal_threshold = goal_threshold
        self.obstacle_ttl = obstacle_ttl

        # --- binding state ----
        self.agent: AgentProfile | None = None

        self.active_mission: Mission | None = None
        self.mission_responder: MissionResponder | None = None

        self.arrive_flag = self.stop_flag = self.battery_depleted = False
        self.sim_time = 0.0
        self.trajectory: list[np.ndarray] = []
        self.pending_obstacles: list[Any] = []

        self.known_obstacles: dict[str, Any] = {}

        self.inbox: queue.Queue = queue.Queue()   # (topic, msg, monotonic arrival time)
        self._link_countdown = 0
        self._complete_countdown = 0   # retransmit COMPLETE until COMPLETE_ACK

        self.inst_responder = HandshakeResponder(self.name, bid_fn=self._estimate_link_bid,
                                                 max_reserved=1)

        self.logger.info("[%s] UNBOUND, waiting on instantiate (ns=%s)", name, namespace)

    def setup_transport(self, transport) -> None:
        self.transport = transport


    def handle_instantiate(self, env: HandshakeEnvelope) -> None:
        """Handled inline (not queued): the timer isn't running when unbound."""


        result = self.inst_responder.handle(env)

        message_type = MessageType.ACTIVATE if INSTANCE_DISCOVERY else MessageType.INSTANTIATE

        # First check if linking doesnt fail to before ack back
        if result.action is HandshakeAction.LINK_SUBJECT:
            if not self._bind(result.subject , result.payload):
                self.transport.publish_to_aggregate(message_type,
                    self.inst_responder.get_revoked(result.subject))
                return

        if result.reply:
            self.transport.publish_to_aggregate(message_type ,result.reply)

        if result.action is HandshakeAction.RELEASE_SUBJECT:
            self._unbind()


    # --- binding ---

    def _bind(self, agent_name: str, discovery: AgentDiscoveryMessage) -> bool:
        """Build the shadow model from the spec the aggregate resolved.
        Returns False if the spec is unusable, so the caller can revoke.
        """
        print("DISCOVERY-CHECKK")
        print(discovery)
        try:
            self.agent = AgentProfile(agent_name, discovery)
        except (DigitalTwinConfigError, ValueError, TypeError):
            # Unknown estimator / sensor / bad params included: revoke instead of half-binding
            self.agent = None
            self.logger.exception("refusing to bind %s", agent_name)
            return False

        self.agent.goal_threshold = self.goal_threshold

        self.trajectory = []
        self.sim_time = 0.0
        self.active_mission = None
        self.arrive_flag = self.stop_flag = False
        self._link_countdown = 0
        self._complete_countdown = 0
        self.known_obstacles = {}
        self.pending_obstacles = []

        # Set mission responder before subscribing to missions. This one bids, and may
        # hold offers on several missions at once.
        self.mission_responder = HandshakeResponder(self.name, bid_fn=self._estimate_mission_bid)

        self.transport.subscribe_agent(self.agent.name, self.agent.interface)

        # Notify the agent immediately
        self._send_link()

        self.transport.start_stepping()

        # From here the aggregate reads our liveness from twin state, so the separate
        # heartbeat would only double-count us as a free instance.
        self.transport.pause_heartbeat()

        self.logger.info(
            "bound %s via %s: kinematics=%s shape=%s r=%.2f drive=%s estimator=%s channels=%s",
            agent_name,
            type(self.agent.interface).__name__,
            self.agent.kinematics.__class__.__name__,
            self.agent.shape,
            self.agent.radius,
            self.drive_agent,
            self.agent.estimator.__class__.__name__,
            self.agent.interface.topic_dict(),
        )
        return True

    def _unbind(self) -> None:
        self.transport.stop_stepping()

        # A real robot keeps its last command until its own watchdog trips, so stop it explicitly
        if self.agent is not None and self.drive_agent:
            self.agent.velocity = np.zeros_like(self.agent.velocity)
            self._send_action()

        self.transport.unsubscribe_agent()

        released = self.agent.name if self.agent else "unknown"
        self.agent = None
        self.mission_responder = None
        self.active_mission = None
        self.trajectory = []
        self.pending_obstacles = []
        self.known_obstacles = {}
        self._complete_countdown = 0

        with self.inbox.mutex:
            self.inbox.queue.clear()

        # Back to being a candidate: no twin state flows any more, so the heartbeat
        # (and discovery, if we were never registered) has to come back.
        self.transport.resume_heartbeat()

        self.logger.info("released %s -- back to UNBOUND", released)

    # ---- Callbacks ---- (queued; agent sensor traffic is routed by the agent itself, see _drain_inbox)

    @handles(MessageType.MISSION)
    def _handle_mission(self, env: HandshakeEnvelope) -> None:
        if not isinstance(env, HandshakeEnvelope):
            return

        # arrived after a release, or our own echo
        if self.agent is None or self.mission_responder is None:
            return

        result = self.mission_responder.handle(env)

        # Bidding is handled inside the responder, only the reply needs to be sent.
        if result.reply:
            self.transport.publish_to_aggregate(MessageType.MISSION, result.reply)

        if result.action is HandshakeAction.LINK_SUBJECT:
            mission = result.payload
            if mission is None:
                self.logger.error("awarded mission %s without a payload", result.subject)
                self.transport.publish_to_aggregate(
                    MessageType.MISSION, self.mission_responder.get_revoked(result.subject))
                return
            # The path is what makes this a route instead of a straight line. The award
            # does not have to repeat it, so the responder hands back whatever it kept.
            if result.hint is not None and getattr(result.hint, "path", None):
                mission.set_path(result.hint.path)
            self.active_mission = mission
            self.active_mission.mission_status = MissionStatus.ACTIVE
            self._complete_countdown = 0
            self.logger.info("mission %s ACTIVE", result.subject)

        elif result.action is HandshakeAction.RELEASE_SUBJECT:
            # Covers both a CANCEL and the COMPLETE_ACK that closes a finished mission.
            self.active_mission = None
            self._complete_countdown = 0
            self.logger.info("mission %s CLEARED", result.subject)

    # --- step ----

    def step(self) -> np.ndarray | None:
        try:
            return self._step()
        except Exception:
            self.logger.exception("step failed at sim_time=%.2f", self.sim_time)
            return None

    def _step(self) -> np.ndarray | None:
        self._drain_inbox()

        if not self.agent:
            return None

        # For waiting on the initial pose message
        if not self.agent.linked:
            self._link_countdown -= 1 #set by _send_link
            if self._link_countdown <= 0:
                self._send_link()

        self._forget_stale_obstacles()
        self._retry_completion()

        self.agent.goal = self._current_goal()
        self.agent.step_physics(self.dt, list(self.known_obstacles.values()), now=time.monotonic(),
                                halt=self.stop_flag or self.agent.battery_depleted)

        self.sim_time += self.dt

        self._check_arrival(self.agent.goal)
        self.trajectory.append(self.agent.state.copy())

        self._publish_state()
        self._publish_obstacles()

        if self.drive_agent:
            self._send_action()

        return self.agent.state

    def _forget_stale_obstacles(self) -> None:
        """An obstacle that left the field of view has to stop blocking the
        planner eventually, or the map fills with things that aren't there."""
        if self.obstacle_ttl <= 0:
            return
        now = time.time()
        for oid in [k for k, o in self.known_obstacles.items()
                    if now - o.timestamp > self.obstacle_ttl]:
            del self.known_obstacles[oid]

    def _drain_inbox(self, budget: int = 512) -> None:
        """Twin topics go to their handler; everything else is agent traffic, routed by the agent."""
        for _ in range(budget):
            try:
                topic, msg, received = self.inbox.get_nowait()
            except queue.Empty:
                return

            handler_name = self._DISPATCH.get(topic)
            try:
                if handler_name is not None:
                    getattr(self, handler_name)(msg)
                elif self.agent is not None:
                    self._ingest_agent(topic, msg, received)
            except Exception:
                self.logger.exception("handling %s failed", topic)

    def _ingest_agent(self, topic: MessageType, msg: Any, received: float) -> None:
        """Sensors turn the message into estimate updates and obstacles; the first one confirms the link."""
        if not self.agent.linked:
            self.logger.info("link confirmed by first %s from %s", topic.value, self.agent.name)
        for obs in self.agent.ingest(topic, msg, received):
            self.known_obstacles[obs.id] = obs
            self.pending_obstacles.append(obs)

    # --- Agent link ----

    def _send_link(self) -> None:
        """Tell the agent who owns it; bridge robots stream anyway, so their first pose is the ack."""
        if not self.agent:
            return
        msg = InitialCommandMessage(instance_name=self.name, agent_name=self.agent.name)
        self.transport.publish_to_agent(MessageType.ACTION, msg)
        self._link_countdown = int(self.loop_freq * 2)  # retry every ~2s
        self.logger.debug("link sent to: %s", self.agent.name)

    def _send_action(self) -> None:
        """Current action as a canonical MotionCommand, the interface turns it into the agent's format."""
        self.transport.publish_to_agent(MessageType.ACTION, self.agent.motion_command())

    # -- mission state ---

    def _current_goal(self) -> np.ndarray | None:
        m = self.active_mission
        if m is None:
            return None
        if m.mission_type == MissionType.TIME_GATED_GOTO and self.sim_time < m.unlock_time:
            return None
        xy = m.next_goal(ugv_pos=(float(self.agent.state[0, 0]), float(self.agent.state[1, 0])))
        return np.array(xy, dtype=float).reshape(-1, 1) if xy is not None else None

    def _check_arrival(self, goal: np.ndarray | None) -> None:
        self.arrive_flag = False
        if goal is None or self.active_mission is None or not self.agent:
            return
        if float(np.linalg.norm(self.agent.state[:2] - goal[:2])) >= self.goal_threshold:
            return

        self.arrive_flag = True
        m = self.active_mission
        if m.mission_type == MissionType.COVERAGE_PATROL:
            m.advance_patrol()
            return

        if m.mission_status != MissionStatus.COMPLETE:
            m.mission_status = MissionStatus.COMPLETE
            self.logger.info("mission %s COMPLETE", m.mission_id)
            self._send_completion(m.mission_id)

    def _send_completion(self, mission_id: str) -> None:
        env = self.mission_responder.get_completed(mission_id) if self.mission_responder else None
        if env is None:
            return
        self.transport.publish_to_aggregate(MessageType.MISSION, env)
        self._complete_countdown = int(self.loop_freq * 2)   # resend every ~2s

    def _retry_completion(self) -> None:
        """Resend mission complete"""
        if not self._complete_countdown:
            return
        self._complete_countdown -= 1
        if self._complete_countdown <= 0 and self.active_mission is not None:
            self._send_completion(self.active_mission.mission_id)


    # --- Outbound payloads ----

    def _publish_state(self) -> None:
        if not self.agent:
            return

        mission_hash = self.active_mission.mission_id if self.active_mission else None

        self.transport.publish_to_aggregate(
            MessageType.TWIN_STATE,
            payloads.TwinStatePayload(
                name=self.agent.name,
                state=tuple(self.agent.state.flatten().tolist()),
                velocity=tuple(self.agent.velocity.flatten().tolist()),
                battery_pct=float(self.agent.battery.status) if self.agent.battery else 100.0,
                arrive_flag=self.arrive_flag,
                sim_time=self.sim_time,
                active_mission_id=mission_hash,
            ),
        )

    def _publish_obstacles(self) -> None:
        for obs in self.pending_obstacles:
            self.transport.publish_to_aggregate(MessageType.OBSTACLE, obs)
        self.pending_obstacles = []

    def _estimate_link_bid(self, hint: Any) -> dict | None:
        """Bidding for the agent lining"""
        if self.agent is not None:
            return None                     # already owns an agent
        return {"instance": self.name, "load": 0.0}

    def _estimate_mission_bid(self, hint: MissionPlanHint) -> MissionBidding | None:
        """Bidding for the mission, None is refuse"""
        if self.agent is None or hint is None or hint.distance is None:
            return None
        bid = self.agent.estimate_traversal(distance=hint.distance)
        if bid is None or bid.time_bidding is None:
            return None
        return bid