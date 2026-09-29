"""
CDR codec for the flexia ROS2 bridge envelopes.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from rosbags.typesys import Stores, get_typestore, get_types_from_msg

CDR_ENCODING = "ros2_cdr_base64"

_STORES = {
    "humble": Stores.ROS2_HUMBLE,
    "jazzy": Stores.ROS2_JAZZY,
}


class RosCodecError(ValueError):
    """Raised when a bridge payload cannot be decoded."""


@dataclass
class RosSerializedMessage:
    """What the bridge carries in _payload, both directions."""
    data: str
    encoding: str = CDR_ENCODING


class RosCodec:
    """Typed CDR (de)serialization from .msg definitions, no handwritten message classes."""

    def __init__(self, distro: str = "humble", extra_msgs: dict[str, str] | None = None):
        self._store = get_typestore(_STORES[distro.lower()])

        for ros_type, definition in (extra_msgs or {}).items():
            self._store.register(get_types_from_msg(definition, ros_type))

    def type(self, ros_type: str) -> type:
        """Message class for a ros type, e.g. geometry_msgs/msg/Twist."""
        return self._store.types[ros_type]

    def decode(self, payload: Any, ros_type: str) -> Any:
        """Full envelope, bare _payload, str or dict -> ROS message."""
        body = self._payload_of(payload)
        if body.get("encoding") != CDR_ENCODING:
            raise RosCodecError(f"unsupported encoding {body.get('encoding')!r}")
        raw = base64.b64decode(body.get("data", ""))
        return self._store.deserialize_cdr(raw, ros_type)

    def encode(self, ros_msg: Any, ros_type: str) -> RosSerializedMessage:
        raw = bytes(self._store.serialize_cdr(ros_msg, ros_type))
        return RosSerializedMessage(data=base64.b64encode(raw).decode("ascii"))

    @staticmethod
    def _payload_of(payload: Any) -> dict:
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError as ex:
                raise RosCodecError(f"not json: {ex}") from ex
        if isinstance(payload, RosSerializedMessage):
            return {"encoding": payload.encoding, "data": payload.data}
        if not isinstance(payload, dict):
            raise RosCodecError(f"unexpected payload {type(payload).__name__}")
        if isinstance(payload.get("_payload"), dict):
            return payload["_payload"]
        return payload
