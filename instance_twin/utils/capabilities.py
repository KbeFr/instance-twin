"""
instance_twin/capabilities.py

Overview of everything an agent spec can configure, read straight off the registries:
per entry its description, fixed specs, and parameters (* = required).
    python -m instance_twin.capabilities
"""
from __future__ import annotations

import inspect
from typing import Any

from core_msgs.agents_contract import AgentKind
from instance_twin.battery.battery_handler import BatteryFactory
from instance_twin.estimation.estimator_handler import EstimatorFactory
from instance_twin.interfaces.interface_handler import AgentInterfaceFactory
from instance_twin.interfaces.ros_interfaces import RosBridgeInterface
from instance_twin.interfaces.ros.ros_topic_mapping import ROS_TOPIC_MAPPING
from instance_twin.irsim_borrowed.controller.controller_handler import ControllerFactory
from instance_twin.irsim_borrowed.geometry.geometryhandler import GeometryFactory
from instance_twin.irsim_borrowed.kinematics.kinematics_handler import AckermannKinematics, KinematicsFactory
from instance_twin.sensors.perception_sensors import PerceptionSensorFactory
from instance_twin.sensors.state_sensors import StateSensorFactory

# Filled in by the twin itself, never by config
_SKIP = {"self", "name", "kinematics", "role", "agent_name", "topic_channels"}
# Class-level facts worth showing next to the parameters
_SPECS = ("state_dim", "action_dim", "vel_min", "vel_max", "default_topic", "components")


def params(fn) -> dict[str, str]:
    """Config parameters of a callable; required ones end in *."""
    return {p.name: f"{p.name}*" if p.default is p.empty else f"{p.name}={p.default!r}"
            for p in inspect.signature(fn).parameters.values()
            if p.name not in _SKIP and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}


def init_params(cls: type) -> dict[str, str]:
    """__init__ parameters up the class chain, as far as each level passes **kwargs on."""
    out: dict[str, str] = {}
    for klass in cls.__mro__[:-1]:
        init = vars(klass).get("__init__")
        if init is None:
            continue
        for key, text in params(init).items():
            out.setdefault(key, text)
        if not any(p.kind is p.VAR_KEYWORD for p in inspect.signature(init).parameters.values()):
            break
    return out


def kinematics_params(cls: type) -> dict[str, str]:
    """create_kinematics owns the parameter list; mode and wheelbase only reach Ackermann."""
    out = params(KinematicsFactory.create_kinematics)
    if not issubclass(cls, AckermannKinematics):
        out.pop("mode", None)
        out.pop("wheelbase", None)
    return out


def specs(cls: type) -> dict[str, Any]:
    """Dimensions, velocity limits, the topic a sensor listens on and what it measures."""
    out: dict[str, Any] = {}
    for key in _SPECS:
        value = getattr(cls, key, None)
        if value is not None:
            out[key.replace("default_", "")] = sorted(value) if isinstance(value, frozenset) else getattr(value, "value", value)
    return out


def interface_specs(cls: type) -> dict[str, Any]:
    """The ROS type each topic travels as; the system's own topics always come as core_msgs."""
    if not issubclass(cls, RosBridgeInterface):
        return {"wire": "core_msgs on every contract topic"}
    mapping = {**ROS_TOPIC_MAPPING, **cls.ros_types}
    return {"distro": cls.ros_distro,
            "ros_types": {topic.value: ch.ros_type for topic, ch in mapping.items()},
            }


# spec key -> (registry, parameter reader, spec reader)
SECTIONS = {
    "kinematics": (KinematicsFactory.available_kinematics, kinematics_params, specs),
    "shape": (GeometryFactory.available_shapes, lambda cls: params(cls.construct_original_geometry), specs),
    "controller": (ControllerFactory.available_controllers, init_params, specs),
    "estimator": (EstimatorFactory.available_estimators, init_params, specs),
    "battery": (BatteryFactory.available_batteries, init_params, specs),
    "perception_sensors": (PerceptionSensorFactory.available_sensors, init_params, specs),
    "state_sensors": (StateSensorFactory.available_sensors, init_params, specs),
    "agent_type": (AgentInterfaceFactory.available_interfaces, init_params, interface_specs),  # params go under interface:
}


def capabilities() -> dict[str, dict[str, dict[str, Any]]]:
    """{spec key: {entry: {"about", "specs", "params"}}}, JSON friendly for a GUI."""
    out = {key: {name: {"about": (cls.__doc__ or "").strip().split("\n")[0],
                        "specs": spec_reader(cls),
                        "params": list(param_reader(cls).values())}
                 for name, cls in sorted(registry().items())}
           for key, (registry, param_reader, spec_reader) in SECTIONS.items()}
    out["kind"] = {kind.value: {"about": "", "specs": {}, "params": []} for kind in AgentKind}
    return out


if __name__ == "__main__":
    for key, entries in capabilities().items():
        print(f"{key}:")
        for name, entry in entries.items():
            print(f"  {name.upper()} : " + (f"  - {entry['about']}" if entry["about"] else ""))
            inline = {k: v for k, v in entry["specs"].items() if not isinstance(v, dict)}
            if inline:
                print("      specs  :   " + "  ".join(f"{k}={v}" for k, v in inline.items()))
            for k, nested in entry["specs"].items():
                if isinstance(nested, dict):
                    print(f"      {k}")
                    for topic, ros_type in nested.items():
                        print(f"        {topic:<18} {ros_type}")
            if entry["params"]:
                print("      params :  " + "  ".join(entry["params"]))
        print()