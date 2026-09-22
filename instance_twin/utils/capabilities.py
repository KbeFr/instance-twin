"""base/irsim_borrowed/capabilities.py

Single source of truth for "what can this build of the twin actually be
configured with" -- kinematics, shape/geometry, and controller names, plus
whatever AgentKind values core_msgs.agents_contract happens to define.

Every name and parameter schema below is read directly off the live
registries in kinematics_handler.py / geometry_handler.py /
controller_handler.py (via their `available_*()` accessors) and, for
per-type parameters, off the actual callable each factory hands kwargs to
-- via `inspect.signature`, not a hand-copied list. Add a new
`@register_kinematics(...)`/`@register_geometry(...)`/`@register_controller(...)`
class anywhere in this package and it shows up here automatically, with
its real parameters, with no second place to remember to update.

This module only knows about the *twin's* own vendored implementations.
It says nothing about what the separate *simulation* container's real
`irsim` package can do -- that lives in that container's
irsim_capabilities.py. The two can genuinely differ (see that module's
docstring), and collapsing them into one list would misrepresent both.
"""
from __future__ import annotations

import inspect
from typing import Any

from instance_twin.irsim_borrowed.controller.controller_handler import ControllerFactory
from instance_twin.irsim_borrowed.geometry.geometryhandler import GeometryFactory
from instance_twin.irsim_borrowed.kinematics.kinematics_handler import (
    AckermannKinematics,
    KinematicsFactory,
)
from irsim_specific.sensors.sensor_factory import SensorFactory


class UnsupportedAgentConfig(ValueError):
    """Raised when a resolved agent config (from a DiscoveryMessage, a
    fallback default_agent_config.yaml entry, or a mix of both) names a
    kinematics/shape/controller/kind this build doesn't have registered,
    is missing a parameter that type requires, or sets a parameter that
    type doesn't recognize.

    Carries every problem found at once (`.problems`), not just the
    first, so a rejection or log line can say everything that's wrong in
    one shot instead of a frustrating fix-one-see-the-next loop.
    """

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

_EMPTY = inspect.Parameter.empty


def _inspect(fn: Any) -> tuple[dict[str, Any], bool]:
    """``({param_name: default}, accepts_extra_kwargs)`` for `fn`.

    Skips ``self``/``name``/``*args``. A default of `_EMPTY` means the
    parameter is required. ``accepts_extra_kwargs`` is True if `fn` takes
    a ``**kwargs`` catch-all (e.g. PolygonGeometry's random-shape
    generator options) -- callers use that to know an unrecognized key
    isn't necessarily a mistake.
    """
    params: dict[str, Any] = {}
    accepts_extra = False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return params, accepts_extra
    for pname, p in sig.parameters.items():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_extra = True
            continue
        if pname in ("self", "name") or p.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        params[pname] = p.default
    return params, accepts_extra


def _entry(raw: dict[str, Any], accepts_extra: bool) -> dict[str, Any]:
    return {
        "params": {k: (None if v is _EMPTY else v) for k, v in raw.items()},
        "required": [k for k, v in raw.items() if v is _EMPTY],
        "accepts_extra": accepts_extra,
    }


def _describe(registry: dict[str, type], ctor) -> dict[str, dict[str, Any]]:
    """``{name: entry}`` for every class in `registry`, reading parameters
    off ``ctor(cls)`` -- the callable that actually receives that class's
    config kwargs (see the call sites below; this differs by registry
    because the three factories don't all wire config the same way)."""
    out: dict[str, dict[str, Any]] = {}
    for reg_name, cls in registry.items():
        raw, accepts_extra = _inspect(ctor(cls))
        out[reg_name] = _entry(raw, accepts_extra)
    return out


def describe_kinematics() -> dict[str, dict[str, Any]]:
    """Unlike shapes/controllers, every kinematics name is built through
    one shared entry point -- ``KinematicsFactory.create_kinematics`` --
    which owns a fixed parameter list and manually forwards the relevant
    ones into each handler class (see its source: non-Ackermann types
    only ever get ``name, noise, alpha``; Ackermann additionally gets
    ``mode, wheelbase``). So the true config surface is that one
    signature, not each handler class's own ``__init__`` -- which the
    factory always satisfies itself and which config-driven code
    (AgentProfile.build) never calls directly.
    """
    raw, accepts_extra = _inspect(KinematicsFactory.create_kinematics)
    raw.pop("role", None)  # caller context (used for a log message), not user config

    out: dict[str, dict[str, Any]] = {}
    for kind_name, cls in KinematicsFactory.available_kinematics().items():
        applicable = dict(raw)
        if not issubclass(cls, AckermannKinematics):
            applicable.pop("mode", None)
            applicable.pop("wheelbase", None)
        out[kind_name] = _entry(applicable, accepts_extra)
    return out


def describe_shapes() -> dict[str, dict[str, Any]]:
    # geometry_handler.__init__ is the generic (self, name, **kwargs) --
    # kwargs flow straight through to construct_original_geometry, which
    # is where each shape's real parameters are declared.
    return _describe(
        GeometryFactory.available_shapes(), lambda cls: cls.construct_original_geometry
    )


def describe_controllers() -> dict[str, dict[str, Any]]:
    # ControllerFactory.create_controller splats config straight into
    # cls(**kwargs), so each controller's own __init__ is the real schema.
    return _describe(ControllerFactory.available_controllers(), lambda cls: cls.__init__)

def describe_sensors() -> dict[str, dict[str, Any]]:
    return _describe(SensorFactory.available_perception_sensors(), lambda cls: cls.__init__)


def list_kinematics() -> list[str]:
    return sorted(KinematicsFactory.available_kinematics())


def list_shapes() -> list[str]:
    return sorted(GeometryFactory.available_shapes())


def list_controllers() -> list[str]:
    return sorted(ControllerFactory.available_controllers())


def list_agent_kinds() -> list[str]:
    """Best-effort: AgentKind lives in core_msgs, not this package, and is
    deliberately not hard-coded here. If it's importable and behaves like
    a normal Enum, read its members straight off it; if that fails for
    any reason, report none (validate_kind then skips the check) rather
    than guess at what the valid values might be.
    """
    try:
        from core_msgs.agents_contract import AgentKind
        return [getattr(k, "value", str(k)) for k in AgentKind]
    except Exception:
        return []


def capabilities_manifest() -> dict[str, Any]:
    """Everything a GUI or validator needs, in one JSON-serializable dict."""
    return {
        "kinematics": describe_kinematics(),
        "shapes": describe_shapes(),
        "controllers": describe_controllers(),
        "sensors" : describe_sensors(),
        "agent_kinds": list_agent_kinds(),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_named_config(
    label: str, config: dict[str, Any] | None, described: dict[str, dict[str, Any]]
) -> str | None:
    config = config or {}
    name = config.get("name")
    if not name:
        return None  # absent entirely -- caller already has its own default for this

    entry = described.get(str(name).lower())
    if entry is None:
        return f"{label} '{name}' is not supported (known: {', '.join(sorted(described))})"

    missing = [p for p in entry["required"] if config.get(p) is None]
    if missing:
        return f"{label} '{name}' is missing required parameter(s): {', '.join(missing)}"

    if not entry["accepts_extra"]:
        allowed = set(entry["params"]) | {"name"}
        extra = sorted(set(config) - allowed)
        if extra:
            return (
                f"{label} '{name}' has unrecognized parameter(s): {', '.join(extra)} "
                f"(accepted: {', '.join(sorted(entry['params'])) or 'none'})"
            )
    return None


def validate_kinematics_config(config: dict[str, Any] | None) -> str | None:
    return _validate_named_config("kinematics", config, describe_kinematics())


def validate_shape_config(config: dict[str, Any] | None) -> str | None:
    return _validate_named_config("shape", config, describe_shapes())


def validate_controller_config(config: dict[str, Any] | None) -> str | None:
    return _validate_named_config("controller", config, describe_controllers())


def validate_kind(value: Any) -> str | None:
    known = list_agent_kinds()
    if not known:
        return None  # AgentKind unavailable/not introspectable -- don't block on it
    val = getattr(value, "value", value)
    if val not in known and value not in known:
        return f"agent kind '{value}' is not supported (known: {', '.join(known)})"
    return None


def validate_discovery(
    *,
    kinematics_config: dict[str, Any] | None,
    shape_config: dict[str, Any] | None,
    controller_config: dict[str, Any] | None,
    kind_value: Any,
) -> list[str]:
    """Collect every problem with a resolved agent config in one pass,
    instead of stopping at the first (or, worse, silently downgrading
    each field to a default one at a time -- see AgentProfile.build)."""
    checks = (
        validate_kinematics_config(kinematics_config),
        validate_shape_config(shape_config),
        validate_controller_config(controller_config),
        validate_kind(kind_value),
    )
    return [problem for problem in checks if problem]


if __name__ == "__main__":
    print(capabilities_manifest())
