from abc import ABC, abstractmethod

GRAVITY = 9.81


#Registry
_battery_registry = {}

def register_battery(name: str):
    """Decorator to register a BatteryHandler subclass.

    Args:
        name (str): Name used in YAML configs (e.g. "anc_only", "twofriction")

    Returns:
        Callable: Class decorator that registers and returns the class unchanged.
    """

    # Normalize registry key to ensure consistency with lookup
    normalized_name = name.lower()

    def decorator(cls):
        # Prevent accidental overrides when the same (normalized) name is
        # registered for multiple different handler classes.
        existing = _battery_registry.get(normalized_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Battery handler '{normalized_name}' is already registered "
                f"for class {existing.__name__}"
            )
        _battery_registry[normalized_name] = cls
        return cls

    return decorator


class BatteryModel(ABC):
    def __init__(self, name : str,  init: float, capacity: float, anc_drain: float,
                 scale: float = 1.0, **kwargs) -> None:
        self.name = name
        self.status = float(init)
        self.capacity_joules = float(capacity)
        self.joules_per_percent = max(float(capacity) / 100.0, 1e-9)
        self.anc_drain = float(anc_drain)
        self.scale = float(scale)

    # ── the two hooks a subclass defines ──────────────────────────────────
    @abstractmethod
    def joules_per_meter(self, mass: float, friction: float) -> float:
        """Translation cost per metre travelled [J/m]."""

    def joules_per_second(self) -> float:
        """Standing/ancillary cost [W]."""
        return self.anc_drain

    def turn_joules(self, w: float, dt: float) -> float:
        """Rotation cost for one step [J]. Default: rotation is free."""
        return 0.0

    # ── everything else is shared ─────────────────────────────────────────
    def predict_joules(self, distance: float, duration: float,
                       mass: float, friction: float) -> float:
        return (
            self.joules_per_meter(mass, friction) * distance
            + self.joules_per_second() * duration
        ) * self.scale

    def to_percent(self, joules: float) -> float:
        return joules / self.joules_per_percent

    def step(self, dt: float, v: float, w: float,
             mass: float, friction: float) -> float:
        """Drain one sim step. Returns the drain in % (all models, always)."""
        joules = self.predict_joules(abs(v) * dt, dt, mass, friction) \
                 + self.turn_joules(w, dt) * self.scale
        drain = self.to_percent(joules)
        self.status = max(0.0, self.status - drain)
        return drain



@register_battery("two_friction")
class TwoFrictionBattery(BatteryModel):
    def joules_per_meter(self, mass: float, friction: float) -> float:
        return 2.0 * friction * mass * GRAVITY

    def turn_joules(self, w: float, dt: float) -> float:
        return 0.01 * abs(w)


@register_battery("anc_only")
class AncillaryOnlyBattery(BatteryModel):
    def joules_per_meter(self, mass: float, friction: float) -> float:
        return 0.0


@register_battery("dummy")
class DummyBatteryModel(BatteryModel):
    def joules_per_meter(self, mass: float, friction: float) -> float:
        return 0.0

    def joules_per_second(self) -> float:
        return 0.0



class BatteryFactory:
    """
    Factory class to create BatteryModels.
    """

    @staticmethod
    def create_battery(name: str = "anc_only", **kwargs) -> BatteryModel:
        name = name.lower()

        cls = _battery_registry.get(name)
        if cls is not None:
            return cls(name, **kwargs)
        raise ValueError(f"Unknown battery type: {name}")

    @staticmethod
    def get_handler_class(name: str) -> type[BatteryModel] | None:
        """Look up a registered handler class by name without instantiation"""
        return _battery_registry.get(name.lower() if name else "")

    @staticmethod
    def available_shapes() -> dict[str, type[BatteryModel]]:
        """Read-only view of the registry: ``{name: handler_class}``."""
        return dict(_battery_registry)
