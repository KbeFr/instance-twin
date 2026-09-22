from abc import ABC, abstractmethod
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

#Registry
_controller_registry = {}

def register_controller(name: str):
    """Decorator to register a ControllerHandler subclass.

    Args:
        name (str): Name used in YAML configs (e.g. "c3bf", "cbf")

    Returns:
        Callable: Class decorator that registers and returns the class unchanged.
    """

    # Normalize registry key to ensure consistency with lookup
    normalized_name = name.lower()

    def decorator(cls):
        # Prevent accidental overrides when the same (normalized) name is
        # registered for multiple different handler classes.
        existing = _controller_registry.get(normalized_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Controller handler '{normalized_name}' is already registered "
                f"for class {existing.__name__}"
            )
        _controller_registry[normalized_name] = cls
        return cls

    return decorator


def obstacle_clearance(anchor_xy: np.ndarray, obstacle) -> tuple[np.ndarray, float]:
    """The (point, residual_radius) a CBF should measure clearance against.
    """
    shape = getattr(obstacle, "shape", "circle")
    geometry = getattr(obstacle, "geometry", None)
    if shape == "circle" or geometry is None:
        return obstacle.position[:, 0], float(getattr(obstacle, "radius", 0.0))

    anchor_point = Point(float(anchor_xy[0]), float(anchor_xy[1]))
    _, closest_surface_point = nearest_points(anchor_point, geometry)
    return np.array([closest_surface_point.x, closest_surface_point.y], dtype=float), 0.0


class ControllerModel(ABC):
    def __init__(self):
        pass    #TODO check to have arguments for fault detection

    @abstractmethod
    def get_action(self, robot, obstacles) -> np.ndarray:
        pass

class ControllerFactory:

    @staticmethod
    def create_controller(name: str = "cbf", **kwargs) -> ControllerModel:
        name = name.lower()
        cls = _controller_registry.get(name) if name else None
        if cls is None:
            raise ValueError(f"Unknown controller type: {name}")
        # Controller __init__ signatures don't accept 'name'
        return cls(**kwargs)

    @staticmethod
    def get_handler_class(name: str) -> type[ControllerModel] | None:
        """Look up a registered handler class by name without instantiation.

        Args:
            name (str): Controller name (e.g. ``"c3bf"``, ``"cbf"``).

        Returns:
            type[ControllerModel] | None: The class, or ``None`` if not found.
        """
        return _controller_registry.get(name.lower() if name else "")

    @staticmethod
    def available_controllers() -> dict[str, type[ControllerModel]]:
        """Read-only view of the registry: ``{name: handler_class}``.
        """
        return dict(_controller_registry)


# avoiding a circular-import error guarantees the registry is populated.
from instance_twin.irsim_borrowed.controller import cbf_qp, c3bf_qp, local_planners  # noqa: E402,F401