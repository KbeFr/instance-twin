from abc import ABC, abstractmethod

import numpy as np
from shapely import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    bounds,
    envelope,
    is_valid,
    make_valid,
    minimum_bounding_radius,
)
from shapely.ops import transform

from instance_twin.irsim_borrowed.geometry.generation import random_generate_polygon
from instance_twin.irsim_borrowed.util.random import rng
from instance_twin.irsim_borrowed.util.util import (
    gen_inequal_from_vertex,
    geometry_transform,
    get_transform,
    is_convex_and_ordered,
)

# Registry was not done in irsim**
_geometry_registry: dict[str, type["GeometryHandler"]] = {}


def register_geometry(name: str):
    """Decorator to register a geometry_handler subclass.

    Args:
        name (str): Name used in YAML configs (e.g. ``"circle"``, ``"rectangle"``).

    Returns:
        Callable: Class decorator that registers and returns the class unchanged.
    """
    normalized_name = name.lower()

    def decorator(cls):
        existing = _geometry_registry.get(normalized_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Geometry handler '{normalized_name}' is already registered "
                f"for class {existing.__name__}"
            )
        _geometry_registry[normalized_name] = cls
        return cls

    return decorator


class GeometryHandler(ABC):
    """
    This class is used to handle the geometry of the object. It reads the shape parameters from yaml file and constructs the geometry of the object.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._original_geometry = self.construct_original_geometry(**kwargs)
        self.geometry = self._original_geometry
        self.wheelbase = kwargs.get("wheelbase")
        self.length, self.width = self.cal_length_width(self._original_geometry)

    @abstractmethod
    def construct_original_geometry(self, **kwargs):
        """
        Construct the original geometry of the object when the state is in the origin of coordinates.

        Args:
            **kwargs: shape parameters

        Returns: Geometry of the object
        """
        pass

    def step(self, state):
        """
        Transform geometry to the new state.

        Args:
            state (np.ndarray): State vector [x, y, theta].

        Returns:
            shapely.geometry.base.BaseGeometry: Transformed geometry.
        """

        self.geometry = geometry_transform(self._original_geometry, state)

        return self.geometry

    def get_init_Gh(self):
        """
        Generate initial G and h for convex object.

        Returns:
            tuple[np.ndarray, np.ndarray, str | None, bool | None]:
                - G matrix: (N, 2)
                - h vector: (N, 1)
                - cone_type (str): "norm2" for circle or "Rpositive" for polygon
                - convex_flag (bool): whether convex constraints are valid
        """

        if self.name == "circle":
            G = np.array([[1, 0], [0, 1], [0, 0]])
            h = np.array([[0], [0], [-self.radius]])
            cone_type = "norm2"
            convex_flag = True

        elif self.name == "polygon" or self.name == "rectangle":
            convex_flag, _ = is_convex_and_ordered(self.original_vertices)

            if convex_flag:
                G, h = gen_inequal_from_vertex(self.original_vertices)
                cone_type = "Rpositive"
            else:
                G, h, cone_type = None, None, None
        else:
            G, h, cone_type, convex_flag = None, None, None, None

        return G, h, cone_type, convex_flag

    def get_Gh(self, **kwargs):
        if self.name == "polygon" or self.name == "rectangle":
            G, h, cone_type, convex_flag = self.get_polygon_Gh(kwargs.get("vertices"))

        elif self.name == "circle":
            G, h, cone_type, convex_flag = self.get_circle_Gh(
                kwargs.get("center"), kwargs.get("radius")
            )

        return G, h, cone_type, convex_flag

    def get_polygon_Gh(self, vertices: np.ndarray | None = None):
        """
        Generate G and h for convex polygon.

        Args:
            vertices: (2, N), N: Edge number of the object

        Returns:
            tuple[np.ndarray | None, np.ndarray | None, str | None, bool]:
                - G matrix: (N, 2)
                - h vector: (N, 1)
                - cone_type (str): "Rpositive" for polygon
                - convex_flag (bool): whether convex constraints are valid
        """

        if self.name == "polygon" or self.name == "rectangle":
            if vertices is None:
                return None, None, None, False
            convex_flag, _ = is_convex_and_ordered(vertices)

            if convex_flag:
                G, h = gen_inequal_from_vertex(vertices)
                cone_type = "Rpositive"
            else:
                G, h, cone_type = None, None, None
        else:
            G, h, cone_type = None, None, None

        return G, h, cone_type, convex_flag

    def get_circle_Gh(self, center: np.ndarray, radius: float):
        """
        Generate G and h for circle.

        Args:
            center: (2, 1) array of center
            radius: float of radius

        Returns:
            tuple[np.ndarray, np.ndarray, str, bool]:
                - G matrix: (3, 2)
                - h vector: (3, 1)
                - cone_type (str): "norm2"
                - convex_flag (bool): True
        """

        assert self.name == "circle"

        G = np.array([[1, 0], [0, 1], [0, 0]])
        h = np.vstack((center, -radius * np.ones((1, 1))))
        cone_type = "norm2"
        convex_flag = True

        return G, h, cone_type, convex_flag

    def cal_length_width(self, geometry):
        min_x, min_y, max_x, max_y = bounds(geometry).tolist()
        length = max_x - min_x
        width = max_y - min_y

        return length, width

    @property
    def vertices(self):
        if self.name == "linestring":
            x = self.geometry.xy[0]
            y = self.geometry.xy[1]
            return np.c_[x, y].T
        return self.geometry.exterior.coords._coords.T[:, :-1]

    @property
    def init_vertices(self):
        """
        return original_vertices: [[x1, y1], [x2, y2]....    [[x1, y1]]]; [x1, y1] will repeat twice
        """
        assert "this property is renamed to be original_vertices"

    @property
    def original_vertices(self) -> np.ndarray | None:
        """
        Get the original vertices of the geometry.
        """

        if self.name == "linestring":
            x = self._original_geometry.xy[0]
            y = self._original_geometry.xy[1]
            return np.c_[x, y].T

        if self.name == "map":
            return None
        return self._original_geometry.exterior.coords._coords.T[:, :-1]

    @property
    def original_centroid(self) -> np.ndarray:
        """
        Get the original centroid of the geometry.

        Returns:
            np.ndarray: The original centroid of the geometry.
        """
        return np.array(self._original_geometry.centroid.xy)

    @property
    def radius(self):
        return minimum_bounding_radius(self._original_geometry)


@register_geometry("circle")
class CircleGeometry(GeometryHandler):
    def __init__(self, name: str = "circle", **kwargs):
        super().__init__(name, **kwargs)

    def construct_original_geometry(
        self,
        radius: float = 0.2,
        center: list | None = None,
        random_shape: bool = False,
        radius_range: list | None = None,
        wheelbase: float | None = None,
    ):
        if radius_range is None:
            radius_range = [0.1, 1.0]
        if center is None:
            center = [0, 0]
        if random_shape:
            radius = rng.uniform(*radius_range)

        if wheelbase is None:
            return Point(center).buffer(radius)
        return Point([center[0] + wheelbase / 2, center[1]]).buffer(radius)


@register_geometry("polygon")
class PolygonGeometry(GeometryHandler):
    def __init__(self, name: str = "polygon", **kwargs):
        super().__init__(name, **kwargs)

    def construct_original_geometry(
        self,
        vertices=None,
        random_shape: bool = False,
        is_convex: bool = False,
        **kwargs,
    ):
        """
        Construct a polygon geometry.

        Args:
            vertices: [[x1, y1], [x2, y2]..]
            random_shape: whether to generate random shape, default is False
            is_convex: whether to generate convex shape, default is False
            **kwargs: see random_generate_polygon()

        Returns:
            Polygon object
        """

        if random_shape:
            if is_convex:
                kwargs.setdefault("spikeyness_range", [0, 0])
                vertices = random_generate_polygon(**kwargs)
            else:
                vertices = random_generate_polygon(**kwargs)

        elif vertices is None:
            print("No vertices provided for polygon. Using default square")
            vertices = [
                (-1, -1),
                (1, -1),
                (1, 1),
                (-1, 1),
            ]

        polygon = Polygon(vertices)

        if is_valid(polygon):
            return polygon
        print("Invalid polygon. Making it valid.")
        valid_polygons = make_valid(polygon)

        polygon = envelope(valid_polygons)

        return make_valid(polygon)


@register_geometry("rectangle")
class RectangleGeometry(GeometryHandler):
    def __init__(self, name: str = "rectangle", **kwargs):
        super().__init__(name, **kwargs)

    def construct_original_geometry(
        self, length: float = 1.0, width: float = 1.0, wheelbase: float | None = None
    ):
        """
        Args
            length: in x axis
            width: in y axis
            wheelbase: for ackermann robot
        """

        if wheelbase is None:
            vertices = [
                (-length / 2, -width / 2),
                (length / 2, -width / 2),
                (length / 2, width / 2),
                (-length / 2, width / 2),
            ]
        else:
            start_x = -(length - wheelbase) / 2
            start_y = -width / 2

            vertices = [
                (start_x, start_y),
                (start_x + length, start_y),
                (start_x + length, start_y + width),
                (start_x, start_y + width),
            ]

        return Polygon(vertices)


@register_geometry("linestring")
class LinestringGeometry(GeometryHandler):
    def __init__(self, name: str = "linestring", **kwargs):
        super().__init__(name, **kwargs)

    def construct_original_geometry(
        self, vertices, random_shape: bool = False, is_convex: bool = True, **kwargs
    ):
        """
        Construct a LineString object.

        Args:
            vertices: [[x1, y1], [x2, y2]..]
            random_shape: whether to generate random shape, default is False
            is_convex: whether to generate convex shape, default is False
            **kwargs: see random_generate_polygon()

        Returns:
            LineString object
        """

        if random_shape:
            if is_convex:
                vertices = random_generate_polygon(spikeyness_range=[0, 0], **kwargs)
            else:
                vertices = random_generate_polygon(**kwargs)

        return LineString(vertices)


@register_geometry("map")
class PointsGeometry(GeometryHandler):
    def __init__(self, name: str = "map", **kwargs):
        super().__init__(name, **kwargs)

    def construct_original_geometry(self, points: np.ndarray, reso: float = 0.1):
        """
        Args:
            points: (2, N) array of points
            reso: resolution for the buffer
        """

        return MultiPoint(points.T).buffer(reso / 2).boundary


########################################3D Geometry Handler #############################################################


class geometry_handler3d(ABC):
    """
    This class is used to handle the 3D geometry of the object. It reads the shape parameters from yaml file and constructs the geometry of the object.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._original_geometry = self.construct_original_geometry(**kwargs)
        self.geometry = self._original_geometry
        self.wheelbase = kwargs.get("wheelbase")
        self.length, self.width, self.depth = self.cal_length_width(
            self._original_geometry
        )

    @abstractmethod
    def construct_original_geometry(self, **kwargs):
        pass

    def step(self, state):
        """
        Transform geometry to the new state.

        Args:
            state (np.ndarray 6*1): [x, y, z, roll, pitch, roll].

        Returns:
            Transformed geometry.
        """

        def transform_with_state(x, y):
            trans, rot = get_transform(state)
            points = np.array([x, y])
            new_points = rot @ points + trans
            return (new_points[0, :], new_points[1, :])

        new_geometry = transform(transform_with_state, self._original_geometry)
        self.geometry = new_geometry

        return new_geometry


class GeometryFactory:
    """
    Factory class to create geometry handlers.
    """

    @staticmethod
    def create_geometry(name: str = "circle", **kwargs) -> GeometryHandler:
        name = name.lower()

        cls = _geometry_registry.get(name)
        if cls is not None:
            return cls(name, **kwargs)

        # elif name == 'sphere3d':
        #     return Sphere3DGeometry(name, **kwargs)
        # elif name == 'cuboid3d':
        #     return Cuboid3DGeometry(name, **kwargs)

        raise ValueError(f"Invalid geometry name: {name}")

    @staticmethod
    def get_handler_class(name: str) -> type[GeometryHandler] | None:
        """Look up a registered handler class by name without instantiation.

        Args:
            name (str): Shape name (e.g. ``"circle"``, ``"rectangle"``).

        Returns:
            type[GeometryHandler] | None: The class, or ``None`` if not found.
        """
        return _geometry_registry.get(name.lower() if name else "")

    @staticmethod
    def available_shapes() -> dict[str, type[GeometryHandler]]:
        """Read-only view of the registry: ``{name: handler_class}``.
        """
        return dict(_geometry_registry)