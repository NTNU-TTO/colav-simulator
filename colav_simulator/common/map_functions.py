"""map_functions.py.

Summary:
Contains functionality diverse functionality for working with maps,
such as computing distance to land polygons, generating random ship starting positions etc.

Author: Trym Tengesdal
"""

import os

os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd
import geopy.distance
import numpy as np
import scipy.spatial as scipy_spatial
import shapely
from osgeo import osr
from seacharts.enc import ENC
from shapely import affinity, ops, strtree
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon

import colav_simulator.common.miscellaneous_helper_methods as mhm


def check_if_pointing_too_close_towards_land(
    csog_state: np.ndarray, enc: ENC, hazards: list | None = None, min_dist_to_hazard: float = 50.0
) -> bool:
    """Checks if the ship is pointing towards land when being too close to it.

    Args:
        csog_state (np.ndarray): The ship`s state [x, y, U, chi]^T.
        enc (ENC): The Electronic Navigational Chart object.
        hazards (Optional[list], optional): List of relevant grounding hazards. Defaults to None.
        min_dist_to_hazard (float, optional): Minimum distance to static hazards. Defaults to 50.0.

    Returns:
        bool: True if the ship is pointing towards land, False otherwise.
    """
    x, y, _U, chi = csog_state
    x_end = x + 10000.0 * np.cos(chi)
    y_end = y + 10000.0 * np.sin(chi)
    if hazards is None:
        hazards = [enc.land.geometry, enc.shore.geometry]
    closest_point = find_closest_collision_free_point_on_segment(
        enc=enc, p1=np.array([x, y]), p2=np.array([x_end, y_end]), hazards=hazards, min_dist=0.0
    )
    d2hazard = np.linalg.norm(closest_point - np.array([x, y]))
    return d2hazard <= min_dist_to_hazard


def create_bbox_from_points(
    enc: ENC, p1: np.ndarray, p2: np.ndarray, buffer: float = 200.0
) -> tuple[float, float, float, float]:
    """Creates a bounding box from two diagonal corner points.

    Args:
        enc (ENC): Electronic Navigational Chart object.
        p1 (np.ndarray): First corner point.
        p2 (np.ndarray): Second corner point.
        buffer (float): Buffer to add around the bounding box. Defaults to 200.0.

    Returns:
        tuple[float, float, float, float]: Bounding box (xmin, ymin, xmax, ymax),
            with x being easting.
    """
    xmin = min(p1[0], p2[0]) - buffer
    xmax = max(p1[0], p2[0]) + buffer
    ymin = min(p1[1], p2[1]) - buffer
    ymax = max(p1[1], p2[1]) + buffer
    xmin = max(xmin, enc.bbox[1])
    xmax = min(xmax, enc.bbox[3])
    ymin = max(ymin, enc.bbox[0])
    ymax = min(ymax, enc.bbox[2])
    return ymin, xmin, ymax, xmax


def local2latlon(
    x: float | list | np.ndarray, y: float | list | np.ndarray, utm_zone: int
) -> tuple[float | list | np.ndarray, float | list | np.ndarray]:
    """Transform coordinates from x (east), y (north) to latitude, longitude.

    Args:
        x (float | list): East coordinate(s) in a local UTM coordinate system.
        y (float | list): North coordinate(s) in a local UTM coordinate system.
        utm_zone (int): UTM zone.

    Raises:
        ValueError: If the input string is not correct.

    Returns:
        Tuple[float | list | np.ndarray, float | list | np.ndarray]: Tuple of latitude and longitude coordinates.
    """
    to_zone = 4326  # Latitude Longitude
    if utm_zone == 32:
        from_zone = 6172  # ETRS89 / UTM zone 32 + NN54 height Møre og Romsdal
    elif utm_zone == 33:
        from_zone = 6173  # ETRS89 / UTM zone 33 + NN54 height
    else:
        raise ValueError('Input "utm_zone" is not correct. Supported zones sofar are 32 and 33.')

    src = osr.SpatialReference()
    src.ImportFromEPSG(from_zone)
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(to_zone)
    transform = osr.CoordinateTransformation(src, tgt)

    if isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray)):
        coordinates = transform.TransformPoints(list(zip(x, y, strict=False)))
        lat = [coord[0] for coord in coordinates]
        lon = [coord[1] for coord in coordinates]
    else:
        lat, lon, _ = transform.TransformPoint(x, y)

    return lat, lon


def latlon2local(
    lat: float | list | np.ndarray, lon: float | list | np.ndarray, utm_zone: int
) -> tuple[float | list | np.ndarray, float | list | np.ndarray]:
    """Transform coordinates from latitude, longitude to UTM32 or UTM33.

    Args:
        lat (float | list): Latitude coordinate(s)
        lon (float | list): Longitude coordinate(s)
        utm_zone (str): UTM zone.

    Raises:
        ValueError: If the input string is not correct.

    Returns:
        Tuple[float | list | np.ndarray, float | list | np.ndarray]: Tuple of east and north coordinates.
    """
    from_zone = 4326  # Latitude Longitude
    if utm_zone == 32:
        to_zone = 6172  # ETRS89 / UTM zone 32 + NN54 height Møre og Romsdal
    elif utm_zone == 33:
        to_zone = 6173  # ETRS89 / UTM zone 33 + NN54 height
    else:
        raise ValueError('Input "utm_zone" is not correct. Supported zones sofar are 32 and 33.')

    src = osr.SpatialReference()
    src.ImportFromEPSG(from_zone)
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(to_zone)
    transform = osr.CoordinateTransformation(src, tgt)

    if isinstance(lat, (list, np.ndarray)) and isinstance(lon, (list, np.ndarray)):
        coordinates = transform.TransformPoints(list(zip(lat, lon, strict=False)))
        x = [coord[0] for coord in coordinates]
        y = [coord[1] for coord in coordinates]
    else:
        x, y, _ = transform.TransformPoint(lat, lon)

    return x, y


def dist_between_latlon_coords(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes the distance between two latitude, longitude coordinates.

    Args:
        lat1 (float): Latitude of first coordinate.
        lon1 (float): Longitude of first coordinate.
        lat2 (float): Latitude of second coordinate.
        lon2 (float): Longitude of second coordinate.

    Returns:
        float: Distance between the two coordinates in meters
    """
    return geopy.distance.distance((lat1, lon1), (lat2, lon2)).m


def create_point_list_from_polygons(polygons: list) -> tuple[np.ndarray, np.ndarray]:
    """Creates a list of x and y coordinates from a list of polygons.

    Args:
        polygons (list): List of shapely polygons.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing the x (north) and y (east) coordinates
            of the polygons.
    """
    px, py, ls = [], [], []
    for i, poly in enumerate(polygons):
        y, x = poly.exterior.coords.xy
        a = np.array(x.tolist())
        b = np.array(y.tolist())
        la, lx = len(a), len(px)
        c = [(i + lx, (i + 1) % la + lx) for i in range(la - 1)]
        px += a.tolist()
        py += b.tolist()
        ls += c

    points = np.array([px, py]).T
    P1, P2 = points[ls][:, 0], points[ls][:, 1]
    return P1, P2


def extract_vertices_from_polygon_list(polygons: list) -> tuple[np.ndarray, np.ndarray]:
    """Creates a list of x and y coordinates from a list of polygons.

    Args:
        polygons (list): List of shapely polygons.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing the x (north) and y (east)
            coordinates of the polygons.
    """
    px, py = [], []
    for _i, poly in enumerate(polygons):
        if isinstance(poly, MultiPolygon):
            for sub_poly in poly:
                y, x = sub_poly.exterior.coords.xy
                px.extend(x[:-1].tolist())
                py.extend(y[:-1].tolist())
        elif isinstance(poly, Polygon):
            y, x = poly.exterior.coords.xy
            px.extend(x[:-1].tolist())
            py.extend(y[:-1].tolist())
        else:
            continue
    return np.array(px), np.array(py)


def extract_safe_sea_area(
    min_depth: int,
    enveloping_polygon: Polygon,
    enc: ENC | None = None,
    as_polygon_list: bool = False,
    buffer: float | None = None,
    show_plots: bool = False,
) -> MultiPolygon | list:
    """Extracts the safe sea area from the ENC as a list of polygons.

    This includes sea polygons that are above the vessel`s minimum depth.

    Args:
        min_depth (int): The minimum depth required for the vessel to avoid
            grounding.
        enveloping_polygon (Polygon): The query polygon.
        enc (ENC | None): Electronic Navigational Chart object used for
            plotting. Defaults to None.
        as_polygon_list (bool): Option for returning the safe sea area as a
            list of polygons. Defaults to False.
        buffer (float | None): Safety buffer for polygons. Defaults to None.
        show_plots (bool): Option for visualization. Defaults to False.

    Returns:
        MultiPolygon | list: The safe sea area.
    """
    seabed = enc.seabed[min_depth].geometry
    if buffer is not None:
        seabed = seabed.buffer(-buffer, cap_style=3, join_style=2)
    safe_sea = seabed.intersection(enveloping_polygon)

    if enc is not None and show_plots:
        enc.start_display()
        enc.draw_polygon(safe_sea, color="green", alpha=0.25, fill=False)

    if as_polygon_list:
        if isinstance(safe_sea, MultiPolygon):
            return list(safe_sea.geoms)
        elif isinstance(safe_sea, Polygon):
            return [safe_sea]
        else:
            return []
    return safe_sea


def create_free_boundary_points_from_enc(enc: ENC, hazards: list) -> tuple[np.ndarray, np.ndarray]:
    """Creates an array of points on the ENC boundary which is free from grounding hazards.

    Args:
        enc (ENC): Electronic Navigational Chart object.
        hazards (list): List of relevant grounding hazards.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of x and y coordinates of the free boundary points.
    """
    (xmin, ymin, xmax, ymax) = enc.bbox
    n_pts_per_side = 50
    x = np.linspace(xmin, xmax, n_pts_per_side)
    y = np.linspace(ymin, ymax, n_pts_per_side)
    points = []
    for i in range(n_pts_per_side):
        p = Point(x[i], ymin)
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = Point(x[i], ymax)
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = Point(xmin, y[i])
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = Point(xmax, y[i])
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)
    # [enc.draw_circle((p.x, p.y), radius=0.5, color="yellow") for p in points]
    Y = np.array([p.x for p in points])
    X = np.array([p.y for p in points])
    return X, Y


def bbox_to_polygon(bbox: tuple[float, float, float, float]) -> Polygon:
    """Converts a bounding box to a polygon.

    Args:
        bbox (tuple[float, float, float, float]): The bounding box (xmin, ymin, xmax, ymax), with x being easting.

    Returns:
        Polygon: The polygon.
    """
    (xmin, ymin, xmax, ymax) = bbox
    return Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])


def point_in_polygon_list(point: Point | np.ndarray, polygons: list) -> bool:
    """Checks if a point is in a list of polygons.

    Args:
        point (Point | np.ndarray): The point to check, with x being easting.
        polygons (list): List of polygons.

    Returns:
        bool: True if the point is in a hazard, False otherwise.
    """
    if isinstance(point, np.ndarray):
        point = Point(point[1], point[0])
    for poly in polygons:
        if point.within(poly) or point.touches(poly):
            return True
    return False


def point_in_point_list(point: Point, points: list) -> bool:
    """Checks if a point is in a list of points.

    Args:
        point (Point): The point to check.
        points (list): List of points.

    Returns:
        bool: True if the point is in a hazard, False otherwise.
    """
    for p in points:
        if point.within(p) or point.touches(p):
            return True
    return False


def create_safe_sea_voronoi_diagram(enc: ENC, vessel_min_depth: int = 5) -> tuple[scipy_spatial.Voronoi, list]:
    """Creates a Voronoi diagram of the safe sea region (i.e. its vertices).

    Args:
        enc (ENC): The Electronic Navigational Chart object.
        vessel_min_depth (float): The safe minimum depth for the vessel to voyage in.

    Returns:
        scipy_spatial.Voronoi: The Voronoi diagram of the safe sea region.
    """
    bbox = enc.bbox
    enc_bbox_poly = bbox_to_polygon(bbox)
    safe_sea = extract_safe_sea_area(vessel_min_depth, enc_bbox_poly, enc, as_polygon_list=True, show_plots=True)
    polygons = []
    for sea_poly in safe_sea:
        if isinstance(sea_poly, MultiPolygon):
            for poly in sea_poly:
                polygons.append(poly)
        elif isinstance(sea_poly, Polygon):
            polygons.append(sea_poly)
        else:
            continue
    px, py = extract_vertices_from_polygon_list(polygons)
    points = np.vstack((py, px)).T
    vor = scipy_spatial.Voronoi(points)
    region_polygons = create_region_polygons_from_voronoi(vor, enc=enc)
    for point in points:
        enc.draw_circle((point[0], point[1]), radius=0.4, color="red")

    # # Keep all voronoi region boundary points that are in the safe sea area
    # safe_points = []
    # for region in vor.regions:
    #     region_vertices = vor.vertices[region]
    #     for vertex in region_vertices:
    #         point = Point(vertex)
    #         if point_in_polygon_list(point, polygons):
    #             safe_points.append((point.x, point.y))
    # settt = set(safe_points)
    # safe_points = list(settt)
    # for point in safe_points:
    #     enc.draw_circle((point[0], point[1]), radius=0.4, color="magenta")
    return vor, region_polygons


def create_safe_sea_triangulation(
    enc: ENC,
    vessel_min_depth: int = 5,
    bbox: tuple[float, float, float, float] | None = None,
    buffer: float | None = None,
    show_plots: bool = True,
) -> list:
    """Creates a constrained delaunay triangulation of the safe sea region.

    Args:
        enc (ENC): Electronic Navigational Chart object.
        vessel_min_depth (int, optional): The safe minimum depth for the vessel to voyage in. Defaults to 5.
        bbox (Optional[Tuple[float, float, float, float]]): Bounding box of the safe sea region to constrain the cdt within.
        buffer (Optional[float], optional): Safety buffer for polygons.
        show_plots (bool, optional): Option for visualization. Defaults to True.

    Returns:
        list: List of triangles.
    """
    if bbox is None:
        bbox = enc.bbox

    safe_sea_poly_list = extract_safe_sea_area(
        vessel_min_depth, bbox_to_polygon(bbox), enc, as_polygon_list=True, buffer=buffer, show_plots=show_plots
    )
    cdt_list = []
    largest_poly_area = 0.0
    for poly in safe_sea_poly_list:
        cdt = constrained_delaunay_triangulation_custom(poly)
        if poly.area > largest_poly_area:
            largest_poly_area = poly.area
            cdt_largest = cdt
        if show_plots:
            # enc.draw_polygon(poly, color="blue", alpha=0.2)
            enc.start_display()
            for triangle in cdt:
                enc.draw_polygon(triangle, color="green", fill=False)
        cdt_list.append(cdt)

    return cdt_largest


def create_region_polygons_from_voronoi(vor: scipy_spatial.Voronoi, enc: ENC | None = None) -> list:
    """Creates a list of polygons from the Voronoi diagram.

    Args:
        vor (scipy_spatial.Voronoi): The Voronoi diagram.
        enc (Optional[ENC], optional): The Electronic Navigational Chart object.

    Returns:
        list: List of polygons.
    """
    polygons = []
    for region in vor.regions:
        if not region:
            continue
        region_vertices = vor.vertices[region]
        if region_vertices.shape[0] < 3:
            continue
        region_poly = Polygon(region_vertices)
        if region_poly.area < 1.0:
            continue
        polygons.append(region_poly)
        if enc:
            enc.start_display()
            enc.draw_polygon(region_poly, color="yellow", alpha=0.5)
    return polygons


def create_ship_polygon(
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    length_scaling: float = 1.0,
    width_scaling: float = 1.0,
) -> Polygon:
    """Creates a ship polygon from the ship`s position, heading, length and width.

    Args:
        x (float): The ship`s north position
        y (float): The ship`s east position
        heading (float): The ship`s heading
        length (float): Length of the ship
        width (float): Width of the ship
        length_scaling (float, optional): Length scale factor. Defaults to 1.0.
        width_scaling (float, optional): Length scale factor. Defaults to 1.0.

    Returns:
        np.ndarray: Ship polygon
    """
    eff_length = length * length_scaling
    eff_width = width * width_scaling

    x_min, x_max = x - eff_length / 2.0, x + eff_length / 2.0 - eff_width
    y_min, y_max = y - eff_width / 2.0, y + eff_width / 2.0
    left_aft, right_aft = (y_min, x_min), (y_max, x_min)
    left_bow, right_bow = (y_min, x_max), (y_max, x_max)
    coords = [left_aft, left_bow, (y, x + eff_length / 2.0), right_bow, right_aft]
    poly = Polygon(coords)
    return affinity.rotate(poly, -heading, origin=(y, x), use_radians=True)


def find_minimum_depth(vessel_draft: float, enc: ENC) -> int:
    """Find the minimum seabed depth for the given vessel draft.

    For it to avoid grounding.

    Args:
        vessel_draft (float): The vessel`s draft.
        enc (ENC): Electronic Navigational Chart object.

    Returns:
        float: The minimum seabed depth required for a safe journey for the vessel.
    """
    lowest_possible_depth = 0
    for depth in enc.seabed:
        if vessel_draft <= float(depth):
            lowest_possible_depth = depth
            break
    return lowest_possible_depth


def extract_relevant_grounding_hazards(vessel_min_depth: int, enc: ENC) -> list:
    """Extracts the relevant grounding hazards from the ENC as a list of (multi) polygons.

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.

    Returns:
        list: The relevant grounding hazards.
    """
    dangerous_seabed = (
        enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry) if vessel_min_depth > 0 else MultiPolygon()
    )
    return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]


def extract_relevant_grounding_hazards_as_union(
    vessel_min_depth: int, enc: ENC, buffer: float | None = None, show_plots: bool = False
) -> list:
    """Extracts the relevant grounding hazards from the ENC as a multipolygon.

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.
        buffer (Optional[float], optional): Buffer for polygons.
        show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        list: The relevant grounding hazards.
    """
    dangerous_seabed = (
        enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry) if vessel_min_depth > 0 else MultiPolygon()
    )
    # return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]
    relevant_hazards = [enc.land.geometry.union(enc.shore.geometry).union(dangerous_seabed)]
    filtered_relevant_hazards = []
    for hazard in relevant_hazards:
        poly = hazard
        if buffer is not None:
            poly = poly.buffer(buffer)

        if isinstance(hazard, Polygon):
            poly = MultiPolygon([Polygon(hazard.exterior)])

        # remove interior
        if isinstance(poly, MultiPolygon):
            poly = MultiPolygon(Polygon(p.exterior) for p in poly.geoms if isinstance(p, Polygon))

        filtered_relevant_hazards.append(poly)

    if show_plots:
        enc.start_display()
        for hazard in filtered_relevant_hazards:
            if isinstance(hazard, MultiPolygon):
                for poly in hazard.geoms:
                    enc.draw_polygon(poly, color="red", fill=False)
            else:
                enc.draw_polygon(hazard, color="red", fill=False)
    return filtered_relevant_hazards


def fill_rtree_with_geometries(geometries: list) -> tuple[strtree.STRtree, list]:
    """Fills an rtree with the given multipolygon geometries. Used for fast spatial queries.

    Args:
        geometries (list): The geometries to fill the rtree with.

    Returns:
        tuple[strtree.STRtree, list]: The rtree containing the geometries, and the Polygon objects used to build it.
    """
    poly_list = []
    for poly in geometries:
        if not isinstance(poly, MultiPolygon):
            msg = "Only MultiPolygon members are supported"
            raise ValueError(msg)
        for sub_poly in poly.geoms:
            poly_list.append(sub_poly)
    return strtree.STRtree(poly_list), poly_list


def generate_random_goal_position(
    rng: np.random.Generator,
    enc: ENC,
    xs_start: np.ndarray,
    safe_sea_cdt: list,
    safe_sea_cdt_weights: list,
    bbox: tuple[float, float, float, float] | None = None,
    min_distance_from_start: float = 300.0,
    max_distance_from_start: float = 10000.0,
    sector_width: float = 60.0 * np.pi / 180.0,
    min_distance_to_land: float = 50.0,
    show_plots: bool = False,
) -> tuple[float, float]:
    """Generates a random goal position for the ship.

    Given its starting state (position, speed and heading).

    Args:
        rng (np.random.Generator): Numpy random generator.
        enc (ENC): Electronic Navigational Chart object.
        xs_start (np.ndarray): Starting CSOG state of the ship [x, y, U, chi]^T.
        safe_sea_cdt (list): List of triangles defining the safe sea region,
            used to sample more efficiently.
        safe_sea_cdt_weights (list): List of weights for the safe sea region
            triangles, used to sample more efficiently.
        bbox (tuple[float, float, float, float] | None): Bounding box. Defaults
            to None.
        min_distance_from_start (float): Minimum distance from the starting
            position. Defaults to 300.0.
        max_distance_from_start (float): Maximum distance from the starting
            position. Defaults to 10000.0.
        sector_width (float): Width of the sector to sample from. Defaults to
            60.0 * np.pi / 180.0.
        min_distance_to_land (float): Minimum distance to land. Defaults to 50.0.
        show_plots (bool): Option for visualization. Defaults to False.

    Returns:
        tuple[float, float]: Goal position (northing, easting) for the ship.
    """
    if bbox is None:
        bbox = enc.bbox
    bbox_poly = bbox_to_polygon(bbox)

    if max_distance_from_start <= min_distance_from_start:
        print(
            "WARNING: Max_distance_from_start must be larger than "
            "min_distance_from_start in goal position sampling. "
            "Setting to default values.."
        )
        max_distance_from_start = min_distance_from_start + 500.0

    northing = xs_start[0] + max_distance_from_start * np.cos(xs_start[3])
    easting = xs_start[1] + max_distance_from_start * np.sin(xs_start[3])
    sector_radius = max(max_distance_from_start, min_distance_from_start)
    n_points = 100
    angle_range = np.linspace(-sector_width / 2.0 + xs_start[3], sector_width / 2.0 + xs_start[3], n_points)
    arc = [
        (xs_start[1] + sector_radius * np.sin(angle), xs_start[0] + sector_radius * np.cos(angle)) for angle in angle_range
    ]
    arc_linestring = LineString(arc)
    sector_poly = Polygon(list(arc_linestring.coords) + [(xs_start[1], xs_start[0])])
    sector_poly = sector_poly.intersection(bbox_poly)
    if show_plots:
        enc.start_display()
        enc.draw_polygon(sector_poly, color="green", fill=True, alpha=0.2)
    max_iter = 3000
    for it in range(max_iter):
        p = mhm.sample_from_triangulation(rng, safe_sea_cdt, safe_sea_cdt_weights)
        easting, northing = p[0], p[1]

        dist2start = np.linalg.norm(np.array([northing, easting]) - np.array([xs_start[0], xs_start[1]]))
        inside_sector = sector_poly.contains(Point(easting, northing))
        dist2land = enc.land.geometry.distance(Point(easting, northing))
        if (
            (min_distance_from_start <= dist2start <= max_distance_from_start)
            and inside_sector
            and (dist2land >= min_distance_to_land)
        ):
            break

        if it == max_iter - 1:
            print("WARNING: No goal position that satisfies the constraints found. Returning a random position...")

    return northing, easting


def generate_random_position_from_draft(
    rng: np.random.Generator,
    enc: ENC,
    draft: float,
    safe_sea_cdt: list | None = None,
    safe_sea_cdt_weights: list | None = None,
    min_hazard_clearance: float = 30.0,
) -> tuple[float, float]:
    """Randomly defining easting and northing coordinates of a ship.

    Inside the safe sea region by considering a ship draft.

    Args:
        rng (np.random.Generator): Numpy random generator.
        enc (ENC): Electronic Navigational Chart object.
        draft (float): Ship's draft in meters.
        safe_sea_cdt (list | None): List of triangles defining the safe sea
            region, used to sample more efficiently. Defaults to None.
        safe_sea_cdt_weights (list | None): List of weights for the safe sea
            region triangles, used to sample more efficiently. Defaults to None.
        min_hazard_clearance (float): Minimum distance to ENC hazards.
            Defaults to 30.0.

    Returns:
        tuple[float, float]: Tuple of starting x and y coordinates for the
            ship.
    """
    depth = find_minimum_depth(draft, enc)
    safe_sea = enc.seabed[depth]
    bbox = enc.bbox

    max_iter = 1000
    northing = enc.bbox[1] + 0.5 * (enc.bbox[3] - enc.bbox[1])
    easting = enc.bbox[0] + 0.5 * (enc.bbox[2] - enc.bbox[0])
    for _ in range(max_iter):
        if safe_sea_cdt is not None:
            p = mhm.sample_from_triangulation(rng, safe_sea_cdt, safe_sea_cdt_weights)
            easting, northing = p[0], p[1]
        else:
            easting, northing = rng.uniform(bbox[0], bbox[2]), rng.uniform(bbox[1], bbox[3])

        inside_bbox = mhm.inside_bbox(np.array([northing, easting]), (bbox[1], bbox[0], bbox[3], bbox[2]))
        d2land = distance_to_enc_hazards(easting, northing, depth, enc)
        if safe_sea.geometry.contains(Point(easting, northing)) and inside_bbox and d2land >= min_hazard_clearance:
            break

    return northing, easting


def distance_to_enc_hazards(x: float, y: float, min_depth: int, enc: ENC, hazards: list | None = None) -> float:
    """Computes the distance to the closest ENC hazard.

    Args:
        x (float): The easting coordinate.
        y (float): The northing coordinate.
        min_depth (int): Minimum depth for the vessel.
        enc (ENC): The ENC object.
        hazards (list | None): List of Multipolygon/Polygon objects that are
            relevant. Used if not None. Defaults to None.

    Returns:
        float: The distance to the closest ENC hazard.
    """
    if hazards is None:
        hazards = extract_relevant_grounding_hazards_as_union(min_depth, enc)

    point = Point(x, y)
    min_dist = 1e12
    for hazard in hazards:
        if hazard.is_empty:
            continue
        dist = point.distance(hazard)
        min_dist = min(min_dist, dist)
    return min_dist


def find_closest_collision_free_point_on_segment(
    enc: ENC, p1: np.ndarray, p2: np.ndarray, draft: float = 5.0, hazards: list | None = None, min_dist: float = 30.0
) -> np.ndarray:
    """Finds the closest collision free point on a line segment between two points.

    Args:
        enc (ENC): Electronic Navigational Chart object
        p1 (np.ndarray): First position [x1, y1]^T. x = north, y = east.
        p2 (np.ndarray): Second position.
        draft (float, optional): Vessel draft. Defaults to 5.0.
        hazards (Optional[list], optional): List of Multipolygon/Polygon objects that are relevant. Used if not none.
        min_dist (float, optional): Minimum distance to the hazard. Defaults to 5.0.

    Returns:
        np.ndarray: The closest collision free point on the line segment.
    """
    if not (p1.shape == (2,) and p2.shape == (2,)):
        msg = "p1 and p2 must be 2D vectors"
        raise ValueError(msg)
    segment = LineString([(p1[1], p1[0]), (p2[1], p2[0])])
    if hazards is None:
        hazards = extract_relevant_grounding_hazards_as_union(find_minimum_depth(draft, enc), enc)

    for hazard_item in hazards:
        if hazard_item.is_empty:
            continue
        hazard = hazard_item.buffer(min_dist)

        # enc.draw_polygon(hazard, color="orange", fill=False)

        if segment.intersects(hazard):
            intersection = segment.intersection(hazard)
            nearest_point = ops.nearest_points(Point(p1[1], p1[0]), intersection)[1]
            # enc.draw_circle((nearest_point.x, nearest_point.y), radius=1.0, color="yellow", fill=False)
            return np.array([nearest_point.y, nearest_point.x])
    return p2


def compute_distance_vectors_to_grounding(
    vessel_trajectory: np.ndarray,
    min_vessel_depth: int,
    enc: ENC,
    disable_bbox_check: bool = False,
    show_plots: bool = False,
) -> np.ndarray:
    """Computes the distance vectors to grounding at each step.

    For the given vessel trajectory or point if n_samples = 1.

    Args:
        vessel_trajectory (np.ndarray): The vessel`s trajectory, 2 x n_samples.
        min_vessel_depth (int): The minimum depth required for the vessel to
            avoid grounding.
        enc (ENC): The ENC to check for grounding.
        disable_bbox_check (bool): Option for disabling the inside bounding
            box check for a position. Defaults to False.
        show_plots (bool): Option for visualization. Defaults to False.

    Returns:
        np.ndarray: The distance to grounding at each step of the vessel
            trajectory.
    """
    n_samples = vessel_trajectory.shape[1]
    x_min, y_min, x_max, y_max = enc.bbox
    bbox_poly = bbox_to_polygon((float(x_min), float(y_min), float(x_max), float(y_max)))
    if show_plots:
        enc.start_display()
    relevant_hazards = extract_relevant_grounding_hazards_as_union(min_vessel_depth, enc)
    distance_vectors = np.ndarray((2, vessel_trajectory.shape[1]))
    for idx in range(n_samples):
        point = Point(vessel_trajectory[0, idx], vessel_trajectory[1, idx])
        nearest_poly_points = []
        for hazard in relevant_hazards:
            if not bbox_poly.contains(point) and not disable_bbox_check:
                distance_vectors[:, idx] = np.array([0.0, 0.0])
                break

            nearest_point = ops.nearest_points(point, hazard)[1]
            nearest_poly_points.append(nearest_point)

            min_dist = 1e12
            min_dist_vec = np.array([1e6, 1e6])
            for _, near_point in enumerate(nearest_poly_points):
                points = [
                    (np.asarray(point.coords.xy[0])[0], np.asarray(point.coords.xy[1])[0]),
                    (np.asarray(near_point.coords.xy[0])[0], np.asarray(near_point.coords.xy[1])[0]),
                ]

                if show_plots:
                    enc.draw_line(points, color="black", width=0.5, marker_type="o")

                dist_vec = np.array([points[1][0] - points[0][0], points[1][1] - points[0][1]])
                if np.linalg.norm(dist_vec) <= min_dist:
                    min_dist_vec = dist_vec
                    min_dist = np.linalg.norm(min_dist_vec)
            distance_vectors[:, idx] = min_dist_vec
    return distance_vectors


def get_distance_vectors_to_obstacles(
    trajectory: np.ndarray,
    do_list: list,
    enc: ENC,
    T: float,
    dt: float,
    min_vessel_depth: int = 5,
    disable_bbox_check: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the distance vectors from the trajectory to the obstacles (dynamic and static).

    Args:
        trajectory (np.ndarray): Trajectory/position data (minimum 2 x n_samples).
        do_list (list): List of dynamic obstacles on the form (ID, state, cov, length, width)
        enc (senc.ENC): ENC object.
        T (float): Prediction horizon.
        dt (float): Time step.
        min_vessel_depth (int): Minimum vessel depth.
        disable_bbox_check (bool): Option for disabling the inside bounding box check for a position.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of distance vectors to dynamic obstacles and list of distance vectors
            to static obstacles.
    """
    distance_vectors_so = compute_distance_vectors_to_grounding(trajectory, min_vessel_depth, enc, disable_bbox_check)
    distance_vectors_do = compute_distance_vectors_to_dynamic_obstacles(trajectory, do_list, T, dt)
    return distance_vectors_do, distance_vectors_so


def compute_minimum_distance_to_collision_and_grounding(
    trajectory: np.ndarray,
    do_list: list,
    enc: ENC,
    T: float,
    dt: float,
    min_vessel_depth: int = 5,
    disable_bbox_check: bool = False,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Check if the trajectory collides with any of the obstacles (dynamic and static) over the prediction horizon.

    Args:
        trajectory (np.ndarray): Trajectory/position data (minimum 2 x n_samples) with EN coordinates.
        do_list (list): List of dynamic obstacles on the form (ID, state, cov, length, width) with EN coordinates.
        enc (ENC): ENC object.
        T (float): Prediction horizon.
        dt (float): Time step.
        min_vessel_depth (int): Minimum allowable vessel depth.
        disable_bbox_check (bool): Option for disabling the inside bounding box check for a position.

    Returns:
        tuple[float, float, np.ndarray, np.ndarray]: The minimum distances to collision and grounding, respectively.
            Also returns the corresponding distance vectors.
    """
    distance_vectors_do, distance_vectors_so = get_distance_vectors_to_obstacles(
        trajectory, do_list, enc, T, dt, min_vessel_depth, disable_bbox_check
    )
    min_dist_so = 1e12
    if distance_vectors_so.size > 0:
        min_dist_so = np.min(np.linalg.norm(distance_vectors_so, axis=0))
    min_dist_do = 1e12
    if distance_vectors_do.size > 0:
        min_dist_do = np.min(np.linalg.norm(distance_vectors_do, axis=0))
    return min_dist_do, min_dist_so, distance_vectors_do, distance_vectors_so


def compute_distance_vectors_to_dynamic_obstacles(trajectory: np.ndarray, do_list: list, T: float, dt: float) -> np.ndarray:
    """Computes the (shortest) distance vectors to dynamic obstacles, assuming EN coordinates.

    Args:
        trajectory (np.ndarray): Trajectory/position data (minimum 2 x n_samples) with EN coordinates
        do_list (list): List of dynamic obstacles on the form (ID, state, cov, length, width) with EN coordinates
        T (float): Prediction horizon.
        dt (float): Time step.

    Returns:
        np.ndarray: (Shortest) Distance vectors to dynamic obstacles.
    """
    if len(do_list) == 0:
        return np.empty(0)
    n_samples = trajectory.shape[1]
    if n_samples <= 1:
        msg = "Trajectory must have at least two samples"
        raise ValueError(msg)
    if n_samples != int(T / dt):
        msg = "Must have n_samples = int(T / dt)"
        raise ValueError(msg)
    distance_vectors = np.ndarray((2, n_samples))
    for k in range(n_samples):
        t = k * dt
        p_k = trajectory[:, k]
        min_do_dist_vec = np.array([1e6, 1e6])
        min_do_dist = 1e12
        for _ID, do_state, _do_cov, _do_length, _do_width in do_list:
            p_do_k = do_state[:2] + np.array([do_state[2], do_state[3]]) * t
            dist_vec = p_do_k - p_k
            if np.linalg.norm(dist_vec) < min_do_dist:
                min_do_dist = np.linalg.norm(dist_vec)
                min_do_dist_vec = dist_vec
        distance_vectors[:, k] = min_do_dist_vec
    return distance_vectors


def compute_distance_vector_to_bbox(
    x: float,
    y: float,
    bbox: tuple[float, float, float, float],
    enc: ENC | None = None,  # noqa: ARG001
) -> np.ndarray:
    """Computes the distance vector to the closest point on the bounding box.

    Args:
        x (float): Easting coordinate.
        y (float): Northing coordinate.
        bbox (tuple[float, float, float, float]): Bounding box (xmin, ymin,
            xmax, ymax).
        enc (ENC | None): Electronic Navigational Chart object. Defaults to None.

    Returns:
        np.ndarray: Distance vector to the closest point on the bounding box.
    """
    south_line = LineString([(bbox[0], bbox[1]), (bbox[2], bbox[1])])
    east_line = LineString([(bbox[2], bbox[1]), (bbox[2], bbox[3])])
    north_line = LineString([(bbox[2], bbox[3]), (bbox[0], bbox[3])])
    west_line = LineString([(bbox[0], bbox[3]), (bbox[0], bbox[1])])
    lines = [north_line, east_line, south_line, west_line]
    min_dist = 1e12
    distance_vector = np.array([1e6, 1e6])
    # if enc is not None:
    #     enc.start_display()

    for line in lines:
        d2line = line.distance(Point(x, y))
        line_point = ops.nearest_points(Point(x, y), line)[1]
        if d2line < min_dist:
            min_dist = d2line
            distance_vector = np.array([line_point.x - x, line_point.y - y])
        # if enc is not None:
        #     enc.draw_line([(x, y), (line_point.x, line_point.y)], color="red")
        #     enc.draw_circle((line_point.x, line_point.y), radius=0.5, color="red")
        #     enc.draw_circle((x, y), radius=0.5, color="blue")

    return distance_vector


def min_distance_to_hazards(hazards: list, x: float, y: float) -> float:
    """Compute the minimum distance to hazards from a given point.

    Args:
        hazards (list): List of Multipolygon/Polygon objects that are relevant
        x (float): Ship's easting coordinate
        y (float): Ship's northing coordinate

    Returns:
        float: Minimum distance to hazards in meters.
    """
    min_dist = 1e12
    for hazard in hazards:
        if hazard.is_empty:
            continue

        dist = hazard.distance(Point(x, y))
        min_dist = min(min_dist, dist)
    return min_dist


def check_if_segment_crosses_grounding_hazards(
    enc: ENC, p1: np.ndarray, p2: np.ndarray, draft: float = 5.0, hazards: list | None = None
) -> bool:
    """Checks if a line segment between two positions/points crosses nearby grounding hazards (land, shore).

    Args:
        enc (ENC): Electronic Navigational Chart object
        p1 (np.ndarray): First position [x1, y1]^T. x = north, y = east.
        p2 (np.ndarray): Second position.
        draft (float): Ship's draft in meters.¨
        hazards (Optional[list]): List of Multipolygon/Polygon objects that are relevant. Used if not none.

    Returns:
        bool: True if path segment crosses land, False otherwise.

    """
    # Create linestring with east as the x value and north as the y value
    p2_reverse = (p2[1], p2[0])
    p1_reverse = (p1[1], p1[0])
    wp_line = LineString([p1_reverse, p2_reverse])

    min_depth = find_minimum_depth(draft, enc)

    if hazards is None:
        hazards = extract_relevant_grounding_hazards_as_union(min_depth, enc)

    for hazard in hazards:
        if hazard.is_empty:
            continue

        intersects_hazards = wp_line.intersects(hazard)
        if intersects_hazards:
            return True

    return False


def generate_ship_sector_polygons(
    pos_x: float, pos_y: float, chi: float, safety_radius: float
) -> tuple[Polygon, Polygon, Polygon]:
    """Generates sector polygons for the ship portside, front and starboardside.

    Args:
        pos_x (float): X position of ship
        pos_y (float): Y position of ship
        chi (float): Course over ground of ship
        safety_radius (int): radius for collision polygons on port, front and starboard of ship

    Returns:
        Tuple[Polygon, Polygon, Polygon]: Tuple of polygons for port, front and starboard collision zones
    """
    ship_center = Point([pos_x, pos_y])

    # zone parameters
    angle = np.pi / 4  # zone intersection angle
    offset = np.pi / 2  # zone offset angle
    num_points = 100

    # Close coast zone port
    angle_range_port = np.linspace(-chi + 2 * offset - angle, -chi + 2 * offset + angle, num_points)
    arc_port = [
        (ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle))
        for angle in angle_range_port
    ]
    arc_line_port = LineString(arc_port)
    zone_port = Polygon(list(arc_line_port.coords) + [ship_center])

    # Close coast zone front
    angle_range_front = np.linspace(-chi + offset - angle, -chi + offset + angle, num_points)
    arc_front = [
        (ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle))
        for angle in angle_range_front
    ]
    arc_line_front = LineString(arc_front)
    zone_front = Polygon(list(arc_line_front.coords) + [ship_center])

    # Close coast zone starboard
    angle_range_starboard = np.linspace(-chi - angle, -chi + angle, num_points)
    arc_starboard = [
        (ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle))
        for angle in angle_range_starboard
    ]
    arc_line_starboard = LineString(arc_starboard)
    zone_starboard = Polygon(list(arc_line_starboard.coords) + [ship_center])

    return zone_port, zone_front, zone_starboard


def distances_to_coast(
    poly_port: Polygon, poly_front: Polygon, poly_starboard: Polygon, poly_ship: Polygon, poly_land: MultiPolygon
) -> tuple[float, float, float]:
    """Calculates distance to coast based on intersection between collision polygons and land polygon.

    Args:
        poly_port (Polygon): Polygon of collision zone on portside of ship
        poly_front (Polygon): Polygon of collision zone in front of ship
        poly_starboard (Polygon): Polygon of collision zone on starboardside of ship
        poly_ship (Polygon): Polygon of ship
        poly_land (MultiPolygon): Polygon of land

    Returns:
        Tuple[float, float, float]: Tuple of distances to land on portside, front and starboardside
    """
    port_intersec_land = poly_port.intersection(poly_land)
    front_intersec_land = poly_front.intersection(poly_land)
    starboard_intersec_land = poly_starboard.intersection(poly_land)

    dist_port = poly_ship.distance(port_intersec_land)
    dist_front = poly_ship.distance(front_intersec_land)
    dist_starboard = poly_ship.distance(starboard_intersec_land)

    return dist_port, dist_front, dist_starboard


def generate_enveloping_polygon(trajectory: np.ndarray, buffer: float) -> Polygon:
    """Creates an enveloping polygon around the trajectory of the vessel.

    Buffered by the given amount.

    Args:
        trajectory (np.ndarray): Trajectory with min shape 2 x n_samples.
        buffer (float): Buffer size.

    Returns:
        Polygon: The query polygon.
    """
    point_list = []
    for k in range(trajectory.shape[1]):
        point_list.append((trajectory[1, k], trajectory[0, k]))
    trajectory_linestring = LineString(point_list).buffer(buffer)
    return trajectory_linestring


def extract_hazards_within_bounding_box(
    hazards: list,
    bbox: tuple[float, float, float, float],
    enc: ENC | None = None,
    show_plots: bool = False,
) -> list:
    """Extracts the hazards that are inside the given bounding box.

    Args:
        hazards (list): List of Multipolygon hazards to consider.
        bbox (tuple[float, float, float, float]): Bounding box to consider in the form (x_min, y_min, x_max, y_max),
        x = easting, y = northing.
        enc (ENC | None): Electronic Navigational Chart object. Defaults to None.
        show_plots (bool): Whether to show plots or not. Defaults to False.

    Returns:
        list: List of hazards inside the bounding box.
    """
    bbox_poly = bbox_to_polygon(bbox)
    intersections = []
    for hazard in hazards:
        if bbox_poly.intersects(hazard):
            overlap = bbox_poly.intersection(hazard)
            if isinstance(overlap, Point):
                intersections.append(Polygon(overlap.buffer(0.1)))
            elif isinstance(overlap, LineString):
                intersections.append(Polygon(overlap.buffer(0.1)))
            elif isinstance(overlap, MultiLineString):
                intersections.append(Polygon(overlap.buffer(0.1)))
            elif isinstance(overlap, MultiPolygon):
                intersections.extend(overlap.geoms)
            elif isinstance(overlap, Polygon):
                intersections.append(overlap)
    multipoly_hazard = MultiPolygon(intersections)

    if enc and show_plots:
        enc.start_display()
        for h in multipoly_hazard.geoms:
            enc.draw_polygon(h, color="red", fill=True, alpha=0.5)

    return [multipoly_hazard]


def extract_polygons_near_trajectory(
    trajectory: np.ndarray,
    geometry_tree: strtree.STRtree,
    buffer: float,
    enc: ENC | None = None,
    clip_to_bbox: bool = True,
    show_plots: bool = False,
) -> tuple[list, Polygon]:
    """Extracts the polygons that are relevant for the trajectory of the vessel.

    Inside a corridor of the given buffer size.

    Args:
        trajectory (np.ndarray): Trajectory to consider.
        geometry_tree (strtree.STRtree): The rtree containing the relevant
            grounding hazard polygons.
        buffer (float): Buffer size.
        enc (ENC | None): Electronic Navigational Chart object used for
            plotting. Defaults to None.
        clip_to_bbox (bool): Whether to clip the polygons to the bounding box
            or not. Defaults to True.
        show_plots (bool): Whether to show plots or not. Defaults to False.

    Returns:
        tuple[list, Polygon]: List of tuples of relevant polygons inside
            query/envelope polygon and the corresponding original polygon they
            belong to. Also returns the query polygon.
    """
    enveloping_polygon = generate_enveloping_polygon(trajectory, buffer)
    if clip_to_bbox:
        bbox_poly = bbox_to_polygon(enc.bbox)
        enveloping_polygon = enveloping_polygon.intersection(bbox_poly)
    polygons_near_trajectory_indices = geometry_tree.query(enveloping_polygon)
    polygons_near_trajectory = [geometry_tree.geometries[idx] for idx in polygons_near_trajectory_indices]
    poly_list = []
    for poly in polygons_near_trajectory:
        relevant_poly_list = []
        intersection_poly = enveloping_polygon.intersection(poly)
        if intersection_poly.area == 0.0 and intersection_poly.length == 0.0:
            continue

        if isinstance(intersection_poly, MultiPolygon):
            for sub_poly in intersection_poly.geoms:
                relevant_poly_list.append(sub_poly)
        elif isinstance(intersection_poly, Polygon):
            relevant_poly_list.append(intersection_poly)
        elif isinstance(intersection_poly, LineString):
            relevant_poly_list.append(intersection_poly.buffer(0.1))
        elif isinstance(intersection_poly, Point):
            relevant_poly_list.append(Polygon(intersection_poly.buffer(0.1)))
        poly_list.append((relevant_poly_list, poly))

    if enc is not None and show_plots:
        enc.start_display()
        enc.draw_polygon(enveloping_polygon, color="yellow", alpha=0.2)
        for poly_sublist, _ in poly_list:
            for poly in poly_sublist:
                enc.draw_polygon(poly, color="red", fill=True, alpha=0.5)

    return poly_list, enveloping_polygon


def extract_boundary_polygons_inside_envelope(
    poly_tuple_list: list, enveloping_polygon: Polygon, enc: ENC | None = None, show_plots: bool = True
) -> list:
    """Extracts the boundary trianguled polygons.

    Relevant for the trajectory of the vessel, inside the given envelope polygon.

    Args:
        poly_tuple_list (list): List of tuples with relevant polygons inside
            query/envelope polygon and the corresponding original polygon they
            belong to.
        enveloping_polygon (Polygon): The query polygon.
        enc (ENC | None): Electronic Navigational Chart object used for
            plotting. Defaults to None.
        show_plots (bool): Whether to show plots or not. Defaults to True.

    Returns:
        list: List of boundary polygons.
    """
    boundary_polygons = []
    for relevant_poly_list, original_polygon in poly_tuple_list:
        for relevant_polygon in relevant_poly_list:
            triangle_boundaries = extract_triangle_boundaries_from_polygon(
                relevant_polygon, enveloping_polygon, original_polygon
            )
            if not triangle_boundaries:
                continue

            if enc is not None and show_plots:
                # enc.draw_polygon(poly, color="pink", alpha=0.3)
                for tri in triangle_boundaries:
                    enc.draw_polygon(tri, color="red", fill=False)

            boundary_polygons.extend(triangle_boundaries)
    return boundary_polygons


def extract_triangle_boundaries_from_polygon(
    polygon: Polygon,
    planning_area_envelope: Polygon,  # noqa: ARG001
    original_polygon: Polygon,
) -> list:
    """Extracts the triangles that comprise the boundary of the polygon.

    Triangles are filtered out if they have two vertices on the envelope boundary and is inside of the original
    polygon.

    Args:
        polygon (Polygon): The polygon in consideration inside the envelope polygon.
        planning_area_envelope (Polygon): A polygon representing the relevant area the vessel is planning to
            navigate in.
        original_polygon (Polygon): The original polygon that the relevant polygon belongs to.

    Returns:
        list: List of shapely polygons representing the boundary triangles for the polygon.
    """
    cdt = constrained_delaunay_triangulation_custom(polygon)
    # return cdt
    original_polygon_boundary = LineString(original_polygon.exterior.coords).buffer(0.0001)
    boundary_triangles = []
    if len(cdt) == 1:
        return cdt

    # Check if triangle has two vertices on the envelope boundary and is inside of the original polygon
    for tri in cdt:
        v_count = 0
        idx_prev = 0
        for idx, v in enumerate(tri.exterior.coords):
            if v_count == 2 and idx_prev == idx - 1 and tri not in boundary_triangles:
                boundary_triangles.append(tri)
                break
            v_point = Point(v)
            if original_polygon_boundary.contains(v_point):
                v_count += 1
                idx_prev = idx

    return boundary_triangles


def constrained_delaunay_triangulation_custom(polygon: Polygon) -> list:
    """Converts a polygon to a list of triangles.

    Basically constrained delaunay triangulation.

    Args:
        polygon (Polygon): The polygon to triangulate.

    Returns:
        list: List of triangles as shapely polygons.
    """
    if polygon.is_empty:
        msg = "Polygon is empty"
        raise ValueError(msg)
    res_intersection_gdf = gpd.GeoDataFrame(geometry=[polygon])
    # Create ID to identify overlapping polygons
    res_intersection_gdf["TRI_ID"] = res_intersection_gdf.index
    # List to keep triangulated geometries
    triangles = []
    # List to keep the original IDs
    triangle_ids = []
    # Triangulate single or multi-polygons
    for i, _ in res_intersection_gdf.iterrows():
        tri_ = ops.triangulate(res_intersection_gdf.geometry.values[i])
        triangles.append(tri_)
        for _ in range(0, len(tri_)):
            triangle_ids.append(res_intersection_gdf.TRI_ID.values[i])
    # Check if it is a single or multi-polygon
    len_list = len(triangles)
    triangles = np.array(triangles).flatten().tolist()
    # unlist geometries for multi-polygons
    if len_list > 1:
        triangles = [item for sublist in triangles for item in sublist]
    # Create triangulated polygons
    filtered_triangles = gpd.GeoDataFrame(triangles)
    filtered_triangles = filtered_triangles.set_geometry(triangles)
    del filtered_triangles[0]
    # Assign original IDs to each triangle
    filtered_triangles["TRI_ID"] = triangle_ids
    # Create new ID for each triangle
    filtered_triangles["LINK_ID"] = filtered_triangles.index
    # Create centroids from all triangles
    filtered_triangles["centroid"] = filtered_triangles.centroid
    filtered_triangles_centroid = filtered_triangles.set_geometry("centroid")
    del filtered_triangles_centroid["geometry"]
    del filtered_triangles["centroid"]
    # Find triangle centroids inside original polygon
    filtered_triangles_join = gpd.sjoin(
        filtered_triangles_centroid[["centroid", "TRI_ID", "LINK_ID"]],
        res_intersection_gdf[["geometry", "TRI_ID"]],
        how="inner",
        predicate="within",
    )
    # Remove overlapping from other triangles (Necessary for multi-polygons overlapping or close to each other)
    filtered_triangles_join = filtered_triangles_join[
        filtered_triangles_join["TRI_ID_left"] == filtered_triangles_join["TRI_ID_right"]
    ]
    # Remove overload triangles from same filtered_triangless
    filtered_triangles = filtered_triangles[filtered_triangles["LINK_ID"].isin(filtered_triangles_join["LINK_ID"])]
    filtered_triangles = filtered_triangles.geometry.values
    # double check
    cdt_triangles = []
    area_eps = 1e-4
    for tri in triangles:
        intersection_poly = tri.intersection(polygon)
        if isinstance(intersection_poly, Point) or isinstance(intersection_poly, LineString):
            continue
        if intersection_poly.area < area_eps:
            continue

        if isinstance(intersection_poly, MultiPolygon) or isinstance(intersection_poly, GeometryCollection):
            for sub_poly in intersection_poly.geoms:
                if sub_poly.area < area_eps or isinstance(sub_poly, Point) or isinstance(sub_poly, LineString):
                    continue
                cdt_triangles.append(sub_poly)
        else:
            cdt_triangles.append(intersection_poly)
    return cdt_triangles


def create_path_polygon(
    waypoints: np.ndarray,
    point_buffer: float | None = 10,
    disk_buffer: float | None = 80,
    hole_buffer: float | None = 10,
    show_annuluses: bool | None = True,
) -> Polygon:
    """Creates a polygon from the waypoints with annuluses around each waypoint if chosen.

    Args:
        waypoints (np.ndarray): Waypoints to create polygon from, 2 x n_samples.
        point_buffer (Optional[float]): Buffer size for the points. Defaults to 10.
        disk_buffer (Optional[float]): Buffer size for the disks around each waypoint. Defaults to 80.
        hole_buffer (Optional[float]): Buffer size for the holes inside each disk. Defaults to 10.
        show_annuluses (Optional[bool]): Option for showing annuluses around each waypoint. Defaults to True.

    Returns:
        Polygon: Path polygon for the waypoints
    """
    lines = [
        LineString([(wp1[1], wp1[0]), (wp2[1], wp2[0])]).buffer(point_buffer)
        for wp1, wp2 in zip(waypoints.T, waypoints[:, 1:].T, strict=False)
    ]
    if show_annuluses:
        points = [Point((wp[1], wp[0])) for wp in waypoints.T]
        disks = [p.buffer(disk_buffer) for p in points]
        holes = [p.buffer(hole_buffer) for p in points]
        path = shapely.unary_union(lines + disks)
        for _i, hole in enumerate(holes):
            path = path.difference(hole)
    else:
        lines.pop(1)
        path = shapely.unary_union(lines)
    return path


def standardize_polygon_intersections(intersection: Point | LineString | MultiLineString) -> Point:
    """Converts a shapely intersection to a point.

    If intersection contains multiple points, the closest one is returned.

    Args:
        intersection (Point | LineString | MultiLineString): The intersection
            to convert.

    Returns:
        Point: Shapely point object containing the closest point of intersection

    """
    if isinstance(intersection, LineString):
        return Point(intersection.coords[0])
    elif isinstance(intersection, Point):
        return intersection
    elif isinstance(intersection, MultiLineString):
        return Point(intersection.geoms[0].coords[0])
