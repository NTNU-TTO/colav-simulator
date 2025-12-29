"""plotters.py.

Summary:
Contains methods for plotting images and data.

Author: Trym Tengesdal
"""

import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seacharts.enc as senc
from matplotlib.collections import PolyCollection
from seacharts.display import colors
from shapely.geometry import MultiPolygon, Polygon

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm


def plot_image(image: np.ndarray, ax: plt.Axes | None = None, title: str | None = None) -> plt.Axes:
    """Plots an image.

    Args:
        image (np.ndarray): Image to plot.
        ax (Optional[plt.Axes]): Matplotlib axes handle.
        title (Optional[str]): Title of the plot.

    Returns:
        plt.Axes: Matplotlib axes handle.
    """
    mpl.use("TkAgg")
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    plt.show(block=False)
    return ax


def plot_trajectory(
    trajectory: np.ndarray,
    enc: senc.ENC,
    color: str,
    edge_style: str | None = None,
    buffer: float | None = 0.5,
    linewidth: float | None = 1.0,
    alpha: float | None = 1.0,
) -> None:
    """Plots the trajectory on the ENC.

    Args:
        trajectory (np.ndarray): Input trajectory, minimum 2 x n_samples.
        enc (senc.ENC): Electronic Navigational Chart object.
        color (str): Color of the trajectory.
        edge_style (str | None): Edge style for the trajectory. Defaults to None.
        buffer (float | None): Buffer of the trajectory. Defaults to 0.5.
        linewidth (float | None): Linewidth of the trajectory. Defaults to 1.0.
        alpha (float | None): Alpha transparency of the trajectory. Defaults to 1.0.
    """
    enc.start_display()
    trajectory_line = []
    for k in range(trajectory.shape[1]):
        trajectory_line.append((trajectory[1, k], trajectory[0, k]))
    return enc.draw_line(
        trajectory_line,
        color=color,
        buffer=buffer,
        linewidth=linewidth,
        edge_style=edge_style,
        alpha=alpha,
    )


def plot_disturbance(
    magnitude: float,
    direction: float,
    name: str,
    enc: senc.ENC,
    color: str,
    linewidth: float | None = 2.5,
    location: str | None = "topright",
    text_location_offset: tuple[float, float] | None = (0.0, 0.0),
) -> list:
    """Plots a disturbance vector on the ENC as a vector arrow inside a circle.

    The name of the disturbance is plotted below the circle, with an offset
    given by text_location_offset.

    Args:
        magnitude (float): Magnitude of the disturbance / length of the
            disturbance vector.
        direction (float): Direction of the disturbance (defined in a
            north-east coordinate system).
        name (str): Name of the disturbance.
        enc (senc.ENC): Electronic Navigational Chart object.
        color (str): Color of the disturbance vector.
        linewidth (float | None): Arrow thickness. Defaults to 2.5.
        location (str | None): Location of the disturbance vector in
            ["topleft", "topright", "bottomleft", "bottomright"]. Defaults to
            "topright".
        text_location_offset (tuple[float, float] | None): Offset of the text
            location. Defaults to (0.0, 0.0).

    Returns:
        list: List of plot handles.
    """
    enc.start_display()
    xmin, ymin, xmax, ymax = enc.bbox  # x is east, y is north
    if location == "topright":
        origin = (xmax - 0.1 * (xmax - xmin), ymax - 0.1 * (ymax - ymin))
    elif location == "topleft":
        origin = (xmin + 0.1 * (xmax - xmin), ymax - 0.1 * (ymax - ymin))
    elif location == "bottomright":
        origin = (xmax - 0.1 * (xmax - xmin), ymin + 0.1 * (ymax - ymin))
    elif location == "bottomleft":
        origin = (xmin + 0.1 * (xmax - xmin), ymin + 0.1 * (ymax - ymin))

    arrow_start = origin
    arrow_end = (origin[0] + magnitude * np.sin(direction), origin[1] + magnitude * np.cos(direction))
    text_location = (
        origin[0] + text_location_offset[0] - 0.8 * magnitude,
        origin[1] - 1.2 * magnitude + text_location_offset[1],
    )

    circle_handle = enc.draw_circle(origin, radius=magnitude, color="white", fill=True, alpha=0.2)
    arrow_handle = enc.draw_arrow(arrow_start, arrow_end, color=color, width=linewidth, fill=True)
    text_handle = enc.draw_text(name, text_location, color=color, size=10)
    return [circle_handle, arrow_handle, text_handle]


def plot_shapely_multipolygon(
    ax: plt.Axes,
    mp: MultiPolygon,
    color: str,
    fill: bool = True,  # noqa: ARG001
    alpha: float = 1.0,
    zorder: int = 1,
) -> tuple[plt.Axes, list]:
    """Plots a shapely MultiPolygon object on a matplotlib axes.

    Args:
        ax (plt.Axes): Matplotlib axes handle.
        mp (MultiPolygon): MultiPolygon object to plot.
        color (str): Color of the MultiPolygon.
        fill (bool): Option for filling the MultiPolygon. Defaults to True.
        alpha (float): Transparency of the MultiPolygon. Defaults to 1.0.
        zorder (int): Z-order of the MultiPolygon. Defaults to 1.

    Returns:
        tuple[plt.Axes, list]: Tuple containing the matplotlib axes handle and
            a list of handles to the plotted MultiPolygon.
    """
    if isinstance(mp, Polygon):
        mp = MultiPolygon([mp])

    poly_verts = [np.array(poly.exterior.coords) for poly in mp.geoms]
    collection = PolyCollection(poly_verts, edgecolor=color, facecolor=color, alpha=alpha, zorder=zorder)
    collection = ax.add_collection(collection)
    return ax, collection


def plot_background(
    ax: plt.Axes,
    enc: senc.ENC,
    show_shore: bool = True,
    show_seabed: bool = True,
    dark_mode: bool = True,
    uniform_seabed_color: bool = False,
    land_color: str | None = None,
    shore_color: str | None = None,
) -> tuple[plt.Axes, dict]:
    """Creates a static background based on the input seacharts.

    Args:
        ax (plt.Axes): Matplotlib axes handle.
        enc (ENC): Electronic Navigational Chart object
        show_shore (bool, optional): Option for showing the shore. Defaults to True.
        show_seabed (bool, optional): Option for showing the seabed. Defaults to True.
        dark_mode (bool, optional): Option for dark mode. Defaults to True.
        uniform_seabed_color (bool): Option for using a uniform color for the
            seabed. Defaults to False.
        land_color (str | None): Color of the land to override the default.
            Defaults to None.
        shore_color (str | None): Color of the shore to override the default.
            Defaults to None.

    Returns:
        tuple[plt.Axes, dict]: Tuple containing the matplotlib axes handle
            and a dictionary of handles to the plotted background.
    """
    handles = {}
    # For every layer put in list and assign a color
    if enc.land:
        color = "#142c38" if dark_mode else colors.color_picker(enc.land.color)
        if land_color is not None:
            color = land_color
        ax, mp_land_handle = plot_shapely_multipolygon(ax, enc.land.geometry, color=color, zorder=enc.land.z_order)
        handles["land"] = mp_land_handle

    if show_shore and enc.shore:
        color = "#142c38" if dark_mode else colors.color_picker(enc.shore.color)
        if shore_color is not None:
            color = shore_color
        ax, mp_shore_handle = plot_shapely_multipolygon(ax, enc.shore.geometry, color=color, zorder=enc.shore.z_order)
        handles["shore"] = mp_shore_handle

    if show_seabed and enc.seabed:
        bins = len(enc.seabed.keys())
        count = 0
        seabed_handles = []
        for _, layer in enc.seabed.items():
            if uniform_seabed_color:
                rank = enc.seabed[0].z_order
                color = colors.color_picker(0, bins)
            else:
                rank = layer.z_order + count
                color = colors.color_picker(count, bins)
            ax, mp_seabed_handle = plot_shapely_multipolygon(ax, layer.geometry, color=color, zorder=rank)
            count += 1
            seabed_handles.append((mp_seabed_handle, color))
        handles["seabed"] = seabed_handles

    x_min, y_min, x_max, y_max = enc.bbox
    ax.set_xlim((x_min, x_max))  # Easting
    ax.set_ylim((y_min, y_max))  # Northing
    return ax, handles


def plot_waypoints(
    waypoints: np.ndarray,
    enc: senc.ENC,
    color: str,
    point_buffer: float | None = 10,
    disk_buffer: float | None = 80,
    hole_buffer: float | None = 10,
    linewidth: float | None = None,
    alpha: float | None = 0.6,
    show_annuluses: bool | None = True,
    draft: float | None = 5.0,  # noqa: ARG001
) -> None:
    """Plot waypoints on the ENC.

    Args:
        waypoints (np.ndarray): Waypoint coordinates.
        enc (senc.ENC): Electronic Navigational Chart object.
        color (str): Color of the waypoint path.
        point_buffer (float | None): Buffer around waypoints. Defaults to 10.
        disk_buffer (float | None): Disk buffer. Defaults to 80.
        hole_buffer (float | None): Hole buffer. Defaults to 10.
        linewidth (float | None): Line width. Defaults to None.
        alpha (float | None): Transparency. Defaults to 0.6.
        show_annuluses (bool | None): Whether to show annuluses. Defaults to True.
        draft (float | None): Draft of the vessel. Defaults to 5.0.
    """
    if waypoints.shape[1] <= 1:
        msg = "Waypoints must have at least two points"
        raise ValueError(msg)
    path = mapf.create_path_polygon(waypoints, point_buffer, disk_buffer, hole_buffer, show_annuluses)

    # hazards = extract_relevant_grounding_hazards_as_union(find_minimum_depth(draft, enc), enc)[0]
    # if path.intersects(hazards):
    #     overlap = path.intersection(hazards)
    #     enc.draw_polygon(overlap, "red", thickness=linewidth, alpha=alpha)
    #     path = path.difference(hazards)
    return enc.draw_polygon(path, color, thickness=linewidth, alpha=alpha)


def plot_dynamic_obstacles(
    dynamic_obstacles: list, color: str, enc: senc.ENC, T: float, dt: float, map_origin: np.ndarray | None = None
) -> None:
    """Plots the dynamic obstacles as ellipses and ship polygons.

    Args:
        dynamic_obstacles (list): List of tuples containing (ID, state, cov, length, width)
        color (string): Color of the ellipses
        enc (ENC): Electronic Navigational Chart object
        T (float): Horizon to predict straight line trajectories for the dynamic obstacles
        dt (float): Time step for the straight line trajectories
        map_origin (np.ndarray, optional): Origin of the map in the form [x, y]^T
    """
    N = int(T / dt)
    enc.start_display()
    dynamic_obstacles_copy = copy.deepcopy(dynamic_obstacles)
    for _ID, state, cov, length, width in dynamic_obstacles_copy:
        if map_origin is not None:
            state[:2] += map_origin
        ellipse_x, ellipse_y = mhm.create_probability_ellipse(cov, 0.67)
        Polygon(zip(ellipse_y + state[1], ellipse_x + state[0], strict=False))
        # enc.draw_polygon(ell_geometry, color=color, alpha=0.4)

        for k in range(0, N, 5):
            do_poly = mapf.create_ship_polygon(
                state[0] + k * dt * state[2],
                state[1] + k * dt * state[3],
                np.arctan2(state[3], state[2]),
                length,
                width,
                length_scaling=1.0,
                width_scaling=1.0,
            )
            enc.draw_polygon(do_poly, color=color)
        do_poly = mapf.create_ship_polygon(
            state[0], state[1], np.arctan2(state[3], state[2]), length, width, length_scaling=1.0, width_scaling=1.0
        )
        enc.draw_polygon(do_poly, color=color)


def plot_rrt_tree(node_list: list, enc: senc.ENC) -> None:
    """Plots an RRT tree given by the list of nodes containing (state, parent_id, id, trajectory, inputs, cost).

    Args:
        node_list (list): List of nodes containing (state, parent_id, id, trajectory, inputs, cost)
        enc (ENC): Electronic Navigational Chart object
    """
    enc.start_display()
    for node in node_list:
        # enc.draw_circle(
        #     (node["state"][1], node["state"][0]), 2.5, color="green", fill=False, thickness=0.8, edge_style=None
        # )
        for sub_node in node_list:
            if node["id"] == sub_node["id"] or sub_node["parent_id"] != node["id"]:
                continue
            points = [(tt[1], tt[0]) for tt in sub_node["trajectory"]]
            if len(points) > 1:
                enc.draw_line(points, color="white", buffer=0.5, linewidth=0.5)
