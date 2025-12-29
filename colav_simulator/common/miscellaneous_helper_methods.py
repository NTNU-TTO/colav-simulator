import math
import os.path
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psutil
from scipy.interpolate import interp1d
from scipy.stats import chi2
from shapely import geometry

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.vessel_data as vd


def print_resource_usage() -> None:  # noqa: D103
    memusage = psutil.virtual_memory()
    cpuusage = psutil.cpu_percent()
    print(f"[System] memory usage: {memusage.percent}% | CPU usage: {cpuusage}%")


def print_process_memory_usage(prefix_str: str) -> None:
    """Prints the memory usage of the current process.

    Args:
        prefix_str (str): Prefix string to print before the memory usage.
    """
    process = psutil.Process(os.getpid())
    print(f"[Process {os.getpid()}] {prefix_str} Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")


def normalize_mpc_param(
    x: np.ndarray,
    param_list: list[str],
    parameter_ranges: dict[str, Any],
    parameter_lengths: dict[str, Any],
    parameter_indices: dict[str, Any],
) -> np.ndarray:
    """Normalize the input parameter.

    Args:
        x (np.ndarray): The unnormalized parameter
        param_list (List[str]): The list of parameters to map.
        parameter_ranges (dict[str, Any]): The parameter ranges.
        parameter_lengths (dict[str, Any]): The parameter lengths.
        parameter_indices (dict[str, Any]): The parameter indices.

    Returns:
        np.ndarray: The normalized parameter
    """
    x_norm = np.zeros_like(x, dtype=np.float32)
    for param_name in param_list:
        param_range = parameter_ranges[param_name]
        param_length = parameter_lengths[param_name]
        pindx = parameter_indices[param_name]
        x_param = x[pindx : pindx + param_length].copy()

        for j in range(len(x_param)):  # pylint: disable=consider-using-enumerate
            if param_name == "Q_p":
                x_param[j] = mf.linear_map(x_param[j], tuple(param_range[j]), (-1.0, 1.0))
            else:
                x_param[j] = mf.linear_map(x_param[j], tuple(param_range), (-1.0, 1.0))
        x_norm[pindx : pindx + param_length] = x_param
    return x_norm


def unnormalize_mpc_param(
    x: np.ndarray,
    param_list: list[str],
    parameter_ranges: dict[str, Any],
    parameter_lengths: dict[str, Any],
    parameter_indices: dict[str, Any],
) -> np.ndarray:
    """Unnormalize the input parameter.

    Args:
        x (np.ndarray): The normalized parameter
        param_list (List[str]): The list of parameters to map.
        parameter_ranges (dict[str, Any]): The parameter ranges.
        parameter_lengths (dict[str, Any]): The parameter lengths.
        parameter_indices (dict[str, Any]): The parameter indices.

    Returns:
        np.ndarray: The unnormalized output as a numpy array
    """
    x_unnorm = np.zeros_like(x, dtype=np.float32)
    for param_name in param_list:
        param_range = parameter_ranges[param_name]
        param_length = parameter_lengths[param_name]
        pindx = parameter_indices[param_name]
        x_param = x[pindx : pindx + param_length].copy()

        for j in range(len(x_param)):  # pylint: disable=consider-using-enumerate
            if param_name == "Q_p":
                x_param[j] = mf.linear_map(x_param[j], (-1.0, 1.0), tuple(param_range[j]))
            else:
                x_param[j] = mf.linear_map(x_param[j], (-1.0, 1.0), tuple(param_range))
        x_unnorm[pindx : pindx + param_length] = x_param
    return x_unnorm


def get_ship_ais_df_list_from_ais_df(df: pd.DataFrame) -> list:
    """Returns a list of DataFrames, where each DataFrame contains AIS_data for a ship.

    Args:
        df (pd.DataFrame): DataFrame containing AIS_data.

    Returns:
        list: List of mmsi [DF_mmsi_1, DF_mmsi_2,..., DF_mmsi_n].
    """
    mmsi_list = df.mmsi.unique().tolist()
    mmsi_df_list = [df[df.mmsi == mmsi].reset_index(drop=True) for mmsi in mmsi_list]
    return mmsi_df_list


def create_arc_length_spline(x: list, y: list) -> tuple[interp1d, interp1d, np.ndarray]:
    """Creates a spline for the arc length of the input x and y coordinates.

    Args:
        x (list): List of x coordinates.
        y (list): List of y coordinates.

    Returns:
        tuple[interp1d, interp1d, np.ndarray]: Tuple of arc length splines for
            x and y coordinates.
    """
    # Interpolate the data to get more points => higher accuracy in the arc length spline
    n_points = len(x)
    y_interp = interp1d(np.arange(n_points), y, kind="linear")
    x_interp = interp1d(np.arange(n_points), x, kind="linear")

    n_expanded_points = 500
    y_expanded = list(y_interp(np.linspace(0, n_points - 1, n_expanded_points)))
    x_expanded = list(x_interp(np.linspace(0, n_points - 1, n_expanded_points)))
    arc_length = [0.0]
    for i in range(1, n_expanded_points):
        pi = np.array([x_expanded[i - 1], y_expanded[i - 1]])
        pj = np.array([x_expanded[i], y_expanded[i]])
        arc_length.append(np.linalg.norm(pi - pj))
    arc_length = np.cumsum(arc_length)
    y_interp_arc_length = interp1d(arc_length, y_expanded, kind="linear")
    x_interp_arc_length = interp1d(arc_length, x_expanded, kind="linear")
    return x_interp_arc_length, y_interp_arc_length, arc_length


def linestring_to_ndarray(line: geometry.LineString) -> np.ndarray:
    """Converts a shapely LineString to a numpy array.

    Args:
        line (LineString): Any LineString object

    Returns:
        np.ndarray: Numpy array containing the coordinates of the LineString
    """
    return np.array(line.coords).transpose()


def ndarray_to_linestring(array: np.ndarray) -> geometry.LineString:
    """Converts a 2D numpy array to a shapely LineString.

    Args:
        array (np.ndarray): Numpy array of 2 x n_samples, containing the coordinates of the LineString

    Returns:
        LineString: Any LineString object
    """
    if not (array.shape[0] == 2 and array.shape[1] > 1):
        msg = "Array must be 2 x n_samples with n_samples > 1"
        raise ValueError(msg)
    return geometry.LineString(list(zip(array[0, :], array[1, :], strict=False)))


def check_if_vessel_is_passed_by(
    p_os: np.ndarray,
    v_os: np.ndarray,
    p_do: np.ndarray,
    v_do: np.ndarray,
    threshold_angle: float = 100.0,
    threshold_distance: float = 50.0,
) -> bool:
    """Checks if a vessel is passed by another vessel.

    Args:
        p_os (np.ndarray): Position of ownship.
        v_os (np.ndarray): Velocity of ownship.
        p_do (np.ndarray): Position of dynamic obstacle.
        v_do (np.ndarray): Velocity of dynamic obstacle.
        threshold_angle (float): Threshold angle in degrees. Defaults to 100.0.
        threshold_distance (float): Threshold distance in meters. Defaults to 50.0.

    Returns:
        bool: True if ownship is passed by dynamic obstacle, False otherwise.
    """
    dist_os_do = p_do - p_os
    L_os_do = dist_os_do / np.linalg.norm(dist_os_do)
    U_os = np.linalg.norm(v_os)
    U_do = np.linalg.norm(v_do)

    os_is_overtaken = np.dot(v_os, v_do) > np.cos(np.deg2rad(68.5)) * U_os * U_do and U_os < U_do and U_os > 0.25
    do_is_overtaken = np.dot(v_do, v_os) > np.cos(np.deg2rad(68.5)) * U_do * U_os and U_do < U_os and U_do > 0.25

    vessel_is_passed: bool = (
        (np.dot(v_os, L_os_do) < np.cos(np.deg2rad(threshold_angle)) * U_os and not os_is_overtaken)
        or (np.dot(v_do, -L_os_do) < np.cos(np.deg2rad(threshold_angle)) * U_do and not do_is_overtaken)
        and np.linalg.norm(dist_os_do) > threshold_distance
    )

    return vessel_is_passed


def compute_triangulation_weights(cdt: list) -> list:
    """Computes the weights of the triangulation.

    Args:
        cdt (list): Constrained Delaunay triangulation.

    Returns:
        list: List of weights for each triangle.
    """
    weights = []
    for triangle in cdt:
        if not (isinstance(triangle, geometry.Polygon) and len(triangle.exterior.coords) >= 4):
            msg = "The triangulation must be a polygon and triangle."
            raise ValueError(msg)
        weights.append(triangle.area)
    total_area = np.sum(np.array(weights))
    normalized_weights_arr = np.array(weights) / total_area
    return normalized_weights_arr.tolist()


def sample_from_triangulation(rng: np.random.Generator, triangulation: list, triangulation_weights: list) -> np.ndarray:
    """Samples a point from a triangulation.

    Args:
        rng (np.random.Generator): Numpy random generator.
        triangulation (list): List of triangles (polygons).
        triangulation_weights (list): List of weights for each triangle.

    Returns:
        np.ndarray: Sampled point.
    """
    random_triangle = rng.choice(triangulation, p=triangulation_weights)
    if not (isinstance(random_triangle, geometry.Polygon) and len(random_triangle.exterior.coords) >= 4):
        msg = "The triangulation must be a polygon and triangle."
        raise ValueError(msg)
    x, y = random_triangle.exterior.coords.xy
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[1], y[1]])
    p3 = np.array([x[2], y[2]])
    random_point = sample_from_triangle_region(rng, p1, p2, p3)
    return random_point


def sample_from_triangle_region(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Samples a point from the triangle region defined by p1, p2 and p3.

    Ref: Osada et. al. 2002.

    Args:
        rng (np.random.Generator): Numpy random generator.
        p1 (np.ndarray): Vertex 1 coordinate.
        p2 (np.ndarray): Vertex 2 coordinate.
        p3 (np.ndarray): Vertex 3 coordinate.

    Returns:
        np.ndarray: Sampled point.
    """
    r_1 = rng.uniform(0, 1)
    r_2 = rng.uniform(0, 1)
    p = (1.0 - np.sqrt(r_1)) * p1 + np.sqrt(r_1) * (1.0 - r_2) * p2 + np.sqrt(r_1) * r_2 * p3
    return p


def compute_path_length(path: np.ndarray) -> float:
    """Computes the length of a path.

    Args:
        path (np.ndarray): Path coordinates 2 x n_samples.

    Returns:
        float: Length of the path.
    """
    if path.shape[0] != 2:
        msg = "Path must be 2 x n_samples"
        raise ValueError(msg)
    if path.shape[1] <= 1:
        msg = "Path must have at least 2 samples"
        raise ValueError(msg)
    length = 0.0
    for k in range(1, path.shape[1]):
        length += np.linalg.norm(path[:, k] - path[:, k - 1])
    return length


def parse_rrt_solution(soln: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parses the RRT solution.

    Args:
        soln (dict): Solution dictionary.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of waypoints, trajectory, inputs and times
            from the solution.
    """
    times = np.array(soln["times"])
    n_samples = len(times)
    trajectory = np.zeros((6, n_samples))
    n_inputs = len(soln["inputs"])
    inputs = np.zeros((3, n_inputs))
    n_wps = len(soln["waypoints"])
    waypoints = np.zeros((3, n_wps))
    if n_samples > 0:
        for k in range(n_wps):
            waypoints[:, k] = np.array(soln["waypoints"][k])
        for k in range(n_samples):
            trajectory[:, k] = np.array(soln["states"][k])
        for k in range(n_inputs):
            inputs[:, k] = np.array(soln["inputs"][k])
    if n_wps == 1:
        waypoints = np.array([waypoints[:, 0], waypoints[:, 0]]).T
    return waypoints, trajectory, inputs, times


def sample_from_waypoint_corridor(rng: np.random.Generator, waypoints: np.ndarray, corridor_width: float) -> np.ndarray:
    """Samples a point from a random segment of the waypoint corridor.

    Args:
        rng (np.random.Generator): Numpy random generator.
        waypoints (np.ndarray): Waypoint data (2 x n_waypoints).
        corridor_width (float): Width of the corridor.

    Returns:
        np.ndarray: Sampled point.
    """
    if waypoints.shape[0] != 2:
        msg = "Waypoints must be 2 x n_waypoints"
        raise ValueError(msg)
    n_wps = waypoints.shape[1]
    if n_wps <= 1:
        msg = "Must have at least 2 waypoints"
        raise ValueError(msg)
    segment_idx = rng.integers(0, n_wps - 1)
    segment_length = np.linalg.norm(waypoints[:, segment_idx + 1] - waypoints[:, segment_idx])
    x = rng.uniform(0.0, segment_length)
    y = rng.uniform(-corridor_width / 2.0, corridor_width / 2.0)
    alpha = np.arctan2(
        waypoints[1, segment_idx + 1] - waypoints[1, segment_idx],
        waypoints[0, segment_idx + 1] - waypoints[0, segment_idx],
    )
    return mf.Rmtrx2D(alpha) @ np.array([x, y]) + waypoints[:, segment_idx]


def check_if_situation_is_risky_enough(
    os_csog_state: np.ndarray, do_csog_state: np.ndarray, t_cpa_threshold: float, d_cpa_threshold: float
) -> bool:
    """Checks if a vessel-vessel situation is risky enough to be considered a COLAV situation.

    Args:
        os_csog_state (np.ndarray): Ownship CSOG state on the form [x, y, U, chi]^T.
        do_csog_state (np.ndarray): Dynamic obstacle CSOG state on the form [x, y, U, chi]^T.
        t_cpa_threshold (float): Threshold for the CPA time.
        d_cpa_threshold (float): Threshold for the CPA distance.

    Returns:
        bool: Whether the situation is risky enough or not.
    """
    p_os = os_csog_state[0:2]
    v_os = np.array([os_csog_state[2] * np.cos(os_csog_state[3]), os_csog_state[2] * np.sin(os_csog_state[3])])
    p_do = do_csog_state[0:2]
    v_do = np.array([do_csog_state[2] * np.cos(do_csog_state[3]), do_csog_state[2] * np.sin(do_csog_state[3])])
    t_cpa, d_cpa, _d_cpa_vec = compute_vessel_pair_cpa(p_os, v_os, p_do, v_do)
    return t_cpa < t_cpa_threshold and d_cpa < d_cpa_threshold


def trajectory_from_waypoints_and_speed(waypoints: np.ndarray, speed_plan: np.ndarray, dt: float, T: float) -> np.ndarray:
    """Creates a simplistic trajectory from the waypoints and speed plan of the vessel.

    Args:
        waypoints (np.ndarray): Waypoint data (2 x n_waypoints).
        speed_plan (np.ndarray): Speed plan data (1 x n_waypoints).
        dt (float): Time step.
        T (float): Total time.

    Returns:
        np.ndarray: Trajectory of the vessel on the form [x, y, U, chi] x n_samples.
    """
    p = waypoints[:, 0]
    p_end = waypoints[:, -1]
    n_wps = waypoints.shape[1]
    n_samples = int(T / dt)
    wp_leg = 0
    traj = np.zeros((4, n_samples))
    for k in range(n_samples):
        speed = speed_plan[wp_leg]
        if np.linalg.norm(p - p_end) < 10.0:
            speed = 0.0

        wp_idx = wp_leg + 1 if wp_leg < n_wps - 1 else wp_leg
        alpha = np.arctan2(waypoints[1, wp_idx] - waypoints[1, wp_idx - 1], waypoints[0, wp_idx] - waypoints[0, wp_idx - 1])

        traj[0:2, k] = p
        traj[2, k] = speed_plan[wp_leg]
        traj[3, k] = alpha

        p = p + speed * np.array([np.cos(alpha), np.sin(alpha)]) * dt

        if np.linalg.norm(p - waypoints[:, wp_idx]) < 1.0 and wp_leg < n_wps - 1:
            wp_leg += 1

    return traj


def compute_actual_vessel_pair_cpa(traj_1: np.ndarray, traj_2: np.ndarray, dt: float) -> tuple[float, float, np.ndarray]:
    """Computes the closest point of approach (CPA) between two vessel trajectories.

    Args:
        traj_1 (np.ndarray): Trajectory of vessel 1 on the form [x, y, U, chi] x n_samples.
        traj_2 (np.ndarray): Trajectory of vessel 2 on the form [x, y, U, chi] x n_samples.
        dt (float): Time step.

    Returns:
        tuple[float, float, np.ndarray]: The time to CPA, distance at CPA and corresponding CPA distance vector.
    """
    dist_vec_traj = traj_2[0:2, :] - traj_1[0:2, :]
    distances = np.linalg.norm(dist_vec_traj, axis=0)
    min_dist_idx = int(np.argmin(distances))
    t_cpa = min_dist_idx * dt
    return t_cpa, distances[min_dist_idx], dist_vec_traj[:, min_dist_idx]


def compute_vessel_pair_cpa(
    p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Computes the closest point of approach (CPA) between two vessel when assumed to travel with constant velocity.

    Args:
        p1 (np.ndarray): Position of vessel 1.
        v1 (np.ndarray): Velocity of vessel 1.
        p2 (np.ndarray): Position of vessel 2.
        v2 (np.ndarray): Velocity of vessel 2.

    Returns:
        tuple[float, float, np.ndarray]: The time to CPA, distance at CPA and corresponding CPA distance vector.
    """
    # Compute the relative position and velocity
    r = p2 - p1
    v = v2 - v1

    if np.dot(v, v) < 0.0001:
        return 0.0, float(np.linalg.norm(r)), p2 - p1

    t_cpa = -np.dot(r, v) / np.dot(v, v)
    d_cpa_vec = r + t_cpa * v
    d_cpa = float(np.linalg.norm(d_cpa_vec))
    return t_cpa, d_cpa, d_cpa_vec


def convert_simulation_data_to_vessel_data(
    sim_data: pd.DataFrame, ship_info: dict[str, Any], utm_zone: int
) -> list[vd.VesselData]:
    """Converts simulation data to vessel data.

    Args:
        sim_data (pd.DataFrame): Simulation data.
        ship_info (dict): Information on all ships in the simulation.
        utm_zone (int): UTM zone used for the planar xy coordinate system.

    Returns:
        List[vd.VesselData]: List of vessel data containers
    """
    vessels = []
    identifier = 0
    for name, ship_i_info in ship_info.items():
        vessel = vd.VesselData(
            id=identifier,
            name=name,
            mmsi=ship_i_info["mmsi"],
            length=ship_i_info["length"],
            width=ship_i_info["width"],
            draft=ship_i_info["draft"],
        )

        X, _, _, vessel.timestamps, vessel.datetimes_utc = extract_trajectory_data_from_dataframe(sim_data[name])
        if X.size == 0:
            continue

        vessel.first_valid_idx, vessel.last_valid_idx = index_of_first_and_last_non_nan(X[0, :])
        n_msgs = len(vessel.timestamps)
        vessel.xy = np.zeros((2, len(X[0, :])))
        vessel.xy[0, :] = X[1, :]  # ENU frame is used in the evaluator
        vessel.xy[1, :] = X[0, :]
        vessel.sog = X[3, :]
        vessel.cog = X[2, :]

        vessel.latlon = np.zeros(vessel.xy.shape) * np.nan
        (
            vessel.latlon[0, vessel.first_valid_idx : vessel.last_valid_idx + 1],
            vessel.latlon[1, vessel.first_valid_idx : vessel.last_valid_idx + 1],
        ) = mapf.local2latlon(
            vessel.xy[0, vessel.first_valid_idx : vessel.last_valid_idx + 1],
            vessel.xy[1, vessel.first_valid_idx : vessel.last_valid_idx + 1],
            utm_zone,
        )

        vessel.forward_heading_estimate = np.zeros(n_msgs) * np.nan
        vessel.backward_heading_estimate = np.zeros(n_msgs) * np.nan
        for k in range(vessel.first_valid_idx, vessel.last_valid_idx):
            vessel.forward_heading_estimate[k] = np.arctan2(
                vessel.xy[0, k + 1] - vessel.xy[0, k], vessel.xy[1, k + 1] - vessel.xy[1, k]
            )
        vessel.forward_heading_estimate[vessel.last_valid_idx] = vessel.forward_heading_estimate[vessel.last_valid_idx - 1]

        for k in range(vessel.first_valid_idx + 1, vessel.last_valid_idx):
            vessel.backward_heading_estimate[k] = np.arctan2(
                vessel.xy[0, k] - vessel.xy[0, k - 1], vessel.xy[1, k] - vessel.xy[1, k - 1]
            )
        vessel.backward_heading_estimate[vessel.first_valid_idx] = vessel.forward_heading_estimate[vessel.first_valid_idx]

        vessel.travel_dist = vd.compute_total_dist_travelled(
            vessel.xy[:, vessel.first_valid_idx : vessel.last_valid_idx + 1]
        )

        # print(f"Vessel {identifier} travelled a distance of {vessel.travel_dist} m")
        # print(f"Vessel status: {vessel.status}")
        vessels.append(vessel)
        identifier += 1
    return vessels


def find_cpa_indices(trajectory_list: list) -> np.ndarray:
    """Find the indices of the ships that are closest to each other.

    Args:
        trajectory_list (list): List of trajectories.

    Returns:
        np.ndarray: Array containing the indices of the ships that are closest to each other.
    """
    n_ships = len(trajectory_list)
    cpa_indices = np.zeros((n_ships, n_ships), dtype=int) * np.nan

    for i in range(n_ships):
        for j in range(n_ships):
            if i == j:
                continue
            cpa_indices[i, j] = find_cpa_index(trajectory_list[i], trajectory_list[j])

    return cpa_indices


def find_cpa_index(trajectory_i: tuple[np.ndarray, np.ndarray], trajectory_j: tuple[np.ndarray, np.ndarray]) -> int | float:
    """Find the index of the closest point of approach between two ships.

    Args:
        trajectory_i (Tuple[np.ndarray, np.ndarray]): Trajectory of ship i with corresponding timestamps.
        trajectory_j (np.ndarray): Trajectory of ship j with corresponding timestamps.

    Returns:
        int: Index of the closest point of approach for the trajectory pair.
    """
    min_dist_idx = np.nan

    timestamps_i = trajectory_i["timestamps"]
    timestamps_j = trajectory_j["timestamps"]
    common_timestamps = np.intersect1d(timestamps_i, timestamps_j)
    relevant_indices_i = np.where(np.isin(timestamps_i, common_timestamps))[0]
    relevant_indices_j = np.where(np.isin(timestamps_j, common_timestamps))[0]
    ranges = np.zeros(relevant_indices_i.size)
    for idx, (t_i, t_j) in enumerate(zip(relevant_indices_i, relevant_indices_j, strict=False)):
        ranges[idx] = np.linalg.norm(trajectory_i["X"][0:2, t_i] - trajectory_j["X"][0:2, t_j])

    finite = np.where(~np.isnan(ranges))[0]
    if finite.size > 0:
        min_dist_idx = np.argmin(ranges[finite])
    return min_dist_idx


def extract_ship_data_from_sim_dataframe(ship_list: list, sim_data: pd.DataFrame) -> dict:
    """Extract the ship data from the simulation data.

    Args:
        ship_list (list): List of ship objects.
        sim_data (pd.DataFrame): Simulation data from the ships.

    Returns:
        dict: Dictionary containing the ship data related to the simulation.
    """
    output = {}
    trajectory_list = []
    colav_data = []
    for i, _ship in enumerate(ship_list):
        X, U, refs, timestamps, _ = extract_trajectory_data_from_dataframe(sim_data[f"Ship{i}"])
        colav_data_i = extract_colav_data_from_dataframe(sim_data[f"Ship{i}"])
        trajectory_list.append({"X": X, "U": U, "refs": refs, "timestamps": timestamps})
        colav_data.append(colav_data_i)

    output["trajectory_list"] = trajectory_list
    output["colav_data_list"] = colav_data
    cpa_indices = find_cpa_indices(trajectory_list)
    output["cpa_indices"] = cpa_indices
    return output


def extract_trajectory_data_from_dataframe(
    ship_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """Extract the trajectory related data from a ship dataframe.

    Args:
        ship_df (Dataframe): Dataframe containing the ship simulation data.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, list, list]: Tuple of array
            containing the trajectory and corresponding relative simulation
            timestamps and UTC timestamps.
    """
    state_list = []
    input_list = []
    reference_list = []
    timestamps = []
    datetimes_utc = []
    for _, ship_df_k in enumerate(ship_df):
        if pd.notna(ship_df_k) and ship_df_k:
            state_list.append(ship_df_k["state"])
            input_list.append(ship_df_k["input"])
            reference_list.append(ship_df_k["references"].reshape(-1))
            timestamps.append(float(ship_df_k["timestamp"]))
            datetime_utc = datetime.strptime(ship_df_k["date_time_utc"], "%d.%m.%Y %H:%M:%S")
            datetimes_utc.append(datetime_utc)
    X = np.array(state_list).T
    U = np.array(input_list).T
    refs = np.array(reference_list).T
    return X, U, refs, timestamps, datetimes_utc


def extract_colav_data_from_dataframe(ship_df: pd.DataFrame) -> list:
    """Extract the COLAV related data from a ship dataframe.

    Args:
        ship_df (pd.DataFrame): Dataframe containing the ship simulation data.

    Returns:
        list: List of COLAV data at each simulation timestamp.
    """
    colav_data = []
    for _, ship_df_k in enumerate(ship_df):
        if pd.notna(ship_df_k) and "colav" in ship_df_k:
            colav_data.append(ship_df_k["colav"])
    return colav_data


def extract_track_data_from_dataframe(ship_df: pd.DataFrame) -> dict:
    """Extract the dynamic obstacle track data from a ship dataframe.

    Args:
        ship_df (Dataframe): Dataframe containing the ship simulation data.

    Returns:
        Tuple[list, list]: List of dynamic obstacle estimates and covariances
    """
    output = {}
    do_estimates = []
    do_covariances = []
    do_NISes = []
    do_timestamps = []

    n_samples = len(ship_df)
    n_do = len(ship_df[n_samples - 1]["do_estimates"])
    do_labels = ship_df[n_samples - 1]["do_labels"]
    for _i in range(n_do):
        do_estimates.append(np.nan * np.ones((4, n_samples)))
        do_covariances.append(np.nan * np.ones((4, 4, n_samples)))
        do_NISes.append(np.nan * np.ones(n_samples))
        do_timestamps.append(np.nan * np.ones(n_samples))

    for k, ship_df_k in enumerate(ship_df):
        for idx, do_idx in enumerate(ship_df_k["do_labels"]):
            do_estimates[do_idx - 1][:, k] = ship_df_k["do_estimates"][idx]
            do_covariances[do_idx - 1][:, :, k] = ship_df_k["do_covariances"][idx]
            do_NISes[do_idx - 1][k] = ship_df_k["do_NISes"][idx]
            do_timestamps[do_idx - 1][k] = ship_df_k["timestamp"]

    output["do_estimates"] = do_estimates
    output["do_covariances"] = do_covariances
    output["do_NISes"] = do_NISes
    output["do_labels"] = do_labels
    output["timestamps"] = do_timestamps
    return output


def inside_bbox(point: np.ndarray, bbox: tuple[float, float, float, float]) -> bool:
    """Checks if a point is inside a bounding box.

    Args:
        point (np.ndarray): Point to check.
        bbox (Tuple[float, float, float, float]): Bounding box defined by [xmin, ymin, xmax, ymax].

    Returns:
        bool: True if point is inside bounding box, False otherwise.
    """
    return point[0] >= bbox[0] and point[0] <= bbox[2] and point[1] >= bbox[1] and point[1] <= bbox[3]


def clip_waypoint_segment_to_bbox(segment: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, bool]:
    """Clips a waypoint segment to within a bounding box.

    Args:
        segment (np.ndarray): Waypoint segment.
        bbox (Tuple[float, float, float, float]): Bounding box defined by [xmin, ymin, xmax, ymax].

    Returns:
        Tuple[np.ndarray, bool: Tuple of the new possibly clipped waypoint segment, and a boolean indicating
            if it was clipped or not.
    """
    segment_linestring = ndarray_to_linestring(segment)
    p1_inside_bbox = inside_bbox(segment[:, 0], bbox)
    p2_inside_bbox = inside_bbox(segment[:, 1], bbox)
    if p1_inside_bbox and p2_inside_bbox or not (p1_inside_bbox or p2_inside_bbox):
        return segment, False

    # check intersection with all bounding box line constraints
    # lower left corner
    p_clip = segment[:, 1]
    left_vertical = ndarray_to_linestring(np.array([[bbox[0], bbox[2]], [bbox[1], bbox[1]]]))
    intersection = segment_linestring.intersection(left_vertical)
    if intersection.geom_type == "Point":
        p_clip = np.array([intersection.x, intersection.y])

    right_vertical = ndarray_to_linestring(np.array([[bbox[0], bbox[2]], [bbox[3], bbox[3]]]))
    intersection = segment_linestring.intersection(right_vertical)
    if intersection.geom_type == "Point":
        p_clip = np.array([intersection.x, intersection.y])

    top_horizontal = ndarray_to_linestring(np.array([[bbox[2], bbox[2]], [bbox[1], bbox[3]]]))
    intersection = segment_linestring.intersection(top_horizontal)
    if intersection.geom_type == "Point":
        p_clip = np.array([intersection.x, intersection.y])

    bottom_horizontal = ndarray_to_linestring(np.array([[bbox[0], bbox[0]], [bbox[1], bbox[3]]]))
    intersection = segment_linestring.intersection(bottom_horizontal)
    if intersection.geom_type == "Point":
        p_clip = np.array([intersection.x, intersection.y])

    # reduce p_clip to avoid end points directly on the bounding box
    p_clip = segment[:, 0] + 0.95 * (p_clip - segment[:, 0])

    return np.array([segment[:, 0], p_clip]).transpose(), True


def check_if_trajectory_is_within_xy_limits(trajectory: np.ndarray, xlimits: list, ylimits: list) -> bool:
    """Checks if the trajectory is within the x and y limits.

    Args:
        trajectory (np.ndarray): Trajectory data.
        xlimits (list): List containing the x limits.
        ylimits (list): List containing the y limits.

    Returns:
        bool: True if trajectory is within limits, False otherwise.
    """
    min_x = np.min(trajectory[0, :])
    max_x = np.max(trajectory[0, :])
    min_y = np.min(trajectory[1, :])
    max_y = np.max(trajectory[1, :])
    return min_x >= xlimits[0] and max_x <= xlimits[1] and min_y >= ylimits[0] and max_y <= ylimits[1]


def update_xy_limits_from_trajectory_data(trajectory: np.ndarray, xlimits: list, ylimits: list) -> tuple[list, list]:
    """Update the x and y limits from the trajectory data.

    Either predefined trajectory or nominal trajectory/waypoints for the ship.

    Args:
        trajectory (np.ndarray): Trajectory or waypoint data.
        xlimits (list): List containing the x limits.
        ylimits (list): List containing the y limits.

    Returns:
        tuple[list, list]: x and y limits.
    """
    min_x = np.min(trajectory[0, :])
    max_x = np.max(trajectory[0, :])
    min_y = np.min(trajectory[1, :])
    max_y = np.max(trajectory[1, :])

    xlimits[0] = min(xlimits[0], min_x)

    xlimits[1] = max(xlimits[1], max_x)

    ylimits[0] = min(ylimits[0], min_y)

    ylimits[1] = max(ylimits[1], max_y)

    return xlimits, ylimits


def create_probability_ellipse(P: np.ndarray, probability: float = 0.99) -> tuple[list, list]:
    """Creates a probability ellipse for a covariance matrix P.

    Uses a given confidence level (default 0.99).

    Args:
        P (np.ndarray): Covariance matrix.
        probability (float): Confidence level. Defaults to 0.99.

    Returns:
        tuple[list, list]: Ellipse data in x and y coordinates.
    """
    # eigenvalues and eigenvectors of the covariance matrix
    eigenval, eigenvec = np.linalg.eig(P[0:2, 0:2])

    largest_eigenval = max(eigenval)
    largest_eigenvec_idx = np.argwhere(eigenval == max(eigenval))[0][0]
    largest_eigenvec = eigenvec[:, largest_eigenvec_idx]

    smallest_eigenval = min(eigenval)
    # if largest_eigenvec_idx == 0:
    #     smallest_eigenvec = eigenvec[:, 1]
    # else:
    #     smallest_eigenvec = eigenvec[:, 0]

    angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
    angle = mf.wrap_angle_to_02pi(angle)

    # Get the ellipse scaling factor based on the confidence level
    chisquare_val = chi2.ppf(q=probability, df=2)

    a = chisquare_val * math.sqrt(largest_eigenval)
    b = chisquare_val * math.sqrt(smallest_eigenval)

    # the ellipse in "body" x and y coordinates
    t = np.linspace(0, 2.0 * np.pi, 200)
    x = a * np.cos(t)
    y = b * np.sin(t)

    R = mf.Rmtrx2D(angle)

    # Rotate to NED by angle phi, N_ell_points x 2
    ellipse_xy = np.array([x, y])
    for i in range(len(ellipse_xy)):
        ellipse_xy[:, i] = R @ ellipse_xy[:, i]

    return ellipse_xy[0, :].tolist(), ellipse_xy[1, :].tolist()


def sample_state_along_waypoints(
    rng: np.random.Generator, waypoints: np.ndarray, speed_plan: np.ndarray, timespan: float
) -> tuple[np.ndarray, float]:
    """Samples a CSOG state along a set of waypoints.

    With course over ground aligned with the waypoint segment chosen, and
    corresponding speed ref.

    Args:
        rng (np.random.Generator): Numpy random generator.
        waypoints (np.ndarray): Waypoint data.
        speed_plan (np.ndarray): Speed plan data.
        timespan (float): Total time span to consider.

    Returns:
        tuple[np.ndarray, float]: Sampled state data along the waypoints, and
            the corresponding approximate vessel time of arrival.
    """
    if not (waypoints.shape[0] == 2 and waypoints.shape[1] > 1):
        msg = "Waypoints must be 2 x n_waypoints, with at least 2 waypoints"
        raise ValueError(msg)
    if speed_plan.size != waypoints.shape[1]:
        msg = "Speed plan must have the same number of elements as waypoints"
        raise ValueError(msg)
    max_iter = 1000
    wp_seg_lengths = np.linalg.norm(waypoints[:, 1:] - waypoints[:, :-1], axis=0)
    wp_seg_times = wp_seg_lengths / speed_plan[:-1]
    for _ in range(max_iter):
        # choose random wp segment
        wp_idx = rng.integers(1, waypoints.shape[1])
        wp_seg_course = np.arctan2(
            waypoints[1, wp_idx] - waypoints[1, wp_idx - 1], waypoints[0, wp_idx] - waypoints[0, wp_idx - 1]
        )
        speed = speed_plan[wp_idx]

        # sample a point along the segment
        path_var = rng.uniform(0.2, 1.0) if wp_idx == 1 else rng.uniform(0.0, 1.0)
        pos = waypoints[:, wp_idx - 1] + path_var * (waypoints[:, wp_idx] - waypoints[:, wp_idx - 1])

        t_arrival = np.sum(wp_seg_times[: wp_idx - 1]) + path_var * wp_seg_times[wp_idx - 1]
        if t_arrival < timespan - 30.0:  # ensure that the vessel arrives at the waypoint before the end of the simulation
            break
    return np.array([pos[0], pos[1], speed, wp_seg_course]), t_arrival


def create_circle(radius: float, n_points: int) -> tuple[list, list]:
    """Creates a circle with a given radius and number of points.

    Args:
        radius (float): Radius of the circle.
        n_points (int): Number of points.

    Returns:
        Tuple[list, list]: Circle data in x and y coordinates.
    """
    t = np.linspace(0, 2.01 * np.pi, n_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return x.tolist(), y.tolist()


def get_list_except_element_idx(input_list: list, idx: int) -> list:
    """Returns a list with all elements of input_list except the element at idx.

    Args:
        input_list (list): List to get elements from
        idx (int): Index of element to exclude

    Returns:
        list: List with all elements of input_list except the element at idx
    """
    output_list = input_list.copy()
    output_list.pop(idx)
    return output_list


def get_relevant_do_states(input_list: list, idx: int, add_empty_cov: bool = False) -> list:
    """Returns a tuple list of relevant dynamic obstacle indices and states.

    To use in tracking/sensor generation, with all elements of input_list except
    the element <idx>, if this index is in the tuple list.

    Args:
        input_list (list): List of (do_idx, do_state, do_length, do_width) to
            get elements from.
        idx (int): Index of element to exclude.
        add_empty_cov (bool): Whether to add an empty covariance matrix to the
            output list. Defaults to False.

    Returns:
        list: List with all (do_idx, do_state) tuples of input_list except the
            element idx, if idx is in the tuple list.
    """
    output_list = []
    for do_idx, do_state, do_length, do_width in input_list:
        if do_idx != idx:
            if add_empty_cov:
                output_list.append((do_idx, do_state, np.zeros((4, 4)), do_length, do_width))
            else:
                output_list.append((do_idx, do_state, do_length, do_width))
    return output_list


def extract_do_states_from_ship_list(t: float, ship_list: list) -> list:
    """Extracts the dynamic obstacle states from the ship list.

    Args:
        t (float): Current time
        ship_list (list): List of Ship objects

    Returns:
        list: List of dynamic obstacle states on the form (do_idx, do_state, do_length, do_width)
    """
    true_do_states = []
    for i, ship_obj in enumerate(ship_list):
        if ship_obj.t_start <= t:
            vxvy_state = convert_state_to_vxvy_state(ship_obj.csog_state)
            true_do_states.append((i, vxvy_state, ship_obj.length, ship_obj.width))
    return true_do_states


def convert_state_to_vxvy_state(xs: np.ndarray) -> np.ndarray:
    """Converts from state(s) to [x, y, Vx, Vy]^T x N.

    Input can be [x, y, U, chi]^T x N or [x, y, psi, u, v, r]^T x N, where U
    is the speed over ground and chi is the course over ground, psi heading, u
    surge, v sway, r yaw rate.

    Args:
        xs (np.ndarray): State(s) to convert.

    Returns:
        np.ndarray: Converted state(s).
    """
    dim = xs.shape[0]
    if xs.ndim == 1:
        if dim == 4:
            return np.array([xs[0], xs[1], xs[2] * np.cos(xs[3]), xs[2] * np.sin(xs[3])])
        else:
            U = np.sqrt(xs[3] ** 2 + xs[4] ** 2)
            chi = np.arctan2(xs[4], xs[3]) + xs[2]
            return np.array([xs[0], xs[1], U * np.cos(chi), U * np.sin(chi)])
    elif dim == 4:
        return np.array([xs[0, :], xs[1, :], xs[2, :] * np.cos(xs[3, :]), xs[2, :] * np.sin(xs[3, :])])
    else:
        U = np.sqrt(np.multiply(xs[3, :], xs[3, :]) + np.multiply(xs[4, :], xs[4, :]))
        chi = np.arctan2(xs[4, :], xs[3, :]) + xs[2, :]
        return np.array([xs[0, :], xs[1, :], U * np.cos(chi), U * np.sin(chi)])


def convert_vxvy_state_to_sog_cog_state(xs: np.ndarray) -> np.ndarray:
    """Converts from a state [x, y, Vx, Vy] x N to [x, y, U, chi] x N.

    Args:
        xs (np.ndarray): State(s) to convert.

    Returns:
        np.ndarray: Converted state.
    """
    if xs.ndim == 1:
        return np.array([xs[0], xs[1], np.sqrt(xs[2] ** 2 + xs[3] ** 2), np.arctan2(xs[3], xs[2])])
    else:
        return np.array(
            [
                xs[0, :],
                xs[1, :],
                np.sqrt(np.multiply(xs[2, :], xs[2, :]) + np.multiply(xs[3, :], xs[3, :])),
                np.arctan2(xs[3, :], xs[2, :]),
            ]
        )


def extract_do_list_from_do_array(do_array: np.ndarray) -> list:
    """Extracts the dynamic obstacle list from the dynamic obstacle array.

    From RL observation with a maximum number of dynamic obstacles.

    Args:
        do_array (np.ndarray): Dynamic obstacle array.

    Returns:
        list: List of dynamic obstacles.
    """
    do_list = []
    if do_array.ndim != 2:
        msg = "Dynamic obstacle array must be 2D"
        raise ValueError(msg)
    nx_do = do_array.shape[0]
    max_num_do = do_array.shape[1]

    for i in range(max_num_do):
        if np.sum(do_array[:, i]) > 1.0:  # A proper DO entry has non-zeros in its vector
            do_state = do_array[0:4, i]
            do_length = do_array[4, i]
            do_width = do_array[5, i]
            do_cov = np.zeros((4, 4))
            if nx_do > 6:
                do_cov = do_array[6:, i].reshape((4, 4))
            do_list.append((i, do_state, do_cov, do_length, do_width))
    return do_list


def convert_3dof_state_to_sog_cog_state(xs: np.ndarray) -> np.ndarray:
    """Converts from a state [x, y, psi, u, v, r]^T x N to [x, y, U, chi]^T x N.

    Args:
        xs (np.ndarray): State(s) to convert.

    Returns:
        np.ndarray: Converted state.
    """
    if xs.ndim == 1:
        heading = xs[2]
        crab_angle = np.arctan2(xs[4], xs[3])
        cog = mf.wrap_angle_to_pmpi(heading + crab_angle)
        speed = np.sqrt(xs[3] ** 2 + xs[4] ** 2)
        return np.array([xs[0], xs[1], speed, cog])
    else:
        heading = xs[2, :]
        crab_angle = np.arctan2(xs[4, :], xs[3, :])
        cog = mf.wrap_angle_to_pmpi(heading + crab_angle)
        speed = np.sqrt(np.multiply(xs[3, :], xs[3, :]) + np.multiply(xs[4, :], xs[4, :]))
        return np.array([xs[0, :], xs[1, :], speed, cog])


def index_of_first_and_last_non_nan(input_list: list | np.ndarray) -> tuple[int, int]:
    """Returns the index of the first and last non-NaN element.

    In a list or numpy array.

    Args:
        input_list (list | np.ndarray): Input list or array.

    Returns:
        tuple[int, int]: Tuple of (first_non_nan_idx, last_non_nan_idx).
    """
    if isinstance(input_list, list):
        input_list = np.array(input_list)

    non_nan_indices = np.where(~np.isnan(input_list))[0]
    if non_nan_indices.size > 0:
        first_non_nan_idx = int(non_nan_indices[0])
        last_non_nan_idx = int(non_nan_indices[-1])
    else:
        first_non_nan_idx = -1
        last_non_nan_idx = -1

    return first_non_nan_idx, last_non_nan_idx


def current_utc_datetime_str(format_str: str) -> str:
    """Returns the current date and time as a string with specified format.

    Args:
        format_str (str): Format of the datetime string, e.g. %Y%m%d_%H_%M_%S.

    Returns:
        str: Formatted datetime string.
    """
    return datetime.utcnow().strftime(format_str)


def current_utc_timestamp() -> int:
    """Returns the current UTC timestamp.

    Returns:
        int: Current UTC timestamp.
    """
    return int(datetime.utcnow().timestamp())


def utc_timestamp_to_local_time(timestamp: int) -> datetime:
    """Converts UTC timestamp to local time.

    Args:
        timestamp (int): UTC timestamp

    Returns:
        datetime: Local time
    """
    return utc_to_local(utc_timestamp_to_datetime(timestamp))


def utc_timestamp_to_datetime(timestamp: int) -> datetime:
    """Converts UTC timestamp to datetime.

    Args:
        timestamp (int): UTC timestamp

    Returns:
        datetime: Datetime object
    """
    return datetime.fromtimestamp(timestamp)


def local_timestamp_from_utc() -> int:
    """Returns the current local time referenced timestamp.

    Returns:
        int: Current local time referenced timestamp.
    """
    return int(datetime.now().astimezone().timestamp())


def utc_to_local(utc_dt: datetime) -> datetime:
    """Convert UTC datetime to local datetime.

    Parameters:
        utc_dt (datetime): UTC datetime

    Returns:
        datetime: Local datetime
    """
    return utc_dt.replace(tzinfo=ZoneInfo("localtime"))


def write_coast_distances_to_file(
    dist_port: float, dist_front: float, dist_starboard: float, safety_radius: float, ship_obj: Any, filepath: str
) -> None:
    """Writes the distance to coast data on-line to the AIS case file.

    Parameters:
        dist_port (float): Distance to port coast
        dist_front (float): Distance to front coast
        dist_starboard (float): Distance to starboard coast
        safety_radius (float): Safety radius
        ship_obj (Ship): Ship object
        filepath (str): Filepath to AIS case file
    """
    new_filepath = filepath + "-radius-" + str(safety_radius)

    if not os.path.isfile(filepath + "-radius-" + str(safety_radius)):
        old_data = pd.read_csv(filepath, sep=";")
    else:
        old_data = pd.read_csv(new_filepath, sep=";")

    columns = ["dcoast_port", "dcoast_front", "dcoast_starboard"]

    if (
        "dcoast_port" not in old_data.columns
        or "dcoast_front" not in old_data.columns
        or "dcoast_starboard" not in old_data.columns
    ):
        old_data[columns[0]] = np.zeros(old_data.index.max() + 1)
        old_data[columns[1]] = np.zeros(old_data.index.max() + 1)
        old_data[columns[2]] = np.zeros(old_data.index.max() + 1)

    # Calculates current timestep for a case file with 60 sec intervals
    current_timestep = int((ship_obj._trajectory_sample - 1) / 60)

    arr1 = old_data["dcoast_port"].tolist()
    arr1[current_timestep] = dist_port
    old_data["dcoast_port"] = arr1
    arr2 = old_data["dcoast_front"].tolist()
    arr2[current_timestep] = dist_front
    old_data["dcoast_front"] = arr2
    arr3 = old_data["dcoast_starboard"].tolist()
    arr3[current_timestep] = dist_starboard
    old_data["dcoast_starboard"] = arr3

    old_data.to_csv(new_filepath, sep=";", index=False)
