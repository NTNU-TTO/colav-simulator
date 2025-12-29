"""guidance.py.

Summary:
Contains class definitions for guidance methods.
Every class must adhere to the model interface IGuidance.

Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm


@dataclass
class LOSGuidanceParams:
    """Parameter class for the LOS guidance method.

    Parameters:
        pass_angle_threshold (float): First threshold for switching between wp segments
        R_a (float): Radius of acceptance, second threshold for switching between wp segments
        K_p (float): Proportional gain in LOS law, K_p = 1 / lookahead distance
        K_i (float): Integral action gain in LOS law.
        max_cross_track_error_int (float): Maximum integrated cross-track error when using integral action
    """

    pass_angle_threshold: float = 90.0
    R_a: float = 25.0
    K_p: float = 0.02
    K_i: float = 0.0001
    max_cross_track_error_int: float = 1000.0
    cross_track_error_int_threshold: float = 50.0

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LOSGuidanceParams":  # noqa: D102
        return LOSGuidanceParams(**config_dict)

    def to_dict(self) -> dict:  # noqa: D102
        return asdict(self)


@dataclass
class Config:
    """Configuration class for managing guidance method parameters."""

    los: LOSGuidanceParams | None = field(default_factory=lambda: LOSGuidanceParams())
    ktp: bool | None = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":  # noqa: D102
        config = Config()
        if "los" in config_dict:
            config.los = cp.convert_settings_dict_to_dataclass(LOSGuidanceParams, config_dict["los"])
            config.ktp = None

        if "ktp" in config_dict:
            config.ktp = True
            config.los = None

        return config

    def to_dict(self) -> dict:  # noqa: D102
        config_dict = {}

        if self.los is not None:
            config_dict["los"] = self.los.to_dict()

        if self.ktp is not None:
            config_dict["ktp"] = ""

        return config_dict


class IGuidance(ABC):
    """The InterfaceGuidance class is abstract and used to force implementation.

    Forces the implementation of the below methods for all subclasses (guidance
    strategies), to comply with the guidance interface.
    """

    @abstractmethod
    def compute_references(
        self, waypoints: np.ndarray, speed_plan: np.ndarray, times: np.ndarray | None, xs: np.ndarray, dt: float
    ) -> np.ndarray:
        """Computes guidance reference states for the ship controller to track.

        Args:
            waypoints (np.ndarray): The waypoints to follow, Dimensions: [2, N]
                composed of the waypoint NE coordinates.
            speed_plan (np.ndarray): Reference speeds at each waypoint.
            times (np.ndarray | None): Optional array corresponding to arrival times
                at each waypoint.
            xs (np.ndarray): The ownship 3DOF state [x, y, psi, u, v, r]^T or [x, y,
                chi, U, 0, 0]^T in case of a kinematic CSOG model.
            dt (float): Time step.

        Returns:
            np.ndarray: References of dimension 9 x N (typically N = 1), consisting
                of ref poses, velocities and accelerations.
        """

    @abstractmethod
    def reset(self) -> None:
        "Resets the guidance system to its initial state."


class GuidanceBuilder:
    @classmethod
    def construct_guidance(cls, config: Config | None = None) -> IGuidance | None:
        """Builds a guidance method from the configuration.

        Args:
            config (Optional[guidance.Config]): Guidance configuration. Defaults to None.

        Returns:
            IGuidance | None: Guidance system as specified by the configuration,
                e.g. a LOSGuidance.
        """
        if config and config.los:
            return LOSGuidance(config.los)
        elif config and config.ktp:
            return KinematicTrajectoryPlanner()
        else:
            return None


class KinematicTrajectoryPlanner(IGuidance):
    """Class which implements a simple kinematic trajectory planner.

    The main functionality converts a path (described by waypoints) and speed plan into a continuous
    trajectory spline for which 3DOF references in position, speed and acceleration can be generated.
    """

    _x_spline: interp.BSpline
    _y_spline: interp.BSpline
    _heading_spline: interp.PchipInterpolator
    _speed_spline: interp.PchipInterpolator

    def __init__(self) -> None:
        self._epsilon: float = 0.00001
        self._s: float = 0.0001
        self._s_dot: float = 0.0
        self._final_arc_length: float = 0.0
        self._s_ddot: float = 0.0
        self._init: bool = False
        self._heading_waypoints: np.ndarray = np.array([])

    def reset(self) -> None:
        """Resets the guidance system to its initial state."""
        self._s = 0.0001
        self._s_dot = 0.0
        self._s_ddot = 0.0
        self._init = False
        self._heading_waypoints = np.array([])
        self._final_arc_length = 0.0

    def compute_splines(  # noqa: PLR0915
        self,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        times: np.ndarray | None = None,
        arc_length_parameterization: bool = True,
    ) -> tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.PchipInterpolator, float]:
        """Converts waypoints and speed plan into C² cubic splines.

        Note:
            The waypoints should be equidistant to ensure that the spline is
            well-behaved. Assumes validated inputs.

        Args:
            waypoints (np.ndarray): 2 x n_wps waypoints array to follow.
            speed_plan (np.ndarray): 1 x n_wps speed plan array to follow.
            times (np.ndarray | None): 1 x n_wps of time instances corresponding to
                the waypoints.
            arc_length_parameterization (bool): Whether to parameterize the splines
                by arc length or not.

        Returns:
            tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.PchipInterpolator, float]:
                Splines for x, y, heading and speed. The final arc length of the
                trajectory is also returned.
        """
        _, n_wps = waypoints.shape
        mod_waypoints = waypoints.copy()
        mod_speed_plan = speed_plan.copy()
        if n_wps == 2:
            wp_near_last = mod_waypoints[:, -1] - 0.05 * (mod_waypoints[:, -1] - mod_waypoints[:, -2])
            mod_waypoints = np.insert(mod_waypoints, 1, wp_near_last, axis=1)
            mod_speed_plan = np.insert(mod_speed_plan, 1, mod_speed_plan[0])

            if times:
                times = np.insert(times, 1, np.mean(times))
                times = np.insert(times, 1, np.mean(times[:2]))
            n_wps += 1

        if n_wps == 3:
            mod_waypoints = np.insert(mod_waypoints, 1, np.mean(mod_waypoints[:, :2], axis=1), axis=1)
            mod_speed_plan = np.insert(mod_speed_plan, 1, np.mean(mod_speed_plan[:2]))
            if times:
                times = np.insert(times, 1, np.mean(times[:2]))
            n_wps += 1

        # add artificial waypoint just beyond the last, to ensure that the spline is
        # well-behaved
        mod_waypoints = np.insert(
            mod_waypoints,
            n_wps,
            mod_waypoints[:, -1] + 2.0 * (mod_waypoints[:, -1] - mod_waypoints[:, -2]),
            axis=1,
        )
        mod_speed_plan = np.insert(mod_speed_plan, n_wps, mod_speed_plan[-1])
        n_wps += 1

        arc_lengths = [np.linalg.norm(mod_waypoints[:, i + 1] - mod_waypoints[:, i]) for i in range(n_wps - 1)]
        arc_lengths = np.array([0.0, *np.cumsum(arc_lengths)])
        final_arc_length = arc_lengths[-2]  # last arc length is due to the artificial point
        self._final_arc_length = final_arc_length
        if times:
            linspace = times
        else:
            linspace = np.linspace(0.0, 1.0, n_wps)

        self._speed_spline = interp.PchipInterpolator(linspace, mod_speed_plan)

        order = 3
        if arc_length_parameterization:
            x_arc_spline, y_arc_spline, arc_lengths = mhm.create_arc_length_spline(
                mod_waypoints[0, :].tolist(), mod_waypoints[1, :].tolist()
            )
            n_points = len(arc_lengths)
            smoothing = 0.005 * (n_points - np.sqrt(2 * n_points))
            expanded_x_values = x_arc_spline(arc_lengths)
            expanded_y_values = y_arc_spline(arc_lengths)
            t_x, c_x, k_x = interp.splrep(arc_lengths, expanded_x_values, s=smoothing, k=order)
            t_y, c_y, k_y = interp.splrep(arc_lengths, expanded_y_values, s=smoothing, k=order)
            self._x_spline = interp.BSpline(t_x, c_x, k_x, extrapolate=False)
            self._y_spline = interp.BSpline(t_y, c_y, k_y, extrapolate=False)

            expanded_speed_values = self._speed_spline(np.linspace(0.0, 1.0, len(arc_lengths)))
            t_speed, c_speed, k_speed = interp.splrep(arc_lengths, expanded_speed_values, s=smoothing, k=order)
            self._speed_spline = interp.BSpline(t_speed, c_speed, k_speed, extrapolate=False)
            x_der_values = self._x_spline(arc_lengths, 1)
            y_der_values = self._y_spline(arc_lengths, 1)
            self._heading_waypoints = mf.unwrap_angle_array(np.arctan2(y_der_values, x_der_values))
            self._heading_spline = interp.PchipInterpolator(arc_lengths, self._heading_waypoints)
        else:
            n_points = len(linspace)
            smoothing = n_points - np.sqrt(2 * n_points)  # default value
            t_x, c_x, k_x = interp.splrep(linspace, waypoints[0, :], s=smoothing, k=order)
            self._x_spline = interp.BSpline(t_x, c_x, k_x, extrapolate=False)

            t_y, c_y, k_y = interp.splrep(linspace, waypoints[1, :], s=smoothing, k=order)
            self._y_spline = interp.BSpline(t_y, c_y, k_y, extrapolate=False)

            x_der_values = self._x_spline(linspace, 1)
            y_der_values = self._y_spline(linspace, 1)
            self._heading_waypoints = mf.unwrap_angle_array(np.arctan2(y_der_values, x_der_values))
            self._heading_spline = interp.PchipInterpolator(linspace, self._heading_waypoints)

        # self.plot_reference_trajectory(waypoints, times)
        return self._x_spline, self._y_spline, self._heading_spline, self._speed_spline, final_arc_length

    def update_path_variable(self, dt: float) -> None:
        """Updates the path variable based on the time dt since the last update.

        Args:
            dt (float): Time step since the last update.
        """
        self._s_dot, self._s_ddot = self._compute_path_variable_derivatives(self._s)
        self._s = mf.sat(self._s + dt * self._s_dot, 0.0, 1.0)

    def compute_references(
        self,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        times: np.ndarray | None,  # noqa: ARG002
        xs: np.ndarray,  # noqa: ARG002
        dt: float,  # noqa: ARG002
    ) -> np.ndarray:
        """Converts waypoints and speed plan into C² cubic spline.

        From which 3DOF reference states (not necessarily feasible) are computed.

        Note:
            The waypoints should be equidistant to ensure that the spline is
            well-behaved. Assumes validated inputs.

        Args:
            waypoints (np.ndarray): 2 x n_wps waypoints array to follow.
            speed_plan (np.ndarray): 1 x n_wps speed plan array to follow.
            times (np.ndarray | None): 1 x n_wps of time instances corresponding to
                the waypoints.
            xs (np.ndarray): n x 1 dimensional state of the ship.
            dt (float): Time step between the previous and current run of this
                function.

        Returns:
            np.ndarray: 9 x 1 dimensional reference state vector.
        """
        self.compute_splines(waypoints, speed_plan, times)

        # der_heading = [0, *np.diff(self._heading_spline(path_values))] / path_values

        self._s_dot, self._s_ddot = self._compute_path_variable_derivatives(self._s)

        eta_ref = self._compute_eta_ref(self._s)
        eta_dot_ref = self._compute_eta_dot_ref(self._s, self._s_dot)
        eta_ddot_ref = self._compute_eta_ddot_ref(self._s, self._s_dot, self._s_ddot)

        self.plot_reference_trajectory(waypoints, times)

        references = np.zeros((9, 1))
        references[:, 0] = np.concatenate((eta_ref, eta_dot_ref, eta_ddot_ref))
        return references

    def compute_reference_trajectory(self, dt: float) -> np.ndarray:
        """Computes the reference trajectory by parameterizing the path variable s by time step dt and the speed spline.

        Args:
            dt (float): Time step to consider between trajectory samples

        Returns:
            np.ndarray: 9 x n_samples reference trajectory array
        """
        ref_traj_list = []
        s_copy = self._s
        s_final = self._x_spline.t[-1]
        eps = 0.2 * dt
        while abs(s_copy - s_final) > eps:
            s_dot, s_ddot = self._compute_path_variable_derivatives(s_copy)
            eta_ref = self._compute_eta_ref(s_copy)
            eta_dot_ref = self._compute_eta_dot_ref(s_copy, s_dot)
            eta_ddot_ref = self._compute_eta_ddot_ref(s_copy, s_dot, s_ddot)
            references_k = np.concatenate((eta_ref, eta_dot_ref, eta_ddot_ref))
            ref_traj_list.append(references_k.tolist())
            s_copy = mf.sat(s_copy + dt * s_dot, 0.0, s_final)
            if s_dot < 1e12 and len(ref_traj_list) > 1000:
                break
        return np.array(ref_traj_list).T

    def get_current_path_variables(self) -> tuple[float, float]:
        return self._s, self._s_dot

    def get_splines(
        self,
    ) -> tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.PchipInterpolator]:
        """Get the splines used for trajectory planning.

        Returns:
            tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.PchipInterpolator]:
                Splines for x, y, heading and speed.
        """
        return self._x_spline, self._y_spline, self._heading_spline, self._speed_spline

    def _compute_path_variable_derivatives(self, s: float) -> tuple[float, float]:
        s_dot = self._speed_spline(s) / np.sqrt(
            self._epsilon + np.power(self._x_spline(s, 1), 2.0) + np.power(self._y_spline(s, 1), 2.0)
        )

        s_ddot = s_dot * (
            self._speed_spline(s, 1)
            / np.sqrt(self._epsilon + np.power(self._x_spline(s, 1), 2.0) + np.power(self._y_spline(s, 1), 2.0))
            - self._speed_spline(s)
            * (self._x_spline(s, 1) * self._x_spline(s, 2) + self._y_spline(s, 1) * self._y_spline(s, 2))
            / np.power(
                np.sqrt(self._epsilon + np.power(self._x_spline(s, 1), 2.0) + np.power(self._y_spline(s, 1), 2.0)),
                3.0,
            )
        )

        return s_dot, s_ddot

    def _compute_eta_ref(self, s: float) -> np.ndarray:
        return np.array([self._x_spline(s), self._y_spline(s), self._heading_spline(s)])

    def _compute_eta_dot_ref(self, s: float, s_dot: float) -> np.ndarray:
        return np.array(
            [
                self._x_spline(s, 1) * s_dot,
                self._y_spline(s, 1) * s_dot,
                self._heading_spline(s, 1) * s_dot,
            ]
        )

    def _compute_eta_ddot_ref(self, s: float, s_dot: float, s_ddot: float) -> np.ndarray:
        return np.array(
            [
                self._x_spline(s, 2) * np.power(s_dot, 2.0) + self._x_spline(s, 1) * s_ddot,
                self._y_spline(s, 2) * np.power(s_dot, 2.0) + self._y_spline(s, 1) * s_ddot,
                self._heading_spline(s, 2) * np.power(s_dot, 2.0) + self._heading_spline(s, 1) * s_ddot,
            ]
        )

    def plot_reference_trajectory(
        self, waypoints: np.ndarray, times: np.ndarray | None, arc_length_parameterization: bool = True
    ) -> None:
        """Plots the trajectory of the reference vehicle.

        Args:
            waypoints (np.ndarray): Input waypoints specified by the user.
            times (Optional[np.ndarray]): Input optional times.
            arc_length_parameterization (bool): Whether to parameterize the splines by arc length or not.
        """
        _, n_wps = waypoints.shape

        if times:
            final_s = times[-1]
            linspace = times
        elif arc_length_parameterization:
            final_s = self._final_arc_length
            linspace = np.linspace(0.0, final_s, n_wps)
        else:
            final_s = 1.0
            linspace = np.linspace(0.0, final_s, n_wps)

        fig, ax = plt.subplots()
        ax.plot(waypoints[1, :], waypoints[0, :], "kx", label="Waypoints", linewidth=2.0)
        ax.plot(
            self._y_spline(np.linspace(0.0, final_s, 300)),
            self._x_spline(np.linspace(0.0, final_s, 300)),
            "b",
            label="Spline",
        )
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.legend()
        ax.grid()

        fig = plt.figure(figsize=(5, 10))
        axs = fig.subplot_mosaic(
            [
                ["x", "y", "psi"],
                ["U", "Udot", "r"],
            ]
        )

        axs["x"].plot(linspace, waypoints[0, :], "rx")
        axs["x"].plot(
            np.linspace(0.0, final_s, 300),
            self._x_spline(np.linspace(0.0, final_s, 300)),
            "b",
            label="x spline",
        )
        axs["x"].legend()
        axs["x"].grid()

        axs["y"].plot(linspace, waypoints[1, :], "rx")
        axs["y"].plot(
            np.linspace(0.0, final_s, 300),
            self._y_spline(np.linspace(0.0, final_s, 300)),
            "b",
            label="y spline",
        )
        axs["y"].legend()
        axs["y"].grid()

        axs["psi"].plot(
            np.linspace(0, final_s, len(self._heading_waypoints)),
            180.0 * np.unwrap(self._heading_waypoints) / np.pi,
            "rx",
            label="Waypoints",
        )
        heading_spline_vals = self._heading_spline(np.linspace(0.0, final_s, 300))
        axs["psi"].plot(
            np.linspace(0.0, final_s, 300),
            180.0 * np.unwrap(heading_spline_vals) / np.pi,
            "b",
            label="heading spline",
        )
        axs["psi"].legend()
        axs["psi"].grid()

        axs["r"].plot(
            np.linspace(0.0, final_s, 300),
            180.0 * self._heading_spline(np.linspace(0.0, final_s, 300), 1) / np.pi,
            "b",
            label="yaw rate spline",
        )
        axs["r"].legend()
        axs["r"].grid()

        axs["U"].plot(linspace, self._speed_spline(linspace), "rx", label="Waypoints")
        axs["U"].plot(
            np.linspace(0.0, final_s, 300),
            self._speed_spline(np.linspace(0.0, final_s, 300)),
            "b",
            label="speed spline",
        )
        axs["U"].legend()
        axs["U"].grid()

        axs["Udot"].plot(
            np.linspace(0.0, final_s, 300),
            self._speed_spline(np.linspace(0.0, final_s, 300), 1),
            "b",
            label="speed der spline",
        )
        axs["Udot"].legend()
        axs["Udot"].grid()
        plt.show(block=False)


class LOSGuidance(IGuidance):
    """Class which implements the Line-of-Sight guidance strategy.

    Internal variables:
        _wp_counter (float): Keeps track of the current waypoint segment
        _e_int (float): Integral of the cross-track error
    """

    def __init__(self, params: LOSGuidanceParams | None = None) -> None:
        if params is not None:
            self._params: LOSGuidanceParams = params
        else:
            self._params = LOSGuidanceParams()
        self._wp_counter: int = 0
        self._e_int: float = 0.0

    def reset(self) -> None:
        """Resets the guidance system to its initial state."""
        self._wp_counter = 0
        self._e_int = 0.0

    def compute_references(
        self,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        times: np.ndarray | None,  # noqa: ARG002
        xs: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Computes references in course and speed using the LOS guidance law.

        Args:
            waypoints (np.ndarray): 2 x n_wps waypoints array to follow.
            speed_plan (np.ndarray): 1 x n_wps speed plan array to follow.
            times (np.ndarray | None): 1 x n_wps of time instances corresponding to
                the waypoints.
            xs (np.ndarray): n x 1 dimensional state of the ship.
            dt (float): Time step between the previous and current run of this
                function.

        Returns:
            np.ndarray: 9 x 1 dimensional reference vector.
        """
        self._find_active_wp_segment(waypoints, xs)

        n_sp_dim = speed_plan.ndim
        if n_sp_dim != 1:
            raise ValueError("Speed plan does not consist of scalar reference values!")
        n_wps = speed_plan.size
        L_wp_segment = np.zeros(2)
        if self._wp_counter >= n_wps - 1:
            L_wp_segment = waypoints[:, self._wp_counter] - waypoints[:, self._wp_counter - 1]
        else:
            L_wp_segment = waypoints[:, self._wp_counter + 1] - waypoints[:, self._wp_counter]

        alpha = np.arctan2(L_wp_segment[1], L_wp_segment[0])
        e = -(xs[0] - waypoints[0, self._wp_counter]) * np.sin(alpha) + (xs[1] - waypoints[1, self._wp_counter]) * np.cos(
            alpha
        )

        if abs(e) < self._params.cross_track_error_int_threshold:
            self._e_int += e * dt

        self._e_int = mf.sat(
            self._e_int,
            -self._params.max_cross_track_error_int,
            self._params.max_cross_track_error_int,
        )

        chi_r = np.arctan2(-(self._params.K_p * e + self._params.K_i * self._e_int), 1)
        chi_d = mf.wrap_angle_to_pmpi(alpha + chi_r)
        U_d = speed_plan[self._wp_counter]

        # print(
        #     f"e_int: {self._e_int} | e: {e} | chi_r: {chi_r * 180.0 / np.pi} | Kp_b: {self._params.K_p * e} " +
        #     f"| Ki_b: {self._params.K_i * self._e_int}"
        # )

        references = np.zeros((9, 1))
        references[:, 0] = np.array([0.0, 0.0, chi_d, U_d, 0.0, 0.0, 0.0, 0.0, 0.0])
        return references

    def _find_active_wp_segment(self, waypoints: np.ndarray, xs: np.ndarray) -> None:
        """Finds the active line segment between waypoints to follow.

            Assumes validated inputs.

        Args:
            waypoints (np.array): 2 x n_wps waypoints array to follow.
            xs (np.array): n x 1 dimensional state of the ship
        """
        _, n_wps = waypoints.shape

        if xs.size < 2:
            raise ValueError("Wrong state dimension!")

        for i in range(self._wp_counter, n_wps - 1):
            d_0wp_vec = waypoints[:, i + 1] - xs[0:2]
            L_wp_segment = waypoints[:, i + 1] - waypoints[:, i]

            segment_passed = self._check_for_wp_segment_switch(L_wp_segment, d_0wp_vec)
            if segment_passed:
                self._wp_counter += 1
                # self._e_int = 0.0
                # print(f"Segment {i} passed!")
            else:
                break

    def _check_for_wp_segment_switch(self, wp_segment: np.ndarray, d_0wp: np.ndarray) -> bool:
        """Checks if a switch should be made from the current to the next waypoint segment.

        Args:
            wp_segment (np.ndarray): 2D vector describing the distance from waypoint
                i to i + 1 in the current segment.
            d_0wp (np.ndarray): 2D distance vector from state to waypoint i + 1.

        Returns:
            bool: If the switch should be made or not.
        """
        wp_segment = mf.normalize_vec(wp_segment)
        d_0wp_norm = np.linalg.norm(d_0wp)
        d_0wp = mf.normalize_vec(d_0wp)

        segment_passed = wp_segment.dot(d_0wp) < np.cos(np.deg2rad(self._params.pass_angle_threshold))

        segment_passed = segment_passed or d_0wp_norm <= self._params.R_a

        return segment_passed
