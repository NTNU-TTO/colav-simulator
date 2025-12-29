"""kuwata_vo.py.

Summary:
A reactive COLAV planning algorithm based on the paper
"Safe Maritime Autonomous Navigation With COLREGS, Using Velocity Obstacles"
by Kuwata et al.

NOTE: Not fully verified and tested.
"""

from dataclasses import dataclass, field
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from seacharts.enc import ENC
from shapely import affinity, geometry

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm


class VOCOLREGSSituation(Enum):
    """Enum for the different COLREGS rules considered in VO."""

    HO = 0  # Head-on
    OT_ing = 1  # Overtaking
    OT_en = 2  # Overtaken
    CR_PS = 3  # Crossing with obstacle on the left/port side
    CR_SS = 4  # Crossing with obstacle on the rigght/starboard side


@dataclass
class VOParams:
    """Parameters for the velocity obstacle algorithm."""

    length_os: float = 10.0  # The length of the ownship [m]
    width_os: float = 5.0  # The width of the ownship [m]
    draft_os: float = 2.0  # The draft of the ownship [m]
    planning_frequency: float = 1.0  # The planning frequency.
    t_max: float = 60.0  # Precollision check maximum time.
    d_min: float = 20.0  # Precollision check minimum distance.
    t_grounding_max: float = 10.0  # Maximum time to consider for grounding.
    t_colregs_update_interval: float = 1.0  # The interval at which the COLREGS situation for an obstacle is updated.
    head_on_width: float = 30.0  # The width of the head-on sector for determining COLREGS constraints/situations.
    overtaking_angle: float = (
        112.0  # The boundary angle for the overtaking sector for determining COLREGS constraints/situations.
    )
    heading_set_limits: list = field(
        default_factory=lambda: [-120.0, 120.0]
    )  # List of heading modifications to consider in the planning [deg].
    heading_set_spacing: float = 2.5  # The spacing between the headings to consider in the planning [deg]
    speed_set_limits: list = field(
        default_factory=lambda: [0.0, 10.0]
    )  # List of speed modifications to consider in the planning. [m/s] Should be set based on the ship min-max speed.
    speed_set_spacing: float = 0.5  # The spacing between the speeds to consider in the planning [m/s]
    safety_buffer: float = (
        15.0  #  A buffer length [m] to add to the obstacle's bounding box to take speed uncertainty into account
    )
    vo_violation_cost: float = 1000.0  # The cost for a velocity obstacle violation.
    grounding_cost: float = 10000.0  # The cost for grounding.
    colregs_violation_cost: float = 100.0  # The cost for a COLREGS violation.
    Q: np.ndarray = field(default_factory=lambda: np.eye(2))  # The weighting matrix for the cost function.

    def to_dict(self) -> dict:  # noqa: D102
        output = {
            "length_os": self.length_os,
            "width_os": self.width_os,
            "draft_os": self.draft_os,
            "planning_frequency": self.planning_frequency,
            "t_max": self.t_max,
            "d_min": self.d_min,
            "t_grounding_max": self.t_grounding_max,
            "t_colregs_update_interval": self.t_colregs_update_interval,
            "head_on_width": float(np.rad2deg(self.head_on_width)),
            "overtaking_angle": float(np.rad2deg(self.overtaking_angle)),
            "heading_set_limits": self.heading_set_limits,
            "heading_set_spacing": self.heading_set_spacing,
            "speed_set_limits": self.speed_set_limits,
            "speed_set_spacing": self.speed_set_spacing,
            "safety_buffer": self.safety_buffer,
            "vo_violation_cost": self.vo_violation_cost,
            "grounding_cost": self.grounding_cost,
            "colregs_violation_cost": self.colregs_violation_cost,
            "Q": self.Q.diagonal().tolist(),
        }
        return output

    @classmethod
    def from_dict(cls, data: dict) -> "VOParams":  # noqa: D102
        output = VOParams(
            length_os=data["length_os"],
            width_os=data["width_os"],
            draft_os=data["draft_os"],
            planning_frequency=data["planning_frequency"],
            t_max=data["t_max"],
            d_min=data["d_min"],
            t_grounding_max=data["t_grounding_max"],
            t_colregs_update_interval=data["t_colregs_update_interval"],
            head_on_width=float(np.deg2rad(data["head_on_width"])),
            overtaking_angle=float(np.deg2rad(data["overtaking_angle"])),
            heading_set_limits=data["heading_set_limits"],
            heading_set_spacing=data["heading_set_spacing"],
            speed_set_limits=data["speed_set_limits"],
            speed_set_spacing=data["speed_set_spacing"],
            safety_buffer=data["safety_buffer"],
            vo_violation_cost=data["vo_violation_cost"],
            grounding_cost=data["grounding_cost"],
            colregs_violation_cost=data["colregs_violation_cost"],
            Q=np.diag(data["Q"]),
        )
        return output


class VO:
    def __init__(self, config: VOParams | None = None) -> None:
        if config:
            self._params = config
        else:
            self._params = VOParams()
        self._initialized = False
        self._t_prev: float = 0.0

        self._poly_os = geometry.Polygon(
            [
                (-self._params.length_os / 2, -self._params.width_os / 2),
                (self._params.length_os / 2, -self._params.width_os / 2),
                (self._params.length_os / 2, self._params.width_os / 2),
                (-self._params.length_os / 2, self._params.width_os / 2),
            ]
        )
        buffer = self._params.safety_buffer
        theta = 0.0
        coords = []
        for _i in range(100):
            theta += 2.01 * np.pi / 100
            coords.append((buffer * np.cos(theta), buffer * np.sin(theta)))

        self._safety_buffer_poly = geometry.Polygon(coords)
        self._relevant_do_list: list = []
        self._colregs_situations: list = []
        self._speed_set = np.arange(
            self._params.speed_set_limits[0], self._params.speed_set_limits[1], self._params.speed_set_spacing
        )
        self._heading_set = np.deg2rad(
            np.arange(
                self._params.heading_set_limits[0], self._params.heading_set_limits[1], self._params.heading_set_spacing
            )
        )
        self._violation_costs: np.ndarray = np.ones((len(self._speed_set), len(self._heading_set)))
        self._total_costs: np.ndarray = np.zeros((len(self._speed_set), len(self._heading_set)))
        self._references = np.zeros((9, 1))

    def get_current_plan(self) -> np.ndarray:
        """Get the current plan."""
        return self._references

    def plan(
        self, t: float, v_ref: np.ndarray, ownship_state: np.ndarray, do_list: list, enc: ENC | None = None
    ) -> np.ndarray:
        if self._initialized and t - self._t_prev < (1.0 / self._params.planning_frequency):
            return self.get_current_plan()

        if not self._initialized:
            self._initialized = True
            self._t_prev = t

        # Nominal solution
        heading_opt = np.arctan2(v_ref[1], v_ref[0])
        speed_opt = np.linalg.norm(v_ref)

        p_os = ownship_state[0:2]
        psi_os = ownship_state[2]
        Rmtrx = mf.Rmtrx2D(psi_os)
        v_os = Rmtrx @ ownship_state[3:5]
        poly_os: geometry.Polygon = affinity.rotate(self._poly_os, psi_os, use_radians=True)
        # poly_os = affinity.translate(self._poly_os, p_os[0], p_os[1])

        self._total_costs = np.zeros((len(self._speed_set), len(self._heading_set)))
        self._violation_costs = np.zeros((len(self._speed_set), len(self._heading_set)))
        for _, do_info in enumerate(do_list):
            id_do, state, _, length, width = do_info
            p_do = state[0:2]
            v_do = state[2:4]
            psi_do = np.arctan2(v_do[1], v_do[0])

            if mhm.check_if_vessel_is_passed_by(p_os, v_os, p_do, v_do):
                if id_do in self._relevant_do_list:
                    idx = self._relevant_do_list.index(id_do)
                    self._colregs_situations.pop(idx)
                    self._relevant_do_list.pop(idx)
                continue

            situation_started = False
            if id_do not in self._relevant_do_list:
                situation_started = self._precollision_check(p_os, v_os, p_do, v_do)

                if situation_started:
                    self._relevant_do_list.append(id_do)
                    self._colregs_situations.append(
                        (id_do, t, self._determine_colregs_situation(p_os, psi_os, p_do, psi_do))
                    )
                else:
                    continue

            self._update_colregs_situation(t, p_os, psi_os, p_do, psi_do, id_do)

            _, _, situation = [x for x in self._colregs_situations if x[0] == id_do][0]

            poly_do = geometry.Polygon(
                [(-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2)]
            )
            poly_do = affinity.rotate(poly_do, psi_do, use_radians=True)
            poly_do = affinity.translate(poly_do, p_do[0], p_do[1])

            expanded_poly_do, expanded_poly_do_buffered = self._compute_expanded_do_polygon(poly_os, poly_do)

            self._update_violation_costs(
                situation,
                expanded_poly_do,
                expanded_poly_do_buffered,
                p_do,
                v_do,
                p_os,
                v_os,
                psi_os,
                enc,
                False,
                poly_do,
                poly_os,
            )

        if self._relevant_do_list:
            heading_opt, speed_opt = self._compute_optimal_controls(v_ref, v_os, psi_os)

        self._t_prev = t
        self._references[2, 0] = mf.wrap_angle_to_pmpi(heading_opt)
        self._references[3, 0] = speed_opt
        return self._references

    def _compute_optimal_controls(self, v_ref: np.ndarray, v_os: np.ndarray, psi_os: float) -> tuple[float, float]:
        """Computes the optimal controls based on the current admissible controls and the VO cost function.

        Args:
            v_ref (np.ndarray): The reference velocity.
            v_os (np.ndarray): The ownship velocity.
            psi_os (float): The ownship heading.

        Returns:
            tuple[float, float]: The optimal heading and speed.
        """
        min_cost = 1e10
        for i, speed in enumerate(self._speed_set):
            for j, heading in enumerate(self._heading_set):
                candidate_heading = heading + psi_os
                candidate_speed = speed + np.linalg.norm(v_os)
                v_os_new = np.array(
                    [
                        candidate_speed * np.cos(candidate_heading),
                        speed * np.sin(candidate_heading),
                    ]
                )
                self._total_costs[i, j] = self._violation_costs[i, j] + (v_ref - v_os_new).transpose() @ self._params.Q @ (
                    v_ref - v_os_new
                )

                if self._total_costs[i, j] < min_cost:
                    min_cost = self._total_costs[i, j]
                    i_opt = i
                    j_opt = j

        heading_opt = self._heading_set[j_opt] + psi_os
        speed_opt = self._speed_set[i_opt] + np.linalg.norm(v_os)
        return heading_opt, speed_opt

    def _update_violation_costs(
        self,
        situation: VOCOLREGSSituation,
        expanded_poly_do: geometry.Polygon,
        expanded_poly_do_buffered: geometry.Polygon,
        p_do: np.ndarray,
        v_do: np.ndarray,
        p_os: np.ndarray,
        v_os: np.ndarray,
        psi_os: float,
        enc: ENC | None = None,  # noqa: ARG002
        show_debug_plots: bool = False,
        poly_do: geometry.Polygon | None = None,
        poly_os: geometry.Polygon | None = None,
    ) -> None:
        """Updates the cost of controls causing any violation (COLREGS, VO, grounding).

        Args:
            situation (VOCOLREGSSituation): The COLREGS situation.
            expanded_poly_do (geometry.Polygon): The expanded dynamic obstacle polygon.
            expanded_poly_do_buffered (geometry.Polygon): The buffered expanded dynamic obstacle polygon.
            p_do (np.ndarray): Dynamic obstacle position.
            v_do (np.ndarray): Dynamic obstacle velocity.
            p_os (np.ndarray): Ownship position.
            v_os (np.ndarray): Ownship velocity.
            psi_os (float): Ownship heading.
            enc (ENC | None): The Electronic Navigational Charts. Defaults to None.
            show_debug_plots (bool): Whether to show debug plots. Defaults to False.
            poly_do (geometry.Polygon | None): The dynamic obstacle polygon, used for
                plotting.
            poly_os (geometry.Polygon | None): The ownship polygon, used for plotting.
        """
        if show_debug_plots:
            if poly_do is None or poly_os is None:
                msg = "poly_do and poly_os must be provided when show_debug_plots is True"
                raise ValueError(msg)
            fig, ax = plot_vo_situation(
                expanded_poly_do,
                expanded_poly_do_buffered,
                affinity.translate(poly_os, p_os[0], p_os[1]),
                v_os,
                poly_do,
                v_do,
            )
            velocity_handles = self.plot_current_velocity_grid(fig, ax, v_os, psi_os)
            ax.plot([0.0, v_os[1] * 2.5 * self._params.t_max], [0.0, v_os[0] * 2.5 * self._params.t_max], "m")
            ray_plot = ax.plot([0.0, 0.0], [0.0, 0.0], "m")[0]

        for j, heading in enumerate(self._heading_set):
            for i, speed in reversed(list(enumerate(self._speed_set))):
                # if not speed_above_avoids_grounding:
                #     break  # No need to check the remaining speeds

                candidate_heading = heading + psi_os
                candidate_speed = speed + np.linalg.norm(v_os)
                v_os_new = np.array(
                    [candidate_speed * np.cos(candidate_heading), candidate_speed * np.sin(candidate_heading)]
                )

                p_diff = p_os - p_do
                v_diff = v_os_new - v_do

                color = "g"
                # First check if VO is violated
                ray = geometry.LineString([p_os, p_os + v_os_new * 3.0 * self._params.t_max])
                if ray.intersects(expanded_poly_do_buffered):
                    self._violation_costs[i, j] = self._params.vo_violation_cost
                    color = "r"
                else:
                    # Check if own-ship follows a velocity in V3, i.e. it moves away from the DO
                    in_v3 = np.dot(p_diff, v_diff) > 0

                    # Check if own-ship follows a velocity in V1, i.e. it moves such that the DO is on its starboard side
                    in_v1 = not in_v3 and (p_diff[0] * v_diff[1] - p_diff[1] * v_diff[0] > 0)

                    if in_v1 and (situation in (VOCOLREGSSituation.HO, VOCOLREGSSituation.OT_ing, VOCOLREGSSituation.CR_SS)):
                        self._violation_costs[i, j] = self._params.colregs_violation_cost
                        color = "r"

                    # Check if velocity leads to collision with grounding hazard within t_max
                    # if speed_above_avoids_grounding and self._check_grounding_risk(p_os, v_os_new, enc):
                    #     speed_above_avoids_grounding = False
                    #     self._violation_costs[i, j] = self._params.grounding_cost

                if show_debug_plots:
                    ray_plot.remove()
                    ray_plot = ax.plot(
                        [0.0, v_os_new[1] * 3.0 * self._params.t_max],
                        [0.0, v_os_new[0] * 3.0 * self._params.t_max],
                        "m",
                    )[0]
                    velocity_handles[i][j].remove()
                    velocity_handles[i][j] = ax.arrow(
                        0, 0, v_os_new[1], v_os_new[0], color=color, width=0.1, head_width=0.3, head_length=0.3
                    )

    def _check_grounding_risk(self, p_os: np.ndarray, v_os: np.ndarray, enc: ENC) -> bool:
        """Checks if the candidate velocity leads to a grounding hazard within t_max.

        Args:
            p_os (np.ndarray): Own-ship position.
            v_os (np.ndarray): Own-ship velocity.
            enc (ENC): Electronic Navigational Chart data.

        Returns:
            bool: True if the candidate velocity leads to a grounding hazard within t_max, False otherwise.
        """
        p2 = p_os + v_os * self._params.t_max
        return mapf.check_if_segment_crosses_grounding_hazards(enc, p2, p_os, self._params.draft_os)

    def _update_colregs_situation(
        self, t: float, p_os: np.ndarray, psi_os: float, p_do: np.ndarray, psi_do: float, id_do: int
    ) -> None:
        """Updates the COLREGS situation of the ownship and the dynamic obstacle.

        Updates if a time threshold is exceeded.

        Args:
            t (float): Current time.
            p_os (np.ndarray): Ownship position.
            psi_os (float): Ownship heading.
            p_do (np.ndarray): Dynamic obstacle position.
            psi_do (float): Dynamic obstacle heading.
            id_do (int): ID of the dynamic obstacle.
        """
        for idx, (id_, t_, _) in enumerate(self._colregs_situations):
            if id_ == id_do and t - t_ > self._params.t_colregs_update_interval:
                self._colregs_situations[idx] = (id_, t, self._determine_colregs_situation(p_os, psi_os, p_do, psi_do))
                return

    def _determine_colregs_situation(
        self, p_os: np.ndarray, psi_os: float, p_do: np.ndarray, psi_do: float
    ) -> VOCOLREGSSituation:
        """Determines the COLREGS situation of the ownship and the dynamic obstacle.

        Args:
            p_os (np.ndarray): Ownship position.
            psi_os (float): Ownship heading.
            p_do (np.ndarray): Dynamic obstacle position.
            psi_do (float): Dynamic obstacle heading.

        Returns:
            VOCOLREGSSituation: The COLREGS situation of the ownship and the dynamic
                obstacle.
        """
        bearing_do_os = mf.compute_bearing(psi_do, p_do, p_os)
        bearing_os_do = mf.compute_bearing(psi_os, p_os, p_do)
        heading_diff = mf.wrap_angle_diff_to_pmpi(psi_do, psi_os)

        if (heading_diff < -np.pi + self._params.head_on_width / 2.0) or (
            heading_diff > np.pi - self._params.head_on_width / 2.0
        ):
            return VOCOLREGSSituation.HO

        if bearing_os_do > self._params.overtaking_angle or bearing_os_do < -self._params.overtaking_angle:
            return VOCOLREGSSituation.OT_en

        if bearing_do_os > self._params.overtaking_angle or bearing_do_os < -self._params.overtaking_angle:
            return VOCOLREGSSituation.OT_ing

        if bearing_do_os < 0:
            return VOCOLREGSSituation.CR_PS

        return VOCOLREGSSituation.CR_SS

    def _precollision_check(self, p_os: np.ndarray, v_os: np.ndarray, p_do: np.ndarray, v_do: np.ndarray) -> bool:
        """Checks if the ownship and the dynamic obstacle are in a dangerous situation.

        Args:
            p_os (np.ndarray): Ownship position.
            v_os (np.ndarray): Ownship velocity.
            p_do (np.ndarray): Dynamic obstacle position.
            v_do (np.ndarray): Dynamic obstacle velocity.

        Returns:
            bool: True if the ownship and the dynamic obstacle are in a dangerous situation.
        """
        t_cpa, d_cpa, __ = mhm.compute_vessel_pair_cpa(p_os, v_os, p_do, v_do)
        return t_cpa <= self._params.t_max and d_cpa <= self._params.d_min

    def _check_if_ray_intersects_vo(self, vo: geometry.Polygon, p_os: np.ndarray, v_os: np.ndarray) -> bool:
        """Checks if the ray from the ownship to the dynamic obstacle intersects the expanded DO shape.

        Args:
            vo (geometry.Polygon): The VO.
            p_os (np.ndarray): Ownship position.
            v_os (np.ndarray): Ownship velocity.

        Returns:
            bool: True if the ray intersects the VO.
        """
        ray = geometry.LineString([p_os, p_os + v_os * self._params.t_max])
        return ray.intersects(vo)

    def _compute_expanded_do_polygon(
        self, poly_os: geometry.Polygon, poly_do: geometry.Polygon
    ) -> tuple[geometry.Polygon, geometry.Polygon]:
        """Computes the minowski sum poly_do + (-poly_os).

        For the ownship (zero origin referenced) and the dynamic obstacle
        (referenced to its position). Returns the expanded DO polygon and the
        expanded DO polygon with the safety buffer (for velocity uncertainty
        added).

        Args:
            poly_os (geometry.Polygon): Ownship polygon.
            poly_do (geometry.Polygon): Dynamic obstacle polygon.

        Returns:
            tuple[geometry.Polygon, geometry.Polygon]: The expanded DO polygon and
                the expanded DO polygon with the safety buffer.
        """
        reflected_poly_os = compute_reflection(poly_os)
        expanded_poly_do = compute_minowski_sum(poly_do, reflected_poly_os)
        expanded_poly_do_w_uncertainty = compute_minowski_sum(expanded_poly_do, self._safety_buffer_poly)
        return expanded_poly_do, expanded_poly_do_w_uncertainty

    def plot_current_velocity_grid(
        self,
        fig: plt.Figure,  # noqa: ARG002
        ax: plt.Axes,
        v_os: np.ndarray,
        psi_os: float,
    ) -> list:
        """Plots the current admissible and inadmissible velocities for the VO-COLAV.

        Args:
            fig (plt.Figure): Figure to plot on.
            ax (plt.Axes): Axes to plot on.
            v_os (np.ndarray): Ownship velocity.
            psi_os (float): Ownship heading.

        Returns:
            list: The handles to all velocity vectors in the grid.
        """
        velocity_handles = []
        for i, speed in enumerate(self._speed_set):
            speed_conditioned_velocity_handles = []
            for j, heading in enumerate(self._heading_set):
                candidate_speed = speed + np.linalg.norm(v_os)
                candidate_heading = heading + psi_os
                color = "g"
                if self._violation_costs[i, j] > 0.0:
                    color = "r"
                ray = ax.arrow(
                    0,
                    0,
                    candidate_speed * np.sin(candidate_heading),
                    candidate_speed * np.cos(candidate_heading),
                    color=color,
                    width=0.1,
                    head_width=0.3,
                    head_length=0.3,
                )
                speed_conditioned_velocity_handles.append(ray)

            velocity_handles.append(speed_conditioned_velocity_handles)

        return velocity_handles


def compute_minowski_sum(poly1: geometry.Polygon, poly2: geometry.Polygon) -> geometry.Polygon:
    """Computes the Minkowski sum of two polygons.

    Args:
        poly1 (geometry.Polygon): First polygon.
        poly2 (geometry.Polygon): Second polygon.

    Returns:
        geometry.Polygon: The Minkowski sum of the two polygons.
    """
    vo_coords = []
    for p1 in poly1.exterior.coords:
        for p2 in poly2.exterior.coords:
            vo_coords.append((p1[0] + p2[0], p1[1] + p2[1]))

    return geometry.Polygon(vo_coords).convex_hull


def compute_reflection(poly: geometry.Polygon) -> geometry.Polygon:
    """Computes the reflection of a polygon.

    Args:
        poly (geometry.Polygon): Polygon to reflect.

    Returns:
        geometry.Polygon: The reflected polygon.
    """
    new_coords = []
    for p in poly.exterior.coords:
        new_coords.append((-p[0], -p[1]))

    return geometry.Polygon(new_coords)


def plot_vo_situation(
    expanded_poly_do: geometry.Polygon,
    expanded_poly_do_buffered: geometry.Polygon,
    poly_os: geometry.Polygon,
    v_os: np.ndarray,
    poly_do: geometry.Polygon,
    v_do: np.ndarray,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the vcurrent velocity obstacle situation.

    Args:
        expanded_poly_do (geometry.Polygon): The expanded dynamic obstacle polygon to avoid.
        expanded_poly_do_buffered (geometry.Polygon): The expanded DO polygon with the safety buffer added.
        poly_os (geometry.Polygon): Ownship polygon.
        v_os (np.ndarray): Ownship velocity.s
        poly_do (geometry.Polygon): Dynamic obstacle polygon.
        v_do (np.ndarray): Dynamic obstacle velocity.
        fig (Optional[plt.Figure]): Figure to plot on. Defaults to None.
        ax (Optional[plt.Axes]): Axes to plot on. Defaults to None.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes for the plot.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    plt.show(block=False)
    center_os_x, center_os_y = poly_os.centroid.xy
    center_os_x = np.array(center_os_x.tolist())[0]
    center_os_y = np.array(center_os_y.tolist())[0]

    os_x, os_y = poly_os.exterior.xy
    os_x = np.array(os_x.tolist())
    os_y = np.array(os_y.tolist())
    os_x = os_x - center_os_x
    os_y = os_y - center_os_y
    ax.plot(os_y, os_x, "b", label="Ownship")

    do_x, do_y = poly_do.exterior.xy
    do_x = np.array(do_x.tolist())
    do_y = np.array(do_y.tolist())
    do_x = do_x - center_os_x
    do_y = do_y - center_os_y
    ax.plot(do_y, do_x, "c", label="DO")

    vo_x, vo_y = expanded_poly_do.exterior.xy
    vo_x = np.array(vo_x.tolist())
    vo_y = np.array(vo_y.tolist())
    vo_x = vo_x - center_os_x
    vo_y = vo_y - center_os_y
    ax.plot(vo_y, vo_x, "r", label="Expanded DO shape")

    vo_buf_x, vo_buf_y = expanded_poly_do_buffered.exterior.xy
    vo_buf_x = np.array(vo_buf_x.tolist())
    vo_buf_y = np.array(vo_buf_y.tolist())
    vo_buf_x = vo_buf_x - center_os_x
    vo_buf_y = vo_buf_y - center_os_y
    ax.plot(vo_buf_y, vo_buf_x, "y", label="Expanded DO shape w/safety buffer")
    plt.arrow(0, 0, v_os[1], v_os[0], color="b", head_width=1.0, head_length=1.0)
    plt.arrow(0, 0, v_do[1], v_do[0], color="g", head_width=1.0, head_length=1.0)
    plt.arrow(0, 0, v_do[1] - v_do[1], v_do[0] + v_os[0], color="k", head_width=1.0, head_length=1.0)

    return fig, ax
