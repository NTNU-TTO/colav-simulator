"""Demonstrates how to use the PQ-RRT* algorithm with the colav-simulator.

Note that the underlying ship model in the planner does not
match the ship model used from the colav-simulator, and thus should be tuned for usage outside this example.

Remember to install the dependencies (COLAV-simulator) and rrt-rs
https://github.com/NTNU-Autoship-Internal/rrt-rs before running this script.

This script is also found in the rrt-rs repository examples folder.

Author: Trym Tengesdal
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import rrt_star_lib  # pyright: ignore[reportMissingImports]
import seacharts.enc as senc
from shapely import strtree

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.paths as dp
import colav_simulator.core.colav.colav_interface as ci
from colav_simulator.behavior_generator import BehaviorGenerationMethod, PQRRTStarParams
from colav_simulator.common import plotters
from colav_simulator.core import guidances, models, stochasticity
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator


@dataclass
class RRTConfig:
    params: PQRRTStarParams = field(default_factory=lambda: PQRRTStarParams())
    model: models.KinematicCSOGParams = field(
        default_factory=lambda: models.KinematicCSOGParams(
            name="KinematicCSOG",
            draft=0.5,
            length=10.0,
            width=3.0,
            T_chi=10.0,
            T_U=7.0,
            r_max=np.deg2rad(4),
            U_min=0.0,
            U_max=15.0,
        )
    )
    los: guidances.LOSGuidanceParams = field(
        default_factory=lambda: guidances.LOSGuidanceParams(
            K_p=0.03,
            K_i=0.0001,
            pass_angle_threshold=90.0,
            R_a=25.0,
            max_cross_track_error_int=1000.0,
            cross_track_error_int_threshold=30.0,
        )
    )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RRTConfig":  # noqa: D102
        config = RRTConfig(
            params=PQRRTStarParams.from_dict(config_dict["params"]),
            model=models.KinematicCSOGParams.from_dict(config_dict["model"]),
            los=guidances.LOSGuidanceParams.from_dict(config_dict["los"]),
        )

        return config


def parse_rrt_solution(soln: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Parses the RRT solution.

    Args:
        soln (dict): Solution dictionary.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: Tuple of waypoints, trajectory,
            inputs, times and cost from the RRT solution.
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
    return waypoints, trajectory, inputs, times, soln["cost"]


@dataclass
class RRTPlannerParams:
    los: guidances.LOSGuidanceParams = field(
        default_factory=lambda: guidances.LOSGuidanceParams(
            K_p=0.035, K_i=0.0, pass_angle_threshold=90.0, R_a=25.0, max_cross_track_error_int=30.0
        )
    )
    rrt: RRTConfig = field(default_factory=lambda: RRTConfig())

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RRTPlannerParams":  # noqa: D102
        config = RRTPlannerParams(
            los=guidances.LOSGuidanceParams.from_dict(config_dict["los"]),
            rrt=RRTConfig.from_dict(config_dict["pq-rrt"]),
        )

        return config


class PQRRTStar(ci.ICOLAV):
    """PQ-RRT* planner."""

    def __init__(self, config: RRTPlannerParams) -> None:
        self._rrt_config = config.rrt
        self._rrt = rrt_star_lib.PQRRTStar(config.rrt.los, config.rrt.model, config.rrt.params)
        self._los = guidances.LOSGuidance(config.los)

        self._rrt_inputs: np.ndarray = np.empty(3)
        self._rrt_trajectory: np.ndarray = np.empty(6)
        self._rrt_waypoints: np.ndarray = np.empty(3)
        self._geometry_tree: strtree.STRtree = strtree.STRtree([])

        self._min_depth: int = 0
        self._map_origin: np.ndarray = np.array([])
        self._references = np.array([])
        self._initialized = False
        self._t_prev: float = 0.0

    def reset(self) -> None:  # noqa: D102
        self._references = np.zeros((9, 1))
        self._t_prev = 0.0
        self._initialized = False
        self._rrt.reset(0)
        self._rrt_inputs = np.empty(3)
        self._rrt_trajectory = np.empty(6)
        self._rrt_waypoints = np.empty(3)
        self._geometry_tree = strtree.STRtree([])
        self._min_depth = 0
        self._map_origin = np.array([])
        self._los.reset()

    def plan(  # noqa: D102
        self,
        t: float,
        waypoints: np.ndarray,  # noqa: ARG002
        speed_plan: np.ndarray,  # noqa: ARG002
        ownship_state: np.ndarray,
        do_list: list[tuple[int, np.ndarray, np.ndarray, float, float]],  # noqa: ARG002
        enc: senc.ENC | None = None,
        goal_state: np.ndarray | None = None,
        w: stochasticity.DisturbanceData | None = None,  # noqa: ARG002
        **kwargs,
    ) -> np.ndarray:
        assert goal_state is not None, "Goal state must be provided to the RRT"  # noqa: S101
        assert enc is not None, "ENC must be provided to the RRT"  # noqa: S101
        if not self._initialized:
            self._min_depth = 5  # mapf.find_minimum_depth(kwargs["os_draft"], enc)
            self._t_prev = t
            self._map_origin = ownship_state[:2]
            self._initialized = True
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(
                self._min_depth, enc, buffer=0.0, show_plots=True
            )
            self._geometry_tree, _ = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)
            safe_sea_triangulation = mapf.create_safe_sea_triangulation(enc, self._min_depth, show_plots=False)
            self._rrt.transfer_bbox(enc.bbox)
            self._rrt.transfer_enc_hazards(relevant_grounding_hazards[0])
            self._rrt.transfer_safe_sea_triangulation(safe_sea_triangulation)
            self._rrt.set_goal_state(goal_state.tolist())

            U_d = ownship_state[3]  # Constant desired speed given by the initial own-ship speed

            rrt_solution: dict = self._rrt.grow_towards_goal(
                ownship_state=ownship_state.tolist(),
                U_d=U_d,
                initialized=False,
                return_on_first_solution=False,
                verbose=True,
            )
            self._rrt_waypoints, self._rrt_trajectory, self._rrt_inputs, times, cost = parse_rrt_solution(rrt_solution)  # noqa: F841

            if enc is not None and t == 0.0:
                plotters.plot_rrt_tree(self._rrt.get_tree_as_list_of_dicts(), enc)
                enc.draw_circle(center=(goal_state[1], goal_state[0]), radius=20.0, color="magenta", alpha=0.7)
                plotters.plot_waypoints(
                    self._rrt_waypoints,
                    enc,
                    color="orange",
                    point_buffer=3.0,
                    disk_buffer=6.0,
                    hole_buffer=3.0,
                )
                ship_poly = mapf.create_ship_polygon(
                    ownship_state[0],
                    ownship_state[1],
                    ownship_state[2],
                    kwargs["os_length"],
                    kwargs["os_width"],
                    2.0,
                    2.0,
                )
                print(f"Num tree nodes: {self._rrt.get_num_nodes()}")
                enc.draw_polygon(ship_poly, color="yellow")

        else:
            self._rrt_trajectory = self._rrt_trajectory[:, 1:]
            self._rrt_inputs = self._rrt_inputs[:, 1:]

        self._t_prev = t
        self._references = self._los.compute_references(
            self._rrt_waypoints[:2, :],
            speed_plan=self._rrt_waypoints[2, :],
            times=None,
            xs=ownship_state,
            dt=t - self._t_prev,
        )
        return self._references

    def get_current_plan(self) -> np.ndarray:  # noqa: D102
        return self._references

    def get_colav_data(self) -> dict:  # noqa: D102
        if not self._initialized:
            return {
                "nominal_trajectory": np.zeros((6, 1)),
                "nominal_inputs": np.zeros((3, 1)),
                "params": self._rrt_config.params,
                "t": self._t_prev,
            }
        else:
            return {
                "nominal_trajectory": self._rrt_trajectory,
                "nominal_inputs": self._rrt_inputs,
                "params": self._rrt_config.params,
                "t": self._t_prev,
            }

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:  # noqa: ARG002, D102
        if self._rrt_trajectory.size > 6:
            plt_handles["colav_nominal_trajectory"].set_xdata(self._rrt_trajectory[1, 0::4])
            plt_handles["colav_nominal_trajectory"].set_ydata(self._rrt_trajectory[0, 0::4])
        return plt_handles


if __name__ == "__main__":
    params = RRTPlannerParams()
    params.rrt.params = PQRRTStarParams(
        max_nodes=3000,
        max_iter=10000,
        max_time=5.0,
        iter_between_direct_goal_growth=500,
        min_node_dist=5.0,
        goal_radius=300.0,
        step_size=0.5,
        min_steering_time=1.0,
        max_steering_time=30.0,
        steering_acceptance_radius=10.0,
        gamma=2000.0,
        max_sample_adjustments=0,
        lambda_sample_adjustment=10.0,
        safe_distance=0.5,
        max_ancestry_level=1,
    )
    rrt = PQRRTStar(params)

    scenario_file = dp.scenarios / "rrt_test.yaml"
    scenario_generator = ScenarioGenerator()
    scenario_generator.behavior_generator.set_ownship_method(BehaviorGenerationMethod.ConstantSpeedAndCourse)

    scenario_data = scenario_generator.generate(config_file=scenario_file, new_load_of_map_data=True, show_plots=False)
    simulator = Simulator()
    simulator.toggle_liveplot_visibility(True)
    # Hint: Close ENC Seacharts plot with RRT tree to speed up livesim
    output = simulator.run([scenario_data], colav_systems=[(0, rrt)], terminate_on_collision_or_grounding=False)
