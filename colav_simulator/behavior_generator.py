"""behavior_generator.py.

Summary:
Contains a class for generating random ship behaviors/waypoints+speed plans/trajectories.

Author: Trym Tengesdal
"""

import time
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np
import seacharts.enc as senc

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
from colav_simulator.common import plotters
from colav_simulator.core import guidances, models, ship

RRT_LIB_FOUND = True
try:
    import rrt_star_lib  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError as err:
    print(f"Warning: rrt_star_lib not found! Error msg: {err}")
    RRT_LIB_FOUND = False


np.set_printoptions(suppress=True, formatter={"float_kind": "{:.2f}".format})


@dataclass
class PQRRTStarParams:
    max_nodes: int = 3000  # Maximum allowable number of nodes in the tree
    max_iter: int = 12000  # Maximum allowable number of iterations
    max_time: float = 5.0  # Maximum allowable runtime in seconds
    iter_between_direct_goal_growth: int = 100  # Number of iterations between direct growth towards the goal
    min_node_dist: float = 10.0  # Minimum distance between nodes in nearest neighbor search
    goal_radius: float = 500.0  # Radius of goal region
    step_size: float = 1.0  # Step size for steering/dynamics
    min_steering_time: float = 1.0  # Minimum steering time allowed
    max_steering_time: float = 30.0  # Maximum steering time allowed
    steering_acceptance_radius: float = 10.0  # Radius of acceptance for steering (using LOS)
    gamma: float = 1500.0  # Nearest neighbor search radius parameter
    max_sample_adjustments: int = 100  # Maximum number of sample adjustments allowed in Potential field sampling
    lambda_sample_adjustment: float = 1.0  # Sample "strength" parameter for Potential field sampling
    safe_distance: float = 0.0  # Safe distance to obstacles (used in Potential field sampling)
    max_ancestry_level: int = (
        2  # Maximum number of levels to go back in the tree when searching for a parent node in the tree wiring
    )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PQRRTStarParams":  # noqa: D102
        config = cls(**config_dict)
        return config

    def to_dict(self) -> dict:  # noqa: D102
        return asdict(self)


@dataclass
class RRTStarParams:
    max_nodes: int = 3000  # Maximum allowable number of nodes in the tree
    max_iter: int = 12000  # Maximum allowable number of iterations
    max_time: float = 5.0  # Maximum allowable runtime in seconds
    iter_between_direct_goal_growth: int = 100  # Number of iterations between direct growth towards the goal
    min_node_dist: float = 10.0  # Minimum distance between nodes in nearest neighbor search
    goal_radius: float = 500.0  # Radius of goal region
    step_size: float = 1.0  # Step size for steering/dynamics
    min_steering_time: float = 1.0  # Minimum steering time allowed
    max_steering_time: float = 30.0  # Maximum steering time allowed
    steering_acceptance_radius: float = 10.0  # Radius of acceptance for steering (using LOS)
    gamma: float = 1500.0  # Nearest neighbor search radius parameter

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RRTStarParams":  # noqa: D102
        config = cls(**config_dict)
        return config

    def to_dict(self) -> dict:  # noqa: D102
        return asdict(self)


@dataclass
class RRTParams:
    max_nodes: int = 3000  # Maximum allowable number of nodes in the tree
    max_iter: int = 12000  # Maximum allowable number of iterations
    max_time: float = 5.0  # Maximum allowable runtime in seconds
    iter_between_direct_goal_growth: int = 100  # Number of iterations between direct growth towards the goal
    goal_radius: float = 500.0  # Radius of goal region
    step_size: float = 1.0  # Step size for steering/dynamics
    min_steering_time: float = 1.0  # Minimum steering time allowed
    max_steering_time: float = 30.0  # Maximum steering time allowed
    steering_acceptance_radius: float = 10.0  # Radius of acceptance for steering (using LOS)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RRTParams":  # noqa: D102
        config = cls(**config_dict)
        return config

    def to_dict(self) -> dict:  # noqa: D102
        return asdict(self)


@dataclass
class RRTConfig:
    params: RRTParams | RRTStarParams | PQRRTStarParams = field(default_factory=lambda: RRTParams())
    model: models.KinematicCSOGParams = field(
        default_factory=lambda: models.KinematicCSOGParams(
            name="KinematicCSOG",
            draft=0.5,
            length=15.0,
            width=4.0,
            T_chi=6.0,
            T_U=6.0,
            r_max=np.deg2rad(10),
            U_min=0.0,
            U_max=10.0,
        )
    )
    los: guidances.LOSGuidanceParams = field(
        default_factory=lambda: guidances.LOSGuidanceParams(
            K_p=0.035,
            K_i=0.0,
            pass_angle_threshold=90.0,
            R_a=25.0,
            max_cross_track_error_int=200.0,
            cross_track_error_int_threshold=30.0,
        )
    )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RRTConfig":  # noqa: D102
        config = RRTConfig()
        if "max_sample_adjustments" in config_dict["params"]:
            config.params = PQRRTStarParams.from_dict(config_dict["params"])
        elif "gamma" in config_dict["params"]:
            config.params = RRTStarParams.from_dict(config_dict["params"])
        else:
            config.params = RRTParams.from_dict(config_dict["params"])
        config.model = models.KinematicCSOGParams.from_dict(config_dict["model"]["csog"])
        config.los = guidances.LOSGuidanceParams.from_dict(config_dict["los"])
        return config

    def to_dict(self) -> dict:  # noqa: D102
        output = asdict(self)
        output["params"] = self.params.to_dict()
        output["model"] = self.model.to_dict()
        output["los"] = self.los.to_dict()
        return output


class BehaviorGenerationMethod(Enum):
    """Enum for the supported methods for generating ship behaviors/waypoints."""

    ConstantSpeedAndCourse = 0  # Constant ship speed and course
    ConstantSpeedRandomWaypoints = 1  # Constant ship speed, uniform distribution to generate ship waypoints
    VaryingSpeedRandomWaypoints = 2  # Uniformly varying ship speed, uniform distribution to generate ship waypoints
    RRT = 3  # Use baseline RRT to generate ship trajectories/waypoints, with constant speed
    RRTStar = 4  # Use Informed RRT* to generate ship trajectories/waypoints, with constant speed
    PQRRTStar = 5  # Use PQ-RRT* to generate ship trajectories/waypoints, with constant speed
    Randomize = 6  # Any of the above methods can be chosen at random


class RRTBehaviorSamplingMethod(Enum):
    """Enum for the supported methods for sampling waypoints and speed plans in target ship RRT-based behavior generation.

    NOTE: Naturally only applicable when RRT-based behavior generation is used.

    Unless Optimal is used, the procedure is as follows:
    1: A position sample is generated based on the sampling method (random,
    CPA-based, OwnshipGoal, etc.)
    2: The tree built by the RRT-based planner is searched for a node that is
    closest to the sample, and the trajectory from the root to this node is
    extracted.
    """

    Optimal = 0  # Sample waypoints and speed plans as the optimal RRT solution
    UniformlyInMap = 1  # Sample waypoints and speed plans from the RRT tree
    # with goal/endpoint near a random map position (inside the planning bbox)
    CourseCorridor = 2  # Sample waypoints and speed plans from the RRT tree
    # with endpoint inside a corridor along ones initial course/heading
    CPA = 3  # Sample waypoints and speed plans from the RRT tree with endpoint
    # in the direction of a random multivariate gaussian centered on the CPA
    # position
    OwnshipGoal = 4  # Sample waypoints and speed plans from the RRT tree with
    # endpoints near a multivariate gaussian sample centered on the ownship goal
    OwnshipWaypointCorridor = 5  # Sample waypoints and speed plans from the
    # RRT tree with endpoint near a sample inside the own-ship waypoint corridor.
    Randomized = 6  # Randomize any of the above methods


@dataclass
class Config:
    """Configuration class for the behavior generator."""

    ownship_method: BehaviorGenerationMethod = BehaviorGenerationMethod.ConstantSpeedAndCourse
    target_ship_method: BehaviorGenerationMethod = BehaviorGenerationMethod.ConstantSpeedAndCourse
    target_ship_rrt_behavior_sampling_method: RRTBehaviorSamplingMethod = RRTBehaviorSamplingMethod.CPA
    n_wps_range: list = field(default_factory=lambda: [2, 4])  # Range of number of waypoints to be generated
    speed_plan_variation_range: list = field(
        default_factory=lambda: [-1.0, 1.0]
    )  # Determines maximal +- change in speed plan from one segment to the next
    waypoint_dist_range: list = field(
        default_factory=lambda: [200.0, 1000.0]
    )  # Range of [min, max] change in distance between randomly created waypoints
    waypoint_ang_range: list = field(
        default_factory=lambda: [-45.0, 45.0]
    )  # Range of [min, max] change in angle between randomly created waypoints
    hazard_buffer: float = 0.0  # Buffer to add to hazards when creating safe sea triangulation
    rrt: RRTConfig | None = field(
        default_factory=lambda: RRTConfig(
            params=RRTParams(), model=models.KinematicCSOGParams(), los=guidances.LOSGuidanceParams()
        )
    )
    rrtstar: RRTConfig | None = field(
        default_factory=lambda: RRTConfig(
            params=RRTStarParams(), model=models.KinematicCSOGParams(), los=guidances.LOSGuidanceParams()
        )
    )
    pqrrtstar: RRTConfig | None = field(
        default_factory=lambda: RRTConfig(
            params=PQRRTStarParams(), model=models.KinematicCSOGParams(), los=guidances.LOSGuidanceParams()
        )
    )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":  # noqa: D102
        config = Config(
            ownship_method=BehaviorGenerationMethod[config_dict["ownship_method"]],
            target_ship_method=BehaviorGenerationMethod[config_dict["target_ship_method"]],
            target_ship_rrt_behavior_sampling_method=RRTBehaviorSamplingMethod[
                config_dict["target_ship_rrt_behavior_sampling_method"]
            ],
            n_wps_range=config_dict["n_wps_range"],
            speed_plan_variation_range=config_dict["speed_plan_variation_range"],
            waypoint_dist_range=config_dict["waypoint_dist_range"],
            waypoint_ang_range=config_dict["waypoint_ang_range"],
        )

        if "pqrrtstar" in config_dict:
            config.pqrrtstar = RRTConfig.from_dict(config_dict["pqrrtstar"])

        if "rrtstar" in config_dict:
            config.rrtstar = RRTConfig.from_dict(config_dict["rrtstar"])

        if "rrt" in config_dict:
            config.rrt = RRTConfig.from_dict(config_dict["rrt"])
        return config

    def to_dict(self) -> dict:  # noqa: D102
        output = asdict(self)
        output["rrt"] = self.rrt.to_dict()
        output["rrtstar"] = self.rrtstar.to_dict()
        output["pqrrtstar"] = self.pqrrtstar.to_dict()
        output["ownship_method"] = self.ownship_method.name
        output["target_ship_method"] = self.target_ship_method.name
        output["target_ship_rrt_behavior_sampling_method"] = self.target_ship_rrt_behavior_sampling_method.name
        return output


class BehaviorGenerator:
    """Class for generating random ship behaviors, using one or multiple methods."""

    def __init__(self, config: Config) -> None:
        self._config: Config = config
        self._enc: senc.ENC = None
        self._safe_sea_cdt: list = None
        self._safe_sea_cdt_weights: list = None

        self._simulation_timespan: float = 0.0
        self._prev_ship_plans: list = []
        self._prev_ship_states: list = []
        self._ship_replan_flags: list = []
        self._planning_bbox_list: list = []
        self._planning_hazard_list: list = []
        self._planning_cdt_list: list = []
        self._rrt_list: list = []
        self._rrtstar_list: list = []
        self._pqrrtstar_list: list = []
        self._grounding_hazards: list = []
        self._seed: int | None = None
        self._bg_method_list: list = []
        self._initialized: bool = False

    def seed(self, seed: int | None = None) -> None:
        """Seeds the behavior generator, i.e. all RRTs/pqrrtstars.

        Args:
            seed (Optional[int]): Integer seed. Defaults to None.
        """
        self._seed = seed
        if len(self._rrt_list) > 0:
            for rrt in self._rrt_list:
                rrt.reset(seed)
            for rrtstar in self._rrtstar_list:
                rrtstar.reset(seed)
            for pqrrtstar in self._pqrrtstar_list:
                pqrrtstar.reset(seed)

    def set_ownship_method(self, method: BehaviorGenerationMethod) -> None:
        """Sets the ownship method.

        Args:
            method (BehaviorGenerationMethod): The ownship method.
        """
        self._config.ownship_method = method

    def set_target_ship_method(self, method: BehaviorGenerationMethod) -> None:
        """Sets the target ship method.

        Args:
            method (BehaviorGenerationMethod): The target ship method.
        """
        self._config.target_ship_method = method

    def reset(self) -> None:
        """Resets the behavior generator, i.e. all RRTs and data structures for the environment."""
        self._initialized = False
        self._prev_ship_plans = []
        self._prev_ship_states = []
        self._ship_replan_flags = []
        self._planning_bbox_list = []
        self._planning_hazard_list = []
        self._planning_cdt_list = []
        self._bg_method_list = []
        self._rrt_list = []
        self._rrtstar_list = []
        self._pqrrtstar_list = []
        self._grounding_hazards = []

    def _initialize_rrts(self, n_rrts: int) -> tuple[list, list, list]:
        """Initializes the RRTs.

        Args:
            n_rrts (int): Number of RRTs to be initialized.
        """
        self._rrt_list = [
            rrt_star_lib.RRT(los=self._config.rrt.los, model=self._config.rrt.model, params=self._config.rrt.params)
            for _ in range(n_rrts)
        ]
        self._rrtstar_list = [
            rrt_star_lib.RRTStar(
                los=self._config.rrtstar.los, model=self._config.rrtstar.model, params=self._config.rrtstar.params
            )
            for _ in range(n_rrts)
        ]
        self._pqrrtstar_list = [
            rrt_star_lib.PQRRTStar(
                los=self._config.pqrrtstar.los, model=self._config.pqrrtstar.model, params=self._config.pqrrtstar.params
            )
            for _ in range(n_rrts)
        ]
        return self._rrt_list, self._rrtstar_list, self._pqrrtstar_list

    def initialize_data_structures(self, n_ships: int) -> None:
        """Initializes data structures and RRTs (if enabled).

        The number of RRTs is determined by the number of ships. If the number
        of ships in the input list is different from the number of RRTs, the
        data structures and RRTs are either extended or truncated to match the
        number of ships.

        Args:
            n_ships (int): Number of ships to be considered in the simulation.
        """
        self.reset()
        self._prev_ship_plans = [None for _ in range(n_ships)]
        self._prev_ship_states = [np.empty(4) for _ in range(n_ships)]
        self._planning_bbox_list = [None for _ in range(n_ships)]
        self._planning_hazard_list = [None for _ in range(n_ships)]
        self._planning_cdt_list = [None for _ in range(n_ships)]
        self._bg_method_list = [None for _ in range(n_ships)]
        self._ship_replan_flags = [True for _ in range(n_ships)]
        if (
            self._config.ownship_method.value >= BehaviorGenerationMethod.RRT.value
            or self._config.target_ship_method.value >= BehaviorGenerationMethod.RRT.value
        ):
            if not RRT_LIB_FOUND:
                msg = "You specified usage of RRT for ship behavior generation, " "but the RRT library is not found."
                raise ValueError(msg)
            self._rrt_list, self._rrtstar_list, self._pqrrtstar_list = self._initialize_rrts(n_ships)

    def setup_enc(self, enc: senc.ENC, safe_sea_cdt: list, safe_sea_cdt_weights: list, show_plots: bool = False) -> None:
        """Sets up the environment for the behavior generator.

        Transfers ENC data and creates a safe sea triangulation.

        Args:
            enc (senc.ENC): Electronic navigational chart.
            safe_sea_cdt (list): Safe sea triangulation.
            safe_sea_cdt_weights (list): Weights for the safe sea triangulation.
            show_plots (bool): Whether to show ENC plots. Defaults to False.
        """
        self._enc = enc
        if show_plots:
            self._enc.start_display()

        # Assume equal min depth for all ships, for now
        self._safe_sea_cdt = safe_sea_cdt
        self._safe_sea_cdt_weights = safe_sea_cdt_weights
        self._grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(
            vessel_min_depth=0, enc=self._enc, buffer=self._config.hazard_buffer, show_plots=show_plots
        )

    def setup(
        self,
        rng: np.random.Generator,
        ship_list: list,
        ship_replan_flags: list,
        simulation_timespan: float,
        show_plots: bool = False,
    ) -> None:
        """Setup the environment for the behavior generator.

        Transfers ENC data to the RRTs if configured, and creates a safe sea
        triangulation for more efficient sampling.

        Args:
            rng (np.random.Generator): Random number generator.
            ship_list (list): List of ships to be considered in simulation.
            ship_replan_flags (list): List of flags indicating whether a ship
                should have a new behavior generated.
            simulation_timespan (float): Simulation timespan.
            show_plots (bool): Whether to show plots. Defaults to False.
        """
        ownship = ship_list[0]
        self._simulation_timespan = simulation_timespan
        self._ship_replan_flags = ship_replan_flags
        for idx, ship_obj in enumerate(ship_list):
            self.setup_ship(
                rng=rng,
                ship_obj=ship_obj,
                replan=ship_replan_flags[idx],
                ownship=ownship,
                simulation_timespan=simulation_timespan,
                show_plots=show_plots,
            )

    def setup_ship(  # noqa: PLR0912, PLR0915
        self,
        rng: np.random.Generator,
        ship_obj: ship.Ship,
        replan: bool,
        simulation_timespan: float,
        ownship: ship.Ship | None = None,
        show_plots: bool = False,
    ) -> None:
        """Sets up a ship for behavior generation.

        Generates a behavior if the ship has not been configured fully yet.

        Args:
            rng (np.random.Generator): Random number generator.
            ship_obj (ship.Ship): Ship to be considered in simulation.
            replan (bool): Flag indicating whether a ship should have a new
                behavior generated when using RRT.
            simulation_timespan (float): Simulation timespan.
            ownship (ship.Ship | None): The ownship, used for generating target
                ship RRT-goals. Defaults to None.
            show_plots (bool): Whether to show plots. Defaults to False.
        """
        method = self._config.target_ship_method if ship_obj.id > 0 else self._config.ownship_method
        if method == BehaviorGenerationMethod.Randomize:
            method = BehaviorGenerationMethod(rng.integers(0, BehaviorGenerationMethod.Randomize.value))

        self._bg_method_list[ship_obj.id] = method
        run_rrt = not np.array_equal(ship_obj.csog_state, self._prev_ship_states[ship_obj.id])
        if (
            method.value < BehaviorGenerationMethod.RRT.value
            or ship_obj.waypoints.size > 0
            or ship_obj.trajectory.size > 0
            or not replan
            or not run_rrt
        ):
            return

        ownship_bbox = None
        if ship_obj.id > 0:
            if ownship is None:
                msg = "Ownship must be specified when generating target ship behavior using RRT."
                raise ValueError(msg)
            target_ship_state = ship_obj.csog_state
            if ownship.waypoints.size > 0:
                ownship_waypoints = ownship.waypoints
            elif ownship.goal_csog_state.size > 0:
                ownship_waypoints = np.hstack(
                    [ownship.csog_state[0:2].reshape(2, 1), ownship.goal_csog_state[0:2].reshape(2, 1)]
                )
            else:
                ownship_waypoints = np.hstack(
                    [ownship.csog_state[0:2].reshape(2, 1), ownship.csog_state[0:2].reshape(2, 1) + 500.0]
                )

            xmax = np.max([target_ship_state[0], *ownship_waypoints[0, :].tolist()])
            ymax = np.max([target_ship_state[1], *ownship_waypoints[1, :].tolist()])
            xmin = np.min([target_ship_state[0], *ownship_waypoints[0, :].tolist()])
            ymin = np.min([target_ship_state[1], *ownship_waypoints[1, :].tolist()])
            pmin = np.array([xmin, ymin])
            pmax = np.array([xmax, ymax])
            bbox = mapf.create_bbox_from_points(self._enc, pmin, pmax, buffer=500.0)

        goal_state = ship_obj.goal_state
        if goal_state.size < 1:
            goal_position = mapf.generate_random_goal_position(
                rng=rng,
                enc=self._enc,
                xs_start=ship_obj.csog_state,
                safe_sea_cdt=self._safe_sea_cdt,
                safe_sea_cdt_weights=self._safe_sea_cdt_weights,
                bbox=ownship_bbox,
                min_distance_from_start=300.0,
                max_distance_from_start=min(1200.0, 5.0 * ship_obj.speed * simulation_timespan),
                show_plots=False,
            )
            goal_state = np.array([goal_position[0], goal_position[1], 0.0, 0.0, 0.0, 0.0])

            ship_obj.set_goal_state(goal_state[:4])

        bbox = mapf.create_bbox_from_points(self._enc, ship_obj.csog_state[:2], goal_state[:2], buffer=800.0)
        relevant_hazards = mapf.extract_hazards_within_bounding_box(
            self._grounding_hazards, bbox, self._enc, show_plots=show_plots
        )
        planning_cdt = mapf.create_safe_sea_triangulation(
            self._enc,
            vessel_min_depth=0,
            bbox=bbox,
            show_plots=False,
        )

        self._planning_bbox_list[ship_obj.id] = bbox
        self._planning_hazard_list[ship_obj.id] = relevant_hazards[0]
        self._planning_cdt_list[ship_obj.id] = planning_cdt
        U_d = ship_obj.csog_state[2]
        if method == BehaviorGenerationMethod.RRT:
            planner = self._rrt_list[ship_obj.id]
        elif method == BehaviorGenerationMethod.RRTStar:
            planner = self._rrtstar_list[ship_obj.id]
        elif method == BehaviorGenerationMethod.PQRRTStar:
            planner = self._pqrrtstar_list[ship_obj.id]

        planner.transfer_bbox(bbox)
        planner.transfer_enc_hazards(relevant_hazards[0])
        planner.transfer_safe_sea_triangulation(planning_cdt)
        planner.set_init_state(ship_obj.state.tolist())
        planner.set_goal_state(goal_state.tolist())
        planner.reset(self._seed)
        rrt_soln = planner.grow_towards_goal(
            ownship_state=ship_obj.state.tolist(),
            U_d=U_d,
            initialized=False,
            return_on_first_solution=False,
            verbose=False,
        )
        waypoints, _, _, _ = mhm.parse_rrt_solution(rrt_soln)
        speed_plan = waypoints[2, :]
        waypoints = waypoints[0:2, :]
        if waypoints.size == 0:
            print(f"WARNING: RRT-based planner failed to generate feasible waypoints for ship {ship_obj.id}!")
            waypoints, speed_plan = self.generate_constant_speed_and_course_waypoints(
                ship_obj.csog_state, ship_obj.draft, ship_obj.length, simulation_timespan
            )
            # plotters.plot_rrt_tree(planner.get_tree_as_list_of_dicts(), self._enc)
        if ship_obj.id == 0:
            if show_plots:
                self._enc.draw_circle((goal_state[1], goal_state[0]), 10, color="gold", alpha=0.4)
            ship_obj.set_nominal_plan(waypoints, speed_plan)
            self._prev_ship_plans[ship_obj.id] = waypoints, speed_plan

        self._prev_ship_states[ship_obj.id] = ship_obj.csog_state

    def plot_rrt_planner_tree(self, ship_id: int) -> None:
        """Plots the RRT planner tree for the specified ship.

        Args:
            ship_id (int): Ship ID.
        """
        if not RRT_LIB_FOUND:
            msg = "You specified usage of RRT for ship behavior generation, " "but the RRT library is not found."
            raise ValueError(msg)
        if ship_id >= len(self._rrt_list):
            msg = "Invalid ship ID."
            raise ValueError(msg)
        if self._bg_method_list[ship_id] == BehaviorGenerationMethod.RRT:
            plotters.plot_rrt_tree(self._rrt_list[ship_id].get_tree_as_list_of_dicts(), self._enc)
        elif self._bg_method_list[ship_id] == BehaviorGenerationMethod.RRTStar:
            plotters.plot_rrt_tree(self._rrtstar_list[ship_id].get_tree_as_list_of_dicts(), self._enc)
        elif self._bg_method_list[ship_id] == BehaviorGenerationMethod.PQRRTStar:
            plotters.plot_rrt_tree(self._pqrrtstar_list[ship_id].get_tree_as_list_of_dicts(), self._enc)

    def visualize_ship_behaviors(self, ship_list: list) -> None:
        """Plots the ship waypoints and trajectories.

        Args:
            ship_list (list): List of ships to be considered in simulation.
        """
        for ship_idx in reversed(range(len(ship_list))):
            ship_obj = ship_list[ship_idx]
            if self._enc is not None and ship_obj.waypoints.size > 1:
                color = "purple" if ship_obj.id == 0 else "orangered"
                plotters.plot_waypoints(
                    ship_obj.waypoints,
                    self._enc,
                    color=color,
                    point_buffer=2.0,
                    disk_buffer=6.0,
                    hole_buffer=2.0,
                )
            if ship_obj.trajectory.size > 1:
                color = "purple" if ship_obj.id == 0 else "orangered"
                plotters.plot_trajectory(ship_obj.trajectory, self._enc, color=color, alpha=0.6)
            color = "magenta" if ship_obj.id == 0 else "red"
            ship_poly = mapf.create_ship_polygon(
                ship_obj.csog_state[0],
                ship_obj.csog_state[1],
                mf.wrap_angle_to_pmpi(ship_obj.csog_state[3]),
                ship_obj.length,
                ship_obj.width,
                5.0,
                5.0,
            )
            self._enc.draw_polygon(ship_poly, color=color, alpha=0.6)

    def generate(
        self,
        rng: np.random.Generator,
        ship_list: list,
        ship_config_list: list,
        simulation_timespan: float,
        show_plots: bool = True,
    ) -> tuple[list, list]:
        """Generates ship behaviors in the form of waypoints + speed plan.

        For ships that have not been configured fully yet.

        Args:
            rng (np.random.Generator): Random number generator.
            ship_list (list): List of ships to be considered in simulation.
            ship_config_list (list): List of ship configurations.
            simulation_timespan (float): Simulation timespan.
            show_plots (bool): Whether to show plots. Defaults to True.

        Returns:
            tuple[list, list]: Tuple containing the waypoints, speed plan and
                possibly trajectory for the ship.
        """
        ownship = ship_list[0]
        for ship_cfg_idx, ship_config_item in enumerate(ship_config_list):
            ship_obj, ship_config = self.generate_ship_behavior(
                rng,
                ship_list[ship_cfg_idx],
                ship_config_item,
                simulation_timespan,
                ownship=ownship,
                reuse_old_behavior=not self._ship_replan_flags[ship_cfg_idx],
            )
            ship_list[ship_cfg_idx] = ship_obj
            ship_config_list[ship_cfg_idx] = ship_config
            self._prev_ship_plans[ship_cfg_idx] = ship_config.waypoints, ship_config.speed_plan
        if show_plots:
            self.visualize_ship_behaviors(ship_list)
        return ship_list, ship_config_list

    def generate_ship_behavior(
        self,
        rng: np.random.Generator,
        ship_obj: ship.Ship,
        ship_config: ship.Config,
        simulation_timespan: float,
        ownship: ship.Ship | None = None,
        reuse_old_behavior: bool = False,
    ) -> tuple[ship.Ship, ship.Config]:
        """Generates a ship behavior using the specified method by the ship configuration.

        Args:
            rng (np.random.Generator): Random number generator.
            ship_obj (ship.Ship): Ship to generate behavior for.
            ship_config (ship.Config): Ship configuration.
            simulation_timespan (float): Simulation timespan.
            ownship (ship.Ship, optional): The ownship, used if an RRT-based method is chosen for the target ship.
            reuse_old_behavior (bool, optional): Whether to reuse the old behavior. Defaults to False.

        Returns:
            Tuple[ship.Ship, ship.Config]: Tuple containing the updated ship and its configuration.
        """
        if ship_obj.trajectory.size > 1 or ship_obj.waypoints.size > 1:
            ship_config.waypoints = ship_obj.waypoints
            ship_config.speed_plan = ship_obj.speed_plan
            return ship_obj, ship_config

        if reuse_old_behavior:
            prev_waypoints, prev_speed_plan = self._prev_ship_plans[ship_obj.id]
            ship_config.waypoints = prev_waypoints
            ship_config.speed_plan = prev_speed_plan
            ship_obj.set_nominal_plan(ship_config.waypoints, ship_config.speed_plan)
            return ship_obj, ship_config

        method = self._bg_method_list[ship_obj.id]
        if method == BehaviorGenerationMethod.ConstantSpeedAndCourse:
            min_dist_to_hazards = ship_obj.length * 4.0 if ship_obj.id == 0 else ship_obj.length * 3.0
            waypoints, speed_plan = self.generate_constant_speed_and_course_waypoints(
                ship_obj.csog_state, ship_obj.draft, simulation_timespan, min_dist_to_hazards=min_dist_to_hazards
            )
        elif method == BehaviorGenerationMethod.ConstantSpeedRandomWaypoints:
            waypoints, _ = self.generate_random_waypoints(
                rng,
                ship_obj.csog_state[0],
                ship_obj.csog_state[1],
                ship_obj.csog_state[3],
                ship_obj.draft,
                min_dist_to_hazards=ship_obj.length * 3.0,
            )
            speed_plan = ship_obj.csog_state[2] * np.ones(waypoints.shape[1])
        elif method == BehaviorGenerationMethod.VaryingSpeedRandomWaypoints:
            waypoints, _ = self.generate_random_waypoints(
                rng,
                ship_obj.csog_state[0],
                ship_obj.csog_state[1],
                ship_obj.csog_state[3],
                ship_obj.draft,
                min_dist_to_hazards=ship_obj.length * 3.0,
            )
            speed_plan = self.generate_random_speed_plan(
                rng,
                ship_obj.csog_state[2],
                U_min=ship_obj.min_speed,
                U_max=ship_obj.max_speed,
                n_wps=waypoints.shape[1],
            )
        elif (
            RRT_LIB_FOUND and BehaviorGenerationMethod.RRT.value <= method.value <= BehaviorGenerationMethod.PQRRTStar.value
        ):
            waypoints, speed_plan, _ = self.generate_rrt_behavior(rng, ship_obj, ownship, method, show_plots=True)

        ship_config.waypoints = waypoints
        ship_config.speed_plan = speed_plan
        ship_obj.set_nominal_plan(ship_config.waypoints, ship_config.speed_plan)
        return ship_obj, ship_config

    def generate_rrt_behavior(  # noqa: C901, PLR0912, PLR0915
        self,
        rng: np.random.Generator,
        ship_obj: ship.Ship,
        ownship: ship.Ship,
        rrt_method: BehaviorGenerationMethod,
        show_plots: bool = True,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates a ship behavior using an RRT-variant.

        Args:
            rng (np.random.Generator): Random number generator.
            ship_obj (ship.Ship): The ship to generate a behavior for.
            ownship (ship.Ship): The ownship.
            rrt_method (BehaviorGenerationMethod): The RRT method to use.
            show_plots (bool): Whether to show plots. Defaults to True.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the
                resulting waypoints, speed plan and trajectory.
        """
        if rrt_method not in (
            BehaviorGenerationMethod.RRT,
            BehaviorGenerationMethod.RRTStar,
            BehaviorGenerationMethod.PQRRTStar,
        ):
            msg = f"Invalid RRT method: {rrt_method}"
            raise ValueError(msg)

        if rrt_method == BehaviorGenerationMethod.RRT:
            planner = self._rrt_list[ship_obj.id]
            # print("Using RRT for behavior generation...")
        elif rrt_method == BehaviorGenerationMethod.RRTStar:
            planner = self._rrtstar_list[ship_obj.id]
            # print("Using RRT* for behavior generation...")
        elif rrt_method == BehaviorGenerationMethod.PQRRTStar:
            planner = self._pqrrtstar_list[ship_obj.id]
            # print("Using PQ-RRT* for behavior generation...")

        if ship_obj.id == 0:
            # If the own-ship uses RRT for behavior/wp+speed plan generation,
            # we use the optimal solution.
            opt_soln = planner.get_optimal_solution()
            waypoints, trajectory, _, _ = mhm.parse_rrt_solution(opt_soln)
            speed_plan = waypoints[2, :]
            waypoints = waypoints[:2, :]
            return waypoints, speed_plan, trajectory

        p_os = ownship.csog_state[0:2]
        v_os = np.array(
            [
                ownship.csog_state[2] * np.cos(ownship.csog_state[3]),
                ownship.csog_state[2] * np.sin(ownship.csog_state[3]),
            ]
        )
        p_target = ship_obj.csog_state[0:2]
        v_target = np.array(
            [
                ship_obj.csog_state[2] * np.cos(ship_obj.csog_state[3]),
                ship_obj.csog_state[2] * np.sin(ship_obj.csog_state[3]),
            ]
        )

        ownship_waypoints = (
            ownship.waypoints
            if ownship.waypoints is not None
            else np.hstack([ownship.csog_state[0:2].reshape(2, 1), ownship.goal_csog_state[0:2].reshape(2, 1)])
        )
        planning_bbox = self._planning_bbox_list[ship_obj.id]

        sampling_method = self._config.target_ship_rrt_behavior_sampling_method
        if sampling_method == RRTBehaviorSamplingMethod.Randomized:
            sampling_method = RRTBehaviorSamplingMethod(rng.integers(0, RRTBehaviorSamplingMethod.Randomized.value))

        if sampling_method == RRTBehaviorSamplingMethod.Optimal:
            opt_soln = planner.get_optimal_solution()
            waypoints, trajectory, _, _ = mhm.parse_rrt_solution(opt_soln)
            speed_plan = waypoints[2, :]
            waypoints = waypoints[:2, :]
            return waypoints, speed_plan, trajectory

        n_samples = 1
        sample_runtimes = np.zeros(n_samples)
        position_covariance = np.diag([100.0**2, 100.0**2])
        corridor_diameter = 150.0
        for s in range(n_samples):
            time_now = time.time()
            for _iter in range(10):
                if sampling_method.value == RRTBehaviorSamplingMethod.UniformlyInMap.value:
                    x_rand = rng.uniform(planning_bbox[1], planning_bbox[3])
                    y_rand = rng.uniform(planning_bbox[0], planning_bbox[2])
                    p_rand = np.array([x_rand, y_rand])
                elif sampling_method.value == RRTBehaviorSamplingMethod.CPA.value:
                    # 3) Minimum dCPA between ownship and target ship
                    p_target = ship_obj.csog_state[0:2]
                    v_target = np.array(
                        [
                            ship_obj.csog_state[2] * np.cos(ship_obj.csog_state[3]),
                            ship_obj.csog_state[2] * np.sin(ship_obj.csog_state[3]),
                        ]
                    )
                    t_cpa, _d_cpa = mf.cpa(p_target, v_target, p_os, v_os)
                    p_target_cpa = p_target + v_target * (
                        t_cpa + 50.0
                    )  # add 50 seconds to CPA time to prevent the ship from stopping at cpa
                    p_rand = rng.multivariate_normal(mean=p_target_cpa, cov=position_covariance)
                elif sampling_method.value == RRTBehaviorSamplingMethod.CourseCorridor.value:
                    p_end = p_target + v_target * 0.6 * self._simulation_timespan
                    corridor_waypoints, _ = mhm.clip_waypoint_segment_to_bbox(
                        np.array([p_target, p_end]).T,
                        (planning_bbox[1], planning_bbox[0], planning_bbox[3], planning_bbox[2]),
                    )
                    corridor_poly = mapf.generate_enveloping_polygon(corridor_waypoints, corridor_diameter)
                    bbox_poly = mapf.bbox_to_polygon(planning_bbox)
                    corridor_poly.intersection(bbox_poly)
                    # if s == 0 and show_plots:
                    #     self._enc.draw_polygon(corridor_poly_inside_bbox, color="black", alpha=0.3)
                    p_rand = mhm.sample_from_waypoint_corridor(rng, corridor_waypoints, 2.0 * corridor_diameter)
                elif sampling_method.value == RRTBehaviorSamplingMethod.OwnshipGoal.value:
                    p_rand = rng.multivariate_normal(mean=ownship_waypoints[:, -1], cov=position_covariance)
                elif sampling_method.value == RRTBehaviorSamplingMethod.OwnshipWaypointCorridor.value:
                    p_rand = mhm.sample_from_waypoint_corridor(rng, ownship_waypoints, 2.0 * corridor_diameter)
                else:
                    raise ValueError(f"Sampling method {sampling_method} not supported!")

                # Distance from start to sample must be at least 200m
                if np.linalg.norm(p_rand - p_target) > 100.0:
                    break

            random_solution = planner.nearest_solution(p_rand.tolist())
            waypoints, trajectory, _, _ = mhm.parse_rrt_solution(random_solution)

            speed_plan = waypoints[2, :]
            waypoints = waypoints[0:2, :]
            t_elapsed = time.time() - time_now
            sample_runtimes[s] = t_elapsed

        # print(
        #     f"RRT-based planner behavior sampling time: "
        #     f"{sample_runtimes.mean():.5f} +/- {sample_runtimes.std():.5f} s | "
        #     f"(min, max): {sample_runtimes.min():.5f}, {sample_runtimes.max():.5f} s"
        # )
        return waypoints, speed_plan, trajectory

    def generate_constant_speed_and_course_waypoints(
        self,
        csog_state: np.ndarray,
        draft: float,
        simulation_timespan: float,
        horizon_modifier: float = 5.0,
        min_dist_to_hazards: float = 20.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates waypoints and speed plan for a ship with constant speed and course.

        Args:
            csog_state (np.ndarray): The ship's CSOG state.
            draft (float): How deep the ship keel is into the water.
            length (float): Length of the ship.
            simulation_timespan (float): Simulation timespan.
            horizon_modifier (float, optional): Endpoint scaling. Defaults to 5.0.
            min_dist_to_hazards (float, optional): Minimum distance to hazards. Defaults to 20.0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the waypoints and speed plan.
        """
        waypoints = np.zeros((2, 2))
        waypoints[:, 0] = csog_state[0:2]
        U = csog_state[2]
        chi = csog_state[3]
        end_position = mapf.find_closest_collision_free_point_on_segment(
            self._enc,
            waypoints[:, 0],
            waypoints[:, 0] + U * np.array([np.cos(chi), np.sin(chi)]) * simulation_timespan * horizon_modifier,
            draft,
            self._grounding_hazards,
            min_dist=min_dist_to_hazards,
        )
        waypoints[:, 1] = end_position
        waypoints, _ = mhm.clip_waypoint_segment_to_bbox(
            waypoints, (self._enc.bbox[1], self._enc.bbox[0], self._enc.bbox[3], self._enc.bbox[2])
        )
        speed_plan = U * np.ones(2)
        return waypoints, speed_plan

    def generate_random_waypoints(
        self,
        rng: np.random.Generator,
        x: float,
        y: float,
        psi: float,
        draft: float = 2.0,
        min_dist_to_hazards: float = 20.0,
        n_wps: int | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Creates random waypoints starting from a ship position and heading.

        Args:
            rng (np.random.Generator): Random number generator.
            x (float): x position (north) of the ship.
            y (float): y position (east) of the ship.
            psi (float): heading of the ship in radians.
            draft (float): How deep the ship keel is into the water.
                Defaults to 2.0.
            min_dist_to_hazards (float): Minimum distance to hazards.
                Defaults to 20.0.
            n_wps (int | None): Number of waypoints to create. Defaults to
                None.

        Returns:
            tuple[np.ndarray, bool]: 2 x n_wps array containing the waypoints,
                and a boolean indicating whether the waypoints were clipped to
                the ENC bbox.
        """
        if n_wps is None:
            n_wps = rng.integers(self._config.n_wps_range[0], self._config.n_wps_range[1] + 1)

        east_min, north_min, east_max, north_max = self._enc.bbox
        waypoints = np.zeros((2, n_wps))
        waypoints[:, 0] = np.array([x, y])
        clipped_to_close_hazard_point = False
        clipped_to_bbox = False
        max_iter = 500
        for i in range(1, n_wps):
            iter_count = -1
            crosses_grounding_hazards = False
            for _ in range(max_iter):
                iter_count += 1

                distance_wp_to_wp = rng.uniform(self._config.waypoint_dist_range[0], self._config.waypoint_dist_range[1])

                alpha = 0
                if i > 1:
                    alpha = np.deg2rad(rng.uniform(self._config.waypoint_ang_range[0], self._config.waypoint_ang_range[1]))

                new_wp = np.array(
                    [
                        waypoints[0, i - 1] + distance_wp_to_wp * np.cos(psi + alpha),
                        waypoints[1, i - 1] + distance_wp_to_wp * np.sin(psi + alpha),
                    ],
                )

                wp_dist_to_hazards = mapf.min_distance_to_hazards(self._grounding_hazards, new_wp[1], new_wp[0])
                if wp_dist_to_hazards < min_dist_to_hazards:
                    continue

                crosses_grounding_hazards = mapf.check_if_segment_crosses_grounding_hazards(
                    enc=self._enc, p1=waypoints[:, i - 1], p2=new_wp, draft=draft, hazards=self._grounding_hazards
                )
                if crosses_grounding_hazards:
                    new_wp = mapf.find_closest_collision_free_point_on_segment(
                        self._enc,
                        waypoints[:, i - 1],
                        new_wp,
                        draft,
                        self._grounding_hazards,
                        min_dist=min_dist_to_hazards,
                    )
                    clipped_to_close_hazard_point = True

                crosses_grounding_hazards = mapf.check_if_segment_crosses_grounding_hazards(
                    enc=self._enc, p1=waypoints[:, i - 1], p2=new_wp, draft=draft, hazards=self._grounding_hazards
                )
                if not crosses_grounding_hazards:
                    break

            waypoints[:, i] = new_wp
            waypoints[:, i - 1 : i + 1], clipped_to_bbox = mhm.clip_waypoint_segment_to_bbox(
                waypoints[:, i - 1 : i + 1], (float(north_min), float(east_min), float(north_max), float(east_max))
            )

            if clipped_to_bbox or clipped_to_close_hazard_point:
                waypoints = waypoints[:, : i + 1]
                break

        if waypoints.shape[1] < 2:
            new_wp = np.array(
                [
                    waypoints[0, i - 1] + 500.0 * np.cos(psi),
                    waypoints[1, i - 1] + 500.0 * np.sin(psi),
                ],
            ).reshape(2, 1)
            waypoints = np.append(waypoints, new_wp, axis=1)
        return waypoints, clipped_to_close_hazard_point or clipped_to_bbox

    def generate_random_speed_plan(
        self, rng: np.random.Generator, U: float, U_min: float = 2.0, U_max: float = 8.0, n_wps: int | None = None
    ) -> np.ndarray:
        """Creates a random speed plan using the input speed and min/max speed.

        Args:
            rng (np.random.Generator): Random number generator.
            U (float): The ship's speed.
            U_min (float): The ship's minimum speed. Defaults to 2.0.
            U_max (float): The ship's maximum speed. Defaults to 8.0.
            n_wps (int | None): Number of waypoints to create. Defaults to
                None.

        Returns:
            np.ndarray: 1 x n_wps array containing the speed plan.
        """
        if n_wps is None:
            n_wps = rng.integers(self._config.n_wps_range[0], self._config.n_wps_range[1] + 1)

        speed_plan = np.zeros(n_wps)
        speed_plan[0] = U
        for i in range(1, n_wps):
            U_mod = rng.uniform(self._config.speed_plan_variation_range[0], self._config.speed_plan_variation_range[1])
            speed_plan[i] = mf.sat(speed_plan[i - 1] + U_mod, U_min, U_max)

            if i == n_wps - 1:
                speed_plan[i] = 0.0

        return speed_plan
