from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seacharts.enc as senc
import yaml

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import colav_simulator.scenario_config as sc
import colav_simulator.viz.visualizer as viz
from colav_simulator.core import stochasticity
from colav_simulator.core.ship import Ship

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.4f}".format})


@dataclass
class Config:
    """Simulation related configuration/parameter class."""

    save_scenario_results: bool = False
    verbose: bool = True
    tracking_from_ownship_only: (
        bool  # Whether to track obstacles from ownship only (True) or all ships track each other (False)
    ) = True
    visualizer: viz.Config = field(default_factory=viz.Config())

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":  # noqa: D102
        config = Config(
            save_scenario_results=config_dict["save_scenario_results"],
            verbose=config_dict["verbose"],
            visualizer=viz.Config.from_dict(config_dict["visualizer"]),
            tracking_from_ownship_only=config_dict["tracking_from_ownship_only"],
        )
        return config

    @classmethod
    def from_file(cls, config_file: Path) -> "Config":  # noqa: D102
        if not config_file.exists():
            msg = f"Configuration file {config_file} does not exist."
            raise FileNotFoundError(msg)
        with open(config_file, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


class Simulator:
    """Class for simulating collision avoidance/maritime vessel scenarios."""

    def __init__(self, config: Config | None = None, config_file: Path | None = dp.simulator_config, **kwargs) -> None:
        """Initializes the simulator.

        Additional key-value arguments can be passed to override the settings in the config file:

        Args:
            config (Optional[Config]): Configuration object.
            config_file (Optional[Path]): Path to configuration file. Defaults to dp.simulator_config.
            kwargs: key-value dictionary of settings to override. This includes:
                scenario_files (List[str]): List of scenario files to run.

        """
        if config is not None:
            self.config: Config = config
        elif config_file is not None:
            self.config = cp.extract(Config, config_file, dp.simulator_schema, **kwargs)
        else:
            raise ValueError("No configuration file or configuration object provided.")

        self.visualizer: viz.Visualizer = viz.Visualizer(self.config.visualizer)
        self.config = self.config

        self.ownship: Ship = None
        self.ship_list: list[Ship] = []
        self.disturbance: stochasticity.Disturbance | None = None
        self.enc: senc.ENC = None
        self.sconfig: sc.ScenarioConfig = None
        self.relevant_grounding_hazards: list = []
        self.relevant_grounding_hazards_as_union = None
        self.recent_sensor_measurements: list = []

        self.t = 0.0
        self.t_start = 0.0
        self.t_end = 0.0
        self.dt = 0.0
        self.timestamp_start = 0

    def toggle_liveplot_visibility(self, visible: bool) -> None:
        """Toggles the visibility of the live plot.

        Args:
            visible (bool): Whether the live plot should be visible or not.
        """
        self.visualizer.toggle_liveplot_visibility(visible)

    def initialize_scenario_episode(
        self,
        ship_list: list[Ship],
        sconfig: sc.ScenarioConfig,
        enc: senc.ENC,
        disturbance: stochasticity.Disturbance | None = None,
        colav_systems: list | None = None,
        trackers: list | None = None,
        seed: int | None = None,
    ) -> None:
        """Initializes the simulation through setting relevant internal state objects.

        Args:
            ship_list (list): 1 x n_ships array of configured Ship objects. Each ship
                is assumed to be properly configured and initialized to its initial state at
                the scenario start (t0).
            sconfig (ScenarioConfig): Scenario episode configuration object.
            enc (senc.ENC): ENC object relevant for the scenario.
            disturbance (stochasticity.Disturbance | None): Disturbance object relevant for the scenario.
            colav_systems (list): List of tuples (ship ID, COLAV system) to use for the selected
                ships involved in the scenario, overrides the existing ones.
            trackers (list): List of tuples (ship ID, tracker system) to use for the selected ships
                involved in the scenario, overrides the existing ones.
            seed (int | None): Seed for the random number generator.
        """
        self.ship_list = None
        self.ship_list = ship_list
        self.sconfig = None
        self.sconfig = sconfig
        self.enc = None
        self.enc = enc
        self.disturbance = None
        self.disturbance = disturbance
        self.ownship = ship_list[0]
        if colav_systems is not None:
            for ship_id, colav_system in colav_systems:
                for _, ship_obj in enumerate(self.ship_list):
                    if ship_obj.id == ship_id:
                        ship_obj.set_colav_system(colav_system)

        if trackers is not None:
            for ship_id, tracker in trackers:
                for _, ship_obj in enumerate(self.ship_list):
                    if ship_obj.id == ship_id:
                        ship_obj.set_tracker(tracker)

        ownship_min_depth = mapf.find_minimum_depth(self.ownship.draft, self.enc)
        self.relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards(ownship_min_depth, self.enc)
        self.relevant_grounding_hazards_as_union = mapf.extract_relevant_grounding_hazards_as_union(
            ownship_min_depth, self.enc
        )

        for ship_obj in self.ship_list:
            ship_obj.reset(seed=seed)

        if self.disturbance is not None:
            self.disturbance.reset(seed=seed)

        self.timestamp_start = mhm.current_utc_timestamp()
        self.t = sconfig.t_start
        self.t_start = sconfig.t_start
        self.t_end = sconfig.t_end
        self.dt = sconfig.dt_sim
        self.recent_sensor_measurements = [None] * len(self.ship_list)

    def run(
        self,
        scenario_data_list: list,
        colav_systems: list | None = None,
        trackers: list | None = None,
        terminate_on_collision_or_grounding: bool = True,
    ) -> list[dict[str, Any]]:
        """Runs through all specified scenarios with their number of episodes.

        Seeds for the random number generator are set and incremented for each scenario episode.

        Args:
            scenario_data_list (list): Premade list of created/configured scenarios. Each entry
                contains a list of ship objects, scenario configuration objects and relevant
                ENC objects.
            colav_systems (list): List of tuples (ship ID, COLAV system) to use for the selected
                ships involved in the scenario, overrides the existing ones.
            trackers (list): List of tuples (ship ID, tracker system) to use for the selected ships
                involved in the scenario, overrides the existing ones.
            terminate_on_collision_or_grounding (bool): Whether to terminate the simulation if a
                collision or grounding occurs.

        Returns:
            List of dictionaries containing the following simulation data for each scenario:
            - episode_simdata_list: List of dictionaries containing the following simulation
                data for each scenario episode:
                - vessel_data: List of data containers containing the vessel simulation data
                    for each ship (used for evaluation).
                - sim_data: List of dictionaries containing the simulated data for each ship.
                - ship_info: Dataframe containing the ship info for each ship.
            - enc: ENC object used in all the scenario episodes.
        """
        if self.config.verbose:
            print("\rSimulator: Started running through scenarios...")

        seed_val = 0

        scenario_simdata_list = []
        for i, (scenario_episode_list, scenario_enc) in enumerate(scenario_data_list):
            scenario_simdata = {}
            if self.config.verbose:
                print(f"\rSimulator: Running scenario nr {i + 1}...")

            episode_simdata_list = []
            for ep, episode_data in enumerate(scenario_episode_list):
                episode_simdata = {}
                ship_list = episode_data["ship_list"]
                episode_disturbance = episode_data["disturbance"]
                episode_config = episode_data["config"]
                scenario_episode_file = episode_config.filename

                self.initialize_scenario_episode(
                    ship_list=ship_list,
                    sconfig=episode_config,
                    enc=scenario_enc,
                    disturbance=episode_disturbance,
                    colav_systems=colav_systems,
                    trackers=trackers,
                    seed=seed_val,
                )

                if self.config.verbose:
                    print(f"\rSimulator: Running scenario episode nr {ep + 1}: {scenario_episode_file}...")
                sim_data, ship_info, sim_times = self.run_scenario_episode(terminate_on_collision_or_grounding)
                if self.config.verbose:
                    print(
                        f"\rSimulator: Finished running through scenario episode nr {ep + 1}: " f"{scenario_episode_file}."
                    )

                self.visualizer.visualize_results(
                    scenario_enc,
                    ship_list,
                    sim_data,
                    sim_times,
                    save_file_path=dp.figure_output / episode_config.name,
                )

                self.visualizer.save_live_plot_animation(dp.animation_output / (episode_config.name + ".gif"))
                self.visualizer.close_live_plot()

                vessel_data = mhm.convert_simulation_data_to_vessel_data(sim_data, ship_info, episode_config.utm_zone)

                episode_simdata["vessel_data"] = vessel_data
                episode_simdata["sim_data"] = sim_data
                episode_simdata["ship_info"] = ship_info
                episode_simdata_list.append(episode_simdata)
                seed_val += 1

            if self.config.verbose:
                print(f"\rSimulator: Finished running through episodes of scenario nr {i + 1}.")

            scenario_simdata["episode_simdata_list"] = episode_simdata_list
            scenario_simdata["enc"] = scenario_enc
            scenario_simdata_list.append(scenario_simdata)

        if self.config.verbose:
            print("\rSimulator: Finished running through scenarios.")
        return scenario_simdata_list

    def is_terminated(
        self,
        verbose: bool = False,
        terminate_on_collision_or_grounding: bool = True,
        prefix_string: str = "",
    ) -> bool:
        """Check whether the current own-ship state is a terminal state.

        Args:
            verbose (bool): Whether to print out the reason for the termination.
            terminate_on_collision_or_grounding (bool): Whether to terminate the simulation if a
                collision or grounding occurs.
            prefix_string (str): Prefix string to add to the printout.

        Returns:
            bool: Whether the current own-ship state is a terminal state
        """
        goal_reached = self.determine_ship_goal_reached(ship_idx=0)
        collided = self.determine_ship_collision(ship_idx=0) and terminate_on_collision_or_grounding
        grounded = self.determine_ship_grounding(ship_idx=0) and terminate_on_collision_or_grounding
        if verbose and collided:
            print(f"{prefix_string}Collision at t = {self.t}!")
        if verbose and grounded:
            print(f"{prefix_string}Grounding at t = {self.t}!")
        if verbose and goal_reached:
            print(f"{prefix_string}Goal reached at t = {self.t}!")
        return collided or grounded or goal_reached

    def is_truncated(self, verbose: bool = False, prefix_string: str = "") -> bool:
        """Check whether the current own-ship state is a truncated state (time limit reached).

        Args:
            verbose (bool): Whether to print out the reason for the truncation.
            prefix_string (str): Prefix string to add to the printout.

        Returns:
            bool: Whether the current own-ship state is a truncated state
        """
        truncated = self.t > self.t_end
        if verbose and truncated:
            print(f"{prefix_string}Time limit reached!")
        return truncated

    def run_scenario_episode(
        self, terminate_on_collision_or_grounding: bool = True
    ) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:  # noqa: PLR0915
        """Runs the simulator for a scenario episode specified by the ship object array.

        Uses a time step dt_sim.

        Args:
            terminate_on_collision_or_grounding (bool): Whether to terminate the simulation if a
                collision or grounding occurs.

        Returns:
            Tuple containing:
            - sim_data: Dataframe containing the ship simulation data for each time step.
            - ship_info: Dictionary containing the ship info for each ship.
            - sim_times: Array containing the simulation times.
        """
        self.visualizer.init_live_plot(self.enc, self.ship_list)

        sim_data = []
        ship_info = {}
        for i, ship_obj in enumerate(self.ship_list):
            ship_info[f"Ship{i}"] = ship_obj.get_ship_info()

        t_end = self.t_end
        while self.t < self.t_end:
            sim_data_dict = self.step()

            sim_data.append(sim_data_dict)

            self.visualizer.update_live_plot(
                self.t,
                self.enc,
                self.ship_list,
                self.recent_sensor_measurements[0],
                self.disturbance.get() if self.disturbance is not None else None,
            )

            terminated = self.is_terminated(
                verbose=True,
                terminate_on_collision_or_grounding=terminate_on_collision_or_grounding,
            )
            truncated = self.is_truncated(verbose=True)
            if terminated or truncated:
                t_end = self.t
                break

        sim_times = np.arange(self.t_start, t_end, self.dt)
        return pd.DataFrame(sim_data), ship_info, sim_times

    def step(self, remote_actor: bool = False) -> dict:
        """Step through the simulation by one time step.

        Args:
            remote_actor (bool): Whether the own-ship is controlled by a remote actor, i.e.
                references are set externally from the Ship object. Used in DRL training.

        Returns:
            Dictionary containing the current time step simulation data for each ship and
            the disturbance data if applicable.
        """
        sim_data_dict = {}

        disturbance_data: stochasticity.DisturbanceData | None = None
        if self.disturbance is not None:
            disturbance_data = self.disturbance.get()
            self.disturbance.update(self.t, self.dt)
            sim_data_dict["currents"] = disturbance_data.currents
            sim_data_dict["wind"] = disturbance_data.wind
            sim_data_dict["waves"] = disturbance_data.waves

        true_do_states = mhm.extract_do_states_from_ship_list(self.t, self.ship_list)
        for i, ship_obj in enumerate(self.ship_list):
            if self.t < ship_obj.t_start:
                sim_data_dict[f"Ship{i}"] = {}
                continue

            tracks, new_measurements = [], []
            if not (i > 0 and self.config.tracking_from_ownship_only):
                relevant_true_do_states = mhm.get_relevant_do_states(true_do_states, i)
                tracks, new_measurements = ship_obj.track_obstacles(self.t, self.dt, relevant_true_do_states)

            self.recent_sensor_measurements[i] = extract_valid_sensor_measurements(
                self.recent_sensor_measurements[i], new_measurements
            )

            if not (i == 0 and remote_actor):  # Skip own-ship planning step if controlled by remote actor
                ship_obj.plan(t=self.t, dt=self.dt, do_list=tracks, enc=self.enc, w=disturbance_data)

            if i > 0 and self.determine_ship_grounding(i):  # Make grounded obstacle ships stop
                ship_obj.set_references(np.zeros((9, 1)))

            sim_data_dict[f"Ship{i}"] = ship_obj.get_sim_data(self.t, self.timestamp_start)
            sim_data_dict[f"Ship{i}"]["sensor_measurements"] = self.recent_sensor_measurements[i]
            sim_data_dict[f"Ship{i}"]["colav"] = ship_obj.get_colav_data()

            ship_obj.forward(self.dt, disturbance_data)

        self.t += self.dt
        return sim_data_dict

    def distance_to_nearby_vessels(self, ship_idx: int = 0) -> np.ndarray:
        """Calculates the distance to nearby vessels for a ship.

        Args:
            ship_idx (int, optional): Index of the ship to calculate the distance to nearby vessels for. Defaults to 0.

        Returns:
            np.ndarray: Array containing the distances to nearby vessels.
        """
        ship_state = self.ship_list[ship_idx].csog_state
        distances = []
        for i, other_ship_obj in enumerate(self.ship_list):
            if i == ship_idx:
                continue
            if other_ship_obj.t_start <= self.t:
                other_ship_state = other_ship_obj.csog_state
                distances.append(np.linalg.norm(ship_state[:2] - other_ship_state[:2]))
            else:
                distances.append(1e12)
        if len(distances) != len(self.ship_list) - 1:
            msg = f"Expected {len(self.ship_list) - 1} distances, got {len(distances)}"
            raise ValueError(msg)
        return np.array(distances)

    def determine_ship_collision(self, ship_idx: int = 0) -> bool:
        """Determines whether a ship is in a collision state.

        Args:
            ship_idx (int, optional): Index of the ship to check for collision. Defaults to 0.

        Returns:
            bool: True if the ship is in a collision state, False otherwise.
        """
        distances = self.distance_to_nearby_vessels(ship_idx)
        other_ship_list = [other_ship_obj for i, other_ship_obj in enumerate(self.ship_list) if i != ship_idx]
        for i, _other_ship_obj in enumerate(other_ship_list):
            if distances[i] <= self.ship_list[ship_idx].length / 2.0:
                return True
        return False

    def distance_to_grounding(self, ship_idx: int = 0) -> float:
        """Calculates the distance to grounding for a ship.

        Args:
            ship_idx (int, optional): Index of the ship to calculate the distance to grounding for. Defaults to 0.

        Returns:
            float: Distance to grounding for the ship.
        """
        ship_state = self.ship_list[ship_idx].csog_state
        d2land = mapf.min_distance_to_hazards(self.relevant_grounding_hazards, ship_state[1], ship_state[0])
        return d2land

    def determine_ship_grounding(self, ship_idx: int = 0) -> bool:
        """Determines whether a ship is in a grounding state.

        Args:
            ship_idx (int, optional): Index of the ship to check for grounding. Defaults to 0.

        Returns:
            bool: True if the ship is in a grounding state, False otherwise.
        """
        d2grounding = self.distance_to_grounding(ship_idx)
        return d2grounding <= self.ship_list[ship_idx].length / 2.0

    def determine_ship_goal_reached(self, ship_idx: int = 0, radius: float | None = None) -> bool:
        """Determines whether the ship has reached its goal.

        Args:
            ship_idx (int, optional): Index of the ship to check for goal reached.
            radius (Optional[float]): Radius around the goal to consider the goal reached.

        Returns:
            bool: True if the own-ship has reached its goal, False otherwise.
        """
        if self.ship_list[ship_idx].goal_csog_state.size > 0:
            goal_state = self.ship_list[ship_idx].goal_csog_state
        elif self.ship_list[ship_idx].waypoints.size > 1:
            goal_state = self.ownship.waypoints[:, -1]
        else:
            return False
        ship_state = self.ship_list[ship_idx].csog_state
        d2goal = np.linalg.norm(ship_state[:2] - goal_state[:2])
        if radius is not None:
            return d2goal <= radius

        scale_factor = 7.0
        return d2goal <= self.ship_list[ship_idx].length * scale_factor


def extract_valid_sensor_measurements(
    recent_sensor_measurements: list[tuple[int, np.ndarray]],
    new_sensor_measurements: list[tuple[int, np.ndarray]],
) -> list:
    """Extracts non-NaN sensor measurements from the recent sensor measurements list.

    Appends them to the most recent sensor measurements list.

    Args:
        recent_sensor_measurements: List of most recent valid (non-nan) sensor measurements
            for the current ship.
        new_sensor_measurements: List of new sensor measurements for the current ship.

    Returns:
        List of updated most recent valid (non-nan) sensor measurements for the current ship.
    """
    for j, sensor_j_measurements in enumerate(new_sensor_measurements):
        if not sensor_j_measurements:
            continue
        valid_meas = []
        for do_idx, do_meas in sensor_j_measurements:
            if not np.isnan(do_meas).any():
                valid_meas.append((do_idx, do_meas))
        if valid_meas:
            if not recent_sensor_measurements:
                recent_sensor_measurements = [None] * len(new_sensor_measurements)
            recent_sensor_measurements[j] = valid_meas
    return recent_sensor_measurements
