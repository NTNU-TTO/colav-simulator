"""
scenario_generator.py

Summary:
    Contains functionality for loading existing scenario definitions,
    and also a ScenarioGenerator class for generating new scenarios. Functionality
    for saving these new scenarios also exists.

Author: Trym Tengesdal, Joachim Miller, Melih Akdag
"""

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import seacharts.enc as senc
import shapely.geometry as geometry
import yaml

import colav_simulator.behavior_generator as bg
import colav_simulator.common.config_parsing as cp
import colav_simulator.common.file_utils as fu
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import colav_simulator.common.plotters as plotters
import colav_simulator.core.ship as ship
import colav_simulator.core.stochasticity as stoch
import colav_simulator.scenario_config as sc

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.2f}".format})


@dataclass
class Config:
    """Configuration class for managing all parameters/settings related to the creation of scenarios.
    All angle ranges are in degrees, and all distances are in meters.
    """

    verbose: bool = False
    manual_episode_accept: bool = False  # Whether or not the user has to accept each generated episode of a scenario.
    behavior_generator: bg.Config = field(default_factory=lambda: bg.Config())

    ho_bearing_range: list = field(
        default_factory=lambda: [-20.0, 20.0]
    )  # Range of [min, max] bearing from the own-ship to the target ship for head-on scenarios
    ho_course_range: list = field(
        default_factory=lambda: [-15.0, 15.0]
    )  # Range of [min, max] course variations of the target ship relative to completely reciprocal head-on scenarios
    ot_bearing_range: list = field(
        default_factory=lambda: [-20.0, 20.0]
    )  # Range of [min, max] bearing from the own-ship to the target ship for overtaking scenarios
    ot_course_range: list = field(
        default_factory=lambda: [-15.0, 15.0]
    )  # Range of [min, max] course variations of the target ship relative to completely parallel overtaking scenarios
    cr_bearing_range: list = field(
        default_factory=lambda: [15.1, 112.5]
    )  # Range of [min, max] bearing from the own-ship to the target ship for crossing scenarios
    cr_course_range: list = field(
        default_factory=lambda: [-15.0, 15.0]
    )  # Range of [min, max] course variations of the target ship relative to completely orthogonal crossing scenarios
    dist_between_ships_range: list = field(
        default_factory=lambda: [200, 10000]
    )  # Range of [min, max] distance variations possible between ships.
    gaussian_csog_state_perturbation_covariance: np.ndarray = field(
        default_factory=lambda: np.diag([25.0, 25.0, 0.5, 3.0])
    )
    perpendicular_csog_state_perturbation_pm_range: list = field(
        default_factory=lambda: [100.0, 0.5, 15.0]
    )  # +- uniform ranges in [distance, speed, course (in deg)]
    t_cpa_threshold: float = (
        200.0  # Threshold for the maximum time to CPA for vessel pairs in a scenario
    )
    d_cpa_threshold: float = 100.0  # Threshold for the maximum distance to CPA for vessel pairs in a scenario
    scenario_files: Optional[list] = (
        None  # Default list of scenario files to load from.
    )
    scenario_folder: Optional[str] = None  # Default scenario folder to load from.

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(
            verbose=config_dict["verbose"],
            manual_episode_accept=config_dict["manual_episode_accept"],
            behavior_generator=bg.Config.from_dict(config_dict["behavior_generator"]),
            ho_bearing_range=config_dict["ho_bearing_range"],
            ho_course_range=config_dict["ho_course_range"],
            ot_bearing_range=config_dict["ot_bearing_range"],
            ot_course_range=config_dict["ot_course_range"],
            cr_bearing_range=config_dict["cr_bearing_range"],
            cr_course_range=config_dict["cr_course_range"],
            dist_between_ships_range=config_dict["dist_between_ships_range"],
            t_cpa_threshold=config_dict["t_cpa_threshold"],
            d_cpa_threshold=config_dict["d_cpa_threshold"],
            gaussian_csog_state_perturbation_covariance=np.diag(
                config_dict["gaussian_csog_state_perturbation_covariance"]
            ),
            perpendicular_csog_state_perturbation_pm_range=config_dict[
                "perpendicular_csog_state_perturbation_pm_range"
            ],
        )
        config.gaussian_csog_state_perturbation_covariance[3, 3] = np.deg2rad(
            config.gaussian_csog_state_perturbation_covariance[3, 3]
        )
        config.perpendicular_csog_state_perturbation_pm_range[2] = np.deg2rad(
            config.perpendicular_csog_state_perturbation_pm_range[2]
        )

        if "scenario_files" in config_dict:
            config.scenario_files = config_dict["scenario_files"]

        if "scenario_folder" in config_dict:
            config.scenario_folder = config_dict["scenario_folder"]
            config.scenario_files = None

        config.behavior_generator = bg.Config.from_dict(
            config_dict["behavior_generator"]
        )
        return config

    def to_dict(self):
        output = asdict(self)
        output["behavior_generator"] = self.behavior_generator.to_dict()
        output["gaussian_csog_state_perturbation_covariance"] = (
            self.gaussian_csog_state_perturbation_covariance.diagonal().tolist()
        )
        output["gaussian_csog_state_perturbation_covariance"][3] = float(
            np.rad2deg(output["gaussian_csog_state_perturbation_covariance"][3])
        )
        output["perpendicular_csog_state_perturbation_pm_range"][2] = float(
            np.rad2deg(output["perpendicular_csog_state_perturbation_pm_range"][2])
        )
        return output

    @classmethod
    def from_file(cls, config_file: Path):
        assert config_file.exists(), f"Configuration file {config_file} does not exist."
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


class ScenarioGenerator:
    """Class for generating maritime traffic scenarios in a given geographical environment."""

    rng: np.random.Generator
    enc: senc.ENC
    behavior_generator: bg.BehaviorGenerator

    def __init__(
        self,
        config: Optional[Config] = None,
        config_file: Optional[Path] = dp.scenario_generator_config,
        enc_config_file: Optional[Path] = dp.seacharts_config,
        init_enc: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Constructor for the ScenarioGenerator.

        Args:
            - config (Config): Configuration object containing all parameters/settings related to the creation of scenarios.
            - config_file (Path, optional): Absolute path to the generator config file. Defaults to dp.scenario_generator_config.
            - enc_config_file (Path, optional): Absolute path to the ENC config file. Defaults to dp.seacharts_config.
            - init_enc (bool, optional): Flag determining whether or not to initialize the ENC object.
            - seed (Optional[int], optional): Integer seed.
            - **kwargs: Keyword arguments for the ScenarioGenerator, can be e.g.:
                    new_data (bool): Flag determining whether or not to read ENC data from shapefiles again.
        """
        self._config: Config = Config()
        if config:
            self._config = config
        elif config_file:
            self._config = cp.extract(Config, config_file, dp.scenario_generator_schema)

        self.safe_sea_cdt: Optional[list] = None
        self.safe_sea_cdt_weights: Optional[list] = None

        if init_enc:
            self.enc = senc.ENC(config_file=enc_config_file, **kwargs)
            self._setup_cdt(show_plots=False)

        self.behavior_generator = bg.BehaviorGenerator(self._config.behavior_generator)
        self.seed(seed)

        self._disturbance_handles: list = []
        self._episode_counter: int = 0
        self._uniform_os_state_update_indices: list = []
        self._os_state_update_indices: list = []
        self._os_plan_update_indices: list = []
        self._do_state_update_indices: list = []
        self._do_plan_update_indices: list = []
        self._disturbance_update_indices: list = []
        self._ep0: int = 0
        self._bad_episode: bool = False
        self._sg_hazards: list = []

        self._prev_disturbance: Optional[stoch.Disturbance] = None
        self._prev_ship_list: list = []
        self._first_csog_states: list = []
        self._ownship_position_generation: sc.OwnshipPositionGenerationMethod = (
            sc.OwnshipPositionGenerationMethod.UniformInTheMapThenGaussian
        )  # set by scenario config
        self._target_position_generation: sc.TargetPositionGenerationMethod = (
            sc.TargetPositionGenerationMethod.BasedOnOwnshipPositionThenGaussian
        )

    def seed(self, seed: Optional[int] = None) -> None:
        """Seeds the random number generator.

        Args:
            seed (Optional[int]): Integer seed.
        """
        self.rng = np.random.default_rng(seed=seed)
        self.behavior_generator.seed(seed=seed)

    def _setup_cdt(self, vessel_min_depth: int = 5, show_plots: bool = False) -> None:
        """Sets up the constrained Delaunay triangulation for the ENC map, for a vessel minimum depth.

        Args:
            show_plots (bool, optional): Wether to show cdt plots or not.
        """
        self.safe_sea_cdt = mapf.create_safe_sea_triangulation(
            self.enc, vessel_min_depth=vessel_min_depth, show_plots=show_plots
        )
        self.safe_sea_cdt_weights = mhm.compute_triangulation_weights(self.safe_sea_cdt)

    def _configure_enc(self, scenario_config: sc.ScenarioConfig) -> senc.ENC:
        """Configures the ENC object based on the scenario config file.

        Args:
            - scenario_config (sc.ScenarioConfig): Scenario config object.

        Returns:
            - (senc.ENC): Configured ENC object.
        """
        self.enc = senc.ENC(
            config_file=dp.seacharts_config,
            utm_zone=scenario_config.utm_zone,
            size=scenario_config.map_size,
            origin=scenario_config.map_origin_enu,
            files=scenario_config.map_data_files,
            new_data=scenario_config.new_load_of_map_data,
            tolerance=scenario_config.map_tolerance,
            buffer=scenario_config.map_buffer,
            figname=scenario_config.name,
        )

        return copy.deepcopy(self.enc)

    def determine_indices_of_episode_parameter_updates(
        self, config: sc.ScenarioConfig
    ) -> None:
        """Determines the episode indices when the OS plan+state, DO state, DO plan and disturbance should be updated/re-randomized.

        Args:
            config (sc.ScenarioConfig): Scenario config object.

        """
        n_episodes = config.episode_generation.n_episodes
        n_constant_os_state_episodes = (
            config.episode_generation.n_constant_os_state_episodes
        )
        n_constant_os_plan_episodes = (
            config.episode_generation.n_constant_os_plan_episodes
        )
        n_constant_do_state_episodes = (
            config.episode_generation.n_constant_do_state_episodes
        )
        n_plans_per_do_state = config.episode_generation.n_plans_per_do_state
        n_constant_do_plans = int(
            np.ceil(n_constant_do_state_episodes / n_plans_per_do_state)
        )
        n_constant_disturbance_episodes = (
            config.episode_generation.n_constant_disturbance_episodes
        )
        delta_uniform_position_sample = (
            config.episode_generation.delta_uniform_position_sample
        )
        self._os_state_update_indices = [-1 for _ in range(n_episodes)]
        self._os_plan_update_indices = [-1 for _ in range(n_episodes)]
        self._do_state_update_indices = [-1 for _ in range(n_episodes)]
        self._do_plan_update_indices = [-1 for _ in range(n_episodes)]
        self._disturbance_update_indices = [-1 for _ in range(n_episodes)]
        self._uniform_os_state_update_indices = [-1 for _ in range(n_episodes)]
        for ep in range(n_episodes):
            if ep % delta_uniform_position_sample == 0:
                self._uniform_os_state_update_indices[ep] = ep

            if ep % n_constant_os_state_episodes == 0:
                self._os_state_update_indices[ep] = ep

            if ep % n_constant_os_plan_episodes == 0:
                self._os_plan_update_indices[ep] = ep

            if ep % n_constant_disturbance_episodes == 0:
                self._disturbance_update_indices[ep] = ep

            if ep % (n_plans_per_do_state * n_constant_do_state_episodes) == 0:
                self._do_state_update_indices[ep] = ep

            if ep % n_constant_do_plans == 0:
                self._do_plan_update_indices[ep] = ep

    def create_file_path_list_from_config(self) -> list:
        """Creates a list of file paths from the config file scenario files or scenario folder.

        Returns:
            list: List of valid file paths.
        """
        if self._config.scenario_files is not None:
            return [dp.scenarios / f for f in self._config.scenario_files]
        else:
            scenario_folder = dp.scenarios / self._config.scenario_folder
            files = [file for file in scenario_folder.iterdir()]
            files.sort()
            return files

    def generate_configured_scenarios(self) -> list:
        """Generates the list of configured scenarios from the class config file.

        Returns:
            list: List of fully configured scenario data definitions.
        """
        files = self.create_file_path_list_from_config()
        scenario_data_list = self.generate_scenarios_from_files(files)
        return scenario_data_list

    def load_scenario_from_folders(
        self,
        folder: Path | List[Path],
        scenario_name: str,
        reload_map: bool = True,
        show: bool = False,
        max_number_of_episodes: Optional[int] = None,
        shuffle_episodes: bool = False,
        merge_scenario_episodes: bool = True,
    ) -> Tuple[list, senc.ENC] | List[Tuple[list, senc.ENC]]:
        """Loads all episode files for the input scenario(s) from folder(s) that match the specified scenario_name(s).

        Args:
            - folder (Path | List[Path]): Path to folder(s) containing scenario files.
            - scenario_name (str | List[str]): Name(s) of the scenario(s). In case of multiple folders, the scenario names should be in the same order as the folders and of the same length.
            - reload_map (bool, optional): Flag determining whether or not to reload the map data. Defaults to True.
            - show (bool, optional): Flag determining whether or not to show the episode setups through seacharts.
            - max_number_of_episodes (Optional[int], optional): Maximum number of episodes to load.
            - shuffle_episodes (bool, optional): Flag determining whether or not to shuffle the episode list.
            - merge_scenario_episodes (bool, optional): Flag determining whether or not to merge the scenario episodes into a single list, and return a single ENC object. This only works if the same map is used for all scenarios.

        Returns:
            - Tuple[list, senc.ENC] | List[Tuple[list, senc.ENC]]: List of scenario files and the corresponding ENC object in case of single folder, or a list of these for multiple folders.
        """
        folder_list = [folder] if isinstance(folder, Path) else folder
        scenario_name_list = (
            [scenario_name] if isinstance(scenario_name, str) else scenario_name
        )
        assert len(folder_list) == len(
            scenario_name_list
        ), "Number of folders and scenario names should match and be the same."

        scenario_data_list = []
        generate_map = True
        enc = None
        for i, folder in enumerate(folder_list):
            if not merge_scenario_episodes:
                generate_map = True
            scenario_episode_list = []
            sname = scenario_name_list[i]
            file_list = [file for file in folder.iterdir()]
            file_list.sort(key=lambda x: x.name.split("_")[-3])
            for file_idx, file in enumerate(file_list):
                if not (sname in file.name and file.suffix == ".yaml"):
                    continue

                if self._config.verbose:
                    print(f"ScenarioGenerator: Loading scenario file: {file.name}...")
                ship_list, disturbance, config = self.load_episode(config_file=file)
                if generate_map:
                    generate_map = False
                    config.new_load_of_map_data = reload_map
                    enc = self._configure_enc(config)
                else:
                    config.new_load_of_map_data = False

                scenario_episode_list.append(
                    {
                        "ship_list": ship_list,
                        "disturbance": disturbance,
                        "config": config,
                    }
                )

                if show:
                    self.visualize_episode(ship_list, disturbance, enc, config)

                self._episode_counter += 1

                if (
                    max_number_of_episodes is not None
                    and file_idx >= max_number_of_episodes - 1
                ):
                    break

            if self._config.verbose:
                print(
                    f"ScenarioGenerator: Finished loading scenario episode files for {sname}."
                )

            if show:
                input("Press enter to continue...")
                self._clear_disturbance_handles()
                enc.close_display()

            if shuffle_episodes:
                self.rng.shuffle(scenario_episode_list)

            if len(folder_list) == 1:
                return (scenario_episode_list, enc)

            scenario_data_list.append((scenario_episode_list, enc))

        if merge_scenario_episodes:
            merged_episode_list = [
                episode for scenario in scenario_data_list for episode in scenario[0]
            ]
            if shuffle_episodes:
                self.rng.shuffle(merged_episode_list)
            enc = scenario_data_list[0][1]
            return (merged_episode_list, enc)

        return scenario_data_list

    def _clear_disturbance_handles(self) -> None:
        if self._disturbance_handles:
            for handle in self._disturbance_handles:
                handle.remove()
            self._disturbance_handles = []

    def visualize_disturbance(
        self, disturbance: stoch.Disturbance | None, enc: senc.ENC
    ) -> None:
        """Visualizes the disturbance object.

        Args:
            disturbance (stoch.Disturbance | None): Disturbance object.
            enc (senc.ENC): ENC object to visualize on.
        """
        if disturbance is None:
            return

        ddata = disturbance.get()
        self._clear_disturbance_handles()

        handles = []
        if ddata.currents is not None and ddata.currents["speed"] > 0.0:
            speed = ddata.currents["speed"]
            handles.extend(
                plotters.plot_disturbance(
                    magnitude=70.0,
                    direction=ddata.currents["direction"],
                    name=f"current: {speed:.2f} m/s",
                    enc=enc,
                    color="white",
                    linewidth=1.0,
                    location="topright",
                    text_location_offset=(0.0, 0.0),
                )
            )

        if ddata.wind is not None and ddata.wind["speed"] > 0.0:
            speed = ddata.wind["speed"]
            handles.extend(
                plotters.plot_disturbance(
                    magnitude=70.0,
                    direction=ddata.wind["direction"],
                    name=f"wind: {speed:.2f} m/s",
                    enc=enc,
                    color="peru",
                    linewidth=1.0,
                    location="topright",
                    text_location_offset=(0.0, -20.0),
                )
            )
        self._disturbance_handles = handles

    def visualize_episode(
        self,
        ship_list: list,
        disturbance: stoch.Disturbance | None,
        enc: senc.ENC,
        config: sc.ScenarioConfig,
    ) -> None:
        """Visualizes a fully defined scenario episode.

        Args:
            ship_list (list): List of ships in the scenario with initialized poses and plans.
            disturbance (stoch.Disturbance | None): Disturbance object for the episode.
            enc (senc.ENC): ENC object.
            config (sc.ScenarioConfig): Scenario config object.
        """
        enc.start_display()
        for ship_obj in reversed(ship_list):
            ship_color = "magenta" if ship_obj.id == 0 else "red"
            plan_color = "purple" if ship_obj.id == 0 else "orangered"
            if ship_obj.waypoints.size > 0:
                plotters.plot_waypoints(
                    ship_obj.waypoints,
                    enc,
                    color=plan_color,
                    point_buffer=2.0,
                    disk_buffer=6.0,
                    hole_buffer=2.0,
                )
            if ship_obj.trajectory.size > 0:
                plotters.plot_trajectory(
                    ship_obj.trajectory, enc, color=plan_color, linewidth=1.0
                )

            ship_poly = mapf.create_ship_polygon(
                ship_obj.csog_state[0],
                ship_obj.csog_state[1],
                mf.wrap_angle_to_pmpi(ship_obj.csog_state[3]),
                ship_obj.length,
                ship_obj.width,
                5.0,
                5.0,
            )
            enc.draw_polygon(ship_poly, color=ship_color, fill=True, alpha=0.6)

        self.visualize_disturbance(disturbance, enc)

    def load_episode(
        self, config_file: Path
    ) -> Tuple[list, stoch.Disturbance, sc.ScenarioConfig]:
        """Loads a fully defined scenario episode from configuration file.

        NOTE: The file must have a ship list with fully specified ship configurations,
        and a corresponding correct number of random ships (excluded the own-ship with ID 0).

        NOTE: The scenario ENC object is not initialized here, but in the `load_scenario_from_folder` function.

        Args:
            - config_file (Path): Absolute path to the scenario config file.

        Returns:
            - Tuple[list, sc.ScenarioConfig]: List of ships in the scenario with initialized poses and plans, the final scenario config object.
        """
        config = cp.extract(sc.ScenarioConfig, config_file, dp.scenario_schema)
        ship_list = []
        disturbance = (
            stoch.Disturbance(config.stochasticity)
            if config.stochasticity is not None
            else None
        )
        for ship_cfg in config.ship_list:
            assert (
                ship_cfg.csog_state.size > 0
                and (
                    (ship_cfg.waypoints.size > 0 and ship_cfg.speed_plan.size > 0)
                    or ship_cfg.goal_csog_state.size > 0
                )
                and ship_cfg.id >= 0
            ), "A fully specified ship config has an id, initial csog_state, waypoints + speed_plan or goal state."
            ship_obj = ship.Ship(
                mmsi=ship_cfg.mmsi, identifier=ship_cfg.id, config=ship_cfg
            )
            ship_list.append(ship_obj)

        return ship_list, disturbance, config

    def generate_scenarios_from_files(self, files: list) -> list:
        """Generates scenarios from each of the input file paths.

        Args:
            - files (list): List of configuration files to generate scenarios from, as Path objects.

        Returns:
            - list: List of episode config data dictionaries and relevant ENC objects, for each scenario.
        """
        scenario_data_list = []
        for i, scenario_file in enumerate(files):
            if self._config.verbose:
                print(
                    f"\rScenario generator: Creating scenario nr {i + 1}: {scenario_file.name}..."
                )
            scenario_episode_list, enc = self.generate(config_file=scenario_file)
            if self._config.verbose:
                print(
                    f"\rScenario generator: Finished creating scenario nr {i + 1}: {scenario_file.name}."
                )
            scenario_data_list.append((scenario_episode_list, enc))
        return scenario_data_list

    def generate(
        self,
        config: Optional[sc.ScenarioConfig] = None,
        config_file: Optional[Path] = None,
        enc: Optional[senc.ENC] = None,
        new_load_of_map_data: Optional[bool] = None,
        show_plots: Optional[bool] = False,
        save_scenario: Optional[bool] = False,
        save_scenario_folder: Optional[Path] = dp.scenarios,
        delete_existing_files: Optional[bool] = False,
        n_episodes: Optional[int] = None,
        episode_idx_save_offset: Optional[int] = 0,
    ) -> Tuple[list, senc.ENC]:
        """Main class function. Creates a maritime scenario, with a number of `n_episodes` based on the input config or config file.

        If specified, the ENC object provides the geographical environment.

        Args:
            - config (sc.ScenarioConfig, optional): Scenario config object.
            - config_file (Path, optional): Absolute path to the scenario config file.
            - enc (ENC, optional): Electronic Navigational Chart object containing the geographical environment.
            - new_load_of_map_data (bool, optional): Flag determining whether or not to read ENC data from shapefiles again.
            - show_plots (bool, optional): Flag determining whether or not to show seacharts debugging plots.
            - save_scenario (bool, optional): Flag determining whether or not to save the scenario definition.
            - save_scenario_folder (Path, optional): Absolute path to the folder where the scenario definition should be saved. Defaults to dp.scenarios.
            - delete_existing_files (bool, optional): Flag determining whether or not to delete existing files at the "save_scenario_folder" path.
            - n_episodes (int, optional): Number of episodes to generate.
            - episode_idx_save_offset (int, optional): Offset for the episode index when saving to file. Defaults to 0.

        Returns:
            - Tuple[list, ENC]: List of scenario episodes, each containing a dictionary of episode information. Also, the corresponding ENC object is returned.
        """
        if save_scenario_folder is not None:
            if not save_scenario_folder.exists():
                save_scenario_folder.mkdir(parents=True, exist_ok=True)
            if delete_existing_files:
                fu.delete_files_in_folder(save_scenario_folder)

        if config is None and config_file is not None:
            config = cp.extract(sc.ScenarioConfig, config_file, dp.scenario_schema)
            config.filename = config_file.name

        if config is None and config_file is None:
            config = cp.extract(
                sc.ScenarioConfig,
                self.create_file_path_list_from_config()[0],
                dp.scenario_schema,
            )

        assert config is not None, "Config should not be none here."
        self._episode_counter = 0
        self._ep0 = episode_idx_save_offset
        show_plots = True if self._config.manual_episode_accept else show_plots
        save_scenario = (
            save_scenario if save_scenario is not None else config.save_scenario
        )
        ais_vessel_data_list = []
        mmsi_list = []
        ais_data_output = sc.process_ais_data(config)
        if ais_data_output:
            ais_vessel_data_list = ais_data_output["vessels"]
            mmsi_list = ais_data_output["mmsi_list"]
            config.map_origin_enu = ais_data_output["map_origin_enu"]
            config.map_size = ais_data_output["map_size"]
        config.map_origin_enu, config.map_size = sc.find_global_map_origin_and_size(
            config
        )
        if new_load_of_map_data is not None:
            config.new_load_of_map_data = new_load_of_map_data

        if enc is not None:
            self.enc = enc
            enc_copy = copy.deepcopy(enc)
        else:
            enc_copy = self._configure_enc(config)
        self._setup_cdt(show_plots=False)
        self._sg_hazards = mapf.extract_relevant_grounding_hazards_as_union(
            vessel_min_depth=1, enc=self.enc
        )

        n_episodes = (
            config.episode_generation.n_episodes if n_episodes is None else n_episodes
        )
        config.episode_generation.n_episodes = n_episodes

        if config.n_random_ships is not None:
            n_random_ships_list = [config.n_random_ships for _ in range(n_episodes)]
        elif config.n_random_ships_range is not None:
            n_random_ships_list = [
                int(
                    self.rng.integers(
                        config.n_random_ships_range[0],
                        config.n_random_ships_range[1],
                        endpoint=True,
                    )
                )
                for _ in range(n_episodes)
            ]
        else:
            n_random_ships_list = [0 for _ in range(n_episodes)]
        max_number_of_ships = max(n_random_ships_list) + 1  # +1 for own-ship

        self.behavior_generator.initialize_data_structures(max_number_of_ships)
        self.behavior_generator.setup_enc(
            self.enc,
            self.safe_sea_cdt,
            self.safe_sea_cdt_weights,
            show_plots=show_plots,
        )
        self.determine_indices_of_episode_parameter_updates(config)
        self._ownship_position_generation = (
            config.episode_generation.ownship_position_generation
        )
        self._target_position_generation = (
            config.episode_generation.target_position_generation
        )

        scenario_episode_list = []
        if show_plots:
            self.enc.start_display()

        self._prev_ship_list = [None for _ in range(max_number_of_ships)]
        self._first_csog_states = [None for _ in range(max_number_of_ships)]
        for ep in range(n_episodes):
            n_random_ships = n_random_ships_list[ep]
            config_copy = copy.deepcopy(config)
            config_copy.n_random_ships = n_random_ships

            ship_list, config_copy = self._create_partially_defined_ships(config_copy)

            episode = {}
            episode["ship_list"], episode["disturbance"], episode["config"] = (
                self.generate_episode(
                    copy.deepcopy(ship_list),
                    config_copy,
                    ais_vessel_data_list,
                    mmsi_list,
                    show_plots=False,  # set to true for debugging
                )
            )
            if (
                self._bad_episode and n_episodes > 1
            ):  # See the check_for_bad_episode method for more information
                print("ScenarioGenerator: Bad episode detected. Skipping episode.")
                continue

            if self._config.manual_episode_accept:
                print("ScenarioGenerator: Accept episode? (y/n)")
                answer = input()  # "y"
                if answer not in ["y", "Y", "yes", "Yes"]:
                    if ep < n_episodes - 1:
                        self._uniform_os_state_update_indices[ep + 1] = ep + 1
                        self._os_plan_update_indices[ep + 1] = ep + 1
                        self._os_state_update_indices[ep + 1] = ep + 1
                        self._do_plan_update_indices[ep + 1] = ep + 1
                    continue

            if show_plots:
                self.visualize_episode(
                    episode["ship_list"],
                    episode["disturbance"],
                    self.enc,
                    episode["config"],
                )

            self._episode_counter += 1
            ep_str = str(self._episode_counter + self._ep0).zfill(3)
            episode["config"].name = f"{config.name}_ep{ep_str}"
            episode["config"].n_random_ships = len(episode["ship_list"]) - 1
            if save_scenario:
                episode["config"].filename = sc.save_scenario_episode_definition(
                    episode["config"], save_scenario_folder
                )

            if self._config.verbose:
                print(
                    f"ScenarioGenerator: Episode {self._episode_counter} of {n_episodes} created. Num target ships: "
                    + str(episode["config"].n_random_ships)
                )

            scenario_episode_list.append(episode)

        if show_plots:
            input(
                "Press enter to continue. Will take a while to load plots if you generated 500+ episodes with visualization on..."
            )
            self._clear_disturbance_handles()
            self.enc.close_display()
        if self._config.verbose:
            print(
                f"ScenarioGenerator: Number of accepted episodes: {self._episode_counter} out of {n_episodes}."
            )

        if self._episode_counter == 0:
            print(
                "WARNING: No episodes were generated. Check the scenario configuration or try different seed and increase the number of episodes."
            )
        return scenario_episode_list, enc_copy

    def _create_partially_defined_ships(
        self, config: sc.ScenarioConfig
    ) -> Tuple[list, sc.ScenarioConfig]:
        """Creates partially defined ship objects and ship configurations for all ships.

        Args:
            config (sc.ScenarioConfig): Scenario config object.

        Returns:
            Tuple[list, sc.ScenarioConfig]: Partially defined list of ships to be considered in simulation, and the updated scenario config object.
        """
        ship_list = []
        ship_config_list = []
        n_cfg_ships = len(config.ship_list)
        for s in range(1 + config.n_random_ships):  # +1 for own-ship
            if s < n_cfg_ships and s == config.ship_list[s].id:
                ship_config = config.ship_list[s]
            else:
                ship_config = ship.Config()
                ship_config.id = s
                ship_config.mmsi = s + 1
            ship_obj = ship.Ship(
                mmsi=ship_config.mmsi, identifier=ship_config.id, config=ship_config
            )
            ship_list.append(ship_obj)
            ship_config_list.append(ship_config)
        config.ship_list = ship_config_list
        return ship_list, config

    def generate_episode(
        self,
        ship_list: list,
        config: sc.ScenarioConfig,
        ais_vessel_data_list: Optional[list],
        mmsi_list: Optional[list],
        show_plots: Optional[bool] = True,
    ) -> Tuple[list, Optional[stoch.Disturbance], sc.ScenarioConfig]:
        """Creates a single maritime scenario episode.

                Some ships in the episode can be partially or fully specified by the AIS ship data, if not none.

                Random plans for each ship will be created unless specified in ship_list entries or loaded from AIS data.
        >
                Args:
                    - ship_list (list): List of ships to be considered in simulation.
                    - config (sc.ScenarioConfig): Scenario config object.
                    - ais_vessel_data_list (Optional[list]): Optional list of AIS vessel data objects.
                    - mmsi_list (Optional[list]): Optional list of corresponding MMSI numbers for the AIS vessels.
                    - show_plots (Optional[bool]): Flag determining whether or not to show seacharts debugging plots.

                Returns:
                    - Tuple[list, Optional[stoch.Disturbance], sc.ScenarioConfig]: List of ships in the scenario with initialized poses and plans, the disturbance object for the episode (if specified) and the final scenario config object.
        """
        ship_replan_flags = self.determine_replanning_flags(ship_list, config)

        ship_list, config = self.transfer_vessel_ais_data(
            ship_list, config, ais_vessel_data_list, mmsi_list
        )

        # Setup and generate own-ship state and behavior first, as this will be used for generating the target ship behavior, depending on the
        # target position generation method used.
        ship_list[0], config, _ = self.generate_ownship_csog_state(ship_list[0], config)

        self.behavior_generator.setup_ship(
            self.rng,
            ship_list[0],
            ship_replan_flags[0],
            config.t_end - config.t_start,
            show_plots=False,
        )
        ship_list[0], config.ship_list[0] = (
            self.behavior_generator.generate_ship_behavior(
                self.rng,
                ship_list[0],
                config.ship_list[0],
                config.t_end - config.t_start,
                reuse_old_behavior=not ship_replan_flags[0],
            )
        )

        # Then generate target ship states and behavior based on the own-ship state and behavior.
        ship_list, config, _ = self.generate_target_ship_csog_states(ship_list, config)
        self.behavior_generator.setup(
            self.rng,
            ship_list,
            ship_replan_flags,
            config.t_end - config.t_start,
            show_plots=show_plots,
        )
        ship_list, config.ship_list = self.behavior_generator.generate(
            self.rng,
            ship_list,
            config.ship_list,
            simulation_timespan=config.t_end - config.t_start,
            show_plots=False,
        )

        ship_list.sort(key=lambda x: x.id)
        config.ship_list.sort(key=lambda x: x.id)

        disturbance = self.generate_disturbance(config)

        self._bad_episode, ship_list, config = self.check_for_bad_episode(
            ship_list, config
        )

        self._prev_ship_list[: len(ship_list)] = copy.deepcopy(ship_list)
        self._prev_ship_list[len(ship_list) :] = [
            None for _ in range(len(ship_list), len(self._prev_ship_list))
        ]

        # self.behavior_generator.visualize_ship_behaviors(ship_list)
        return ship_list, disturbance, config

    def check_for_bad_episode(
        self,
        ship_list: list,
        config: sc.ScenarioConfig,
        minimum_os_plan_length: float = 300.0,
        minimum_do_plan_length: float = 100.0,
        show_plots: bool = False,
    ) -> Tuple[bool, list, sc.ScenarioConfig]:
        """Checks if the episode is bad, i.e. if any of the ships are outside the map,
        the plan is less than the minimum length.

        Args:
            ship_list (list): List of ships to be considered in simulation.
            config (sc.ScenarioConfig): Scenario config object.
            minimum_plan_length (float, optional): Minimum length of the plan. Defaults to 400.0.
            show_plots (bool, optional): Flag determining whether or not to show seacharts debugging plots.

        Returns:
            Tuple[bool, list, sc.ScenarioConfig]: Tuple of boolean determining if the episode is bad, the new ship list and the new scenario config object. If the episode is OK, we may still want to prune the DOs with bad paths (i.e. too short paths).
        """
        if show_plots:
            os_poly_handle = None
            os_traj_handle = None
            do_traj_handle = None
            do_poly_handle = None

        ownship = ship_list[0]
        if ownship.waypoints.size > 1:
            os_simple_traj = mhm.trajectory_from_waypoints_and_speed(
                ownship.waypoints,
                ownship.speed_plan,
                config.dt_sim,
                config.t_end - config.t_start,
            )
        elif ownship.goal_csog_state.size > 0:
            os_simple_traj = mhm.trajectory_from_waypoints_and_speed(
                np.array([ownship.csog_state[0:2], ownship.goal_csog_state[0:2]]).T,
                np.array([ownship.speed, ownship.goal_csog_state[2]]),
                config.dt_sim,
                config.t_end - config.t_start,
            )

        n_do = len(ship_list) - 1
        bad_do_path_indices = []
        bad_episode = False
        for ship_obj in ship_list:
            in_safe_sea = mapf.point_in_polygon_list(
                geometry.Point(ship_obj.csog_state[1], ship_obj.csog_state[0]),
                self.safe_sea_cdt,
            )
            path_length = np.sum(
                np.linalg.norm(np.diff(ship_obj.waypoints, axis=1), axis=0)
            )

            if ship_obj.id == 0 and (
                not in_safe_sea or (path_length < minimum_os_plan_length)
            ):
                bad_episode = True
                break

            if ship_obj.id > 0:
                start_idx_ship = int(np.floor(ship_obj.t_start / config.dt_sim))
                dist_os_to_ship = np.linalg.norm(
                    os_simple_traj[:2, start_idx_ship] - ship_obj.state[:2]
                )
                traj_do = mhm.trajectory_from_waypoints_and_speed(
                    ship_obj.waypoints,
                    ship_obj.speed_plan,
                    config.dt_sim,
                    config.t_end - ship_obj.t_start,
                )
                t_cpa, d_cpa, _ = mhm.compute_actual_vessel_pair_cpa(
                    os_simple_traj[
                        :, start_idx_ship : start_idx_ship + len(traj_do[0, :])
                    ],
                    traj_do,
                    config.dt_sim,
                )
                do_path_crosses_hazards = False
                for wp_idx in range(1, ship_obj.waypoints.shape[1]):
                    do_path_crosses_hazards = (
                        mapf.check_if_segment_crosses_grounding_hazards(
                            enc=self.enc,
                            p1=traj_do[:2, wp_idx - 1],
                            p2=traj_do[:2, wp_idx],
                            draft=ship_obj.draft,
                            hazards=self._sg_hazards,
                        )
                    )
                    if do_path_crosses_hazards:
                        break

                if show_plots:
                    os_poly = mapf.create_ship_polygon(
                        ownship.csog_state[0],
                        ownship.csog_state[1],
                        mf.wrap_angle_to_pmpi(ownship.heading),
                        ownship.length,
                        ownship.width,
                        5.0,
                        5.0,
                    )
                    do_poly = mapf.create_ship_polygon(
                        ship_obj.csog_state[0],
                        ship_obj.csog_state[1],
                        mf.wrap_angle_to_pmpi(ship_obj.heading),
                        ship_obj.length,
                        ship_obj.width,
                        5.0,
                        5.0,
                    )
                    os_traj_handle = plotters.plot_trajectory(
                        os_simple_traj[:, start_idx_ship:],
                        self.enc,
                        color="purple",
                        linewidth=1.0,
                    )
                    os_poly_handle = self.enc.draw_polygon(
                        os_poly, color="magenta", fill=True, alpha=0.6
                    )
                    do_traj_handle = plotters.plot_trajectory(
                        traj_do, self.enc, color="orangered", linewidth=1.0
                    )
                    do_poly_handle = self.enc.draw_polygon(
                        do_poly, color="red", fill=True, alpha=0.6
                    )
                    if os_traj_handle:
                        os_poly_handle.remove()
                        os_traj_handle.remove()
                        do_traj_handle.remove()
                        do_poly_handle.remove()

                if (
                    ship_obj.waypoints.size == 0
                    or not in_safe_sea
                    or path_length < minimum_do_plan_length
                    or dist_os_to_ship < self._config.dist_between_ships_range[0]
                    or t_cpa > self._config.t_cpa_threshold
                    or d_cpa > self._config.d_cpa_threshold
                    or do_path_crosses_hazards
                ):
                    bad_do_path_indices.append(ship_obj.id)
                    continue

        bad_episode = bad_episode or (len(bad_do_path_indices) == n_do and n_do > 0)
        if bad_episode:
            return True, ship_list, config

        # Prune DOs with bad paths
        new_ship_list = [
            ship_obj for ship_obj in ship_list if ship_obj.id not in bad_do_path_indices
        ]
        new_ship_config_list = [
            ship_cfg
            for ship_cfg in config.ship_list
            if ship_cfg.id not in bad_do_path_indices
        ]
        id_counter = 0
        for ship_obj, ship_config in zip(new_ship_list, new_ship_config_list):
            ship_obj.set_id(id_counter)
            ship_config.id = id_counter
            id_counter += 1
        config.ship_list = new_ship_config_list

        return False, new_ship_list, config

    def determine_replanning_flags(
        self, ship_list: list, config: sc.ScenarioConfig
    ) -> list:
        """Determines the flags for whether or not to generate a new plan for each ship.

        Args:
            ship_list (list): List of ships to be considered in simulation.
            config (sc.ScenarioConfig): Scenario config object.

        Returns:
            list: List of booleans determining whether or not to generate a new plan for each ship.
        """
        replan_list = [False for _ in range(len(ship_list))]
        ep = self._episode_counter
        uniform_in_map_sample = (
            ep % config.episode_generation.delta_uniform_position_sample == 0
        )
        for ship_cfg_idx, _ in enumerate(config.ship_list):
            if uniform_in_map_sample:
                replan_list[ship_cfg_idx] = True

            elif ship_cfg_idx == 0 and ep == self._os_plan_update_indices[ep]:
                replan_list[ship_cfg_idx] = True

            elif ship_cfg_idx > 0 and ep == self._do_plan_update_indices[ep]:
                replan_list[ship_cfg_idx] = True
        return replan_list

    def transfer_vessel_ais_data(
        self,
        ship_list: list,
        config: sc.ScenarioConfig,
        ais_vessel_data_list: Optional[list],
        mmsi_list: Optional[list],
    ) -> Tuple[list, sc.ScenarioConfig]:
        """Transfers AIS vessel data to the ship objects and ship configurations, if available.

        Args:
            - ship_list (list): List of ships to be considered in simulation.
            - config (sc.ScenarioConfig): Scenario config object.
            - ais_vessel_data_list (Optional[list]): Optional list of AIS vessel data objects.
            - mmsi_list (Optional[list]): Optional list of corresponding MMSI numbers for the AIS vessels.

        Returns:
            - Tuple[list, sc.ScenarioConfig]: List of partially initialized ships in the scenario, and the corresponding updated scenario config object.
        """
        if not (ais_vessel_data_list or mmsi_list):
            return ship_list, config

        for ship_cfg_idx, ship_config in enumerate(config.ship_list):
            use_ais_ship_trajectory = True

            # The own-ship (with index 0) will not use the predefined AIS trajectory, but can use the AIS data
            # for the initial state.
            idx = 0
            if ship_cfg_idx == 0:
                use_ais_ship_trajectory = False

            if ship_config.mmsi in mmsi_list:
                idx = [
                    i
                    for i in range(len(ais_vessel_data_list))
                    if ais_vessel_data_list[i].mmsi == ship_config.mmsi
                ][0]

            ais_vessel = ais_vessel_data_list.pop(idx)
            while ais_vessel.status.value not in config.allowed_nav_statuses:
                ais_vessel = ais_vessel_data_list.pop(idx)

            ship_list[ship_cfg_idx].transfer_vessel_ais_data(
                ais_vessel,
                use_ais_ship_trajectory,
                ship_config.t_start,
                ship_config.t_end,
            )
            ship_config.csog_state = ship_list[ship_cfg_idx].csog_state
            ship_config.mmsi = ship_list[ship_cfg_idx].mmsi

        return ship_list, config

    def generate_disturbance(
        self, config: sc.ScenarioConfig
    ) -> Optional[stoch.Disturbance]:
        """Generates a disturbance object from the scenario config.

        Args:
            - config (sc.ScenarioConfig): Scenario config object.

        Returns:
            - stoch.Disturbance: Disturbance object.
        """
        if config.stochasticity is None:
            return None

        ep = self._episode_counter
        if ep == self._disturbance_update_indices[ep]:
            disturbance = stoch.Disturbance(config.stochasticity)
        else:
            disturbance = self._prev_disturbance
        self._prev_disturbance = copy.deepcopy(disturbance)
        return disturbance

    def generate_ownship_csog_state(
        self, ownship: ship.Ship, config: sc.ScenarioConfig
    ) -> Tuple[ship.Ship, sc.ScenarioConfig, np.ndarray]:
        """Generates the initial own-ship pose for the scenario episode.

        Args:
            ownship (ship.Ship): Own-ship object.
            config (sc.ScenarioConfig): Scenario config object.

        Returns:
            Tuple[ship.Ship, sc.ScenarioConfig, np.ndarray]: Partially initialized own-ship in the scenario with pose set, the updated scenario config object and the generated/set own-ship csog state.
        """
        if ownship.csog_state.size > 0:
            return ownship, config, ownship.csog_state

        # Use 90% of the maximum speed as the maximum speed for the ships
        ep = self._episode_counter
        uniform_in_map_sample = self._uniform_os_state_update_indices[ep] == ep
        if ep == self._os_state_update_indices[ep]:
            csog_state = self.generate_random_csog_state(
                self._ownship_position_generation,
                U_min=3.5,
                U_max=0.6 * ownship.max_speed,
                draft=ownship.draft,
                min_hazard_clearance=np.min([30.0, ownship.length * 3.0]),
                first_episode_csog_state=(
                    self._first_csog_states[0][0] if not uniform_in_map_sample else None
                ),
            )
        else:
            csog_state = self._prev_ship_list[0].csog_state
        ownship.set_initial_state(csog_state)
        config.ship_list[0].csog_state = csog_state
        return ownship, config, csog_state

    def generate_target_ship_csog_states(
        self, ship_list: list, config: sc.ScenarioConfig
    ) -> Tuple[list, sc.ScenarioConfig, list]:
        """Generates the initial ship poses for the scenario episode.

        Args:
            ship_list (list): List of ships to be considered in simulation.
            config (sc.ScenarioConfig): Scenario config object.

        Returns:
            Tuple[list, sc.ScenarioConfig, list]: List of partially initialized ships in the scenario with poses set, the updated scenario config object and list of generated/set csog states.
        """
        ep = self._episode_counter
        uniform_in_map_sample = self._uniform_os_state_update_indices[ep] == ep
        ownship = ship_list[0]
        csog_state_list = [(ownship.csog_state, ownship.t_start, None)]
        for ship_cfg_idx, ship_config in enumerate(config.ship_list[1:]):
            ship_cfg_idx += 1  # Skip own-ship
            if ship_config.csog_state is not None:
                csog_state_list.append(
                    (ship_config.csog_state, ship_config.t_start, None)
                )
                continue

            ship_obj = ship_list[ship_cfg_idx]

            if (
                ep == self._do_state_update_indices[ep]
                or uniform_in_map_sample
                or self._prev_ship_list[ship_cfg_idx] is None
            ):
                csog_state, t_start, os_csog_state_basis = (
                    self.generate_target_ship_csog_state(
                        config,
                        ownship,
                        U_min=2.0,
                        U_max=0.8 * ship_obj.max_speed,
                        draft=ship_obj.draft,
                        min_hazard_clearance=np.min([15.0, ship_obj.length * 3.0]),
                        first_episode_csog_state=(
                            None
                            if (uniform_in_map_sample)
                            else self._first_csog_states[ship_cfg_idx]
                        ),
                    )
                )
            else:
                csog_state = self._prev_ship_list[ship_cfg_idx].csog_state
                t_start = self._prev_ship_list[ship_cfg_idx].t_start
                os_csog_state_basis = self._first_csog_states[ship_cfg_idx][2]

            ship_config.csog_state = csog_state
            ship_config.t_start = t_start
            ship_obj.set_initial_state(ship_config.csog_state, t_start=t_start)
            csog_state_list.append(
                (ship_config.csog_state, ship_config.t_start, os_csog_state_basis)
            )

        if ep % config.episode_generation.delta_uniform_position_sample == 0:
            self._first_csog_states[: len(ship_list)] = csog_state_list

        return ship_list, config, csog_state_list

    def generate_target_ship_csog_state(
        self,
        config: sc.ScenarioConfig,
        ownship: ship.Ship,
        U_min: float = 2.0,
        U_max: float = 8.0,
        draft: float = 2.0,
        min_hazard_clearance: float = 30.0,
        first_episode_csog_state: Optional[
            Tuple[np.ndarray, float | None, np.ndarray | None]
        ] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Generates a position for the target ship based on the perspective of the first ship/own-ship,
        such that the scenario is of the input type.

        Args:
            - config (sc.ScenarioConfig): Scenario config.
            - ownship (ship.Ship): Own-ship object.
            - U_min (float, optional): Obstacle minimum speed. Defaults to 2.0.
            - U_max (float, optional): Obstacle maximum speed. Defaults to 8.0.
            - draft (float, optional): Draft of target ship. Defaults to 2.0.
            - min_hazard_clearance (float, optional): Minimum distance between target ship and grounding hazards. Defaults to 100.0.
            - first_episode_csog_state (Optional[Tuple[np.ndarray, float | None]]): First scenario episode target ship COG-SOG state, (possibly) start time and (possibly) own-ship COG-SOG state used for generating the target ship COG-SOG state.

        Returns:
            - Tuple[np.ndarray, float]: Target ship COG-SOG state = [x, y, speed, course] and the start time for the target ship. Also
                returns the own-ship COG-SOG state used for generating the target ship COG-SOG state.
        """
        if first_episode_csog_state is not None:
            t_start = first_episode_csog_state[1]
            os_csog_state_basis = first_episode_csog_state[2]
            if (
                self._target_position_generation
                == sc.TargetPositionGenerationMethod.BasedOnOwnshipPositionThenGaussian
                or self._target_position_generation
                == sc.TargetPositionGenerationMethod.BasedOnOwnshipWaypointsThenGaussian
            ):
                return (
                    self.generate_gaussian_csog_state(
                        mean=first_episode_csog_state[0],
                        cov=self._config.gaussian_csog_state_perturbation_covariance,
                        draft=draft,
                        os_csog_state_basis=os_csog_state_basis,
                        min_hazard_clearance=min_hazard_clearance,
                    ),
                    t_start,
                    os_csog_state_basis,
                )
            elif first_episode_csog_state is not None and (
                self._target_position_generation
                == sc.TargetPositionGenerationMethod.BasedOnOwnshipPositionThenPerpendicular
                or self._target_position_generation
                == sc.TargetPositionGenerationMethod.BasedOnOwnshipWaypointsThenPerpendicular
            ):
                return (
                    self.generate_perpendicular_csog_state(
                        initial_csog_state=first_episode_csog_state[0],
                        os_csog_state_basis=os_csog_state_basis,
                        U_min=U_min,
                        U_max=U_max,
                        draft=draft,
                        min_hazard_clearance=min_hazard_clearance,
                    ),
                    t_start,
                    os_csog_state_basis,
                )

        scenario_type = config.type
        if scenario_type == sc.ScenarioType.MS:
            scenario_type = self.rng.choice(
                [
                    sc.ScenarioType.HO,
                    sc.ScenarioType.OT_ing,
                    sc.ScenarioType.OT_en,
                    sc.ScenarioType.CR_GW,
                    sc.ScenarioType.CR_SO,
                ]
            )

        os_csog_state_basis = ownship.csog_state
        t_start = 0.0
        if (
            self._target_position_generation
            == sc.TargetPositionGenerationMethod.BasedOnOwnshipWaypoints
            or self._target_position_generation
            == sc.TargetPositionGenerationMethod.BasedOnOwnshipWaypointsThenGaussian
            or self._target_position_generation
            == sc.TargetPositionGenerationMethod.BasedOnOwnshipWaypointsThenPerpendicular
        ):
            os_csog_state_basis, t_start = mhm.sample_state_along_waypoints(
                self.rng,
                ownship.waypoints,
                ownship.speed_plan,
                config.t_end - config.t_start,
            )

        ot_speed_margin = 1.0
        if (
            scenario_type == sc.ScenarioType.OT_en
            and U_max - ot_speed_margin <= os_csog_state_basis[2]
        ):
            print(
                f"WARNING: ScenarioType = OT_en: Own-ship speed should be below the maximum target ship speed minus margin of {ot_speed_margin}. Selecting a different scenario type..."
            )
            scenario_type = self.rng.choice(
                [
                    sc.ScenarioType.HO,
                    sc.ScenarioType.OT_ing,
                    sc.ScenarioType.CR_GW,
                    sc.ScenarioType.CR_SO,
                ]
            )

        if (
            scenario_type == sc.ScenarioType.OT_ing
            and U_min >= os_csog_state_basis[2] - ot_speed_margin
        ):
            print(
                f"WARNING: ScenarioType = OT_ing: Own-ship speed minus margin of {ot_speed_margin} should be above the minimum target ship speed. Selecting a different scenario type..."
            )
            scenario_type = self.rng.choice(
                [
                    sc.ScenarioType.HO,
                    sc.ScenarioType.OT_en,
                    sc.ScenarioType.CR_GW,
                    sc.ScenarioType.CR_SO,
                ]
            )

        min_depth = mapf.find_minimum_depth(draft, self.enc)
        max_iter = 2000
        y_min, x_min, y_max, x_max = self.enc.bbox
        distance_os_ts = self.rng.uniform(
            self._config.dist_between_ships_range[0],
            self._config.dist_between_ships_range[1],
        )
        bearing = self.rng.uniform(0.0, 2.0 * np.pi)
        if scenario_type == sc.ScenarioType.OT_en:
            bearing = self.rng.uniform(
                self._config.ot_bearing_range[0], self._config.ot_bearing_range[1]
            )
            x = os_csog_state_basis[0] - distance_os_ts * np.cos(
                os_csog_state_basis[3] + bearing
            )
            y = os_csog_state_basis[1] - distance_os_ts * np.sin(
                os_csog_state_basis[3] + bearing
            )
        else:
            bearing = self.rng.uniform(
                self._config.ot_bearing_range[0], self._config.ot_bearing_range[1]
            )
            x = os_csog_state_basis[0] + distance_os_ts * np.cos(
                os_csog_state_basis[3] + bearing
            )
            y = os_csog_state_basis[1] + distance_os_ts * np.sin(
                os_csog_state_basis[3] + bearing
            )
        speed = self.rng.uniform(U_min, U_max)
        accepted = False
        for i in range(max_iter):
            if scenario_type == sc.ScenarioType.HO:
                bearing = self.rng.uniform(
                    self._config.ho_bearing_range[0], self._config.ho_bearing_range[1]
                )
                speed = self.rng.uniform(U_min, U_max)
                course_modifier = 180.0 + self.rng.uniform(
                    self._config.ho_course_range[0], self._config.ho_course_range[1]
                )

            elif scenario_type == sc.ScenarioType.OT_ing:
                bearing = self.rng.uniform(
                    self._config.ot_bearing_range[0], self._config.ot_bearing_range[1]
                )
                speed = self.rng.uniform(
                    U_min, os_csog_state_basis[2] - ot_speed_margin
                )
                course_modifier = self.rng.uniform(
                    self._config.ot_course_range[0], self._config.ot_course_range[1]
                )

            elif scenario_type == sc.ScenarioType.OT_en:
                bearing = self.rng.uniform(
                    self._config.ot_bearing_range[0], self._config.ot_bearing_range[1]
                )
                speed = self.rng.uniform(os_csog_state_basis[2], U_max)
                course_modifier = self.rng.uniform(
                    self._config.ot_course_range[0], self._config.ot_course_range[1]
                )

            elif scenario_type == sc.ScenarioType.CR_GW:
                bearing = self.rng.uniform(
                    self._config.cr_bearing_range[0], self._config.cr_bearing_range[1]
                )
                speed = self.rng.uniform(U_min, U_max)
                course_modifier = -90.0 + self.rng.uniform(
                    self._config.cr_course_range[0], self._config.cr_course_range[1]
                )

            elif scenario_type == sc.ScenarioType.CR_SO:
                bearing = self.rng.uniform(
                    -self._config.cr_bearing_range[1], -self._config.cr_bearing_range[0]
                )
                speed = self.rng.uniform(U_min, U_max)
                course_modifier = 90.0 + self.rng.uniform(
                    self._config.cr_course_range[0], self._config.cr_course_range[1]
                )

            else:
                bearing = self.rng.uniform(0.0, 2.0 * np.pi)
                speed = self.rng.uniform(U_min, U_max)
                course_modifier = self.rng.uniform(0.0, 359.999)

            bearing = np.deg2rad(bearing)
            course = os_csog_state_basis[3] + np.deg2rad(course_modifier)

            distance_os_ts = self.rng.uniform(
                self._config.dist_between_ships_range[0],
                self._config.dist_between_ships_range[1],
            )
            if scenario_type == sc.ScenarioType.OT_en:
                x = os_csog_state_basis[0] - distance_os_ts * np.cos(
                    os_csog_state_basis[3] + bearing
                )
                y = os_csog_state_basis[1] - distance_os_ts * np.sin(
                    os_csog_state_basis[3] + bearing
                )
            else:
                x = os_csog_state_basis[0] + distance_os_ts * np.cos(
                    os_csog_state_basis[3] + bearing
                )
                y = os_csog_state_basis[1] + distance_os_ts * np.sin(
                    os_csog_state_basis[3] + bearing
                )

            inside_bbox = mhm.inside_bbox(
                np.array([x, y]), (x_min, y_min, x_max, y_max)
            )
            risky_enough = mhm.check_if_situation_is_risky_enough(
                os_csog_state_basis,
                np.array([x, y, speed, course]),
                self._config.t_cpa_threshold,
                self._config.d_cpa_threshold,
            )
            pointing_towards_land = mapf.check_if_pointing_too_close_towards_land(
                np.array([x, y, speed, course]),
                enc=self.enc,
                hazards=self._sg_hazards,
                min_dist_to_hazard=100.0,
            )

            d2hazards = mapf.distance_to_enc_hazards(
                y, x, min_depth=min_depth, enc=self.enc, hazards=self._sg_hazards
            )

            if (
                risky_enough
                and d2hazards >= min_hazard_clearance
                and inside_bbox
                and not pointing_towards_land
            ):
                accepted = True
                break

        if not accepted:
            print(
                "WARNING: No acceptable starting state found for the target ship! Using a random state projected onto the safe sea.."
            )
            # self.enc.draw_circle((y, x), radius=10.0, color="orange", fill=True, alpha=0.6)
            start_pos = np.array([x, y]) + speed * 500.0 * np.array(
                [np.cos(course), np.sin(course)]
            )
            end_pos = np.array([x, y])
            new_start_pos = mapf.find_closest_collision_free_point_on_segment(
                self.enc, start_pos, end_pos, draft, min_dist=min_hazard_clearance
            )
            x, y = new_start_pos[0], new_start_pos[1]
            # self.enc.draw_circle((y, x), radius=10.0, color="red", fill=True, alpha=0.6)
        return np.array([x, y, speed, course]), float(t_start), os_csog_state_basis

    def generate_gaussian_csog_state(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        draft: float,
        os_csog_state_basis: Optional[np.ndarray] = None,
        min_hazard_clearance: float = 30.0,
        show_plots: bool = False,
    ) -> np.ndarray:
        """Generates a COG-SOG state from a Gaussian distribution around the input mean (first episodic csog state) and covariance.

        Args:
            mean (np.ndarray): Mean of the Gaussian distribution, i.e. the first episodic csog state = [x, y, speed, course].
            cov (np.ndarray): Covariance of the Gaussian distribution.
            draft (float, optional): Draft of ship. Defaults to 2.0.
            os_csog_state_basis (Optional[np.ndarray], optional): Own-ship COG-SOG state used for generating the target ship COG-SOG state. Defaults to None.
            min_hazard_clearance (float, optional): Minimum distance between ship and ENC hazards. Defaults to 30.0.
            show_plots (bool, optional): Flag determining whether or not to show seacharts debugging plots. Defaults to False.

        Returns:
            np.ndarray: Array containing the random vessel state = [x, y, speed, course]
        """
        min_depth = mapf.find_minimum_depth(draft, self.enc)
        hazards = self._sg_hazards
        if min_depth > 1:
            hazards = mapf.extract_relevant_grounding_hazards_as_union(
                min_depth, self.enc
            )

        if show_plots and os_csog_state_basis is not None:
            ship_poly = mapf.create_ship_polygon(
                os_csog_state_basis[0],
                os_csog_state_basis[1],
                mf.wrap_angle_to_pmpi(os_csog_state_basis[3]),
                10.0,
                3.0,
                5.0,
                5.0,
            )
            self.enc.draw_polygon(ship_poly, color="yellow", alpha=0.6)

        max_iter = 300
        for _ in range(max_iter):
            perturbed_state = self.rng.multivariate_normal(mean, cov)
            d2hazards = mapf.distance_to_enc_hazards(
                perturbed_state[1],
                perturbed_state[0],
                min_depth=min_depth,
                enc=self.enc,
                hazards=hazards,
            )

            risky_enough = True
            if os_csog_state_basis is not None:
                risky_enough = mhm.check_if_situation_is_risky_enough(
                    os_csog_state_basis,
                    perturbed_state,
                    self._config.t_cpa_threshold,
                    self._config.d_cpa_threshold,
                )

            if show_plots and os_csog_state_basis is not None:
                ship_poly = mapf.create_ship_polygon(
                    perturbed_state[0],
                    perturbed_state[1],
                    mf.wrap_angle_to_pmpi(perturbed_state[3]),
                    10.0,
                    3.0,
                    5.0,
                    5.0,
                )
                self.enc.draw_polygon(ship_poly, color="pink", alpha=0.6)

            if d2hazards >= min_hazard_clearance and risky_enough:
                break

        return perturbed_state

    def generate_perpendicular_csog_state(
        self,
        initial_csog_state: np.ndarray,
        os_csog_state_basis: Optional[np.ndarray] = None,
        U_min: float = 2.0,
        U_max: float = 10.0,
        draft: float = 0.5,
        min_hazard_clearance: float = 20.0,
        show_plots: bool = False,
    ) -> np.ndarray:
        """Generates a COG-SOG state on a line perpendicular to the input initial_csog_state with given course, that (hopefully)
        statisfies the minimum land clearance.

        Args:
            initial_csog_state (np.ndarray): Initial target COG-SOG state to generate the perpendicular line from.
            os_csog_state_basis (Optional[np.ndarray], optional): Own-ship COG-SOG state used for generating the target ship COG-SOG state. Defaults to None.
            U_min (float, optional): Minimum speed of the ship. Defaults to 2.0.
            U_max (float, optional): Maximum speed of the ship. Defaults to 10.0.
            draft (float): Draft of the ship.
            min_hazard_clearance (float, optional): Minimum land clearance. Defaults to 30.0.
            show_plots (bool, optional): Flag determining whether or not to show seacharts debugging plots. Defaults to False.


        Returns:
            np.ndarray: Array containing the perpendicular vessel state = [x, y, speed, course]
        """
        min_depth = 1
        if draft > 1.0:
            min_depth = mapf.find_minimum_depth(draft, self.enc)

        perp_course = mf.wrap_angle_to_pmpi(initial_csog_state[3] + np.pi / 2.0)
        x, y = initial_csog_state[0], initial_csog_state[1]
        U = initial_csog_state[2]
        max_iter = 300
        dist_range_max = self._config.perpendicular_csog_state_perturbation_pm_range[0]
        speed_range_abs = self._config.perpendicular_csog_state_perturbation_pm_range[1]
        course_range_abs = self._config.perpendicular_csog_state_perturbation_pm_range[
            2
        ]
        if show_plots and os_csog_state_basis is not None:
            ship_poly = mapf.create_ship_polygon(
                os_csog_state_basis[0],
                os_csog_state_basis[1],
                mf.wrap_angle_to_pmpi(os_csog_state_basis[3]),
                10.0,
                3.0,
                5.0,
                5.0,
            )
            self.enc.draw_polygon(ship_poly, color="yellow", alpha=0.6)

        for _ in range(max_iter):
            dist_from_initial = self.rng.uniform(-dist_range_max, dist_range_max)
            x = x + dist_from_initial * np.cos(perp_course)
            y = y + dist_from_initial * np.sin(perp_course)
            d2hazard = mapf.distance_to_enc_hazards(
                y, x, min_depth=min_depth, enc=self.enc, hazards=self._sg_hazards
            )

            speed = self.rng.uniform(U - speed_range_abs, U + speed_range_abs)
            speed = np.clip(speed, U_min, U_max)
            course = self.rng.uniform(
                initial_csog_state[3] - course_range_abs,
                initial_csog_state[3] + course_range_abs,
            )
            course = mf.wrap_angle_to_pmpi(course)
            csog_state = np.array([x, y, speed, course])
            risky_enough = True
            if os_csog_state_basis is not None:
                risky_enough = mhm.check_if_situation_is_risky_enough(
                    os_csog_state_basis,
                    csog_state,
                    self._config.t_cpa_threshold,
                    self._config.d_cpa_threshold,
                )

            if show_plots and os_csog_state_basis is not None:
                ship_poly = mapf.create_ship_polygon(
                    csog_state[0],
                    csog_state[1],
                    mf.wrap_angle_to_pmpi(csog_state[3]),
                    10.0,
                    3.0,
                    5.0,
                    5.0,
                )
                self.enc.draw_polygon(ship_poly, color="pink", alpha=0.6)

            if d2hazard >= min_hazard_clearance and risky_enough:
                break

        return csog_state

    def generate_random_csog_state(
        self,
        method: sc.OwnshipPositionGenerationMethod | sc.TargetPositionGenerationMethod,
        U_min: float = 1.0,
        U_max: float = 10.0,
        draft: float = 5.0,
        min_hazard_clearance: float = 50.0,
        first_episode_csog_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Creates a random COG-SOG state which adheres to the ship's draft and maximum speed.

        Args:
            - method (sc.OwnshipPositionGenerationMethod | sc.TargetPositionGenerationMethod): Method for generating the position.
            - U_min (float, optional): Minimum speed of the ship. Defaults to 1.0.
            - U_max (float, optional): Maximum speed of the ship. Defaults to 10.0.
            - draft (float, optional): How deep the ship keel is into the water. Defaults to 5.
            - min_hazard_clearance (float, optional): Minimum distance between ship and land. Defaults to 50.0.
            - first_episode_csog_state (Optional[np.ndarray], optional): First scenario episode ship COG-SOG state.

        Returns:
            - np.ndarray: Array containing the vessel state = [x, y, speed, course]
        """
        if first_episode_csog_state is not None and (
            method == sc.OwnshipPositionGenerationMethod.UniformInTheMapThenGaussian
            or method
            == sc.TargetPositionGenerationMethod.BasedOnOwnshipPositionThenGaussian
        ):
            return self.generate_gaussian_csog_state(
                mean=first_episode_csog_state,
                cov=self._config.gaussian_csog_state_perturbation_covariance,
                draft=draft,
                min_hazard_clearance=min_hazard_clearance,
            )
        elif first_episode_csog_state is not None and (
            method
            == sc.TargetPositionGenerationMethod.BasedOnOwnshipPositionThenPerpendicular
            or method
            == sc.TargetPositionGenerationMethod.BasedOnOwnshipWaypointsThenPerpendicular
        ):
            return self.generate_perpendicular_csog_state(
                initial_csog_state=first_episode_csog_state,
                draft=draft,
                min_hazard_clearance=min_hazard_clearance,
            )

        x, y = mapf.generate_random_position_from_draft(
            self.rng,
            self.enc,
            draft,
            self.safe_sea_cdt,
            self.safe_sea_cdt_weights,
            min_hazard_clearance,
        )
        speed = self.rng.uniform(U_min, U_max)
        distance_vectors = mapf.compute_distance_vectors_to_grounding(
            np.array([y, x]).reshape(-1, 1),
            mapf.find_minimum_depth(draft, self.enc),
            self.enc,
        )
        dist_vec = distance_vectors[:, 0]
        angle_to_land = np.arctan2(dist_vec[0], dist_vec[1])
        dist_vec_to_bbox = mapf.compute_distance_vector_to_bbox(
            y, x, self.enc.bbox, self.enc
        )
        angle_to_bbox = np.arctan2(dist_vec_to_bbox[0], dist_vec_to_bbox[1])
        # If the ship is close to the bounding box or land, we want to make sure it is not heading straight into it.
        course = self.rng.uniform(0.0, 2.0 * np.pi)
        if np.linalg.norm(dist_vec) < 2.0 * min_hazard_clearance:
            course = angle_to_land + np.pi + self.rng.uniform(-np.pi / 2.0, np.pi / 2.0)

        if np.linalg.norm(dist_vec_to_bbox) < 2.0 * min_hazard_clearance:
            course = angle_to_bbox + np.pi + self.rng.uniform(-np.pi / 2.0, np.pi / 2.0)

        return np.array([x, y, speed, mf.wrap_angle_to_pmpi(course)])

    @property
    def enc_bbox(self) -> np.ndarray:
        """Returns the bounding box of the considered ENC area.

        Returns:
            - np.ndarray: Array containing the ENC bounding box = [min_x, min_y, max_x, max_y]
        """
        size = self.enc.size
        origin = self.enc.origin

        return np.array(
            [origin[0], origin[1], origin[0] + size[0], origin[1] + size[1]]
        )

    @property
    def enc_origin(self) -> np.ndarray:
        return np.array([self.enc.origin[1], self.enc.origin[0]])
