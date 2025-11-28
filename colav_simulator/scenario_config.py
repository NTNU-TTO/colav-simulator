"""
scenario_config.py

Summary:
    Contains class definitions for the configuration of a maritime COLAV scenario.

Author: Trym Tengesdal
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

import colav_simulator.common.file_utils as file_utils
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import colav_simulator.core.ship as ship
import colav_simulator.core.stochasticity as stoch


class OwnshipPositionGenerationMethod(Enum):
    """Enum for the different possible methods of generating ownship positions in a scenario."""

    UniformlyInMap = 0  # Positions are uniformly generated in the map (safe sea area)
    UniformInTheMapThenGaussian = 1  # Every "delta_uniform_position_sample" (default=10000000) position is uniformly generated in the map, then the next positions are generated through a Gaussian centered around the first position.
    MapUniformThenPerpendicular = 2  # Every "delta_uniform_position_sample" (default=10000000) position is uniformly generated in the map, then the next positions are generated along a line perpendicular to the first OS positon and heading direction


class TargetPositionGenerationMethod(Enum):
    """Enum for the different possible methods of generating target positions in a scenario."""

    BasedOnOwnshipPosition = 0  # Positions are generated based on the own-ship position and heading direction, uniformly
    BasedOnOwnshipPositionThenGaussian = 1  # Positions are generated based on the own-ship position and heading direction, with subsequent positions generated through a Gaussian centered around the first position.
    BasedOnOwnshipPositionThenPerpendicular = 2  # Positions are generated based on the own-ship initial state, with subsequent positions generated along a line perpendicular to the first position and heading direction
    BasedOnOwnshipWaypoints = 3  # Positions are generated based on the own-ship waypoints and heading direction, uniformly in a corridor around the waypoints.
    BasedOnOwnshipWaypointsThenGaussian = 4  # Positions are generated based on the own-ship waypoints and heading direction, with subsequent positions generated through a Gaussian centered around the first position.
    BasedOnOwnshipWaypointsThenPerpendicular = 5  # Positions are generated based on the own-ship waypoints, with subsequent positions generated along a line perpendicular to the first position and heading direction


class ScenarioType(Enum):
    """Enum for the different possible scenario/situation types.

    Explanation:
        SS: Only one ship (the own-ship) in the scenario.
        HO: Head on scenario.
        OT_ing: Overtaking scenario (own-ship overtakes and should give-way).
        OT_en: Overtaken scenario (own-ship is overtaken and should stand-on).
        CR_GW: Crossing scenario where own-ship has give-way duties.
        CR_SO: Crossing scenario where own-ship has stand-on duties.
        MS: Multiple ships scenario without any specification of COLREGS situations.
    """

    SS = 0
    HO = 1
    OT_ing = 2
    OT_en = 3
    CR_GW = 4
    CR_SO = 5
    MS = 6


@dataclass
class RLConfig:
    """Configuration class for an RL agent."""

    observation_type: Optional[dict] = field(
        default_factory=lambda: {
            "dict_observation": [
                "navigation_3dof_state_observation",
                "lidar_like_observation",
            ]
        }
    )
    action_type: Optional[str] = "continuous_autopilot_reference_action"
    action_sample_time: Optional[float] = (
        None  # Time between each action sample, equal to simulation time step if not specified.
    )

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RLConfig()
        if "observation_type" in config_dict:
            config.observation_type = config_dict["observation_type"]
        if "action_type" in config_dict:
            config.action_type = config_dict["action_type"]
        if "action_sample_time" in config_dict:
            config.action_sample_time = config_dict["action_sample_time"]
        return config

    def to_dict(self):
        output = asdict(self)
        return output


@dataclass
class EpisodeGenerationConfig:
    """Class describing how the episodes are generated, i.e. how often the own-ship plan+state, dynamic obstacle state, dynamic obstacle plan and disturbance are updated/re-randomized."""

    n_episodes: Optional[int] = (
        1  # Number of episodes to run for the scenario. Each episode is a new random realization of the scenario, with unique own-ship dynamic obstacle states+plans, and disturbance realizations.
    )
    n_constant_os_state_episodes: Optional[int] = (
        1  # Number of episodes to run with the same own-ship state before generating a new one.
    )
    n_constant_os_plan_episodes: Optional[int] = (
        None  # Number of episodes to run with the same own-ship plan before generating a new one.
    )
    n_constant_do_state_episodes: Optional[int] = (
        1  # Number of episodes to run with the same initial dynamic obstacle state before generating a new one.
    )
    n_plans_per_do_state: Optional[int] = (
        None  # Number of plans per initial dynamic obstacle state.
    )
    n_constant_disturbance_episodes: Optional[int] = (
        None  # Number of episodes to run with the same disturbance realizzation (applicable only if stocastic disturbances are used), before generating a new one.
    )
    ownship_position_generation: OwnshipPositionGenerationMethod = (
        OwnshipPositionGenerationMethod.UniformInTheMapThenGaussian  # Method for generating ship positions in the scenario.
    )
    target_position_generation: TargetPositionGenerationMethod = (
        TargetPositionGenerationMethod.BasedOnOwnshipWaypointsThenGaussian
    )
    delta_uniform_position_sample: Optional[int] = (
        10000000  # Number of episodes/position samples between each UniformlyInTheMap position sample (for the ownship). Not applicable if position_generation is set to UniformlyInMap.
    )

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = EpisodeGenerationConfig()
        if "n_episodes" in config_dict:
            config.n_episodes = config_dict["n_episodes"]
        if "n_constant_os_state_episodes" in config_dict:
            config.n_constant_os_state_episodes = config_dict[
                "n_constant_os_state_episodes"
            ]
        if "n_constant_os_plan_episodes" in config_dict:
            config.n_constant_os_plan_episodes = config_dict[
                "n_constant_os_plan_episodes"
            ]
        if "n_constant_do_state_episodes" in config_dict:
            config.n_constant_do_state_episodes = config_dict[
                "n_constant_do_state_episodes"
            ]
        if "n_plans_per_do_state" in config_dict:
            config.n_plans_per_do_state = config_dict["n_plans_per_do_state"]
        if "n_constant_disturbance_episodes" in config_dict:
            config.n_constant_disturbance_episodes = config_dict[
                "n_constant_disturbance_episodes"
            ]
        if "delta_uniform_position_sample" in config_dict:
            config.delta_uniform_position_sample = config_dict[
                "delta_uniform_position_sample"
            ]
        if "ownship_position_generation" in config_dict:
            config.ownship_position_generation = OwnshipPositionGenerationMethod[
                config_dict["ownship_position_generation"]
            ]
        if "target_position_generation" in config_dict:
            config.target_position_generation = TargetPositionGenerationMethod[
                config_dict["target_position_generation"]
            ]
        return config

    def to_dict(self):
        config_dict = asdict(self)
        config_dict["ownship_position_generation"] = (
            self.ownship_position_generation.name
        )
        config_dict["target_position_generation"] = self.target_position_generation.name
        return config_dict


@dataclass
class ScenarioConfig:
    """Configuration class for a maritime COLAV scenario."""

    name: str
    save_scenario: bool
    t_start: float
    t_end: float
    dt_sim: float
    type: ScenarioType
    utm_zone: int
    map_data_files: list  # List of file paths to .gdb database files used by seacharts to create the map
    new_load_of_map_data: bool  # If True, seacharts will process .gdb files into shapefiles. If false, it will use existing shapefiles.
    map_size: Optional[Tuple[float, float]] = (
        None  # Size of the map considered in the scenario (in meters) referenced to the origin.
    )
    map_origin_enu: Optional[Tuple[float, float]] = (
        None  # Origin of the map considered in the scenario (in UTM coordinates per now)
    )
    map_tolerance: Optional[int] = 0  # Tolerance for the map simplification process
    map_buffer: Optional[int] = 0  # Buffer for the map simplification process
    ais_data_file: Optional[Path] = None  # Path to the AIS data file, if considered
    ship_data_file: Optional[Path] = (
        None  # Path to the ship information data file associated with AIS data, if considered
    )
    allowed_nav_statuses: Optional[list] = (
        None  # List of AIS navigation statuses that are allowed in the scenario
    )
    episode_generation: Optional[EpisodeGenerationConfig] = field(
        default_factory=lambda: EpisodeGenerationConfig()
    )
    n_random_ships: Optional[int] = (
        None  # Fixed number of random ships in the scenario, excluding the own-ship, if considered
    )
    n_random_ships_range: Optional[list] = (
        None  # Variable range of number of random ships in the scenario, excluding the own-ship, if considered
    )
    ship_list: Optional[list] = field(
        default_factory=[]
    )  # List of ship configurations for the scenario, does not have to be equal to the number of ships in the scenario.
    filename: Optional[str] = None  # Filename of the scenario, stored after creation
    stochasticity: Optional[stoch.Config] = (
        None  # Configuration class containing stochasticity parameters for the scenario
    )
    rl: Optional[RLConfig] = (
        None  # Configuration class containing COLAVEnvironment observation and action  parameters for the scenario
    )

    @classmethod
    def parse_episode_generation_config(
        cls, config_dict: dict
    ) -> EpisodeGenerationConfig:
        """Parses the episodic generation configuration dictionary, ensures backwards compatibility with old scenario config files.

        Args:
            config_dict (dict): Scenario generation configuration dictionary.
        """
        if "episode_generation" in config_dict:
            return EpisodeGenerationConfig.from_dict(config_dict["episode_generation"])

        n_constant_os_plan_episodes = 1
        if "n_constant_os_plan_episodes" in config_dict:
            n_constant_os_plan_episodes = config_dict["n_constant_os_plan_episodes"]
        n_constant_do_state_episodes = 1
        if "n_constant_do_state_episodes" in config_dict:
            n_constant_do_state_episodes = config_dict["n_constant_do_state_episodes"]
        n_plans_per_do_state = 1
        if "n_plans_per_do_state" in config_dict:
            n_plans_per_do_state = config_dict["n_plans_per_do_state"]
        n_constant_disturbance_episodes = 1
        if "n_constant_disturbance_episodes" in config_dict:
            n_constant_disturbance_episodes = config_dict[
                "n_constant_disturbance_episodes"
            ]
        n_episodes = 1
        if "n_episodes" in config_dict:
            n_episodes = config_dict["n_episodes"]

        ownship_position_generation = (
            OwnshipPositionGenerationMethod.UniformInTheMapThenGaussian
        )
        if "ownship_position_generation" in config_dict:
            ownship_position_generation = OwnshipPositionGenerationMethod[
                config_dict["ownship_position_generation"]
            ]

        target_position_generation = (
            TargetPositionGenerationMethod.BasedOnOwnshipWaypointsThenGaussian
        )
        if "target_position_generation" in config_dict:
            target_position_generation = TargetPositionGenerationMethod[
                config_dict["target_position_generation"]
            ]

        return EpisodeGenerationConfig(
            n_episodes=n_episodes,
            n_constant_os_plan_episodes=n_constant_os_plan_episodes,
            n_constant_do_state_episodes=n_constant_do_state_episodes,
            n_plans_per_do_state=n_plans_per_do_state,
            n_constant_disturbance_episodes=n_constant_disturbance_episodes,
            ownship_position_generation=ownship_position_generation,
            target_position_generation=target_position_generation,
        )

    def to_dict(self) -> dict:
        output = {
            "name": self.name,
            "save_scenario": self.save_scenario,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "dt_sim": self.dt_sim,
            "type": self.type.name,
            "episode_generation": self.episode_generation.to_dict(),
            "n_random_ships": self.n_random_ships,
            "n_random_ships_range": self.n_random_ships_range,
            "utm_zone": self.utm_zone,
            "map_data_files": self.map_data_files,
            "map_tolerance": self.map_tolerance,
            "map_buffer": self.map_buffer,
            "map_size": [float(si) for si in self.map_size]
            if self.map_size is not None
            else None,
            "map_origin_enu": [float(mo) for mo in self.map_origin_enu]
            if self.map_origin_enu is not None
            else None,
            "new_load_of_map_data": self.new_load_of_map_data,
            "ais_data_file": str(self.ais_data_file)
            if self.ais_data_file is not None
            else None,
            "ship_data_file": str(self.ship_data_file)
            if self.ship_data_file is not None
            else None,
            "allowed_nav_statuses": self.allowed_nav_statuses,
            "filename": self.filename if self.filename is not None else None,
            "stochasticity": self.stochasticity.to_dict()
            if self.stochasticity is not None
            else None,
            "rl": self.rl.to_dict() if self.rl is not None else None,
            "ship_list": [],
        }

        if self.ship_list is not None:
            for ship_config in self.ship_list:
                output["ship_list"].append(ship_config.to_dict())
        return output

    @staticmethod
    def handle_map_data_files(map_data_files: list) -> list:
        if map_data_files is None:
            return None

        enc_data_dir = Path.home() / "enc_data"
        resolved_map_data_files = []
        missing_files = []

        for file_path in map_data_files:
            path = Path(file_path)
            if path.is_absolute():
                resolved_path = path
            else:
                resolved_path = enc_data_dir / path

            if not resolved_path.exists():
                missing_files.append(str(resolved_path))
            else:
                resolved_map_data_files.append(str(resolved_path))

        if missing_files:
            raise FileNotFoundError(
                f"Map data file(s) not found: {', '.join(missing_files)}"
            )

        return resolved_map_data_files

    @classmethod
    def from_dict(cls, config_dict: dict):
        map_data_files = (
            config_dict["map_data_files"] if "map_data_files" in config_dict else None
        )
        if map_data_files is not None:
            map_data_files = cls.handle_map_data_files(map_data_files)

        config = ScenarioConfig(
            name=config_dict["name"],
            save_scenario=config_dict["save_scenario"],
            t_start=config_dict["t_start"],
            t_end=config_dict["t_end"],
            dt_sim=config_dict["dt_sim"],
            type=ScenarioType[config_dict["type"]],
            utm_zone=config_dict["utm_zone"],
            map_data_files=map_data_files,
            map_size=tuple(config_dict["map_size"])
            if "map_size" in config_dict
            else None,
            map_origin_enu=tuple(config_dict["map_origin_enu"])
            if "map_origin_enu" in config_dict
            else None,
            map_tolerance=config_dict["map_tolerance"]
            if "map_tolerance" in config_dict
            else 0,
            map_buffer=config_dict["map_buffer"] if "map_buffer" in config_dict else 0,
            n_random_ships=config_dict["n_random_ships"]
            if "n_random_ships" in config_dict
            else None,
            n_random_ships_range=config_dict["n_random_ships_range"]
            if "n_random_ships_range" in config_dict
            else None,
            ais_data_file=(
                Path(config_dict["ais_data_file"])
                if "ais_data_file" in config_dict
                and config_dict["ais_data_file"] is not None
                else None
            ),
            new_load_of_map_data=config_dict["new_load_of_map_data"],
            filename=config_dict["filename"] if "filename" in config_dict else None,
            stochasticity=(
                stoch.Config.from_dict(config_dict["stochasticity"])
                if "stochasticity" in config_dict
                else None
            ),
            rl=RLConfig.from_dict(config_dict["rl"]) if "rl" in config_dict else None,
            ship_list=[],
        )

        if config.rl is not None and config.rl.action_sample_time is None:
            config.rl.action_sample_time = config.dt_sim

        ep_gen_cfg = ScenarioConfig.parse_episode_generation_config(config_dict)
        config.episode_generation = ep_gen_cfg
        if config.ais_data_file is not None:
            if len(config.ais_data_file.parts) == 1:
                config.ais_data_file = dp.ais_data / config.ais_data_file

            config.ship_data_file = Path(config_dict["ship_data_file"])
            if len(config.ship_data_file.parts) == 1:
                config.ship_data_file = dp.ais_data / config.ship_data_file

            config.allowed_nav_statuses = config_dict["allowed_nav_statuses"]

        if "ship_list" in config_dict:
            config.ship_list = []
            for ship_config in config_dict["ship_list"]:
                config.ship_list.append(ship.Config.from_dict(ship_config))
        return config


def save_scenario_episode_definition(
    scenario_config: ScenarioConfig, folder: Path
) -> str:
    """Saves the the scenario episode defined by the preliminary scenario configuration and list of configured ships.

    Uses the config to create a unique scenario name and filename. The scenario is saved in the default scenario save folder.

    Args:
        - scenario_config (ScenarioConfig): Scenario configuration
        - folder (Path): Absolute path to the folder where the scenario definition should be saved.

    Returns:
        - str: The filename of the saved scenario.
    """
    if not folder.exists():
        folder.mkdir(parents=False)
    scenario_config_dict: dict = scenario_config.to_dict()
    scenario_config_dict["save_scenario"] = False
    scenario_config_dict.pop("n_random_ships_range")
    current_datetime_str = mhm.current_utc_datetime_str("%d%m%Y_%H%M%S")
    scenario_config_dict["name"] = (
        scenario_config_dict["name"] + "_" + current_datetime_str
    )
    filename = scenario_config.name + "_" + current_datetime_str + ".yaml"
    scenario_config_dict["filename"] = filename
    save_file = folder / filename
    with save_file.open(mode="w") as file:
        yaml.dump(scenario_config_dict, file)
    return filename


def find_global_map_origin_and_size(
    config: ScenarioConfig,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Finds the global map origin and size encompassing all ships in the scenario.

    Args:
        - config (sc.ScenarioConfig): Scenario configuration

    Returns:
        - Tuple[np.ndarray, np.ndarray]: Global map origin and size
    """
    assert config.map_origin_enu is not None and config.map_size is not None
    assert config.ship_list is not None
    map_origin_enu = config.map_origin_enu
    map_size = config.map_size

    min_east = map_origin_enu[0]
    min_north = map_origin_enu[1]
    max_east = map_origin_enu[0] + map_size[0]
    max_north = map_origin_enu[1] + map_size[1]
    for ship_config in config.ship_list:
        if ship_config.csog_state is not None:
            csog_state = ship_config.csog_state
            if csog_state[0] < min_north:
                min_north = csog_state[0]
            if csog_state[0] > max_north:
                max_north = csog_state[0]
            if csog_state[1] < min_east:
                min_east = csog_state[1]
            if csog_state[1] > max_east:
                max_east = csog_state[1]

        if ship_config.waypoints is not None:
            waypoints = ship_config.waypoints
            if np.min(waypoints[0, :]) < min_north:
                min_north = np.min(waypoints[0, :])
            if np.max(waypoints[0, :]) > max_north:
                max_north = np.max(waypoints[0, :])
            if np.min(waypoints[1, :]) < min_east:
                min_east = np.min(waypoints[1, :])
            if np.max(waypoints[1, :]) > max_east:
                max_east = np.max(waypoints[1, :])

    map_origin_enu = min_east, min_north
    map_size = max_east - min_east, max_north - min_north
    return map_origin_enu, map_size


def process_ais_data(config: ScenarioConfig) -> dict:
    """Processes AIS data from file, returns a dict containing AIS VesselData, ship MMSIs and the coordinate frame origin and size.

    Args:
        - config (ScenarioConfig): Configuration object containing all parameters/settings related to the creation of a scenario.

    Returns:
        - dict: Dictionary containing processed AIS data.
    """
    output = {}
    if config.ais_data_file is not None:
        output = file_utils.read_ais_data(
            config.ais_data_file,
            config.ship_data_file,
            config.utm_zone,
            config.map_origin_enu,
            config.map_size,
            config.dt_sim,
        )
    return output
