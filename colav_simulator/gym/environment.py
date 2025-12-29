"""Wraps the colav-simulator for use with Gymnasium.

Author: Trym Tengesdal
"""

import pathlib

import gymnasium as gym
import numpy as np
import seacharts.enc as senc

import colav_simulator.core.ship as cs_ship
import colav_simulator.core.stochasticity as stoch
import colav_simulator.gym.action as csgym_action
import colav_simulator.gym.observation as csgym_obs
import colav_simulator.gym.reward as rw
import colav_simulator.scenario_config as sc
import colav_simulator.scenario_generator as sg
import colav_simulator.simulator as cssim


class COLAVEnvironment(gym.Env):
    """An environment for performing ship collision-free planning tasks.

    The environment is centered on the own-ship (single-agent), and consists of
    a maritime scenario with possibly multiple other vessels and grounding hazards
    from ENC data.
    """

    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 30, "video.frames_per_second": 30}
    observation_type: csgym_obs.ObservationType
    action_type: csgym_action.ActionType
    scenario_config: sc.ScenarioConfig
    scenario_data_tup: tuple

    def __init__(  # noqa: PLR0915
        self,
        simulator_config: cssim.Config | None = None,
        scenario_generator_config: sg.Config | None = None,
        scenario_config: sc.ScenarioConfig | pathlib.Path | None = None,
        scenario_file_folder: pathlib.Path | list[pathlib.Path] | None = None,
        reload_map: bool | None = True,
        rewarder_class: rw.IReward | None = rw.Rewarder,
        rewarder_kwargs: dict | None = None,
        action_type: str | None = None,
        action_type_class: csgym_action.ActionType | None = None,
        action_kwargs: dict | None = None,
        action_sample_time: float | None = None,
        observation_type: dict | str | None = None,
        observation_kwargs: dict | None = None,
        render_mode: str | None = "rgb_array",
        render_update_rate: float | None = None,
        verbose: bool | None = True,
        show_loaded_scenario_data: bool | None = False,
        shuffle_loaded_scenario_data: bool | None = False,
        max_number_of_episodes: int | None = None,
        merge_loaded_scenario_episodes: bool | None = False,
        identifier: int | str | None = "COLAVEnv",
        seed: int | None = None,
    ) -> None:
        """Initializes the environment.

        Note that the scenario config object takes precedence over the scenario
        config file, which again takes precedence over the scenario file list.

        For action and observation types defined under `gym/action` and
        `gym/observation`, specify usage of these by providing the class name as
        a lower snake case string with the `action_type` and `observation_type`
        parameters.

        For external action types and the reward class you want to use, provide
        the class as an argument with the `action_type_class` and `rewarder_class`
        parameters, respectively.


        Args:
            simulator_config (Optional[cssim.Config]): Simulator configuration.
            scenario_generator_config (Optional[sg.Config]): Scenario generator configuration.
            scenario_config (Optional[sc.ScenarioConfig | Path]): Scenario
                configuration, either config object or path.
            scenario_file_folder (Optional[List[Path] | Path): Folder path(s) to
                scenario episode files.
            reload_map (Optional[bool]): Whether to reload the scenario ENC map.
                NOTE: Might cause issues with vectorized environments due to race
                conditions.
            rewarder_class (Optional[rw.IReward]): Rewarder class.
            rewarder_kwargs (Optional[dict]): Rewarder keyword arguments.
            action_type (Optional[str]): Action type.
            action_type_class (Optional[csgym_action.ActionType]): Action type
                class. Typically used for externally developed action types.
            action_kwargs (Optional[dict]): Keyword arguments passed to the action type upon construction.
            action_sample_time (Optional[float]): Action sample time, i.e. the time between each applied action.
            observation_type (Optional[dict | str]): Observation type.
            observation_kwargs (Optional[dict]): Keyword arguments passed to the observation type.
            render_mode (Optional[str]): Render mode.
            render_update_rate (Optional[float]): Render update rate.
            verbose (Optional[bool]): Wheter to print debugging info or not.
            show_loaded_scenario_data (Optional[bool]): Whether to show the loaded scenario data or not.
            shuffle_loaded_scenario_data (Optional[bool]): Whether to shuffle the loaded scenario data or not.
            max_number_of_episodes (Optional[int]): Maximum number of episodes
                to generate/load. Defaults to none (i.e. no limit).
            merge_loaded_scenario_episodes (Optional[bool]): Whether to merge
                multiple loaded scenario episodes into one scenario or not.
            identifier (Optional[int | str]): Identifier for the environment.
            seed (Optional[int]): Seed for the random number generator.
        """
        if observation_kwargs is None:
            observation_kwargs = {}
        if action_kwargs is None:
            action_kwargs = {}
        if rewarder_kwargs is None:
            rewarder_kwargs = {}
        super().__init__()
        if scenario_config is None and scenario_file_folder is None:
            msg = "Either scenario config or scenario file folder must be provided!"
            raise ValueError(msg)

        # Dummy spaces, must be overwritten by _define_spaces after call to reset the environment
        self.action_kwargs = action_kwargs
        self.action_type_class = action_type_class
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)
        self.action_type = None
        self.observation_kwargs = observation_kwargs
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)
        self.observation_type = None
        self.simulator_config: cssim.Config = simulator_config
        self.simulator: cssim.Simulator = cssim.Simulator(config=simulator_config)
        self.scenario_generator: sg.ScenarioGenerator = sg.ScenarioGenerator(config=scenario_generator_config, seed=seed)
        self.scenario_config: sc.ScenarioConfig | pathlib.Path | None = scenario_config
        self.scenario_file_folder: pathlib.Path | None = scenario_file_folder
        self.scenario_data_tup: tuple | None = None
        self.reload_map: bool = reload_map
        self._has_init_generated: bool = False
        self.max_number_of_episodes: int | None = max_number_of_episodes

        self._seed: int | None = seed
        self.env_id = identifier
        self.done = False
        self.steps: int = 0
        self.last_info: dict = {}
        self.last_reward: float = 0.0
        self.terminal_info: dict = {}
        self.episodes: int = 0
        self.n_episodes: int = 0
        self.ownship: cs_ship.Ship | None = None
        self.render_mode = render_mode
        self.render_update_rate = render_update_rate
        self.verbose: bool = verbose
        self.current_frame: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)
        if self.scenario_file_folder is not None:
            self._load(
                scenario_file_folder=self.scenario_file_folder,
                reload_map=self.reload_map,
                show=show_loaded_scenario_data,
                shuffle=shuffle_loaded_scenario_data,
                merge_loaded_scenario_episodes=merge_loaded_scenario_episodes,
                max_number_of_episodes=self.max_number_of_episodes,
            )
            self.loaded_scenario_data = True
        else:
            self._generate(
                scenario_config=self.scenario_config,
                reload_map=self.reload_map,
                max_number_of_episodes=self.max_number_of_episodes,
            )

        if not isinstance(self.scenario_config, sc.ScenarioConfig):
            msg = "Scenario config not initialized properly!"
            raise ValueError(msg)
        self._has_init_generated = True
        (scenario_episode_list, scenario_enc) = self.scenario_data_tup
        episode_data = scenario_episode_list[0]
        self.n_episodes = len(scenario_episode_list)

        self.simulator.initialize_scenario_episode(
            ship_list=episode_data["ship_list"],
            sconfig=episode_data["config"],
            enc=scenario_enc,
            disturbance=episode_data["disturbance"],
            seed=seed,
        )
        self.ownship = self.simulator.ownship

        self.action_type_cfg = action_type if action_type is not None else self.scenario_config.rl.action_type
        self.observation_type_cfg = (
            observation_type if observation_type is not None else self.scenario_config.rl.observation_type
        )
        self.dt_action = action_sample_time if action_sample_time is not None else self.scenario_config.rl.action_sample_time

        self.rewarder_class = rewarder_class
        self.rewarder_kwargs = rewarder_kwargs
        self.rewarder = self.rewarder_class(env=self, **self.rewarder_kwargs)

        # if self.episodes == 0:
        #     self._clear_render()
        #     tracemalloc.start(40)
        #     self.t_prev_malloc_snapshot = tracemalloc.take_snapshot()

        self._define_spaces()

    def close(self) -> None:
        """Closes the environment. To be called after usage."""
        self.done = True
        if self.simulator.visualizer is not None:
            self.simulator.visualizer.close_live_plot()  # not thread safe unless the matplotlib backend is set to 'Agg'
            self.simulator.visualizer = None

    def _define_spaces(self) -> None:
        """Defines the action and observation spaces."""
        if self.scenario_config is None:
            msg = "Scenario config not initialized!"
            raise ValueError(msg)

        if self.action_type is not None and self.observation_type is not None:
            return

        self.dt_action = self.dt_action if self.dt_action is not None else self.simulator.dt
        if self.dt_action % self.simulator.dt != 0.0:
            msg = "Action sampling time must be a multiple of simulator time step!"
            raise ValueError(msg)

        if self.action_type_class is None:
            self.action_type = csgym_action.action_factory(
                self, self.action_type_cfg, sample_time=self.dt_action, **self.action_kwargs
            )
        else:
            self.action_type = self.action_type_class(self, sample_time=self.dt_action, **self.action_kwargs)
        self.action_space = self.action_type.space()

        self.observation_type = csgym_obs.observation_factory(self, self.observation_type_cfg, **self.observation_kwargs)
        self.observation_space = self.observation_type.space()

    def _is_terminated(self) -> bool:
        """Check whether the current state is a terminal state.

        Returns:
            bool: Whether the current state is a terminal state
        """
        return bool(self.simulator.is_terminated(self.verbose, prefix_string=f"[{self.env_id.upper()}] "))

    def _is_truncated(self) -> bool:
        """Check whether the current state is a truncated state (time limit reached).

        Returns:
            bool: Whether the current state is a truncated state
        """
        return bool(self.simulator.is_truncated(self.verbose, prefix_string=f"[{self.env_id.upper()}] "))

    def _load(
        self,
        scenario_file_folder: pathlib.Path | list[pathlib.Path],
        reload_map: bool = True,
        show: bool = False,
        shuffle: bool = False,
        merge_loaded_scenario_episodes: bool = False,
        max_number_of_episodes: int | None = None,
    ) -> None:
        """Load scenario episodes from files or a folder.

        Args:
            scenario_file_folder (pathlib.Path | List[pathlib.Path]): Folder
                path(s) where all episodes for the scenario(s) are found.
            reload_map (bool): Whether to reload the scenario(s) map.
            show (bool): Whether to show the scenario(s) data or not.
            shuffle (bool): Whether to shuffle the scenario(s) episode data or not.
            merge_loaded_scenario_episodes (bool): Whether to merge the scenario episodes into one scenario or not.
            max_number_of_episodes (Optional[int]): Maximum number of episodes to load.
        """
        name = (
            scenario_file_folder.name
            if isinstance(scenario_file_folder, pathlib.Path)
            else [f.name for f in scenario_file_folder]
        )
        self.scenario_data_tup = self.scenario_generator.load_scenario_from_folders(
            scenario_file_folder,
            scenario_name=name,
            reload_map=reload_map,
            show=show,
            max_number_of_episodes=max_number_of_episodes,
            shuffle_episodes=shuffle,
            merge_scenario_episodes=merge_loaded_scenario_episodes,
        )
        self.scenario_config = self.scenario_data_tup[0][0]["config"]

    def _generate(
        self,
        scenario_config: sc.ScenarioConfig | pathlib.Path | None = None,
        reload_map: bool | None = None,
        max_number_of_episodes: int | None = None,
    ) -> None:
        """Generate new scenario from the input configuration.

        Args:
            scenario_config (Optional[sc.ScenarioConfig | pathlib.Path]): Scenario
                configuration or path to scenario config file.
            reload_map (bool): Whether to reload the scenario map.
            max_number_of_episodes (Optional[int]): Maximum number of episodes to generate.
        """
        if isinstance(scenario_config, pathlib.Path):
            self.scenario_data_tup = self.scenario_generator.generate(
                config_file=scenario_config,
                new_load_of_map_data=reload_map,
                n_episodes=max_number_of_episodes,
                show_plots=False,
            )
        else:
            self.scenario_data_tup = self.scenario_generator.generate(
                config=scenario_config,
                new_load_of_map_data=reload_map,
                n_episodes=max_number_of_episodes,
                show_plots=False,
            )
        self.scenario_config = self.scenario_data_tup[0][0]["config"]

    def _info(
        self, obs: csgym_obs.Observation, action: csgym_action.Action, action_result: csgym_action.ActionResult
    ) -> dict:
        """Returns a dictionary of additional information as defined by the environment.

        Args:
            obs (Observation): Observation vector from the environment
            action (Optional[Action]): Action vector applied by the agent.
            action_result (ActionResult): Result of the action applied by the agent.

        Returns:
            dict: Dictionary of additional information
        """
        self.last_info = {
            "episode_name": self.simulator.sconfig.name,
            "duration": self.simulator.t,
            "timesteps": self.steps,
            "episode_nr": self.episodes,
            "goal_reached": self.simulator.determine_ship_goal_reached(ship_idx=0),
            "collision": self.simulator.determine_ship_collision(ship_idx=0),
            "grounding": self.simulator.determine_ship_grounding(ship_idx=0),
            "distance_to_collision": np.min(self.simulator.distance_to_nearby_vessels(ship_idx=0)),
            "distance_to_grounding": self.simulator.distance_to_grounding(ship_idx=0),
            "actor_failure": not action_result.success,
            "truncated": self._is_truncated(),
            "os_heading": self.ownship.heading,
            "os_speed": self.ownship.speed,
            "os_course": self.ownship.course,
            "reward": self.last_reward,
            "reward_components": self.rewarder.get_last_rewards_as_dict(),
            "render_frame": self.simulator.visualizer.get_live_plot_image(),
            "action": action,
            "unnorm_action": self.action_type.unnormalize(action),
            "os_state": self.ownship.state,
            "actor_info": action_result.info,
            "observation": obs,
            "disturbance": self.simulator.disturbance.get() if self.simulator.disturbance is not None else None,
        }
        return self.last_info

    def seed(self, seed: int | None = None, options: dict | None = None) -> None:
        """Re-seed the environment. This is useful for reproducibility.

        Args:
            seed (Optional[int]): Seed for the random number generator.
            options (Optional[dict]): Options for the environment.
        """
        self._seed = seed if seed is not None else self._seed
        super().reset(seed=None, options=options)
        self._seed = seed if seed is not None else self._seed
        self.scenario_generator.seed(seed=self._seed)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[csgym_obs.Observation, dict]:
        """Reset the environment to a new scenario episode.

        If a scenario config or config file is provided, a new scenario is
        generated. Otherwise, the next episode of the current scenario is used,
        if any.

        Args:
            seed (Optional[int]): Seed for the random number generator.
            options (Optional[dict]): Options for the environment.

        Returns:
            Tuple[Observation, dict]: Initial observation and additional information
        """
        self.seed(seed=seed, options=options)
        # t_now = tracemalloc.take_snapshot()
        # stats = t_now.compare_to(self.t_prev_malloc_snapshot, "traceback")
        # print(f"Memory usage env {self.env_id}:")
        # # for stat in stats[:5]:
        # #     print(stat)

        # for entry in stats[:7]:
        #     print("\nEntry: {}".format(entry))
        #     print("Traceback:")
        #     for line in entry.traceback:
        #         print("  {}".format(line))
        # print("----------------------------------------------------------------------------")

        self.steps = 0  # Actions performed, not necessarily equal to the simulator steps
        self.last_reward = 0.0
        self.done = False

        if self.episodes == self.n_episodes or self.scenario_data_tup[0] == []:
            if self.scenario_file_folder is not None:
                self._load(
                    scenario_file_folder=self.scenario_file_folder,
                    reload_map=self.reload_map,
                    show=False,
                    shuffle=True,
                    merge_loaded_scenario_episodes=True,
                    max_number_of_episodes=self.max_number_of_episodes,
                )
                self.loaded_scenario_data = True
            else:
                self._generate(
                    scenario_config=self.scenario_config,
                    reload_map=self.reload_map,
                    max_number_of_episodes=self.max_number_of_episodes,
                )

        if self.scenario_config is None:
            msg = "Scenario config not initialized!"
            raise ValueError(msg)
        (scenario_episode_list, scenario_enc) = self.scenario_data_tup

        episode_data = scenario_episode_list.pop(0)

        # episode_data["disturbance"].disable_wind()
        self.simulator.initialize_scenario_episode(
            ship_list=episode_data["ship_list"],
            sconfig=episode_data["config"],
            enc=scenario_enc,
            disturbance=episode_data["disturbance"],
            seed=self._seed,
        )
        self.ownship = self.simulator.ownship

        self._define_spaces()
        self._init_render()

        obs = self.observation_type.observe()
        info = self._info(
            obs,
            action=np.zeros(self.action_space.shape[0]),
            action_result=csgym_action.ActionResult(success=True, info={}),
        )

        self.episodes += 1  # Episodes performed
        return obs, info

    def step(self, action: csgym_action.Action) -> tuple[csgym_obs.Observation, float, bool, bool, dict]:
        """Perform an action in the environment.

        Returns the new observation, the reward, whether the task is terminated,
        and additional information.

        Args:
            action (Action): Action vector applied by the agent

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: New observation, reward,
                whether the task is terminated, whether the state is truncated,
                and additional information.
        """
        self.dt_action = self.action_type.get_sampling_time()
        n_steps_between_actions = int(self.dt_action / self.simulator.dt)
        action_kwargs = {
            "applied": False
        }  # Used for action types where it is important to know if the action has been applied
        for _ in range(n_steps_between_actions):
            action_result = self.action_type.act(action, **action_kwargs)
            action_kwargs["applied"] = True

            _ = self.simulator.step(remote_actor=True)

            terminated = self._is_terminated()
            truncated = self._is_truncated()
            if not action_result.success:
                print(f"[{self.env_id.upper()}] Actor failure at t = {self.time}!")

            terminated = terminated or not action_result.success
            if terminated or truncated:
                break

        obs = self.observation_type.observe()
        rewarder_kwargs = {"num_steps": self.steps, "truncated": truncated, "terminated": terminated}
        reward = self.rewarder(obs, action, **rewarder_kwargs)
        self.last_reward = reward

        info = self._info(obs, action, action_result)
        if terminated or truncated:
            self.terminal_info = info

        self.steps += 1
        return obs, reward, terminated, truncated, info

    def _init_render(self) -> None:
        """Initializes the renderer."""
        if self.render_mode == "rgb_array":
            self.simulator.visualizer = None
            self.simulator.visualizer = cssim.viz.Visualizer(config=self.simulator.config.visualizer)
            self.simulator.visualizer.toggle_liveplot_visibility(show=True)
            if self.render_update_rate is not None:
                self.simulator.visualizer.set_update_rate(self.render_update_rate)
            self.simulator.visualizer.init_live_plot(
                self.enc, self.simulator.ship_list, fignum=self.env_id, disable_frame_storage=True
            )
            self.simulator.visualizer.update_live_plot(
                self.simulator.t,
                self.enc,
                self.simulator.ship_list,
                self.simulator.recent_sensor_measurements,
                self.simulator.disturbance.get() if self.simulator.disturbance is not None else None,
                remote_actor=True,
            )

    def render(self) -> np.ndarray | None:
        """Renders the environment in 2D."""
        img = None
        # t_now = time.time()
        if self.render_mode == "rgb_array":
            self.simulator.visualizer.update_live_plot(
                self.simulator.t,
                self.enc,
                self.simulator.ship_list,
                self.simulator.recent_sensor_measurements,
                self.simulator.disturbance.get() if self.simulator.disturbance is not None else None,
                remote_actor=True,
            )
            img = self.simulator.visualizer.get_live_plot_image()
        # print(f"Render time env {self.env_id}: {time.time() - t_now}")
        return img

    @property
    def liveplot_image(self) -> np.ndarray:
        """The current live plot image."""
        if self.simulator.visualizer is not None and self.render_mode == "rgb_array":
            return self.simulator.visualizer.get_live_plot_image()

    @property
    def liveplot_zoom_width(self) -> float:
        """The width of the live plot."""
        return self.simulator.visualizer.zoom_window_width

    @property
    def enc(self) -> senc.ENC:
        """The ENC used for the environment."""
        return self.simulator.enc

    @property
    def time(self) -> float:
        """The current simulation time."""
        return self.simulator.t

    @property
    def time_step(self) -> float:
        """The current simulation time step."""
        return self.simulator.dt

    @property
    def time_truncated(self) -> float:
        """The truncation time (max time)."""
        return self.simulator.t_end

    @property
    def ship_list(self) -> list:
        """The ships in the environment."""
        return self.simulator.ship_list

    @property
    def dynamic_obstacles(self) -> list:
        """The dynamic obstacles in the environment, seen from the own-ship perspective (which has ID 0)."""
        return self.simulator.ship_list[1:]

    @property
    def relevant_grounding_hazards(self) -> list:
        """The nearby ownship grounding hazards in the environment."""
        return self.simulator.relevant_grounding_hazards

    @property
    def relevant_grounding_hazards_as_union(self) -> list:
        """The nearby ownship grounding hazards in the environment as a single multipolygon."""
        return self.simulator.relevant_grounding_hazards_as_union

    @property
    def disturbance(self) -> stoch.Disturbance | None:
        """The current disturbance data."""
        return self.simulator.disturbance
