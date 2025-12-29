"""Contains class for logging data from the COLAV environment.

Author: Trym Tengesdal
"""

import pickle
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import scipy.ndimage as scimg

import colav_simulator.common.miscellaneous_helper_methods as mhm


class EpisodeData(NamedTuple):
    """Data for a single episode in the COLAV environment."""

    name: str
    timestamp: str
    timesteps: int
    duration: float
    distances_to_grounding: np.ndarray
    distances_to_collision: np.ndarray
    goal_reached: bool
    collision: bool
    grounding: bool
    truncated: bool
    rewards: np.ndarray
    cumulative_reward: float
    mean_reward: float
    std_reward: float
    reward_components: list[dict[str, float]]
    frames: list[np.ndarray] | None
    actions: list[np.ndarray]
    obs: list[np.ndarray]
    ownship_states: list[np.ndarray]
    actor_infos: list[dict[str, Any]]
    actor_failure: bool = False


class Logger:
    """Logs data from the COLAV environment. Supports saving and loading to/from pickle files.

    Args:
        experiment_name (str): The name of the experiment.
        log_dir (Path): The directory where the log files are saved.
        save_freq (int, optional): The frequency (in episodes) of saving the data to a pickle file.
        n_envs (int, optional): The number of environments.
        max_num_logged_episodes (int, optional): The maximum number of episodes to log before saving and resetting.
        minimal_logging (bool, optional): Whether to log the bare minimum (rewards, collisions,
            groundings, goal reached, and truncated).
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: Path,
        n_envs: int = 1,
        max_num_logged_episodes: int = 100,
        minimal_logging: bool = False,
    ) -> None:
        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        self.max_num_logged_episodes: int = max_num_logged_episodes
        self.minimal_logging: bool = minimal_logging
        self.env_data: list[EpisodeData] = []
        self.pos: int = 0
        self.prev_pos: int = 0
        self.reset_logger()
        self.experiment_name: str = experiment_name
        self.log_dir: Path = log_dir
        self.img_dir: Path = log_dir / "terminal_frames"
        if not self.img_dir.exists():
            self.img_dir.mkdir(parents=True)
        self.store_actor_info: bool = True

        self.log_frame_freq: int = 10
        self.log_count: list[int] = [0 for _ in range(n_envs)]

        self.episode_name: list[str] = ["" for _ in range(n_envs)]
        self.duration: list[float] = [0.0 for _ in range(n_envs)]
        self.rewards: list[float] = [[] for _ in range(n_envs)]
        self.timesteps: list[int] = [0 for _ in range(n_envs)]
        self.prev_episode_nr: int = -1
        self.episode_nr: int = 0

        self.reward_components: list[list[dict[str, float]]] = [[] for _ in range(n_envs)]
        self.collision: list[bool] = [False for _ in range(n_envs)]
        self.grounding: list[bool] = [False for _ in range(n_envs)]
        self.goal_reached: list[bool] = [False for _ in range(n_envs)]
        self.truncated: list[bool] = [False for _ in range(n_envs)]
        self.frames: list[list[np.ndarray]] = [[] for _ in range(n_envs)]
        self.distances_to_grounding: list[list[float]] = [[] for _ in range(n_envs)]
        self.distances_to_collision: list[list[float]] = [[] for _ in range(n_envs)]
        self.actions: list[list[np.ndarray]] = [[] for _ in range(n_envs)]
        self.obs: list[list[np.ndarray | dict[str, np.ndarray]]] = [[] for _ in range(n_envs)]
        self.ownship_states: list[list[np.ndarray]] = [[] for _ in range(n_envs)]
        self.actor_infos: list[list[dict[str, Any]]] = [[] for _ in range(n_envs)]
        self.actor_failure: list[bool] = [False for _ in range(n_envs)]

    def save_as_pickle(self, name: str | None = None) -> None:
        """Saves the environment data to a pickle file.

        Args:
            name (Optional[str]): The name of the pickle file, without the .pkl extension.
        """
        if name is None:
            name = self.experiment_name + "_env_training_data"

        # Don't save if there is no data or no new data
        if self.pos in (0, self.prev_pos):
            return

        with open(self.log_dir / (name + ".pkl"), "ba") as f:
            pickle.dump(self.env_data[: self.pos], f)

        self.reset_logger()

    def load_from_pickle(self, name: str | None) -> None:
        """Loads the environment data from a pickle file.

        Args:
            name (Optional[str]): Name of the pickle file, without the .pkl extension.
        """
        if name is None:
            name = self.experiment_name
        # Because multiple objects might be stored in the same file
        # we need to load them one by one
        self.env_data = []
        with open(self.log_dir / (name + ".pkl"), "rb") as f:
            while 1:
                try:
                    edata = pickle.load(f)
                    self.env_data.extend(edata)
                except EOFError:
                    break

        self.env_data = list(dict(enumerate(self.env_data)).values())
        # print(f"Loaded {len(self.env_data)} episodes from {name}.pkl")

    def __call__(self, cs_env_infos: list[dict[str, Any]]) -> None:
        """Logs data from the input env info dictionary.

        Args:
            cs_env_infos (List[dict[str, Any]]): List of environment info dictionaries for the COLAV environment(s).
        """
        for env_idx, info in enumerate(cs_env_infos):
            self._log_env_info(info, env_idx)

    def _dump_last_frame_as_img(self, frame: np.ndarray, env_idx: int) -> None:
        """Dumps the last frame of the episode as an image.

        Args:
            frame (np.ndarray): The frame to dump.
            env_idx (int): The index of the environment.
        """
        ctxt = ""
        if self.collision[env_idx]:
            ctxt = "collision"
        elif self.grounding[env_idx]:
            ctxt = "grounding"
        elif self.goal_reached[env_idx]:
            ctxt = "goal_reached"
        elif self.truncated[env_idx]:
            ctxt = "truncated"
        elif self.actor_failure[env_idx]:
            ctxt = "actor_failure"
        img_name = f"{self.episode_name[env_idx]}_env{env_idx}_ep{self.episode_nr}_{ctxt}_tf.png"

        img_path = self.img_dir / img_name
        cv2.imwrite(str(img_path), frame)

    def _log_env_info(self, info: dict[str, Any], env_idx: int) -> None:  # noqa: PLR0915
        """Logs the environment info to the logger.

        Args:
            info (dict[str, Any]): The environment info dictionary.
            env_idx (int): The index of the environment.
        """
        if "episode_name" not in info:
            print(f"Episode name not found in info dictionary!: Info = {info}")
            return

        self.episode_name[env_idx] = info["episode_name"]
        self.duration[env_idx] = info["duration"]
        self.timesteps[env_idx] = info["timesteps"]

        self.rewards[env_idx].append(info["reward"])
        self.collision[env_idx] = info["collision"]
        self.grounding[env_idx] = info["grounding"]
        self.goal_reached[env_idx] = info["goal_reached"]
        self.truncated[env_idx] = info["truncated"]
        self.actor_failure[env_idx] = info["actor_failure"]
        self.reward_components[env_idx].append(info["reward_components"])

        done = (
            self.collision[env_idx]
            or self.grounding[env_idx]
            or self.goal_reached[env_idx]
            or self.truncated[env_idx]
            or self.actor_failure[env_idx]
        )

        if not self.minimal_logging:
            self.distances_to_collision[env_idx].append(info["distance_to_collision"])
            self.distances_to_grounding[env_idx].append(info["distance_to_grounding"])
            self.ownship_states[env_idx].append(info["os_state"])
            # self.actions[env_idx].append(info["unnorm_action"])
            # self.obs[env_idx].append(info["unnorm_observation"])

        if (
            not self.minimal_logging
            and info["render_frame"] is not None
            and info["render_frame"].size > 10
            and ((self.log_count[env_idx] % self.log_frame_freq == 0) or done)
        ):
            stored_frame = info["render_frame"].copy()

            rotate = False
            if rotate:
                stored_frame = scimg.rotate(stored_frame, np.rad2deg(info["os_course"]), reshape=False)

            npx, npy = stored_frame.shape[:2]
            center_pixel_x = int(stored_frame.shape[0] // 2)
            center_pixel_y = int(stored_frame.shape[1] // 2)
            if rotate:
                cutoff_index_below_vessel = int(0.1 * npx)  # corresponds to 100 m for a 1200 m zoom width
                cutoff_index_above_vessel = int(0.4 * npx)
                cutoff_laterally = int(0.25 * npy)
            else:
                cutoff_index_below_vessel = int(0.3 * npx)
                cutoff_index_above_vessel = int(0.3 * npx)
                cutoff_laterally = int(0.3 * npy)
            cutoff_laterally = cutoff_laterally if cutoff_laterally <= center_pixel_y else center_pixel_y
            cutoff_index_below_vessel = (
                cutoff_index_below_vessel if cutoff_index_below_vessel <= center_pixel_y else center_pixel_y
            )
            cutoff_index_above_vessel = (
                cutoff_index_above_vessel if cutoff_index_above_vessel <= center_pixel_x else center_pixel_x
            )
            cropped_img = stored_frame[
                center_pixel_x - cutoff_index_above_vessel : center_pixel_x + cutoff_index_below_vessel,
                center_pixel_y - cutoff_laterally : center_pixel_y + cutoff_laterally,
            ]
            reduced_frame = cv2.resize(cropped_img, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            # plotters.plot_image(image=reduced_frame, title="Reduced frame for logging")
            self.frames[env_idx].append(reduced_frame)

            # if done:
            #     self._dump_last_frame_as_img(reduced_frame, env_idx)

        # Special case for an NMPC actor
        if not self.minimal_logging and self.store_actor_info and "actor_info" in info and "optimal" in info["actor_info"]:
            actor_info = info["actor_info"]
            stored_actor_info = {}
            stored_actor_info["mpc_runtime"] = actor_info["runtime"]
            stored_actor_info["optimal"] = actor_info["optimal"]
            stored_actor_info["dnn_input_features"] = actor_info["dnn_input_features"]
            stored_actor_info["old_mpc_params"] = actor_info["old_mpc_params"]
            stored_actor_info["new_mpc_params"] = actor_info["new_mpc_params"]
            stored_actor_info["norm_mpc_action"] = actor_info["norm_mpc_action"]
            stored_actor_info["applied_refs"] = actor_info["applied_refs"]
            self.actor_infos[env_idx].append(stored_actor_info)

        self.log_count[env_idx] += 1

        if done:
            self.add_episode_data(env_idx)
            self.reset_data_structures(env_idx)

    def add_episode_data(self, env_idx: int) -> None:
        """Adds the data from the current episode to the environment data list.

        Args:
            env_idx (int): The index of the environment.
        """
        # Don't log if there is only one timestep
        if len(self.rewards[env_idx]) == 1 or self.episode_nr == self.prev_episode_nr:
            return

        datetime_t = mhm.utc_timestamp_to_datetime(mhm.current_utc_timestamp())
        datetime_str = datetime_t.strftime("%d.%m.%Y %H:%M:%S")

        episode_data = EpisodeData(
            name=self.episode_name[env_idx],
            timestamp=datetime_str,
            timesteps=self.timesteps[env_idx],
            duration=self.duration[env_idx],
            rewards=np.array(self.rewards[env_idx], dtype=np.float32),
            cumulative_reward=np.sum(self.rewards[env_idx], dtype=np.float32),
            mean_reward=np.mean(self.rewards[env_idx], dtype=np.float32),
            std_reward=np.std(self.rewards[env_idx], dtype=np.float32),
            reward_components=self.reward_components[env_idx],
            goal_reached=self.goal_reached[env_idx],
            collision=self.collision[env_idx],
            grounding=self.grounding[env_idx],
            truncated=self.truncated[env_idx],
            distances_to_grounding=np.array(self.distances_to_grounding[env_idx], dtype=np.float32),
            distances_to_collision=np.array(self.distances_to_collision[env_idx], dtype=np.float32),
            actions=np.array(self.actions[env_idx], dtype=np.float32),
            obs=self.obs[env_idx],
            ownship_states=np.array(self.ownship_states[env_idx], dtype=np.float32),
            frames=np.array(self.frames[env_idx], dtype=np.uint8),
            actor_infos=self.actor_infos[env_idx],
            actor_failure=self.actor_failure[env_idx],
        )
        self.env_data[self.pos] = episode_data
        self.prev_episode_nr = self.episode_nr
        self.pos += 1
        self.episode_nr += 1
        # mhm.print_process_memory_usage(prefix_str=f"Env {env_idx} | Episode {self.episode_nr} |")
        if self.pos >= self.max_num_logged_episodes:
            self.save_as_pickle()

    def reset_logger(self) -> None:
        """Resets the logger."""
        self.env_data: list[EpisodeData] = [
            EpisodeData(
                name="",
                timestamp="",
                timesteps=0,
                duration=0.0,
                distances_to_grounding=np.array([], dtype=np.float32),
                distances_to_collision=np.array([], dtype=np.float32),
                goal_reached=False,
                collision=False,
                grounding=False,
                truncated=False,
                rewards=np.array([], dtype=np.float32),
                cumulative_reward=0.0,
                mean_reward=0.0,
                std_reward=0.0,
                reward_components=[],
                frames=[],
                actions=[],
                obs=[],
                ownship_states=[],
                actor_infos=[],
                actor_failure=False,
            )
        ] * self.max_num_logged_episodes
        self.pos = 0
        self.prev_pos = 0

    def reset_data_structures(self, env_idx: int) -> None:
        """Resets the data structures in preparation of new episode.

        Args:
            env_idx (int): The index of the environment.
        """
        self.log_count[env_idx] = 0
        self.timesteps[env_idx] = 0
        self.duration[env_idx] = 0.0
        self.rewards[env_idx] = []
        self.reward_components[env_idx] = []
        self.goal_reached[env_idx] = False
        self.collision[env_idx] = False
        self.grounding[env_idx] = False
        self.truncated[env_idx] = False
        self.distances_to_grounding[env_idx] = []
        self.distances_to_collision[env_idx] = []
        self.actions[env_idx] = []
        self.obs[env_idx] = []
        self.ownship_states[env_idx] = []
        self.frames[env_idx] = []
        self.actor_infos[env_idx] = []
        self.actor_failure[env_idx] = False


if __name__ == "__main__":
    experiment_name = "sac_rlmpc1"
    log_dir = Path.home() / "Desktop" / "machine_learning" / "rlmpc" / experiment_name
    logger = Logger(experiment_name=experiment_name, log_dir=log_dir)
    logger.load_from_pickle(f"{experiment_name}_env_training_data")

    print(f"Ep0: {logger.env_data[0]}")
