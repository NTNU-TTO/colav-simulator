"""Test module for gym.py.

Shows how to use the gym environment, and how to save a video + gif of the simulation.
"""

import gymnasium as gym
import numpy as np

import colav_simulator.common.image_helper_methods as ihm
import colav_simulator.common.paths as dp
from colav_simulator.gym.environment import COLAVEnvironment  # noqa: F401


def test_gym() -> None:
    config_file = dp.scenarios / "rl_scenario_smaller.yaml"

    # scenario_generator = ScenarioGenerator(seed=0)
    # scenario_data = scenario_generator.generate(
    #     config_file=config_file,
    #     new_load_of_map_data=True,
    #     save_scenario=True,
    #     save_scenario_folder=dp.scenarios / "test_data" / "rl_scenario_smaller",
    #     show_plots=True,
    #     episode_idx_save_offset=0,
    #     delete_existing_files=True,
    # )

    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_config": config_file,
        "reload_map": True,
        "render_mode": "rgb_array",
        "render_update_rate": 0.5,
        "seed": 1,
    }
    env = gym.make(id=env_id, **env_config)

    record = False
    if record:
        video_path = dp.animation_output / "demo.mp4"
        env = gym.wrappers.RecordVideo(env, video_path.as_posix(), episode_trigger=lambda x: x == 0)

    env.reset(seed=1)
    frames = []
    for _i in range(100):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))

        frames.append(env.render())

        if terminated or truncated:
            env.reset()

    env.close()

    save_gif = False
    if save_gif:
        ihm.save_frames_as_gif(frames, dp.animation_output / "demo.gif")


if __name__ == "__main__":
    test_gym()
