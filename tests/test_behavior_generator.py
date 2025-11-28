"""
Test file showing how the behavior generator can be used to generate scenarios.
"""

import colav_simulator.common.paths as dp
from colav_simulator.behavior_generator import (
    BehaviorGenerationMethod,
    RRTBehaviorSamplingMethod,
)
from colav_simulator.scenario_generator import Config, ScenarioGenerator


def test_behavior_generator() -> None:
    sg_config = Config()
    sg_config.manual_episode_accept = False
    sg_config.behavior_generator.ownship_method = (
        BehaviorGenerationMethod.ConstantSpeedAndCourse
    )
    sg_config.behavior_generator.target_ship_method = (
        BehaviorGenerationMethod.RRTStar
    )  # NOTE: Remember to install the rrt-star-lib package first.
    sg_config.behavior_generator.target_ship_rrt_behavior_sampling_method = (
        RRTBehaviorSamplingMethod.OwnshipWaypointCorridor
    )
    sg_config.behavior_generator.rrtstar.params.max_time = 10.0
    sg_config.verbose = True
    scenario_generator = ScenarioGenerator(config=sg_config)
    scenario_generator.seed(12)

    scenario_episode_list, scenario_enc = scenario_generator.generate(
        config_file=dp.scenarios / "boknafjorden_generation_test.yaml",
        new_load_of_map_data=True,
        show_plots=False,
        n_episodes=1,
    )


if __name__ == "__main__":
    test_behavior_generator()
