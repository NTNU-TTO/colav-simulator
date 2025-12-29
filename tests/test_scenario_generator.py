"""Test the scenario generator, showing the main functionalities."""

import colav_simulator.common.paths as dp
from colav_simulator.scenario_generator import ScenarioGenerator


def test_scenario_generator() -> None:
    scenario_generator = ScenarioGenerator(seed=0)
    scenario_name = "rlmpc_scenario_ms_channel"

    scenario_generator.generate(
        config_file=dp.scenarios / (scenario_name + ".yaml"),
        new_load_of_map_data=True,
        save_scenario=True,
        save_scenario_folder=dp.scenarios / "saved" / scenario_name,
        show_plots=False,
        episode_idx_save_offset=0,
        n_episodes=20,
        delete_existing_files=True,
    )

    scenario_generator.load_scenario_from_folders(
        folder=dp.scenarios / "saved" / scenario_name,
        scenario_name=scenario_name,
        reload_map=False,
        max_number_of_episodes=1000,
        shuffle_episodes=False,
        show=False,
    )


if __name__ == "__main__":
    test_scenario_generator()
