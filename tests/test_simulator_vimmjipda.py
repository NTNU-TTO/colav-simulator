"""
Test module for the Simulator class.

Shows how to use the simulator with a colav system and the VIMMJIPDA tracker.

Requires installation of the vimmjipda package at https://github.com/NTNU-Autoship-Internal/vimmjipda
"""

from pathlib import Path

import vimmjipda.vimmjipda_tracker_interface as vti

import colav_simulator.common.paths as dp
import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.scenario_generator as sg
import colav_simulator.simulator as sim


def test_simulator_vimmjipda() -> None:
    vimmjipda_config_path = (
        Path.home() / "Desktop/autotuning/autotuning/vimmjipda/config/vimmjipda.yaml"
    )  # Path to the vimmjipda.yaml file, modify for your system
    vimmjipda_params = vti.VIMMJIPDAParams.from_yaml(vimmjipda_config_path)
    vimmjipda_tracker = vti.VIMMJIPDA(params=vimmjipda_params)
    sbmpc_obj = ci.SBMPCWrapper()

    scenario_generator = sg.ScenarioGenerator()
    simconfig = sim.Config.from_file(dp.simulator_config)
    simconfig.visualizer.zoom_window_width = 1500.0
    simconfig.visualizer.show_liveplot_target_tracks = True
    simconfig.visualizer.show_liveplot_measurements = True
    simulator = sim.Simulator(config=simconfig)
    simulator.toggle_liveplot_visibility(True)

    scenario_data_list = scenario_generator.generate_configured_scenarios()
    output1 = simulator.run(
        scenario_data_list,
        colav_systems=[(0, sbmpc_obj)],
        trackers=[(0, vimmjipda_tracker)],
    )

    scenario_name = "rlmpc_scenario_ms_channel_vimmjipda"
    scenario_data = scenario_generator.generate(
        config_file=dp.scenarios / (scenario_name + ".yaml"),
        new_load_of_map_data=True,
        save_scenario=True,
        save_scenario_folder=dp.scenarios / "saved" / scenario_name,
        show_plots=False,
        episode_idx_save_offset=0,
        n_episodes=1,
        delete_existing_files=True,
    )
    scenario_data_list = [scenario_data]
    output2 = simulator.run(
        scenario_data_list,
        colav_systems=[(0, sbmpc_obj)],
        trackers=[(0, vimmjipda_tracker)],
    )


if __name__ == "__main__":
    test_simulator_vimmjipda()
