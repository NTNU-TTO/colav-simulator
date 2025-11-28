import tempfile
from pathlib import Path
from unittest.mock import patch

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.file_utils as fu
import colav_simulator.common.paths as dp
from colav_simulator.scenario_config import ScenarioConfig


def test_parse_simple_scenario():
    config_file = dp.scenarios / "head_on.yaml"
    config = cp.extract(ScenarioConfig, config_file, dp.scenario_schema)
    assert config.name == "head_on1"
    assert config.type.name == "HO"
    assert config.n_random_ships == 1
    assert len(config.ship_list) == 2


def test_parse_rl_scenario():
    config_file = dp.scenarios / "rl_scenario.yaml"
    config = cp.extract(ScenarioConfig, config_file, dp.scenario_schema)
    assert config.name == "rl_scenario"
    assert config.rl is not None
    assert config.rl.action_type == "relative_course_speed_reference_sequence_action"
    assert config.stochasticity is not None
    assert config.episode_generation.n_episodes == 1


def test_yaml_round_trip_simple():
    config_file = dp.scenarios / "crossing_give_way.yaml"
    original_dict = fu.read_yaml_into_dict(config_file)
    config = cp.extract(ScenarioConfig, config_file, dp.scenario_schema)
    round_trip_dict = config.to_dict()

    assert round_trip_dict["name"] == original_dict["name"]
    assert round_trip_dict["type"] == original_dict["type"]
    assert round_trip_dict["utm_zone"] == original_dict["utm_zone"]
    assert round_trip_dict["t_start"] == original_dict["t_start"]
    assert round_trip_dict["t_end"] == original_dict["t_end"]
    assert round_trip_dict["dt_sim"] == original_dict["dt_sim"]
    assert (
        round_trip_dict["new_load_of_map_data"] == original_dict["new_load_of_map_data"]
    )
    assert len(round_trip_dict["map_data_files"]) == len(
        original_dict["map_data_files"]
    )
    for resolved_path, original_path in zip(
        round_trip_dict["map_data_files"], original_dict["map_data_files"]
    ):
        assert Path(resolved_path).name == Path(original_path).name


def test_yaml_round_trip_with_optional_fields():
    config_file = dp.scenarios / "rl_scenario.yaml"
    original_dict = fu.read_yaml_into_dict(config_file)
    config: ScenarioConfig = cp.extract(ScenarioConfig, config_file, dp.scenario_schema)
    round_trip_dict = config.to_dict()

    assert (
        round_trip_dict["episode_generation"]["n_episodes"]
        == original_dict["episode_generation"]["n_episodes"]
    )
    assert (
        round_trip_dict["episode_generation"]["ownship_position_generation"]
        == original_dict["episode_generation"]["ownship_position_generation"]
    )
    assert round_trip_dict["rl"]["action_type"] == original_dict["rl"]["action_type"]
    assert round_trip_dict["stochasticity"] is not None
    assert original_dict["stochasticity"] is not None


def test_handle_map_data_files_relative_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        enc_data_dir = tmp_path / "enc_data"
        enc_data_dir.mkdir()
        test_file = enc_data_dir / "test_relative.gdb"
        test_file.touch()

        with patch("colav_simulator.scenario_config.Path.home", return_value=tmp_path):
            result = ScenarioConfig.handle_map_data_files(["test_relative.gdb"])
            assert len(result) == 1
            assert Path(result[0]).is_absolute()
            assert Path(result[0]).parent == enc_data_dir
            assert Path(result[0]).name == "test_relative.gdb"


def test_handle_map_data_files_absolute_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_absolute.gdb"
        test_file.touch()

        result = ScenarioConfig.handle_map_data_files([str(test_file)])
        assert len(result) == 1
        assert result[0] == str(test_file)
        assert Path(result[0]).is_absolute()
