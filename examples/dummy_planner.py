"""Demonstrates how to use an external planning algorithm with the colav-simulator.

Author: Trym Tengesdal
"""

from dataclasses import dataclass, field

import matplotlib as mpl
import matplotlib.pyplot as plt

import colav_simulator.common.paths as dp
import colav_simulator.core.colav.colav_interface as ci
from colav_simulator.core import guidances, stochasticity

mpl.use("Agg")
import numpy as np
import seacharts.enc as senc

from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator


@dataclass
class DummyPlannerParams:
    los: guidances.LOSGuidanceParams = field(
        default_factory=lambda: guidances.LOSGuidanceParams(
            K_p=0.035, K_i=0.0, pass_angle_threshold=90.0, R_a=25.0, max_cross_track_error_int=30.0
        )
    )


class DummyPlanner(ci.ICOLAV):
    """Dummy planner."""

    def __init__(self, config: DummyPlannerParams) -> None:
        self._config = config
        self._los = guidances.LOSGuidance(config.los)
        self._references = np.zeros((9, 1))
        self._t_prev = 0.0

    def reset(self) -> None:  # noqa: D102
        self._references = np.zeros((9, 1))
        self._t_prev = 0.0
        self._los.reset()

    def plan(  # noqa: D102
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list[tuple[int, np.ndarray, np.ndarray, float, float]],  # noqa: ARG002
        enc: senc.ENC | None = None,  # noqa: ARG002
        goal_state: np.ndarray | None = None,  # noqa: ARG002
        w: stochasticity.DisturbanceData | None = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> np.ndarray:
        assert waypoints is not None, "Waypoints must be provided to the dummy planner"  # noqa: S101
        assert speed_plan is not None, "Speed plan must be provided to the dummy planner"  # noqa: S101

        # Insert all your fancy planning here

        self._references = self._los.compute_references(
            waypoints, speed_plan, times=None, xs=ownship_state, dt=t - self._t_prev
        )
        self._t_prev = t
        return self._references

    def get_current_plan(self) -> np.ndarray:  # noqa: D102
        return self._references

    def get_colav_data(self) -> dict:  # noqa: D102
        return {
            "nominal_trajectory": self._references[:6, :],
            "nominal_inputs": np.zeros((3, 1)),
            "params": self._config.los,
            "t": self._t_prev,
        }

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:  # noqa: ARG002, D102
        return plt_handles


if __name__ == "__main__":
    dummy_planner = DummyPlanner(DummyPlannerParams())

    scenario_file = dp.scenarios / "head_on.yaml"
    scenario_generator = ScenarioGenerator()
    scenario_data = scenario_generator.generate(config_file=scenario_file, new_load_of_map_data=True)
    simulator = Simulator()
    simulator.toggle_liveplot_visibility(True)
    output = simulator.run([scenario_data], colav_systems=[(0, dummy_planner)])
    print("done")
