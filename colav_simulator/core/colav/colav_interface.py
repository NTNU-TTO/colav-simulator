"""colav_interface.py.

Summary:
Contains the interface used by all COLAV planning algorithms that
wants to be run with the COLAV simulator.

To add a new COLAV planning algorithm internally to the simulator:

1: Import necessary algorithm modules in this file.
2: Add the algorithm name as a type to the COLAVType enum.
3: Add the algorithm as an optional entry to the LayerConfig class.
4: Create a new wrapper class for your COLAV algorithm, which implements
   (inherits as this is python) the ICOLAV interface. It should take in a Config
   object as input.
5: Add an entry in the COLAVBuilder class, which builds it from config if the
   type matches. See an example for the Kuwata VO and SBMPC below.
6: Add configuration support for the algorithm by expanding the `colav` entry
   under `schemas/scenario.yaml` in the `ship_list` section.

Alternatively (AND EASIER), to be able to use a third-party COLAV planning algorithm:

1: Import this module in your own code.
2: Create a wrapper class for your COLAV algorithm that implements the ICOLAV interface.
3: Provide your third-party algorithm to the simulator at run-time (see Simulator class in simulator.py).

Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import seacharts.enc as senc

import colav_simulator.common.config_parsing as cp
import colav_simulator.core.colav.kuwata_vo_alg.kuwata_vo as kvo
import colav_simulator.core.colav.sbmpc.sbmpc as sb_mpc
import colav_simulator.core.guidances as guidance
from colav_simulator.core import stochasticity


class COLAVType(Enum):
    """Enum for the different COLAV algorithms currently compatible with the simulator."""

    VO = 0  # Kuwata VO, with LOS guidance to provide velocity references.
    SBMPC = 1  # SB-MPC, provide trajectory offsets


@dataclass
class LayerConfig:
    """Configuration class for the parameters of a single layer/algorithm.

    In the COLAV planning hierarchy, each layer represents a specific COLAV
    algorithm, and the parameters are specific to the algorithm. For example
    with three layers:

    - The first layer will be a static obstacle collision-free planner, run
      e.g. only at the start of the mission.
    - The second layer is a mid-level MPC-based COLAV system, that can handle
      both static and dynamic obstacles (and the COLREGS).
    - The third layer is a lower level reactive VO-based COLAV, that handles
      emergency maneuvers and close encounters if the mid-level planner fails.

    NOTE: This class is typically only used when you want to configure the
          COLAV system parameters from a scenario file. However, an easier
          option is to configure the COLAV system externally, and pass the
          COLAV object to the simulator at run-time (see examples/dummy_planner.py
          for an example of this). This is recommended if you want to use a
          third-party COLAV algorithm.
    """

    vo: kvo.VOParams | None = field(default_factory=lambda: kvo.VOParams())
    los: guidance.LOSGuidanceParams | None = None
    sbmpc: sb_mpc.SBMPCParams | None = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LayerConfig":  # noqa: D102
        config = LayerConfig()
        if "vo" in config_dict:
            config.vo = cp.convert_settings_dict_to_dataclass(kvo.VOParams, config_dict["vo"])

        if "los" in config_dict:
            config.los = cp.convert_settings_dict_to_dataclass(guidance.LOSGuidanceParams, config_dict["los"])

        if "sbmpc" in config_dict:
            config.sbmpc = cp.convert_settings_dict_to_dataclass(sb_mpc.SBMPCParams, config_dict["sbmpc"])

        return config

    def to_dict(self) -> dict:  # noqa: D102
        config_dict = {}

        if self.vo is not None:
            config_dict["vo"] = self.vo.to_dict()

        if self.los is not None:
            config_dict["los"] = self.los.to_dict()

        if self.sbmpc is not None:
            config_dict["sbmpc"] = self.sbmpc.to_dict()

        return config_dict


@dataclass
class Config:
    """Configuration class for managing COLAV system parameters for all considered layers in the COLAV hierarchy."""

    name: COLAVType = COLAVType.VO
    layer1: LayerConfig = field(default_factory=lambda: LayerConfig())
    layer2: LayerConfig | None = None
    layer3: LayerConfig | None = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":  # noqa: D102
        config = Config(name=COLAVType[config_dict["name"]], layer1=LayerConfig.from_dict(config_dict["layer1"]))

        if "layer2" in config_dict:
            config.layer2 = LayerConfig.from_dict(config_dict["layer2"])

        if "layer3" in config_dict:
            config.layer3 = LayerConfig.from_dict(config_dict["layer3"])

        return config

    def to_dict(self) -> dict:  # noqa: D102
        config_dict = {"name": self.name.name, "layer1": self.layer1.to_dict()}

        if self.layer2 is not None:
            config_dict["layer2"] = self.layer2.to_dict()

        if self.layer3 is not None:
            config_dict["layer3"] = self.layer3.to_dict()

        return config_dict


class ICOLAV(ABC):
    @abstractmethod
    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list[tuple[int, np.ndarray, np.ndarray, float, float]],
        enc: senc.ENC | None = None,
        goal_state: np.ndarray | None = None,  # noqa: ARG002
        w: stochasticity.DisturbanceData | None = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> np.ndarray:
        """Main COLAV planning function.

        Args:
            t (float): The current time since the start of the simulation.
            waypoints (np.ndarray): The waypoints to follow, typically used for
                COLAV planners assuming a nominal path/trajectory as input.
                Dimensions: [2, N] composed of the waypoint NE coordinates.
            speed_plan (np.ndarray): Reference speeds at each waypoint, typically
                used for COLAV planners assuming a nominal path/trajectory as
                input.
            ownship_state (np.ndarray): The ownship state [x, y, psi, u, v, r]^T.
                Used as start state in case of high level planners.
            do_list (list[tuple[int, np.ndarray, np.ndarray, float, float]]): List
                of dynamic obstacles in the vicinity of the ship, on the format
                (ID, state, covariance, length, width). The state is on the
                format [x, y, Vx, Vy]^T.
            enc (senc.ENC | None): The relevant Electronic Navigational Chart
                (ENC) for static obstacle info.
            goal_state (np.ndarray | None): The goal state [x, y, psi, u, v, r]^T,
                typically used for high level COLAV planners where no nominal
                path/trajectory is assumed.
            w (stochasticity.DisturbanceData | None): The stochastic disturbance
                data.
            **kwargs: Additional arguments to the COLAV planning algorithm, e.g.
                the own-ship length.

        Returns:
            np.ndarray: The planned poses, velocities and accelerations
                (vstacked as a 9 x N array, N >= 1 being the number of samples)
                from the COLAV planning algorithm. Must be compatible with the
                control system you are using.
        """

    @abstractmethod
    def reset(self):
        """Resets the COLAV planning algorithm to its initial state."""

    @abstractmethod
    def get_current_plan(self) -> np.ndarray:
        """Returns the current planned trajectory.

        Returns:
            np.ndarray: The most recent planned poses, velocities and
                accelerations (vstacked as a 9 x N array, N >= 1 being the
                number of samples) over the COLAV planning horizon (if any).
                Must be compatible with the control system you are using.
        """

    @abstractmethod
    def get_colav_data(self) -> dict:
        """Returns the plotting data relevant for the COLAV planning algorithm.

        This includes e.g. the predicted trajectory, considered obstacles,
        optimal inputs etc. Used for plotting and logging.

        Returns:
            dict: The relevant data used in the COLAV planning algorithm.
        """

    @abstractmethod
    def plot_results(
        self,
        ax_map: plt.Axes,
        enc: senc.ENC,
        plt_handles: dict,
        **kwargs,  # noqa: ARG002
    ) -> dict:
        """Plots the COLAV planning algorithm results data.

        E.g. the predicted trajectory, considered obstacles, optimal inputs
        etc.

        Args:
            ax_map (plt.Axes): Map axes to plot on.
            enc (senc.ENC): ENC object.
            plt_handles (dict): Dictionary of plot handles.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary of plot handles.
        """


class VOWrapper(ICOLAV):
    """The VO wrapper is a Kuwata VO-based reactive COLAV planning system.

    Where LOS-guidance is used to provide velocity references.
    """

    def __init__(self, config: Config, **kwargs) -> None:  # noqa: ARG002
        if config.layer1.vo is None:
            msg = "Kuwata VO must be on the first layer for the VO wrapper."
            raise ValueError(msg)
        self._vo = kvo.VO(config.layer1.vo)

        if not (config.layer2 and config.layer2.los is not None):
            msg = "LOS guidance must be on the second layer for the VO wrapper."
            raise ValueError(msg)
        self._los = guidance.LOSGuidance(config.layer2.los)

        self._t_prev = 0.0
        self._initialized = False

    def reset(self):
        """Resets the VO-COLAV to its initial state."""
        self._t_prev = 0.0
        self._initialized = False
        self._los.reset()

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list[tuple[int, np.ndarray, np.ndarray, float, float]],
        enc: senc.ENC | None = None,
        goal_state: np.ndarray | None = None,  # noqa: ARG002
        w: stochasticity.DisturbanceData | None = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> np.ndarray:
        if not self._initialized:
            self._t_prev = t
            self._initialized = True

        references = self._los.compute_references(waypoints, speed_plan, None, ownship_state, t - self._t_prev)
        self._t_prev = t
        course_ref = references[2, 0]
        speed_ref = references[3, 0]
        vel_ref = np.array([speed_ref * np.cos(course_ref), speed_ref * np.sin(course_ref)])
        return self._vo.plan(t, vel_ref, ownship_state, do_list, enc)

    def get_current_plan(self) -> np.ndarray:
        return self._vo.get_current_plan()

    def get_colav_data(self) -> dict:
        return {}

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:  # noqa: ARG002
        return plt_handles


class SBMPCWrapper(ICOLAV):
    """SBMPC is here implemented as a COLAV planning algorithm that provides trajectory offsets to the nominal LOS guidance.

    NOTE: No land consideration is added in this implementation.
    """

    def __init__(
        self,
        config: Config = Config(
            name=COLAVType.SBMPC,
            layer1=LayerConfig(sbmpc=sb_mpc.SBMPCParams()),
            layer2=LayerConfig(los=guidance.LOSGuidanceParams()),
        ),
        **kwargs,  # noqa: ARG002
    ) -> None:
        if config.layer1.sbmpc is None:
            msg = "SBMPC must be on the first layer for the SBMPC wrapper."
            raise ValueError(msg)
        self._sbmpc = sb_mpc.SBMPC(config.layer1.sbmpc)

        if config.layer2.los is None:
            msg = "LOS guidance must be on the second layer for the SBMPC wrapper."
            raise ValueError(msg)
        self._los = guidance.LOSGuidance(config.layer2.los)

        self._t_prev = 0.0
        self._initialized = False
        self._t_run_sbmpc_last = 0.0
        self._speed_os_best = 1.0
        self._course_os_best = 0.0

    def reset(self):
        """Resets the SBMPC-COLAV to its initial state."""
        self._t_prev = 0.0
        self._initialized = False
        self._t_run_sbmpc_last = 0.0
        self._speed_os_best = 1.0
        self._course_os_best = 0.0
        self._los.reset()

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list[tuple[int, np.ndarray, np.ndarray, float, float]],
        enc: senc.ENC | None = None,
        goal_state: np.ndarray | None = None,  # noqa: ARG002
        w: stochasticity.DisturbanceData | None = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> np.ndarray:
        if not self._initialized or t < 0.0001:
            self._t_prev = t
            self._initialized = True

        references = self._los.compute_references(waypoints, speed_plan, None, ownship_state, t - self._t_prev)
        self._t_prev = t
        course_ref = references[2, 0]
        speed_ref = references[3, 0]
        if t - self._t_run_sbmpc_last >= 5.0:
            self._speed_os_best, self._course_os_best = self._sbmpc.get_optimal_ctrl_offset(
                speed_ref, course_ref, ownship_state, do_list, enc
            )
            self._t_run_sbmpc_last = t
            # print(
            #     f"[SBMPC] Course output: {np.rad2deg(course_ref + self._course_os_best)} | "
            #     f"Best course offset: {np.rad2deg(self._course_os_best)} | "
            #     f"Nominal course ref: {np.rad2deg(course_ref)} | "
            #     f"Speed output: {speed_ref * self._speed_os_best} | "
            #     f"Best speed offset: {self._speed_os_best} | "
            #     f"Nominal speed ref: {speed_ref}"
            # )
        references[2, 0] = course_ref + self._course_os_best
        references[3, 0] = speed_ref * self._speed_os_best
        return references

    def get_current_plan(self) -> np.ndarray:
        refs = np.zeros((9, 1))
        return refs

    def get_colav_data(self) -> dict:
        return {}

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:  # noqa: ARG002
        return plt_handles


class COLAVBuilder:
    @classmethod
    def construct_colav(cls, config: Config | None = None) -> ICOLAV | None:
        """Builds a colav system from the configuration and the default ones provided in this project.

        ..if specified.

        Args:
            config (Config | None): COLAV configuration. Defaults to None.

        Returns:
            ICOLAV | None: The COLAV system (if any config), e.g. Kuwata VO.
        """
        if config and config.name == COLAVType.VO:
            return VOWrapper(config)
        elif config and config.name == COLAVType.SBMPC:
            return SBMPCWrapper(config)
        else:
            return None
