"""stochasticity.py.

Summary:
Contains class definitions for various stochastic disturbance models,
from which wind, wave, current++ factors can be superposed on ship objects.
Every disturbance class must adhere to the interface IDisturbance.

Author: Trym Tengesdal
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf


class MovingAverageFilter:
    def __init__(self, input_dim: int = 2, window_size: int = 10):
        """Initializes a moving average filter with a specified window size.

        Args:
            input_dim (int, optional): Dimension of input. Defaults to 2.
            window_size (int, optional): Size of averaging window. Defaults to 10.
        """
        self._window_size: int = window_size
        self._window: np.ndarray = np.nan * np.ones((input_dim, window_size))

    def update(self, input_val: np.ndarray) -> np.ndarray:
        """Updates the filter with the input value, and returns the new moving average.

        Args:
            input_val (np.ndarray): Input value

        Returns:
            np.ndarray: New moving average estimate
        """
        self._window = np.concatenate((input_val.reshape(-1, 1), self._window[:, :-1]), axis=1)
        average = np.nanmean(self._window, axis=1)
        return average


@dataclass
class GaussMarkovDisturbanceParams:
    """Parameters for a gauss-markov disturbance model with random speed and direction (of e.g. wind or current).

    Initial speed and direction are optional, and if not configured, they are drawn from the specified ranges.

    If constant is set to True, the speed and direction will remain constant. If add_impulse_noise is set to True,
    impulse noise will be added to the speed and direction dynamics (within the specified ranges).
    """

    constant: bool = True
    initial_speed: float = 3.0
    initial_direction: float = 0.0
    speed_range: tuple[float, float] = (0.0, 3.0)
    direction_range: tuple[float, float] = (-np.pi, np.pi)
    mu_speed: float = 1e-5
    mu_direction: float = 1e-05
    sigma_speed: float = 0.005
    sigma_direction: float = 0.005
    add_impulse_noise: bool = False  # Add impulse noise to the speed and direction dynamics if constant=False
    speed_impulses: list[float] = field(
        default_factory=lambda: [-1.0, 1.0]
    )  # List of impulse noise values to randomly choose from
    direction_impulses: list[float] = field(
        default_factory=lambda: [-np.pi / 4, np.pi / 4]
    )  # List of impulse noise values to randomly choose from
    impulse_times: list[float] = field(default_factory=lambda: [70.0])

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GaussMarkovDisturbanceParams":  # noqa: D102
        params = GaussMarkovDisturbanceParams()
        if "initial_speed" in config_dict:
            params.initial_speed = config_dict["initial_speed"]
        else:
            params.initial_speed = random.uniform(*config_dict["speed_range"])

        if "initial_direction" in config_dict:
            params.initial_direction = np.deg2rad(config_dict["initial_direction"])
        else:
            params.initial_direction = random.uniform(*config_dict["direction_range"])

        params.speed_range = tuple(config_dict["speed_range"])
        params.direction_range = tuple(np.deg2rad(config_dict["direction_range"]))
        params.mu_speed = config_dict["mu_speed"]
        params.mu_direction = config_dict["mu_direction"]
        params.sigma_speed = config_dict["sigma_speed"]
        params.sigma_direction = config_dict["sigma_direction"]
        params.constant = config_dict["constant"]
        if "add_impulse_noise" in config_dict:
            params.add_impulse_noise = config_dict["add_impulse_noise"]
        if "speed_impulses" in config_dict:
            params.speed_impulses = config_dict["speed_impulses"]
        if "direction_impulses" in config_dict:
            params.direction_impulses = np.deg2rad(config_dict["direction_impulses"])
        if "impulse_times" in config_dict:
            params.impulse_times = config_dict["impulse_times"]
        else:
            rng = np.random.default_rng()
            params.impulse_times = rng.integers(low=20, high=150, size=1).tolist()
            params.impulse_times.sort()
        return params

    def to_dict(self) -> dict:  # noqa: D102
        config_dict = {
            "constant": self.constant,
            "initial_speed": self.initial_speed,
            "initial_direction": float(np.rad2deg(self.initial_direction)),
            "speed_range": list(self.speed_range),
            "direction_range": list(self.direction_range),
            "mu_speed": self.mu_speed,
            "mu_direction": self.mu_direction,
            "sigma_speed": self.sigma_speed,
            "sigma_direction": self.sigma_direction,
            "add_impulse_noise": self.add_impulse_noise,
            "speed_impulses": self.speed_impulses,
            "direction_impulses": [float(np.rad2deg(di)) for di in self.direction_impulses],
            "impulse_times": [float(t) for t in self.impulse_times],
        }
        config_dict["direction_range"] = [
            float(np.rad2deg(self.direction_range[0])),
            float(np.rad2deg(self.direction_range[1])),
        ]
        return config_dict


@dataclass
class Config:
    "Configuration class for managing environment disturbance/stochasticity parameters."

    wind: GaussMarkovDisturbanceParams | None = field(default_factory=lambda: GaussMarkovDisturbanceParams())
    waves: dict | None = None
    currents: GaussMarkovDisturbanceParams | None = field(default_factory=lambda: GaussMarkovDisturbanceParams())

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":  # noqa: D102
        config = Config()
        if "wind" in config_dict:
            config.wind = cp.convert_settings_dict_to_dataclass(GaussMarkovDisturbanceParams, config_dict["wind"])

        if "currents" in config_dict:
            config.currents = cp.convert_settings_dict_to_dataclass(GaussMarkovDisturbanceParams, config_dict["currents"])

        return config

    def to_dict(self) -> dict:  # noqa: D102
        config_dict: dict = {}
        if self.wind is not None:
            config_dict["wind"] = self.wind.to_dict()

        if self.currents is not None:
            config_dict["currents"] = self.currents.to_dict()

        return config_dict


class IDisturbance(ABC):
    @abstractmethod
    def update(self, t: float, dt: float) -> None:
        """Updates the disturbance process from time t to t + dt.

        Args:
            t (float): Current time.
            dt (float): Time step.
        """

    @abstractmethod
    def get(self) -> np.ndarray:
        """Fetches disturbance vector at time t.

        Returns:
            - np.ndarray: Disturbance data
        """

    def reset(self, seed: int | None) -> None:  # noqa: B027
        """Resets the disturbance process.

        Args:
            seed (int | None): Random seed.
        """


class GaussMarkovDisturbance(IDisturbance):
    """Gauss-Markov disturbance model with random speed and direction (of e.g. wind or current) in the North-East frame."""

    def __init__(self, params: GaussMarkovDisturbanceParams):
        self._params: GaussMarkovDisturbanceParams = params
        self._speed: float = params.initial_speed
        self._direction: float = params.initial_direction
        self._impulse_counter: int = 0
        self._rng = np.random.default_rng()
        self._impulse_applied: bool = False
        self._speed_impulse: float = 0.0
        self._direction_impulse: float = 0.0

        self._params.add_impulse_noise = False

    def reset(self, seed: int | None) -> None:
        self._speed = self._params.initial_speed
        self._direction = self._params.initial_direction
        self._speed_impulse = 0.0
        self._direction_impulse = 0.0
        self._impulse_applied = False
        self._impulse_counter = 0
        self._rng = np.random.default_rng(seed)

    def update(self, t: float, dt: float) -> None:
        """Update speed and direction dynamics in the gauss-markov process.

        Args:
            t (float): Current time.
            dt (float): Time step.
        """
        if self._params.constant:
            return

        w_V = self._rng.normal(0.0, self._params.sigma_speed)
        w_beta = self._rng.normal(0.0, self._params.sigma_direction)
        V_dot = -self._params.mu_speed * self._speed + w_V
        beta_dot = -self._params.mu_direction * self._direction + w_beta

        if self._params.add_impulse_noise and (
            self._params.impulse_times[self._impulse_counter]
            <= t
            < self._params.impulse_times[self._impulse_counter] + 3.0 * dt
        ):
            self._speed_impulse = self._rng.choice(self._params.speed_impulses)
            self._direction_impulse = self._rng.choice(self._params.direction_impulses)
            self._impulse_counter += 1 if self._impulse_counter < len(self._params.impulse_times) - 1 else 0
            self._impulse_applied = True
        elif self._impulse_applied:
            # subtract impulse noise from the speed and direction dynamics to avoid permanent disturbance change
            self._speed -= self._speed_impulse
            self._direction -= self._direction_impulse
            self._impulse_applied = False
            self._speed_impulse = 0.0
            self._direction_impulse = 0.0

        # "Euler integration"
        self._speed = mf.sat(
            self._speed + V_dot * dt + self._speed_impulse, self._params.speed_range[0], self._params.speed_range[1]
        )
        self._direction = mf.sat(
            self._direction + beta_dot * dt + self._direction_impulse,
            self._params.direction_range[0],
            self._params.direction_range[1],
        )
        self._direction = mf.wrap_angle_to_pmpi(self._direction)

    def get(self) -> np.ndarray:
        """Fetches gauss-markov process disturbance vector [speed, direction] at time t.

        Returns:
            - np.ndarray: Disturbance speed and direction in the North-East frame
        """
        return np.array([self._speed, self._direction])


@dataclass
class DisturbanceData:
    """Class used for containing disturbance data.

    Dictionaries are used preliminarily for containing info on the different
    disturbances, to be flexible wrt the disturbance models that are used.
    """

    wind: dict
    waves: dict
    currents: dict

    def __init__(self):
        self.wind = {}
        self.waves = {}
        self.currents = {}

    def print(self):
        if self.wind:
            print("Wind speed:", self.wind["speed"], "m/s, Wind direction:", np.rad2deg(self.wind["direction"]), "deg")
        if self.currents:
            print(
                "Current speed:",
                self.currents["speed"],
                "m/s, Current direction:",
                np.rad2deg(self.currents["direction"]),
                "deg",
            )


class Disturbance:
    """Class for managing the different disturbances affecting the ship motion (wind, waves, currents)."""

    def __init__(self, config: Config):
        self._wind: GaussMarkovDisturbance | None = None
        self._waves: bool | None = None
        self._currents: GaussMarkovDisturbance | None = None

        if config.wind is not None:
            self._wind = GaussMarkovDisturbance(config.wind)

        if config.currents is not None:
            self._currents = GaussMarkovDisturbance(config.currents)

    def reset(self, seed: int | None) -> None:
        """Resets the disturbance processes.

        Args:
            seed (int | None): Random seed.
        """
        if self._wind is not None:
            self._wind.reset(seed)

        if self._currents is not None:
            self._currents.reset(seed)

    def disable_wind(self):
        self._wind = None

    def disable_currents(self):
        self._currents = None

    def enable_wind(self, config: GaussMarkovDisturbanceParams):
        self._wind = GaussMarkovDisturbance(config)

    def enable_currents(self, config: GaussMarkovDisturbanceParams):
        self._currents = GaussMarkovDisturbance(config)

    def update(self, t: float, dt: float) -> None:
        """Updates the disturbance processes from time t to t + dt.

        Args:
            t (float): Current time.
            dt (float): Time step.
        """
        if self._wind is not None:
            self._wind.update(t, dt)

        if self._currents is not None:
            self._currents.update(t, dt)

    def get(self) -> DisturbanceData:
        """Fetches the disturbance data at time t."""
        disturbance_data = DisturbanceData()

        if self._wind is not None:
            disturbance_data.wind = {}
            wind = self._wind.get()
            disturbance_data.wind["speed"] = wind[0]  # min(wind[0], 4.0)
            disturbance_data.wind["direction"] = wind[1]

        if self._currents is not None:
            disturbance_data.currents = {}
            current = self._currents.get()
            disturbance_data.currents["speed"] = current[0]  # min(current[0], 2.0)
            disturbance_data.currents["direction"] = current[1]

        return disturbance_data
