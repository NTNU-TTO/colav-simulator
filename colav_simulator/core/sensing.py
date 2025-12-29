"""sensing.py.

Summary:
Contains class definitions for various sensors.
Every sensor must adhere to the ISensor interface.

Author: Trym Tengesdal, Ragnar Wien
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf


class ISensor(ABC):
    @abstractmethod
    def R(self, xs: np.ndarray) -> np.ndarray:
        """Returns the measurement noise covariance matrix for the input state."""

    @abstractmethod
    def H(self, xs: np.ndarray) -> np.ndarray:
        """Returns the measurement matrix for the input state."""

    @abstractmethod
    def h(self, xs: np.ndarray) -> np.ndarray:
        """Returns the measurement function for the input state."""

    @abstractmethod
    def reset(self, seed: int | None) -> None:
        """Resets the sensor to its initial state, optionally seeding the rng for measurement generation."""

    @abstractmethod
    def seed(self, seed: int | None) -> None:
        """Sets the seed for the sensor's random number generator."""

    @abstractmethod
    def generate_measurements(
        self, t: float, true_do_states: list[tuple[int, np.ndarray, float, float]], ownship_state: np.ndarray
    ) -> list[tuple[int, np.ndarray]]:
        """Generates sensor measurements from the input tuple list of true dynamic obstacle info.

        Takes (do_idx, do_state, do_length, do_width) x n_do, and own-ship state.

        Args:
            t (float): Current time.
            true_do_states (list[tuple[int, np.ndarray, float, float]]): List of tuples
                containing the dynamic obstacle index, state, length, and width.
            ownship_state (np.ndarray): Own-ship state vector.

        Returns:
            list[tuple[int, np.ndarray]]: List of tuples containing the dynamic obstacle
                index and the measurement.
        """


@dataclass
class RadarParams:
    """Configuration parameters for a radar sensor."""

    max_range: float = 1000.0
    measurement_rate: float = 1.0
    R_ne: np.ndarray = field(default_factory=lambda: np.diag([5.0**2, 5.0**2]))  # north-east meas cov used by the tracker
    R_ne_true: np.ndarray = field(
        default_factory=lambda: np.diag([5.0**2, 5.0**2])
    )  #  north-east meas cov that reflects the true noise characteristics. Used to generate measurements
    generate_clutter: bool = False
    clutter_cardinality_expectation: int = 5
    detection_probability: float = 0.9
    include_polar_meas_noise: bool = False
    R_polar_true: np.ndarray = field(default_factory=lambda: np.diag([8.0**2, ((np.pi / 180) * 1) ** 2]))  # meas cov

    @classmethod
    def from_dict(self, config_dict: dict) -> "RadarParams":  # noqa: D102
        return RadarParams(
            max_range=config_dict["max_range"],
            measurement_rate=config_dict["measurement_rate"],
            R_ne=np.diag(config_dict["R_ne"]),
            R_ne_true=np.diag(config_dict["R_ne_true"]),
            generate_clutter=config_dict["generate_clutter"],
            clutter_cardinality_expectation=config_dict["clutter_cardinality_expectation"],
            detection_probability=config_dict["detection_probability"],
            include_polar_meas_noise=config_dict["include_polar_meas_noise"],
            R_polar_true=np.diag(config_dict["R_polar_true"]),
        )

    def to_dict(self) -> dict:  # noqa: D102
        output_dict = asdict(self)
        output_dict["R_ne"] = self.R_ne.diagonal().tolist()
        output_dict["R_ne_true"] = self.R_ne_true.diagonal().tolist()
        output_dict["R_polar_true"] = self.R_polar_true.diagonal().tolist()
        return output_dict


class AISClass(Enum):
    """AIS class A and B transponder types."""

    A = 0
    B = 1


@dataclass
class AISParams:
    """AIS parameter class."""

    max_range: float = 5000.0
    ais_class: AISClass = AISClass.A
    R: np.ndarray = field(
        default_factory=lambda: np.diag([5.0**2, 5.0**2, 0.1**2, 0.08**2])
    )  # meas cov for a state vector of [x, y, Vx, Vy], used by the tracker
    R_true: np.ndarray = field(
        default_factory=lambda: np.diag([5.0**2, 5.0**2, 0.1**2, 0.08**2])
    )  # meas cov that reflects the true noise characteristics. Used to generate measurements

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AISParams":  # noqa: D102
        return AISParams(
            max_range=config_dict["max_range"],
            ais_class=AISClass[config_dict["ais_class"]],
            R=np.diag(config_dict["R"]),
            R_true=np.diag(config_dict["R_true"]),
        )

    def to_dict(self) -> dict:  # noqa: D102
        output_dict = {
            "max_range": self.max_range,
            "ais_class": self.ais_class.name,
            "R": self.R.diagonal().tolist(),
            "R_true": self.R_true.diagonal().tolist(),
        }
        return output_dict


@dataclass
class Config:
    """Class for holding sensor(s) configuration parameters."""

    sensor_list: list = field(default_factory=lambda: [RadarParams()])

    def to_dict_list(self) -> list:
        output_list: list = []
        for sensor in self.sensor_list:
            sensor_dict = {}
            if isinstance(sensor, RadarParams):
                sensor_dict["radar"] = sensor.to_dict()
            elif isinstance(sensor, AISParams):
                sensor_dict["ais"] = sensor.to_dict()
            output_list.append(sensor_dict)

        return output_list

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":  # noqa: D102
        config = Config(sensor_list=[])
        for sensor_dict in config_dict:
            if "radar" in sensor_dict:
                config.sensor_list.append(cp.convert_settings_dict_to_dataclass(RadarParams, sensor_dict["radar"]))
            elif "ais" in sensor_dict:
                config.sensor_list.append(cp.convert_settings_dict_to_dataclass(AISParams, sensor_dict["ais"]))

        return config


class SensorSuiteBuilder:
    @classmethod
    def construct_sensors(cls, config: Config | None = None) -> list:
        """Builds a list of sensors from the configuration.

        Args:
            config (Optional[Config]): Configuration of ship sensors

        Returns:
            List[Sensor]: List of sensors.
        """
        if config:
            sensors: list = []
            for sensor_config in config.sensor_list:
                if isinstance(sensor_config, RadarParams):
                    sensors.append(Radar(sensor_config))
                elif isinstance(sensor_config, AISParams):
                    sensors.append(AIS(sensor_config))
        else:
            sensors = [Radar()]
        return sensors


class Radar(ISensor):
    """Implements functionality for a radar sensor."""

    def __init__(self, params: RadarParams = RadarParams()) -> None:
        self.type: str = "radar"
        self._H: np.ndarray = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self._params: RadarParams = params
        self._prev_meas_time: float = 0.0
        self._initialized: bool = False
        self._rng: np.random.Generator = np.random.default_rng()

    def reset(self, seed: int | None) -> None:
        self.seed(seed)
        self._prev_meas_time = 0.0
        self._initialized = False

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def R(self, xs: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._params.R_ne_true

    def H(self, xs: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._H

    def h(self, xs: np.ndarray) -> np.ndarray:
        return self._H @ xs

    def generate_measurements(
        self, t: float, true_do_states: list[tuple[int, np.ndarray, float, float]], ownship_state: np.ndarray
    ) -> list[tuple[int, np.ndarray]]:
        measurements = []
        if not self._initialized or t < 0.0001:
            self._prev_meas_time = t
            self._initialized = True

        detection_probability = self._params.detection_probability if self._params.generate_clutter else 1.0

        if (t - self._prev_meas_time) < (1.0 / self._params.measurement_rate):
            return [(do_tup[0], np.nan * np.ones(2)) for do_tup in true_do_states]

        for _i, (do_idx, do_state, _do_length, _do_width) in enumerate(true_do_states):
            dist_ownship_to_do = np.sqrt((do_state[0] - ownship_state[0]) ** 2 + (do_state[1] - ownship_state[1]) ** 2)
            do_detection_check = self._rng.random()
            if dist_ownship_to_do <= self._params.max_range and do_detection_check <= detection_probability:
                cartesian_meas_noise = self._rng.multivariate_normal(np.zeros(2), self._params.R_ne_true)
                if self._params.include_polar_meas_noise:
                    angle_ownship_to_do = np.arctan2(do_state[1] - ownship_state[1], do_state[0] - ownship_state[0])
                    ownship_to_do_polar_coords = np.array([dist_ownship_to_do, angle_ownship_to_do])
                    polar_meas_noise = self._rng.multivariate_normal(np.zeros(2), self._params.R_polar_true)
                    distorted_ownship_to_do_polar_coords = ownship_to_do_polar_coords + polar_meas_noise
                    distorted_ownship_to_do_cart_coords = np.array(
                        [
                            ownship_state[0]
                            + distorted_ownship_to_do_polar_coords[0] * np.cos(distorted_ownship_to_do_polar_coords[1]),
                            ownship_state[1]
                            + distorted_ownship_to_do_polar_coords[0] * np.sin(distorted_ownship_to_do_polar_coords[1]),
                        ]
                    )  # Cartesian coords of dynamic obstacle distorted by polar measurement noise
                    polar_meas_noise_cart_coords = distorted_ownship_to_do_cart_coords - self.h(
                        do_state
                    )  # cartesian coords of polar measurement noise relative to dynamic obstacle
                    meas_noise = (
                        polar_meas_noise_cart_coords + cartesian_meas_noise
                    ) / 2  # Midpoint between cartesian and polar measurement noise in cartesian coords
                else:
                    meas_noise = cartesian_meas_noise

                z = self.h(do_state) + meas_noise
            else:
                z = np.nan * np.ones(2)
            measurements.append((do_idx, z))

        self._prev_meas_time = t
        z_clutter = self.generate_clutter(ownship_state)
        measurements.extend(z_clutter)
        return measurements

    def generate_clutter(self, ownship_state: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Generates clutter measurements around the ownship using a Poisson distribution.

        Args:
            ownship_state (np.ndarray): The ownship state vector on the form [x, y, Vx, Vy].

        Returns:
            List[Tuple[int, np.ndarray]]: List of clutter measurements with -1 as the index (non-existent dynamic obstacle).
        """
        clutter = []
        if not self._params.generate_clutter:
            return clutter

        cardinality = self._rng.poisson(self._params.clutter_cardinality_expectation, 1)
        r = self._params.max_range * np.sqrt(self._rng.uniform(0, 1, cardinality))
        theta = self._rng.uniform(0, 2 * np.pi, cardinality)
        x = r * np.cos(theta) + ownship_state[0]
        y = r * np.sin(theta) + ownship_state[1]
        for i in range(cardinality[0]):
            clutter.append((-1, np.array([x[i], y[i]])))
        return clutter

    @property
    def max_range(self) -> float:
        return self._params.max_range

    @property
    def params(self) -> RadarParams:
        return self._params


class AIS(ISensor):
    """Class for simulating AIS transponder measurements.

    AIS/VDES:
        Measurement rate depends on:
        Class A	Anchored / Moored	 Every 3 Minutes
        Class A	Sailing 0-14 knots	 Every 10 Seconds
        Class A	Sailing 14-23 knots	 Every 6 Seconds
        Class A	Sailing 0-14 knots and changing course	 Every 3.33 Seconds
        Class A	Sailing 14-23 knots and changing course	 Every 2 Seconds
        Class A	Sailing faster than 23 knots	 Every 2 Seconds
        Class A	Sailing faster than 23 knots and changing course	 Every 2 Seconds
        Class B	Stopped or sailing up to 2 knots	 Every 3 Minutes
        Class B	Sailing faster than 2 knots	 Every 30 Seconds

     R_realistic values from paper considering quantization effects
     https://link.springer.com/chapter/10.1007/978-3-319-55372-6_13#Sec14
    def R_realistic(self, x: np.ndarray):
        R_GNSS = np.diag([0.5, 0.5, 0.1, 0.1])**2
        R_v = np.diag([x[2]**2, x[3]**2, 0, 0])
        self._R = R_GNSS + (1/12)*R_v
    """

    type: str = "ais"
    _H: np.ndarray = np.eye(4)

    def __init__(self, params: AISParams = AISParams()) -> None:
        self._params: AISParams = params
        self._prev_meas_time: list = []
        self._initialized: bool = False
        self._rng: np.random.Generator = np.random.default_rng()

    def reset(self, seed: int | None) -> None:
        self.seed(seed)
        self._prev_meas_time = []
        self._initialized = False

    def seed(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    def R(self, xs: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._params.R

    def H(self, xs: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._H

    def h(self, xs: np.ndarray) -> np.ndarray:
        z = self._H @ xs
        return z

    def generate_measurements(
        self, t: float, true_do_states: list[tuple[int, np.ndarray, float, float]], ownship_state: np.ndarray
    ) -> list[tuple[int, np.ndarray]]:
        measurements = []
        if not self._initialized or t < 0.0001:
            self._prev_meas_time = [t] * len(true_do_states)
            self._initialized = True

        if len(true_do_states) > len(self._prev_meas_time):
            diff = len(true_do_states) - len(self._prev_meas_time)
            self._prev_meas_time.extend([t] * diff)

        for i, (do_idx, do_state, _, _) in enumerate(true_do_states):
            dist_ownship_to_do = np.sqrt((do_state[0] - ownship_state[0]) ** 2 + (do_state[1] - ownship_state[1]) ** 2)

            if (t - self._prev_meas_time[i]) >= (
                1.0 / self._measurement_rate(do_state)
            ) and dist_ownship_to_do <= self._params.max_range:
                z = self.h(do_state) + self._rng.multivariate_normal(np.zeros(4), self._params.R_true)
                self._prev_meas_time[i] = t
            else:
                z = np.nan * np.ones(4)
            measurements.append((do_idx, z))

        return measurements

    def _measurement_rate(self, xs: np.ndarray) -> float:
        """Returns the measurement rate for the input state.

        This depends on the input state's speed and AIS class (and also if the
        course is changing, but this is not considered here (yet)).

        Args:
            xs (np.ndarray): The state vector of the dynamic obstacle = [x, y, Vx, Vy].

        Returns:
            float: The measurement rate in Hz.
        """
        sog = mf.ms2knots(float(np.linalg.norm(xs[2:4])))
        rate = 1.0
        if self._params.ais_class == AISClass.A:
            if sog <= 0.001:
                rate = 1.0 / 180.0
            elif sog > 0.001 and sog <= 14.0:
                rate = 1.0 / 10.0
            elif sog > 14.0 and sog <= 23.0:
                rate = 1.0 / 6.0
            elif sog > 23.0:
                rate = 1.0 / 2.0
        elif self._params.ais_class == AISClass.B:
            if sog <= 2.0:
                rate = 1.0 / 180.0
            elif sog > 2.0:
                rate = 1.0 / 30.0
        return rate

    @property
    def max_range(self) -> float:
        return self._params.max_range

    @property
    def params(self) -> AISParams:
        return self._params
