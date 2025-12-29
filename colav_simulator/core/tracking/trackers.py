"""trackers.py.

Summary:
Contains class definitions for dynamic obstacle
target trackers. Every tracker must adhere to the
ITracker interface.

Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as la

import colav_simulator.common.config_parsing as cp
import colav_simulator.core.sensing as sens


class ITracker(ABC):
    @abstractmethod
    def track(
        self,
        t: float,
        dt: float,
        true_do_states: list[tuple[int, np.ndarray, float, float]],
        ownship_state: np.ndarray,
    ) -> tuple[list[tuple[int, np.ndarray, np.ndarray, float, float]], list[tuple[int, np.ndarray]]]:
        """Tracks/updates estimates on dynamic obstacles.

        Based on sensor measurements generated from the input true dynamic obstacle
        states.

        Args:
            t (float): Current time (assumed >= 0).
            dt (float): Time since last update.
            true_do_states (list[tuple[int, np.ndarray, float, float]]): List of
                tuples of true dynamic obstacle indices and states (do_idx, [x, y,
                Vx, Vy], length, width) x n_do. Used for simulating sensor
                measurements.
            ownship_state (np.ndarray): Ownship state vector on the form [x, y, Vx,
                Vy] used for simulating sensor measurements.

        Returns:
            tuple[list[tuple[int, np.ndarray, np.ndarray, float, float]], list[tuple[int, np.ndarray]]]:
                List of updated dynamic obstacle tracks (ID, state, cov, length,
                width), sorted in ascending order by the distance from the ownship.
                Also, a list the sensor measurements used.
        """

    @abstractmethod
    def set_sensor_list(self, sensor_list: list[sens.ISensor]) -> None:
        """Sets the sensor list for the tracker.

        Args:
            sensor_list (List[sens.ISensor]): List of sensors used by the tracker.
        """

    @abstractmethod
    def get_track_information(
        self, ownship_state: np.ndarray
    ) -> tuple[list[tuple[int, np.ndarray, np.ndarray, float, float]], list[float]]:
        """Returns the dynamic obstacle track information.

        Returns (ID, state, cov, length, width), sorted in ascending order by the
        distance from the ownship. Also, it returns the associated Normalized
        Innovation error Squared (NIS) values for the most recent update step for
        each track, and the track labels.

        Args:
            ownship_state (np.ndarray): Ownship state vector on the form [x, y, Vx,
                Vy] used for simulating sensor measurements.

        Returns:
            tuple[list[tuple[int, np.ndarray, np.ndarray, float, float]], list[float]]:
                List of tracks and list of NISes.
        """

    @abstractmethod
    def reset(self) -> None:
        """Resets the tracker to its initial state."""


@dataclass
class KFParams:
    """Class for holding KF parameters."""

    P_0: np.ndarray = field(default_factory=lambda: np.diag([49.0, 49.0, 0.5, 0.5]))
    q: float = 0.4

    def to_dict(self) -> dict:  # noqa: D102
        output_dict = {"P_0": self.P_0.diagonal().tolist(), "q": self.q}
        return output_dict

    @classmethod
    def from_dict(cls, config_dict: dict) -> "KFParams":  # noqa: D102
        return KFParams(P_0=np.diag(config_dict["P_0"]), q=config_dict["q"])


@dataclass
class Config:
    """Class for holding tracker configuration parameters."""

    god_tracker: bool | None = True
    kf: KFParams | None = None

    def to_dict(self) -> dict:  # noqa: D102
        output_dict = {}
        if self.kf is not None:
            output_dict["kf"] = self.kf.to_dict()
        if self.god_tracker is not None:
            output_dict["god_tracker"] = ""

        return output_dict

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":  # noqa: D102
        config = Config()

        if "kf" in config_dict:
            config.kf = cp.convert_settings_dict_to_dataclass(KFParams, config_dict["kf"])
            config.god_tracker = None
        elif "god_tracker" in config_dict:
            config.god_tracker = True
            config.kf = None

        return config


class TrackerBuilder:
    @classmethod
    def construct_tracker(cls, sensors: list, config: Config | None = None) -> ITracker:
        """Builds a tracker from the configuration.

        Args:
            sensors (list): Sensors used by the tracker.
            config (Optional[Config]): Tracker configuration. Defaults to None.

        Returns:
            ITracker: The tracker.
        """
        if config and config.kf:
            return KF(sensors, config.kf)
        elif config and config.god_tracker:
            return GodTracker(sensors)
        else:
            return KF(sensors)


class GodTracker(ITracker):
    """This tracker is used to simulate perfect knowledge of dynamic obstacles."""

    def __init__(self, sensor_list: list[sens.ISensor] | None = None) -> None:
        self.sensors: list[sens.ISensor] = sensor_list

        self._initialized: bool = False
        self._labels: list = []
        self._xs_upd: list = []
        self._P_upd: list = []
        self._length_upd: list = []
        self._t_prev: float = -1.0
        self._width_upd: list = []
        self._recent_sensor_measurements: list = []

    def reset(self) -> None:
        self._initialized = False
        self._labels = []
        self._xs_upd = []
        self._P_upd = []
        self._length_upd = []
        self._t_prev = -1.0
        self._width_upd = []
        self._recent_sensor_measurements = []

    def set_sensor_list(self, sensor_list: list[sens.ISensor]) -> None:
        self.sensors = sensor_list

    def track(
        self,
        t: float,
        dt: float,  # noqa: ARG002
        true_do_states: list[tuple[int, np.ndarray, float, float]],
        ownship_state: np.ndarray,
    ) -> tuple[list[tuple[int, np.ndarray, np.ndarray, float, float]], list[tuple[int, np.ndarray]]]:
        # If the function is run at the same time as the previous, return the same tracks
        if self.sensors is None:
            msg = "Sensor list must be set."
            raise ValueError(msg)
        if t <= self._t_prev:
            tracks, _ = self.get_track_information(ownship_state)
            return tracks, self._recent_sensor_measurements

        self._t_prev = t

        for do_idx, do_state, do_length, do_width in true_do_states:
            if do_idx not in self._labels:
                self._labels.append(do_idx)
                self._xs_upd.append(do_state)
                self._P_upd.append(np.zeros((4, 4)))
                self._length_upd.append(do_length)
                self._width_upd.append(do_width)

        # Only generate measurements for initialized tracks
        sensor_measurements = []
        for sensor in self.sensors:
            z = sensor.generate_measurements(t, true_do_states, ownship_state)
            sensor_measurements.append(z)
        self._recent_sensor_measurements = sensor_measurements

        tracks = []
        n_tracked_do = len(true_do_states)
        for i in range(n_tracked_do):
            self._xs_upd[i] = true_do_states[i][1]
            tracks.append(
                (
                    self._labels[i],
                    self._xs_upd[i],
                    self._P_upd[i],
                    self._length_upd[i],
                    self._width_upd[i],
                )
            )
        tracks_sorted_by_distance = sorted(tracks, key=lambda x: np.linalg.norm(x[1][:2] - ownship_state[:2]))
        return tracks_sorted_by_distance, sensor_measurements

    def get_track_information(self, ownship_state: np.ndarray) -> tuple[list, list]:
        tracks = []
        for i, label in enumerate(self._labels):
            tracks.append(
                (
                    label,
                    self._xs_upd[i],
                    self._P_upd[i],
                    self._length_upd[i],
                    self._width_upd[i],
                )
            )
        tracks_sorted_by_distance = sorted(tracks, key=lambda x: np.linalg.norm(x[1][:2] - ownship_state[:2]))
        return tracks_sorted_by_distance, [0.0 for _ in range(len(tracks_sorted_by_distance))]


class KF(ITracker):
    """The KF class implements a linear Kalman filter based tracker."""

    def __init__(self, sensor_list: list[sens.ISensor] | None = None, params: KFParams | None = None) -> None:
        if params is not None:
            self._params: KFParams = params
        else:
            self._params = KFParams()

        self._model = CVModel(self._params.q)

        self.sensors: list[sens.ISensor] = sensor_list

        self._track_initialized: list = []
        self._track_terminated: list = []
        self._labels: list = []
        self._xs_p: list = []
        self._P_p: list = []
        self._xs_upd: list = []
        self._P_upd: list = []
        self._length_upd: list = []  # List of DO length estimates. Assumed known
        self._width_upd: list = []  # List of DO width estimates. Assumed known
        self._NIS: list = []
        self._t_prev: float = -1.0
        self._recent_sensor_measurements: list = []

    def reset(self) -> None:
        self._track_initialized = []
        self._track_terminated = []
        self._labels = []
        self._xs_p = []
        self._P_p = []
        self._xs_upd = []
        self._P_upd = []
        self._length_upd = []
        self._width_upd = []
        self._NIS = []
        self._t_prev = -1.0
        self._recent_sensor_measurements = []

    def set_sensor_list(self, sensor_list: list[sens.ISensor]) -> None:
        self.sensors = sensor_list

    def track(
        self,
        t: float,
        dt: float,
        true_do_states: list[tuple[int, np.ndarray, float, float]],
        ownship_state: np.ndarray,
    ) -> tuple[list[tuple[int, np.ndarray, np.ndarray, float, float]], list[tuple[int, np.ndarray]]]:
        if self.sensors is None:
            msg = "Sensor list must be set."
            raise ValueError(msg)
        # If the function is run at the same time as the previous, return the same tracks
        if t <= self._t_prev:
            tracks, _ = self.get_track_information(ownship_state)
            return tracks, self._recent_sensor_measurements

        self._t_prev = t
        max_sensor_range = max(sensor.max_range for sensor in self.sensors)
        for _i, (do_idx, do_state, do_length, do_width) in enumerate(true_do_states):
            dist_ownship_to_do = np.linalg.norm(do_state[:2] - ownship_state[:2])
            if do_idx not in self._labels and dist_ownship_to_do < max_sensor_range:
                self._labels.append(do_idx)
                self._track_initialized.append(False)
                self._track_terminated.append(False)
                self._xs_upd.append(do_state)
                self._P_upd.append(self._params.P_0)
                self._xs_p.append(do_state)
                self._P_p.append(self._params.P_0)
                self._length_upd.append(do_length)
                self._width_upd.append(do_width)
                self._NIS.append(np.nan)
            elif do_idx in self._labels:
                self._track_initialized[self._labels.index(do_idx)] = True

        n_tracked_do = len(self._xs_upd)
        sensor_measurements = []
        for sensor in self.sensors:
            z = sensor.generate_measurements(t, true_do_states, ownship_state)
            sensor_measurements.append(z)
        self._recent_sensor_measurements = sensor_measurements

        tracks = []
        for i in range(n_tracked_do):
            if self._track_initialized[i] and not self._track_terminated[i]:
                self._xs_p[i], self._P_p[i] = self.predict(self._xs_upd[i], self._P_upd[i], dt)
                self._xs_upd[i] = self._xs_p[i]
                self._P_upd[i] = self._P_p[i]

                if sensor_measurements:
                    for sensor_id in range(len(self.sensors)):
                        sensed_dos = [do_meas[0] for do_meas in sensor_measurements[sensor_id]]

                        if self._labels[i] in sensed_dos:  # Automatic data association
                            z = sensor_measurements[sensor_id][sensed_dos.index(self._labels[i])][1]

                            self._xs_upd[i], self._P_upd[i], NIS_i = self.update(
                                self._xs_upd[i], self._P_upd[i], z, sensor_id
                            )

                            if not np.isnan(NIS_i):
                                self._NIS[i] = NIS_i

            tracks.append(
                (
                    self._labels[i],
                    self._xs_upd[i],
                    self._P_upd[i],
                    self._length_upd[i],
                    self._width_upd[i],
                )
            )

        # print(f"xs_p: {self._xs_p}, xs_upd: {self._xs_upd}")
        # print(f"P_p: {self._P_p}")
        # print(f"P_upd: {self._P_upd}")
        tracks_sorted_by_distance = sorted(tracks, key=lambda x: np.linalg.norm(x[1][:2] - ownship_state[:2]))
        return tracks_sorted_by_distance, sensor_measurements

    def predict(self, xs_upd: np.ndarray, P_upd: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = self._model.F(dt)
        Q = self._model.Q(dt)

        x_pred = self._model.f(xs_upd, dt)
        P_pred = F @ P_upd @ F.T + Q

        return x_pred, P_pred

    def innovation(self, xs_p: np.ndarray, P_p: np.ndarray, z: np.ndarray, sensor_id: int) -> tuple[np.ndarray, np.ndarray]:
        zbar = self.sensors[sensor_id].h(xs_p)
        v = z - zbar

        H = self.sensors[sensor_id].H(xs_p)
        R = self.sensors[sensor_id].R(xs_p)
        S = H @ P_p @ H.T + R

        return v, S

    def update(
        self, xs_p: np.ndarray, P_p: np.ndarray, z: np.ndarray, sensor_id: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if any(np.isnan(z)):
            return xs_p, P_p, np.nan

        v, S = self.innovation(xs_p, P_p, z, sensor_id)
        H = self.sensors[sensor_id].H(xs_p)

        K = P_p @ H.T @ la.inv(S)
        x_upd = xs_p + K @ v
        P_upd = P_p - K @ H @ P_p

        return x_upd, P_upd, NIS(v, S)

    def get_track_information(self, ownship_state: np.ndarray) -> tuple[list, list]:
        tracks = []
        for i, label in enumerate(self._labels):
            tracks.append(
                (
                    label,
                    self._xs_upd[i],
                    self._P_upd[i],
                    self._length_upd[i],
                    self._width_upd[i],
                )
            )
        tracks_sorted_by_distance = sorted(tracks, key=lambda x: np.linalg.norm(x[1][:2] - ownship_state[:2]))
        return tracks_sorted_by_distance, [0.0 for _ in range(len(tracks_sorted_by_distance))]


def NIS(v: np.ndarray, S: np.ndarray) -> float:
    """Calculate the Normalized Innovation Squared (NIS).

    Args:
        v (np.ndarray): Innovation vector.
        S (np.ndarray): Innovation covariance matrix.

    Returns:
        float: NIS value.
    """
    return v.T @ la.inv(S) @ v


class CVModel:
    """The CVModel class implements a constant velocity model."""

    def __init__(self, q: float) -> None:
        self._q: float = q

    def f(self, xs: np.ndarray, dt: float) -> np.ndarray:
        """Returns the r.h.s of the prediction model state transition function.

        Args:
            xs (np.ndarray): State vector [x, y, Vx, Vy]
            dt (float): Time step

        Returns:
            np.ndarray: New state vector dt seconds ahead

        """
        return xs + np.array([xs[2] * dt, xs[3] * dt, 0.0, 0.0])

    def F(self, dt: float) -> np.ndarray:
        """Returns the Jacobian of the prediction model state transition function.

        Args:
            xs (np.ndarray): xs (np.ndarray): State vector [x, y, Vx, Vy]
            dt (float): Time step

        Returns:
            np.ndarray: Jacobian of the prediction model state transition function
        """
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def Q(self, dt: float) -> np.ndarray:
        """Returns the process noise covariance matrix.

        Args:
            dt (float): Time step

        Returns:
            np.ndarray: Process noise covariance matrix
        """
        return (
            np.array(
                [
                    [dt**3 / 3.0, 0.0, dt**2 / 2.0, 0.0],
                    [0.0, dt**3 / 3.0, 0.0, dt**2 / 2.0],
                    [dt**2 / 2.0, 0.0, dt, 0.0],
                    [0.0, dt**2 / 2.0, 0.0, dt],
                ]
            )
            * self._q
        )
