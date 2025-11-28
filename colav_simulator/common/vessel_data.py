"""
vessel_data.py

Summary:
    Contains a VesselData class, used to
    store trajectory data and other information for the vessel.

Author: Trym Tengesdal, Inger Berge Hagen
"""

import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters
import seacharts.enc as senc
import shapely.geometry as geometry
from scipy.interpolate import interp1d

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm


class Status(Enum):
    """Status of the vessel (by AIS standards)"""

    UnderWayUsingEngine = 0  # Underway using engine
    AtAnchor = 1  # At anchor
    NotUnderCommand = 2  # Not under command
    RestrictedManeuverability = 3  # Restricted maneuverability
    ConstrainedByHerDraught = 4  # Constrained by her draught
    Moored = 5  # Moored
    Aground = 6  # Aground
    EngagedInFishing = 7  # Engaged in fishing
    UnderwaySailing = 8  # Underway sailing
    NotDefined = 9  # Catch-all for remaining values 9 - 15


@dataclass
class VesselData:
    """The VesselData class is used by the Evaluator class to store the trajectory and other information for a vessel.

    NOTE: States must be given in a local ENU frame to work with the EvalTool class.
    """

    xy: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )  # Position trajectory of the vessel: [x, y] x n_msgs, where x is east and y is north.
    latlon: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )  # Position trajectory of the vessel: [lat, lon]  x n_msgs
    sog: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )  # Speed over ground in m/s, array x n_msgs
    cog: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )  # Course over ground array x n_msgs
    timestamps: np.ndarray = field(default_factory=lambda: np.empty(0))
    datetimes_utc: np.ndarray = field(default_factory=lambda: np.empty(0))
    name: str = ""
    type: int = 0
    id: int = 0
    mmsi: int = 0
    imo: int = 0
    callsign: str = ""
    width: float = 5.0
    length: float = 20.0
    draft: float = 3.0
    min_depth: int = -1
    travel_dist: float = -1.0
    maneuver_detect_idx: np.ndarray = field(default_factory=lambda: np.empty(0))
    delta_cog: np.ndarray = field(default_factory=lambda: np.empty(0))
    delta_sog: np.ndarray = field(default_factory=lambda: np.empty(0))
    sog_der: np.ndarray = field(default_factory=lambda: np.empty(0))
    maneuver_der: np.ndarray = field(default_factory=lambda: np.empty(0))
    cog_maneuvers_idx: np.ndarray = field(default_factory=lambda: np.empty(0))
    sog_maneuvers_idx: np.ndarray = field(default_factory=lambda: np.empty(0))

    first_valid_idx: int = -1
    last_valid_idx: int = -1
    nav_status: np.ndarray = field(default_factory=lambda: np.empty(0))
    status: Status = Status.UnderWayUsingEngine

    heading: list = field(default_factory=lambda: [])  # from AIS if available
    forward_heading_estimate: list = field(default_factory=lambda: [])
    backward_heading_estimate: list = field(default_factory=lambda: [])

    grounding_dist: float = -1.0
    grounding_dist_vec: np.ndarray = field(default_factory=lambda: np.empty(0))
    grounding_idx: int = -1

    @classmethod
    def create_from_ais_data(
        cls,
        t_0_global: datetime,
        t_end_global: datetime,
        identifier: int,
        ship_ais_df: pd.DataFrame,
        ship_info_df: pd.DataFrame = None,
        utm_zone: int = 33,
        sample_interval: float = 1.0,
    ):
        """Create a VesselData object from AIS data contained in DataFrames.

        Interpolates the data to within [t_0_global, t_end_global] for a desired sampling interval.
        This means that some of the data could contain NaN values, due to the vessel AIS data not
        necessarily covering the entire time interval.

        Assumes availability of the following columns in the AIS DataFrame:
        mmsi;date_time_utc;sog;cog;true_heading;nav_status;calc_speed;lon;lat

        Args:
            t_0_global (datetime): Start time, minimum over all considered vessels, from the AIS data.
            t_end_global (datetime): End time, maximum over all considered vessels, from the AIS data.
            identifer (int): Identifier (ID) for the vessel.
            ship_ais_df (pd.DataFrame): DataFrame containing AIS data for the vessel.
            ship_info_df (Optional[pd.DataFrame]): DataFrame containing information about the vessel.
            utm_zone (int, optional): UTM zone for the ENU frame used.
            sample_interval (float, optional): Desired sampling interval for the trajectory. Defaults to 1.0 seconds.

        Returns:
            VesselData: Object containing vessel data.
        """
        vessel = VesselData(id=identifier, mmsi=int(ship_ais_df.at[0, "mmsi"]))
        name = str(vessel.mmsi)
        if ship_info_df is not None:
            if vessel.mmsi in ship_info_df.mmsi.tolist():
                name_df = ship_info_df.loc[ship_info_df.mmsi == vessel.mmsi]
                idx = name_df.index[0]
                name = str(name_df.name[idx])
                vessel.length = float(ship_info_df.at[idx, "length"])
                vessel.width = float(ship_info_df.at[idx, "width"])
                vessel.type = int(ship_info_df.at[idx, "type"])
                vessel.imo = int(ship_info_df.at[idx, "imo"])
                vessel.callsign = str(ship_info_df.at[idx, "callsign"])
        vessel.name = name

        # Remove NaNs and interpolate to desired sampling interval
        first_valid_idx = ship_ais_df.nav_status.first_valid_index()
        last_valid_idx = ship_ais_df.nav_status.last_valid_index()
        vessel.nav_status = ship_ais_df.nav_status.tolist()[
            first_valid_idx : last_valid_idx + 1
        ]
        vessel.status = determine_status(vessel.nav_status)

        if first_valid_idx > 0:
            print(f"Nav status: {ship_ais_df.nav_status.tolist()}")

        datetimes_utc = ship_ais_df["date_time_utc"].tolist()
        datetimes_utc = datetimes_utc[first_valid_idx : last_valid_idx + 1]
        vessel.datetimes_utc = datetimes_utc

        dataset_timespan = (t_end_global - t_0_global).total_seconds()

        if len(datetimes_utc) < 2:
            return None

        original_times = []
        for i in range(len(datetimes_utc)):  # pylint: disable=consider-using-enumerate
            original_times.append((datetimes_utc[i] - t_0_global).total_seconds())

        interpolated_times = np.arange(0.0, dataset_timespan, sample_interval)

        # find the first and last valid index for the vessel
        vessel.timestamps = interpolated_times
        n_msgs = len(interpolated_times)
        vessel.latlon = np.zeros((2, n_msgs))
        lat_interpolated = interp1d(
            original_times, ship_ais_df.lat.tolist(), kind="linear", bounds_error=False
        )
        vessel.latlon[0, :] = lat_interpolated(interpolated_times)

        lon_interpolated = interp1d(
            original_times, ship_ais_df.lon.tolist(), kind="linear", bounds_error=False
        )
        vessel.latlon[1, :] = lon_interpolated(interpolated_times)

        first_valid_idx = int(np.argwhere(~np.isnan(vessel.latlon[0, :])).T[0][0])
        last_valid_idx = int(np.argwhere(~np.isnan(vessel.latlon[0, :])).T[0][-1])

        vessel.xy = np.empty((2, n_msgs)) * np.nan
        (
            vessel.xy[0, first_valid_idx : last_valid_idx + 1],
            vessel.xy[1, first_valid_idx : last_valid_idx + 1],
        ) = mapf.latlon2local(
            vessel.latlon[0, first_valid_idx : last_valid_idx + 1],
            vessel.latlon[1, first_valid_idx : last_valid_idx + 1],
            utm_zone,
        )

        vessel.forward_heading_estimate = np.zeros(n_msgs) * np.nan
        vessel.backward_heading_estimate = np.zeros(n_msgs) * np.nan
        for k in range(first_valid_idx, last_valid_idx):
            vessel.forward_heading_estimate[k] = np.arctan2(
                vessel.xy[0, k + 1] - vessel.xy[0, k],
                vessel.xy[1, k + 1] - vessel.xy[1, k],
            )
        vessel.forward_heading_estimate[last_valid_idx] = (
            vessel.forward_heading_estimate[last_valid_idx - 1]
        )

        for k in range(first_valid_idx + 1, last_valid_idx):
            vessel.backward_heading_estimate[k] = np.arctan2(
                vessel.xy[0, k] - vessel.xy[0, k - 1],
                vessel.xy[1, k] - vessel.xy[1, k - 1],
            )
        vessel.backward_heading_estimate[first_valid_idx] = (
            vessel.forward_heading_estimate[first_valid_idx]
        )

        cog_interpolated = interp1d(
            original_times, ship_ais_df.cog.tolist(), kind="linear", bounds_error=False
        )
        vessel.cog = mf.wrap_angle_to_pmpi(
            np.deg2rad(cog_interpolated(interpolated_times))
        )

        sog_interpolated = interp1d(
            original_times, ship_ais_df.sog.tolist(), kind="linear", bounds_error=False
        )
        vessel.sog = mf.knots2mps(sog_interpolated(interpolated_times))

        vessel.first_valid_idx = first_valid_idx
        vessel.last_valid_idx = last_valid_idx

        vessel.travel_dist = compute_total_dist_travelled(
            vessel.xy[:, first_valid_idx : last_valid_idx + 1]
        )

        if vessel.travel_dist < 500.0:
            vessel.status = Status.AtAnchor
            return None

        print(f"Vessel {identifier} travelled a distance of {vessel.travel_dist} m")
        # print(f"Vessel status: {vessel.status}")

        return vessel

    def compute_course_and_speed_derivatives(
        self,
        epsilon_d_course: float,
        epsilon_speed: float,
        epsilon_d_speed: float,
        epsilon_dist: float = 500.0,
    ) -> None:
        """Compute derivatives of speed and course based on the vessel trajectory (state) data.

        Also finds out when the vessel performs a maneuver.

        Args:
            epsilon_d_course (float): Course change threshold.
            epsilon_speed (float): Speed change threshold.
            epsilon_d_speed (float): Speed derivative threshold.
        """

        # Only compute derivatives if the vessel has traveled more than 500 meters
        if self.travel_dist < epsilon_dist:
            self.status = Status.AtAnchor

        else:
            speed = self.sog.copy()

            target_area = ~np.isnan(speed)
            speed[target_area] = filters.gaussian_filter(speed[target_area], sigma=2)

            target_area = [
                np.logical_and(
                    np.logical_and(target_area[i], target_area[i + 2]),
                    target_area[i + 1],
                )
                for i in range(len(target_area) - 2)
            ]
            target_area = np.append(False, target_area)
            target_area = np.append(target_area, False)
            if speed.size >= 3:
                self.sog_der = np.zeros(speed.size)
                speed = speed[~np.isnan(speed)]
                try:
                    self.sog_der[target_area] = [
                        np.dot([speed[i], speed[i + 1], speed[i + 2]], [-0.5, 0.0, 0.5])
                        for i in range(len(speed) - 2)
                    ]
                except Exception as e:
                    print(e)
                    warnings.warn("self reentering the area may cause problems.")

            # Calculate derivatives of yaw
            cog_smoothed = self.cog.copy()
            cog_diff = np.append([0], self.cog[1:] - self.cog[:-1], 0)

            cog_diff[np.isnan(cog_diff)] = 0.0
            cog_diff[abs(cog_diff) < np.pi] = 0.0
            cog_diff[cog_diff < -np.pi] = -2.0 * np.pi
            cog_diff[cog_diff > np.pi] = (
                2.0 * np.pi
            )  # cog_diff is now 2pi or -2pi at jumps from pi to -pi or opposite

            cumsum_cog_diff = np.cumsum(cog_diff, axis=0)

            target_area = ~np.isnan(cog_smoothed)

            cog_smoothed[target_area] = (
                cog_smoothed[target_area] - cumsum_cog_diff[target_area]
            )  # avoids counting sudden changes from pi to -pi or opposite count as maneuvers

            cog_smoothed[target_area] = filters.gaussian_filter(
                cog_smoothed[target_area], sigma=2
            )

            target_area = [
                np.logical_and(target_area[i], target_area[i + 2])
                for i in range(len(target_area) - 2)
            ]
            target_area = np.append(False, target_area)
            target_area = np.append(target_area, False)
            self.maneuver_der = np.zeros((3, self.cog.size))
            if cog_smoothed.size >= 3:
                cog_smoothed = cog_smoothed[~np.isnan(cog_smoothed)]
                self.maneuver_der[0, target_area] = [
                    np.dot(
                        [cog_smoothed[i], cog_smoothed[i + 1], cog_smoothed[i + 2]],
                        [-0.5, 0, 0.5],
                    )
                    for i in range(len(cog_smoothed) - 2)
                ]
                self.maneuver_der[1, target_area] = [
                    np.dot(
                        [cog_smoothed[i], cog_smoothed[i + 1], cog_smoothed[i + 2]],
                        [1.0, -2.0, 1.0],
                    )
                    for i in range(len(cog_smoothed) - 2)
                ]

                # Added again apparently because target area is changed
                target_area = [
                    np.logical_and(target_area[i], target_area[i + 2])
                    for i in range(len(target_area) - 2)
                ]
                target_area = np.append(False, target_area)
                target_area = np.append(target_area, False)
                self.maneuver_der[2, target_area] = [
                    np.dot(
                        [
                            cog_smoothed[i],
                            cog_smoothed[i + 1],
                            cog_smoothed[i + 2],
                            cog_smoothed[i + 3],
                            cog_smoothed[i + 4],
                        ],
                        [-0.5, 1.0, 0.0, -1.0, 0.5],
                    )
                    for i in range(len(cog_smoothed) - 4)
                ]
                self.maneuver_der[1, :] = np.zeros(self.cog.size)
                self.maneuver_der[1, target_area] = [
                    np.dot(
                        [
                            cog_smoothed[i],
                            cog_smoothed[i + 1],
                            cog_smoothed[i + 2],
                            cog_smoothed[i + 3],
                            cog_smoothed[i + 4],
                        ],
                        [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0],
                    )
                    for i in range(len(cog_smoothed) - 4)
                ]
            self.find_maneuver_detect_index_der(
                epsilon_d_course, epsilon_speed, epsilon_d_speed
            )

    def find_maneuver_detect_index_der(
        self, epsilon_d_course: float, epsilon_speed: float, epsilon_d_speed: float
    ) -> None:
        """Finds indices where a maneuver, course or speed, is started.

        The indices are saved to the array `VesselData.maneuver_detect_idx`.

        The magnitude of the change is saved in the `VesselData.delta_cog` or `VesselData.delta_sog` array.

        For an index marking a course change, the corresponding element in `VesselData.delta_sog` = 0.
        For a course change `VesselData.delta_cog` = 0.

        Start and stop indices for course and speed maneuvers are saved in
        `VesselData.course_maneuver_idx` and `VesselData.speed_maneuver_idx`, respectively.

        Args:
            epsilon_d_course (float): Course change threshold  for maneuver detection, [rad/s].
            epsilon_speed (float): Speed threshold for vessel to be considered moving, [m/s].
            epsilon_d_speed (float): Speed change threshold for maneuver detection, [m/s^2].
        """

        # Course change detection
        cont_man = False
        # third_derivative_zeros_idx = []
        # first_derivative_zeros_idx = []
        course_maneuvers = []
        detection_idx = []
        end_idx = []
        n_msgs = len(self.cog)
        for i in range(n_msgs):
            if i > 0 and len(detection_idx) > 0 and cont_man:
                # Maneuver stops if first derivative goes to zero
                if (
                    np.sign(self.maneuver_der[0, i])
                    + np.sign(self.maneuver_der[0, i - 1])
                    == 0
                    or self.maneuver_der[1, i] == 0
                ):
                    # first_derivative_zeros_idx.append(i)
                    cont_man = False
                    end_idx.append(i)

            # Maneuver stops if speed falls below threshold
            if self.sog[i] < epsilon_speed:
                if cont_man:
                    cont_man = False
                    end_idx.append(i)
                continue

            # # Register third derivative zeros / zero-crossings
            # if np.sign(self.maneuver_der[2, i]) + np.sign(self.maneuver_der[2, i - 1]) == 0 or self.maneuver_der[2, i] == 0:
            #     third_derivative_zeros_idx.append(i)

            # Maneuver if first derivative above threshold
            if np.abs(self.maneuver_der[0, i]) >= epsilon_d_course and not cont_man:
                cont_man = True
                detection_idx.append(i)

        if len(detection_idx) > len(end_idx):
            end_idx.append(i)

        # first_derivative_zeros_idx = np.array(end_idx)
        # third_derivative_zeros_idx = np.array(third_derivative_zeros_idx)

        while len(detection_idx) > 0:
            start_idx = detection_idx.pop(0)
            stop_idx = end_idx.pop(0)
            course_maneuvers.append([start_idx, stop_idx])

        # Speed change detection
        speed_changes = np.abs(self.sog_der) >= epsilon_d_speed
        speed_maneuvers = []
        cont_man = False
        for i, change in enumerate(speed_changes):
            if not cont_man:
                if change:
                    start_idx = i
                    cont_man = True
            else:
                if not change:
                    stop_idx = i - 1
                    cont_man = False
                    speed_maneuvers.append([start_idx, stop_idx])

        # Book-keeping
        n_maneuvers = len(course_maneuvers) + len(speed_maneuvers)
        self.maneuver_detect_idx = np.zeros(n_maneuvers, dtype=int)
        self.delta_cog = np.zeros(n_maneuvers)
        self.delta_sog = np.zeros(n_maneuvers)
        final_course_man = []
        final_speed_man = []

        for i in range(n_maneuvers):
            if speed_maneuvers and course_maneuvers:
                add_speed_maneuver = speed_maneuvers[0][0] < course_maneuvers[0][0]
            elif speed_maneuvers:
                add_speed_maneuver = True
            else:
                add_speed_maneuver = False

            if add_speed_maneuver:
                [start_idx, stop_idx] = speed_maneuvers.pop(0)
                final_speed_man.append([start_idx, stop_idx])
                self.maneuver_detect_idx[i] = start_idx
                self.delta_sog[i] = np.sum(self.sog_der[start_idx:stop_idx])
            else:
                [start_idx, stop_idx] = course_maneuvers.pop(0)
                final_course_man.append([start_idx, stop_idx])
                self.maneuver_detect_idx[i] = start_idx
                self.delta_cog[i] = np.sum(self.maneuver_der[0, start_idx:stop_idx])

        self.sog_maneuvers_idx = np.array(final_speed_man, dtype=int)
        self.cog_maneuvers_idx = np.array(final_course_man, dtype=int)

    def find_maneuver_detect_index(
        self, epsilon_d_course: float, epsilon_d_speed: float
    ) -> None:
        """
        Find indices i where the *vessel*'s speed and/or course change exceeds *epsilon_speed*
        and/or *epsilon_d_course* respectively. The change is defined as the difference between
        the speed/course at index i and index i + step_length, where the step length is defined by the sample frequency
        of the *vessel*'s state such that the time between sample i and i + step_length is one second.

        Args:
            epsilon_d_course (float): Course change threshold  for maneuver detection, [rad/s].
            epsilon_d_speed (float): Speed change threshold for maneuver detection, [m/s^2].

        Sets the following parameters:

            * :attr:`VesselData.maneuver_detect_idx`
            * :attr:`VesselData.delta_cog`
            * :attr:`VesselData.delta_sog`
        """
        dt = self.timestamps[1] - self.timestamps[0]
        step_length = int(round(1 / dt))
        i_maneuver_detect = []
        delta_course_tot_list = []
        delta_speed_tot_list = []
        delta_speed_tot = 0
        delta_course_tot = 0
        is_course_maneuver_prev = is_speed_maneuver_prev = False
        sign_delta_course_prev = sign_delta_speed_prev = 0
        n_msgs = len(self.timestamps)
        for i in range(step_length, n_msgs, step_length):
            delta_speed_curr = abs(self.sog[i] - self.sog[i - step_length])
            delta_course_curr = mf.wrap_angle_diff_to_02pi(
                self.xy[2, i], self.xy[2, i - step_length]
            )

            is_course_maneuver = delta_course_curr > epsilon_d_course
            is_speed_maneuver = delta_speed_curr > epsilon_d_speed

            if is_course_maneuver:
                sign_delta_course_curr = np.sign(
                    self.sog[i] - self.sog[i - step_length]
                )
                sign_change_course = bool(
                    sign_delta_course_curr + sign_delta_course_prev
                )
                is_new_course_maneuver = (
                    not is_course_maneuver_prev
                ) or sign_change_course
                is_continued_course_maneuver = (
                    is_course_maneuver_prev and not sign_change_course
                )
                is_end_course_maneuver = False
                sign_delta_course_prev = sign_delta_course_curr
            else:
                sign_delta_course_prev = 0
                is_new_course_maneuver = False
                is_continued_course_maneuver = False
                is_end_course_maneuver = is_course_maneuver_prev

            if is_speed_maneuver:
                if (
                    mf.wrap_angle_to_02pi(self.cog[i - step_length] + delta_course_curr)
                    == self.cog[i]
                ):
                    sign_delta_speed_curr = 1
                else:
                    sign_delta_speed_curr = -1
                sign_change_speed = bool(sign_delta_speed_curr + sign_delta_speed_prev)
                is_new_speed_maneuver = (
                    not is_speed_maneuver_prev
                ) or sign_change_speed
                is_continued_speed_maneuver = (
                    is_speed_maneuver_prev and not sign_change_speed
                )
                is_end_speed_maneuver = False
                sign_delta_speed_prev = sign_delta_speed_curr
            else:
                sign_delta_speed_prev = 0
                is_new_speed_maneuver = False
                is_continued_speed_maneuver = False
                is_end_speed_maneuver = is_speed_maneuver_prev

            if is_new_course_maneuver and is_new_speed_maneuver:  # New maneuver started
                i_maneuver_detect.append(i)
                delta_course_tot += delta_course_curr
                delta_speed_tot += delta_speed_curr
            elif is_new_course_maneuver:
                i_maneuver_detect.append(i)
                delta_speed_tot_list.append(np.nan)
                delta_course_tot += delta_course_curr
            elif is_new_speed_maneuver:
                i_maneuver_detect.append(i)
                delta_course_tot_list.append(np.nan)
                delta_speed_tot += delta_speed_curr

            if (
                is_continued_course_maneuver
            ):  # Continued maneuver from previous time step
                delta_course_tot += delta_course_curr
            if is_continued_speed_maneuver:
                delta_speed_tot += delta_speed_curr

            if is_end_course_maneuver:  # End of maneuver
                delta_course_tot_list.append(delta_course_tot)
                delta_course_tot = 0
            if is_end_speed_maneuver:
                delta_speed_tot_list.append(delta_speed_tot)
                delta_speed_tot = 0

            is_course_maneuver_prev = is_course_maneuver
            is_speed_maneuver_prev = is_speed_maneuver

        if is_course_maneuver:  # Save size of maneuver in progress at end of trajectory
            delta_course_tot_list.append(delta_course_tot)
        if is_speed_maneuver:
            delta_speed_tot_list.append(delta_speed_tot)

        self.maneuver_detect_idx = np.array(i_maneuver_detect, dtype=int)
        self.delta_cog = np.array(delta_course_tot_list)
        self.delta_sog = np.array(delta_speed_tot_list)

    def downsample_data(self, sample_time: float) -> None:
        current_sample_time = self.timestamps[1] - self.timestamps[0]
        step = int(sample_time / current_sample_time)
        self.xy = self.xy[:, ::step]
        self.latlon = self.latlon[:, ::step]
        self.cog = self.cog[::step]
        self.sog = self.sog[::step]
        self.timestamps = self.timestamps[::step]
        self.forward_heading_estimate = self.forward_heading_estimate[::step]
        self.backward_heading_estimate = self.backward_heading_estimate[::step]
        self.sog_der = self.sog_der[::step]
        self.maneuver_der = self.maneuver_der[::step]
        self.first_valid_idx, self.last_valid_idx = mhm.index_of_first_and_last_non_nan(
            self.cog
        )

    def compute_closest_grounding_dist(self, enc: senc.ENC):
        """Compute the distance to the closest grounding point along the trajectory.

        Sets the following vessel attributes:
            grounding_dist (float): The distance to the closest grounding point.
            grounding_dist_vec (np.ndarray): The distance vector to the closest grounding point.
            grounding_idx (int): The sample index at which the closest grounding point occurs.

        Args:
            enc (senc.ENC): The Electronic Navigational Chart to check against.

        Returns
            Tuple[float, float]: The distance to the closest grounding point, the distance vector and the sample index at which this occurs.
        """
        self.grounding_dist, self.grounding_dist_vec, self.grounding_idx = (
            mapf.compute_closest_grounding_dist(
                self.xy[:, self.first_valid_idx : self.last_valid_idx + 1],
                self.min_depth,
                enc,
            )
        )

    def plot_trajectory(self) -> plt.Axes:
        _, ax = plt.subplots()
        out = ax.plot(
            self.xy[0, self.first_valid_idx : self.last_valid_idx],
            self.xy[1, self.first_valid_idx : self.last_valid_idx],
            label=self.name,
        )
        for k in range(self.first_valid_idx, self.last_valid_idx, 5):
            ship_poly = mapf.create_ship_polygon(
                self.xy[1, k], self.xy[0, k], self.cog[k], self.length, self.width
            )
            y_ship, x_ship = ship_poly.exterior.xy
            ax.fill(y_ship, x_ship, alpha=0.5, facecolor="b")

        plt.show(block=False)
        return out

    def plot_sog_profile(self) -> None:
        _, ax = plt.subplots()
        ax.plot(self.sog, label=self.name)
        ax.legend(loc="best")
        # plt.show(block=False)

    def plot_cog_profile(self) -> None:
        _, ax = plt.subplots()
        ax.plot(np.rad2deg(self.xy[2]), label=self.name)
        ax.legend(loc="best")
        # plt.show(block=False)

    def plot_maneuver_detection_information(
        self, epsilon_d_course: float, epsilon_speed: float, epsilon_d_speed: float
    ) -> Tuple[list, list]:
        """Plots course, 1st, 2nd and 3rd derivative of course, speed and 1st derivative of speed along with thresholds
        used in maneuver detection.

        Args:
            vessel: Vessel data.
            n_msgs: Number of messages to plot.
            epsilon_speed: Threshold for speed change.
            epsilon_d_speed: Threshold for 1st derivative of speed.

        Returns:
            Tuple[list, list]: Output lists of figure and axes objects.
        """
        figs = []
        axes = []

        scale = 30  # Scaling factor for log plots.
        stat_seqs = np.where(self.sog <= 2)[0]
        stat_seqs = np.split(stat_seqs, np.where(np.diff(stat_seqs) != 1)[0] + 1)

        styles = ["-", ":"]
        style = "-"
        stat_idx = np.empty(0)
        if len(stat_seqs) > 0 and len(stat_seqs[0]) > 0:
            stat_idx = np.array(
                [[seq[0], seq[-1] + 1] for seq in stat_seqs]
            ).flatten()  # start and end idx of static periods

            if stat_idx[0] != 0:  # Moving at start
                styles = [":", "-"] * int(len(stat_idx) / 2)
                style = "-"
            else:  # Static at start
                styles = ["-", ":"] * int(len(stat_idx) / 2)
                style = ":"
            if stat_idx[-1] != len(self.cog):  # Moving at end
                stat_idx = np.append(stat_idx, len(self.cog))

        cp = np.concatenate(
            (stat_idx, self.cog_maneuvers_idx), axis=None
        )  # Change points

        if len(cp) > 0:
            fig1, ax1 = plt.subplots(nrows=4, ncols=1)
            fig1.suptitle(f"Vessel {self.id} course maneuvers")
            axes1 = ax1.flatten()
            idx_order = np.argsort(cp)
            cp = cp[idx_order]

            color = "r"
            colors = ["r", "b"]
            changes = styles + ["r", "b"]
            if len(self.cog_maneuvers_idx) > 0:
                if self.cog_maneuvers_idx[0, 0] != 0:  # Const at start
                    colors = ["r", "b"] * len(self.cog_maneuvers_idx)
                    color = "b"
                else:  # Maneuvering at start
                    colors = ["b", "r"] * len(self.cog_maneuvers_idx)
                    color = "r"

            changes = styles + colors
            changes = np.array(changes)
            changes = changes[idx_order]

            # Plot from start until first change point
            first_cp = int(cp[0])
            axes1[0].plot(
                self.timestamps[0 : first_cp + 1],
                np.rad2deg(self.cog[0 : first_cp + 1]),
                color=color,
                linestyle=style,
            )
            axes1[0].set_ylabel("Course [deg]")
            axes1[1].plot(
                self.timestamps[0 : first_cp + 1],
                np.rad2deg(self.maneuver_der[0, 0 : first_cp + 1]),
                color=color,
                linestyle=style,
            )
            axes1[1].set_ylabel("Course derivative [deg/s]")
            axes1[2].plot(
                self.timestamps[0 : first_cp + 1],
                scale * np.rad2deg(self.maneuver_der[1, 0 : first_cp + 1]),
                color=color,
                linestyle=style,
            )
            axes1[2].set_ylabel("Course 2nd derivative [deg/s²]")
            axes1[3].plot(
                self.timestamps[0 : first_cp + 1],
                scale * np.rad2deg(self.maneuver_der[2, 0 : first_cp + 1]),
                color=color,
                linestyle=style,
            )
            axes1[3].set_ylabel("Course 3rd derivative [deg/s^3]")
            axes1[3].set_xlabel("Time [s]")

            for i in range(0, len(cp) - 1):
                start = int(cp[i])
                end = int(cp[i + 1]) + 1  # Add one to include endpoint in line.
                if end > len(self.cog):
                    end = len(self.cog)

                if changes[i] == ":" or changes[i] == "-":
                    style = changes[i]
                elif changes[i] == "r" or changes[i] == "b":
                    color = changes[i]

                axes1[0].plot(
                    self.timestamps[start:end],
                    np.rad2deg(self.cog[start:end]),
                    color=color,
                    linestyle=style,
                )
                axes1[1].plot(
                    self.timestamps[start:end],
                    np.rad2deg(self.maneuver_der[0, start:end]),
                    color=color,
                    linestyle=style,
                )
                axes1[2].plot(
                    self.timestamps[start:end],
                    scale * np.rad2deg(self.maneuver_der[1, start:end]),
                    color=color,
                    linestyle=style,
                )
                axes1[3].plot(
                    self.timestamps[start:end],
                    scale * np.rad2deg(self.maneuver_der[2, start:end]),
                    color=color,
                    linestyle=style,
                )

            # axes1[0].set_title('Course', loc='center')
            # axes1[0].set_ylabel("[deg]")
            # axes1[0].set_ylim(-75, 40)
            # # axes1[1].set_title('First derivative', loc='center')
            # axes1[1].set_ylabel(r"symlog([deg/sec])")
            # axes1[1].set_yscale("symlog")
            # axes1[1].set_ylim(-6, 10)
            # # axes1[2].set_title('Second derivative', loc='center')
            # axes1[2].set_ylabel(r"symlog([deg/sec$^2$])")
            # axes1[2].set_yscale("symlog")
            # # axes1[3].set_title('Third derivative', loc='center')
            # axes1[3].set_ylabel(r"symlog([deg/sec$^3$])")
            # axes1[3].set_yscale("symlog")
            # axes1[3].set_ylim(scale * -0.6, scale * (0.9 + 0.2))

            sample_interval = self.timestamps[1] - self.timestamps[0]
            for man in self.cog_maneuvers_idx:
                start_pts0 = axes1[0].scatter(
                    man[0] * sample_interval,
                    np.rad2deg(self.cog[man[0]]),
                    color="red",
                    marker="o",
                    label="Detection points",
                )
                end_pts0 = axes1[0].scatter(
                    man[1] * sample_interval,
                    np.rad2deg(self.cog[man[1]]),
                    color="red",
                    marker="x",
                    label="End points",
                )
                start_pts1 = axes1[1].scatter(
                    man[0] * sample_interval,
                    np.rad2deg(self.maneuver_der[0, man[0]]),
                    color="red",
                    marker="o",
                    label="Detection points",
                )
                end_pts1 = axes1[1].scatter(
                    man[1] * sample_interval,
                    np.rad2deg(self.maneuver_der[0, man[1]]),
                    color="red",
                    marker="x",
                    label="End points",
                )
                start_pts3 = axes1[3].scatter(
                    man[0] * sample_interval,
                    scale * np.rad2deg(self.maneuver_der[2, man[0]]),
                    color="red",
                    marker="o",
                    label="Detection points",
                )
                end_pts3 = axes1[3].scatter(
                    man[1] * sample_interval,
                    scale * np.rad2deg(self.maneuver_der[2, man[1]]),
                    color="red",
                    marker="x",
                    label="End points",
                )

            lim1 = axes1[1].axhline(
                np.rad2deg(epsilon_d_course),
                color="darkgreen",
                alpha=0.5,
                label=r"+/- $\epsilon_{\dot{\chi}}$",
            )
            axes1[1].axhline(
                np.rad2deg(-epsilon_d_course), color="darkgreen", alpha=0.5
            )

            if len(self.cog_maneuvers_idx) > 0:
                handles = [start_pts0, end_pts0]
                labels = [start_pts0.get_label(), end_pts0.get_label()]
                axes1[0].legend(
                    handles,
                    labels,
                    loc="upper right",
                    bbox_to_anchor=(1.005, 1.006),
                    prop={"size": 10},
                )
                handles = [start_pts1, end_pts1, lim1]
                labels = [
                    start_pts1.get_label(),
                    end_pts1.get_label(),
                    lim1.get_label(),
                ]
                axes1[1].legend(
                    handles,
                    labels,
                    loc="upper right",
                    bbox_to_anchor=(1.005, 1.006),
                    prop={"size": 10},
                )
                handles = [start_pts3, end_pts3]
                labels = [start_pts3.get_label(), end_pts3.get_label()]
                axes1[3].legend(
                    handles,
                    labels,
                    loc="upper right",
                    bbox_to_anchor=(1.005, 1.006),
                    prop={"size": 10},
                )

            n_msgs = len(self.timestamps)
            for a in axes1:
                a.yaxis.set_major_locator(plt.MaxNLocator(4))
                a.yaxis.set_label_coords(-0.08, 0.5)
                a.set_xlim(0, n_msgs * sample_interval)
                a.set_xlabel("[sec]", fontdict={"family": "Times"})
                # a.set_position([0.135, 0.22, 0.775, 0.75])

            figs.append(fig1)
            axes.extend(axes1)

        # SPEED MANEUVERS ----------------------------------------------------------------------------------------------

        cp = np.concatenate(
            (stat_idx, self.sog_maneuvers_idx), axis=None
        )  # Change points

        if len(cp) > 0:
            fig2, ax2 = plt.subplots(nrows=2, ncols=1)
            fig2.suptitle(f"Vessel {self.id} speed maneuvers")
            axes2 = ax2.flatten()

            idx_order = np.argsort(cp)
            cp = cp[idx_order]

            if stat_idx[0] != 0:  # Moving at start
                style = "-"
            else:  # Static at start
                style = ":"

            if len(self.sog_maneuvers_idx) > 0:
                if self.sog_maneuvers_idx[0, 0] != 0:  # Const at start
                    colors = ["r", "b"] * len(self.sog_maneuvers_idx)
                    color = "b"
                else:  # Maneuvering at start
                    colors = ["b", "r"] * len(self.sog_maneuvers_idx)
                    color = "r"

            changes = styles + colors
            changes = np.array(changes)
            changes = changes[idx_order]

            # Plot from start until first change point
            first_cp = int(cp[0])
            axes2[0].plot(
                self.timestamps[0 : first_cp + 1],
                self.sog[0 : first_cp + 1],
                color=color,
                linestyle=style,
            )
            axes2[1].plot(
                self.timestamps[0 : first_cp + 1],
                self.sog_der[0 : first_cp + 1],
                color=color,
                linestyle=style,
            )
            for i in range(0, len(cp) - 1):
                start = int(cp[i])
                end = int(cp[i + 1]) + 1  # Add one to include endpoint in line.
                if end > len(self.sog_der):
                    end = len(self.sog_der)
                if changes[i] == ":" or changes[i] == "-":
                    style = changes[i]
                elif changes[i] == "r" or changes[i] == "b":
                    color = changes[i]
                axes2[0].plot(
                    self.timestamps[start:end],
                    self.sog[start:end],
                    color=color,
                    linestyle=style,
                )
                axes2[1].plot(
                    self.timestamps[start:end],
                    self.sog_der[start:end],
                    color=color,
                    linestyle=style,
                )

            lim0 = axes2[0].axhline(
                epsilon_speed, color="darkgreen", alpha=0.5, label=r"+/- $\epsilon_{U}$"
            )
            axes2[0].axhline(epsilon_speed, color="darkgreen", alpha=0.5)
            axes2[0].legend(
                [lim0], [lim0.get_label()], loc="lower right", prop={"size": 11}
            )

            lim1 = axes2[1].axhline(
                epsilon_d_speed,
                color="darkgreen",
                alpha=0.5,
                label=r"+/- $\epsilon_{\dot{U}}$",
            )
            axes2[1].axhline(-epsilon_d_speed, color="darkgreen", alpha=0.5)
            axes2[1].legend(
                [lim1],
                [lim1.get_label()],
                loc="upper right",
                bbox_to_anchor=(1.005, 1.006),
                prop={"size": 11},
            )

            # ax[0].set_title('Speed', loc='center')
            axes2[0].set_ylabel("[m/sec]")
            axes2[0].set_xlabel("[sec]")
            axes2[0].set_xlim(0, n_msgs * sample_interval)
            axes2[0].set_ylim(-0.5, 4)
            axes2[0].yaxis.set_major_locator(plt.MaxNLocator(5))
            # axes2[0].set_position([0.135, 0.22, 0.775, 0.75])

            # ax[1].set_title('Acceleration', loc='center')
            axes2[1].set_ylabel(r"[m/sec$^2$]")
            axes2[1].set_xlabel("[sec]")
            axes2[1].set_xlim(0, n_msgs * sample_interval)
            # axes2[1].set_position([0.135, 0.22, 0.775, 0.75])

            figs.append(fig2)
            axes.extend(axes2)

        if figs:
            plt.show(block=False)
        return figs, axes


def determine_status(nav_status_arr: np.ndarray) -> Status:
    mean_status = int(np.mean(nav_status_arr))
    if mean_status == 0:
        status = Status.UnderWayUsingEngine
    elif mean_status == 1:
        status = Status.AtAnchor
    elif mean_status == 2:
        status = Status.NotUnderCommand
    elif mean_status == 3:
        status = Status.RestrictedManeuverability
    elif mean_status == 4:
        status = Status.ConstrainedByHerDraught
    elif mean_status == 5:
        status = Status.Moored
    elif mean_status == 6:
        status = Status.Aground
    elif mean_status == 7:
        status = Status.EngagedInFishing
    elif mean_status == 8:
        status = Status.UnderwaySailing
    else:
        status = Status.NotDefined
    return status


def compute_total_dist_travelled(xy: np.ndarray) -> float:
    """Computes the total distance travelled by a vessel.

    Args:
        xy (np.ndarray): Trajectory in UTM (East-North) coordinates (x, y).

    Returns:
        float: Total distance travelled by the vessel.
    """
    if xy.size < 2:
        return -1.0

    return float(
        np.linalg.norm(
            [
                xy[0, -1] - xy[0, 0],
                xy[1, -1] - xy[1, 0],
            ]
        )
    )
    # return np.sum(np.sqrt((np.diff(xy[0, :]) ** 2 + np.diff(xy[1, :]) ** 2)))


def compute_new_cpa(
    vessel: VesselData,
    obst: VesselData,
    deviating_traj_segment: geometry.LineString,
) -> Tuple[dict, dict, float, int]:
    """
    Calculates new DCPA and CPA positions for obstacle and ownship for an alternative own-ship path.

    Args:
        vessel (VesselData): Ownship vessel data.
        obst (VesselData): Obstacle vessel data.
        deviating_traj_segment (geometry.LineString): The part of the alternative trajectory that deviates from the original path.

    Returns:
        Tuple[dict, dict, float, int]: New DCPA and CPA positions for the ownship and obstacle, and the new DCPA + its index
    """
    dubins_p0, dubins_p1 = deviating_traj_segment.boundary.geoms
    dubins_p0_np = np.array(dubins_p0.coords)
    dubins_p1_np = np.array(dubins_p1.coords)
    own_traj_arr = vessel.xy[
        :, vessel.first_valid_idx : vessel.last_valid_idx + 1
    ].transpose()

    dists = np.sqrt(np.sum((own_traj_arr - dubins_p0_np) ** 2, axis=1))
    start_idx = int(
        np.argmin(dists)
    )  # Index where the alternative path deviates from the original trajectory
    dists = np.sqrt(np.sum((own_traj_arr - dubins_p1_np) ** 2, axis=1))
    end_idx = int(
        np.argmin(dists)
    )  # Index where the alternative path rejoins from the original trajectory

    dt = vessel.timestamps[1] - vessel.timestamps[0]
    dist_dt = (
        vessel.sog[vessel.first_valid_idx + start_idx] * dt
    )  # Calculate approximate travel length per time step
    num_vert = int(round(deviating_traj_segment.length / dist_dt))
    if num_vert == 0:
        num_vert = 1
    interpol_dubins_traj = geometry.LineString(
        [
            deviating_traj_segment.interpolate(
                float(n) / float(num_vert), normalized=True
            )
            for n in range(num_vert + 1)
        ]
    )
    interpol_dubins_traj_arr = np.array(interpol_dubins_traj.coords)
    alt_traj_arr = np.concatenate(
        (
            own_traj_arr[vessel.first_valid_idx : start_idx],
            interpol_dubins_traj_arr,
            own_traj_arr[end_idx : vessel.last_valid_idx + 1],
        )
    ).transpose()

    new_dcpa = 1e20
    new_cpa_idx = 0
    for i, _ in enumerate(obst.timestamps):
        if i < obst.first_valid_idx or i > obst.last_valid_idx:
            continue

        temp_dist = float(np.linalg.norm(obst.xy[:, i] - alt_traj_arr[:, i]))
        if temp_dist < new_dcpa:
            new_dcpa = temp_dist
            new_cpa_idx = i

    cpa_forward_heading_estimate = math.atan2(
        (alt_traj_arr[1, new_cpa_idx + 1] - alt_traj_arr[1, new_cpa_idx]),
        (alt_traj_arr[0, new_cpa_idx + 1] - alt_traj_arr[0, new_cpa_idx]),
    )
    cpa_forward_heading_estimate = np.rad2deg(
        -mf.wrap_angle_to_pmpi(cpa_forward_heading_estimate - np.pi / 2)
    )
    obst_cpa_forward_heading_estimate = np.rad2deg(
        obst.forward_heading_estimate[new_cpa_idx]
    )
    own_cpa = {
        "x": alt_traj_arr[0, new_cpa_idx],
        "y": alt_traj_arr[1, new_cpa_idx],
        "psi": cpa_forward_heading_estimate,
    }
    obst_cpa = {
        "x": obst.xy[0, new_cpa_idx],
        "y": obst.xy[1, new_cpa_idx],
        "psi": obst_cpa_forward_heading_estimate,
    }
    return own_cpa, obst_cpa, new_dcpa, new_cpa_idx
