"""file_utils.py.

Summary:
Contains general non-math related utility functions.

Author: Trym Tengesdal
"""

from pathlib import Path

import pandas as pd
import yaml

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.vessel_data as vd


def read_yaml_into_dict(file_name: Path) -> dict:
    """Reads a yaml file into a dictionary.

    Args:
        file_name (Path): Path to the yaml file.

    Returns:
        dict: Dictionary containing the yaml file data.
    """
    with file_name.open(mode="r", encoding="utf-8") as file:
        output_dict = yaml.safe_load(file)
    return output_dict


def delete_files_in_folder(folder: Path) -> None:
    """Deletes all files in the specified folder, but not the folder itself.

    Does nothing if the folder does not exist.

    Args:
        folder (Path): Path to the folder containing the files to be deleted.
    """
    if not folder.is_dir():
        return

    for file in folder.iterdir():
        if file.is_file():
            file.unlink()


def read_ais_data(
    ais_path: Path,
    ship_info_path: Path | None = None,
    utm_zone: int = 33,
    map_origin_enu: tuple[float, float] | None = None,
    map_size: tuple[float, float] | None = None,
    sample_interval: float = 1.0,
) -> dict:
    """Reads the ais data file and creates a list of VesselData instances.

    For each vessel recorded in the data. The list of MMSI`s and map
    origin/reference (in ENU) are also returned.

    Args:
        ais_path (Path): Path to the ais data file.
        ship_info_path (Path | None): Path to the ship information data file.
            Defaults to None.
        utm_zone (int): UTM zone of the coordinate system. Defaults to 33.
        map_origin_enu (tuple[float, float] | None): Origin of the coordinate
            system in ENU coordinates. Defaults to None.
        map_size (tuple[float, float] | None): Size of the considered area,
            relative to the origin, ENU coordinates. Defaults to None.
        sample_interval (float): Sampling interval used for interpolation on
            the vessel data. Defaults to 1.0.

    Returns:
        dict: Dictionary containing:
        - List of VesselData instances
        - List of vessel MMSI
        - Reference/origin of the local coordinate system in ENU coordinates.
        - Timespan of the data.
        - Tuple of size_x, size_y of the map area (+ buffer) containing the
            data, referenced to the origin.
        - Extent of the map area (+ buffer) containing the data, in lat/lon
            coordinates ([lat_min, lat_max, lon_min, lon_max]).
    """
    output = {}
    vessels = []
    mmsi_list = []
    ship_info_df = None
    if ais_path.is_file():
        ais_df = pd.read_csv(ais_path, sep=";", parse_dates=["date_time_utc"], infer_datetime_format=True)
    else:
        raise FileNotFoundError(f"AIS data file not found: {ais_path}")

    if ship_info_path is not None and ship_info_path.is_file():
        ship_info_df = pd.read_csv(
            ship_info_path, sep=";", dtype={"mmsi": "uint32", "length": "float16", "width": "float16"}
        )

    origin_buffer = 0.01
    lat0 = min(ais_df.lat) - 0.01
    lon0 = min(ais_df.lon) - 0.01
    lat_max = max(ais_df.lat) + origin_buffer
    lon_max = max(ais_df.lon) + origin_buffer

    if map_origin_enu is None:
        map_origin_enu = mapf.latlon2local(lat0, lon0, utm_zone=utm_zone)

    size: list = [0.0, 0.0]
    if map_size is None:
        size[0] = mapf.dist_between_latlon_coords(lat0, lon0, lat0, lon_max)
        size[1] = mapf.dist_between_latlon_coords(lat0, lon0, lat_max, lon0)
    else:
        size = list(map_size)

    t_0 = ais_df.date_time_utc.min()
    t_end = ais_df.date_time_utc.max()

    count = 0
    ship_ais_df_list = mhm.get_ship_ais_df_list_from_ais_df(ais_df)
    for ship_ais_df in ship_ais_df_list:
        vessel = vd.VesselData.create_from_ais_data(
            t_0_global=t_0,
            t_end_global=t_end,
            identifier=count,
            ship_ais_df=ship_ais_df,
            ship_info_df=ship_info_df,
            utm_zone=utm_zone,
            sample_interval=sample_interval,
        )
        if vessel is not None and vessel.status != vd.Status.AtAnchor:
            vessels.append(vessel)
            mmsi_list.append(vessel.mmsi)
            # vessel.plot_trajectory()
            count += 1

    output["vessels"] = vessels
    output["mmsi_list"] = mmsi_list
    output["map_origin_enu"] = list(map_origin_enu)
    output["map_size"] = size
    output["timespan"] = [t_0, t_end]
    output["lla_extent"] = [lat0, lon0, lat_max, lon_max]
    return output
