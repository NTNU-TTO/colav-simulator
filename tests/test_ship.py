"""
Test module for the Ship class. Use it to test the ship's behavior in a scenario, and/or tune your specific guidance algorithm + controller.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.plotters as plotters
import colav_simulator.core.controllers as controllers
import colav_simulator.core.guidances as guidances
import colav_simulator.core.models as models
import colav_simulator.core.sensing as sensorss
import colav_simulator.core.ship as ship
import colav_simulator.core.stochasticity as stochasticity
import colav_simulator.core.tracking.trackers as trackers
from colav_simulator.scenario_config import OwnshipPositionGenerationMethod
from colav_simulator.scenario_generator import ScenarioGenerator


def test_ship() -> None:
    fig_size = [25, 13]  # figure1 size in cm
    dpi_value = 150  # figure dpi value
    horizon = 150.0
    dt = 0.5  # NOTE: time step affects the dynamics accuracy and also control performance

    utm_zone = 33
    map_size = [1400.0, 1500.0]
    map_orig = np.array([6573700.0, -31924.0])
    map_data_files = [
        str(Path.home() / "enc_data" / "Rogaland_utm33.gdb")
    ]  # You need to create this folder and put a downloaded Rogaland .gdb file in it. See the README.md file for more information.

    # Put new_data to True to load map data in ENC if it is not already loaded
    scenario_generator = ScenarioGenerator(
        init_enc=True,
        new_data=True,
        utm_zone=utm_zone,
        size=map_size,
        origin=[map_orig[1], map_orig[0]],
        files=map_data_files,
    )
    origin = scenario_generator.enc_origin

    model = models.Viknes()
    ctrl_params = controllers.FLSCParams(
        K_p_u=3.0,
        K_i_u=0.3,
        K_p_chi=2.2,
        K_d_chi=4.0,
        K_i_chi=0.1,
        max_speed_error_int=4.0,
        speed_error_int_threshold=0.5,
        max_chi_error_int=90.0 * np.pi / 180.0,
        chi_error_int_threshold=20.0 * np.pi / 180.0,
    )
    controller = controllers.FLSC(model.params, ctrl_params)
    sensor_list = [sensorss.Radar()]
    tracker = trackers.KF(sensor_list=sensor_list)
    guidance_params = guidances.LOSGuidanceParams(
        K_p=0.01,
        K_i=0.0004,
        R_a=80.0,
        max_cross_track_error_int=500.0,
        cross_track_error_int_threshold=100.0,
        pass_angle_threshold=90.0,
    )
    guidance_method = guidances.LOSGuidance(guidance_params)

    ownship = ship.Ship(
        mmsi=1,
        identifier=0,
        model=model,
        controller=controller,
        tracker=tracker,
        sensors=sensor_list,
        guidance=guidance_method,
    )

    scenario_generator.seed(1)
    csog_state = scenario_generator.generate_random_csog_state(
        method=OwnshipPositionGenerationMethod.UniformInTheMapThenGaussian,
        draft=ownship.draft,
        min_hazard_clearance=100.0,
        U_min=2.0,
        U_max=ownship.max_speed,
    )
    # csog_state = np.array(
    #     [6574229.448438326, -31157.753734698883, 5.805685027679189, -131.1969202238676 * np.pi / 180.0]
    # )
    # ownship.set_initial_state(csog_state)
    ownship._state = np.array([6574512.209, -31657.653, 1.513, 4.012, 2.421, -0.057])

    disturbance_config = stochasticity.Config()
    disturbance = stochasticity.Disturbance(disturbance_config)
    # disturbance.disable_wind()
    # disturbance._currents = None
    # disturbance._wind = None

    rng = np.random.default_rng(seed=1)
    enc = scenario_generator.enc
    safe_sea_cdt = scenario_generator.safe_sea_cdt
    safe_sea_cdt_weights = scenario_generator.safe_sea_cdt_weights
    scenario_generator.behavior_generator.initialize_data_structures(1)
    scenario_generator.behavior_generator.setup_enc(
        enc=enc, safe_sea_cdt=safe_sea_cdt, safe_sea_cdt_weights=safe_sea_cdt_weights
    )
    scenario_generator.behavior_generator.setup_ship(
        rng=rng,
        ship_obj=ownship,
        replan=True,
        simulation_timespan=horizon,
        show_plots=True,
    )

    n_wps = 4
    waypoints, _ = scenario_generator.behavior_generator.generate_random_waypoints(
        rng,
        x=csog_state[0],
        y=csog_state[1],
        psi=csog_state[3],
        draft=ownship.draft,
        n_wps=n_wps,
    )
    speed_plan = (
        4.0 * np.ones(waypoints.shape[1])
    )  # = scenario_generator.generate_random_speed_plan(U=5.0, n_wps=waypoints.shape[1])
    ownship.set_nominal_plan(waypoints=waypoints, speed_plan=speed_plan)

    disturbance_config = stochasticity.Config()
    disturbance_config.currents = stochasticity.GaussMarkovDisturbanceParams(
        constant=False,
        initial_speed=1.9496385863730494,
        initial_direction=-2.9234709165353365,
        speed_range=(0.0, 2.0),
        direction_range=(-3.141592653589793, 3.141592653589793),
        mu_speed=0.0001,
        mu_direction=0.0005,
        sigma_speed=0.005,
        sigma_direction=0.005,
        add_impulse_noise=False,
        speed_impulses=[5.0],
        direction_impulses=np.array([0.785]),
        impulse_times=[82.0],
    )
    disturbance_config.wind = stochasticity.GaussMarkovDisturbanceParams(
        constant=False,
        initial_speed=2.185481173407966,
        initial_direction=171.7304503635763,
        speed_range=(0.0, 4.0),
        direction_range=(-3.141592653589793, 3.141592653589793),
        mu_speed=0.0001,
        mu_direction=0.0005,
        sigma_speed=0.005,
        sigma_direction=0.005,
        add_impulse_noise=False,
        speed_impulses=[1.0, 2.5, 2.0],
        direction_impulses=np.array([-1.571, -0.785, 0.0, 0.785, 1.571, 2.094, 3.142]),
        impulse_times=[131.0],
    )
    disturbance = stochasticity.Disturbance(disturbance_config)
    # disturbance._currents = None
    # disturbance._wind = None

    n_x, n_u = model.dims
    n_r = 9
    n_samples = round(horizon / dt)
    disturbances = np.zeros((10, n_samples))
    trajectory = np.zeros((n_x, n_samples))
    refs = np.zeros((n_r, n_samples))
    tau = np.zeros((n_u, n_samples))
    time = np.zeros(n_samples)

    traj = np.array(
        [
            [
                6574512.0,
                6574505.5,
                6574501.0,
                6574499.0,
                6574498.5,
                6574500.5,
                6574504.0,
                6574507.5,
                6574509.0,
                6574509.5,
                6574510.0,
                6574510.5,
                6574511.0,
                6574511.0,
                6574511.5,
                6574512.0,
                6574512.5,
                6574513.0,
                6574513.0,
                6574513.5,
                6574514.0,
                6574514.5,
                6574515.0,
                6574515.0,
                6574515.5,
                6574516.0,
                6574516.5,
                6574517.0,
                6574517.5,
                6574517.5,
                6574518.0,
            ],
            [
                -31657.652,
                -31652.16,
                -31645.988,
                -31639.988,
                -31633.645,
                -31626.3,
                -31618.887,
                -31611.207,
                -31602.885,
                -31594.441,
                -31586.01,
                -31577.588,
                -31569.176,
                -31560.773,
                -31552.38,
                -31543.998,
                -31535.625,
                -31527.262,
                -31518.906,
                -31510.559,
                -31502.22,
                -31493.889,
                -31485.564,
                -31477.246,
                -31468.932,
                -31460.623,
                -31452.318,
                -31444.016,
                -31435.715,
                -31427.416,
                -31419.117,
            ],
            [
                2.599,
                2.32,
                2.04,
                1.761,
                1.482,
                1.203,
                1.002,
                1.281,
                1.52,
                1.521,
                1.522,
                1.523,
                1.523,
                1.524,
                1.524,
                1.524,
                1.523,
                1.523,
                1.523,
                1.523,
                1.522,
                1.522,
                1.522,
                1.521,
                1.521,
                1.521,
                1.521,
                1.521,
                1.521,
                1.521,
                1.521,
            ],
            [
                4.686,
                4.086,
                3.486,
                2.886,
                3.486,
                4.086,
                4.241,
                4.235,
                4.229,
                4.224,
                4.219,
                4.213,
                4.208,
                4.203,
                4.198,
                4.194,
                4.189,
                4.184,
                4.18,
                4.176,
                4.172,
                4.169,
                4.166,
                4.163,
                4.161,
                4.159,
                4.157,
                4.156,
                4.155,
                4.154,
                4.154,
            ],
            [
                143.111,
                151.123,
                158.905,
                166.202,
                173.502,
                181.287,
                189.378,
                197.531,
                205.681,
                213.827,
                221.969,
                230.109,
                238.244,
                246.376,
                254.504,
                262.628,
                270.749,
                278.868,
                286.988,
                295.109,
                303.232,
                311.358,
                319.488,
                327.621,
                335.755,
                343.881,
                351.991,
                360.088,
                368.182,
                376.282,
                384.387,
            ],
            [
                4.0,
                4.012,
                3.77,
                3.528,
                3.771,
                4.014,
                4.077,
                4.076,
                4.074,
                4.072,
                4.07,
                4.069,
                4.067,
                4.065,
                4.063,
                4.061,
                4.06,
                4.06,
                4.06,
                4.061,
                4.062,
                4.064,
                4.066,
                4.067,
                4.067,
                4.059,
                4.051,
                4.046,
                4.048,
                4.051,
                4.054,
            ],
        ],
        dtype=np.float32,
    )

    ref_counter = 2
    t_prev_upd = 0.0
    chi_ref = traj[2, 1]
    U_ref = traj[3, 1]
    for k in range(n_samples):
        time[k] = k * dt
        disturbance_data = disturbance.get()
        # disturbance_data.currents = {"speed": 1.0429484, "direction": 0.59327924}
        # disturbance_data.wind = {"speed": 6.546729, "direction": -3.7480865}
        if disturbance_data.wind:
            disturbances[0, k] = disturbance_data.wind["speed"]
            disturbances[1, k] = disturbance_data.wind["direction"]
        if disturbance_data.currents:
            disturbances[2, k] = disturbance_data.currents["speed"]
            disturbances[3, k] = disturbance_data.currents["direction"]

        # ownship.plan(time[k], dt, [], None, w=disturbance_data)
        if time[k] - t_prev_upd >= 2.0:
            ref_counter += 1 if ref_counter < traj.shape[1] - 1 else 0
            t_prev_upd = time[k]
            chi_ref = traj[2, ref_counter]
            U_ref = traj[3, ref_counter]

        ownship.set_references(
            np.array([0.0, 0.0, chi_ref, U_ref, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        trajectory[:, k], tau[:, k], refs[:, k] = ownship.forward(
            dt, w=disturbance_data
        )
        disturbance.update(time[k], dt)

    # Plots
    scenario_generator.enc.start_display()
    if disturbance_data.currents and disturbance_data.currents["speed"] > 0.0:
        plotters.plot_disturbance(
            magnitude=100.0,
            direction=disturbance_data.currents["direction"],
            name="current: " + str(disturbance_data.currents["speed"]) + " m/s",
            enc=enc,
            color="white",
            linewidth=1.0,
            location="topright",
            text_location_offset=(0.0, 0.0),
        )

    if disturbance_data.wind and disturbance_data.wind["speed"] > 0.0:
        plotters.plot_disturbance(
            magnitude=100.0,
            direction=disturbance_data.wind["direction"],
            name="wind: " + str(disturbance_data.wind["speed"]) + " m/s",
            enc=enc,
            color="peru",
            linewidth=1.0,
            location="topright",
            text_location_offset=(0.0, -20.0),
        )
    plotters.plot_waypoints(
        traj[:2, :],
        scenario_generator.enc,
        "orange",
        point_buffer=2.0,
        disk_buffer=4.0,
        hole_buffer=2.0,
        alpha=0.6,
    )
    plotters.plot_trajectory(trajectory, scenario_generator.enc, "black")
    for k in range(0, n_samples, 40):
        ship_poly = mapf.create_ship_polygon(
            trajectory[0, k],
            trajectory[1, k],
            trajectory[2, k],
            ownship.length,
            ownship.width,
        )
        scenario_generator.enc.draw_polygon(ship_poly, "magenta", fill=True)

    # States
    fig = plt.figure(
        figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value
    )
    axs = fig.subplot_mosaic(
        [
            ["xy", "chi", "r"],
            ["U", "u", "v"],
        ]
    )
    axs["xy"].plot(
        traj[1, :] - origin[1], traj[0, :] - origin[0], "rx", label="Waypoints"
    )
    axs["xy"].plot(
        trajectory[1, :] - origin[1],
        trajectory[0, :] - origin[0],
        "k",
        label="Trajectory",
    )
    axs["xy"].set_xlabel("East (m)")
    axs["xy"].set_ylabel("North (m)")
    axs["xy"].grid()
    axs["xy"].legend()

    axs["u"].plot(time, refs[3], "r--", label="Surge reference")
    axs["u"].plot(time, trajectory[3], "k", label="Surge")
    axs["u"].set_xlabel("Time (s)")
    axs["u"].set_ylabel("North (m)")
    axs["u"].grid()
    axs["u"].legend()

    axs["v"].plot(time, refs[4], "r--", label="Sway reference")
    axs["v"].plot(time, trajectory[4], "k", label="Sway")
    axs["v"].set_xlabel("Time (s)")
    axs["v"].set_ylabel("East (m)")
    axs["v"].grid()
    axs["v"].legend()

    # heading_error = mf.wrap_angle_diff_to_pmpi(refs[2, :], trajectory[2, :])
    crab = np.arctan2(trajectory[4, :], trajectory[3, :])
    axs["chi"].plot(
        time,
        np.rad2deg(mf.wrap_angle_to_pmpi(refs[2, :])),
        "r--",
        label="Course reference",
    )
    axs["chi"].plot(
        time,
        np.rad2deg(mf.wrap_angle_to_pmpi(trajectory[2, :] + crab)),
        "k",
        label="Course",
    )
    axs["chi"].set_xlabel("Time (s)")
    axs["chi"].set_ylabel("Course (deg)")
    axs["chi"].grid()
    axs["chi"].legend()

    axs["r"].plot(
        time,
        np.rad2deg(mf.wrap_angle_to_pmpi(refs[5, :])),
        "r--",
        label="Yaw rate reference",
    )
    axs["r"].plot(
        time, np.rad2deg(mf.wrap_angle_to_pmpi(trajectory[5])), "k", label="Yaw rate"
    )
    axs["r"].set_xlabel("Time (s)")
    axs["r"].set_ylabel("Angular rate rate (deg/s)")
    axs["r"].grid()
    axs["r"].legend()

    U_d = np.sqrt(refs[3] ** 2 + refs[4] ** 2)
    U = np.sqrt(trajectory[3, :] ** 2 + trajectory[4, :] ** 2)
    axs["U"].plot(time, U_d, "r--", label="Speed reference")
    axs["U"].plot(time, U, "k", label="Speed")
    axs["U"].set_xlabel("Time (s)")
    axs["U"].set_ylabel("Speed (m/s)")
    axs["U"].grid()
    axs["U"].legend()

    # Disturbances
    fig = plt.figure(
        figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value
    )
    axs = fig.subplot_mosaic(
        [["wind_speed", "wind_direction"], ["current_speed", "current_direction"]]
    )

    axs["wind_speed"].plot(time, disturbances[0, :], "k", label="Wind speed")
    axs["wind_speed"].set_xlabel("Time (s)")
    axs["wind_speed"].set_ylabel("Speed (m/s)")
    axs["wind_speed"].grid()
    axs["wind_speed"].legend()

    axs["wind_direction"].plot(
        time, np.rad2deg(disturbances[1, :]), "k", label="Wind direction"
    )
    axs["wind_direction"].set_xlabel("Time (s)")
    axs["wind_direction"].set_ylabel("Direction (deg)")
    axs["wind_direction"].grid()
    axs["wind_direction"].legend()

    axs["current_speed"].plot(time, disturbances[2, :], "k", label="Current speed")
    axs["current_speed"].set_xlabel("Time (s)")
    axs["current_speed"].set_ylabel("Speed (m/s)")
    axs["current_speed"].grid()
    axs["current_speed"].legend()

    axs["current_direction"].plot(
        time, np.rad2deg(disturbances[3, :]), "k", label="Current direction"
    )
    axs["current_direction"].set_xlabel("Time (s)")
    axs["current_direction"].set_ylabel("Direction (deg)")
    axs["current_direction"].grid()
    axs["current_direction"].legend()

    # Inputs
    if n_u == 3:
        fig = plt.figure(
            figsize=(mf.cm2inch(fig_size[0]), mf.cm2inch(fig_size[1])), dpi=dpi_value
        )
        axs = fig.subplot_mosaic(
            [
                ["X"],
                ["Y"],
                ["N"],
            ]
        )
        axs["X"].plot(time, tau[0, :], "k", label="Surge force")
        axs["X"].set_xlabel("Time (s)")
        axs["X"].set_ylabel("Force (N)")
        axs["X"].grid()
        axs["X"].legend()

        axs["Y"].plot(time, tau[1, :], "k", label="Sway force")
        axs["Y"].set_xlabel("Time (s)")
        axs["Y"].set_ylabel("Force (N)")
        axs["Y"].grid()
        axs["Y"].legend()

        axs["N"].plot(time, tau[2, :], "k", label="Yaw moment")
        axs["N"].set_xlabel("Time (s)")
        axs["N"].set_ylabel("Moment (Nm)")
        axs["N"].grid()
        axs["N"].legend()

    plt.show()


if __name__ == "__main__":
    test_ship()
