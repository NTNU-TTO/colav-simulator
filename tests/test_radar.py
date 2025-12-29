"""Shows how the radar sensor can be used to generate measurements for a single ownship and a single dynamic obstacle."""

import matplotlib.pyplot as plt
import numpy as np

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm
from colav_simulator.core import sensing


def test_radar() -> None:  # noqa: PLR0915
    """Test the radar sensor."""
    rparams = sensing.RadarParams()
    rparams.generate_clutter = True
    rparams.measurement_rate = 0.5
    rparams.clutter_cardinality_expectation = 20
    rparams.include_polar_meas_noise = True
    rparams.max_range = 2000.0
    radar = sensing.Radar(rparams)
    radar.reset(seed=0)

    ownship_state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    true_do_states = [
        (0, np.array([50.0, 0.0, 0.0, 2.0]), 10.0, 3.0)
    ]  # dynamic obstacle info on the form (ID, state, length, width)

    _, ax1 = plt.subplots()
    ax1.set_aspect("equal")
    ax1.set_xlabel("East [m]")
    ax1.set_ylabel("North [m]")
    ax1.set_xlim(-rparams.max_range, rparams.max_range)
    ax1.set_ylim(-rparams.max_range, rparams.max_range)
    dt = 0.5
    for k in range(50):
        t = k * dt
        ownship_state_vxvy = mhm.convert_state_to_vxvy_state(ownship_state)
        meas = radar.generate_measurements(t, true_do_states, ownship_state_vxvy)

        for m in meas:
            if np.any(np.isnan(m[1])):
                continue
            mcolor = "ro"  # clutter
            if m[0] >= 0:
                mcolor = "go"  # target measurement
            ax1.plot(m[1][1], m[1][0], mcolor, markersize=5)

        os_poly = mapf.create_ship_polygon(
            x=ownship_state[0], y=ownship_state[1], heading=ownship_state[2], length=10.0, width=2.0
        )
        ax1.fill(*os_poly.exterior.xy, color="b", alpha=0.5)

        do_poly = mapf.create_ship_polygon(
            x=true_do_states[0][1][0], y=true_do_states[0][1][1], heading=np.pi / 2, length=10.0, width=3.0
        )
        ax1.fill(*do_poly.exterior.xy, true_do_states[0][1][0], color="orangered")

        ownship_state = ownship_state + dt * np.array(
            [
                ownship_state[3] * np.cos(ownship_state[2]),
                ownship_state[3] * np.sin(ownship_state[2]),
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        do_state = true_do_states[0][1] + dt * np.array(
            [
                true_do_states[0][1][2],
                true_do_states[0][1][3],
                0.0,
                0.0,
            ]
        )
        true_do_states = [(0, do_state, 10.0, 3.0)]

    ownship_state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    true_do_states = [
        (0, np.array([50.0, 0.0, 0.0, 2.0]), 10.0, 3.0)
    ]  # dynamic obstacle info on the form (ID, state, length, width)

    rparams.generate_clutter = False
    radar = sensing.Radar(rparams)
    radar.reset(seed=0)
    fig2, ax2 = plt.subplots()
    ax2.set_aspect("equal")
    ax2.set_xlabel("East [m]")
    ax2.set_ylabel("North [m]")
    ax2.set_xlim(-50.0, 0.1 * rparams.max_range)
    ax2.set_ylim(0.0, 0.1 * rparams.max_range)
    for k in range(50):
        t = k * dt

        ownship_state_vxvy = mhm.convert_state_to_vxvy_state(ownship_state)
        meas = radar.generate_measurements(t, true_do_states, ownship_state_vxvy)

        for m in meas:
            if np.any(np.isnan(m[1])):
                continue
            mcolor = "ro"  # clutter
            if m[0] >= 0:
                mcolor = "go"  # target measurement
            ax2.plot(m[1][1], m[1][0], mcolor, markersize=5)

        os_poly = mapf.create_ship_polygon(
            x=ownship_state[0], y=ownship_state[1], heading=ownship_state[2], length=10.0, width=2.0
        )
        ax2.fill(*os_poly.exterior.xy, color="b", alpha=0.5)

        do_poly = mapf.create_ship_polygon(
            x=true_do_states[0][1][0], y=true_do_states[0][1][1], heading=np.pi / 2, length=10.0, width=3.0
        )
        ax2.fill(*do_poly.exterior.xy, true_do_states[0][1][0], color="orangered")

        ownship_state = ownship_state + dt * np.array(
            [
                ownship_state[3] * np.cos(ownship_state[2]),
                ownship_state[3] * np.sin(ownship_state[2]),
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        do_state = true_do_states[0][1] + dt * np.array(
            [
                true_do_states[0][1][2],
                true_do_states[0][1][3],
                0.0,
                0.0,
            ]
        )
        true_do_states = [(0, do_state, 10.0, 3.0)]

    radar.reset(seed=0)
    plt.show(block=False)


if __name__ == "__main__":
    test_radar()
