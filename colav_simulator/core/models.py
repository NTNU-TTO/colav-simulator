"""
models.py

Summary:
    Contains class definitions for various models.
    Every model class must adhere to the interface IModel.

Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import numpy as np

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf
import colav_simulator.core.stochasticity as stochasticity


@dataclass
class KinematicCSOGParams:
    """Parameter class for the Course and SPeed over Ground model."""

    name: str = "KinematicCSOG"
    draft: float = 0.5
    length: float = 10.0
    ship_vertices: np.ndarray = field(default_factory=lambda: np.empty(2))
    width: float = 3.0
    T_chi: float = 3.0
    T_U: float = 5.0
    r_max: float = float(np.deg2rad(4))
    U_min: float = 0.0
    U_max: float = 15.0

    @classmethod
    def from_dict(self, params_dict: dict):
        params = KinematicCSOGParams(
            draft=params_dict["draft"],
            length=params_dict["length"],
            width=params_dict["width"],
            ship_vertices=np.empty(2),
            T_chi=params_dict["T_chi"],
            T_U=params_dict["T_U"],
            r_max=np.deg2rad(params_dict["r_max"]),
            U_min=params_dict["U_min"],
            U_max=params_dict["U_max"],
        )
        params.ship_vertices = np.array(
            [
                [params.length / 2.0, -params.width / 2.0],
                [params.length / 2.0, params.width / 2.0],
                [-params.length / 2.0, params.width / 2.0],
                [-params.length / 2.0, -params.width / 2.0],
            ]
        ).T
        return params

    def to_dict(self):
        output_dict = asdict(self)
        output_dict["ship_vertices"] = self.ship_vertices.tolist()
        output_dict["r_max"] = float(np.rad2deg(self.r_max))
        return output_dict


@dataclass
class ViknesParams:
    """Parameters for the Viknes vessel (read only / fixed)."""

    name: str = "Viknes"
    draft: float = 0.5
    length: float = 8.45
    width: float = 2.71
    ship_vertices: np.ndarray = field(
        default_factory=lambda: np.array(
            [[3.75, 1.5], [4.25, 0.0], [3.75, -1.5], [-3.75, -1.5], [-3.75, 1.5]]
        ).T
    )
    l_r: float = 4.0  # Distance from CG to rudder
    M_rb: np.ndarray = field(
        default_factory=lambda: np.diag([3980.0, 3980.0, 19703.0])
    )  # Rigid body mass matrix
    M_a: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    D_c: np.ndarray = field(
        default_factory=lambda: np.diag([0.0, 0.0, 3224.0])
    )  # Third order/cubic damping
    D_q: np.ndarray = field(
        default_factory=lambda: np.diag([135.0, 2000.0, 0.0])
    )  # Second order/quadratic damping
    D_l: np.ndarray = field(
        default_factory=lambda: np.diag([50.0, 200.0, 1281.0])
    )  # First order/linear damping
    Fx_limits: np.ndarray = field(
        default_factory=lambda: np.array([-6550.0, 13100.0])
    )  # Force limits in x
    Fy_limits: np.ndarray = field(
        default_factory=lambda: np.array([-645.0, 645.0])
    )  # Force limits in y
    r_max: float = float(np.deg2rad(15))
    U_min: float = 0.0
    U_max: float = 10.0
    A_Fw: float = 3.0 * 1.5  # guesstimate of frontal area
    A_Lw: float = 8.0 * 1.5  # guesstimate of lateral area
    rho_air: float = 1.225  # Density of air
    CD_l_AF_0: float = (
        0.55  # Longitudinal resistance used to compute wind coefficients in wind model
    )
    CD_l_AF_pi: float = 0.65
    CD_t: float = (
        0.85  # Transversal resistance used to compute wind coefficients in wind model
    )
    delta_crossforce: float = 0.60  # Cross-force parameter
    s_L: float = 0.0  # x-coordinate of transverse prject area A_Lw wrt the main section

    # NB! Very crude assumed/guessed values.
    A_Fw: float = 3.5 * width  # Guess 3.5 m height
    A_Lw: float = (
        0.45 * 3.5 * length
    )  # Guess 3.5 m height, the cab covers about half of the length, front is open
    rho_air: float = 1.225  # Density of air
    CD_l_AF_0: float = 0.55  # Guess longitudinal resistance used to compute wind coefficients in wind model (gamma_w = 0). Table 10.3 Fossen 2011.
    CD_l_AF_pi: float = 0.60  # Guess longitudinal resistance used to compute wind coefficients in wind model (gamma_w = pi) Table 10.3 Fossen 2011.
    CD_t: float = 0.85  # Guess transversal resistance used to compute wind coefficients in wind model. Table 10.3 Fossen 2011: Research vessel, chosen due to expectation of lower sim speeds
    delta_crossforce: float = (
        0.60  # Guess cross-force parameter. Table 10.3 Fossen 2011.
    )
    s_L: float = -1.0  # Guess x-coordinate of the centre of "A_lw", vessel is asymmetric, see sideprofile


@dataclass
class RVGunnerusParams:
    """Parameters for the R/V Gunnerus vessel (read only / fixed)."""

    name: str = "R/V Gunnerus"
    rho: float = 1000.0  # Density of water
    draft: float = 2.7
    length: float = 31.25  # (Loa)
    width: float = 9.6
    ship_vertices: np.ndarray = field(
        default_factory=lambda length=length, width=width: np.array(
            [
                [0.9 * length / 2.0, width / 2.0],
                [length / 2.0, 0.0],
                [0.9 * length / 2.0, -width / 2.0],
                [-length / 2.0, -width / 2.0],
                [-length / 2.0, width / 2.0],
            ]
        ).T
    )

    M_rb: np.ndarray = field(
        default_factory=lambda: np.array(
            [[574127.69, 0.0, 0.0], [0.0, 574127.69, 0.0], [0.0, 0.0, 41237080.0]]
        )
    )  # Rigid body mass matrix, m = 574127.69 kg, I_z = 41237080 kgm²
    M_a: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [26599.24721721, 0.0, 0.0],
                [0.0, 132619.44, -473277.78],
                [0.0, -571162.06, 13320142.0],
            ]
        )
    )  # Added mass matrix
    D_l: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.11759831e03, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 2.22886400e04, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 1.94968656e06],
            ]
        )
    )  # First order/linear damping
    D_u: np.ndarray = field(
        default_factory=lambda: np.array(
            [[1671.60006965, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
    )  # Quadratic damping matrix that multiplies with relative surge speed u_r
    D_v: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0.0, 0.0, 0.0], [0.0, 23611.351393, 0.0], [0.0, 0.0, 0.0]]
        )
    )  # Quadratic damping matrix that multiplies with relative sway speed v_r
    D_r: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 3.65427334e08],
            ]
        )
    )  # Quadratic damping matrix that multiplies with yaw rate r

    # Actuator parameters
    r_t: np.array = field(
        default_factory=lambda: np.array([-15.5600097, 0.0, -4.4])
    )  # Thruster position in BODY coordinates
    T_azimuth_angle: float = 1.0  # Time constant for azimuth angle
    T_propeller_speed: float = 1.0  # Time constant for propeller speed
    max_azimuth_angle_der: float = float(
        np.pi / 6.0
    )  # Max azimuth angle derivative (rad/s)
    max_propeller_speed_der: float = (
        20.0 / 60.0
    )  # Max propeller speed derivative (rad/s)
    rudder_area: float = 9.0

    # The limits are guesstimates based on the real ship main propeller power (500 kW) and max speed (12.6 knots)
    Fx_limits: np.ndarray = field(
        default_factory=lambda: np.array([-154.273, 154.273]) * 1000.0
    )  # Force limits in x (unscaled)
    Fy_limits: np.ndarray = field(
        default_factory=lambda: np.array([-0.6 * 154.273, 0.6 * 154.273]) * 1000.0
    )  # Force limits in y (unscaled)

    U_min: float = 0.0  # Min speed
    U_max: float = mf.knots2ms(12.6)  # Max speed
    r_max: float = 6.0 * np.pi / 180.0  # Max yaw rate, just a guess

    A_Fw: float = 12.0 * 9.6  # guesstimate of frontal area
    A_Lw: float = 12.0 * 31.25  # guesstimate of lateral area
    rho_air: float = 1.225  # Density of air
    CD_l_AF_0: float = (
        0.55  # Longitudinal resistance used to compute wind coefficients in wind model
    )
    CD_l_AF_pi: float = 0.65
    CD_t: float = (
        0.85  # Transversal resistance used to compute wind coefficients in wind model
    )
    delta_crossforce: float = 0.60  # Cross-force parameter
    s_L: float = 0.0  # x-coordinate of transverse prject area A_Lw wrt the main section


@dataclass
class CyberShip2Params:
    """Parameters for the CyberShip2 vessel (read only / fixed)."""

    name: str = "CyberShip2"
    rho: float = 1000.0  # Density of water
    draft: float = 5.0
    length: float = 1.255 * 70.0
    width: float = 0.29 * 70.0
    ship_vertices: np.ndarray = field(
        default_factory=lambda length=length, width=width: np.array(
            [
                [0.9 * length / 2.0, width / 2.0],
                [length / 2.0, 0.0],
                [0.9 * length / 2.0, -width / 2.0],
                [-length / 2.0, -width / 2.0],
                [-length / 2.0, width / 2.0],
            ]
        ).T
    )

    # The parameters below are scaled up 70 times to match the size of the real ship
    M_rb: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [23.800, 0.0, 0.0],
                [0.0, 23.800, 23.800 * 0.046],
                [0.0, 23.800 * 0.046, 1.760],
            ]
        )
    )  # Rigid body mass, m = 23.800 kg, I_z = 1.760 kgm², x_g = 0.046 matrix
    M_a: np.ndarray = field(
        default_factory=lambda: np.array(
            [[2.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]
        )
    )  # Added mass matrix
    D_l: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0.72253, 0.0, 0.0], [0.0, 0.88965, 7.250], [0.0, -0.03130, 1.9]]
        )
    )  # First order/linear damping
    # Nonlinear damping related parameters:
    X_uu: float = -1.32742
    X_uuu: float = -5.86643

    Y_vv: float = -36.47287
    Y_vr: float = -0.845
    Y_rv: float = -0.805
    Y_rr: float = -3.450

    N_vv: float = 3.95645
    N_rv: float = 0.130
    N_vr: float = 0.080
    N_rr: float = -0.750

    B: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
                [0.078, -0.078, 0.466, 0.549, 0.549],
            ]
        )
    )  # Actuator configuration matrix
    # Main propeller thruster 1 and 2 coefficients:
    T_nn_plus: float = 3.65034e-03
    T_nu_plus: float = 1.52468e-04
    T_nn_minus: float = 5.10256e-03
    T_nu_minus: float = 4.55822e-02
    # Bow thruster force coefficient
    T_n3n3: float = 1.56822e-04

    d_rud: float = 0.08  # Rudder diameter
    k_u: float = 0.5  # Induced velocity factor on fluid at rudder surface
    # Rudder lift force coefficient
    L_delta_plus: float = 6.43306
    L_ddelta_plus: float = 5.83594
    L_delta_minus: float = 3.19573
    L_ddelta_minus: float = 2.34356

    Fx_limits: np.ndarray = field(
        default_factory=lambda: np.array([-4.0, 8.0])
    )  # Force limits in x (unscaled)
    Fy_limits: np.ndarray = field(
        default_factory=lambda: np.array([-4.0, 4.0])
    )  # Force limits in y (unscaled)
    N_limits: np.ndarray = field(
        default_factory=lambda: np.array([-2.5, 2.5])
    )  # Torque limits in z (unscaled)
    max_propeller_speed: float = 33.0  # (unscaled)
    max_rudder_angle: float = 35.0 * np.pi / 180.0  # (unscaled)

    U_min: float = 0.0  # Min speed
    U_max: float = 10.0  # Max speed
    r_max: float = np.inf * np.pi / 180.0  # Max yaw rate

    scaling_factor: float = 70.0  # Scaling factor for the ship size


@dataclass
class Config:
    """Configuration class for managing model parameters."""

    csog: Optional[KinematicCSOGParams] = field(
        default_factory=lambda: KinematicCSOGParams()
    )
    viknes: Optional[ViknesParams] = None
    cybership2: Optional[CyberShip2Params] = None
    rvgunnerus: Optional[RVGunnerusParams] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()
        if "csog" in config_dict:
            config.csog = cp.convert_settings_dict_to_dataclass(
                KinematicCSOGParams, config_dict["csog"]
            )
            config.viknes = None

        if "viknes" in config_dict:
            config.viknes = ViknesParams()
            config.csog = None
            config.cybership2 = None

        if "cybership2" in config_dict:
            config.cybership2 = CyberShip2Params()
            config.csog = None
            config.viknes = None

        if "rvgunnerus" in config_dict:
            config.rvgunnerus = RVGunnerusParams()
            config.csog = None
            config.viknes = None
            config.cybership2 = None

        return config

    def to_dict(self) -> dict:
        config_dict = {}

        if self.csog is not None:
            config_dict["csog"] = self.csog.to_dict()

        if self.viknes is not None:
            config_dict["viknes"] = ""

        if self.cybership2 is not None:
            config_dict["cybership2"] = ""

        if self.rvgunnerus is not None:
            config_dict["rvgunnerus"] = ""

        return config_dict


class IModel(ABC):
    @abstractmethod
    def dynamics(
        self,
        xs: np.ndarray,
        u: np.ndarray,
        w: Optional[stochasticity.DisturbanceData] = None,
    ) -> np.ndarray:
        """The r.h.s of the ODE x_k+1 = f(x_k, u_k) for the considered model in discrete time.

        Args:
            xs (np.ndarray): The state vector x_k
            u (np.ndarray): The input vector u_k
            w (stochasticity.DisturbanceData): Optional data containing disturbance information. The model will extract relevant parts of the structure.

        NOTE: The state and input dimension may change depending on the model. Make sure to check compatibility between the controller you are using and the model.
        """

    @abstractmethod
    def bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the lower and upper bounds for the inputs and states in the model.
        The output is on the form (lbu, ubu, lbx, ubx)."""


class ModelBuilder:
    @classmethod
    def construct_model(cls, config: Optional[Config] = None) -> IModel:
        """Builds a ship model from the configuration

        Args:
            config (Optional[models.Config]): Model configuration. Defaults to None.

        Returns:
            Model: Model as specified by the configuration, e.g. a KinematicCSOG model.
        """
        if config and config.csog:
            return KinematicCSOG(config.csog)
        elif config and config.viknes:
            return Viknes()
        elif config and config.cybership2:
            return CyberShip2()
        elif config and config.rvgunnerus:
            return RVGunnerus()
        else:
            return KinematicCSOG()


class KinematicCSOG(IModel):
    """Implements a planar kinematic model using Course over ground (COG) and Speed over ground (SOG):

    x_k+1 = x_k + U_k cos(chi_k)
    y_k+1 = y_k + U_k sin(chi_k)
    chi_k+1 = chi_k + (1 / T_chi)(chi_d - chi_k)
    U_k+1 = U_k + (1 / T_U)(U_d - U_k)


    where x,y are the planar coordinates, U the vessel SOG and chi the vessel COG => xs = [x, y, chi, U, 0, 0]
    """

    _n_x: int = 4
    _n_u: int = 2

    def __init__(self, params: Optional[KinematicCSOGParams] = None) -> None:
        if params is not None:
            self._params: KinematicCSOGParams = params
        else:
            self._params = KinematicCSOGParams()

    def dynamics(
        self,
        xs: np.ndarray,
        u: np.ndarray,
        w: Optional[stochasticity.DisturbanceData] = None,
    ) -> np.ndarray:
        """Computes r.h.s of ODE x_k+1 = f(x_k, u_k), where

        Args:
            xs (np.ndarray): State x_k = [x_k, y_k, chi_k, U_k, 0.0, 0.0]
            u (np.ndarray): Input equal to [chi_d, U_d, 0]
            w (stochasticity.DisturbanceData): Optional data containing disturbance information. The model will extract relevant parts of the structure.

        Returns:
            np.ndarray: New state x_k+1.
        """
        if len(u) != 3:
            raise ValueError("Dimension of input array should be 3!")

        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        chi_d = u[0]
        U_d = mf.sat(u[1], 0.0, self._params.U_max)

        chi_diff = mf.wrap_angle_diff_to_pmpi(chi_d, xs[2])
        xs[3] = mf.sat(xs[3], 0.0, self._params.U_max)

        ode_fun = np.zeros(6)
        ode_fun[0] = xs[3] * np.cos(xs[2])
        ode_fun[1] = xs[3] * np.sin(xs[2])
        ode_fun[2] = mf.sat(
            chi_diff / self._params.T_chi, -self._params.r_max, self._params.r_max
        )
        ode_fun[3] = (U_d - xs[3]) / self._params.T_U

        return ode_fun

    def bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lbu = np.array([-np.inf, 0.0, -np.inf])
        ubu = np.array([np.inf, self._params.U_max, np.inf])
        lbx = np.array(
            [
                -np.inf,
                -np.inf,
                -np.inf,
                -self._params.U_max,
                -self._params.U_max,
                -self._params.r_max,
            ]
        )
        ubx = np.array(
            [
                np.inf,
                np.inf,
                np.inf,
                self._params.U_max,
                self._params.U_max,
                self._params.r_max,
            ]
        )
        return lbu, ubu, lbx, ubx

    @property
    def params(self):
        return self._params

    @property
    def dims(self):
        """Returns the ACTUAL state and input dimensions considered in the model.

        NOTE: Not to be mistaken with the model interface state (6) and input (3) dimension requirements."""
        return self._n_x, self._n_u


class Viknes(IModel):
    """Implements a 3DOF underactuated vessel maneuvering model for the Viknes vessel:

    eta_dot = Rpsi(eta) * nu
    nu_dot = nu_c_dot + (M_rb + M_a)^-1 (- C_rb(nu_r) * nu_r - C_a(nu_r) * nu_r - (D_l + D_nl(nu_r)) * nu_r + tau + tau_wind)

    with eta = [x, y, psi]^T, nu = [u, v, r]^T, xs = [eta, nu]^T, nu_c_dot = [r * v_c, -r * u_c, 0]^T and nu_r = nu - nu_c.

    Disturbances from winds and currents have been added, with a simple model for wind forces and moments (Blendermann model) as in Fossen (2011). A future enhancement is to include support for wave disturbances as well.

    NOTE: When using Euler`s method, keep the time step small enough (e.g. around 0.1 or less) to ensure numerical stability.

    Ref: See e.g. https://github.com/cybergalactic/FossenHandbook?tab=readme-ov-file Chapter 10, slide 55.
    """

    _n_x: int = 6
    _n_u: int = 3

    def __init__(self) -> None:
        self._params: ViknesParams = ViknesParams()

    def dynamics(
        self,
        xs: np.ndarray,
        u: np.ndarray,
        w: Optional[stochasticity.DisturbanceData] = None,
    ) -> np.ndarray:
        """Computes r.h.s of ODE x_k+1 = f(x_k, u_k), where

        Args:
            xs (np.ndarray): State xs = [eta, nu]^T
            u (np.ndarray): Input vector u = tau (generalized forces in X, Y and N)
            w (stochasticity.DisturbanceData, optional): Optional data containing disturbance information. The model will extract relevant parts of the structure.

        Returns:
            np.ndarray: New state xs.
        """
        if u.size != self._n_u:
            raise ValueError("Dimension of input array should be 3!")
        if xs.size != self._n_x:
            raise ValueError("Dimension of state should be 6!")

        u[0] = mf.sat(u[0], self._params.Fx_limits[0], self._params.Fx_limits[1])
        u[1] = mf.sat(u[1], self._params.Fy_limits[0], self._params.Fy_limits[1])
        u[2] = mf.sat(
            u[2],
            self._params.Fy_limits[0] * self._params.l_r,
            self._params.Fy_limits[1] * self._params.l_r,
        )

        eta = xs[0:3]
        eta[2] = mf.wrap_angle_to_pmpi(eta[2])
        nu = xs[3:6]

        V_c = 0.0
        beta_c = 0.0
        V_w = 0.0
        beta_w = 0.0
        tau_wind = np.zeros(3)
        if w is not None and "speed" in w.wind:
            V_w = w.wind["speed"]
            beta_w = w.wind["direction"]
            # Compute wind forces and moments
            nu_w = mf.Rmtrx(eta[2]).T @ np.array(
                [V_w * np.cos(beta_w), V_w * np.sin(beta_w), 0.0]
            )
            u_rw = nu[0] - nu_w[0]
            v_rw = nu[1] - nu_w[1]
            V_rw = np.sqrt(u_rw**2 + v_rw**2)
            gamma_rw = -np.arctan2(v_rw, u_rw)
            gamma_w = eta[2] - beta_w - np.pi  # wind angle of attack
            gamma_w = mf.wrap_angle_to_pmpi(gamma_w)
            tau_wind = self.compute_wind_forces(V_rw, gamma_rw)
        if w is not None and "speed" in w.currents:
            V_c = w.currents["speed"]
            beta_c = w.currents["direction"]

        nu_c = mf.Rmtrx(eta[2]).T @ np.array(
            [V_c * np.cos(beta_c), V_c * np.sin(beta_c), 0.0]
        )
        nu_c_dot = np.array(
            [nu[2] * nu_c[1], -nu[2] * nu_c[0], 0.0]
        )  # under the assumption of irrotational current
        nu_r = nu - nu_c

        Minv = np.linalg.inv(self._params.M_rb + self._params.M_a)
        C_RB = mf.coriolis_matrix_rigid_body(self._params.M_rb, nu_r)
        C_A = mf.coriolis_matrix_added_mass(self._params.M_a, nu_r)
        Cvv = C_RB @ nu_r + C_A @ nu_r

        Dvv = (
            mf.Dmtrx(self._params.D_l, self._params.D_q, self._params.D_c, nu_r) @ nu_r
        )

        tau = u

        ode_fun = np.zeros(6)
        ode_fun[0:3] = mf.Rmtrx(eta[2]) @ nu
        ode_fun[3:6] = Minv @ (-Cvv - Dvv + tau + tau_wind) + nu_c_dot

        return ode_fun

    def _compute_wind_coefficients(self, gamma_rw: float) -> Tuple[float, float, float]:
        """Computes the wind coefficients based on 8.32 - 8.35 in Fossen 2011. See also _compute_wind_forces

        Args:
            gamma_rw (float): Relative wind angle of attack

        Returns:
            Tuple[float, float, float]: Wind coefficients for X, Y and N forces and moments
        """
        A_ratio = self._params.A_Lw / self._params.A_Fw
        if abs(gamma_rw) <= np.pi / 2.0:
            CD_l = self._params.CD_l_AF_0 / A_ratio
        elif abs(gamma_rw) > np.pi / 2.0:
            CD_l = self._params.CD_l_AF_pi / A_ratio
        CD_t = self._params.CD_t
        CD_ratio = CD_l / CD_t
        C_X = (
            -CD_l
            * A_ratio
            * (
                np.cos(gamma_rw)
                / (
                    1.0
                    - (
                        0.5
                        * self._params.delta_crossforce
                        * (1.0 - CD_ratio)
                        * (np.sin(2.0 * gamma_rw)) ** 2
                    )
                )
            )
        )
        C_Y = (
            CD_t
            * np.sin(gamma_rw)
            / (
                1.0
                - (
                    0.5
                    * self._params.delta_crossforce
                    * (1.0 - CD_ratio)
                    * (np.sin(2.0 * gamma_rw)) ** 2
                )
            )
        )
        C_N = (
            self._params.s_L / self._params.length - 0.18 * (gamma_rw - 0.5 * np.pi)
        ) * C_Y
        return C_X, C_Y, C_N

    def compute_wind_forces(self, V_rw: float, gamma_rw: float) -> np.ndarray:
        """Computes the wind forces and moments based on 8.23 in Fossen 2011.

        Args:
            V_rw (float): Relative wind speed
            gamma_rw (float): Relative wind angle of attack

        Returns:
            np.ndarray: Wind forces and moments
        """
        C_X, C_Y, C_N = self._compute_wind_coefficients(gamma_rw)
        Fx = 0.5 * self._params.rho_air * V_rw**2 * C_X * self._params.A_Fw
        Fy = 0.5 * self._params.rho_air * V_rw**2 * C_Y * self._params.A_Lw
        N = (
            0.5
            * self._params.rho_air
            * V_rw**2
            * C_N
            * self._params.A_Lw
            * self._params.length
        )
        return np.array([Fx, Fy, N])

    def bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lbu = np.array(
            [
                self._params.Fx_limits[0],
                self._params.Fy_limits[0],
                self._params.Fy_limits[0] * self._params.l_r,
            ]
        )
        ubu = np.array(
            [
                self._params.Fx_limits[1],
                self._params.Fy_limits[1],
                self._params.Fy_limits[1] * self._params.l_r,
            ]
        )
        lbx = np.array(
            [
                -np.inf,
                -np.inf,
                -np.inf,
                -self._params.U_max,
                -self._params.U_max,
                -self._params.r_max,
            ]
        )
        ubx = np.array(
            [
                np.inf,
                np.inf,
                np.inf,
                self._params.U_max,
                self._params.U_max,
                self._params.r_max,
            ]
        )
        return lbu, ubu, lbx, ubx

    @property
    def params(self):
        "Returns the parameters of the considered model."
        return self._params

    @property
    def dims(self):
        """Returns the ACTUAL state and input dimensions considered in the model.

        NOTE: Not to be mistaken with the model interface state (6) and input (3) dimension requirements."""
        return self._n_x, self._n_u


class RVGunnerus(IModel):
    """Implements a 3DOF underactuated vessel maneuvering model for the R/V Gunnerus vessel with linear+quadratic viscous loads.

    An actuator model for a single azimuth thruster (by combining the two existing azimuth pods into one at the centerline for simplicity) is included,
    but not used in the current framework. This removes the need for thrust allocation

    The model is implemented originally by Mathias Marley in the MCSim_python repository, managed by the Marine Cybernetics laboratory https://www.ntnu.edu/imt/lab/cybernetics.

    Disturbances from winds and currents have been added, with a simple model for wind forces and moments (Blendermann model) as in Fossen (2011). A future enhancement is to include support for wave disturbances as well.

    NOTE: When using Eulers method, keep the time step small enough (e.g. around 0.1 or less) to ensure numerical stability.

    Ref: See e.g. https://github.com/cybergalactic/FossenHandbook?tab=readme-ov-file Chapter 10, slide 55.
    """

    _n_x: int = 6
    _n_u: int = 3

    def __init__(self) -> None:
        self._params: RVGunnerusParams = RVGunnerusParams()

    def dynamics(
        self,
        xs: np.ndarray,
        u: np.ndarray,
        w: Optional[stochasticity.DisturbanceData] = None,
    ) -> np.ndarray:
        """Computes r.h.s of ODE x_k+1 = f(x_k, u_k), where

        Args:
            xs (np.ndarray): State xs = [eta, nu]^T
            u (np.ndarray): Input vector u = tau (generalized forces in X, Y and N), essentially only X and Y as N = l_r * Y here for the case with 1 azimuth thruster.
            w (stochasticity.DisturbanceData, optional): Optional data containing disturbance information. The model will extract relevant parts of the structure.

        Returns:
            np.ndarray: New state xs.
        """
        if u.size != self._n_u:
            raise ValueError("Dimension of input array should be 3!")
        if xs.size != self._n_x:
            raise ValueError("Dimension of state should be 6!")

        eta = xs[0:3]
        eta[2] = mf.wrap_angle_to_pmpi(eta[2])
        nu = xs[3:6]

        # Guesstimate limits
        u[0] = mf.sat(u[0], self._params.Fx_limits[0], self._params.Fx_limits[1])
        u[1] = mf.sat(u[1], self._params.Fy_limits[0], self._params.Fy_limits[1])
        u[2] = mf.sat(
            u[2],
            abs(self._params.r_t[0]) * self._params.Fy_limits[0],
            abs(self._params.r_t[0]) * self._params.Fy_limits[1],
        )

        V_c = 0.0
        beta_c = 0.0
        V_w = 0.0
        beta_w = 0.0
        tau_wind = np.zeros(3)
        if w is not None and "speed" in w.wind and "direction" in w.wind:
            V_w = w.wind["speed"]
            beta_w = w.wind["direction"]
            # Compute wind forces and moments
            nu_w = mf.Rmtrx(eta[2]).T @ np.array(
                [V_w * np.cos(beta_w), V_w * np.sin(beta_w), 0.0]
            )
            u_rw = nu[0] - nu_w[0]
            v_rw = nu[1] - nu_w[1]
            V_rw = np.sqrt(u_rw**2 + v_rw**2)
            gamma_rw = -np.arctan2(v_rw, u_rw)
            gamma_w = eta[2] - beta_w - np.pi  # wind angle of attack
            gamma_w = mf.wrap_angle_to_pmpi(gamma_w)
            tau_wind = self.compute_wind_forces(V_rw, gamma_rw)
        if w is not None and "speed" in w.currents and "direction" in w.currents:
            V_c = w.currents["speed"]
            beta_c = w.currents["direction"]

        nu_c = mf.Rmtrx(eta[2]).T @ np.array(
            [V_c * np.cos(beta_c), V_c * np.sin(beta_c), 0.0]
        )
        nu_c_dot = np.array(
            [nu[2] * nu_c[1], -nu[2] * nu_c[0], 0.0]
        )  # under the assumption of irrotational current
        nu_r = nu - nu_c

        Minv = np.linalg.inv(self._params.M_rb + self._params.M_a)
        C_RB = mf.coriolis_matrix_rigid_body(self._params.M_rb, nu_r)
        C_A = mf.coriolis_matrix_added_mass(self._params.M_a, nu_r)
        Cvv = C_RB @ nu_r + C_A @ nu_r
        Dvv = (
            self._params.D_l
            + self._params.D_u * abs(nu_r[0])
            + self._params.D_v * abs(nu_r[1])
            + self._params.D_r * abs(nu_r[2])
        ) @ nu_r

        # Compute thruster forces and moments (NOT USED IN CURRENT FRAMEWORK for simplicity),
        # would then need azimuth angle and propeller speed as input
        # u_t = nu[0] - nu_c[0] # surge velocity at thruster location
        # v_t = nu[1] - nu_c[1] + nu[2] * self._params.r_t[0] # sway velocity at thruster location
        # Fx_thruster, Fy_thruster = self._compute_thruster_forces(u_t, v_t, u[0], u[1])
        tau = u

        ode_fun = np.zeros(6)
        ode_fun[0:3] = mf.Rmtrx(eta[2]) @ nu
        ode_fun[3:6] = Minv @ (-Cvv - Dvv + tau + tau_wind) + nu_c_dot

        return ode_fun

    def _compute_wind_coefficients(self, gamma_rw: float) -> Tuple[float, float, float]:
        """Computes the wind coefficients based on 8.32 - 8.35 in Fossen 2011. See also _compute_wind_forces

        Args:
            gamma_rw (float): Relative wind angle of attack

        Returns:
            Tuple[float, float, float]: Wind coefficients for X, Y and N forces and moments
        """
        A_ratio = self._params.A_Lw / self._params.A_Fw
        if abs(gamma_rw) <= np.pi / 2.0:
            CD_l = self._params.CD_l_AF_0 / A_ratio
        elif abs(gamma_rw) > np.pi / 2.0:
            CD_l = self._params.CD_l_AF_pi / A_ratio
        CD_t = self._params.CD_t
        CD_ratio = CD_l / CD_t
        C_X = (
            -CD_l
            * A_ratio
            * (
                np.cos(gamma_rw)
                / (
                    1.0
                    - 0.5
                    * self._params.delta_crossforce
                    * (1.0 - CD_ratio)
                    * np.sin(2.0 * gamma_rw) ** 2
                )
            )
        )
        C_Y = (
            CD_t
            * np.sin(gamma_rw)
            / (
                1.0
                - 0.5
                * self._params.delta_crossforce
                * (1.0 - CD_ratio)
                * np.sin(2.0 * gamma_rw) ** 2
            )
        )
        C_N = (
            self._params.s_L / self._params.length - 0.18 * (gamma_rw - 0.5 * np.pi)
        ) * C_Y
        return C_X, C_Y, C_N

    def compute_wind_forces(self, V_rw: float, gamma_rw: float) -> np.ndarray:
        """Computes the wind forces and moments based on 8.23 in Fossen 2011.

        Args:
            V_rw (float): Relative wind speed
            gamma_rw (float): Relative wind angle of attack

        Returns:
            np.ndarray: Wind forces and moments
        """
        C_X, C_Y, C_N = self._compute_wind_coefficients(gamma_rw)
        Fx = 0.5 * self._params.rho_air * V_rw**2 * C_X * self._params.A_Fw
        Fy = 0.5 * self._params.rho_air * V_rw**2 * C_Y * self._params.A_Lw
        N = (
            0.5
            * self._params.rho_air
            * V_rw**2
            * C_N
            * self._params.A_Lw
            * self._params.length
        )
        return np.array([Fx, Fy, N])

    def _compute_thruster_forces(
        self, u_t: float, v_t: float, azimuth: float, propeller_speed: float
    ) -> Tuple[float, float]:
        """Computes the forces and moments from the thrusters

        Args:
            - u_t (float): surge velocity at thruster location
            - v_t (float): sway velocity at thruster location
            - azimuth (float): azimuth angle of the thruster
            - propeller_speed (float): propeller speed

        Returns:
            Tuple[float, float]: the forces in the x and y direction
        """
        U_t = np.sqrt(u_t**2 + v_t**2)  # total speed
        inflow_angle = np.arctan2(v_t, u_t)  # inflow angle

        aoa = -inflow_angle + azimuth  # foil angle of attack
        aoa_max = 35.0 * np.pi / 180

        if np.abs(aoa) > aoa_max:
            print(
                "Warning (thrusters.RVGazimuth_man): Angle of attack="
                + str(round(aoa * 180 / np.pi))
                + "deg, assumed validity range is +/- 30deg"
            )

        C_d = 0.3 + np.abs(aoa) * 0.3  # drag coefficient
        C_l = 0.5 * np.sin(2 * aoa)  # lift coefficient

        F_drag = (
            0.5 * self._params.rho * self._params.rudder_area * C_d * U_t**2
        )  # drag force on foil (parallel to fluid velocity)
        F_lift = (
            0.5 * self._params.rho * self._params.rudder_area * C_l * U_t**2
        )  # lift force on foil (normal to fluid velocity)
        F_foil_x = -F_drag * np.cos(aoa) + F_lift * np.sin(
            aoa
        )  # force in foil x direction
        F_foil_y = F_lift * np.cos(aoa) + F_drag * np.sin(
            aoa
        )  # force in foil y direction

        C_t = 2.2 * 1  # *2.8 # thruster coefficient (just a guess)
        F_thrust = C_t * (propeller_speed * 60.0) ** 2  # propeller thrust force

        # decompose loads in body-fixed surge and sway force
        F_thrust_x = (
            F_thrust * np.cos(azimuth)
            + F_foil_x * np.cos(azimuth)
            - F_foil_y * np.sin(azimuth)
        )
        F_thrust_y = (
            F_thrust * np.sin(azimuth)
            + F_foil_x * np.sin(azimuth)
            + F_foil_y * np.cos(azimuth)
        )
        return F_thrust_x, F_thrust_y

    def bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lbu = np.array(
            [
                self._params.Fx_limits[0],
                self._params.Fy_limits[0],
                self._params.Fy_limits[0] * abs(self._params.r_t[0]),
            ]
        )
        ubu = np.array(
            [
                self._params.Fx_limits[1],
                self._params.Fy_limits[1],
                self._params.Fy_limits[1] * abs(self._params.r_t[0]),
            ]
        )
        lbx = np.array(
            [
                -np.inf,
                -np.inf,
                -np.inf,
                -self._params.U_max,
                -self._params.U_max,
                -np.inf,
            ]
        )
        ubx = np.array(
            [np.inf, np.inf, np.inf, self._params.U_max, self._params.U_max, np.inf]
        )
        return lbu, ubu, lbx, ubx

    @property
    def params(self):
        "Returns the parameters of the considered model."
        return self._params

    @property
    def dims(self):
        """Returns the ACTUAL state and input dimensions considered in the model.

        NOTE: Not to be mistaken with the model interface state (6) and input (3) dimension requirements."""
        return self._n_x, self._n_u


class CyberShip2(IModel):
    """Implements a modified version of the 3DOF nonlinear maneuvering model for the Cybership 2 vessel with consideration of currents. The model is on the form

    eta_dot = Rpsi(eta) * nu
    M_rb * nu_dot + C_rb(nu) * nu + M_A * nu_v_r_dot + C_A(nu_r) * nu_r + D(nu_r) * nu_r = tau + tau_wind + tau_waves

    with tau = B f_c(u, nu_r) if the actuator model is used. Here, the actuator model is not used per now.

    The input vector is thus set to u = tau
    #(= [n_1, n_2, n_3, delta_1, delta_2]^T if actuator model is used).
    #
    # State vector is xs = [eta, nu]^T, where eta = [x, y, psi]^T and nu = [u, v, r]^T.

    See "A Nonlinear Ship Manoeuvering Model: Identification and adaptive control with experiments for a model ship" https://www.mic-journal.no/ABS/MIC-2004-1-1/ for more details.

    NOTE: When using Euler`s method, keep the time step small enough (e.g. around 0.1 or less) to ensure numerical stability.
    """

    _n_x: int = 6
    _n_u: int = 3

    def __init__(self) -> None:
        self._params: CyberShip2Params = CyberShip2Params()

    def bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the bounds on the forces the actuators can produce on the ship."""
        lamb = self._params.scaling_factor
        lbu = np.array(
            [
                self._params.Fx_limits[0] * lamb * lamb * lamb,
                self._params.Fy_limits[0] * lamb * lamb * lamb,
                self._params.N_limits[0] * lamb * lamb * lamb * lamb * lamb,
            ]
        )
        ubu = np.array(
            [
                self._params.Fx_limits[1],
                self._params.Fy_limits[1],
                self._params.N_limits[1],
            ]
        )
        lbx = np.array(
            [
                -np.inf,
                -np.inf,
                -np.inf,
                -self._params.U_max,
                -self._params.U_max,
                -np.inf,
            ]
        )
        ubx = np.array(
            [np.inf, np.inf, np.inf, self._params.U_max, self._params.U_max, np.inf]
        )
        return lbu, ubu, lbx, ubx

    @property
    def params(self):
        "Returns the parameters of the considered model."
        return self._params

    @property
    def dims(self):
        """Returns the ACTUAL state and input dimensions considered in the model.

        NOTE: Not to be mistaken with the model interface state (6) and input (3) dimension requirements."""
        return self._n_x, self._n_u

    def nonlinear_damping_matrix(self, nu_r: np.ndarray) -> np.ndarray:
        """Computes the nonlinear damping matrix D_nl(nu_r)

        Args:
            nu (np.ndarray): Velocity vector nu = [u, v, r]^T or relative velocity vector nu_r = [u_r, v_r, r]^T

        Returns:
            np.ndarray: Nonlinear damping matrix D_nl(nu)
        """
        d_11 = self._params.X_uu * abs(nu_r[0]) + self._params.X_uuu * nu_r[0] ** 2
        d_22 = self._params.Y_vv * abs(nu_r[1]) + self._params.Y_rv * abs(nu_r[2])
        d_23 = self._params.Y_vr * abs(nu_r[1]) + self._params.Y_rr * abs(nu_r[2])
        d_32 = self._params.N_vv * abs(nu_r[1]) + self._params.N_rv * abs(nu_r[2])
        d_33 = self._params.N_vr * abs(nu_r[1]) + self._params.N_rr * abs(nu_r[2])
        return np.array([[-d_11, 0.0, 0.0], [0.0, -d_22, -d_23], [0.0, -d_32, -d_33]])

    def dynamics(
        self,
        xs: np.ndarray,
        u: np.ndarray,
        w: Optional[stochasticity.DisturbanceData] = None,
    ) -> np.ndarray:
        """Computes r.h.s of ODE x_k+1 = f(x_k, u_k), where

        Args:
            xs (np.ndarray): State xs = [eta, nu]^T (for the real ship, must be scaled to be used with the CS2-model)
            u (np.ndarray):  Input vector of generalized forces [X, Y, N]^T. Input equal to [n_1, n_2, n_3, delta_1, delta_2]^T consisting of the two main propeller speeds, bow propeller speed and main propeller rudder angles. Values already scaled to be used with the CS2-model.
            w (stochasticity.DisturbanceData, optional): Optional data containing disturbance information. The model will extract relevant parts of the structure.

        Returns:
            np.ndarray: New state xs.
        """
        # if u.size != self._n_u:
        #     raise ValueError("Dimension of input array should be 5!")
        if xs.size != self._n_x:
            raise ValueError("Dimension of state should be 6!")

        eta = xs[0:3]
        eta[2] = mf.wrap_angle_to_pmpi(eta[2])

        nu = xs[3:6]

        V_c = 0.0
        beta_c = 0.0
        if w is not None and "speed" in w.currents:
            V_c = w.currents["speed"]
            beta_c = w.currents["direction"]

        # Current in BODY frame
        nu_c = mf.Rmtrx(eta[2]).T @ np.array(
            [V_c * np.cos(beta_c), V_c * np.sin(beta_c), 0.0]
        )

        # Scale down velocities, relevant states and references to model size
        nu_scaled = nu.copy()
        nu_scaled[0] = nu[0] / np.sqrt(self._params.scaling_factor)
        nu_scaled[1] = nu[1] / np.sqrt(self._params.scaling_factor)
        nu_scaled[2] = nu[2] * np.sqrt(self._params.scaling_factor)

        nu_c_scaled = nu_c.copy()
        nu_c_scaled[0] = nu_c[0] / np.sqrt(self._params.scaling_factor)
        nu_c_scaled[1] = nu_c[1] / np.sqrt(self._params.scaling_factor)
        nu_c_scaled[2] = nu_c[2] * np.sqrt(self._params.scaling_factor)

        nu_r_scaled = nu_scaled - nu_c_scaled

        tau = u  # self.input_to_generalized_forces(u, nu_r_scaled)

        Minv = np.linalg.inv(self._params.M_rb + self._params.M_a)

        C_RB = mf.coriolis_matrix_rigid_body(self._params.M_rb, nu_scaled)
        C_A = mf.coriolis_matrix_added_mass(self._params.M_a, nu_r_scaled)
        Cvv = C_RB @ nu_scaled + C_A @ nu_r_scaled

        D_nl = self.nonlinear_damping_matrix(nu_r_scaled)
        Dvv = self._params.D_l @ nu_r_scaled + D_nl @ nu_r_scaled

        ode_fun = np.zeros(6)
        ode_fun[0:3] = mf.Rmtrx(eta[2]) @ nu
        ode_fun[3:6] = Minv @ (-Cvv - Dvv + tau)

        return ode_fun

    # The below functions are used if the actuator model is employed (need proper thrust allocation to control the ship in that case)
    def input_to_generalized_forces(
        self, u: np.ndarray, nu_r: np.ndarray
    ) -> np.ndarray:
        """Computes generalized forces tau from the input u and relative velocity nu_r.

        Args:
            u (np.ndarray): Input equal to [n_1, n_2, n_3, delta_1, delta_2]^T consisting of the two main propeller speeds, bow propeller speed and main propeller rudder angles.
            nu_r (np.ndarray): Relative velocity vector nu_r = [u_r, v_r, r]^T

        Returns:
            np.ndarray: Generalized forces tau = [X, Y, N]^T
        """
        T_1 = self.main_propeller_speed_to_thrust_force(u[0], nu_r[0])
        T_2 = self.main_propeller_speed_to_thrust_force(u[1], nu_r[0])
        T_3 = self.bow_propeller_speed_to_thrust_force(u[2])
        L_1 = self.main_propeller_rudder_angle_to_lift_force(u[3], T_1, nu_r[0])
        L_2 = self.main_propeller_rudder_angle_to_lift_force(u[4], T_2, nu_r[0])
        tau = self._params.B @ np.array([T_1, T_2, T_3, L_1, L_2])
        return tau

    def main_propeller_speed_to_thrust_force(self, n: float, u_r: float) -> float:
        """Computes the thrust force T from the main propeller speed n and relative surge speed using the actuator model.

        Args:
            n (float): Main propeller speed n
            u_r (float): Relative surge speed u_r

        Returns:
            float: Thrust force T
        """
        n_top = max(0.0, u_r * self._params.T_nu_plus / self._params.T_nn_plus)
        n_bot = min(0.0, u_r * self._params.T_nu_minus / self._params.T_nn_minus)
        T = 0.0
        if n >= n_top:
            T = (
                self._params.T_nn_plus * abs(n) * n
                - self._params.T_nu_plus * abs(n) * u_r
            )
        elif n <= n_bot:
            T = (
                self._params.T_nn_minus * abs(n) * n
                - self._params.T_nu_minus * abs(n) * u_r
            )
        return T

    def bow_propeller_speed_to_thrust_force(self, n: float) -> float:
        """Computes the thrust force T from the bow propeller speed n using the actuator model.

        Args:
            n (float): Propeller speed n

        Returns:
            float: Thrust force T
        """
        return self._params.T_n3n3 * abs(n) * n

    def main_propeller_rudder_angle_to_lift_force(
        self, delta: float, T: float, u_r: float
    ) -> float:
        """Computes the lift force L from the main propeller rudder angle delta, thruster force and relative surge speed using the actuator model.

        Args:
            delta (float): Main propeller rudder angle delta
            T (float): Thruster force T
            u_r (float): Relative surge speed u_r

        Returns:
            float: Lift force L
        """
        u_rud = u_r
        if u_r >= 0.0:
            u_rud = u_r + self._params.k_u * (
                np.sqrt(
                    max(
                        0.0,
                        8.0 * T / (np.pi * self._params.rho * self._params.d_rud**2)
                        + u_r**2,
                    )
                )
                - u_r
            )

        if u_rud >= 0.0:
            L = (
                (
                    self._params.L_delta_plus * delta
                    - self._params.L_ddelta_plus * abs(delta) * delta
                )
                * abs(u_rud)
                * u_rud
            )
        else:
            L = (
                (
                    self._params.L_delta_minus * delta
                    - self._params.L_ddelta_minus * abs(delta) * delta
                )
                * abs(u_rud)
                * u_rud
            )
        return L
