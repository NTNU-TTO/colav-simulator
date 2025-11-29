"""
controllers.py

Summary:
    Contains class definitions for various control strategies.
    Every controller must adhere to the interface IController.

Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf


@dataclass
class MIMOPIDParams:
    "Parameters for a Proportional-Integral-Derivative controller."

    wn: np.ndarray = field(default_factory=lambda: np.diag([0.3, 0.2, 0.35]))
    zeta: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 1.0]))
    eta_diff_max: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_dict(self):
        output_dict = {}
        output_dict["wn"] = self.wn.diagonal().tolist()
        output_dict["zeta"] = self.zeta.diagonal().tolist()
        output_dict["eta_diff_max"] = self.eta_diff_max.tolist()


@dataclass
class SHPIDParams:
    "Parameters for a PID controller for surge, sway + heading control with feedback linearization of the mass+coriolis+damping."

    K_p: np.ndarray = field(default_factory=lambda: np.diag([5.0, 1.3, 1.4]))
    K_d: np.ndarray = field(default_factory=lambda: np.diag([0.0, 5.0, 15.0]))
    K_i: np.ndarray = field(default_factory=lambda: np.diag([0.25, 0.1, 0.1]))
    z_diff_max: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 2.0, 15.0 * np.pi / 180.0])
    )
    V: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
    )

    def __init__(
        self,
        K_p: np.ndarray,
        K_d: np.ndarray,
        K_i: np.ndarray,
        z_diff_max: np.ndarray,
    ) -> None:
        self.K_p = K_p
        self.K_d = K_d
        self.K_i = K_i
        self.z_diff_max = z_diff_max

    @classmethod
    def from_dict(cls, config_dict: dict):
        params = SHPIDParams(
            K_p=np.diag(config_dict["K_p"]),
            K_d=np.diag(config_dict["K_d"]),
            K_i=np.diag(config_dict["K_i"]),
            z_diff_max=np.array(config_dict["z_diff_max"]),
        )
        params.z_diff_max[2] = np.deg2rad(params.z_diff_max[2])
        return params

    def to_dict(self):
        output_dict = {}
        output_dict["K_p"] = self.K_p.diagonal().tolist()
        output_dict["K_d"] = self.K_d.diagonal().tolist()
        output_dict["K_i"] = self.K_i.diagonal().tolist()
        self.z_diff_max[2] = np.rad2deg(self.z_diff_max[2])
        output_dict["z_diff_max"] = self.z_diff_max.tolist()


@dataclass
class FLSCParams:
    "Parameters for the feedback linearizing surge-course controller."

    K_p_u: float = 3.0
    K_i_u: float = 0.3
    K_p_chi: float = 2.2
    K_d_chi: float = 4.0
    K_i_chi: float = 0.1
    max_speed_error_int: float = 4.0
    speed_error_int_threshold: float = 0.5
    max_chi_error_int: float = 90.0 * np.pi / 180.0
    chi_error_int_threshold: float = 20.0 * np.pi / 180.0

    def to_dict(self):
        output = asdict(self)
        output["max_chi_error_int"] = float(np.rad2deg(output["max_chi_error_int"]))
        output["chi_error_int_threshold"] = float(
            np.rad2deg(output["chi_error_int_threshold"])
        )
        return output


@dataclass
class PassThroughInputsParams:
    "Parameters for the pass-through inputs controller."

    rudder_propeller_mapping: bool = True

    def to_dict(self):
        return asdict(self)

    def from_dict(self, config_dict: dict):
        return PassThroughInputsParams(**config_dict)


@dataclass
class Config:
    """Configuration class for managing controller parameters."""

    pid: Optional[MIMOPIDParams] = None
    flsc: Optional[FLSCParams] = None
    pass_through_cs: Optional[bool] = True
    pass_through_inputs: Optional[PassThroughInputsParams] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(
            pid=None, flsc=None, pass_through_cs=None, pass_through_inputs=None
        )
        if "pid" in config_dict:
            config.pid = MIMOPIDParams(
                wn=np.diag(config_dict["pid"]["wn"]),
                zeta=np.diag(config_dict["pid"]["zeta"]),
                eta_diff_max=np.array(config_dict["pid"]["eta_diff_max"]),
            )

        if "flsc" in config_dict:
            config.flsc = cp.convert_settings_dict_to_dataclass(
                FLSCParams, config_dict["flsc"]
            )
            config.flsc.max_chi_error_int = np.deg2rad(config.flsc.max_chi_error_int)
            config.flsc.chi_error_int_threshold = np.deg2rad(
                config.flsc.chi_error_int_threshold
            )

        if "pass_through_cs" in config_dict:
            config.pass_through_cs = True

        if "pass_through_inputs" in config_dict:
            config.pass_through_inputs = PassThroughInputsParams(
                **config_dict["pass_through_inputs"]
            )

        return config

    def to_dict(self) -> dict:
        config_dict = {}
        if self.pid is not None:
            config_dict["pid"] = self.pid.to_dict()

        if self.flsc is not None:
            config_dict["flsc"] = self.flsc.to_dict()

        if self.pass_through_cs is not None:
            config_dict["pass_through_cs"] = ""

        if self.pass_through_inputs is not None:
            config_dict["pass_through_inputs"] = self.pass_through_inputs.to_dict()

        return config_dict


class IController(ABC):
    @abstractmethod
    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
        """Computes inputs using the specific controller strategy.

        References should be of dimension 9 x 1 consisting of pose, velocity and acceleration refs. E.g.
        for LOS-guidance with a PID controller (FLSH) for surge and course control, the refs are typically
        [0, 0, chi_ref, U_ref, 0, 0, 0, 0, 0]^T.

        Args:
            refs (np.ndarray): Desired/references = [x, y, psi, u, v, r, ax, ay, rdot]
            xs (np.ndarray): State typically on the form [x, y, psi, u, v, r]^T, or [x, y, chi, U, 0, 0]^T for a CSOG kinematic model.
            dt (float): Time step

        Returns:
            np.ndarray: 3 x 1 inputs u.

        """

    @abstractmethod
    def reset(self) -> None:
        """Resets the controller to its initial state."""


class ControllerBuilder:
    @classmethod
    def construct_controller(
        cls, model_params: Any, config: Optional[Config] = None
    ) -> IController:
        """Builds a controller from the configuration

        Args:
            model_params (Any): Model parameters used by the controller.
            config (Optional[controllers.Config]): Model configuration. Defaults to None.

        Returns:
            Model: Model as specified by the configuration, e.g. a MIMOPID controller.
        """
        if config and config.pid:
            return MIMOPID(model_params, config.pid)
        elif config and config.flsc:
            return FLSC(model_params, config.flsc)
        elif config and config.pass_through_cs:
            return PassThroughCS()
        elif config and config.pass_through_inputs:
            return PassThroughInputs(model_params, config.pass_through_inputs)
        else:
            return PassThroughCS()


@dataclass
class PassThroughCS(IController):
    """This controller just feeds through the course (heading) and forward speed references."""

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
        """Takes out relevant parts of references as inputs directly

        Args:
            refs (np.ndarray): Desired/references = [x, y, psi, u, v, r, ax, ay, rdot]
            xs (np.ndarray): State typically on the form [x, y, psi, u, v, r]^T, or [x, y, chi, U, 0, 0]^T for a CSOG kinematic model.
            dt (float): Time step

        Returns:
            np.ndarray: 3 x 1 inputs u = [chi_d, U_d, 0.0]^T, where chi_d and U_d are the desired course and speed over ground
        """

        if len(refs) != 9:
            raise ValueError("Dimension of reference array should be equal to 9!")
        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        # Course taken directly as heading reference.
        return np.array([refs[2], np.sqrt(refs[3] ** 2 + refs[4] ** 2), 0.0])

    def reset(self) -> None:
        return


@dataclass
class PassThroughInputs(IController):
    """This controller feeds through the force inputs u = [X, Y, N] or u = [X, Y], depending on the model and mapping chosen.

    If the model is a rudder-propeller model, the references are [X, Y], and the mapping is [X, Y] -> [X, Y, -Y * l_r],
    where l_r is the distance from the center of rotation to the rudder.

    Otherwise, the mapping is [X, Y, N] -> [X, Y, N]."""

    def __init__(
        self, model_params, params: Optional[PassThroughInputsParams] = None
    ) -> None:
        self._model_params = model_params
        if params is None:
            self.params = PassThroughInputsParams()
        else:
            self.params = params

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
        """Takes out relevant parts of references as inputs directly

        Args:
            refs (np.ndarray): Force inputs = [X, Y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]^T or [X, Y, -Y * l_r, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]^T depending on the model.
            xs (np.ndarray): State typically on the form [x, y, psi, u, v, r]^T, or [x, y, chi, U, 0, 0]^T for a CSOG kinematic model.
            dt (float): Time step

        Returns:
            np.ndarray: 3 x 1 inputs u
        """

        if len(refs) != 9:
            raise ValueError("Dimension of reference array should be equal to 9!")
        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        if self.params.rudder_propeller_mapping:
            return np.array([refs[0], refs[1], -refs[1] * self._model_params.l_r])
        else:
            return np.array([refs[0], refs[1], refs[2]])

    def reset(self) -> None:
        return


@dataclass
class MIMOPID(IController):
    """Implements a multiple input multiple output (MIMO) Proportional-Integral-Derivative (PID) controller

    tau = J_Theta(eta)^T *
          (-K_p (eta_d - eta)
           -K_d (eta_dot - eta_dot_d)
           -K_i integral_{0}^{t} (eta(l) - eta_d(l)) dl)

    for a system

    eta_dot = J_Theta(eta) * nu
    M * nu_dot + C(nu) * nu + D(nu) * nu = tau

    where J_Theta(eta) = R(eta) for the 3DOF case with eta = [x, y, psi]^T, nu = [u, v, r]^T and xs = [eta, nu]^T.

    See Fossen 2011, Ch. 12.
    """

    def __init__(self, model_params, params: Optional[MIMOPIDParams] = None) -> None:
        self._model_params = model_params
        if params is not None:
            self._params: MIMOPIDParams = params
        else:
            self._params = MIMOPIDParams()

        self._eta_diff_int: np.ndarray = np.zeros(3)

    def _reset_integrator(self):
        self._eta_diff_int = np.zeros(3)

    def reset(self):
        self._reset_integrator()

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
        """Computes inputs based on the PID law.

        Args:
            refs (np.ndarray): Desired/reference state xs_d = [eta_d, eta_dot_d, eta_ddot_d]^T
            xs (np.ndarray): State xs = [eta, nu]^T
            dt (float): Time step

        Returns:
            np.ndarray: Inputs u = tau to apply to the system.
        """
        if len(refs) != 9:
            raise ValueError("Dimension of reference array should be equal to 9!")
        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        eta = xs[0:3]
        nu = xs[3:]

        R_n_b = mf.Rmtrx(eta[2])
        eta_dot = R_n_b @ nu

        Mmtrx = self._model_params.M_rb + self._model_params.M_a
        Dmtrx = mf.Dmtrx(
            self._model_params.D_l, self._model_params.D_q, self._model_params.D_c, nu
        )

        K_p, K_d, K_i = pole_placement(Mmtrx, Dmtrx, self._params.wn, self._params.zeta)

        eta_d = refs[0:3]
        eta_diff = eta - eta_d
        eta_diff[2] = mf.wrap_angle_diff_to_pmpi(eta[2], eta_d[2])

        eta_dot_d = refs[3:6]
        eta_dot_diff = eta_dot - eta_dot_d

        self._eta_diff_int = mf.sat(
            self._eta_diff_int + eta_diff * dt, np.zeros(3), self._params.eta_diff_max
        )

        tau = -K_p @ eta_diff - K_d @ eta_dot_diff - K_i @ self._eta_diff_int
        tau = R_n_b.T @ tau

        return tau


def pole_placement(
    Mmtrx: np.ndarray, Dmtrx: np.ndarray, wn: np.ndarray, zeta
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dynamic positioning controller pole placement based on Ex. 12.7 in Fossen 2011.

    Args:
        Mmtrx (np.ndarray): Mass matrix
        Dmtrx (np.ndarray): Damping matrix
        wn (np.ndarray): Natural frequency vector [wn_x, wn_y, wn_psi] > 0
        zeta (float): Relative damping ratio > 0
    """
    K_p = wn @ wn @ Mmtrx
    K_d = 2.0 * zeta @ wn @ Mmtrx - Dmtrx
    K_i = (1.0 / 10.0) * wn @ K_p
    return K_p, K_d, K_i


@dataclass
class FLSC(IController):
    """Implements a feedback-linearizing surge-course (FLSC) controller for a single thruster+rudder
    (NOT true in practice, as the vessel has an outboard engine) Viknes vessel using

    Fx = (C(nu) * nu)[0] + (D(nu) * nu)[0] + M[0, 0] * (K_p,u * (u_d - u) + int_0^t K_i,u * (u_d - u))
    Fy = (M[2, 2] / l_r) * (K_p,chi * (chi_d - chi) + K_d,chi * (r_d - r) + int_0^t K_i,chi * (chi_d - chi))

    for the system

    eta_dot = J_Theta(eta) * nu
    M * nu_dot + C(nu) * nu + D(nu) * nu = tau

    with a rudder placed l_r units away from CG, such that we set tau = [Fx, 0.0, Fy * l_r]^T.
    Here, J_Theta(eta) = R(eta) for the 3DOF case with eta = [x, y, psi]^T, nu = [u, v, r]^T and xs = [eta, nu]^T.
    """

    def __init__(
        self,
        model_params,
        params: Optional[FLSCParams] = None,
        enable_separate_low_speed_control: bool = False,
    ) -> None:
        self._model_params = model_params
        if params is not None:
            self._params: FLSCParams = params
        else:
            self._params = FLSCParams()

        self._low_speed_ctrl_params = None
        if enable_separate_low_speed_control:
            # Tuned for the viknes model with disturbances considered
            self._low_speed_ctrl_params = FLSCParams(
                K_p_u=3.0,
                K_i_u=0.2,
                K_p_chi=1.5,
                K_d_chi=4.0,
                K_i_chi=0.1,
                max_speed_error_int=4.0,
                speed_error_int_threshold=0.4,
                max_chi_error_int=90.0 * np.pi / 180.0,
                chi_error_int_threshold=15.0 * np.pi / 180.0,
            )
        self._speed_error_int: float = 0.0
        self._chi_error_int: float = 0.0
        self._chi_d_prev: float = 0.0
        self._chi_prev: float = 0.0
        self._t: float = 0.0

    def reset(self):
        self._speed_error_int = 0.0
        self._chi_error_int = 0.0
        self._chi_d_prev = 0.0
        self._chi_prev = 0.0
        self._t = 0.0

    def update_integrators(
        self, speed_error: float, chi_error: float, dt: float
    ) -> None:
        """Updates the integrators for the FLSC controller.

        Args:
            speed_error (float): Current speed error
            chi_error (float): Current course error
            dt (float): Time step
        """
        if abs(speed_error) <= self._params.speed_error_int_threshold:
            self._speed_error_int += speed_error * dt

        # if abs(speed_error) < 0.01:
        #     self._speed_error_int = 0.0

        if abs(chi_error) <= self._params.chi_error_int_threshold:
            self._chi_error_int = mf.unwrap_angle(self._chi_error_int, chi_error * dt)

        # if abs(chi_error) < 0.1 * np.pi / 180.0:
        #     self._chi_error_int = 0.0

        self._speed_error_int = mf.sat(
            self._speed_error_int,
            -self._params.max_speed_error_int,
            self._params.max_speed_error_int,
        )
        self._chi_error_int = mf.sat(
            self._chi_error_int,
            -self._params.max_chi_error_int,
            self._params.max_chi_error_int,
        )

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
        """Computes inputs based on the proposed control law.

        Args:
            refs (np.ndarray): Desired/reference states [eta_d^T, eta_dot_d^T, eta_ddot_d^T]^T
            (equal to [0, 0, chi_d, U_d, 0, 0, 0, 0, 0]^T in case of LOSGuidance)
            xs (np.ndarray): State xs = [eta^T, nu^T]^T
            dt (float): Time step

        Returns:
            np.ndarray: Inputs u = tau to apply to the system.
        """
        if len(refs) != 9:
            raise ValueError("Dimension of reference array should be equal to 9!")

        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        eta = xs[0:3]
        nu = xs[3:]

        # assume eta_dot_d is in BODY frame
        u_d = refs[3]
        chi_d: float = mf.wrap_angle_to_pmpi(refs[2])
        chi_d_unwrapped = mf.unwrap_angle(self._chi_d_prev, chi_d)
        r_d = mf.sat(refs[5], -self._model_params.r_max, self._model_params.r_max)

        psi = mf.wrap_angle_to_pmpi(eta[2])
        speed = np.sqrt(nu[0] ** 2 + nu[1] ** 2)
        crab = np.arctan2(
            nu[1], nu[0]
        )  # if speed > 1.5 else 0.0  # to avoid oscillations when speed is low
        chi = mf.wrap_angle_to_pmpi(psi + crab)
        chi_unwrapped = mf.unwrap_angle(self._chi_prev, chi)

        self._chi_d_prev = chi_d
        self._chi_prev = eta[2]
        chi_error_prev = chi_d - chi

        Cvv = np.zeros(3)
        Dvv = np.zeros(3)
        Mmtrx = self._model_params.M_rb + self._model_params.M_a
        if self._model_params.name == "Viknes":
            Cvv = mf.Cmtrx(Mmtrx, nu) @ nu
            Dvv = (
                mf.Dmtrx(
                    self._model_params.D_l,
                    self._model_params.D_q,
                    self._model_params.D_c,
                    nu,
                )
                @ nu
            )
            l_r = self._model_params.l_r
        elif self._model_params.name == "R/V Gunnerus":
            C_RB = mf.coriolis_matrix_rigid_body(self._model_params.M_rb, nu)
            C_A = mf.coriolis_matrix_added_mass(self._model_params.M_a, nu)
            Cvv = C_RB @ nu + C_A @ nu
            Dvv = (
                self._model_params.D_l
                + self._model_params.D_u * abs(nu[0])
                + self._model_params.D_v * abs(nu[1])
                + self._model_params.D_r * abs(nu[2])
            ) @ nu
            l_r = abs(self._model_params.r_t[0])

        speed_error = u_d - speed
        chi_error: float = mf.wrap_angle_diff_to_pmpi(chi_d_unwrapped, chi_unwrapped)
        chi_error_unwrapped = mf.unwrap_angle(chi_error_prev, chi_error)
        self.update_integrators(speed_error, chi_error_unwrapped, dt)

        params = self._params
        if self._low_speed_ctrl_params is not None:
            params = self._params if speed > 1.5 else self._low_speed_ctrl_params

        # if chi_error > 0.2 and u_d < 1.0:
        # self._t += dt
        # if self._t > 50.0:
        #     print(
        #         f"speed error: {speed_error} | chi error: {180.0 * chi_error_unwrapped / np.pi} | speed error int: {self._speed_error_int} | chi error int: {180.0 * self._chi_error_int / np.pi}"
        #     )

        tau_X = (
            Cvv[0]
            + Dvv[0]
            + Mmtrx[0, 0]
            * (params.K_p_u * speed_error + params.K_i_u * self._speed_error_int)
        )
        tau_N = (Mmtrx[2, 2] / l_r) * (
            params.K_p_chi * chi_error
            + params.K_d_chi * (r_d - nu[2])
            + params.K_i_chi * self._chi_error_int
        )

        tau = np.array([float(tau_X), 0.0, float(l_r * tau_N)])

        return tau
