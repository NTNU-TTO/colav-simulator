"""math_functions.py.

Summary:
Contains common math functions.

Author: Trym Tengesdal
"""

import math

import numpy as np


def find_dependent_rows(A: np.ndarray) -> list:
    """Find the indices of linearly dependent rows in a rank-deficient matrix A.

    Args:
        A (numpy.ndarray): The input matrix

    Returns:
        List[int]: A list of indices of linearly dependent rows in the original matrix
    """
    rows, cols = A.shape
    row_order = np.arange(rows)  # Track the original row indices

    # Perform Gaussian elimination to row echelon form
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        # Find the pivot row and swap
        max_row = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
        A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
        row_order[[pivot_row, max_row]] = row_order[[max_row, pivot_row]]  # Swap the row order as well
        pivot = A[pivot_row, col]
        if np.isclose(pivot, 0):
            continue
        # Normalize the pivot row
        A[pivot_row] = A[pivot_row] / pivot
        # Eliminate below
        for row in range(pivot_row + 1, rows):
            A[row] = A[row] - A[row, col] * A[pivot_row]
        pivot_row += 1

    # Identify dependent rows
    dependent_rows = []
    for row in range(rows):
        if np.all(np.isclose(A[row], 0)):
            dependent_rows.append(row_order[row])

    return sorted(dependent_rows)


def cpa(p_A: np.ndarray, v_A: np.ndarray, p_B: np.ndarray, v_B: np.ndarray) -> tuple[float, float]:
    """Computes the closest point of approach (CPA) between two objects A and B.

    Args:
        p_A (np.ndarray): Position of object A.
        v_A (np.ndarray): Velocity of object A.
        p_B (np.ndarray): Position of object B.
        v_B (np.ndarray): Velocity of object B.

    Returns:
        Tuple[float, float]: Tuple containing the time and distance to CPA.
    """
    p_AB = p_B - p_A
    v_AB = v_B - v_A
    v_AB_norm = np.linalg.norm(v_AB)
    if v_AB_norm < 0.000001:
        return np.inf, np.inf
    else:
        t_cpa = float(-np.dot(p_AB, v_AB) / (v_AB_norm * v_AB_norm))
        d_cpa = float(np.linalg.norm(p_AB + t_cpa * v_AB))
        return t_cpa, d_cpa


def linear_map(v: float, x: tuple[float, float], y: tuple[float, float]) -> float:
    """Linearly maps v from x to y.

    Args:
        v (float): Value to map
        x (Tuple[float, float]): Input range
        y (Tuple[float, float]): Output range

    Returns:
        float: Mapped value
    """
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def wrap_min_max(x: float | np.ndarray, x_min: float | np.ndarray, x_max: float | np.ndarray) -> float | np.ndarray:
    """Wraps input x to [x_min, x_max).

    Args:
        x (float or np.ndarray): Unwrapped value
        x_min (float or np.ndarray): Minimum value
        x_max (float or np.ndarray): Maximum value

    Returns:
        float or np.ndarray: Wrapped value
    """
    if isinstance(x, np.ndarray):
        return x_min + np.mod(x - x_min, x_max - x_min)
    else:
        return x_min + (x - x_min) % (x_max - x_min)


def wrap_angle_to_pmpi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wraps input angle to [-pi, pi).

    Args:
        angle (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle
    """
    if isinstance(angle, np.ndarray):
        return wrap_min_max(angle, -np.pi * np.ones(angle.size), np.pi * np.ones(angle.size))
    else:
        return wrap_min_max(angle, -np.pi, np.pi)


def wrap_angle_to_02pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wraps input angle to [0, 2pi).

    Args:
        angle (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle
    """
    if isinstance(angle, np.ndarray):
        return wrap_min_max(angle, np.zeros(angle.size), 2 * np.pi * np.ones(angle.size))
    else:
        return wrap_min_max(angle, 0, 2 * np.pi)


def wrap_angle_diff_to_02pi(a_1: float | np.ndarray, a_2: float | np.ndarray) -> float | np.ndarray:
    """Wraps angle difference a_1 - a_2 to within [0, 2pi).

    Args:
        a_1 (float or np.ndarray): Angle in radians
        a_2 (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle difference
    """
    diff = wrap_angle_to_02pi(a_1) - wrap_angle_to_02pi(a_2)
    if isinstance(diff, np.ndarray):
        return wrap_min_max(diff, np.zeros(diff.size), 2 * np.pi * np.ones(diff.size))
    else:
        return wrap_min_max(diff, 0, 2 * np.pi)


def wrap_angle_diff_to_pmpi(a_1: float | np.ndarray, a_2: float | np.ndarray) -> float | np.ndarray:
    """Wraps angle difference a_1 - a_2 to within [-pi, pi).

    Args:
        a_1 (float or np.ndarray): Angle in radians
        a_2 (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle difference
    """
    diff = a_1 - a_2
    if isinstance(diff, np.ndarray):
        return wrap_min_max(diff, -np.pi * np.ones(diff.size), np.pi * np.ones(diff.size))
    else:
        return wrap_min_max(diff, -np.pi, np.pi)


def unwrap_angle(a_prev: float, a: float) -> float:
    """Unwraps angle a to a_prev + angle difference.

    Args:
        a_prev (float): Previous angle in radians
        a (float): Angle in radians

    Returns:
        float: Unwrapped angle
    """
    return a_prev + float(wrap_angle_diff_to_pmpi(a, a_prev))


def unwrap_angle_array(angles: np.ndarray) -> np.ndarray:
    """Unwraps an array of angles.

    Args:
        angles (np.ndarray): Array of angles in radians

    Returns:
        np.ndarray: Unwrapped angles
    """
    unwrapped = [angles[0]]
    for i in range(1, angles.size):
        unwrapped.append(unwrap_angle(unwrapped[i - 1], angles[i]))
    return np.array(unwrapped)


def compute_bearing(psi: float, p1: np.ndarray, p2: np.ndarray) -> float:
    """Computes bearing in [-180, 180) from p1 to p2, where the object at p1 has heading psi.

    Args:
        psi (float): Current heading of object at p1.
        p1 (np.ndarray): Position of first object.
        p2 (np.ndarray): Position of second object.
    """
    return float(wrap_angle_diff_to_pmpi(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]), psi))


def knots2mps(knots: float) -> float:
    """Converts from knots to miles per second.

    Args:
        knots (float): Knots to convert.

    Returns:
        float: Knots converted to mps.
    """
    mps = knots * 1.852 / 3.6
    return mps


def cm2inch(cm: float) -> float:  # inch to cm
    """Converts from cm to inches.

    Args:
        cm (float): centimetres to convert.

    Returns:
        float: Resulting inches.
    """
    return cm / 2.54


def mps2knots(mps: float) -> float:
    """Converts from miles per second to knots.

    Args:
        mps (float): Mps to convert.

    Returns:
        float: mps converted to knots.
    """
    knots = mps * 3.6 / 1.852
    return knots


def ms2knots(ms: float) -> float:
    """Converts from m/s to knots.

    Args:
        ms (float): m/s to convert.

    Returns:
        float: Resulting knots.
    """
    return ms * 1.94384


def knots2ms(knots: float) -> float:
    """Converts from knots to m/s.

    Args:
        knots (float): Knots to convert.

    Returns:
        float: Resulting m/s.
    """
    return knots * 0.514444


def normalize_vec(v: np.ndarray) -> np.ndarray:
    """Normalize vector v to length 1.

    Args:
        v (np.ndarray): Vector to normalize.

    Returns:
        np.ndarray: Normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm < 0.000001:
        return v
    else:
        return v / norm


def sat(x: float | np.ndarray, x_min: float | np.ndarray, x_max: float | np.ndarray) -> float | np.ndarray:
    """Saturates a signal x such that x_min <= x <= x_max.

    Args:
        x (float | np.ndarray): Signal to saturate.
        x_min (float | np.ndarray): Minimum value.
        x_max (float | np.ndarray): Maximum value.

    Returns:
        float | np.ndarray: Saturated signal.
    """
    return np.clip(x, x_min, x_max)


def coriolis_matrix_rigid_body(M_RB: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Calculates the rigid body Coriolis matrix C_RB(v).

    Assumes decoupled surge and sway-yaw dynamics, as in eq. 7.13 in Fossen
    (2011).

    Args:
        M_RB (np.ndarray): Rigid body mass matrix.
        nu (np.ndarray): Body-frame velocity nu = [u, v, r]^T.

    Returns:
        np.ndarray: Coriolis matrix C_RB(v).
    """
    c13 = -M_RB[1, 2] * nu[2] - M_RB[1, 1] * nu[1]
    c23 = M_RB[0, 0] * nu[0]
    return np.array([[0, 0, c13], [0, 0, c23], [-c13, -c23, 0]])


def coriolis_matrix_added_mass(M_A: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Calculates the added mass Coriolis matrix C_A(v).

    Assumes decoupled surge and sway-yaw dynamics, for non-symmetric added mass
    matrix.

    Args:
        M_A (np.ndarray): Added mass matrix.
        nu (np.ndarray): Body-frame velocity nu = [u, v, r]^T.

    Returns:
        np.ndarray: Coriolis matrix C_A(v).
    """
    c13 = -M_A[1, 1] * nu[1] - 0.5 * (M_A[2, 1] + M_A[1, 2]) * nu[2]
    c23 = M_A[0, 0] * nu[0]
    return np.array([[0, 0, c13], [0, 0, c23], [-c13, -c23, 0]])


def Cmtrx(Mmtrx: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Calculates coriolis matrix C(v).

    Assumes decoupled surge and sway-yaw dynamics, as in eq. (7.12) - (7.15) in
    Fossen2011.

    Args:
        Mmtrx (np.ndarray): Mass matrix (M_RB + M_A).
        nu (np.ndarray): Body-frame velocity nu = [u, v, r]^T.

    Returns:
        np.ndarray: Coriolis matrix C(v).
    """
    c13 = -(Mmtrx[1, 1] * nu[1] + Mmtrx[1, 2] * nu[2])
    c23 = Mmtrx[0, 0] * nu[0]

    return np.array([[0, 0, c13], [0, 0, c23], [-c13, -c23, 0]])


def Dmtrx(D_l: np.ndarray, D_q: np.ndarray, D_c: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Calculates damping matrix D(nu).

    Assumes decoupled surge and sway-yaw dynamics, as in eq. (7.24) in
    Fossen2011+.

    Args:
        D_l (np.ndarray): Linear damping matrix.
        D_q (np.ndarray): Quadratic damping matrix.
        D_c (np.ndarray): Cubic damping matrix.
        nu (np.ndarray): Body-frame velocity nu = [u, v, r]^T.

    Returns:
        np.ndarray: Damping matrix D = D_l + D_q(nu) + D_c(nu).
    """
    return D_l + D_q * np.abs(nu) + D_c * (nu * nu)


def Smtrx(a: np.ndarray) -> np.ndarray:
    """Computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.

    The cross product satisfies: a x b = S(a)b.

    Args:
        a (np.ndarray): Input vector.

    Returns:
        np.ndarray: Skew-symmetric matrix S(a).
    """
    S = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    return S


def Hmtrx(r: np.ndarray) -> np.ndarray:
    """Computes the 6x6 system transformation matrix.

    H = [np.eye(3)     S'
         np.zeros((3,3)) np.eye(3) ]

    Property: inv(H(r)) = H(-r). If r = r_bg is the vector from the CO to the
    CG, the model matrices in CO and CG are related by:
    M_CO = H(r_bg)' * M_CG * H(r_bg).

    Generalized position and force satisfy:
    eta_CO = H(r_bg)' * eta_CG and tau_CO = H(r_bg)' * tau_CG.

    Args:
        r (np.ndarray): Input vector.

    Returns:
        np.ndarray: System transformation matrix H.
    """
    H = np.identity(6, float)
    H[0:3, 3:6] = Smtrx(r).T

    return H


def Rzyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """Computes the Euler angle rotation matrix R in SO(3).

    Uses the zyx convention.

    Args:
        phi (float): Roll angle in radians.
        theta (float): Pitch angle in radians.
        psi (float): Yaw angle in radians.

    Returns:
        np.ndarray: Rotation matrix R.
    """
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)

    R: np.ndarray = np.array(
        [
            [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
            [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
            [-sth, cth * sphi, cth * cphi],
        ]
    )

    return R


def Rmtrx(psi: float) -> np.ndarray:
    """Computes the 3x3 rotation matrix of an angle psi about the z-axis.

    Args:
        psi (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotation matrix R.
    """
    return np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])


def Rmtrx2D(psi: float) -> np.ndarray:
    """Computes the 2D rotation matrix.

    Rmtrx = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]]).

    Args:
        psi (float): Rotation angle in radians.

    Returns:
        np.ndarray: 2D rotation matrix.
    """
    return np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])


def Jtheta(Theta: np.ndarray) -> np.ndarray:
    """Computes the transformation matrix.

    eta_dot = J_Theta(eta) * nu using the zyx convention. eta = [x, y, z, phi,
    theta, psi].

    Args:
        Theta (np.ndarray): Euler angles [phi, theta, psi]^T.

    Returns:
        np.ndarray: Transformation matrix J.
    """
    phi = Theta[0]
    theta = Theta[1]
    psi = Theta[2]
    Jmtrx = np.zeros((6, 6))
    Jmtrx[0:3, 0:3] = Rzyx(phi, theta, psi)
    Jmtrx[3:6, 3:6] = Tzyx(phi, theta)
    return Jmtrx


def Tzyx(phi: float, theta: float) -> np.ndarray:
    """Computes the Euler angle attitude transformation matrix T.

    Uses the zyx convention.

    Args:
        phi (float): Roll angle in radians.
        theta (float): Pitch angle in radians.

    Returns:
        np.ndarray: Transformation matrix T.
    """
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)

    try:
        Tmtrx = np.array([[1, sphi * sth / cth, cphi * sth / cth], [0, cphi, -sphi], [0, sphi / cth, cphi / cth]])

    except ZeroDivisionError:
        print("Tzyx is singular for theta = +-90 degrees.")

    return Tmtrx


def m2c(M: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Computes the Coriolis and centripetal matrix C.

    From the mass matrix M and generalized velocity vector nu (Fossen 2021,
    Ch. 3).

    Args:
        M (np.ndarray): Mass matrix.
        nu (np.ndarray): Generalized velocity vector.

    Returns:
        np.ndarray: Coriolis and centripetal matrix C.
    """
    M = 0.5 * (M + M.T)  # systematization of the inertia matrix

    #  6-DOF model
    if len(nu) == 6:
        M11mtrx = M[0:3, 0:3]
        M12mtrx = M[0:3, 3:6]
        M21mtrx = M12mtrx.T
        M22mtrx = M[3:6, 3:6]

        nu1 = nu[0:3]
        nu2 = nu[3:6]
        dt_dnu1 = np.matmul(M11mtrx, nu1) + np.matmul(M12mtrx, nu2)
        dt_dnu2 = np.matmul(M21mtrx, nu1) + np.matmul(M22mtrx, nu2)

        # Cmtrx  = [  zeros(3,3)      -Smtrx(dt_dnu1)
        #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
        Cmtrx = np.zeros((6, 6))
        Cmtrx[0:3, 3:6] = -Smtrx(dt_dnu1)
        Cmtrx[3:6, 0:3] = -Smtrx(dt_dnu1)
        Cmtrx[3:6, 3:6] = -Smtrx(dt_dnu2)

    else:  # 3-DOF model (surge, sway and yaw)
        # Cmtrx = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
        #      0             0             M(1,1)*nu(1)
        #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]
        Cmtrx = np.zeros((3, 3))
        Cmtrx[0, 2] = -M[1, 1] * nu[1] - M[1, 2] * nu[2]
        Cmtrx[1, 2] = M[0, 0] * nu[0]
        Cmtrx[2, 0] = -Cmtrx[0, 2]
        Cmtrx[2, 1] = -Cmtrx[1, 2]

    return Cmtrx
