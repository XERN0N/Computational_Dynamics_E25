"""
Author:     Oskar Minds
Date:       Oct. 2025

Description
-----------
Example solution for exercise Q1.2 in Computation Dynamics.
Formula references are according to Jain, A. ()
Required libraries:
    - numpy
    - scipy
    - matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

def skew_symmetric_matrix(a: npt.ArrayLike) -> npt.NDArray:
    """
    Gets the skew symmetric matrix of a vector.

    Parameters
    ----------
    a : array_like
        The vector with shape (3,).

    Returns
    -------
    npt.NDArray
        The skew symmetric version of vector 'a' with shape (3, 3).
    """
    a = np.asarray(a)
    return np.array([[    0, -a[2],  a[1]],
                     [ a[2],     0, -a[0]],
                     [-a[1],  a[0],     0]])

def quaternion_to_rotation_matrix(quaternion: npt.ArrayLike) -> npt.NDArray:
    """
    Gets the rotation matrix from a quaternion based on Equation B.14.

    Parameters
    ----------
    quaternion : array_like
        The quaternion with shape (4,) ordered as in Equation B.12.

    Returns
    -------
    npt.NDArray
        The rotation matrix with shape (3, 3).
    """
    q, q0 = np.split(quaternion, (3,))
    q_tilde = skew_symmetric_matrix(q)
    # Relating the rotation matric to the quaternion using equation B.14.
    return np.eye(3) + 2 * (q0 * np.eye(3) + q_tilde) @ q_tilde

def quaternion_equation_of_motion(time: float, quaternion: npt.ArrayLike, angular_velocity: npt.ArrayLike) -> npt.NDArray:
    """
    Calculates the derivative of the quaternion with respect to time based on the quaternion and the
    angular velocity.

    Parameters
    ----------
    time : float
        Current time (ignored but its presence is needed for the integrator).
    quaternion : array_like
        The quaternion with shape (4,).
    angular_velocity : array_like
        The angular velocity of the body with shape (3,).

    Returns
    -------
    npt.NDArray
        The derivative of the quaternion with respect to time with shape (4,).
    """
    quaternion = np.asarray(quaternion)
    angular_velocity = np.asarray(angular_velocity)
    # Calculates the derivative of the quaternion with respect to time acc. to Equation B.34.
    return 1/2 * np.block([
        [-skew_symmetric_matrix(angular_velocity), angular_velocity.reshape(3, 1)],
        [                       -angular_velocity,                              0]
    ]) @ quaternion

# Parameters.
t_eval = np.linspace(0, 1, 100)
initial_condition = (0, 0, 0, 1)
angular_velocity = (1.0, 0.1, 0.6)

# Solving using the scipy.integrate.solve_ivp integrator (same as ode45 in matlab).
solution = solve_ivp(quaternion_equation_of_motion, (t_eval[0], t_eval[-1]), initial_condition, t_eval=t_eval, args=(angular_velocity,))

# Combining all rotation matrices for the entire simulation into one matrix with shape (100, 3, 3).
rotations = np.array([quaternion_to_rotation_matrix(quaternion) for quaternion in solution.y.T])

# Determining the basis vectors for all time steps.
x_basis_vector = (rotations @ (1, 0, 0)).T
y_basis_vector = (rotations @ (0, 1, 0)).T
z_basis_vector = (rotations @ (0, 0, 1)).T

# Plotting the final position of the basis vectors.
ax = plt.figure().add_subplot(projection='3d')
ax.plot((0, x_basis_vector[0, -1]), (0, x_basis_vector[1, -1]), (0, x_basis_vector[2, -1]), color="blue" , label='$\\mathbb{F}_{\\^x}$')
ax.plot((0, y_basis_vector[0, -1]), (0, y_basis_vector[1, -1]), (0, y_basis_vector[2, -1]), color="red"  , label='$\\mathbb{F}_{\\^y}$')
ax.plot((0, z_basis_vector[0, -1]), (0, z_basis_vector[1, -1]), (0, z_basis_vector[2, -1]), color="green", label='$\\mathbb{F}_{\\^z}$')

# Plotting the reference frame.
ax.plot((0, 1.5), (0,   0), (0,   0), color='grey', label='$\\mathbb{G}$')
ax.plot((0,   0), (0, 1.5), (0,   0), color='grey')
ax.plot((0,   0), (0,   0), (0, 1.5), color='grey')

# Plotting the trajectory of the moving frame.
ax.plot(x_basis_vector[0], x_basis_vector[1], x_basis_vector[2], color="blue",  linestyle="--")
ax.plot(y_basis_vector[0], y_basis_vector[1], y_basis_vector[2], color="red",   linestyle="--")
ax.plot(z_basis_vector[0], z_basis_vector[1], z_basis_vector[2], color="green", linestyle="--")

# Optional plot settings.
ax.set_aspect('equal')
ax.set_axis_off()
legend = ax.legend(prop={'size': 15})
legend.set_bbox_to_anchor((0.7, 0.6))

plt.show()