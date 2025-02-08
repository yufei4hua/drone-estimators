"""This file contains modified methods and classes taken from the filterpy library.

Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import absolute_import, annotations, division, print_function

from typing import TYPE_CHECKING, Callable

import numpy as np
from flax.struct import dataclass
from lsy_models.dataclasses import QuadrotorState
from scipy.linalg import block_diag

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class UKFData:
    """TODO."""

    state: QuadrotorState
    covariance: NDArray[np.floating]  # Covariance matrix

    sigmas_f: NDArray[np.floating]
    sigmas_h: NDArray[np.floating]

    u: NDArray[np.floating]  # input
    z: NDArray[np.floating]  # measurement
    dt: np.floating

    K: NDArray[np.floating]  # Kalman gain
    y: NDArray[np.floating]  # residual

    S: NDArray[np.floating]  # system uncertainty
    # SI: NDArray[np.floating]  # inverse system uncertainty

    @classmethod
    def create(
        cls,
        dim_u: int = 4,
        dim_z: int = 7,
        forces_dist: bool | None = False,
        torques_dist: bool | None = False,
    ) -> UKFData:
        """TODO."""
        state = QuadrotorState.create_empty(forces_dist, torques_dist)
        dim_x = QuadrotorState.get_state_dim(state)

        covariance = np.eye(dim_x)

        sigmas_f = np.zeros(dim_x)
        sigmas_h = np.zeros(dim_z)

        u = np.zeros(dim_u)  # input
        z = np.zeros(dim_z)  # measurement
        dt = 1

        K = np.zeros((dim_x, dim_z))  # Kalman gain
        y = np.zeros((dim_z))  # residual
        S = np.zeros((dim_z, dim_z))  # system uncertainty
        # SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        return cls(state, covariance, sigmas_f, sigmas_h, u, z, dt, K, y, S)  # , SI)


@dataclass
class UKFSettings:
    """TODO."""

    SPsettings: SigmaPointsSettings
    Q: NDArray[np.floating]
    R: NDArray[np.floating]
    fx: Callable
    hx: Callable

    @classmethod
    def create(
        cls,
        SPsettings: SigmaPointsSettings,
        Q: NDArray[np.floating],
        R: NDArray[np.floating],
        fx: Callable[[QuadrotorState, NDArray], QuadrotorState],
        hx: Callable[[QuadrotorState, NDArray], QuadrotorState],
    ) -> UKFSettings:
        """TODO."""
        return cls(SPsettings, Q, R, fx, hx)


def ukf_predict_correct(data: UKFData, settings: UKFSettings) -> UKFData:
    """TODO."""
    xp = data.covariance.__array_namespace__()
    #### Predict
    # Calculate sigma pointstf.transform.rotation.x,
    sigmas = ukf_calculate_sigma_points(data, settings)
    sigma_states = QuadrotorState.from_array(data.state, sigmas)

    # Pass sigma points through dynamics
    sigma_states_dot = settings.fx(sigma_states, data.u)
    sigmas_dot = QuadrotorState.as_array(sigma_states_dot)

    # Integrate dynamics if continuous
    # TODO implement proper integrator
    # TODO watch out for quaternion integration!
    sigmas_f = sigmas + sigmas_dot * data.dt
    # data = data.replace(sigmas_f=sigmas_f)

    # Compute prior with unscented transform
    x, P = ukf_unscented_transform(
        sigmas_f, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.Q
    )

    # save prior
    # data = data.replace(x=x, covariance=P)

    #### Correct
    # Pass prior sigmas through measurment function h(x,u,dt) to get measurement sigmas
    # sigmas_h = settings.hx(sigmas_f, data.u, data.dt)
    sigmas_h = sigmas_f[..., :7]  # TODO remove this Ghetto version
    # data = data.replace(sigmas_h=sigmas_h)

    # Pass mean and covariance of prediction through unscented transform
    zp, S = ukf_unscented_transform(
        sigmas_h, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.R
    )
    # SI = xp.linalg.inv(data.S)
    # data = data.replace(S=S, SI=SI)
    # data = data.replace(S=S)

    # compute cross variance
    Pxz = ukf_cross_variance(x, zp, sigmas_f, sigmas_h, settings.SPsettings.Wc)
    # K = xp.dot(Pxz, data.SI)       # Kalman gain
    # K @ S = Pxz => K = Pxz @ S^-1 => or: S.T @ K.T = Pxz.T
    K = xp.linalg.solve(S.T, Pxz.T).T
    y = xp.subtract(data.z, zp)  # residual
    # data = data.replace(K=K, y=y)

    # Update Gaussian state estimate (x, P)
    x = x + xp.dot(K, y)
    P = P - xp.dot(K, xp.dot(S, K.T))

    # Save posterior
    state = QuadrotorState.from_array(data.state, x)
    data = data.replace(state=state, covariance=P)

    return data


# Legacy code
# def ukf_predict(data: UKFData, settings: UKFSettings) -> UKFData:
#     """TODO."""
#     # Calculate sigma points
#     sigmas = ukf_calculate_sigma_points(data, settings)
#     sigma_states = QuadrotorState.from_array(data.state, sigmas)

#     # Pass sigma points through dynamics
#     sigma_states_dot = settings.fx(sigma_states, data.u)
#     sigmas_dot = QuadrotorState.as_array(sigma_states_dot)

#     # Integrate dynamics if continuous
#     # TODO implement proper integrator
#     # TODO watch out for quaternion integration!
#     sigmas_f = sigmas + sigmas_dot * data.dt
#     data = data.replace(sigmas_f=sigmas_f)

#     # Compute prior with unscented transform
#     x, P = ukf_unscented_transform(
#         data.sigmas_f, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.Q
#     )

#     # save prior
#     data = data.replace(x=x, covariance=P)

#     return data


# def ukf_correct(data: UKFData, settings: UKFSettings) -> UKFData:
#     """TODO."""
#     xp = data.covariance.__array_namespace__()
#     # Pass prior sigmas through measurment function h(x,u,dt) to get measurement sigmas
#     sigmas_h = settings.hx(data.sigmas_f, data.u, data.dt)
#     data = data.replace(sigmas_h=sigmas_h)

#     # Pass mean and covariance of prediction through unscented transform
#     zp, S = ukf_unscented_transform(
#         data.sigmas_h, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.R
#     )
#     # SI = xp.linalg.inv(data.S)
#     # data = data.replace(S=S, SI=SI)
#     data = data.replace(S=S)

#     # compute cross variance
#     Pxz = ukf_cross_variance(data.x, zp, data.sigmas_f, data.sigmas_h, settings.SPsettings.Wc)
#     # K = xp.dot(Pxz, data.SI)       # Kalman gain
#     # K @ S = Pxz => K = Pxz @ S^-1 => or: S.T @ K.T = Pxz.T
#     K = xp.linalg.solve(S.T, Pxz.T).T
#     y = xp.subtract(data.z, zp)  # residual
#     data = data.replace(K=K, y=y)

#     # Update Gaussian state estimate (x, P)
#     x = data.x + xp.dot(data.K, data.y)
#     P = data.P - xp.dot(data.K, xp.dot(data.S, data.K.T))

#     # Safe posterior
#     data = data.replace(x=x, P=P, x_post=x, P_post=P)

#     return data


def ukf_calculate_sigma_points(data: UKFData, settings: UKFSettings) -> NDArray[np.floating]:
    """TODO."""
    xp = data.covariance.__array_namespace__()
    U = xp.linalg.cholesky(
        (settings.SPsettings.lambda_ + settings.SPsettings.n) * data.covariance, upper=True
    )

    state_array = QuadrotorState.as_array(data.state)
    sigma_center = state_array
    # TODO for the quaternions use more sophisitcated approach to keep length 1!
    sigma_pos = xp.subtract(state_array, -U)
    sigma_neg = xp.subtract(state_array, U)
    return xp.vstack((sigma_center, sigma_pos, sigma_neg))


def ukf_unscented_transform(
    sigmas: NDArray[np.floating],
    Wm: NDArray[np.floating],
    Wc: NDArray[np.floating],
    noise_cov: NDArray[np.floating] = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """TODO."""
    xp = sigmas.__array_namespace__()
    x = xp.dot(Wm, sigmas)

    # new covariance is the sum of the outer product of the residuals
    # times the weights
    y = sigmas - x[xp.newaxis, :]  # TODO replace with None?
    P = xp.dot(y.T, xp.dot(xp.diag(Wc), y))

    if noise_cov is not None:
        P += noise_cov

    return (x, P)


def ukf_cross_variance(
    x: NDArray[np.floating],
    z: NDArray[np.floating],
    sigmas_f: NDArray[np.floating],
    sigmas_h: NDArray[np.floating],
    Wc: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute cross variance of the state `x` and measurement `z`."""
    xp = x.__array_namespace__()
    # The slicing brings Wc in the correct shape to be broadcast
    # The einsum as set up here takes the outer product of all the stacked vectors
    Pxz = Wc[:, None, None] * xp.einsum(
        "bi,bj->bij", xp.subtract(sigmas_f, x), xp.subtract(sigmas_h, z)
    )
    Pxz = xp.sum(Pxz, axis=0)
    return Pxz


@dataclass
class SigmaPointsSettings:
    """TODO."""

    n: int
    alpha: float
    beta: float
    kappa: float
    lambda_: float
    Wc: NDArray[np.floating]
    Wm: NDArray[np.floating]

    @classmethod
    def create(cls, n: int, alpha: float, beta: float, kappa: float = 0.0) -> SigmaPointsSettings:
        """TODO."""
        lambda_ = alpha**2 * (n + kappa) - n
        c = 0.5 / (n + lambda_)
        Wc0 = np.array([lambda_ / (n + lambda_) + (1 - alpha**2 + beta)])
        Wm0 = np.array([lambda_ / (n + lambda_)])
        Wc = np.full(2 * n, c)
        Wm = np.full(2 * n, c)
        Wc = np.concat((Wc0, Wc))
        Wm = np.concat((Wm0, Wm))

        return cls(n, alpha, beta, kappa, lambda_, Wc, Wm)


def order_by_derivative(Q: NDArray[np.floating], dim: int, block_size: int) -> NDArray[np.floating]:
    """TODO."""
    xp = Q.__array_namespace__()

    N = dim * block_size

    D = xp.zeros((N, N))

    Q = xp.array(Q)
    for i, x in enumerate(Q.ravel()):
        f = xp.eye(block_size) * x

        ix, iy = (i // dim) * block_size, (i % dim) * block_size
        D[ix : ix + block_size, iy : iy + block_size] = f

    return D


def Q_discrete_white_noise(
    dim: int, dt: float = 1.0, var: float = 1.0, block_size: int = 1, order_by_dim: bool = True
) -> NDArray[np.floating]:
    """TODO."""
    if not (dim == 2 or dim == 3 or dim == 4):
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]]
    elif dim == 3:
        Q = [
            [0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
            [0.5 * dt**3, dt**2, dt],
            [0.5 * dt**2, dt, 1],
        ]
    else:
        Q = [
            [(dt**6) / 36, (dt**5) / 12, (dt**4) / 6, (dt**3) / 6],
            [(dt**5) / 12, (dt**4) / 4, (dt**3) / 2, (dt**2) / 2],
            [(dt**4) / 6, (dt**3) / 2, dt**2, dt],
            [(dt**3) / 6, (dt**2) / 2, dt, 1.0],
        ]

    if order_by_dim:
        return block_diag(*[Q] * block_size) * var
    return order_by_derivative(np.array(Q), dim, block_size) * var
