"""Based on https://github.com/utiasDSL/crazyflow/blob/d38bb5c75fe6624972ccc18d89789d3636cfd8cd/crazyflow/sim/integration.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lsy_models.utils import rotation as R

from lsy_estimators.datacls import UKFData

if TYPE_CHECKING:
    from numpy.typing import NDArray


def integrate_UKFData(state: UKFData, state_dot: UKFData) -> UKFData:
    next_pos, next_quat, next_vel, next_angvel, next_forces_motor = _integrate(
        state.pos,
        state.quat,
        state.vel,
        state.angvel,
        state_dot.pos,
        state_dot.quat,
        state_dot.vel,
        state_dot.angvel,
        state.dt,
        state.forces_motor,
        state_dot.forces_motor,
    )
    # TODO unit norm of quaternion!
    # TODO implement different integrator types later
    return state.replace(
        pos=next_pos,
        quat=next_quat,
        vel=next_vel,
        angvel=next_angvel,
        forces_motor=next_forces_motor,
    )


def _integrate(
    pos: NDArray,
    quat: NDArray,
    vel: NDArray,
    angvel: NDArray,
    pos_dot: NDArray,
    quat_dot: NDArray,
    vel_dot: NDArray,
    angvel_dot: NDArray,
    dt: float,
    forces_motor: NDArray | None = None,
    forces_motor_dot: NDArray | None = None,
) -> NDArray:  # TODO is actually tuple
    """Integrate the dynamics forward in time.

    Args:
        pos: The position of the drone.
        quat: The orientation of the drone as a quaternion.
        vel: The velocity of the drone.
        ang_vel: The angular velocity of the drone.
        dpos: The derivative of the position of the drone.
        drot: The derivative of the quaternion of the drone (3D angular velocity).
        dvel: The derivative of the velocity of the drone.
        dang_vel: The derivative of the angular velocity of the drone.
        dt: The time step to integrate over.

    Returns:
        The next position, quaternion, velocity, and roll, pitch, and yaw rates of the drone.
    """
    next_pos = pos + pos_dot * dt
    next_quat = (R.from_quat(quat) * R.from_rotvec(angvel * dt)).as_quat()
    next_vel = vel + vel_dot * dt
    next_angvel = angvel + angvel_dot * dt
    next_forces_motor = None
    if forces_motor is not None:
        next_forces_motor = forces_motor + forces_motor_dot * dt

    return next_pos, next_quat, next_vel, next_angvel, next_forces_motor
