"""Based on https://github.com/utiasDSL/crazyflow/blob/d38bb5c75fe6624972ccc18d89789d3636cfd8cd/crazyflow/sim/integration.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from array_api_typing import Array

    from drone_estimators.structs import UKFData


def integrate_UKFData(state: UKFData, state_dot: UKFData) -> UKFData:
    """Integrates UKFData properly."""
    next_pos, next_quat, next_vel, next_ang_vel, next_forces_motor = _integrate(
        state.pos,
        state.quat,
        state.vel,
        state.ang_vel,
        state_dot.pos,
        state_dot.quat,
        state_dot.vel,
        state_dot.ang_vel,
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
        ang_vel=next_ang_vel,
        forces_motor=next_forces_motor,
    )


def _integrate(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    pos_dot: Array,
    quat_dot: Array,
    vel_dot: Array,
    ang_vel_dot: Array,
    dt: float,
    forces_motor: Array | None = None,
    forces_motor_dot: Array | None = None,
) -> Array:  # TODO is actually tuple
    """Integrate the dynamics forward in time.

    Args:
        pos: The position of the drone.
        quat: The orientation of the drone as a quaternion.
        vel: The velocity of the drone.
        ang_vel: The angular velocity of the drone.
        pos_dot: The derivative of the position of the drone.
        quat_dot: The derivative of the quaternion of the drone.
        vel_dot: The derivative of the velocity of the drone.
        ang_vel_dot: The derivative of the angular velocity of the drone.
        dt: The time step to integrate over.
        forces_motor: The forces for the motors.
        forces_motor_dot: The derivative of the motor forces.

    Returns:
        The next position, quaternion, velocity, and roll, pitch, and yaw rates of the drone.
    """
    next_pos = pos + pos_dot * dt
    next_quat = (R.from_quat(quat) * R.from_rotvec(ang_vel * dt)).as_quat()
    next_vel = vel + vel_dot * dt
    next_ang_vel = ang_vel + ang_vel_dot * dt
    next_forces_motor = None
    if forces_motor is not None:
        next_forces_motor = forces_motor + forces_motor_dot * dt

    return next_pos, next_quat, next_vel, next_ang_vel, next_forces_motor
