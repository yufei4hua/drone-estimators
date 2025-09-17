"""This file implements some wrappers to get the dynamics from 'drone-models'."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from drone_models.so_rpy_rotor_drag import dynamics as so_rpy_rotor_drag_dynamics
from drone_models.so_rpy_rotor_drag.params import SoRpyRotorDragParams

if TYPE_CHECKING:
    from collections.abc import Callable

    from array_api_typing import Array


def get_dynamics(
    model_type: str, drone_model: str
) -> Callable[
    [Array, Array, Array, Array, Array, Array | None, Array | None, Array | None],
    tuple[Array, Array, Array, Array, Array | None],
]:
    """Simplifies drone-model dynamics by adding all arguments for the given drone_model."""
    match model_type:
        case "so_rpy_rotor_drag_dynamics":
            params = SoRpyRotorDragParams.load(drone_model)
            return partial(
                so_rpy_rotor_drag_dynamics,
                mass=params.mass,
                gravity_vec=params.gravity_vec,
                J=params.J,
                J_inv=params.J_inv,
                KF=params.KF,
                KM=params.KM,
                thrust_time_coef=params.thrust_time_coef,
                acc_coef=params.acc_coef,
                cmd_f_coef=params.cmd_f_coef,
                rpy_coef=params.rpy_coef,
                rpy_rates_coef=params.rpy_rates_coef,
                cmd_rpy_coef=params.cmd_rpy_coef,
                drag_linear_coef=params.drag_linear_coef,
                drag_square_coef=params.drag_square_coef,
            )
        case _:
            raise NotImplementedError(f"Model type {model_type} not supported.")


def observation_function(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    cmd: Array,
    rotor_vel: Array | None = None,
    dist_f: Array | None = None,
    dist_t: Array | None = None,
) -> Array:
    """Return the observable part of the state.

    This is basically not necessary, since we always get position and orientation
    from Vicon. However, for sake of completeness, this observation function is added.
    """
    xp = pos.__array_namespace__()
    return xp.concat((pos, quat), axis=-1)
