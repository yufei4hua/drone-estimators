"""This file implements some wrappers to get the dynamics from 'drone-models'."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from drone_models.core import parametrize

if TYPE_CHECKING:
    from collections.abc import Callable

    from array_api_typing import Array


def dynamics_function(model:str, config:str):
    """TODO."""
    # Idea: 
    # from drone_models.{model}.model import dynamics as fn
    
    # return parametrize(fn, config)
    ...


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
