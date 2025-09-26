"""Testing the selfimplemented rotations against scipy rotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from drone_models import available_models, model_features
from drone_models.drones import available_drones

from drone_estimators.estimator import KalmanFilter
from drone_estimators.utils.dynamics import dynamics_function

if TYPE_CHECKING:
    from typing import Callable


@pytest.mark.unit
def test_placeholder():
    """Placeholder test."""
    pass


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_type", available_drones)
@pytest.mark.unit
def test_model_loading(model_name: str, model: Callable, drone_type: str):
    """Tests if the models for the kalman filters can be imported."""
    dynamics_function(model_name, drone_type)


# TODO test if the whole filterpy chain is jitable


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_type", available_drones)
@pytest.mark.unit
def test_kalman(model_name: str, model: Callable, drone_type: str):
    """Tests if the Kalman filter can be imported and stepped."""
    supports_dynamics = model_features(dynamics_function(model_name, drone_type))["rotor_dynamics"]
    kf = KalmanFilter(1 / 200, model_name, drone_type, estimate_rotor_vel=supports_dynamics)

    kf.predict(1 / 240, np.array([0.0, 0.0, 0.0, 0.5]))

    kf.correct(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
