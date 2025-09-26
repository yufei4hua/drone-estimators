$$
\huge \displaystyle \hat{\boldsymbol{\mu}}_{x,k+1} = \check{\boldsymbol{\mu}}_{x,k+1} + \mathbf{K}_{k+1} \mathbf{z}_{k+1} \\
\hat{\boldsymbol{\Sigma}}_{xx, k+1} = ( \mathbf{I}-\mathbf{K}_{k+1} \mathbf{C}_k+1) \check{\boldsymbol{\Sigma}}_{xx, k+1}
$$

---

Drone state estimators @ LSY. Contains model free (smoothing) and model based (EKF, UKF) state estimators for drones.

[![Python Version]][Python Version URL] [![Ruff Check]][Ruff Check URL] [![Tests]][Tests URL]

[Python Version]: https://img.shields.io/badge/python-3.11+-blue.svg
[Python Version URL]: https://www.python.org

[Ruff Check]: https://github.com/utiasDSL/drone-estimators/actions/workflows/ruff.yml/badge.svg?style=flat-square
[Ruff Check URL]: https://github.com/utiasDSL/drone-estimators/actions/workflows/ruff.yml

[Tests]: https://github.com/utiasDSL/drone-estimators/actions/workflows/testing.yml/badge.svg
[Tests URL]: https://github.com/utiasDSL/drone-estimators/actions/workflows/testing.yml

## Installation
Clone repository:

`git clone git@github.com:utiasDSL/drone-estimators.git`

If you already have an environment to install the estimators in, use:

`pip install -e .`

Otherwise, you can first start a pixi environment and then install the package with 

`pixi shell -e jazzy`

`pip install -e .`

## Usage
Either use the estimators directly:

`from drone_estimators.estimator import KalmanFilter`

or run the ROS2 node with:

`python drone_estimators/ros_nodes/ros2_node.py`

For the latter, you need to add all drones you want to estimate to the `estimators.toml` file, or create your own file and call

`python drone_estimators/ros_nodes/ros2_node.py --settings <your_estimators.toml>`