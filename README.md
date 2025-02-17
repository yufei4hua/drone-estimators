# estimators
Drone state estimators @ LSY


## Installation
Clone repository

`pip install -e .`

## Usage
Either use the estimators directly:

`from lsy_estimators.estimator import KalmanFilter`

or run the ROS2 node with:

`python ros_nodes/ros2node.py --drones <DRONE_NAME1> <DRONE_NAME2> ...`

Drone names for the Crazyflies are cfXX, e.g. cf6 or cf52, as in Vicon.