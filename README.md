# estimators
Drone state estimators @ LSY


## Installation
Clone repository

`pip install -e .`

## Usage
Either use the estimators directly:

`from lsy_estimators.estimator import KalmanFilter`

or run the ROS2 node with:

`python lsy_estimators/ros_nodes/ros2_node.py`

For the latter, you need to add all drones you want to estimate to the `estimators.toml` file, or create your own file and call

<<<<<<< HEAD
`python ros_nodes/ros2node.py --settings ros_nodes/<your_estimators.toml>`
=======
`python lsy_estimators/ros_nodes/ros2_node.py --settings <your_estimators.toml>`
>>>>>>> dev
