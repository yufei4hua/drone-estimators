from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import rclpy
from geometry_msgs.msg import (
    TransformStamped,
    TwistStamped,
    Vector3,
    WrenchStamped,
)  # Message types: https://docs.ros2.org/foxy/api/geometry_msgs/index-msg.html
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from lsy_estimators.estimator import KalmanFilter
from ros_nodes.ros2utils import (
    create_transform,
    create_twist,
    create_wrench,
    find_transform,
    header2sec,
    tf2array,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from std_msgs.msg import Header

    from lsy_estimators.datacls import UKFData


class EstimatorNode(Node):
    """This class starts an estimator and the necessary subscribers and publishers.

    Note: Since all results are published individually, one might want to
    synchronize on the receiving end. For that see
    https://docs.ros.org/en/rolling/p/flex_sync/
    https://github.com/ros2/message_filters

    In the future, a custom message type might be advantageous. However,
    to keep it simple, we use standard message types.
    """

    def __init__(self, drone: str):
        super().__init__(f"Estimator_{drone}")  # Can put text into init to give it a name
        self.lock = threading.Lock()
        self.drone = drone
        sec, _ = self.get_clock().now().seconds_nanoseconds()
        self.time_stamp_last = float(sec)
        self.perf_timings = deque(maxlen=5000)

        # TODO add additional information from Node call
        self.estimator = KalmanFilter(
            dt=1.0 / 200,
            estimate_forces_motor=True,
            estimate_forces_dist=False,
            estimate_torques_dist=False,
        )

        # A better implementation would be:
        # https://docs.ros.org/en/foxy/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html
        # But for now this works and we don't care about a fixed publishing frequency
        self.subscriber_frames = self.create_subscription(TFMessage, "/tf", self.estimate_state, 2)

        self.subscriber_control = self.create_subscription(
            TFMessage, f"/control_{drone}", self.update_control, 2
        )

        self.publisher_transform = self.create_publisher(
            TransformStamped, f"/estimated_state_transform_{drone}", 2
        )  # pos, quat
        self.publisher_twist = self.create_publisher(
            TwistStamped, f"/estimated_state_twist_{drone}", 2
        )  # vel, angvel
        self.publisher_forces = self.create_publisher(
            Float64MultiArray, f"/estimated_state_forces_{drone}", 2
        )  # f_motors
        self.publisher_wrench = self.create_publisher(
            WrenchStamped, f"/estimated_state_wrench_{drone}", 2
        )  # f_dis, t_dis

    def estimate_state(self, msg: TFMessage):
        """Estimates the full drone state based on the new measurements."""
        if not self.lock.acquire(blocking=False):
            self.get_logger().warning(
                "New measurements before finishing estimation. Can't keep up..."
            )
            return

        try:
            tf = find_transform(msg.transforms, self.drone)

            if tf is None:
                self.get_logger().warning(
                    f"Drone {self.drone} could not have been found. Occluded?"
                )
            else:
                header, pos, quat = tf2array(tf)
                time_stamp = header2sec(header)
                dt = time_stamp - self.time_stamp_last

                # self.get_logger().info(
                #     f"New Measurement for {self.drone}: time={time_stamp}, pos={pos}, quat={quat}"
                # )

                # time_stamp = self._header2sec(header)
                # time_stamp = header.stamp.seconds_nanoseconds()

                # if dt > 1 => reset estimator

                if dt > 1e-3:
                    # self.get_logger().info(f"dt={dt}")
                    self.time_stamp_last = time_stamp
                    t1 = time.perf_counter()
                    estimated_state = self.estimator.step(np.concat((pos, quat)), dt)
                    t2 = time.perf_counter()
                    self.perf_timings.append(t2 - t1)
                    self.get_logger().info(
                        f"Step time avg = {(np.mean(self.perf_timings)) * 1000:.3f}ms",
                        throttle_duration_sec=1.0,
                    )
                    self.publish_state(header, estimated_state)
        finally:
            self.lock.release()  # Ensure lock is released

    def update_control(self, control):  # TODO add type, TODO get control for current drone only
        """TODO."""
        ...
        # roll = control.cmd_roll
        # pitch = control.cmd_pitch
        # yaw = control.cmd_pitch
        # pwm_thrust = control.cmd_thrust

        # thrust = pwm2thrust(pwm_thrust)

        # self.estimator.set_input(np.array([thrust, roll, pitch, yaw]))

    def publish_state(self, header: Header, state: UKFData):  # TODO state type
        """TODO."""
        # TODO time this

        # self.get_logger().info(f"Published for {self.drone}: {state}")

        transform = create_transform(header, state.pos, state.quat)
        self.publisher_transform.publish(transform)

        twist = create_twist(header, state.vel, state.angvel)
        self.publisher_twist.publish(twist)

        forces = Float64MultiArray(data=state.forces_motor)
        # This type doesn't have a stamp!
        self.publisher_forces.publish(forces)

        wrench = create_wrench(header, state.forces_dist, state.torques_dist)
        self.publisher_wrench.publish(wrench)


def launch_estimators(drones: list):
    """TODO."""
    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor()

    # create one node for each drone and add it to the executor
    node = EstimatorNode(drones[0])
    print(f"[ESTIMATOR]: Added estimator for {drones[0]}")
    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-drones", nargs="+", help="List of estimated drones", required=True)
    # Format: python ros2launch.py -drones "cf01" "cf02" ...
    args = parser.parse_args()

    launch_estimators(args.drones)
