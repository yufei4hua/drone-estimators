from __future__ import annotations

import argparse
import sys
import threading
from typing import TYPE_CHECKING

import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from lsy_estimators.estimator import KalmanFilter

if TYPE_CHECKING:
    from geometry_msgs.msg import TransformStamped
    from numpy.typing import NDArray


class EstimatorNode(Node):
    """This class starts an estimator and the necessary subscribers and publishers."""

    def __init__(self, drone: str):
        super().__init__(f"Estimator_{drone}")  # Can put text into init to give it a name
        self.drone = drone
        sec, _ = self.get_clock().now().seconds_nanoseconds()
        self.stamp_last = float(sec)

        # TODO add additional information from Node call
        self.estimator = KalmanFilter(dt=1.0 / 200)

        self.subscriber_frames = self.create_subscription(TFMessage, "/tf", self.estimate_state, 10)

        # self.subscriber_control = self.create_subscription(TFMessage, "/control", self.step, 10)

        # self.publisher_state = self.create_publisher(String, "topic", 10) # TODO

    def _tf2np(
        self, tf: TransformStamped
    ) -> tuple[np.floating, NDArray[np.floating], NDArray[np.floating]]:
        """Converts a Transform into numpy arrays."""
        stamp = float(tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9)

        pos = np.array(
            [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
        )

        quat = np.array(
            [
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w,
            ]
        )

        return stamp, pos, quat

    def _find_transform(self, transforms: list[TransformStamped], drone: str) -> TransformStamped:
        """Finds the transform of the given drone."""
        return next((tf for tf in transforms if tf.child_frame_id == drone), None)

    def estimate_state(self, msg: TFMessage):
        """Estimates the full drone state based on the new measurements."""
        tf = self._find_transform(msg.transforms, self.drone)

        if tf is None:
            self.get_logger().warning(f"Drone {self.drone} could not have been found. Occluded?")
        else:
            stamp, pos, quat = self._tf2np(tf)
            print(f"New Measurement for {self.drone}: time={stamp}, pos={pos}, quat={quat}")

            dt = stamp - self.stamp_last
            self.stamp_last = stamp

            if dt > 1e-6:
                print("Stepping estimator")
                self.estimator.step(np.concat((pos, quat)), dt)

    def update_control(self, control):  # TODO add type, TODO get control for current drone only
        """TODO."""
        roll = control.cmd_roll
        pitch = control.cmd_pitch
        yaw = control.cmd_pitch
        pwm_thrust = control.cmd_thrust

        thrust = pwm2thrust(pwm_thrust)

        self.estimator.set_input(np.array([thrust, roll, pitch, yaw]))

    def publish_state(self, state):  # TODO state type
        """TODO."""
        # self.publisher_state.publish(state) # TODO

        print(f"Would have published {state}")

        # for transform in msg.transforms:
        #     if transform.child_frame_id == "cf97":
        #         self.get_logger().info(f"Header: {transform.header}")
        #         self.get_logger().info(f"ID: {transform.child_frame_id}")
        #         self.get_logger().info(f"Transform: {transform.transform}")
        #         print(transform.transform.translation.x)


def launch_estimators(drones: list):
    """TODO."""
    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor()

    # create one node for each drone and add it to the executor
    for drone in args.drones:
        node = EstimatorNode(drone)
        executor.add_node(node)
        print(f"[ESTIMATOR]: Added estimator for {drone}")

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = node.create_rate(1)  # 1Hz

    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-drones", nargs="+", help="List of estimated drones", required=True)
    # Format: python ros2launch.py -drones "cf01" "cf02" ...
    args = parser.parse_args()

    launch_estimators(args.drones)
