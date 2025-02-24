"""This file contains the estimator node for ROS 2.

Essentially this file provides a wrapper for the estimators to be used with ROS data. The estimates get published to ROS as well.

TODO Subscribed and published topics...
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import signal
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rclpy
import toml

# Message types: https://docs.ros2.org/foxy/api/geometry_msgs/index-msg.html
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from munch import Munch, munchify
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from lsy_estimators.estimator import KalmanFilter
from lsy_estimators.estimator_legacy import StateEstimator
from ros_nodes.ros2utils import (
    append_measurement,
    append_state,
    create_array,
    create_pose,
    create_twist,
    create_wrench,
    find_transform,
    header2sec,
    tf2array,
)

if TYPE_CHECKING:
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

    def __init__(self, settings: Munch):
        """TODO."""
        super().__init__(f"Estimator_{settings.drone_name}")
        self.input_needed = False
        self.initial_observation = None
        self.settings = settings
        sec, _ = self.get_clock().now().seconds_nanoseconds()
        self.time_stamp_last_measurement = sec
        self.time_stamp_last_command = sec
        self.perf_timings = deque(maxlen=5000)

        # This is for storing the data (DEBUG_SAVE_DATA)
        self.data_meas = defaultdict(list)  # {"time": [], "pos": [], "quat": [], "command": []}
        self.data_est = defaultdict(list)

        match self.settings.estimator_type:
            case "legacy":
                self.estimator = StateEstimator((0.0001, 0.007, 0.09, 0.005, 0.07))
                if (
                    settings.estimate_forces_motor
                    or settings.estimate_forces_dist
                    or settings.estimate_torques_dist
                ):
                    self.get_logger().warning(
                        "Legacy estimator does not support force or torque estimation!"
                    )
            case "ukf":
                self.input_needed = True
                self.estimator = KalmanFilter(
                    dt=1.0 / 200,
                    model=settings.dynamics_model,  # mellinger_rpyt, fitted_DI_rpy
                    estimate_forces_motor=settings.estimate_forces_motor,
                    estimate_forces_dist=settings.estimate_forces_dist,
                    estimate_torques_dist=settings.estimate_torques_dist,
                )
            case _:
                raise NotImplementedError(
                    f"Estimator type {self.settings.estimator_type} not implemented."
                )

        # A better implementation would be:
        # https://docs.ros.org/en/foxy/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html
        # But for now this works and we don't care about a fixed publishing frequency
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.subscriber_frames = self.create_subscription(
            TFMessage, "/tf", self.estimate_state, qos_profile
        )

        self.subscriber_control = self.create_subscription(
            Float64MultiArray,
            f"/command_{self.settings.drone_name}",
            self.update_control,
            qos_profile,
        )

        # pos, quat
        self.publisher_pose = self.create_publisher(
            PoseStamped, f"/estimated_state_pose_{self.settings.drone_name}", qos_profile
        )
        # vel, angvel
        self.publisher_twist = self.create_publisher(
            TwistStamped, f"/estimated_state_twist_{self.settings.drone_name}", qos_profile
        )
        # f_motors
        self.publisher_forces = self.create_publisher(
            Float64MultiArray, f"/estimated_state_forces_{self.settings.drone_name}", qos_profile
        )
        # f_dis, t_dis
        self.publisher_wrench = self.create_publisher(
            WrenchStamped, f"/estimated_state_wrench_{self.settings.drone_name}", qos_profile
        )

        self.get_logger().info(f"Started estimator (process {os.getpid()})")

    def estimate_state(self, msg: TFMessage):
        """Estimates the full drone state based on the new measurements."""
        tf = find_transform(msg.transforms, self.settings.drone_name)
        if tf is None:
            self.get_logger().warning(
                f"Drone {self.settings.drone_name} could not have been found. Occluded?",
                throttle_duration_sec=0.5,
            )
        else:
            header, pos_meas, quat_meas = tf2array(tf)
            time_stamp = header2sec(header)
            dt = time_stamp - self.time_stamp_last_measurement

            if self.initial_observation is None:
                self.initial_observation = (pos_meas, quat_meas)
                self.estimator.set_state(pos_meas, quat_meas)

            # self.get_logger().info(
            #     f"New Measurement for {self.settings.drone_name}: time={time_stamp}, pos_meas={pos_meas}, quat_meas={quat_meas}"
            # )

            # rot = R.from_quat(quat_meas)
            # print(f"rpy_meas={rot.as_euler('xyz', degrees=True)}")

            if dt < 0:  # This should only be executed when playing rosbags
                self.get_logger().warning("dt < 0s! Assuming rosbag is played. Setting time to now")
                self.time_stamp_last_measurement = time_stamp
                self.time_stamp_last_command = time_stamp
            elif dt > 1e-4:  # accepting the new data point
                self.time_stamp_last_measurement = time_stamp
                t1 = time.perf_counter()
                estimated_state = self.estimator.step(pos_meas, quat_meas, dt)
                self.publish_state(header, estimated_state)
                t2 = time.perf_counter()
                perf_time = t2 - t1

                if dt > 9e-3:
                    self.get_logger().warning(
                        f"Can't keept up... Time since last estimation: {dt * 1e3:.2f}ms.",
                        throttle_duration_sec=0.1,
                    )

                # AFTER publishing, we have time to store and print timings
                if DEBUG_TIMINGS:
                    self.perf_timings.append(perf_time)
                    t_avg = np.mean(self.perf_timings) * 1000
                    t_max = np.max(self.perf_timings) * 1000
                    t_min = np.min(self.perf_timings) * 1000
                    self.get_logger().info(
                        f"t_avg={t_avg:.3f}ms, t_min={t_min:.3f}ms, t_max={t_max:.3f}ms, datapoints {len(self.perf_timings)}",
                        throttle_duration_sec=2.0,
                    )
                # AFTER publishing, we have time to store the data
                if DEBUG_SAVE_DATA:
                    if not self.input_needed:
                        append_measurement(
                            self.data_meas, time_stamp, pos_meas, quat_meas, [0.0, 0.0, 0.0, 0.0]
                        )
                    else:
                        append_measurement(
                            self.data_meas, time_stamp, pos_meas, quat_meas, self.estimator.data.u
                        )
                    append_state(self.data_est, time_stamp, estimated_state)
            else:
                self.get_logger().info(
                    f"Received too high frequency measurements (dt = {dt * 1000:.1f}ms). Skipping data point..."
                )

            # print(f"time_stamp={time_stamp}, time_step_command={self.time_stamp_last_command}")
            # Before finishing, we should also check how recent the control inputs were
            dt_cmd = time_stamp - self.time_stamp_last_command
            if dt_cmd > 1:
                # TODO Move this into Kalman Filter
                # TODO Change varQ_forces_motor based on how old the estimate is
                # TODO change process and input noise depending on measurement
                if self.input_needed:
                    self.get_logger().warning(
                        f"Haven't received a new command in {dt_cmd:.0f}s. Setting it to zero.",
                        throttle_duration_sec=5.0,
                    )
                    self.estimator.set_input(np.zeros(4))

    def update_control(self, control: Float64MultiArray):
        """TODO."""
        # Storing the time of the current command
        # Note: Not using now() under the assumption that the measurements are very frequent (200Hz)
        # This is to allow us to also play rosbags without breaking functionality
        self.time_stamp_last_command = self.time_stamp_last_measurement

        # The command is as it is sent to the drone, meaning for attitude interface:
        # roll (deg), pitch (deg), yaw (deg), thrust (PWM)
        rpyt = np.array(control.data)

        # WARNING: Remove the following lines later.
        # This is only necessary because data was published wrongly (PYTR) for the current rosbags
        rpyt = np.roll(rpyt, 1)

        rpyt[..., :-1] = rpyt[..., :-1] * np.pi / 180  # to rad

        # self.get_logger().info(f"set input to {rpyt}")

        self.estimator.set_input(rpyt)

    def publish_state(self, header: Header, state: UKFData):  # TODO state type
        """TODO."""
        # TODO time this

        # self.get_logger().info(f"Published for {self.settings.drone_name}: {state}", throttle_duration_sec=0.5)
        # self.get_logger().info(
        #     f"Published for {self.settings.drone_name}: {state.pos}, {state.quat}, {state.vel}, {state.angvel}"
        # )

        transform = create_pose(header, self.settings.drone_name, state.pos, state.quat)
        self.publisher_pose.publish(transform)

        twist = create_twist(header, self.settings.drone_name, state.vel, state.angvel)
        self.publisher_twist.publish(twist)

        forces = create_array(header, self.settings.drone_name, state.forces_motor)
        # This type doesn't have a stamp!
        self.publisher_forces.publish(forces)

        wrench = create_wrench(
            header, self.settings.drone_name, state.forces_dist, state.torques_dist
        )
        self.publisher_wrench.publish(wrench)

    def shutdown(self, _, __):
        """TODO."""
        self.get_logger().info("Terminating")
        if DEBUG_SAVE_DATA:
            self.get_logger().info("Saving data...")
            filename = f"data_{self.settings.drone_name}_"
            info = f"{self.settings.estimator_type}"
            if self.settings.estimator_type != "legacy":
                info = info + f"_{self.settings.dynamics_model}"
            with open(filename + info + ".pkl", "wb") as f:
                pickle.dump(self.data_est, f)
            with open(filename + "measurement" + ".pkl", "wb") as f:
                pickle.dump(self.data_meas, f)
        self.subscriber_frames.destroy()
        self.subscriber_control.destroy()
        time.sleep(0.1)
        self.publisher_pose.destroy()
        self.publisher_twist.destroy()
        self.publisher_forces.destroy()
        self.publisher_wrench.destroy()
        # time.sleep(0.1)
        self.destroy_node()


def launch_estimators(estimators: dict):
    """TODO."""
    rclpy.init()
    processes = []
    seen_drone_names = []
    try:
        for k, v in estimators.items():
            if k.startswith("estimator"):
                settings = v
                name = settings.drone_name
                if name in seen_drone_names:
                    print(f"[ESTIMATOR]: Estimator for {name} already existing. Check settings")
                else:
                    seen_drone_names.append(name)
                    stop_event = mp.Event()
                    # not sure if daemon should be True or False
                    p = mp.Process(target=launch_node, args=(settings, stop_event), daemon=True)
                    processes.append((p, stop_event))
                    p.start()

        while True:
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Terminating nodes...")
                break

    finally:
        for _, stop_event in processes:
            stop_event.set()

        for process, _ in processes:
            process.join(timeout=2)

        for process, _ in processes:
            if process.is_alive():
                print(f"Force terminating process {process.pid}")
                process.terminate()
                process.join()

        if rclpy.ok():
            rclpy.shutdown()
        print("All nodes terminated.")


def launch_node(settings: Munch, stop_event: threading.Event):
    """TODO."""
    node = EstimatorNode(settings)

    signal.signal(signal.SIGINT, node.shutdown)

    try:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        pass

        # print(f"destroying node {drone}")
        # node.destroy_node()
    time.sleep(0.5)
    rclpy.shutdown()


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=3)  # TODO remove

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings", default="ros_nodes/estimators.toml", help="Path to Settings file"
    )
    args = parser.parse_args()
    path = args.settings

    path = Path(Path(__file__).parents[1] / path)
    with open(path, "r") as f:
        estimators = munchify(toml.load(f))

    # Global debug settings. Defaulting to False
    debug_settings = estimators.get("debug", {})
    DEBUG_TIMINGS = debug_settings.get("timings", False)
    DEBUG_SAVE_DATA = debug_settings.get("save_data", False)

    launch_estimators(estimators)

    # only launch first one. For testing only... TODO remove
    # settings = estimators["estimator1"]
    # rclpy.init()
    # node = EstimatorNode(settings)
    # rclpy.spin(node)
    # rclpy.shutdown()
