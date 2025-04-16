"""This file contains some util functions for ROS 2 nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped, WrenchStamped
from std_msgs.msg import Float64MultiArray

if TYPE_CHECKING:
    from collections import defaultdict

    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from rclpy.time import Time
    from std_msgs.msg import Header
    from torch import Tensor

    from lsy_estimators.structs import UKFData

    Array = NDArray | JaxArray | Tensor


def find_transform(
    transforms: list[TransformStamped], child_frame_id: str
) -> TransformStamped | None:
    """Finds the transform of a given child frame."""
    return next((tf for tf in transforms if tf.child_frame_id == child_frame_id), None)


def header2sec(header: Header) -> float:
    """Extracts the current time in seconds from a message header."""
    return float(header.stamp.sec + header.stamp.nanosec * 1e-9)


def tf2array(tf: TransformStamped) -> tuple[Header, Array, Array]:
    """Converts a Transform into numpy arrays."""
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

    return tf.header, pos, quat


def time2sec_nsec(t: float) -> tuple[int, int]:
    sec = int(t)
    nsec = int(1e9 * (t - sec))
    return sec, nsec


def create_pose(time_stamp: float, drone: str, pos: Array, quat: Array) -> PoseStamped:
    """Creates a PoseStamped from position and orientation."""
    pose = PoseStamped()
    sec, nsec = time2sec_nsec(time_stamp)
    pose.header.stamp.sec = sec
    pose.header.stamp.nanosec = nsec
    pose.header.frame_id = "world"
    pose.pose.position.x = pos[0]
    pose.pose.position.y = pos[1]
    pose.pose.position.z = pos[2]
    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]

    return pose


def create_twist(time_stamp: float, drone: str, vel: Array, ang_vel: Array) -> TwistStamped:
    """Creates a TwistStamped based on velocity and angular velocity."""
    twist = TwistStamped()
    sec, nsec = time2sec_nsec(time_stamp)
    twist.header.stamp.sec = sec
    twist.header.stamp.nanosec = nsec
    twist.header.frame_id = drone
    twist.twist.linear.x = vel[0]
    twist.twist.linear.y = vel[1]
    twist.twist.linear.z = vel[2]
    twist.twist.angular.x = ang_vel[0]
    twist.twist.angular.y = ang_vel[1]
    twist.twist.angular.z = ang_vel[2]

    return twist


def create_array(time_stamp: float, drone: str, data: Array | None) -> Float64MultiArray:
    """Creates a generic Float64Array with the length of data.

    If data is None, an array zeros of length 4 is created.
    """
    if data is not None:
        array = Float64MultiArray()
        array.data = list(data)
        return array
    else:
        array = Float64MultiArray()
        array.data = [0.0, 0.0, 0.0, 0.0]
        return array


def create_wrench(
    time_stamp: float, drone: str, force: Array | None, torque: Array | None
) -> WrenchStamped:
    """Creates a WrenchStamped based on force and torque.

    If force or torque is None, it gets set to zeros.
    """
    wrench = WrenchStamped()
    sec, nsec = time2sec_nsec(time_stamp)
    wrench.header.stamp.sec = sec
    wrench.header.stamp.nanosec = nsec
    wrench.header.frame_id = drone
    if force is not None:
        wrench.wrench.force.x = force[0]
        wrench.wrench.force.y = force[1]
        wrench.wrench.force.z = force[2]
    else:
        wrench.wrench.force.x = 0.0
        wrench.wrench.force.y = 0.0
        wrench.wrench.force.z = 0.0
    if torque is not None:
        wrench.wrench.torque.x = 0.0
        wrench.wrench.torque.y = 0.0
        wrench.wrench.torque.z = 0.0

    return wrench


def append_state(data: defaultdict[str, list], time: float, state: UKFData):
    """Appends each states data to the corresponding of list in the dictionary."""
    data["time"].append(time)
    data["pos"].append(state.pos)
    data["quat"].append(state.quat)
    data["vel"].append(state.vel)
    data["ang_vel"].append(state.ang_vel)
    data["covariance"].append(state.covariance)
    if state.forces_motor is not None:
        data["forces_motor"].append(state.forces_motor)
    else:
        data["forces_motor"] = []
    if state.forces_dist is not None:
        data["forces_dist"].append(state.forces_dist)
    else:
        data["forces_dist"] = []
    if state.forces_dist is not None:
        data["torques_dist"].append(state.torques_dist)
    else:
        data["torques_dist"] = []


def append_measurement(
    data: defaultdict[str, list], time: float, pos: Array, quat: Array, command: Array
):
    """Appends each measurment data to the corresponding of list in the dictionary."""
    data["time"].append(time)
    data["pos"].append(pos)
    data["quat"].append(quat)
    data["command"].append(command)
