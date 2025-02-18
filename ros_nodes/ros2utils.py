"""This file contains some util functions for ROS 2 nodes."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped, WrenchStamped
from std_msgs.msg import Float64MultiArray

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from std_msgs.msg import Header
    from torch import Tensor

    from lsy_estimators.datacls import UKFData

    Array = NDArray | JaxArray | Tensor


def find_transform(transforms: list[TransformStamped], child_frame_id: str) -> TransformStamped:
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


def create_pose(header: Header, drone: str, pos: Array, quat: Array) -> PoseStamped:
    """Creates a PoseStamped from position and orientation."""
    pose = PoseStamped()
    pose.header = header
    pose.header.frame_id = "world"
    pose.pose.position.x = pos[0]
    pose.pose.position.y = pos[1]
    pose.pose.position.z = pos[2]
    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]

    return pose


def create_twist(header: Header, drone: str, vel: Array, angvel: Array) -> TwistStamped:
    """Creates a TwistStamped based on velocity and angular velocity."""
    twist = TwistStamped()
    twist.header = header
    twist.header.frame_id = drone
    twist.twist.linear.x = vel[0]
    twist.twist.linear.y = vel[1]
    twist.twist.linear.z = vel[2]
    twist.twist.angular.x = angvel[0]
    twist.twist.angular.y = angvel[1]
    twist.twist.angular.z = angvel[2]

    return twist


def create_array(header: Header, drone: str, data: Array | None) -> Float64MultiArray:
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
    header: Header, drone: str, force: Array | None, torque: Array | None
) -> WrenchStamped:
    """Creates a WrenchStamped based on force and torque.

    If force or torque is None, it gets set to zeros.
    """
    wrench = WrenchStamped()
    wrench.header = header
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
    """TODO."""
    data["time"].append(time)
    data["pos"].append(state.pos)
    data["quat"].append(state.quat)
    data["vel"].append(state.vel)
    data["angvel"].append(state.angvel)
    if state.forces_motor is not None:
        data["forces_motor"].append(state.forces_motor)
    if state.forces_motor is not None:
        data["forces_dist"].append(state.forces_dist)
    if state.forces_motor is not None:
        data["torques_dist"].append(state.torques_dist)


def append_measurement(
    data: defaultdict[str, list], time: float, pos: Array, quat: Array, command: Array
):
    """TODO."""
    data["time"].append(time)
    data["pos"].append(pos)
    data["quat"].append(quat)
    data["command"].append(command)
