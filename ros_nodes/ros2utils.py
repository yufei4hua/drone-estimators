from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped, Vector3, WrenchStamped
from std_msgs.msg import Float64MultiArray

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from std_msgs.msg import Header


def find_transform(transforms: list[TransformStamped], child_frame_id: str) -> TransformStamped:
    """Finds the transform of a given child frame."""
    return next((tf for tf in transforms if tf.child_frame_id == child_frame_id), None)


def header2sec(header: Header):
    return float(header.stamp.sec + header.stamp.nanosec * 1e-9)


def tf2array(tf: TransformStamped) -> tuple[Header, NDArray[np.floating], NDArray[np.floating]]:
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


def create_pose(
    header: Header, drone: str, pos: NDArray[np.floating], quat: NDArray[np.floating]
) -> TransformStamped:
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


def create_twist(
    header: Header, drone: str, vel: NDArray[np.floating], angvel: NDArray[np.floating]
) -> TwistStamped:
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


def create_array(header: Header, drone: str, data: NDArray | None) -> Float64MultiArray:
    if data is not None:
        array = Float64MultiArray()
        array.data = list(data)
        return array
    else:
        array = Float64MultiArray()
        array.data = [0.0, 0.0, 0.0, 0.0]
        return array


def create_wrench(
    header: Header,
    drone: str,
    force: NDArray[np.floating] | None,
    torque: NDArray[np.floating] | None,
) -> TransformStamped:
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
