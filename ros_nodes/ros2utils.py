from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from geometry_msgs.msg import TransformStamped, TwistStamped, Vector3, WrenchStamped

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


def create_transform(
    header: Header, pos: NDArray[np.floating], quat: NDArray[np.floating]
) -> TransformStamped:
    transform = TransformStamped()
    transform.header = header
    # TODO child frame id? should stay the same??
    transform.transform.translation.x = pos[0]
    transform.transform.translation.y = pos[1]
    transform.transform.translation.z = pos[2]
    transform.transform.rotation.x = quat[0]
    transform.transform.rotation.y = quat[1]
    transform.transform.rotation.z = quat[2]
    transform.transform.rotation.w = quat[3]

    return transform


def create_twist(
    header: Header, vel: NDArray[np.floating], angvel: NDArray[np.floating]
) -> TransformStamped:
    twist = TwistStamped()
    twist.header = header
    twist.twist.linear.x = vel[0]
    twist.twist.linear.y = vel[1]
    twist.twist.linear.z = vel[2]
    twist.twist.angular.x = angvel[0]
    twist.twist.angular.y = angvel[1]
    twist.twist.angular.z = angvel[2]

    return twist


def create_wrench(
    header: Header, force: NDArray[np.floating], torque: NDArray[np.floating]
) -> TransformStamped:
    wrench = WrenchStamped()
    wrench.header = header
    if force is not None:
        wrench.wrench.force.x = force[0]
        wrench.wrench.force.y = force[1]
        wrench.wrench.force.z = force[2]
    if torque is not None:
        wrench.wrench.torque.x = force[0]
        wrench.wrench.torque.y = force[1]
        wrench.wrench.torque.z = force[2]

    return wrench
