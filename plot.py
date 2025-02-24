from __future__ import annotations

import os
import pickle
import time
from collections import deque
from typing import TYPE_CHECKING

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray


def setaxs1(axs1, t_start, t_end):
    axs1[0, 0].set_title("Position x [m]")
    axs1[1, 0].set_title("Position y [m]")
    axs1[2, 0].set_title("Position z [m]")
    # axs1[0, 0].set_ylim(-1.5, 1.5)
    # axs1[1, 0].set_ylim(-1.5, 1.5)
    # axs1[2, 0].set_ylim(-0.05, 1.5)
    axs1[0, 1].set_title("Position Error x [m]")
    axs1[1, 1].set_title("Position Error y [m]")
    axs1[2, 1].set_title("Position Error z [m]")
    err_pos = 1e-3
    axs1[0, 1].set_ylim(-err_pos, err_pos)
    axs1[1, 1].set_ylim(-err_pos, err_pos)
    axs1[2, 1].set_ylim(-err_pos, err_pos)
    axs1[0, 2].set_title("Velocity x [m/s]")
    axs1[1, 2].set_title("Velocity y [m/s]")
    axs1[2, 2].set_title("Velocity z [m/s]")
    axs1[0, 2].set_ylim(-1.5, 1.5)
    axs1[1, 2].set_ylim(-1.5, 1.5)
    axs1[2, 2].set_ylim(-1.5, 1.5)
    axs1[0, 3].set_title("Velocity Error x [m/s]")
    axs1[1, 3].set_title("Velocity Error y [m/s]")
    axs1[2, 3].set_title("Velocity Error z [m/s]")
    err_vel = 1.5e-1
    axs1[0, 3].set_ylim(-err_vel, err_vel)
    axs1[1, 3].set_ylim(-err_vel, err_vel)
    axs1[2, 3].set_ylim(-err_vel, err_vel)

    # Setting legend and grid
    for ax in axs1.flat:
        ax.legend()
        ax.grid()
        ax.set_xlim(t_start, t_end)


def setaxs2(axs2, t_start, t_end):
    axs2[0, 0].set_title("Euler Angle roll [degree]")
    axs2[1, 0].set_title("Euler Angle pitch [degree]")
    axs2[2, 0].set_title("Euler Angle yaw [degree]")
    # axs2[0, 0].set_ylim(-0.3, 0.3)
    # axs2[1, 0].set_ylim(-0.3, 0.3)
    # axs2[2, 0].set_ylim(-0.3, 0.3)
    axs2[0, 1].set_title("Euler Angle Error roll [degree]")
    axs2[1, 1].set_title("Euler Angle Error pitch [degree]")
    axs2[2, 1].set_title("Euler Angle Error yaw [degree]")
    err_rpy = 1e-0
    axs2[0, 1].set_ylim(-err_rpy, err_rpy)
    axs2[1, 1].set_ylim(-err_rpy, err_rpy)
    axs2[2, 1].set_ylim(-err_rpy, err_rpy)
    axs2[0, 2].set_title("Angular velocity x [rad/s]")
    axs2[1, 2].set_title("Angular velocity y [rad/s]")
    axs2[2, 2].set_title("Angular velocity z [rad/s]")
    # axs2[0, 2].set_ylim(-2, 2)
    # axs2[1, 2].set_ylim(-2, 2)
    # axs2[2, 2].set_ylim(-2, 2)
    axs2[0, 3].set_title("Angular velocity error x [rad/s]")
    axs2[1, 3].set_title("Angular velocity error y [rad/s]")
    axs2[2, 3].set_title("Angular velocity error z[rad/s]")
    err_angvel = 5e-1
    axs2[0, 3].set_ylim(-err_angvel, err_angvel)
    axs2[1, 3].set_ylim(-err_angvel, err_angvel)
    axs2[2, 3].set_ylim(-err_angvel, err_angvel)

    for ax in axs2.flat:
        ax.legend()
        ax.grid()
        ax.set_xlim(t_start, t_end)


def setaxs3(axs3, t_start, t_end):
    axs3[0, 0].set_title("Disturbance Force x [N]")
    axs3[1, 0].set_title("Disturbance Force y [N]")
    axs3[2, 0].set_title("Disturbance Force z [N]")
    axs3[0, 0].set_ylim(-0.1, 0.1)
    axs3[1, 0].set_ylim(-0.1, 0.1)
    axs3[2, 0].set_ylim(-0.1, 0.1)
    axs3[0, 1].set_title("Force Error [N]")
    axs3[1, 1].set_title("Force Error [N]")
    axs3[2, 1].set_title("Force Error [N]")
    axs3[0, 1].set_title("Disturbance Torque x [Nm]")
    axs3[1, 1].set_title("Disturbance Torque y [Nm]")
    axs3[2, 1].set_title("Disturbance Torque z [Nm]")
    axs3[0, 1].set_ylim(-0.002, 0.002)
    axs3[1, 1].set_ylim(-0.002, 0.002)
    axs3[2, 1].set_ylim(-0.002, 0.002)
    axs3[0, 3].set_title("Torque Error [Nm]")
    axs3[1, 3].set_title("Torque Error [Nm]")
    axs3[2, 3].set_title("Torque Error [Nm]")

    for ax in axs3.flat:
        ax.legend()
        ax.grid()
        ax.set_xlim(t_start, t_end)


def plotaxs1(
    axs1,
    data,
    label="unkown",
    linestyle="-",
    color="tab:blue",
    alpha=0.0,
    t_vertical=None,
    order="",
    weight=0,
    t_start=0,
    t_end=10,
):
    ### Pos and vel
    axs1[0, 0].plot(data["time"], data["pos"][:, 0], linestyle, label=label, color=color)
    # axs1[0, 0].fill_between(
    #     data["time_est"],
    #     -3 * data["P_post"][:, 0] + data["pos"][:, 0],
    #     3 * data["P_post"][:, 0] + data["pos"][:, 0],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )

    axs1[1, 0].plot(data["time"], data["pos"][:, 1], linestyle, label=label, color=color)

    # axs1[1, 0].fill_between(
    #     data["time_est"],
    #     -3 * data["P_post"][:, 1] + data["pos_est"][:, 1],
    #     3 * data["P_post"][:, 1] + data["pos_est"][:, 1],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )  # plotting 3 std

    axs1[2, 0].plot(data["time"], data["pos"][:, 2], linestyle, label=label, color=color)

    # axs1[2, 0].fill_between(
    #     data["time_est"],
    #     -3 * data["P_post"][:, 2] + data["pos_est"][:, 2],
    #     3 * data["P_post"][:, 2] + data["pos_est"][:, 2],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )  # plotting 3 std

    if len(data["pos_error"]) > 0:
        axs1[0, 1].plot(data["time"], data["pos_error"][:, 0], label=label, color=color)
        axs1[1, 1].plot(data["time"], data["pos_error"][:, 1], label=label, color=color)
        axs1[2, 1].plot(data["time"], data["pos_error"][:, 2], label=label, color=color)

    axs1[0, 2].plot(data["time"], data["vel"][:, 0], linestyle, label=label, color=color)

    # axs1[0, 2].fill_between(
    #     data["time_est"],
    #     -3 * data["P_post"][:, 6] + data["vel_est"][:, 0],
    #     3 * data["P_post"][:, 6] + data["vel_est"][:, 0],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )  # plotting 3 std

    axs1[1, 2].plot(data["time"], data["vel"][:, 1], linestyle, label=label, color=color)

    # axs1[1, 2].fill_between(
    #     data["time_est"],
    #     -3 * data["P_post"][:, 7] + data["vel_est"][:, 1],
    #     3 * data["P_post"][:, 7] + data["vel_est"][:, 1],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )  # plotting 3 std

    axs1[2, 2].plot(data["time"], data["vel"][:, 2], linestyle, label=label, color=color)

    # axs1[2, 2].fill_between(
    #     data["time_est"],
    #     -3 * data["P_post"][:, 8] + data["vel_est"][:, 2],
    #     3 * data["P_post"][:, 8] + data["vel_est"][:, 2],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )  # plotting 3 std

    if len(data["vel_error"]) > 0:
        axs1[0, 3].plot(data["time"], data["vel_error"][:, 0], label=label, color=color)
        axs1[1, 3].plot(data["time"], data["vel_error"][:, 1], label=label, color=color)
        axs1[2, 3].plot(data["time"], data["vel_error"][:, 2], label=label, color=color)


def plotaxs2(
    axs2,
    data,
    label="unkown",
    linestyle="-",
    color="tab:blue",
    alpha=0.0,
    t_vertical=None,
    order="",
    weight=0,
    t_start=0,
    t_end=10,
):
    ### rpy and rpy dot

    axs2[0, 0].plot(data["time"], data["rpy"][:, 0], linestyle, label=label, color=color)
    # axs2[0, 0].fill_between(
    #     data["time_est"],
    #     -3 * data["P_post"][:, 3] + data["euler_est"][:, 0],
    #     3 * data["P_post"][:, 3] + data["euler_est"][:, 0],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )  # plotting 3 std

    axs2[1, 0].plot(data["time"], data["rpy"][:, 1], linestyle, label=label, color=color)
    # axs2[1, 0].fill_between(
    #     data["time_est"],
    #     -3 * data["P_post"][:, 4] + data["euler_est"][:, 1],
    #     3 * data["P_post"][:, 4] + data["euler_est"][:, 1],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )  # plotting 3 std

    axs2[2, 0].plot(data["time"], data["rpy"][:, 2], linestyle, label=label, color=color)
    # axs2[2, 0].fill_between(
    #     data["time_est"],
    #     -3 * data["P_post"][:, 5] + data["euler_est"][:, 2],
    #     3 * data["P_post"][:, 5] + data["euler_est"][:, 2],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )  # plotting 3 std

    if len(data["rpy_error"]) > 0:
        axs2[0, 1].plot(data["time"], data["rpy_error"][:, 0], label=label, color=color)
        axs2[1, 1].plot(data["time"], data["rpy_error"][:, 1], label=label, color=color)
        axs2[2, 1].plot(data["time"], data["rpy_error"][:, 2], label=label, color=color)

    axs2[0, 2].plot(data["time"], data["angvel"][:, 0], linestyle, label=label, color=color)
    # axs2[0, 2].fill_between(
    #     data["time_est"],
    #     -3 * data["P_post"][:, 9] + data["euler_rate_est"][:, 0],
    #     3 * data["P_post"][:, 9] + data["euler_rate_est"][:, 0],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )  # plotting 3 std

    axs2[1, 2].plot(data["time"], data["angvel"][:, 1], linestyle, label=label, color=color)
    # axs2[1, 2].fill_between(
    #     data["time_est"],
    #     -3 * data["P_post"][:, 10] + data["euler_rate_est"][:, 1],
    #     3 * data["P_post"][:, 10] + data["euler_rate_est"][:, 1],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )  # plotting 3 std

    axs2[2, 2].plot(data["time"], data["angvel"][:, 2], linestyle, label=label, color=color)
    # axs2[2, 2].fill_between(
    #     data["time"],
    #     -3 * data["P_post"][:, 11] + data["euler_rate_est"][:, 2],
    #     3 * data["P_post"][:, 11] + data["euler_rate_est"][:, 2],
    #     alpha=alpha,
    #     linewidth=0,
    #     color="tab:orange",
    # )  # plotting 3 std

    if len(data["angvel_error"]) > 0:
        axs2[0, 3].plot(data["time"], data["angvel_error"][:, 0], label=label, color=color)
        axs2[1, 3].plot(data["time"], data["angvel_error"][:, 1], label=label, color=color)
        axs2[2, 3].plot(data["time"], data["angvel_error"][:, 1], label=label, color=color)


def plotaxs3(
    axs3,
    data,
    label="unkown",
    linestyle="-",
    color="tab:blue",
    alpha=0.0,
    t_vertical=None,
    order="",
    weight=0,
    t_start=0,
    t_end=10,
):
    ### force and torque
    if len(data["forces_dist"]) > 0:  # check if the posterior even contains the force
        axs3[0, 0].plot(data["time"], data["forces_dist"][:, 0], label=label, color=color)
        # axs3[0, 0].fill_between(
        #     data["time"],
        #     -3 * data["P_post"][:, 12] + data["force_est"][:, 0],
        #     3 * data["P_post"][:, 12] + data["force_est"][:, 0],
        #     alpha=alpha,
        #     linewidth=0,
        # )  # plotting 3 std
        # try:
        #     axs3[0, 0].vlines(
        #         x=[t_vertical[order.index("x")], t_vertical[order.index("x") + 1]],
        #         ymin=-0.1,
        #         ymax=0.1,
        #         colors="red",
        #         linestyles="--",
        #     )
        # except:
        #     ...  # We do not care if the index cant be found. Simply dont plot

        axs3[1, 0].plot(data["time"], data["forces_dist"][:, 1], label=label, color=color)
        # axs3[1, 0].fill_between(
        #     data["time"],
        #     -3 * data["P_post"][:, 13] + data["force_est"][:, 1],
        #     3 * data["P_post"][:, 13] + data["force_est"][:, 1],
        #     alpha=alpha,
        #     linewidth=0,
        # )  # plotting 3 std
        # try:
        #     axs3[1, 0].vlines(
        #         x=[t_vertical[order.index("y")], t_vertical[order.index("y") + 1]],
        #         ymin=-0.1,
        #         ymax=0.1,
        #         colors="red",
        #         linestyles="--",
        #     )
        # except:
        #     ...  # We do not care if the index cant be found. Simply dont plot

        axs3[2, 0].plot(data["time"], data["forces_dist"][:, 2], label=label, color=color)
        # axs3[2, 0].fill_between(
        #     data["time"],
        #     -3 * data["P_post"][:, 14] + data["force_est"][:, 2],
        #     3 * data["P_post"][:, 14] + data["force_est"][:, 2],
        #     alpha=alpha,
        #     linewidth=0,
        # )  # plotting 3 std
        # try:
        #     axs3[2, 0].vlines(
        #         x=[t_vertical[order.index("z")], t_vertical[order.index("z") + 1]],
        #         ymin=-0.1,
        #         ymax=0.1,
        #         colors="red",
        #         linestyles="--",
        #     )
        # except:
        #     ...  # We do not care if the index cant be found. Simply dont plot
        # if weight > 0.0:
        #     axs3[2, 0].hlines(
        #         y=-weight / 1000 * 9.81,
        #         xmin=t_start,
        #         xmax=t_start + 60,
        #         colors="green",
        #         linestyles="--",
        #     )

    if len(data["torques_dist"]) > 0:  # check if the posterior even contains the torque
        axs3[0, 1].plot(data["time"], data["torques_dist"][:, 0], label=label, color=color)
        # axs3[0, 1].fill_between(
        #     data["time"],
        #     -3 * data["P_post"][:, 15] + data["torque_est"][:, 0],
        #     3 * data["P_post"][:, 15] + data["torque_est"][:, 0],
        #     alpha=alpha,
        #     linewidth=1,
        # )  # plotting 3 std

        axs3[1, 1].plot(data["time"], data["torques_dist"][:, 1], label=label, color=color)
        # axs3[1, 1].fill_between(
        #     data["time"],
        #     -3 * data["P_post"][:, 16] + data["torque_est"][:, 1],
        #     3 * data["P_post"][:, 16] + data["torque_est"][:, 1],
        #     alpha=alpha,
        #     linewidth=1,
        # )  # plotting 3 std

        axs3[2, 1].plot(data["time"], data["torques_dist"][:, 2], label=label, color=color)
        # axs3[2, 1].fill_between(
        #     data["time"],
        #     -3 * data["P_post"][:, 17] + data["torque_est"][:, 2],
        #     3 * data["P_post"][:, 17] + data["torque_est"][:, 2],
        #     alpha=alpha,
        #     linewidth=1,
        # )  # plotting 3 std


def plotaxs3single(axs3, alpha, data, t_vertical=None, order="", weight=0, t_start=0, t_end=10):
    ### force and torque
    if data["P_post"].shape[1] >= 15:  # check if the posterior even contains the force
        axs3.set_title("Estimated Disturbance Force", fontsize=20)
        axs3.set_xlabel("Time [s]", fontsize=16)
        axs3.set_ylabel("Force [N]", fontsize=16)
        axs3.plot(data["time_est"], data["force_est"][:, 0], label="$F_x$")  # , color="red"
        # axs3.plot(data["time_est"], data["force_est_fxtdo"][:, 0], label='FxTDO')

        axs3.set_ylim(-0.05, 0.05)

        # axs3.set_title('Disturbance Force y [N]')
        # axs3.plot(data["time_est"], data["force_est"][:, 1], label='UKF')
        # axs3[1, 0].plot(data["time_est"], data["force_est_fxtdo"][:, 1], label='FxTDO')

        axs3.plot(data["time_est"], data["force_est"][:, 2], label="$F_z$")  # , color="blue"
        # axs3.plot(data["time_est"], data["force_est_fxtdo"][:, 2], label='FxTDO')

        # for ax in axs3.flat:
        axs3.legend(prop={"size": 16})
        axs3.grid()
        axs3.set_xlim(t_start, t_end)


def slice_data(data, t1, t2):
    """Slices the data between t1 and t2"""
    data_sliced = data.copy()

    # Slice time_meas
    start_idx = np.searchsorted(data["time_meas"], t1, side="left")
    end_idx = np.searchsorted(data["time_meas"], t2, side="right") - 1
    data_sliced["time_meas"] = data_sliced["time_meas"][start_idx:end_idx]
    data_sliced["pos_meas"] = data_sliced["pos_meas"][start_idx:end_idx]
    data_sliced["euler_meas"] = data_sliced["euler_meas"][start_idx:end_idx]
    data_sliced["vel_meas"] = data_sliced["vel_meas"][start_idx:end_idx]
    data_sliced["euler_rate_meas"] = data_sliced["euler_rate_meas"][start_idx:end_idx]

    data_sliced["pos_error"] = data_sliced["pos_error"][start_idx:end_idx]
    data_sliced["euler_error"] = data_sliced["euler_error"][start_idx:end_idx]
    data_sliced["vel_error"] = data_sliced["vel_error"][start_idx:end_idx]
    data_sliced["euler_rate_error"] = data_sliced["euler_rate_error"][start_idx:end_idx]

    data_sliced["time_est"] = data_sliced["time_est"][start_idx:end_idx]
    data_sliced["pos_est"] = data_sliced["time_meas"][start_idx:end_idx]
    data_sliced["euler_est"] = data_sliced["euler_est"][start_idx:end_idx]
    data_sliced["vel_est"] = data_sliced["vel_est"][start_idx:end_idx]
    data_sliced["euler_rate_est"] = data_sliced["euler_rate_est"][start_idx:end_idx]
    data_sliced["force_est"] = data_sliced["force_est"][start_idx:end_idx]
    data_sliced["torque_est"] = data_sliced["torque_est"][start_idx:end_idx]
    data_sliced["P_post"] = data_sliced["P_post"][start_idx:end_idx]
    data_sliced["cmd"] = data_sliced["cmd"][start_idx:end_idx]

    data_sliced["force_est_fxtdo"] = data_sliced["force_est_fxtdo"][start_idx:end_idx]

    return data_sliced


def plots(data_meas, estimator_types, estimator_datasets, animate=False, order="", weight=0):
    """Plot the measurement and estimator data.

    #Args:
        order: Order of the external forces applied in 10s intervals.
        weight: Extra weight in [g] added to the drone
    """
    alpha = 0.4  # for 3 std fill in between plots
    pad = 2.0
    figsize = (18, 12)
    # Initialize the plot with 3 rows and 4 columns
    fig1, axs1 = plt.subplots(3, 4, figsize=figsize)  # pos and vel
    fig1.tight_layout(pad=pad)
    fig2, axs2 = plt.subplots(3, 4, figsize=figsize)  # rpy and rpy dot
    fig2.tight_layout(pad=pad)
    fig3, axs3 = plt.subplots(3, 4, figsize=figsize)  # force and torque
    # fig3, axs3 = plt.subplots(3, 2, figsize=figsize) # force and torque
    # fig3, axs3 = plt.subplots(1, figsize=figsize)  # force and torque
    fig3.tight_layout(pad=pad)

    # axs3.tick_params(axis="both", which="major", labelsize=12)
    # axs3.tick_params(axis="both", which="minor", labelsize=10)

    # SMALL_SIZE = 8
    # MEDIUM_SIZE = 10
    # BIGGER_SIZE = 18

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=100)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ##################################################
    ### Data preprocessing
    ##################################################
    # Calculating measured vel and angvel from finite differences
    filter_length, filter_order = 51, 1  # length/200Hz = length in [s]
    dt_avg = np.mean(np.diff(data_meas["time"]))

    # pos_meas_filtered = savgol_filter(data_meas["pos"], filter_length, filter_order, axis=0)
    # data_meas["pos"] = pos_meas_filtered  # TODO remove?
    # data_meas["vel"] = np.gradient(pos_meas_filtered, data_meas["time"], axis=0)
    # data_meas["vel"] = savgol_filter(data_meas["vel"], filter_length, filter_order, axis=0)
    data_meas["vel"] = savgol_filter(data_meas["pos"], 51, 1, deriv=1, delta=dt_avg, axis=0)

    quat_meas_filtered = savgol_filter(data_meas["quat"], 7, 2, axis=0)
    data_meas["quat"] = quat_meas_filtered  # TODO remove?
    data_meas = quat2rpy(data_meas)

    rot = R.from_euler("xyz", data_meas["rpy"], degrees=True)
    rpy_dot = savgol_filter(data_meas["rpy"], 11, 1, deriv=1, delta=dt_avg, axis=0)
    data_meas["angvel"] = rpy_dot / 180 * np.pi
    # data_meas["angvel"] = rot.apply(rpy_dot / 180 * np.pi)

    # data_meas["angvel"] = quat2angvel(quat_meas_filtered, data_meas["time"])

    # estimator_datasets[0]["angvel"]

    # dquat = np.gradient(quat_meas_filtered, data_meas["time"], axis=0)
    # dquat_filtered = savgol_filter(data_meas["quat"], 7, 2, deriv=1, delta=dt_avg, axis=0)
    # data_meas["angvel"] = dquat2angvel(
    #     quat_meas_filtered, dquat_filtered, np.diff(data_meas["time"], prepend=1.0 / 200)
    # )
    # data_meas["angvel"] = savgol_filter(data_meas["angvel"], 7, 2, deriv=1, delta=dt_avg, axis=0)

    # Interpolating maybe? TODO

    # Error calculation of estimates to "ground truth" (=filtered measurements)
    for i in range(len(estimator_types)):
        data_est = estimator_datasets[i]
        estimator_times = data_est["time"]
        measurement_times = data_meas["time"]
        data_new = {}
        for k, v in data_est.items():
            if k != "time" and k != "forces_dist" and k != "torques_dist" and k != "forces_motor":
                # interpolate measurement to fit estimator data
                interpolation = interp1d(
                    measurement_times, data_meas[k], kind="linear", axis=0, fill_value="extrapolate"
                )
                # values2_interp = interp_func(time1)
                # interpolation = np.interp(estimator_times, measurement_times, data_meas[k], )
                data_new[f"{k}_error"] = interpolation(estimator_times) - v
        for k, v in data_new.items():
            data_est[k] = v

        pos = rmse(data_est["pos_error"])
        quat = rmse(data_est["quat_error"])
        vel = rmse(data_est["vel_error"])
        angvel = rmse(data_est["angvel_error"])
        print(f"{estimator_types[i]} RMSE: pos={pos}, quat={quat}, vel={vel}, angvel={angvel}")
        # print(f"estimator {estimator_types[i]} keys={data_est.keys()}")

    # Skipping datapoints for faster plotting performance
    # Note: This decreases high frequency effects
    step = 20

    ##################################################
    ### Plotting
    ##################################################
    colors = list(mcolors.TABLEAU_COLORS.values())
    # First, plot measurements
    plotaxs1(axs1, data_meas, label="meas", linestyle="--", color=colors[0])
    plotaxs2(axs2, data_meas, label="meas", linestyle="--", color=colors[0])
    # plotaxs3(axs3, data_meas, label="meas", linestyle="--", color="tab:blue")

    for i in range(len(estimator_types)):
        name = estimator_types[i]
        data = estimator_datasets[i]

        plotaxs1(axs1, data, label=name, linestyle="-", color=colors[i + 1])
        plotaxs2(axs2, data, label=name, linestyle="-", color=colors[i + 1])
        plotaxs3(axs3, data, label=name, linestyle="-", color=colors[i + 1])

    # TODO axis title, grid, legend etc
    setaxs1(axs1, data_meas["time"][0], data_meas["time"][-1])
    setaxs2(axs2, data_meas["time"][0], data_meas["time"][-1])
    setaxs3(axs3, data_meas["time"][0], data_meas["time"][-1])

    # plotaxs1(axs1, alpha, data, t_vertical=None, order="", weight=0, t_start=0, t_end=50)
    # plotaxs2(axs2, alpha, data, t_vertical=None, order="", weight=0, t_start=0, t_end=50)
    # plotaxs3single(axs3, alpha, data, t_vertical=None, order="", weight=0, t_start=0, t_end=50)

    # plt.rc('xtick', labelsize=50)
    # plt.rc('ytick', labelsize=50)

    if animate:
        FPS = 5  # Frames per second
        T = 10  # Amount of seconds displayed in the x axis
        time_divider = 5  # needed to speed up the recording later and make it look smooth
        ani_start = time.perf_counter() / time_divider

        # def clear():

        #     for ax in axs2.flat:
        #         ax.clear()
        #     for ax in axs3.flat:
        #         ax.clear()

        def getTimes():
            t = time.perf_counter() / time_divider - ani_start
            if t < T:
                t2 = t
                t1 = 0
            else:
                t2 = t
                t1 = t - T
            return t1, t2

        def update1(frame):
            t1, t2 = getTimes()

            for ax in axs3.flat:
                ax.set_xlim(t1, t2)

        def update2(frame):
            t1, t2 = getTimes()

            for ax in axs2.flat:
                ax.set_xlim(t1, t2)

        def update3(frame):
            t1, t2 = getTimes()

            # for ax in axs3.flat:
            #     ax.set_xlim(t1, t2)
            axs3.set_xlim(t1, t2)

        # ani1 = animation.FuncAnimation(fig1, update1, interval=1000/FPS, blit=False)
        # ani2 = animation.FuncAnimation(fig2, update2, interval=1000/FPS, blit=False)
        ani3 = animation.FuncAnimation(fig3, update3, interval=1000 / FPS, blit=False)
    # else:

    # plotaxs1(axs1, alpha, data, t_vertical = None, order = "", weight = 0, t_start = 0, t_end = 50)
    # plotaxs2(axs2, alpha, data, t_vertical = None, order = "", weight = 0, t_start = 0, t_end = 50)
    # plotaxs3(axs3, alpha, data, t_vertical = None, order = "", weight = 0, t_start = 0, t_end = 50)

    plt.show()


def list2array(data: dict[str, list]) -> dict[str, NDArray]:
    """Converts a dictionary of lists to a dictionary of arrays."""
    for k, v in data.items():
        data[k] = np.array(data[k])
    return data


def quat2rpy(data: dict[str, NDArray]) -> dict[str, NDArray]:
    """Converts the orientation in the data to euler angles."""
    data["rpy"] = R.from_quat(data["quat"]).as_euler("xyz", degrees=True)
    return data


def quat2axis_angle(q1, q2):
    """Computes the angle and axis of rotation between two quaternions.

    Parameters:
        q1 (array-like): First quaternion [x, y, z, w]
        q2 (array-like): Second quaternion [x, y, z, w]

    Returns:
        tuple: (angle in radians, rotation axis as a unit vector)
    """
    # Compute the relative quaternion (q_delta)
    q_delta = R.from_quat(q2) * R.from_quat(q1).inv()

    # Extract the rotation angle and axis
    angle = 2 * np.arccos(q_delta.as_quat()[..., -1])
    axis = q_delta.as_rotvec()
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-6:
        axis = axis / axis_norm
    else:
        axis = np.array([1, 0, 0])  # Default axis if too small

    return angle, axis


def quat2angvel(quat, times):
    """Computes the angular velocity in 3D given a quaternion time series."""  #
    q1 = quat
    q2 = np.roll(quat, 1, axis=0)
    dt = np.diff(times, prepend=1 / 200)
    print(np.mean(dt))
    angle, axis = quat2axis_angle(q1, q2)
    print(angle)
    angvel = (angle / dt)[..., None] * axis
    return angvel


def dquat2angvel(quat, dquat, dt):
    # see https://ahrs.readthedocs.io/en/latest/filters/angular.html
    # Get both rotations and their difference
    q_delta = R.from_quat(quat + dquat) * R.from_quat(quat).inv()
    # Convert that into axis/angle representation
    # Calculate angular velocity = dangle/dt
    angle = 2 * np.arccos(np.clip(q_delta.as_quat()[..., -1], -1.0, 1.0))
    axis = q_delta.as_rotvec()
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-6:
        axis = axis / axis_norm
    else:
        axis = np.array([1, 0, 0])  # Default axis if too small

    angvel = (angle / dt)[..., None] * axis

    return angvel
    # return R.from_quat(quat).apply(angvel)  # RPY rates


def rmse(error_array):
    """Calculated the RMSE of a time series error."""
    error_value = np.sum(error_array, axis=-1)
    return np.sqrt(np.mean(error_value**2))


if __name__ == "__main__":
    drone_name = "cf6"
    estimator_types = ["legacy", "ukf_fitted_DI_rpy"]  # , "ukf_mellinger_rpyt"
    estimator_datasets = []

    path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path, f"data_{drone_name}_measurement.pkl"), "rb") as f:
        data_meas = pickle.load(f)
        data_meas = list2array(data_meas)
        data_meas = quat2rpy(data_meas)

    start_time = data_meas["time"][0]
    data_meas["time"] -= start_time

    # Load all estimator datasets
    for estimator_type in estimator_types:
        with open(os.path.join(path, f"data_{drone_name}_{estimator_type}.pkl"), "rb") as f:
            data_est = pickle.load(f)
            data_est = list2array(data_est)
            data_est = quat2rpy(data_est)
            data_est["time"] -= start_time
            estimator_datasets.append(data_est)

    plots(data_meas, estimator_types, estimator_datasets, animate=False, order="", weight=0)
    # plots(data_meas, data_est, None, order="xyz", weight=0)
    # plots(data_meas, data_est, None, order="", weight=5)
