# -*- coding: utf-8 -*-
"""
Comparison between geometric projection curves for air and water.
"""

# Module importation
import os
import string
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Other modules
import source.processing as processing
from source.geometric_rolloff import MatlabGeometricMengine


# Functions
def format_geometric_calibration(calibration):
    """
    Function to format correctly all the calibration bands (red, green, blue) object (MatlabGeometricMengine)
    inside a dictionary.
    :return:
    """
    gvar = {}
    for a in calibration["fp"].keys():
        gvar[a] = MatlabGeometricMengine(calibration["fp"][a], calibration["ierror"][a])
    return gvar


if __name__ == "__main__":

    # Path for folder calibration
    path_calib = os.path.dirname(__file__)

    # Object figure function
    ff = processing.FigureFunctions()

    # Pre-allocation
    plt.style.use("../../figurestyle.mplstyle")
    fig1, ax1 = plt.subplots(3, 2, sharey=True, sharex=True)
    fig2, ax2 = plt.figure(figsize=ff.set_size(443.86319, height_ratio=0.45)), []
    ax2.append(fig2.add_subplot(121))
    ax2.append(fig2.add_subplot(122, projection="3d"))

    # Open Water calibration
    geo_water = h5py.File(path_calib + "/calibrationfiles/geometric-calibration-water.h5")
    geo_water_close = geo_water["lens-close"]["20200730_112353"]
    geo_water_far = geo_water["lens-far"]["20200730_143716"]

    geometric_water_close = format_geometric_calibration(geo_water_close)
    geometric_water_far = format_geometric_calibration(geo_water_far)

    # Open Air calibration
    geo_air = h5py.File(path_calib + "/calibrationfiles/geometric-calibration-air.h5")
    geo_air_close = geo_air["lens-close"]["20190104_192404"]
    geo_air_far = geo_air["lens-far"]["20190104_214037"]

    geometric_air_close = format_geometric_calibration(geo_air_close)
    geometric_air_far = format_geometric_calibration(geo_air_far)

    # Radial euclidean distance for interpolation
    r_dws = np.linspace(0, 810 * 1.05, 1000)

    ls = ["-", "-.", ":"]
    lab = ["602 nm", "544 nm", "484 nm"]
    color = ['#d62728', '#2ca02c', '#1f77b4']

    for i, k in enumerate(geo_water_close["fp"].keys()):

        # Plot water
        res_w_c = geometric_water_close[k].get_results()
        z_w_c = np.interp(r_dws, res_w_c[0], res_w_c[1])  # interpolation

        res_w_f = geometric_water_far[k].get_results()
        z_w_f = np.interp(r_dws, res_w_f[0], res_w_f[1])  # interpolation

        ax1[i, 0].plot(r_dws, z_w_c, color=k, label="close water")
        ax1[i, 1].plot(r_dws, z_w_f, color=k, label="far water")

        # Plot air
        res_a_c = geometric_air_close[k].get_results()
        z_a_c = np.interp(r_dws, res_a_c[0], res_a_c[1])  # interpolation

        res_a_f = geometric_air_far[k].get_results()
        z_a_f = np.interp(r_dws, res_a_f[0], res_a_f[1])  # interpolation

        ax1[i, 0].plot(r_dws, z_a_c, linestyle="-.", color=k, label="close air")
        ax1[i, 1].plot(r_dws, z_a_f, linestyle="-.", color=k, label="far air")

        ax1[i, 0].set_ylabel(r"$\theta~[\degree]$")

        ax1[i, 0].legend(loc="best")
        ax1[i, 1].legend(loc="best")

        # Paper figure (figure 2)
        ax2[0].plot(r_dws, z_a_c, color=color[i], linestyle=ls[i], label=lab[i])
        ax2[0].plot(r_dws, z_w_c, color=color[i], linestyle=ls[i], label=lab[i])

    ax1[2, 0].set_xlabel(r"radial euclidean distance $r$")
    ax1[2, 1].set_xlabel(r"radial euclidean distance $r$")

    # ____________ Figure for paper
    ax2[0].text(-0.05, 1.05, "(" + string.ascii_lowercase[0] + ")", transform=ax2[0].transAxes, size=11, weight='bold')
    ax2[0].set_xticks(np.arange(0, 1000, 100))
    ax2[0].set_xlabel(r"$\rho$ [px]")
    ax2[0].set_ylabel(r"$\theta$ [˚]")
    ax2[0].legend(loc="best")

    ax2[1].text2D(-0.05, 1.05, "(" + string.ascii_lowercase[1] + ")", transform=ax2[1].transAxes, size=11, weight='bold')
    ax2[1] = geometric_water_close["red"].draw_targets(ax2[1])
    ax2[1].view_init(elev=35, azim=-122)
    ax2[1].set_xlim((-150, 150))
    ax2[1].set_zlim((-50, 150))
    ax2[1].set_ylim((0, 200))
    ax2[1].set_yticks(np.arange(0, 250, 50))
    ax2[1].set_zticks(np.arange(-50, 200, 50))

    fig2.tight_layout()

    plt.show()
