# -*- coding: utf-8 -*-
"""
Comparison between geometric projection curves for air and water.
"""

# Module importation
import os
import deepdish
import numpy as np
import matplotlib.pyplot as plt

# Other modules
from source.geometric_rolloff import MatlabGeometricMengine

if __name__ == "__main__":

    path_calib = os.path.dirname(__file__)

    fig1, ax1 = plt.subplots(3, 2, sharey=True, sharex=True)

    r_dws = np.linspace(0, 810 * 1.05, 1000)

    # Water
    geo_water = deepdish.io.load(path_calib + "/calibrationfiles/geometric-calibration-water.h5")
    geo_water_close = geo_water["lens-close"]["20200730_112353"]
    geo_water_far = geo_water["lens-far"]["20200730_143716"]

    geometric_water_close = {}
    geometric_water_far = {}
    for i, k in enumerate(geo_water_close["fp"].keys()):
        geometric_water_close[k] = MatlabGeometricMengine(geo_water_close["fp"][k],  geo_water_close["ierror"][k])
        geometric_water_far[k] = MatlabGeometricMengine(geo_water_far["fp"][k], geo_water_far["ierror"][k])

        # Plot
        res_w_c = geometric_water_close[k].get_results()
        z_w_c = np.interp(r_dws, res_w_c[0], res_w_c[1])  # interpolation

        res_w_f = geometric_water_far[k].get_results()
        z_w_f = np.interp(r_dws, res_w_f[0], res_w_f[1])  # interpolation

        ax1[i, 0].plot(r_dws, z_w_c, color=k, label="close water")
        ax1[i, 1].plot(r_dws, z_w_f, color=k, label="far water")

    # Air
    geo_air = deepdish.io.load(path_calib + "/calibrationfiles/geometric-calibration-air.h5")
    geo_air_close = geo_air["lens-close"]["20190104_192404"]
    geo_air_far = geo_air["lens-far"]["20190104_214037"]

    geometric_air_close = {}
    geometric_air_far = {}

    for i, k in enumerate(geo_air_close["fp"].keys()):
        geometric_air_close[k] = MatlabGeometricMengine(geo_air_close["fp"][k],  geo_air_close["ierror"][k])
        geometric_air_far[k] = MatlabGeometricMengine(geo_air_far["fp"][k], geo_air_far["ierror"][k])

        # Plot
        res_a_c = geometric_air_close[k].get_results()
        z_a_c = np.interp(r_dws, res_a_c[0], res_a_c[1])  # interpolation

        res_a_f = geometric_air_far[k].get_results()
        z_a_f = np.interp(r_dws, res_a_f[0], res_a_f[1])  # interpolation

        ax1[i, 0].plot(r_dws, z_a_c, linestyle="-.", color=k, label="close air")
        ax1[i, 1].plot(r_dws, z_a_f, linestyle="-.", color=k, label="far air")

        ax1[i, 0].set_ylabel(r"$\theta~[\degree]$")

        ax1[i, 0].legend(loc="best")
        ax1[i, 1].legend(loc="best")

    ax1[2, 0].set_xlabel(r"radial euclidean distance $r$")

    plt.show()
