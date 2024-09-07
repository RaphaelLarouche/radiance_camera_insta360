# -*- coding: utf-8 -*-
"""

"""


# Module importation
import h5py
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt

import source.processing as processing
from source.geometric_rolloff import MatlabGeometricMengine
from media_comparisons import format_geometric_calibration

# Classes and functions


if __name__ == "__main__":

    # Information
    sn = "2BW7X7"

    #rho = np.linspace(0, 1496, 1000)

    # Geometric calibration files
    geo_calib = h5py.File(f"calibrationfiles/geometric-calibration-2BW7X7.h5", "r")

    # Loop
    calib_list = ["air/nocover/front/20230212_115034", "air/nocover/back/20230212_121752",
                  "water/nocover/front/20230404_122521", "water/nocover/back/20230404_124245",
                  "air/cover/front/20230406_114342", "air/cover/back/20230406_120016",
                  "water/cover/front/20230411_171230", "water/cover/back/20230411_172456"]

    ls = {"air/nocover/front/20230212_115034": "-",
          "air/nocover/back/20230212_121752": "-",
          "water/nocover/front/20230404_122521": "-.",
          "water/nocover/back/20230404_124245": "-.",
          "air/cover/front/20230406_114342": "--",
          "air/cover/back/20230406_120016": "--",
          "water/cover/front/20230411_171230": ":",
          "water/cover/back/20230411_172456": ":"}

    # Pre-allocation
    fig1, ax1 = plt.subplots(1, 3, sharey=True, sharex=True)

    for i, cl in enumerate(calib_list):

        geo = format_geometric_calibration(geo_calib[cl])

        name_split = cl.split("/")

        for b, k in enumerate(geo.keys()):

            res = geo[k].get_results()

            # Max points
            rpoints, _, _, _, _, _ = geo[k].reprojection_errors(geo[k].fisheye_params)
            rho = np.linspace(0, rpoints.max(), 500)

            zenith_interp = np.interp(rho, res[0], res[1])

            ax1[b].plot(rho, zenith_interp, linestyle=ls[cl], label=f"{name_split[0]}/{name_split[1]}/{name_split[2]}")

    ax1[0].axvline(1496, color="k", alpha=0.5)
    ax1[1].axvline(1496, color="k", alpha=0.5)
    ax1[2].axvline(1496, color="k", alpha=0.5)

    ax1[0].set_ylabel(r"$\theta$ [˚]")

    ax1[0].set_xlabel(r"$\rho$ [px]")
    ax1[1].set_xlabel(r"$\rho$ [px]")
    ax1[2].set_xlabel(r"$\rho$ [px]")

    ax1[0].legend(loc="best", fontsize=7, frameon=False)
    ax1[1].legend(loc="best", fontsize=7, frameon=False)
    ax1[2].legend(loc="best", fontsize=7, frameon=False)

    fig1.tight_layout()

    # 2C9JCA
    geo_calib_2C9 = h5py.File(f"calibrationfiles/geometric-calibration-2C9JCA.h5", "r")

    calib_list_2C9 = ["air/nocover/front/20230322_150036", "air/nocover/back/20230322_153728", "water/nocover/front/20230404_114030",
                      "water/nocover/back/20230404_115610"]

    ls_2 = {"air/nocover/front/20230322_150036": "-", "air/nocover/back/20230322_153728": "-",
            "water/nocover/front/20230404_114030": "-.", "water/nocover/back/20230404_115610": "-."}

    fig2, ax2 = plt.subplots(1, 3, sharey=True, sharex=True)

    for i, cl in enumerate(calib_list_2C9):

        geo = format_geometric_calibration(geo_calib_2C9[cl])

        name_split = cl.split("/")

        for b, k in enumerate(geo.keys()):

            res = geo[k].get_results()

            # Max points
            rpoints, _, _, _, _, _ = geo[k].reprojection_errors(geo[k].fisheye_params)
            #rho = np.linspace(0, rpoints.max(), 500)
            rho = np.linspace(0, 1496, 500)

            zenith_interp = np.interp(rho, res[0], res[1])

            ax2[b].plot(rho, zenith_interp, linestyle=ls_2[cl], label=f"{name_split[0]}/{name_split[1]}/{name_split[2]}")

    ax2[0].axvline(1496, color="k", alpha=0.5)
    ax2[1].axvline(1496, color="k", alpha=0.5)
    ax2[2].axvline(1496, color="k", alpha=0.5)

    ax2[0].set_ylabel(r"$\theta$ [˚]")

    ax2[0].set_xlabel(r"$\rho$ [px]")
    ax2[1].set_xlabel(r"$\rho$ [px]")
    ax2[2].set_xlabel(r"$\rho$ [px]")

    ax2[0].legend(loc="best", fontsize=7, frameon=False)
    ax2[1].legend(loc="best", fontsize=7, frameon=False)
    ax2[2].legend(loc="best", fontsize=7, frameon=False)

    fig2.tight_layout()

    plt.show()