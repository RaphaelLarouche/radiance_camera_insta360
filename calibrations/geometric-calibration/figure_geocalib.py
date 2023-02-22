# -*- coding: utf-8 -*-
"""

"""

# Module importation
import os
import string
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Other modules
from source.processing import FigureFunctions
import source.processing as processing
from media_comparisons import format_geometric_calibration as format_geo


# Function and classes
def common_projection_curves(r, fl):
    """

    :param r:
    :param fl:
    :return:
    """

    ste = 2 * 180/np.pi * np.arctan2(r, 2 * fl)  # degrees
    equi = 180/np.pi * (r / fl)
    ortho = 180/np.pi * np.arcsin(r / fl)

    return ste, equi, ortho


def projection_curves_graph(calib, fig_ax=(None, None)):
    """
    Only showing green band (as the spectral shifts are very small).
    :param calib:
    :param fig_ax:
    :return:
    """

    # Create figure
    if fig_ax == (None, None):
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = fig_ax

    # Get other curves of projection
    focal = calib["green"].fisheye_params["Intrinsics"]["MappingCoefficients"][0]
    maxima_rho = 810 # ?
    radial_x = np.linspace(0, maxima_rho, 1000)
    stereo, equidis, orthograph = common_projection_curves(radial_x, focal)

    # Plot projected points (geometric calibration)
    r, zen, rmap, residuals, n_im = calib["green"].get_results()
    #rmap, rfitted, _, _, _, _ = calib["green"].reprojection_errors(calib["green"].fisheye_params)
    ang_map = np.arctan2(rmap, calib["green"].imagingfunction(rmap, calib["green"].mapping_coefficients)) * 180 / np.pi
    #z_dws = np.interp(radial_x, r, zen)  # interpolation
    z_dws = np.arctan2(radial_x, calib["green"].imagingfunction(radial_x, calib["green"].mapping_coefficients)) * 180 / np.pi

    # Plot everything

    ax.plot(rmap.ravel(), ang_map.ravel(), marker="s", markersize=2, linestyle="none", markeredgecolor='#1b9e77', markerfacecolor="none", alpha=0.9, label="Reprojected points")
    ax.plot(radial_x, z_dws, linewidth=1.2, color='#1b9e77', linestyle="-.", label="Projection green band")

    ax.plot(radial_x, stereo, linewidth=0.8, color="#a6cee3", label="Stereographic")
    #ax.plot(radial_x, equidis, linewidth=0.8, color="#1f78b4", label="Equidistant")
    ax.plot(radial_x, orthograph, linewidth=0.8, color="#b2df8a", label="Orthographic")

    ax.set_yticks(np.arange(-10, 110, 10))
    ax.set_ylim((-4.616729775163446, 96.95132527843236))
    ax.legend(loc="best", fontsize=7, frameon=False)
    #ax.set_xlabel(r"$\rho$ [px]")
    ax.set_xlabel(r"Radial distance $\rho$ [px]")
    ax.set_ylabel(r"$\theta$ [˚]")

    return fig, ax


def reprojection_errors_graph(calib, fig_ax=(None, None)):
    """
    Figure for euclidean reprojection errors.

    :param calib:
    :param fig_ax:
    :return:
    """
    # Create figure
    if fig_ax == (None, None):
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = fig_ax

    # Marker and color
    marker = ["o", "s", "^"]
    colo = ['#d95f02', '#1b9e77', '#7570b3']
    lab = ["red: 603 nm", "green: 544 nm", "blue: 484 nm"]
    lkeys = {"red: 603 nm":"red", "green: 544 nm":"green", "blue: 484 nm":"blue"}

    ano_txt = ""

    # Plot
    for i, band in enumerate(lab):
        print(lkeys[band])
        current_calib = calib[lkeys[band]]
        r, _, _, _, _, eucl_err = current_calib.reprojection_errors(current_calib.fisheye_params)
        ax.scatter(r, eucl_err, s=3, marker=marker[i], edgecolor=colo[i], facecolor="none", label=lab[i])

        res = current_calib.get_results()
        print(res[-2].mean())
        print(eucl_err.mean())

        ano_txt+="$\overline{{\epsilon}}_{{{0}}}$ = {1:.2f} px\n".format(lab[i].split(':')[0], eucl_err.mean())

    # Set other figure parameters
    ax.text(0.65, 0.02, ano_txt, transform=ax.transAxes, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim((10**-3, 40))
    ax.set_ylabel(r"Reprojection errors $\epsilon_{i}$ [px]")
    ax.set_xlabel(r"Radial distance $\rho$ [px]")
    ax.legend(loc="best", fontsize=7, frameon=False)

    return fig, ax


def spectral_aberrations(calib, fig_ax=(None, None)):
    """

    :param calib:
    :param fig_ax:
    :return:
    """

    # Create figure
    if fig_ax == (None, None):
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = fig_ax

    maxima_rho = 810  # ?
    radial_x = np.linspace(0, maxima_rho, 1000)

    # Reference curve
    zen_ref = np.arctan2(radial_x, calib["green"].imagingfunction(radial_x, calib["green"].mapping_coefficients)) * 180 / np.pi

    colo = ['#d95f02', '#1b9e77', '#7570b3']
    lstyle = ["-", "-.", ":"]
    lab = ["red: 603 nm", "green: 544 nm", "blue: 484 nm"]
    lkeys = {"red: 603 nm": "red", "green: 544 nm": "green", "blue: 484 nm": "blue"}

    for i, k in enumerate(lab):
        print(lkeys[k])
        # Current zenith
        zen = np.arctan2(radial_x, calib[lkeys[k]].imagingfunction(radial_x, calib[lkeys[k]].mapping_coefficients)) * 180 / np.pi

        # Difference
        df = zen - zen_ref
        print('Max {0}: {1:.4f}'.format(lab[i], np.absolute(df).max()))

        ax.plot(radial_x, df, linestyle=lstyle[i], linewidth=1.0, color=colo[i], label=lab[i])

    ax.set_xlabel(r"Radial distance $\rho$ [px]")
    ax.set_ylabel(r"$\theta_{i}-\theta_{green}$ [˚]")
    ax.legend(loc="best", fontsize=7, frameon=False)

    return fig, ax


if __name__ == "__main__":

    plt.style.use("../../figurestyle.mplstyle")

    ff = FigureFunctions()

    # Open geometric calibrations
    base_path = os.path.dirname(__file__)
    geo_water_file = h5py.File(base_path + "/calibrationfiles/geometric-calibration-water.h5")
    geometric_water = format_geo(geo_water_file["lens-close"]["20200730_112353"])  # Construct geometric objects

    # Open Air calibration
    geo_air_file = h5py.File(base_path + "/calibrationfiles/geometric-calibration-air.h5")
    geometric_air = format_geo(geo_air_file["lens-close"]["20190104_192404"])

    # Figure 1 - with reprojection errors
    fig1, ax1 = plt.subplots(3, 2, sharex=True, figsize=ff.set_size(height_ratio=1.1))

    ax1[0, 0].text(-0.05, 1.05, "(" + string.ascii_lowercase[0] + ")", transform=ax1[0, 0].transAxes, size=11, weight='bold')
    ax1[0, 1].text(-0.05, 1.05, "(" + string.ascii_lowercase[1] + ")", transform=ax1[0, 1].transAxes, size=11, weight='bold')
    ax1[1, 0].text(-0.05, 1.05, "(" + string.ascii_lowercase[2] + ")", transform=ax1[1, 0].transAxes, size=11, weight='bold')
    ax1[1, 1].text(-0.05, 1.05, "(" + string.ascii_lowercase[3] + ")", transform=ax1[1, 1].transAxes, size=11, weight='bold')
    ax1[2, 0].text(-0.05, 1.05, "(" + string.ascii_lowercase[4] + ")", transform=ax1[2, 0].transAxes, size=11, weight='bold')
    ax1[2, 1].text(-0.05, 1.05, "(" + string.ascii_lowercase[5] + ")", transform=ax1[2, 1].transAxes, size=11, weight='bold')

    projection_curves_graph(geometric_air, fig_ax=(fig1, ax1[0, 0]))
    projection_curves_graph(geometric_water, fig_ax=(fig1, ax1[0, 1]))

    spectral_aberrations(geometric_air, fig_ax=(fig1, ax1[1, 0]))
    spectral_aberrations(geometric_water, fig_ax=(fig1, ax1[1, 1]))

    reprojection_errors_graph(geometric_air, fig_ax=(fig1, ax1[2, 0]))
    reprojection_errors_graph(geometric_water, fig_ax=(fig1, ax1[2, 1]))

    ax1[0, 0].set_xlabel(None)
    ax1[0, 1].set_xlabel(None)
    ax1[1, 0].set_xlabel(None)
    ax1[1, 1].set_xlabel(None)

    ax1[1, 0].set_yticks(np.arange(-0.35, 0.1, 0.05))
    ax1[1, 0].set_ylim((-0.28, 0.03))

    ax1[1, 1].set_yticks(np.arange(-0.20, 0.6, 0.1))
    ax1[1, 1].set_ylim((-0.15, 0.55))

    ax1[2, 0].legend(loc="upper left", fontsize=7)

    # Figure 2 - without reprojection errors
    fig2, ax2 = plt.subplots(2, 2, sharex=True, figsize=ff.set_size(height_ratio=0.8))

    ax2[0, 0].text(-0.05, 1.05, "(" + string.ascii_lowercase[0] + ")", transform=ax2[0, 0].transAxes, size=11, weight='bold')
    ax2[0, 1].text(-0.05, 1.05, "(" + string.ascii_lowercase[1] + ")", transform=ax2[0, 1].transAxes, size=11, weight='bold')
    ax2[1, 0].text(-0.05, 1.05, "(" + string.ascii_lowercase[2] + ")", transform=ax2[1, 0].transAxes, size=11, weight='bold')
    ax2[1, 1].text(-0.05, 1.05, "(" + string.ascii_lowercase[3] + ")", transform=ax2[1, 1].transAxes, size=11, weight='bold')

    projection_curves_graph(geometric_air, fig_ax=(fig2, ax2[0, 0]))
    projection_curves_graph(geometric_water, fig_ax=(fig2, ax2[0, 1]))

    spectral_aberrations(geometric_air, fig_ax=(fig2, ax2[1, 0]))
    spectral_aberrations(geometric_water, fig_ax=(fig2, ax2[1, 1]))

    ax2[0, 0].set_xlabel("")
    ax2[0, 1].set_xlabel("")

    # Figure 3 - reprojection errors
    fig3, ax3 = plt.subplots(1, 2, sharex=True, figsize=ff.set_size(height_ratio=0.5))

    ax3[0].text(-0.05, 1.05, "(" + string.ascii_lowercase[0] + ")", transform=ax3[0].transAxes, size=11, weight='bold')
    ax3[1].text(-0.05, 1.05, "(" + string.ascii_lowercase[1] + ")", transform=ax3[1].transAxes, size=11, weight='bold')

    reprojection_errors_graph(geometric_air, fig_ax=(fig1, ax3[0]))
    reprojection_errors_graph(geometric_water, fig_ax=(fig1, ax3[1]))

    ax3[0].set_xlim(ax2[0, 0].get_xlim())
    ax3[1].set_xlim(ax2[0, 0].get_xlim())

    # Saving figures
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    fig1.savefig("figures/geometric_close.png", format="png", dpi=600)
    fig2.savefig("figures/geo_close_no_reproj_err.png", format="png", dpi=600)
    fig3.savefig("figures/reproj_err.png", format="png", dpi=600)

    plt.show()
