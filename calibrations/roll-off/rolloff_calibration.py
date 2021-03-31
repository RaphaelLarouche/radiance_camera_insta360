# -*- coding: utf-8 -*-
"""
Roll-off calibration. Roll-off represents the irradiance fall-off across the image plane due to vignetting and
diminution of the optic etendue.
"""

# Module importation
import os
import time
import deepdish
import numpy as np
import matplotlib.pyplot as plt

# Other modules
from source.processing import ProcessImage, FigureFunctions
from source.geometric_rolloff import RolloffFunctions


# Functions
def plot_rolloff(axe, angles, rolloff, err, mark, cl, lab=""):
    """
    Plot function.

    :param axe: matplotlib axe
    :param angles: angles degrees (array)
    :param rolloff: roll-off (array)
    :param err: errors (array)
    :param mark: marker (str)
    :param cl: color (str)
    :param lab: label (str)
    :return:
    """

    if lab:
        axe.errorbar(angles, rolloff,
                     yerr=err, marker=mark, linestyle="", markerfacecolor=cl,
                     markeredgecolor="grey", ecolor="grey", elinewidth=1,
                     markersize=4, label=lab)
    else:
        axe.errorbar(angles, rolloff,
                     yerr=err, marker=mark, linestyle="", markerfacecolor=cl,
                     markeredgecolor="grey", ecolor="grey", elinewidth=1,
                     markersize=4, label=lab)

    axe.set_ylabel("Roll-off")

    return axe


if __name__ == "__main__":

    # Input lens to analyzed
    while True:
        answer = input("Which lens do you want to analyze? (c/f): ")
        if answer.lower() in ["c", "f"]:
            break

    # Instance of figurefunctions
    ff = FigureFunctions()

    # Instance of ProcessImage
    process = ProcessImage()

    # Open files
    generalpath = "/Volumes/MYBOOK/data-i360/calibrations/relative-illumination/"
    pathgeo = os.path.dirname(os.path.dirname(__file__))

    if answer.lower() == "c":

        # Roll-off images
        path_00 = generalpath + "lensclose/00"
        path_90 = generalpath + "lensclose/90"

        # Geometric calibration
        geocalib = deepdish.io.load(pathgeo + "/geometric-calibration/calibrationfiles/geometric-calibration-water.h5", "/lens-close/20200730_112353/")
        FishParams = geocalib["fp"]

        # AIR roll-off
        rcal = RolloffFunctions(geocalib["fp"], geocalib["ierror"], "close")
    else:

        # Roll-off images
        path_00 = generalpath + "/lensfar/00"
        path_90 = generalpath + "/lensfar/90"

        # Geometric calibration
        geocalib = deepdish.io.load(pathgeo + "/geometric-calibration/calibrationfiles/geometric-calibration-water.h5", "/lens-far/20200730_143716/")
        FishParams = geocalib["fp"]

        # Air roll-off
        rcal = RolloffFunctions(geocalib["fp"], geocalib["ierror"], "far")

    # Dictionary to open DNG files using readDNG_insta360
    wlens = {"c": "close", "f": "far"}

    # Loop
    imlist_00 = rcal.imageslist(path_00)
    imtotal, roff_centro, centro = rcal.rolloff_centroid_water(imlist_00, 0, azimuth="0")
    imlist_90 = rcal.imageslist(path_90)
    imtotal_90, roff_centro_90, centro_90 = rcal.rolloff_centroid_water(imlist_90, 1, azimuth="90")

    # New method - still needs improvement A, B = rcal.rolloff_angular_range(path_00, 0, [0, 1728], [775, 1100], -2, 1)

    # Clipping image
    imtotal = np.clip(imtotal + imtotal_90, 0, 2**14 - 1)

    # Normalization
    roff_centro_norm = roff_centro.copy()
    roff_centro_norm["DN_avg"] /= np.max(roff_centro["DN_avg"], axis=0)
    roff_centro_norm["DN_std"] /= np.max(roff_centro["DN_avg"], axis=0)

    roff_centro_90_norm = roff_centro_90.copy()
    roff_centro_90_norm["DN_avg"] /= np.max(roff_centro_90["DN_avg"], axis=0)
    roff_centro_90_norm["DN_std"] /= np.max(roff_centro_90["DN_avg"], axis=0)

    print(np.mean(roff_centro["DN_std"], axis=0))

    # Figures
    plt.style.use("../../figurestyle.mplstyle")

    # Fig1 - zenith, azimuth
    # Fig2 - image total
    fig1, ax1 = plt.subplots(1, 1)
    ax1.axhline(y=rcal.geo["red"].intrinsics["DistortionCenter"][1], xmin=0, xmax=3455//2)
    ax1.axvline(x=rcal.geo["red"].intrinsics["DistortionCenter"][0], ymin=0, ymax=3455//2)

    ax1.imshow(imtotal[:, :, 0])
    ax1.plot(centro["x"][:, 0], centro["y"][:, 0], "r+")
    ax1.plot(centro_90["x"][:, 0], centro_90["y"][:, 0], "r+")

    # Fig3 - rolloff using data around centroid
    fig2, ax2 = plt.subplots(3, 1, sharex=True, figsize=(6.4, 7))

    # Fig 8 - same as fig2 but in black and white
    fig3, ax3 = plt.subplots(3, 1, sharex=True, figsize=(6.4, 7))

    # Fig 9
    fig4, ax4 = plt.subplots(1, 1, figsize=ff.set_size(fraction=0.7, height_ratio=0.53))

    # Fitting
    theta = np.linspace(0, 80, 100)
    marker = ["o", "s", "d"]

    color = iter(['#d62728', '#2ca02c', '#1f77b4'])
    lab = ["602 nm", "544 nm", "484 nm"]
    ls = ["-", "-.", ":"]

    fitresults = np.empty((3, 5))

    legp, legst = [], []

    for band in range(roff_centro.shape[1]):
        # Fit
        print("8 degree fit")
        popt2, pcov2, rsquared2, perr2 = rcal.rolloff_curvefit(roff_centro_norm["a"][:, band], roff_centro_norm["DN_avg"][:, band])
        popt90, pcov90, rsquared90, perr90 = rcal.rolloff_curvefit(roff_centro_90_norm["a"][:, band], roff_centro_90_norm["DN_avg"][:, band])

        # Fit
        atot = np.append(roff_centro_norm["a"][:, band], roff_centro_90_norm["a"][:, band])
        rtot = np.append(roff_centro_norm["DN_avg"][:, band], roff_centro_90_norm["DN_avg"][:, band])

        poptall, pcovall, rsquareall, perrall = rcal.rolloff_curvefit(atot, rtot)  # 90˚ and 0˚ azimuth

        # Saving fit results
        fitresults[band, :] = poptall

        # Plots
        col = next(color)

        plot_rolloff(ax2[band], roff_centro_norm["a"][:, band], roff_centro_norm["DN_avg"][:, band], roff_centro_norm["DN_std"][:, band], "o", col, "0˚ azimuth "+lab[band])
        plot_rolloff(ax2[band], roff_centro_90_norm["a"][:, band], roff_centro_90_norm["DN_avg"][:, band], roff_centro_90_norm["DN_std"][:, band], "s", col, "90˚ azimuth "+lab[band])

        plot_rolloff(ax3[band], roff_centro_norm["a"][:, band], roff_centro_norm["DN_avg"][:, band], roff_centro_norm["DN_std"][:, band], "o", "none", "$\phi = $0˚($k=1$ uncertainty)")
        plot_rolloff(ax3[band], roff_centro_90_norm["a"][:, band], roff_centro_90_norm["DN_avg"][:, band], roff_centro_90_norm["DN_std"][:, band], "s", "none", "$\phi = $90˚ ($k=1$ uncertainty)")

        ax3[band].plot(theta, process.rolloff_polynomial(theta, *poptall), color="k", linewidth=1.7, linestyle="-", label="Fit")
        ax3[band].text(52, 0.91, "{0}\n$r^{{2}}={1:.5f}$".format(lab[band], rsquareall), fontsize=9)

        ax2[band].plot(theta, process.rolloff_polynomial(theta, *poptall), color=col, linewidth=1.7, linestyle="-", label="Polynomial fit")
        ax2[band].text(52, 0.91, "$k=1$ standard uncertainty\n$r^{{2}}={0:.5f}$".format(rsquareall), fontsize=9)

        # Figure 9
        ax4.plot(roff_centro_norm["a"][:, band], roff_centro_norm["DN_avg"][:, band], marker=marker[band], markersize=3, linestyle="none", markeredgecolor="k", markerfacecolor="none", alpha=0.5, label=lab[band])
        ax4.plot(roff_centro_90_norm["a"][:, band], roff_centro_90_norm["DN_avg"][:, band], marker=marker[band], markersize=3, linestyle="none", markeredgecolor="k", markerfacecolor="none", alpha=0.5,)
        p = ax4.plot(theta, process.rolloff_polynomial(theta, *poptall), color="k", linestyle=ls[band])

        legp.append(p[0])
        legst.append("$a_{0}$ = %.2E $a_{2}$ = %.2E $a_{4}$ = %.2E $a_{6}$ = %.2E  $a_{8}$ = %.2E" % tuple(fitresults[band, :]))

    leg = plt.legend(legp, legst, fontsize=5, frameon=False)

    ax4.add_artist(leg)

    # Fig3 parameters
    ax2[0].set_xlim((0, 80))
    ax2[1].set_xlim((0, 80))
    ax2[2].set_xlabel(r"$\theta$ [˚]")

    ax2[0].legend(loc="best")
    ax2[1].legend(loc="best")
    ax2[2].legend(loc="best")

    # Fig3 parameters
    ax3[0].set_yticks(np.arange(0.4, 1.2, 0.2))
    ax3[1].set_yticks(np.arange(0.4, 1.2, 0.2))
    ax3[2].set_yticks(np.arange(0.4, 1.2, 0.2))
    ax3[0].set_xlim((0, 80))
    ax3[1].set_xlim((0, 80))
    ax3[2].set_xlabel(r"$\theta$ [˚]")

    ax3[0].legend(loc="best", fontsize=10)

    # Figure 9
    ax4.set_xticks(np.arange(0, 90, 10))
    ax4.set_xlabel(r"$\theta$ [˚]")
    ax4.set_ylabel(r"$R(\theta)$")
    ax4.legend(loc="best")

    fig3.tight_layout()
    fig4.tight_layout()

    # Saving data
    save_answer = process.save_results()
    timestr = time.strftime("%Y%m%d", time.localtime(os.stat(imlist_00[0])[-2]))

    if save_answer == "y":

        pathname = "calibrationfiles/rolloff_w.h5"
        timestr = time.strftime("%Y%m%d", time.localtime(os.stat(imlist_00[0])[-2]))

        if answer.lower() == "c":
            process.create_hdf5_dataset(pathname, "lens-close/" + timestr, "fit-coefficients", fitresults)
        else:
            process.create_hdf5_dataset(pathname, "lens-far/" + timestr, "fit-coefficients", fitresults)

        # Saving figure
        fig4.savefig("figures/roll-off-{}.pdf".format(wlens[answer.lower()]), format="pdf", dpi=600)

    plt.show()
