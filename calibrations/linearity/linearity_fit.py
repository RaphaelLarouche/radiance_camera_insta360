# -*- coding: utf-8 -*-
"""
Linearity figure and (linear) fit.
"""

# Module importation
import os
import string
import deepdish
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Other modules
from source.processing import ProcessImage, FigureFunctions
from source.geometric_rolloff import MatlabGeometricMengine


# Functions
def averaging_3rdimension(imstack, exposure_param):
    """

    :param imstack:
    :param exposure_param:
    :return:
    """
    exp_param_unique = np.unique(exposure_param)
    imstack_avg = np.empty((imstack.shape[0], imstack.shape[1], exp_param_unique.shape[0]))
    for n, exp in enumerate(exp_param_unique):
        cond = exp == exposure_param
        curr_im = imstack[:, :, cond.astype(bool)]

        imstack_avg[:, :, n] = curr_im.mean(axis=2)

    return imstack_avg


def estimators_stde(slope, intercept, x, y):
    """
    Standard error of the estimates for a linear regression.

    :param slope:
    :param intercept:
    :param x:
    :param y:
    :return:
    """
    y_i = slope * x + intercept
    ssr = np.sum((y - y_i) ** 2)  # sum of squared residual
    mse = np.sqrt(ssr / (x.shape[0] - 2))

    std_slope = mse * np.sqrt(1 / np.sum((x - x.mean()) ** 2))
    std_intercept = std_slope * np.sqrt(np.sum(x ** 2) / x.shape[0])

    return std_slope, std_intercept


def plot_linear_regression(ax, x, y, text, postext=(80, 40), oneone=False):
    """

    :param fig:
    :param ax:
    :param row:
    :param x:
    :param y:
    :param err_y:
    :return:
    """

    slo, inte, r, _, stde_slope = stats.linregress(x, y)
    _, stde_inter = estimators_stde(slo, inte, x, y)

    x_regression = np.linspace(x.min() * 0.8, x.max() * 1.2, 50)
    ax.plot(x_regression, slo * x_regression + inte, label="linear fit")
    if oneone:
        ax.plot(x_regression, x_regression, linestyle="-.", color="black", label="1:1")
    ax.text(postext[0], postext[1], text.format(slo, stde_slope * 1.96, inte, stde_inter * 1.96, r ** 2), fontsize=6)
    return ax


if __name__ == "__main__":

    # ProcessImage object
    pp = ProcessImage()

    # FigureFunctions object
    ff = FigureFunctions()

    # Figure style
    plt.style.use("../../figurestyle.mplstyle")

    path_i360 = os.path.dirname(os.path.dirname(__file__))
    filepath_exp = "/Volumes/MYBOOK/data-i360/calibrations/linearity/integration-time/"
    filepath_iso = "/Volumes/MYBOOK/data-i360/calibrations/linearity/iso-gain/"

    while True:
        answer = input("Which lens do you want to analyze? (c/f): ")
        if answer.lower() in ["c", "f"]:
            break

    if answer.lower() == "c":

        # Path list
        imlist_exp = pp.imageslist(filepath_exp + "lensclose")
        imlist_iso = pp.imageslist(filepath_iso + "lensclose")
        imlist_exp_bl = pp.imageslist_dark(filepath_exp + "lensclose")
        imlist_iso_bl = pp.imageslist_dark(filepath_iso + "lensclose")

        # Geometric calibration
        geocalib = deepdish.io.load(path_i360 + "/geometric-calibration/calibrationfiles/geometric-calibration-water.h5", "/lens-close/20200730_112353/")

        wim = "close"

    elif answer.lower() == "f":

        # Path list
        imlist_exp = pp.imageslist(filepath_exp + "lensfar")
        imlist_iso = pp.imageslist(filepath_iso + "lensfar")
        imlist_exp_bl = pp.imageslist_dark(filepath_exp + "lensfar")
        imlist_iso_bl = pp.imageslist_dark(filepath_iso + "lensfar")

        # Geometric calibration
        geocalib = deepdish.io.load(path_i360 + "/geometric-calibration/calibrationfiles/geometric-calibration-water.h5", "/lens-far/20200730_143716/")

        wim = "far"
    else:
        raise ValueError("Not valid choice.")

    # Geometric classes
    geo = {}
    zen = {}
    for i in geocalib["fp"].keys():
        geo[i] = MatlabGeometricMengine(geocalib["fp"][i], geocalib["ierror"][i])
        r, z, a, = geo[i].angular_coordinates()
        zen[i] = z
    channel = {0: "red", 1: "green", 2: "blue"}

    # Illuminated images stack
    imstack_exp, exp_expln, iso_expln, bl_expln = pp.imagestack(imlist_exp, wim)
    imstack_iso, exp_isoln, iso_isoln, bl_isoln = pp.imagestack(imlist_iso, wim)

    imstack_exp_bl, exp_bl, iso_bl, _ = pp.imagestack(imlist_exp_bl, wim)

    # Averaging
    imstack_exp_avg = averaging_3rdimension(imstack_exp, exp_expln)  # 4 Image average
    imstack_iso_avg = averaging_3rdimension(imstack_iso, iso_isoln)  # 4 Image average

    # Dark removal
    imstack_exp_avg -= bl_expln[None, None, ::4]
    imstack_exp_avg = np.clip(imstack_exp_avg, 0, None)

    imstack_iso_avg -= bl_isoln[None, None, ::4]
    imstack_iso_avg = np.clip(imstack_iso_avg, 0, None)

    # Loop
    # Pre-allocation
    mask_zen = 5  # degrees

    # Exposure time
    dn_exp = np.empty((imstack_exp_avg.shape[2], 3))
    noise_exp = np.empty((imstack_exp_avg.shape[2], 3))

    for i in range(imstack_exp_avg.shape[2]):

        im_dws = pp.dwnsampling(imstack_exp_avg[:, :, i], "RGGB")

        for b in range(im_dws.shape[2]):
            curr_im = im_dws[:, :, b]
            mask_angular = zen[channel[b]] <= mask_zen

            pixel_values = curr_im[mask_angular]
            dn_exp[i, b] = pixel_values.mean()
            noise_exp[i, b] = pixel_values.std()

    # Exposure time
    dn_iso = np.empty((imstack_iso_avg.shape[2], 3))
    noise_iso = np.empty((imstack_iso_avg.shape[2], 3))

    for i in range(imstack_iso_avg.shape[2]):

        im_dws = pp.dwnsampling(imstack_iso_avg[:, :, i], "RGGB")

        for b in range(im_dws.shape[2]):
            curr_im = im_dws[:, :, b]
            mask_angular = zen[channel[b]] <= mask_zen

            pixel_values = curr_im[mask_angular]
            dn_iso[i, b] = pixel_values.mean()
            noise_iso[i, b] = pixel_values.std()

    # Figures
    fs = ff.set_size(subplots=(2, 3))

    fig1, ax1 = plt.subplots(2, 3, sharey="row", figsize=(fs[0], fs[1] * 1.7))
    fig2, ax2 = plt.subplots(2, 3, sharey="row", figsize=(fs[0], fs[1] * 1.7))

    tintms = np.unique(exp_expln) * 1000
    iso = np.unique(iso_isoln) * 0.01

    x = np.linspace(tintms.min() * 0.8, tintms.max() * 1.2, 50)
    x_iso = np.linspace(iso.min() * 0.8, iso.max() * 1.2, 50)

    tx_exp = "$DN_{{i}} = m \cdot t_{{int}} + b$\n$m = ({0:.2f}\pm{1:.2f})$\n$b = ({2:.0f}\pm{3:.0f})$\n$R^{{2}} = {4:.6f}$"
    tx_iso = "$DN_{{i}} = m \cdot ISO \cdot 0.01 + b$\n$m = ({0:.2f}\pm{1:.2f})$\n$b = ({2:.0f}\pm{3:.0f})$\n$R^{{2}} = {4:.6f}$"

    for b in range(dn_exp.shape[1]):

        # Figure 1 ___ exp
        # linear regression
        if b == 0 or b == 1:
            tint_reg, dn_exp_reg = tintms[:-2], dn_exp[:-2, b]
        else:
            tint_reg, dn_exp_reg = tintms.copy(), dn_exp[:, b].copy()

        ax1[0, b].errorbar(tintms, dn_exp[:, b], yerr=noise_exp[:, b], linestyle="none", marker=".", label="averaged $DN$")

        # Linear regression
        plot_linear_regression(ax1[0, b], tint_reg, dn_exp_reg, tx_exp, postext=(80, 40))

        ax1[0, b].set_xscale("log")
        ax1[0, b].set_yscale("log")

        ax1[0, b].set_xlabel("exposure time $t_{int}$ [ms]")

        ax1[0, b].text(-0.05, 1.05, "(" + string.ascii_lowercase[b] + ")", transform=ax1[0, b].transAxes, size=11, weight='bold')

        # Figure 1 ___ iso

        ax1[1, b].errorbar(iso, dn_iso[:, b], yerr=noise_iso[:, b], linestyle="none", marker=".", label="averaged $DN$")
        # Linear regression
        plot_linear_regression(ax1[1, b], iso[1:], dn_iso[1:, b], tx_iso, postext=(1, 2000))

        ax1[1, b].set_xscale("log")
        ax1[1, b].set_yscale("log")

        ax1[1, b].set_xlabel("$ISO \cdot 0.01$")

        # Figure 2
        ax2[0, b].errorbar(tintms / tintms[0], dn_exp[:, b] / dn_exp[0, b], linestyle="none", marker=".", label="averaged $DN$")
        plot_linear_regression(ax2[0, b], tint_reg / tint_reg[0], dn_exp_reg / dn_exp_reg[0], tx_exp, postext=(10, 1), oneone=True)
        ax2[1, b].errorbar(iso / iso[1], dn_iso[:, b] / dn_iso[1, b], linestyle="none", marker=".", label="averaged $DN$")
        plot_linear_regression(ax2[1, b], iso / iso[1], dn_iso[:, b] / dn_iso[1, b], tx_iso, postext=(7, 1), oneone=True)

        ax2[0, b].set_xscale("log")
        ax2[0, b].set_yscale("log")

        ax2[0, b].set_xlabel("$t_{int} / t_{int, 1}$")
        ax2[0, b].set_ylabel("$DN / DN_{t_{int, 1}}$")

        ax2[1, b].set_xscale("log")
        ax2[1, b].set_yscale("log")

        ax2[1, b].set_xlabel("$ISO / ISO_{2}$")  # DIVIDED BY THE SECOND Measurements!
        ax2[1, b].set_ylabel("$DN / DN_{ISO_{2}}$")

    # Figure 1
    ax1[0, 0].set_ylabel("$DN_{i}$")
    ax1[1, 0].set_ylabel("$DN_{i}$")

    # Saving figures
    fig1.tight_layout()
    fig2.tight_layout()

    optic_correspondance = {"c": "close", "f": "far"}

    fig1.savefig("figures/linearity-fit-{0}.pdf".format(optic_correspondance[answer.lower()]), format="pdf", dpi=600)

    plt.show()
