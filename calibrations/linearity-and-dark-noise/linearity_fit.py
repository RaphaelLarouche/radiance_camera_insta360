# -*- coding: utf-8 -*-
"""
Linearity figure and (linear) fit.
"""

# Module importation
import os
import h5py
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


def plot_linear_regression(ax, x, y, text, ls, postext=(80, 40), oneone=False, color=False):
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
    stde_slope_calc, stde_inter = estimators_stde(slo, inte, x, y)
    #print(f'Scipy stde: {stde_slope}, own calculation stde: {stde_slope_calc}')
    x_regression = np.linspace(x.min() * 0.8, x.max() * 1.2, 50)
    if color:
        ax.plot(x_regression, slo * x_regression + inte, color=color, linestyle=ls, label="linear fit")
    else:
        ax.plot(x_regression, slo * x_regression + inte, linestyle=ls, label="linear fit")
    if oneone:
        ax.plot(x_regression, x_regression, linestyle="-.", color="black", label="1:1")

    return ax, (slo, inte, stde_slope, stde_inter, r)


if __name__ == "__main__":

    # ProcessImage object
    pp = ProcessImage()

    # FigureFunctions object
    ff = FigureFunctions()

    # Figure style
    plt.style.use(os.path.abspath(os.path.join(__file__, "../../..")) + "/figurestyle.mplstyle")

    path_i360 = os.path.dirname(os.path.dirname(__file__))

    # if Windows
    volume_path = "/Volumes/MYBOOK/"
    filepath_exp = volume_path + "data-i360/calibrations/linearity/integration-time/"
    filepath_iso = volume_path + "data-i360/calibrations/linearity/iso-gain/"

    # if Mac
    #filepath_exp = "/Volumes/MYBOOK/data-i360/calibrations/linearity/integration-time/"
    #filepath_iso = "/Volumes/MYBOOK/data-i360/calibrations/linearity/iso-gain/"

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
        geocalib = h5py.File(path_i360 + "/geometric-calibration/calibrationfiles/geometric-calibration-water.h5")
        geocalib = geocalib["/lens-close/20200730_112353/"]

        wim = "close"

    elif answer.lower() == "f":

        # Path list
        imlist_exp = pp.imageslist(filepath_exp + "lensfar")
        imlist_iso = pp.imageslist(filepath_iso + "lensfar")
        imlist_exp_bl = pp.imageslist_dark(filepath_exp + "lensfar")
        imlist_iso_bl = pp.imageslist_dark(filepath_iso + "lensfar")

        # Geometric calibration
        geocalib = h5py.File(path_i360 + "/geometric-calibration/calibrationfiles/geometric-calibration-water.h5")
        geocalib = geocalib["/lens-far/20200730_143716/"]

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
    imstack_exp_avg -= bl_expln.astype(float)[None, None, ::4]
    imstack_exp_avg = np.clip(imstack_exp_avg, 0, None)

    imstack_iso_avg -= bl_isoln.astype(float)[None, None, ::4]
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
    fig3, ax3 = plt.subplots(1, 1, figsize=ff.set_size(subplots=(1, 1)), sharex=True)

    tintms = np.unique(exp_expln) * 1000
    iso = np.unique(iso_isoln) * 0.01

    x = np.linspace(tintms.min() * 0.8, tintms.max() * 1.2, 50)
    x_iso = np.linspace(iso.min() * 0.8, iso.max() * 1.2, 50)

    #tx_exp = "$DN_{{i}} = m \cdot t_{{int}} + b$\n$m = ({0:.2f}\pm{1:.2f})$\n$b = ({2:.0f}\pm{3:.0f})$\n$R^{{2}} = {4:.6f}$\n$r={5:.7f}$"
    tx_exp = "$y_{{DN,i}} = m \cdot t_{{int}} + b$\n$m = ({0:.2f}\pm{1:.2f})$\n$b = ({2:.0f}\pm{3:.0f})$\n$R^{{2}} = {4:.6f}$\n$r={5:.7f}$"
    #tx_iso = "$DN_{{i}} = m \cdot ISO \cdot 0.01 + b$\n$m = ({0:.1f}\pm{1:.1f})$\n$b = ({2:.0f}\pm{3:.0f})$\n$R^{{2}} = {4:.6f}$\n$r={5:.5f}$"
    tx_iso = "$y_{{DN,i}} = m \cdot S_{{ISO}} \cdot 0.01 + b$\n$m = ({0:.1f}\pm{1:.1f})$\n$b = ({2:.0f}\pm{3:.0f})$\n$R^{{2}} = {4:.6f}$\n$r={5:.5f}$"
    #tx_iso_green = "$DN_{{i}} = m \cdot ISO \cdot 0.01 + b$\n$m = ({0:.0f}\pm{1:.0f})$\n$b = ({2:.0f}\pm{3:.0f})$\n$R^{{2}} = {4:.6f}$\n$r={5:.5f}$"
    tx_iso_green = "$y_{{DN,i}} = m \cdot S_{{ISO}} \cdot 0.01 + b$\n$m = ({0:.0f}\pm{1:.0f})$\n$b = ({2:.0f}\pm{3:.0f})$\n$R^{{2}} = {4:.6f}$\n$r={5:.5f}$"

    # color prop
    #prop_cycle = plt.rcParams['axes.prop_cycle']
    #col = prop_cycle.by_key()['color']
    col = ['#d95f02', '#1b9e77', '#7570b3']

    # Linestyle
    ls = ["-", "-.", ":"]
    ms = ["o", "s", "^"]
    band_label = {0: 'r', 1: 'g', 2: 'b'}

    for b in range(dn_exp.shape[1]):

        # Figure 1 ___ exp
        # linear regression
        if b == 0 or b == 1:
            tint_reg, dn_exp_reg, noise_exp_reg = tintms[:-2], dn_exp[:-2, b], noise_exp[:-2, b]
        else:
            tint_reg, dn_exp_reg, noise_exp_reg = tintms.copy(), dn_exp[:, b].copy(), noise_exp[:, b].copy()

        ax1[0, b].errorbar(tintms, dn_exp[:, b], yerr=noise_exp[:, b], color=col[b], linestyle="none", marker=ms[b], markersize=3, label="averaged $DN$")
        # Linear regression plot
        pearson_r_texp = stats.pearsonr(tint_reg, dn_exp_reg)
        _, resu_fit = plot_linear_regression(ax1[0, b], tint_reg, dn_exp_reg, tx_exp, ls[b], postext=(0.45, 0.05), color=col[b])
        ax1[0, b].text(0.45, 0.05, tx_exp.format(resu_fit[0], resu_fit[2] * 1.96, resu_fit[1], resu_fit[3] * 1.96, resu_fit[4] ** 2, pearson_r_texp.statistic), transform=ax1[0, b].transAxes, fontsize=6)

        estimates = tint_reg * resu_fit[0] + resu_fit[1]

        delta_slope_exp = resu_fit[2] * stats.t.ppf(0.975, tint_reg.shape[0] - 2)
        delta_intercept_exp = resu_fit[3] * stats.t.ppf(0.975, tint_reg.shape[0] - 2)
        delta_estimates_exp = np.sqrt((tint_reg * delta_slope_exp) ** 2 + (delta_intercept_exp ** 2))

        nl = 100 * (dn_exp_reg - estimates) / estimates
        delta_nl_exp = 100 * np.sqrt(((1/estimates) * noise_exp_reg) ** 2 + ((dn_exp_reg / estimates ** 2) * delta_estimates_exp) ** 2)
        #delta_nl_exp = 100 * np.sqrt(((1 / estimates) * noise_exp[:, b]) ** 2)

        #ax1_twin = ax1[0, b].twinx()
        #ax1_twin.plot(tint_reg, nl, linewidth=0.8, marker=".", color='grey')
        #ax1_twin.set_ylabel('Residuals [%]', color="grey")
        #ax1_twin.tick_params(axis='y', colors='grey')

        # Axe NL
        #ax3.plot(dn_exp_reg, nl, color=col[b], linewidth=0.8, marker=ms[b], markersize=3, label=f'${band_label[b]}$ band, exposure time')
        ax3.errorbar(dn_exp_reg, nl, yerr=delta_nl_exp, color=col[b], linewidth=0.8, marker=ms[b], ms=3, label=f'${band_label[b]}$ band, exposure time')
        # axnl = ax1[0, b].inset_axes([0.1, 0.6, 0.4, 0.3], transform=ax1[0, b].transAxes)
        # axnl.plot(dn_exp_reg, nl, linewidth=0.8, color=col[b], marker=ms[b])
        # axnl.set_xscale('log')
        # axnl.set_ylabel('Linearity [%]', fontsize=3)
        # axnl.set_xlabel("$DN_{i}$ [ADU]", fontsize=3)
        # axnl.set_xticklabels(axnl.get_xticklabels(), fontsize=3)
        # axnl.set_yticklabels(axnl.get_yticklabels(), fontsize=3)

        ax1[0, b].set_xscale("log")
        ax1[0, b].set_yscale("log")

        ax1[0, b].set_xlabel("exposure time $t_{int}$ [ms]")
        ax1[0, b].text(0.02, 0.90, "(" + string.ascii_lowercase[b] + ")", transform=ax1[0, b].transAxes, size=11, weight='bold')

        # Figure 1 ___ iso

        ax1[1, b].errorbar(iso, dn_iso[:, b], yerr=noise_iso[:, b], color=col[b], linestyle="none", marker=ms[b], markersize=3, label="averaged $DN$")
        pearson_r_iso = stats.pearsonr(iso, dn_iso[:, b])
        # Linear regression
        if b == 2:
            _, resu_fit_iso = plot_linear_regression(ax1[1, b], iso[1:], dn_iso[1:, b], tx_iso, ls[b], postext=(0.2, 0.5), color=col[b])
            ax1[1, b].text(0.2, 0.5, tx_iso.format(resu_fit_iso[0],
                                                   resu_fit_iso[2] * 1.96,
                                                   resu_fit_iso[1],
                                                   resu_fit_iso[3] * 1.96,
                                                   resu_fit_iso[4] ** 2,
                                                   pearson_r_iso.statistic), transform=ax1[1, b].transAxes, fontsize=6)

        elif b == 1:
            _, resu_fit_iso = plot_linear_regression(ax1[1, b], iso[1:], dn_iso[1:, b], tx_iso_green, ls[b], postext=(0.38, 0.05), color=col[b])
            ax1[1, b].text(0.35, 0.05, tx_iso_green.format(resu_fit_iso[0],
                                                   resu_fit_iso[2] * 1.96,
                                                   resu_fit_iso[1],
                                                   resu_fit_iso[3] * 1.96,
                                                   resu_fit_iso[4] ** 2,
                                                   pearson_r_iso.statistic), transform=ax1[1, b].transAxes, fontsize=6)
        else:
            _, resu_fit_iso = plot_linear_regression(ax1[1, b], iso[1:], dn_iso[1:, b], tx_iso, ls[b], postext=(0.38, 0.05), color=col[b])
            ax1[1, b].text(0.35, 0.05, tx_iso.format(resu_fit_iso[0],
                                                   resu_fit_iso[2] * 1.96,
                                                   resu_fit_iso[1],
                                                   resu_fit_iso[3] * 1.96,
                                                   resu_fit_iso[4] ** 2,
                                                   pearson_r_iso.statistic), transform=ax1[1, b].transAxes, fontsize=6)

        estimate_iso = iso * resu_fit_iso[0] + resu_fit_iso[1]

        delta_slope_iso = resu_fit_iso[2] * stats.t.ppf(0.975, iso[1:].shape[0] - 2)
        delta_intercept_iso = resu_fit_iso[3] * stats.t.ppf(0.975, iso[1:].shape[0] - 2)
        delta_estimates_iso = np.sqrt((iso * delta_slope_iso) ** 2 + (delta_intercept_iso ** 2))

        nl_iso = 100 * (dn_iso[:, b] - estimate_iso) / estimate_iso
        delta_nl_iso = 100 * np.sqrt(((1/estimate_iso) * noise_iso[:, b]) ** 2 + ((dn_iso[:, b] / estimate_iso ** 2) * delta_estimates_iso) ** 2)

        #ax3.plot(dn_iso[:, b], nl_iso, color=col[b], linestyle='--', linewidth=0.8, marker=ms[b], markersize=3, label=f'${band_label[b]}$ band, ISO gain')
        ax3.errorbar(dn_iso[:, b], nl_iso, yerr=delta_nl_iso, color=col[b], markerfacecolor='none', linestyle='--', linewidth=0.8, marker=ms[b], ms=3, label=f'${band_label[b]}$ band, ISO gain')

        print(f'{band_label[b]} band')
        print('NL exposure time')
        print(nl)
        print(delta_nl_exp)
        print('NL ISO gain')
        print(nl_iso)
        print(delta_nl_iso)

        ax1[1, b].set_xscale("log")
        ax1[1, b].set_yscale("log")

        #ax1[1, b].set_xlabel("$ISO \cdot 0.01$")
        ax1[1, b].set_xlabel("$S_{ISO} \cdot 0.01$")
        ax1[1, b].text(0.02, 0.90, "(" + string.ascii_lowercase[3 + b] + ")", transform=ax1[1, b].transAxes, size=11, weight='bold')

        # Figure 2
        ax2[0, b].errorbar(tintms / tintms[0], dn_exp[:, b] / dn_exp[0, b], linestyle="none", marker=".", label="averaged $DN$")
        plot_linear_regression(ax2[0, b], tint_reg / tint_reg[0], dn_exp_reg / dn_exp_reg[0], tx_exp, "-", postext=(0.45, 0.05), oneone=True)

        ax2[1, b].errorbar(iso / iso[1], dn_iso[:, b] / dn_iso[1, b], linestyle="none", marker=".", label="averaged $DN$")
        plot_linear_regression(ax2[1, b], iso / iso[1], dn_iso[:, b] / dn_iso[1, b], tx_iso, "-", postext=(0.45, 0.05), oneone=True)

        ax2[0, b].set_xscale("log")
        ax2[0, b].set_yscale("log")

        ax2[0, b].set_xlabel("$t_{int} / t_{int, 1}$")
        ax2[0, b].set_ylabel("$DN / DN_{t_{int, 1}}$")

        ax2[1, b].set_xscale("log")
        ax2[1, b].set_yscale("log")

        ax2[1, b].set_xlabel("$ISO / ISO_{2}$")  # DIVIDED BY THE SECOND Measurements!
        ax2[1, b].set_ylabel("$DN / DN_{ISO_{2}}$")

    # Figure 1
    #ax1[0, 0].set_ylabel("$DN_{i}$ [ADU]")
    #ax1[1, 0].set_ylabel("$DN_{i}$ [ADU]")
    ax1[0, 0].set_ylabel("$y_{DN,i}$ [ADU]")
    ax1[1, 0].set_ylabel("$y_{DN,i}$ [ADU]")

    #ax3[0].set_title('Exposure time', fontsize=10)
    #ax3[0].set_xscale('log')
    #ax3[0].set_xlabel("$DN_{i}$ [ADU]")
    #ax3[0].set_ylabel("Linearity [%]")
    #ax3[0].legend(loc='best')

    ax3.set_xscale('log')
    ax3.set_xlabel("$y_{DN,i}$ [ADU]")
    ax3.set_ylabel("Nonlinearity [%]")
    ax3.legend(loc='best')

    #ax3[1].set_title('ISO gain', fontsize=10)
    #ax3[1].set_xscale('log')
    #ax3[1].set_xlabel("$DN_{i}$ [ADU]")
    #ax3[1].set_ylabel("Linearity [%]")
    #ax3[1].legend(loc='best')

    # Saving figures
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    optic_correspondance = {"c": "close", "f": "far"}

    fig1.savefig("figures/linearity-fit-{0}.pdf".format(optic_correspondance[answer.lower()]), format="pdf", dpi=600)
    fig1.savefig("figures/linearity-fit-{0}.png".format(optic_correspondance[answer.lower()]), format="png", dpi=600)
    fig1.savefig("figures/linearity-fit-{0}.jpg".format(optic_correspondance[answer.lower()]), format="jpg", dpi=600)

    if answer.lower() == "c":
        fig3.savefig("figures/S4.pdf", format="pdf", dpi=600)
        fig3.savefig("figures/S4.jpg", format="jpg", dpi=600)
    plt.show()
