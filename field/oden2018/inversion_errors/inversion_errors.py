# -*- coding : utf-8 -*-
"""

"""
# Module importation
import os
import string
import pandas
import pickle
import numpy as np
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt

# Other modules
from field.oden2018.oden_dort_vs_hl import load_zenith_radiance


# Classes and functions
def load_depths(path):
    """
    Get depths according to hermes.pickle file.

    :param path: absolute path (str)
    :return: depths (list)
    """
    with open(path + "/hermes.pickle", 'rb') as handle:
        hermes = pickle.load(handle)

    return hermes['zetanom']


def show_iops(path_secret, path_fitted):
    # Load data
    #secret_data = pandas.read_csv(os.path.join(path_secret, "eudos_iops.csv"))
    #fit_data = pandas.read_csv(os.path.join(path_fitted, "eudos_iops.csv"))

    # Figure initialization
    plt.style.use(os.path.abspath(os.path.join(__file__, "../../../..")) + "/figurestyle.mplstyle")
    fig, ax = plt.subplots(1, 5, figsize=(6.6929, 6.6929 * 3/4 * 0.7), sharey=True)

    ax, er = plot_all_inv_data(ax, path_secret, path_fitted)

    # Invert yaxis
    ax[0].invert_yaxis()

    ax[1].set_xscale("log")
    ax[3].set_xscale("log")

    # X ticks and lims
    ax[0].set_xticks(np.arange(0, 1.1, 0.1))
    ax[0].set_xlim((-0.02, 0.5))
    ax[1].set_xticks(np.logspace(-1, 4, 6))
    ax[1].set_xlim((0.05, 2000))
    ax[2].set_xticks(np.arange(0.7, 1.05, 0.05))
    ax[2].set_xlim((0.83, 1.02))
    ax[3].set_xticks(np.logspace(-3, 3, 7))
    ax[3].set_xlim((0.007, 300))
    ax[4].set_xticks(np.arange(-0.5, 1.5, 0.5))
    ax[4].set_xlim((-0.01, 1.01))

    # Y X labels
    ax[0].set_ylabel("Depths [cm]")

    ax[0].set_xlabel("$a~\mathrm{[m^{-1}]}$")
    ax[1].set_xlabel("$b~\mathrm{[m^{-1}]}$")
    ax[2].set_xlabel("$g$")
    ax[3].set_xlabel("$b\cdot(1-g)~\mathrm{[m^{-1}]}$")
    ax[4].set_xlabel("$S$")

    # Letters
    ax[0].text(-0.05, 1.02, "(" + string.ascii_lowercase[0] + ")", transform=ax[0].transAxes, size=9, weight='bold')
    ax[1].text(-0.05, 1.02, "(" + string.ascii_lowercase[1] + ")", transform=ax[1].transAxes, size=9, weight='bold')
    ax[2].text(-0.05, 1.02, "(" + string.ascii_lowercase[2] + ")", transform=ax[2].transAxes, size=9, weight='bold')
    ax[3].text(-0.05, 1.02, "(" + string.ascii_lowercase[3] + ")", transform=ax[3].transAxes, size=9, weight='bold')
    ax[4].text(-0.05, 1.02, "(" + string.ascii_lowercase[4] + ")", transform=ax[4].transAxes, size=9, weight='bold')

    ax[0].legend(loc="best", frameon=False, fontsize=7)
    ax[1].legend(loc="best", frameon=False, fontsize=7)
    ax[2].legend(loc="best", frameon=False, fontsize=7)
    ax[3].legend(loc="best", frameon=False, fontsize=7)
    ax[4].legend(loc="best", frameon=False, fontsize=7)

    # Text
    ax[0].text(0.35, 0.15, "$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %"
                           "\n$\overline{{∣e_{{g}}∣}}= {1:.1f}$ %"
                           "\n$\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(er[0][0], er[0][1], er[0][2]), transform=ax[0].transAxes, fontsize=7)
    ax[1].text(0.35, 0.15, "$\overline{{∣e∣}}= {0:.1f}$ %".format(er[1]), transform=ax[1].transAxes, fontsize=7)
    ax[2].text(0.4, 0.15, "$\overline{{∣e∣}}= {0:.1f}$ %".format(er[2]), transform=ax[2].transAxes, fontsize=7)
    ax[3].text(0.4, 0.15, "$\overline{{∣e∣}}= {0:.1f}$ %".format(er[3]), transform=ax[3].transAxes, fontsize=7)
    ax[4].text(0.2, 0.15, "$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %"
                           "\n$\overline{{∣e_{{g}}∣}}= {1:.1f}$ %"
                           "\n$\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(er[4][0], er[4][1], er[4][2]), transform=ax[4].transAxes, fontsize=7)

    # Layout
    #fig.tight_layout()
    fig.subplots_adjust(left=0.09,
                        bottom=0.15,
                        right=0.98,
                        top=0.95,
                        wspace=0.13,
                        hspace=0.35)

    return fig, ax


def show_inversion_errors(path_secret, path_fitted, S=True):
    """

    :param depths:
    :param path_secret:
    :param path_fitted:
    :param S:
    :return:
    """
    # Load data
    secret_data = pandas.read_csv(os.path.join(path_secret, "eudos_iops.csv"))
    fit_data = pandas.read_csv(os.path.join(path_fitted, "eudos_iops.csv"))

    # Figure initialization
    plt.style.use(os.path.abspath(os.path.join(__file__, "../../../..")) + "/figurestyle.mplstyle")
    if S:
        fig, ax = plt.subplots(2, 5, figsize=(6.6929, 6.6929 * 3/4), sharey=True)
    else:
        fig, ax = plt.subplots(2, 4, figsize=(6.6929, 6.6929 * 3 / 4), sharey=True)

    wl = [600.0, 540.0, 480.0]  # wavelength in nm
    real_wl = {600.0: "603 nm", 540.0: "544 nm", 480.0: "484 nm"}
    depths_meter = secret_data["depths"]

    CMAR = matplotlib.cm.get_cmap("Reds", 10 + 1)
    CMAG = matplotlib.cm.get_cmap("Greens", 10 + 1)
    CMAB = matplotlib.cm.get_cmap("Blues", 10 + 1)
    col = [CMAR(7), CMAG(7), CMAB(7)]

    # Scattering secret
    b_secret = secret_data["b_" + str(wl[0])]  # scattering coefficient
    g_secret = secret_data["g"]  # anisotropy coefficient of the phase function
    beff_secret = b_secret * (1 - g_secret)

    ax[0, 1].plot(b_secret[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color="k", label="unknown")
    ax[0, 2].plot(g_secret[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color="k", label="unknown")
    ax[0, 3].plot(beff_secret[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color="k", label="unknown")

    # Scattering fitted
    b_fitted = fit_data["b_" + str(wl[0])]  # scattering coefficient
    g_fitted = fit_data["g"]  # anisotropy coefficient of the phase function
    beff_fitted = b_fitted * (1 - g_fitted)

    ax[0, 1].plot(b_fitted[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color="k", label="fitted")
    ax[0, 2].plot(g_fitted[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color="k", label="fitted")
    ax[0, 3].plot(beff_fitted[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color="k", label="fitted")

    # Errors
    b_err = 100 * (b_fitted - b_secret) / b_secret
    g_err = 100 * (g_fitted - g_secret) / g_secret
    beff_err = 100 * (beff_fitted - beff_secret) / beff_secret

    mask_2m = depths_meter[1:] <= 2.0

    b_err_avg = np.abs(b_err[1:][mask_2m]).mean()
    g_err_avg = np.abs(g_err[1:][mask_2m]).mean()
    beff_err_avg = np.abs(beff_err[1:][mask_2m]).mean()

    ax[1, 1].plot(b_err[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color="k")
    ax[1, 2].plot(g_err[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color="k")
    ax[1, 3].plot(beff_err[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color="k")

    # Loop through the different wavelengths
    ae_list, se_list = [], []

    for i, w in enumerate(wl):

        # IOPs
        a_secret = secret_data["a_" + str(w)]  # absorption coefficient
        a_fitted = fit_data["a_" + str(w)]

        # S param
        S_secret = (1 + ((b_secret/a_secret) * (1 - g_secret))) ** (-1/2)
        S_fitted = (1 + ((b_fitted/a_fitted) * (1 - g_fitted))) ** (-1/2)

        # Errors
        a_err = 100 * (a_fitted - a_secret) / a_secret
        S_err = 100 * (S_fitted - S_secret) / S_secret

        # Plot
        ax[0, 0].plot(a_secret[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color=col[i])
        ax[0, 0].plot(a_fitted[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color=col[i], label=real_wl[w])
        if S:
            ax[0, 4].plot(S_secret[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color=col[i])
            ax[0, 4].plot(S_fitted[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color=col[i])
            ax[1, 4].plot(S_err[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color=col[i])

            se_list.append(np.abs(S_err[1:][mask_2m]).mean())

        ae_list.append(np.abs(a_err[1:][mask_2m]).mean())

        ax[1, 0].plot(a_err[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color=col[i])

    # Invert yaxis
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_ylabel("Depths [cm]")
    ax[1, 0].set_ylabel("Depths [cm]")

    ax[0, 0].set_xlabel("$a~\mathrm{[m^{-1}]}$")
    ax[0, 1].set_xlabel("$b~\mathrm{[m^{-1}]}$")
    ax[0, 2].set_xlabel("$g$")
    ax[0, 3].set_xlabel("$b\cdot(1-g)~\mathrm{[m^{-1}]}$")

    ax[1, 0].set_xlabel("errors [%]")
    ax[1, 1].set_xlabel("errors [%]")
    ax[1, 2].set_xlabel("errors [%]")
    ax[1, 3].set_xlabel("errors [%]")

    # Letters
    ax[0, 0].text(-0.05, 1.02, "(" + string.ascii_lowercase[0] + ")", transform=ax[0, 0].transAxes, size=9, weight='bold')
    ax[0, 1].text(-0.05, 1.02, "(" + string.ascii_lowercase[1] + ")", transform=ax[0, 1].transAxes, size=9, weight='bold')
    ax[0, 2].text(-0.05, 1.02, "(" + string.ascii_lowercase[2] + ")", transform=ax[0, 2].transAxes, size=9, weight='bold')
    ax[0, 3].text(-0.05, 1.02, "(" + string.ascii_lowercase[3] + ")", transform=ax[0, 3].transAxes, size=9, weight='bold')
    if S:
        ax[0, 4].text(-0.05, 1.02, "(" + string.ascii_lowercase[4] + ")", transform=ax[0, 4].transAxes, size=9, weight='bold')
        ax[1, 0].text(-0.05, 1.02, "(" + string.ascii_lowercase[5] + ")", transform=ax[1, 0].transAxes, size=9, weight='bold')
        ax[1, 1].text(-0.05, 1.02, "(" + string.ascii_lowercase[6] + ")", transform=ax[1, 1].transAxes, size=9, weight='bold')
        ax[1, 2].text(-0.05, 1.02, "(" + string.ascii_lowercase[7] + ")", transform=ax[1, 2].transAxes, size=9, weight='bold')
        ax[1, 3].text(-0.05, 1.02, "(" + string.ascii_lowercase[8] + ")", transform=ax[1, 3].transAxes, size=9, weight='bold')
        ax[1, 4].text(-0.05, 1.02, "(" + string.ascii_lowercase[9] + ")", transform=ax[1, 4].transAxes, size=9, weight='bold')

        ax[0, 4].set_xlabel("$S$")
        ax[1, 4].set_xlabel("errors [%]")

    else:
        ax[1, 0].text(-0.05, 1.02, "(" + string.ascii_lowercase[4] + ")", transform=ax[1, 0].transAxes, size=9, weight='bold')
        ax[1, 1].text(-0.05, 1.02, "(" + string.ascii_lowercase[5] + ")", transform=ax[1, 1].transAxes, size=9, weight='bold')
        ax[1, 2].text(-0.05, 1.02, "(" + string.ascii_lowercase[6] + ")", transform=ax[1, 2].transAxes, size=9, weight='bold')
        ax[1, 3].text(-0.05, 1.02, "(" + string.ascii_lowercase[7] + ")", transform=ax[1, 3].transAxes, size=9, weight='bold')

    fig.tight_layout()

    return fig, ax, [ae_list, b_err_avg, g_err_avg, beff_err_avg, se_list]


def figure_inversion_err(p_secret, p_fitted, p_secret_noise, p_fitted_noise):
    """
    """

    # Figure initialization
    plt.style.use(os.path.abspath(os.path.join(__file__, "../../../..")) + "/figurestyle.mplstyle")

    fig, ax = plt.subplots(2, 5, figsize=(6.6929, 6.6929 * 3/4), sharey=True)

    ax[0, :], err_nonoise = plot_all_inv_data(ax[0, :], p_secret, p_fitted)
    ax[1, :], err_noise = plot_all_inv_data(ax[1, :], p_secret_noise, p_fitted_noise)

    ax[0, 1].set_xscale("log")
    ax[1, 1].set_xscale("log")

    ax[0, 3].set_xscale("log")
    ax[1, 3].set_xscale("log")

    # Xticks and Xlims
    ax[0, 0].set_xticks(np.arange(0, 1.1, 0.1))
    ax[0, 0].set_xlim((-0.02, 0.42))
    ax[0, 1].set_xticks(np.logspace(-1, 3, 5))
    ax[0, 1].set_xlim((0.09, 1100))
    ax[0, 2].set_xticks(np.arange(0.7, 1.05, 0.05))
    ax[0, 2].set_xlim((0.84, 1.01))
    ax[0, 3].set_xticks(np.logspace(-3, 3, 7))
    ax[0, 3].set_xlim((0.015, 200))
    ax[0, 4].set_xticks(np.arange(-0.5, 1.5, 0.5))
    ax[0, 4].set_xlim((-0.01, 1.01))

    ax[1, 0].set_xlim((0.01, 0.31))
    ax[1, 1].set_xticks(np.logspace(-1, 3, 5))
    ax[1, 1].set_xlim((0.06, 1400))
    ax[1, 2].set_xticks(np.arange(0.7, 1.1, 0.1))
    ax[1, 2].set_xlim((0.77, 1.01))
    ax[1, 3].set_xticks(np.logspace(-3, 3, 7))
    ax[1, 3].set_xlim((0.007, 200))
    ax[1, 4].set_xticks(np.arange(-0.5, 1.5, 0.5))
    ax[1, 4].set_xlim((-0.01, 1.01))

    ax[0, 0].legend(loc="best", frameon=False, fontsize=5)
    ax[0, 1].legend(loc="best", frameon=False, fontsize=5)
    ax[0, 2].legend(loc="best", frameon=False, fontsize=5)
    ax[0, 3].legend(loc="best", frameon=False, fontsize=5)
    ax[0, 4].legend(loc="best", frameon=False, fontsize=5)

    ax[0, 0].set_xlabel("$a~\mathrm{[m^{-1}]}$")
    ax[0, 1].set_xlabel("$b~\mathrm{[m^{-1}]}$")
    ax[0, 2].set_xlabel("$g$")
    ax[0, 3].set_xlabel("$b\cdot(1-g)~\mathrm{[m^{-1}]}$")
    ax[0, 4].set_xlabel("$S$")

    ax[0, 0].invert_yaxis()
    ax[0, 0].set_ylabel("Depths [cm]")
    ax[1, 0].set_ylabel("Depths [cm]")

    ax[1, 0].legend(loc="best", frameon=False, fontsize=5)
    ax[1, 1].legend(loc="best", frameon=False, fontsize=5)
    ax[1, 2].legend(loc="best", frameon=False, fontsize=5)
    ax[1, 3].legend(loc="best", frameon=False, fontsize=5)
    ax[1, 4].legend(loc="best", frameon=False, fontsize=5)

    ax[1, 0].set_xlabel("$a~\mathrm{[m^{-1}]}$")
    ax[1, 1].set_xlabel("$b~\mathrm{[m^{-1}]}$")
    ax[1, 2].set_xlabel("$g$")
    ax[1, 3].set_xlabel("$b\cdot(1-g)~\mathrm{[m^{-1}]}$")
    ax[1, 4].set_xlabel("$S$")

    # Text
    ax[0, 0].text(0.01, 0.43, "$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %"
                              "\n$\overline{{∣e_{{g}}∣}}= {1:.1f}$ %"
                              "\n$\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(err_nonoise[0][0], err_nonoise[0][1], err_nonoise[0][2]), transform=ax[0, 0].transAxes, fontsize=5)
    ax[0, 1].text(0.03, 0.7, "$\overline{{∣e∣}}= {0:.1f}$ %".format(err_nonoise[1]), transform=ax[0, 1].transAxes, fontsize=6)
    ax[0, 2].text(0.03, 0.7, "$\overline{{∣e∣}}= {0:.1f}$ %".format(err_nonoise[2]), transform=ax[0, 2].transAxes, fontsize=6)
    ax[0, 3].text(0.4, 0.2, "$\overline{{∣e∣}}= {0:.1f}$ %".format(err_nonoise[3]), transform=ax[0, 3].transAxes, fontsize=6)
    ax[0, 4].text(0.01, 0.43, "$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %"
                              "\n$\overline{{∣e_{{g}}∣}}= {1:.1f}$ %"
                              "\n$\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(err_nonoise[4][0], err_nonoise[4][1], err_nonoise[4][2]), transform=ax[0, 4].transAxes, fontsize=5)

    ax[1, 0].text(0.55, 0.43, "$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %"
                              "\n$\overline{{∣e_{{g}}∣}}= {1:.1f}$ %"
                              "\n$\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(err_noise[0][0], err_noise[0][1], err_noise[0][2]), transform=ax[1, 0].transAxes, fontsize=5)
    ax[1, 1].text(0.03, 0.7, "$\overline{{∣e∣}}= {0:.1f}$ %".format(err_noise[1]), transform=ax[1, 1].transAxes, size=6)
    ax[1, 2].text(0.03, 0.7, "$\overline{{∣e∣}}= {0:.1f}$ %".format(err_noise[2]), transform=ax[1, 2].transAxes, size=6)
    ax[1, 3].text(0.4, 0.2, "$\overline{{∣e∣}}= {0:.1f}$ %".format(err_noise[3]), transform=ax[1, 3].transAxes, size=6)
    ax[1, 4].text(0.1, 0.09, "$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %"
                              "\n$\overline{{∣e_{{g}}∣}}= {1:.1f}$ %"
                              "\n$\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(err_noise[4][0], err_noise[4][1], err_noise[4][2]), transform=ax[1, 4].transAxes, fontsize=5)

    # Titles
    #ax[0, 0].set_title("$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %, $\overline{{∣e_{{g}}∣}}= {1:.1f}$ %, $\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(err_nonoise[0][0], err_nonoise[0][1], err_nonoise[0][2]), fontsize=6)
    #ax[0, 4].set_title("$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %, $\overline{{∣e_{{g}}∣}}= {1:.1f}$ %, $\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(err_nonoise[4][0], err_nonoise[4][1], err_nonoise[4][2]), fontsize=6)

    #ax[1, 0].set_title("$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %, $\overline{{∣e_{{g}}∣}}= {1:.1f}$ %, $\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(err_noise[0][0], err_noise[0][1], err_noise[0][2]), fontsize=6)
    #ax[1, 4].set_title("$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %, $\overline{{∣e_{{g}}∣}}= {1:.1f}$ %, $\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(err_noise[4][0], err_noise[4][1], err_noise[4][2]), fontsize=6)

    #fig.tight_layout()

    # Letters
    ax[0, 0].text(-0.05, 1.01, "(" + string.ascii_lowercase[0] + ")", transform=ax[0, 0].transAxes, size=9, weight='bold')
    ax[0, 1].text(-0.05, 1.01, "(" + string.ascii_lowercase[1] + ")", transform=ax[0, 1].transAxes, size=9, weight='bold')
    ax[0, 2].text(-0.05, 1.01, "(" + string.ascii_lowercase[2] + ")", transform=ax[0, 2].transAxes, size=9, weight='bold')
    ax[0, 3].text(-0.05, 1.01, "(" + string.ascii_lowercase[3] + ")", transform=ax[0, 3].transAxes, size=9, weight='bold')
    ax[0, 4].text(-0.05, 1.01, "(" + string.ascii_lowercase[4] + ")", transform=ax[0, 4].transAxes, size=9, weight='bold')

    ax[1, 0].text(-0.05, 1.01, "(" + string.ascii_lowercase[5] + ")", transform=ax[1, 0].transAxes, size=9, weight='bold')
    ax[1, 1].text(-0.05, 1.01, "(" + string.ascii_lowercase[6] + ")", transform=ax[1, 1].transAxes, size=9, weight='bold')
    ax[1, 2].text(-0.05, 1.01, "(" + string.ascii_lowercase[7] + ")", transform=ax[1, 2].transAxes, size=9, weight='bold')
    ax[1, 3].text(-0.05, 1.01, "(" + string.ascii_lowercase[8] + ")", transform=ax[1, 3].transAxes, size=9, weight='bold')
    ax[1, 4].text(-0.05, 1.01, "(" + string.ascii_lowercase[9] + ")", transform=ax[1, 4].transAxes, size=9, weight='bold')

    fig.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.98,
                        top=0.97,
                        wspace=0.13,
                        hspace=0.35)

    return fig, ax


def plot_all_inv_data(ax, path_secret, path_fitted):
    """

    :param ax:
    :param path_secret:
    :param path_fitted:
    :return:
    """

    # Load data
    secret_data = pandas.read_csv(os.path.join(path_secret, "eudos_iops.csv"))
    fit_data = pandas.read_csv(os.path.join(path_fitted, "eudos_iops.csv"))

    wl = [600.0, 540.0, 480.0]  # wavelength in nm
    real_wl = {600.0: "603 nm", 540.0: "544 nm", 480.0: "484 nm"}
    depths_meter = secret_data["depths"]
    label_wl = {600.0: "r", 540.0: "g", 480.0: "b"}

    # Scattering secret
    b_secret = secret_data["b_" + str(wl[0])]  # scattering coefficient
    g_secret = secret_data["g"]  # anisotropy coefficient of the phase function
    beff_secret = b_secret * (1 - g_secret)

    ax[1].plot(b_secret[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color="k", label="unknown")
    ax[2].plot(g_secret[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color="k", label="unknown")
    ax[3].plot(beff_secret[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color="k", label="unknown")

    # Scattering fitted
    b_fitted = fit_data["b_" + str(wl[0])]  # scattering coefficient
    g_fitted = fit_data["g"]  # anisotropy coefficient of the phase function
    beff_fitted = b_fitted * (1 - g_fitted)

    ax[1].plot(b_fitted[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color="k", label="inverted")
    ax[2].plot(g_fitted[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color="k", label="inverted")
    ax[3].plot(beff_fitted[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color="k", label="inverted")

    # Errors
    b_err = 100 * (b_fitted - b_secret) / b_secret
    g_err = 100 * (g_fitted - g_secret) / g_secret
    beff_err = 100 * (beff_fitted - beff_secret) / beff_secret

    mask_2m = depths_meter[1:] <= 2.0

    b_err_avg = np.abs(b_err[1:][mask_2m]).mean()
    g_err_avg = np.abs(g_err[1:][mask_2m]).mean()
    beff_err_avg = np.abs(beff_err[1:][mask_2m]).mean()

    # Loop through the different wavelengths
    ae_list, se_list = [], []

    CMAR = matplotlib.cm.get_cmap("Reds", 10 + 1)
    CMAG = matplotlib.cm.get_cmap("Greens", 10 + 1)
    CMAB = matplotlib.cm.get_cmap("Blues", 10 + 1)
    col = [CMAR(7), CMAG(7), CMAB(7)]

    for i, w in enumerate(wl):

        # IOPs
        a_secret = secret_data["a_" + str(w)]  # absorption coefficient
        a_fitted = fit_data["a_" + str(w)]

        # S param
        S_secret = (1 + ((b_secret/a_secret) * (1 - g_secret))) ** (-1/2)
        S_fitted = (1 + ((b_fitted/a_fitted) * (1 - g_fitted))) ** (-1/2)

        # Errors
        a_err = 100 * (a_fitted - a_secret) / a_secret
        S_err = 100 * (S_fitted - S_secret) / S_secret

        # Plot
        # Dummy
        if i == 0:
            ax[0].plot(np.array([]), np.array([]), linestyle="-", linewidth=0.9, color="k", label="unknown")
            ax[0].plot(np.array([]), np.array([]), linestyle="-.", linewidth=0.9, color="k", label="inverted")

            ax[4].plot(np.array([]), np.array([]), linestyle="-", linewidth=0.9, color="k", label="unknown")
            ax[4].plot(np.array([]), np.array([]), linestyle="-.", linewidth=0.9, color="k", label="inverted")

        ax[0].plot(a_secret[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color=col[i])
        ax[0].plot(a_fitted[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color=col[i])
        #ax[0].plot(a_fitted[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color=col[i], label="$\overline{{∣e_{0}∣}}= {1:.1f}$ %".format(label_wl[w], np.abs(a_err[1:][mask_2m]).mean()))
        #ax[0].plot(a_fitted[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color=col[i], label=real_wl[w])

        ax[4].plot(S_secret[1:], depths_meter[1:] * 100, linestyle="-", linewidth=0.9, color=col[i])
        ax[4].plot(S_fitted[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color=col[i])
        #ax[4].plot(S_fitted[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color=col[i], label="$\overline{{∣e_{0}∣}}= {1:.1f}$ %".format(label_wl[w], np.abs(S_err[1:][mask_2m]).mean()))
        #ax[4].plot(S_fitted[1:], depths_meter[1:] * 100, linestyle="-.", linewidth=0.9, color=col[i], label=real_wl[w])

        se_list.append(np.abs(S_err[1:][mask_2m]).mean())
        ae_list.append(np.abs(a_err[1:][mask_2m]).mean())

    return ax, [ae_list, b_err_avg, g_err_avg, beff_err_avg, se_list]


if __name__ == "__main__":

    # Open IOPS
    # Path of code
    path_code = os.path.dirname(os.path.abspath(__file__))
    # Path of secret data
    rel_path_sdata1 = r"data/inversion_errors/fit_errors_1/secret_iops"  # difficult fit
    rel_path_sdata = r"data/inversion_errors/fit_errors_2/secret_iops"
    path_sdata1 = os.path.join(os.path.dirname(path_code), rel_path_sdata1)
    path_sdata = os.path.join(os.path.dirname(path_code), rel_path_sdata)

    # Path of fitted data
    rel_path_fdata1 = r"data/inversion_errors/fit_errors_1/secret_irradiance_fit"  # difficult fit
    rel_path_fdata = r"data/inversion_errors/fit_errors_2/secret_irradiance_fit"
    path_fdata1 = os.path.join(os.path.dirname(path_code), rel_path_fdata1)
    path_fdata = os.path.join(os.path.dirname(path_code), rel_path_fdata)

    # Inversion error with noise
    rel_path_sdata_noise = r"data/inversion_errors/fit_errors_plus_noise_2/original_files"
    rel_path_fdata_noise = r"data/inversion_errors/fit_errors_plus_noise_2/fit_files"

    path_sdata_noise = os.path.join(os.path.dirname(path_code),  rel_path_sdata_noise)
    path_fdata_noise = os.path.join(os.path.dirname(path_code),  rel_path_fdata_noise)

    # Load depths
    depths_sdata = load_depths(path_sdata)
    depths_fdata = load_depths(path_fdata)

    fig1, ax1, avg_errors = show_inversion_errors(path_sdata, path_fdata, S=True)

    # Set xlim and xticks
    # First row
    ax1[0, 0].set_xticks(np.arange(-0.2, 0.8, 0.2))
    ax1[0, 0].set_xlim((-0.01, 0.5))

    ax1[0, 1].set_xscale("log")
    ax1[0, 1].set_xticks(np.logspace(-1, 3, 3))

    ax1[0, 2].set_xticks(np.arange(0.7, 1.05, 0.05))
    ax1[0, 2].set_xlim((0.84, 1.01))

    ax1[0, 3].set_xscale("log")
    ax1[0, 3].set_xticks(np.logspace(-3, 3, 7))
    ax1[0, 3].set_xlim((0.007, 300))

    # Second row
    #ax1[1, 0].set_xticks(np.arange(-0.2, 0.8, 0.2))
    ax1[1, 0].set_xlim((-95, 35))

    # Add average errors

    ax1[1, 1].text(0.3, 0.1, "$\overline{{∣e∣}}= {0:.1f}$ %".format(avg_errors[1]), transform=ax1[1, 1].transAxes, size=6)
    ax1[1, 2].text(0.3, 0.1, "$\overline{{∣e∣}}= {0:.1f}$ %".format(avg_errors[2]), transform=ax1[1, 2].transAxes, size=6)
    ax1[1, 3].text(0.35, 0.1, "$\overline{{∣e∣}}= {0:.1f}$ %".format(avg_errors[3]), transform=ax1[1, 3].transAxes, size=6)

    if ax1.shape[1] == 5:
        ax1[1, 0].text(0.01, 0.05,
                       "$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %\n"
                       "$\overline{{∣e_{{g}}∣}}= {1:.1f}$ %\n"
                       "$\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(avg_errors[0][0],
                                                                    avg_errors[0][1],
                                                                    avg_errors[0][2]), transform=ax1[1, 0].transAxes,
                       size=6)
        ax1[1, 4].text(0.015, 0.05,
                    "$\overline{{∣e_{{r}}∣}}= {0:.1f}$ %\n"
                    "$\overline{{∣e_{{g}}∣}}= {1:.1f}$ %\n"
                    "$\overline{{∣e_{{b}}∣}}= {2:.1f}$ %".format(avg_errors[4][0],
                                                          avg_errors[4][1],
                                                          avg_errors[4][2]), transform=ax1[1, 4].transAxes, size=6)

        ax1[0, 4].set_xticks(np.arange(-0.5, 1.5, 0.5))
        ax1[0, 4].set_xlim((-0.01, 1.01))

    # Set legends
    ax1[0, 0].legend(loc="best", frameon=False, fontsize=7)
    ax1[0, 1].legend(loc="best", frameon=False, fontsize=7)

    fig1.subplots_adjust(wspace=0.15)

    # Figure 2
    fig2, ax2 = figure_inversion_err(path_sdata, path_fdata, path_sdata_noise, path_fdata_noise)

    # Figure 3
    fig3, ax3 = show_iops(path_sdata1, path_fdata1)

    fig1.savefig("../figures/inversion_error.pdf", format="pdf", dpi=600)
    fig1.savefig("../figures/inversion_error.png", format="png", dpi=600)

    fig2.savefig("../figures/inv_errors_wn_noise.pdf", format="pdf", dpi=600)
    fig2.savefig("../figures/inv_errors_wn_noise.png", format="png", dpi=600)

    fig3.savefig("../figures/FigS2.pdf", format="pdf", dpi=600)
    fig3.savefig("../figures/FigS2.png", format="png", dpi=600)
