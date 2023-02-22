# -*- coding: utf-8 -*-
"""
Script to build roll-off figures.
"""

# Module importation
import string
import h5py
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Other modules
from source.processing import ProcessImage, FigureFunctions
from source.geometric_rolloff import RolloffFunctions


# Classes and functions
def rolloff_fit(angles, rolloff):
    """
    Curve fit for roll-off.

    :param angles:
    :param rolloff:
    :return:
    """

    pp = ProcessImage()

    popt, pcov = curve_fit(pp.rolloff_polynomial, angles, rolloff)
    rsquared, perr = pp.rsquare(pp.rolloff_polynomial, popt, pcov, angles, rolloff)

    return popt, pcov, rsquared, perr


def show_roll_off(data_w, data_a, fig_ax=(None, None)):
    """
    Create graph for roll-off visualization.
    :param data_w:
    :param data_a:
    :return:
    """

    pim = ProcessImage()

    if fig_ax == (None, None):
        fig, ax = plt.subplots(2, 1)
    else:
        fig, ax = fig_ax

    ax[0].text(0.9, 0.9, "(" + string.ascii_lowercase[0] + ")", transform=ax[0].transAxes, size=11, weight='bold')
    ax[1].text(0.9, 0.9, "(" + string.ascii_lowercase[1] + ")", transform=ax[1].transAxes, size=11, weight='bold')
    ax[0].text(0.1, 0.65, "in-air", transform=ax[0].transAxes, size=8)
    ax[1].text(0.1, 0.65, "in-water", transform=ax[1].transAxes, size=8)

    r0deg_w = data_w["roll-off-0degree"][:]
    r90deg_w = data_w["roll-off-90degree"][:]
    r0deg_a = data_a["roll-off-0degree"][:]
    r90deg_a = data_a["roll-off-90degree"][:]

    #marker = ["o", "s", "d"]
    #colo = ['#d62728', '#2ca02c', '#1f77b4']
    lab = ["red: 603 nm", "green: 544 nm", "blue: 484 nm"]
    lab_fit = ["fit red: ", "fit green: ", "fit blue: "]
    lstyle = ["-", "-.", ":"]
    marker = ["o", "s", "^"]
    colo = ['#d95f02', '#1b9e77', '#7570b3']
    th_air = np.linspace(0, 90, 50)
    th_water = np.linspace(0, 75, 50)

    leg_p1, legst_p1 = [], []
    leg_p2, legst_p2 = [], []

    leg_p3, legst_p3 = [], []
    leg_p4, legst_p4 = [], []

    for n in range(r0deg_w.shape[1]):
        # In-air
        all_a_air = np.append(r0deg_a["a"][:, n], r90deg_a["a"][:, n])
        all_rolloff_air = np.append(r0deg_a["DN_avg"][:, n], r90deg_a["DN_avg"][:, n])
        popt_air, pcov_air, rsquared_air, perr_air = rolloff_fit(all_a_air, all_rolloff_air)  # fit 90˚ and 0˚ azimuth

        p1 = ax[0].plot(r0deg_a["a"][:, n], r0deg_a["DN_avg"][:, n], marker=marker[n], markersize=2, linestyle="none", markeredgecolor=colo[n], markerfacecolor="none", alpha=0.9)
        ax[0].plot(r90deg_a["a"][:, n], r90deg_a["DN_avg"][:, n], marker=marker[n], markersize=2, linestyle="none", markeredgecolor=colo[n], markerfacecolor="none", alpha=0.9)
        p2 = ax[0].plot(th_air, pim.rolloff_polynomial(th_air, *popt_air), color=colo[n], linewidth=1, linestyle=lstyle[n], label="Fit")

        print(lab_fit[n] + ": air 70 deg: {0:.4f}".format(pim.rolloff_polynomial(70, *popt_air)))
        print(lab_fit[n] + ": air 80 deg: {0:.4f}".format(pim.rolloff_polynomial(80, *popt_air)))
        print(lab_fit[n] + ": air 90 deg: {0:.4f}".format(pim.rolloff_polynomial(90, *popt_air)))

        # Legend _air
        leg_p1.append(p1[0])
        leg_p2.append(p2[0])
        legst_p1.append(lab[n])
        legst_p2.append(lab_fit[n] + "$a_{0}$ = %.2E $a_{2}$ = %.2E $a_{4}$ = %.2E $a_{6}$ = %.2E  $a_{8}$ = %.2E" % tuple(popt_air))

        # In-water
        all_a_water = np.append(r0deg_w["a"][:, n], r90deg_w["a"][:, n])
        all_rolloff_water = np.append(r0deg_w["DN_avg"][:, n], r90deg_w["DN_avg"][:, n])
        popt_water, pcov_water, rsquared_water, perr_water = rolloff_fit(all_a_water, all_rolloff_water)  # fit 90˚ and 0˚ azimuth

        p3 = ax[1].plot(r0deg_w["a"][:, n], r0deg_w["DN_avg"][:, n], marker=marker[n], markersize=2, linestyle="none", markeredgecolor=colo[n], markerfacecolor="none", alpha=0.9)
        ax[1].plot(r90deg_w["a"][:, n], r90deg_w["DN_avg"][:, n], marker=marker[n], markersize=2, linestyle="none", markeredgecolor=colo[n], markerfacecolor="none", alpha=0.9)
        p4 = ax[1].plot(th_water, pim.rolloff_polynomial(th_water, *popt_water), color=colo[n], linewidth=1, linestyle=lstyle[n], label="Fit")

        print(lab_fit[n] + ": water 70 deg: {0:.4f}".format(pim.rolloff_polynomial(70, *popt_water)))

        # Legend _water
        leg_p3.append(p3[0])
        leg_p4.append(p4[0])
        legst_p3.append(lab[n])
        legst_p4.append(lab_fit[n] + "$a_{0}$ = %.2E $a_{2}$ = %.2E $a_{4}$ = %.2E $a_{6}$ = %.2E  $a_{8}$ = %.2E" % tuple(popt_water))

    ax[0].legend(leg_p1 + leg_p2, legst_p1 + legst_p2, fontsize=6, frameon=False)
    ax[1].legend(leg_p3 + leg_p4, legst_p3 + legst_p4, fontsize=6, frameon=False)

    ax[1].set_xticks(np.arange(0, 100, 10))

    ax[0].set_ylim((0.2, 1.0338603520703975))
    ax[1].set_ylim((0.2, 1.0338603520703975))

    ax[0].set_ylabel(r"$R(\theta)$")
    ax[1].set_xlabel(r"$\theta$ [˚]")
    ax[1].set_ylabel(r"$R(\theta)$")

    fig.tight_layout()
    return fig, ax


def show_roll_off_both_medium(data_w, data_a, fig_ax=(None, None)):
    """

    :param data_w:
    :param data_a:
    :param fig_ax:
    :return:
    """

    pim = ProcessImage()

    if fig_ax == (None, None):
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = fig_ax

    # Data
    r0deg_w = data_w["roll-off-0degree"][:]
    r90deg_w = data_w["roll-off-90degree"][:]
    r0deg_a = data_a["roll-off-0degree"][:]
    r90deg_a = data_a["roll-off-90degree"][:]

    lab = ["red band (603 nm)", "green band (544 nm)", "blue band (484 nm)"]

    lstyle = ["-.", ":", "-"]
    marker_a = ["o", "s", "^"]
    marker_w = [">", "P", "h"]
    colo = ['#d95f02', '#1b9e77', '#7570b3']
    #colo_d = ['#1f77b4', '#17becf']
    colo_d =['#1f78b4', '#a6cee3']
    th_air = np.linspace(0, 90, 50)
    th_water = np.linspace(0, 75, 50)

    for n in range(r0deg_a.shape[1]):
        # In-air
        ax.plot(r0deg_a["a"][:, n], r0deg_a["DN_avg"][:, n], marker=marker_a[n], markersize=2.5, linestyle="none", markeredgecolor=colo_d[0], markerfacecolor="none", alpha=0.9, label=r"air " + lab[n])
        ax.plot(r90deg_a["a"][:, n], r90deg_a["DN_avg"][:, n], marker=marker_a[n], markersize=2.5, linestyle="none", markeredgecolor=colo_d[0], markerfacecolor="none", alpha=0.9)

    for n in range(r0deg_w.shape[1]):
        # In-water
        ax.plot(r0deg_w["a"][:, n], r0deg_w["DN_avg"][:, n], marker=marker_w[n], markersize=2.5, linestyle="none", markeredgecolor=colo_d[1], markerfacecolor="none", alpha=0.9, label=r"water " + lab[n])
        ax.plot(r90deg_w["a"][:, n], r90deg_w["DN_avg"][:, n], marker=marker_w[n], markersize=2.5, linestyle="none", markeredgecolor=colo_d[1], markerfacecolor="none", alpha=0.9)

    for n in range(r0deg_a.shape[1]):
        # In-air
        print("In air " + lab[n])
        all_a_air = np.append(r0deg_a["a"][:, n], r90deg_a["a"][:, n])
        all_rolloff_air = np.append(r0deg_a["DN_avg"][:, n], r90deg_a["DN_avg"][:, n])
        popt_air, pcov_air, rsquared_air, perr_air = rolloff_fit(all_a_air, all_rolloff_air)  # fit 90˚ and 0˚ azimuth
        ax.plot(th_air, pim.rolloff_polynomial(th_air, *popt_air), color=colo_d[0], linewidth=0.9, linestyle=lstyle[n], label="Fit, R-squared: {0:.4f}".format(rsquared_air))

    for n in range(r0deg_w.shape[1]):
        # In-water
        print("In water " + lab[n])
        all_a_water = np.append(r0deg_w["a"][:, n], r90deg_w["a"][:, n])
        all_rolloff_water = np.append(r0deg_w["DN_avg"][:, n], r90deg_w["DN_avg"][:, n])
        popt_water, pcov_water, rsquared_water, perr_water = rolloff_fit(all_a_water, all_rolloff_water)  # fit 90˚ and 0˚ azimuth
        ax.plot(th_water, pim.rolloff_polynomial(th_water, *popt_water), color=colo_d[1], linewidth=0.9, linestyle=lstyle[n], label="Fit, R-squared: {0:.4f}".format(rsquared_water))

    ax.set_xticks(np.arange(0, 100, 10))

    ax.set_ylim((0.2, 1.0338603520703975))
    ax.set_ylim((0.2, 1.0338603520703975))

    ax.set_ylabel(r"$R(\theta)$")
    ax.set_xlabel(r"$\theta$ [˚]")

    ax.legend(fontsize=6, ncol=2, frameon=False)

    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":

    # Figure function
    ff = FigureFunctions()

    # Opening data
    data_air = h5py.File("calibrationfiles/roll-off-data-air.h5")
    data_water = h5py.File("calibrationfiles/roll-off-data-water.h5")

    while True:
        answer = input("Which lens do you want to analyze? (c/f): ")
        if answer.lower() in ["c", "f"]:
            break

    dict_wlens = {"c": "lens-close", "f": "lens-far"}
    wlens = dict_wlens[answer.lower()]

    data_air = data_air[wlens]["20170102"]
    if wlens == "lens-close":
        data_water = data_water[wlens]["20190501"]
    else:
        data_water = data_water[wlens]["20190503"]

    # Get band-averaged relative errors
    rel_err_water0 = (data_water["roll-off-0degree"][:]["DN_std"] / data_water["roll-off-0degree"][:]["DN_avg"]) * 100
    rel_err_water90 = (data_water["roll-off-90degree"][:]["DN_std"] / data_water["roll-off-90degree"][:]["DN_avg"]) * 100
    rel_err_water = np.append(rel_err_water0, rel_err_water90, axis=0)
    angtot = np.append(data_water["roll-off-0degree"][:]["a"], data_water["roll-off-90degree"][:]["a"], axis=0)

    print(rel_err_water.mean(axis=0))

    rel_err_air0 = (data_air["roll-off-0degree"][:]["DN_std"] / data_air["roll-off-0degree"][:]["DN_avg"]) * 100
    rel_err_air90 = (data_air["roll-off-90degree"][:]["DN_std"] / data_air["roll-off-90degree"][:]["DN_avg"]) * 100
    rel_err_air = np.append(rel_err_air0, rel_err_air90, axis=0)

    print(rel_err_air.mean(axis=0))

    # Figures
    plt.style.use("../../figurestyle.mplstyle")

    #fig1, ax1 = plt.subplots(2, 1, sharex=True, figsize=ff.set_size(subplots=(1, 1), height_ratio=0.9))
    #fig1 = plt.figure(figsize=ff.set_size(subplots=(2, 1), fraction=0.7))
    fig1, ax1 = plt.subplots(2, 1, sharex=True, figsize=ff.set_size(fraction=0.75, height_ratio=1))
    fig2, ax2 = plt.subplots(1, 1, figsize=ff.set_size(fraction=0.7, height_ratio=0.7))

    # Plot figure
    fig1, ax1 = show_roll_off(data_water, data_air, fig_ax=(fig1, ax1))
    fig2, ax2 = show_roll_off_both_medium(data_water, data_air, fig_ax=(fig2, ax2))

    # Saving figure
    fig1.savefig("figures/roll-off-air-water-{0}.png".format(wlens), format="png", dpi=600)
    fig2.savefig("figures/roll-off-air-water-sf-{0}.png".format(wlens), format="png", dpi=600)
    fig2.savefig("figures/roll-off-air-water-sf-{0}.pdf".format(wlens), format="pdf", dpi=600)

    plt.show()
