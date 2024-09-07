# -*- coding: utf-8 -*-
"""
File to construct the relative spectral response figure.
"""

# Module importation
import os
import h5py
import scipy
import numpy as np
import matplotlib.pyplot as plt

# Other module importation
from source.processing import FigureFunctions


# Functions
def rsr_statistics_simps(wavelength, rsr, verbose=True):
    """
    Printing spectral response statistics.
    :param wavelength:
    :param rsr:
    :return:
    """
    eff_bw_array = np.zeros(rsr.shape[1])
    eff_wl_array = eff_bw_array.copy()
    max_wl_array = eff_wl_array.copy()

    for band in range(rsr.shape[1]):
        eff_bw_array[band] = scipy.integrate.simpson(rsr[:, band], x=wavelength)
        eff_wl_array[band] = scipy.integrate.simpson(rsr[:, band] * wavelength, x=wavelength) / eff_bw_array[band]
        max_wl_array[band] = wavelength[np.argmax(rsr[:, band])]

        if verbose:
            print("Band no. {0} statistics".format(band))
            print("Effective bw: {0:.4f}, effective wl: {1:.4f}, maximum wl: {2:.4f}". format(eff_bw_array[band],
                                                                                              eff_wl_array[band],
                                                                                              max_wl_array[band]))

    return eff_wl_array, eff_bw_array, max_wl_array


def rsr_statistics(wavelength, rsr, verbose=True):
    """
    Printing spectral response statistics.
    :param wavelength:
    :param rsr:
    :return:
    """
    eff_bw_array = np.zeros(rsr.shape[1])
    eff_wl_array = eff_bw_array.copy()
    max_wl_array = eff_wl_array.copy()

    for band in range(rsr.shape[1]):
        eff_bw_array[band] = np.trapz(rsr[:, band], x=wavelength)
        eff_wl_array[band] = np.trapz(rsr[:, band] * wavelength, x=wavelength) / eff_bw_array[band]
        max_wl_array[band] = wavelength[np.argmax(rsr[:, band])]

        if verbose:
            print("Band no. {0} statistics".format(band))
            print("Effective bw: {0:.4f}, effective wl: {1:.4f}, maximum wl: {2:.4f}". format(eff_bw_array[band],
                                                                                              eff_wl_array[band],
                                                                                              max_wl_array[band]))

    return eff_wl_array, eff_bw_array, max_wl_array


def format_figure_rsr(data_rsr, bandwidth=False):
    """
    Function to create figure.

    :param data_rsr: hdf5 rsr calibration object
    :return: (fig, ax)
    """
    ff = FigureFunctions()
    #fig, ax = plt.subplots(figsize=ff.set_size(443.86319, fraction=0.7))
    fig, ax = plt.subplots(figsize=ff.set_size(fraction=0.6, height_ratio=0.75))

    rsr = data_rsr["rsr_peak_norm"][:]
    rsr_rel_unc = data_rsr["rsr_relative_unc"][:]
    wl = data_rsr["wavelength"][:]

    # Band statistics
    eff_ww, bw, max_w = rsr_statistics(wl, rsr)

    # graph parameters
    lstyle = ["-", "-.", ":"]
    m = ["o", "s", "^"]
    #col = ['#d62728', '#2ca02c', '#1f77b4']
    col = ['#d95f02', '#1b9e77', '#7570b3']
    bname = ["r", "g", "b"]
    channel_name = ["red channel", "green channel", "blue channel"]
    ann_px_offset_y = [17, 12, 16]
    ann_px_offset_x = [-10, -20, -45]
    #ann_px_offset_y = [0.1, 0.047, 0.09375]
    #ann_px_offset_x = [5, -10, -10]

    bw_ht = np.array([0.6, 0.5, 0.4])

    # loop
    for n in range(rsr.shape[1]):

        # Format label
        #lab = r"{0}: $\lambda_{{eff, {1}}}={2:.1f}$ nm, $BW_{1}={3:.1f}$ nm".format(channel_name[n], bname[n], eff_ww[n], bw[n])
        #lab = r"{0}: $(\lambda_{{eff, {1}}} \pm BW_{1}) = ({2:.1f} \pm {3:.1f})$ nm".format(channel_name[n], bname[n],
                                                                                    #eff_ww[n], bw[n])

        lab = r"$(\lambda_{{eff, {0}}} \pm BW_{0}) = ({1:.1f} \pm {2:.1f})$ nm".format(bname[n], eff_ww[n], bw[n])

        # Plot
        ax.plot(wl, rsr[:, n], color=col[n], marker=m[n], markersize=2, markeredgecolor=col[n], markerfacecolor="none", linestyle=lstyle[n], linewidth=0.9, label=lab)
        rsr_unc = rsr[:, n] * rsr_rel_unc[:, n]
        ax.fill_between(wl, rsr[:, n]-rsr_unc, rsr[:, n]+rsr_unc, color="lightgrey", alpha=0.7)

        # Channel names
        ax.annotate(channel_name[n].split(" ")[0] + ", $\lambda_{{eff}}$ = {0:.0f} nm".format(eff_ww[n]), (max_w[n], 1.), (ann_px_offset_x[n], ann_px_offset_y[n]),
                    xycoords="data", textcoords='offset points', fontsize=7)
        #ax.annotate(channel_name[n].split(" ")[0] + ", {0:.0f} nm".format(eff_ww[n]), (eff_ww[n], 1.), (ann_px_offset_x[n], ann_px_offset_y[n]),
        #            xycoords="data", textcoords='offset points', fontsize=7,
        #            arrowprops=dict(arrowstyle="->", color="k"))

        #ax.annotate("$\lambda_{{eff, {0}}}$={1:.0f} nm".format(bname[n], eff_ww[n]), (eff_ww[n], 1), (-50, -60),
        #            xycoords="data", textcoords='offset pixels', fontsize=7,
        #            arrowprops=dict(arrowstyle="->", color="k", shrinkA=0.2, shrinkB=0.2))

        # Bandwidth plot
        if bandwidth:
            ax.errorbar(eff_ww[n], bw_ht[n], xerr=bw[n] / 2, color=col[n], marker=m[n], markersize=2, markeredgecolor=col[n], markerfacecolor="none", linestyle=lstyle[n])

    ax.set_ylim(-0.1, 1.25)
    #ax.set_xlim(386.5, 705)
    #ax.grid()
    #ax.set_ylabel("$RSR_{i}(\lambda)$")
    #ax.set_ylabel("Relative spectral response, $RSR_{i}$")
    ax.set_ylabel("Relative spectral response, $S_{R,i}$")
    ax.set_xlabel("Wavelength [nm]")
    #ax.legend(loc="best", fontsize=7, frameon=False)

    fig.tight_layout()

    return fig, ax


def two_curves_one_graph(data1, data2, shaded=False):

    #ff = FigureFunctions()
    #fig, ax = plt.subplots(figsize=ff.set_size(443.86319, fraction=0.7))
    figs_inch = 84 / 25.54
    fig, ax = plt.subplots(figsize=(figs_inch, figs_inch * 0.7))

    # Data
    rsr1 = data1["rsr_peak_norm"][:]
    rsr_rel_unc1 = data1["rsr_relative_unc"][:]
    wl1 = data1["wavelength"][:]

    rsr2 = data2["rsr_peak_norm"][:]
    rsr_rel_unc2 = data2["rsr_relative_unc"][:]
    wl2 = data2["wavelength"][:]

    # graph parameters
    lstyle = ["-", "-.", ":"]
    m = ["o", "s", "^"]
    #col = ['#d62728', '#2ca02c', '#1f77b4']
    col = ['#d95f02', '#1b9e77', '#7570b3']
    emp = np.empty(2)
    emp[:] = np.NaN

    # loop
    for n in range(rsr1.shape[1]):

        ax.plot(wl1, rsr1[:, n], color=col[n], marker=m[n], markersize=3, markeredgecolor=col[n], markerfacecolor="none", linestyle=lstyle[0])
        ax.plot(wl2, rsr2[:, n], color=col[n], marker=m[n], markersize=3, markeredgecolor=col[n], markerfacecolor="none", linestyle=lstyle[2])

        if shaded:
            ax.fill_between(wl1, rsr1[:, n] * (1 - rsr_rel_unc1[:, n]), rsr1[:, n] * (1 + rsr_rel_unc1[:, n]), color="lightgrey", alpha=0.7)
            ax.fill_between(wl2, rsr2[:, n] * (1 - rsr_rel_unc2[:, n]), rsr2[:, n] * (1 + rsr_rel_unc2[:, n]), color="lightgrey", alpha=0.7)

    ax.plot(emp, emp, color="k", linestyle=lstyle[0], label="Lens 1")
    ax.plot(emp, emp, color="k", linestyle=lstyle[2], label="Lens 2")

    ax.set_ylim((ax.get_ylim()[0], 1.2))
    ax.set_xlim(386.5, 705)
    ax.grid()
    ax.set_ylabel("Relative spectral response, $RSR_{i}(\lambda)$")
    ax.set_xlabel("Wavelength [nm]")

    ax.legend(loc="best")

    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":

    # Figure style
    plt.style.use(os.path.abspath(os.path.join(__file__, "../../..")) + "/figurestyle.mplstyle")

    # Open data
    data_c = h5py.File("calibrationfiles/rsr_20200610.h5")
    data_c = data_c["lens-close"]

    data_f = h5py.File("calibrationfiles/rsr_20200710.h5")
    data_f = data_f["lens-far"]

    #data_fluo = h5py.File("calibrationfiles/rsr_fluorolog_20220914.h5")
    #data_fluo = h5py.File("calibrationfiles/rsr_fluorolog_test220220914.h5")
    data_fluo = h5py.File("calibrationfiles/rsr_fluorolog_20221103.h5")
    #data_fluo = data_fluo["lens-close"]

    # Figure
    fig1, ax1 = format_figure_rsr(data_c, bandwidth=False)
    fig2, ax2 = format_figure_rsr(data_f, bandwidth=False)
    fig3, ax3 = format_figure_rsr(data_fluo["lens-close"], bandwidth=False)
    ax3.set_rasterized(True)
    fig4, ax4 = two_curves_one_graph(data_c, data_f)
    fig5, ax5 = format_figure_rsr(data_fluo["lens-far"])


    dflu_lc = data_fluo["lens-close"]
    lstyle = ["-", "-.", ":"]
    m = ["o", "s", "^"]
    col = ['#d95f02', '#1b9e77', '#7570b3']
    for n in range(dflu_lc["rsr_peak_norm"][:].shape[1]):

        curr_rsr = dflu_lc["rsr_peak_norm"][:][:, n]
        ax4.plot(dflu_lc["wavelength"][:], curr_rsr, color=col[n], marker=m[n], markersize=3, markeredgecolor=col[n], markerfacecolor="none", linestyle="--")

    # Saving figure
    fig1.savefig("figures/rsr_close.png", format="png", dpi=600)
    fig2.savefig("figures/rsr_far.png", format="png", dpi=600)
    fig3.savefig("figures/rsr_close_fluo.png", format="png", dpi=600)
    fig3.savefig("figures/Fig.4.jpg", format='jpg', dpi=600)
    fig3.savefig("figures/Fig.4.pdf", format='pdf', dpi=600)

    plt.show()
