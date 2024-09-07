# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import string
import pandas
import matplotlib.cm
import matplotlib.pyplot as plt

import field.oden2018.oden_2018_aops_iops as rfct
from bdc_process_stations import create_label, get_ice_freeboard


# Function and classes
def graph_aops(radcl, ifb):
    """

    :param radcl:
    :return:
    """
    if radcl.station == "station_2":
        p = "data/super_recu_bdc_fit_final/eudos_iops.csv"
    elif radcl.station == "station_3":
        p = "data/bdc_3_fit/eudos_iops.csv"
    else:
        p = "None"
        ValueError("Station not taken in charge right now.")

    z_depth = radcl.ed["depth"][1:]
    z_depth[0] = 0.0
    print(z_depth)
    ed_hl, eu_hl, eo_hl, edo_hl, euo_hl = rfct.construct_irradiance_profile_HL(z_depth, path=p)

    # Mean cosine hl
    ud_hl = ed_hl / edo_hl
    uu_hl = eu_hl / euo_hl
    u_hl = (ed_hl - eu_hl) / eo_hl

    enet_hl = ed_hl - eu_hl
    a_hl = -np.gradient(enet_hl, z_depth/100, axis=0, edge_order=2) * (1/eo_hl)
    kd_hl = -np.gradient(ed_hl, z_depth/100, axis=0, edge_order=2) * (1/ed_hl)

    # Loop
    plt.style.use(os.path.abspath(os.path.join(__file__, "../../..")) + "/figurestyle.mplstyle")
    fig, ax = plt.subplots(3, 3, sharey=True, figsize=(6.6929, 6.6929 * 0.9))
    #fig, ax = plt.subplots(2, 3, sharey=True, figsize=(6.6929, 6.6929 * 0.8))

    CMAR = matplotlib.cm.get_cmap("Reds", 10 + 1)
    CMAG = matplotlib.cm.get_cmap("Greens", 10 + 1)
    CMAB = matplotlib.cm.get_cmap("Blues", 10 + 1)

    #col = ["#d55e00", "#009e73", "#0072b2"]
    col = [CMAR(7), CMAG(7), CMAB(7)]

    wl_label = {"r": "603 nm", "g": "544 nm", "b": "484 nm"}

    mask_d = ifb <= radcl.ed["depth"]

    for i, b in enumerate(["r", "g", "b"]):

        # Irradiances
        ax[0, 0].plot(radcl.ed[b][mask_d], radcl.ed["depth"][mask_d], linewidth=0.9, marker="o", markersize=2, color=col[i], label=wl_label[b])
        ax[0, 1].plot(radcl.eu[b][mask_d], radcl.eu["depth"][mask_d], linewidth=0.9, marker="o", markersize=2,  color=col[i], label=wl_label[b])
        ax[0, 2].plot(radcl.eo[b][mask_d], radcl.eo["depth"][mask_d], linewidth=0.9, marker="o", markersize=2,  color=col[i], label=wl_label[b])

        #ax[0, 0].plot(radcl.ed[b], radcl.ed["depth"], linestyle="none", markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])
        #ax[0, 1].plot(radcl.eu[b], radcl.eu["depth"], linestyle="none",  markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])
        #ax[0, 2].plot(radcl.eo[b], radcl.eo["depth"], linestyle="none",  markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])

        # Hydrolight
        ax[0, 0].plot(ed_hl[:, i], z_depth, linewidth=0.9,  linestyle="--",  color=col[i])
        ax[0, 1].plot(eu_hl[:, i], z_depth, linewidth=0.9,  linestyle="--",  color=col[i])
        ax[0, 2].plot(eo_hl[:, i], z_depth, linewidth=0.9,  linestyle="--", color=col[i])

        # Average cosines
        ax[1, 0].plot(radcl.u_d[b], radcl.u_d["depth"], linewidth=0.9, marker="o", markersize=2,  color=col[i], label=wl_label[b])
        ax[1, 1].plot(radcl.u_u[b], radcl.u_u["depth"], linewidth=0.9, marker="o", markersize=2,  color=col[i], label=wl_label[b])
        ax[1, 2].plot(radcl.u[b], radcl.u["depth"], linewidth=0.9, marker="o", markersize=2,  color=col[i], label=wl_label[b])

        #ax[1, 0].plot(radcl.u_d[b], radcl.u_d["depth"], linestyle="none", markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])
        #ax[1, 1].plot(radcl.u_u[b], radcl.u_u["depth"], linestyle="none", markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])
        #ax[1, 2].plot(radcl.u[b], radcl.u["depth"], linestyle="none", markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])

        ax[1, 0].plot(ud_hl[:, i], z_depth, linewidth=0.8, linestyle="--", color=col[i])
        ax[1, 1].plot(uu_hl[:, i], z_depth, linewidth=0.8, linestyle="--", color=col[i])
        ax[1, 2].plot(u_hl[:, i], z_depth, linewidth=0.8, linestyle="--", color=col[i])

        ax[2, 0].plot(radcl.ed[b][mask_d] - radcl.eu[b][mask_d], radcl.ed["depth"][mask_d], marker="o", markersize=2, linewidth=0.9, color=col[i], label=wl_label[b])
        ax[2, 0].plot(enet_hl[:, i], z_depth, linestyle="--", linewidth=0.9, color=col[i])
        ax[2, 1].plot(kd_hl[:, i], z_depth, linestyle="--", linewidth=0.9, color=col[i])

        ax[2, 1].plot(radcl.K_d[b][mask_d], radcl.K_d["depth"][mask_d], marker="o", markersize=2, linewidth=0.9, color=col[i], label=wl_label[b])
        ax[2, 2].plot(a_hl[:, i], z_depth, linestyle="--", linewidth=0.9, color=col[i])

        ax[2, 2].plot(radcl.mu_a[b][mask_d], radcl.mu_a["depth"][mask_d], marker="o", markersize=2, linewidth=0.9, color=col[i], label=wl_label[b])

    # First row param
    ax[0, 0].set_xscale("log")
    ax[0, 1].set_xscale("log")
    ax[0, 2].set_xscale("log")

    ax[0, 0].set_ylabel("Depth [cm]")
    ax[0, 0].set_xlabel("$E_{d}[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax[0, 1].set_xlabel("$E_{u}[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax[0, 2].set_xlabel("$E_{o}[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")

    #ax[0, 0].legend(loc="best", frameon=False, fontsize=7)
    #ax[0, 1].legend(loc="best", frameon=False, fontsize=7)
    #ax[0, 2].legend(loc="best", frameon=False, fontsize=7)

    ax[0, 0].text(-0.05, 1.05, "(" + string.ascii_lowercase[0] + ")", transform=ax[0, 0].transAxes, size=10, weight='bold')
    ax[0, 1].text(-0.05, 1.05, "(" + string.ascii_lowercase[1] + ")", transform=ax[0, 1].transAxes, size=10, weight='bold')
    ax[0, 2].text(-0.05, 1.05, "(" + string.ascii_lowercase[2] + ")", transform=ax[0, 2].transAxes, size=10, weight='bold')

    # Second row param
    ax[1, 0].invert_yaxis()

    ax[1, 0].set_ylabel("Depth [cm]")
    ax[1, 0].set_xlabel("$\overline{\mu_{d}}$")
    ax[1, 1].set_xlabel("$\overline{\mu_{u}}$")
    ax[1, 2].set_xlabel("$\overline{\mu}$")

    #ax[1, 0].legend(loc="best", frameon=False, fontsize=7)
    #ax[1, 1].legend(loc="best", frameon=False, fontsize=7)
    #ax[1, 2].legend(loc="best", frameon=False, fontsize=7)

    ax[1, 0].text(-0.05, 1.05, "(" + string.ascii_lowercase[3] + ")", transform=ax[1, 0].transAxes, size=10, weight='bold')
    ax[1, 1].text(-0.05, 1.05, "(" + string.ascii_lowercase[4] + ")", transform=ax[1, 1].transAxes, size=10, weight='bold')
    ax[1, 2].text(-0.05, 1.05, "(" + string.ascii_lowercase[5] + ")", transform=ax[1, 2].transAxes, size=10, weight='bold')

    ax[2, 2].set_xscale("log")

    # Third row param
    ax[2, 0].set_xscale("log")
    ax[2, 0].set_ylabel("Depth [cm]")
    ax[2, 0].set_xlabel("$(E_{d} - E_{u})~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax[2, 1].set_xlabel("$K_{d}~[\mathrm{m^{-1}}]$")
    ax[2, 2].set_xlabel("$a~[\mathrm{m^{-1}}]$")

    #ax[2, 0].legend(loc="best", frameon=False, fontsize=7)
    #ax[2, 1].legend(loc="best", frameon=False, fontsize=7)
    #ax[2, 2].legend(loc="best", frameon=False, fontsize=7)

    ax[2, 0].text(-0.05, 1.05, "(" + string.ascii_lowercase[6] + ")", transform=ax[2, 0].transAxes, size=10, weight='bold')
    ax[2, 1].text(-0.05, 1.05, "(" + string.ascii_lowercase[7] + ")", transform=ax[2, 1].transAxes, size=10, weight='bold')
    ax[2, 2].text(-0.05, 1.05, "(" + string.ascii_lowercase[8] + ")", transform=ax[2, 2].transAxes, size=10, weight='bold')

    fig.tight_layout()

    return fig, ax


def graph_iops(radcl, ifb):
    """

    :param radcl:
    :return:
    """

    if radcl.station == "station_2":
        p = "data/super_recu_bdc_fit_final/eudos_iops.csv"
    elif radcl.station == "station_3":
        p = "data/bdc_3_fit/eudos_iops.csv"

    z_hr = np.arange(0, 100, 1)

    ed_hl, eu_hl, eo_hl, _, _ = rfct.construct_irradiance_profile_HL(z_hr, path=p)

    mask_d = ifb <= radcl.ed["depth"]

    enet_hl = ed_hl - eu_hl
    a_hl = -np.gradient(enet_hl, z_hr/100, axis=0, edge_order=2) * (1/eo_hl)
    kd_hl = -np.gradient(ed_hl, z_hr/100, axis=0, edge_order=2) * (1/ed_hl)

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(6.6929, 6.6929 * 0.5))

    col = ["#d55e00", "#009e73", "#0072b2"]
    wl_label = {"r": "603 nm", "g": "544 nm", "b": "484 nm"}
    b = ["r", "g", "b"]
    for i in range(3):

        ax[0].plot(radcl.ed[b[i]][mask_d] - radcl.eu[b[i]][mask_d] , radcl.ed["depth"][mask_d] , linewidth=0.9, color=col[i], label=list(wl_label.values())[i])
        ax[0].plot(enet_hl[:, i], z_hr, linestyle="-.", linewidth=0.9, color=col[i])

        ax[1].plot(kd_hl[:, i][1:-1], z_hr[1:-1], linestyle="-.", linewidth=0.9, color=col[i])
        ax[1].plot(radcl.K_d[b[i]][mask_d] , radcl.K_d["depth"][mask_d] , linewidth=0.9, color=col[i], label=list(wl_label.values())[i])

        ax[2].plot(a_hl[:, i][1:-1], z_hr[1:-1], linestyle="-.", linewidth=0.9, color=col[i])
        ax[2].plot(radcl.mu_a[b[i]][mask_d] , radcl.mu_a["depth"][mask_d] , linewidth=0.9, color=col[i], label=list(wl_label.values())[i])

    ax[0].invert_yaxis()
    ax[0].set_xscale("log")
    ax[0].set_ylabel("Depth [cm]")
    ax[0].set_xlabel("$(E_{d} - E_{u})~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")

    ax[1].set_xlabel("$K_{d}~[\mathrm{m^{-1}}]$")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("$a~[\mathrm{m^{-1}}]$")

    ax[0].annotate(txt_annotate, (0.6, 0.45), xycoords="axes fraction", fontsize=6)
    ax[1].annotate(txt_annotate, (0.6, 0.45), xycoords="axes fraction", fontsize=6)
    ax[2].annotate(txt_annotate, (0.6, 0.13), xycoords="axes fraction", fontsize=6)

    ax[0].legend(loc="best", frameon=False)
    ax[1].legend(loc="best", frameon=False)
    ax[2].legend(loc="best", frameon=False)

    return fig, ax


def draw_layers(axe, snow_thickness, ice_freebord, ice_thickness):

    [ax.axhspan(0, snow_thickness, facecolor="dodgerblue", alpha=0.00) for ax in axe.ravel()]
    [ax.axhspan(snow_thickness, ice_freebord, facecolor="dodgerblue", alpha=0.1) for ax in axe.ravel()]
    [ax.axhspan(ice_freebord, ice_thickness, facecolor="dodgerblue", alpha=0.2) for ax in axe.ravel()]
    [ax.axhspan(ice_thickness, rc_st2.mu_a["depth"].max() + 100, facecolor="dodgerblue", alpha=0.5) for ax in axe.ravel()]

    [ax.axhline(0, linewidth=0.4, color="gray", alpha=0.5) for ax in axe.ravel()]
    [ax.axhline(snow_thickness, linewidth=0.4, color="gray", alpha=0.5) for ax in axe.ravel()]
    [ax.axhline(ice_freebord, linewidth=0.4, color="gray", alpha=0.5) for ax in axe.ravel()]
    [ax.axhline(ice_thickness, linewidth=0.4, color="gray", alpha=0.5) for ax in axe.ravel()]

    #[ax.set_yticks((np.arange(-50, 400, 50))) for ax in ax2.ravel()]

    [ax.annotate("sea ice", (0.02, 0.86), xycoords="axes fraction", fontsize=5) for ax in axe.ravel()]
    [ax.annotate("water level", (0.02, 0.77), xycoords="axes fraction", fontsize=5) for ax in axe.ravel()]
    [ax.annotate("seawater", (0.02, 0.03), xycoords="axes fraction", fontsize=5) for ax in axe.ravel()]

    return axe


if __name__ == "__main__":

    ifb_st2 = get_ice_freeboard("data/station_2_data.txt")
    ifb_st3 = get_ice_freeboard("data/station_3_data.txt")

    label_st2 = np.array(list(create_label("data/station_2_data.txt").keys()))
    label_st3 = np.array(list(create_label("data/station_3_data.txt").keys()))

    #rc_st2 = rfct.RadClass(data_path="data/baiedeschaleurs-03232022.h5", station="station_2", data_type="camera", freeboard=0.0)
    rc_st2 = rfct.RadClass(data_path="data/baiedeschaleurs-03232022-imf-fluo.h5", station="station_2", data_type="camera", freeboard=0.0)
    #rc_st2 = rfct.RadClass(data_path="data/baiedeschaleurs-03232022.h5", station="station_2", data_type="camera", freeboard=0.11)
    rc_st3 = rfct.RadClass(data_path="data/baiedeschaleurs-03232022-imf-fluo.h5", station="station_3", data_type="camera", freeboard=11.0)

    # Figures
    plt.style.use(os.path.abspath(os.path.join(__file__, "../../..")) + "/figurestyle.mplstyle")
    #fig1, ax1 = graph_aops(rc_st2, 0.1)
    #fig1, ax1 = graph_aops(rc_st2, ifb_st2)
    fig1, ax1 = graph_aops(rc_st2, 5)
    #fig2, ax2 = graph_aops(rc_st3, 0.1)
    fig2, ax2 = graph_aops(rc_st3, ifb_st3)
    #fig2, ax2 = plt.subplots(3, 3, sharey=True, figsize=(6.6929, 6.6929 * 0.9))

    # Text annotation
    txt_annotate = "- measurements\n-- simulations"
    ax1[0, 0].annotate(txt_annotate, (0.05, 0.8), xycoords="axes fraction", fontsize=6)
    ax1[0, 1].annotate(txt_annotate, (0.6, 0.05), xycoords="axes fraction", fontsize=6)
    ax1[0, 2].annotate(txt_annotate, (0.05, 0.8), xycoords="axes fraction", fontsize=6)

    ax1[1, 0].annotate(txt_annotate, (0.05, 0.05), xycoords="axes fraction", fontsize=6)
    ax1[1, 1].annotate(txt_annotate, (0.6, 0.2), xycoords="axes fraction", fontsize=6)
    ax1[1, 2].annotate(txt_annotate, (0.6, 0.4), xycoords="axes fraction", fontsize=6)

    ax1[2, 0].annotate(txt_annotate, (0.05, 0.8), xycoords="axes fraction", fontsize=6)
    ax1[2, 1].annotate(txt_annotate, (0.6, 0.05), xycoords="axes fraction", fontsize=6)
    ax1[2, 2].annotate(txt_annotate, (0.05, 0.6), xycoords="axes fraction", fontsize=6)

    # Legend
    ax1[0, 0].legend(loc="best", frameon=False, fontsize=7)
    ax1[0, 1].legend(loc="best", frameon=False, fontsize=7)
    ax1[0, 2].legend(loc="best", frameon=False, fontsize=7)

    ax1[1, 0].legend(loc="best", frameon=False, fontsize=7)
    ax1[1, 1].legend(loc="best", frameon=False, fontsize=7)
    ax1[1, 2].legend(loc="best", frameon=False, fontsize=7)

    ax1[2, 0].legend(loc="best", frameon=False, fontsize=7)
    ax1[2, 1].legend(loc="best", frameon=False, fontsize=7)
    ax1[2, 2].legend(loc="best", frameon=False, fontsize=7)

    # Text annotation
    ax2[0, 0].annotate(txt_annotate, (0.5, 0.05), xycoords="axes fraction", fontsize=6)
    ax2[0, 1].annotate(txt_annotate, (0.5, 0.05), xycoords="axes fraction", fontsize=6)
    ax2[0, 2].annotate(txt_annotate, (0.5, 0.05), xycoords="axes fraction", fontsize=6)

    ax2[1, 0].annotate(txt_annotate, (0.6, 0.8), xycoords="axes fraction", fontsize=6)
    ax2[1, 1].annotate(txt_annotate, (0.05, 0.5), xycoords="axes fraction", fontsize=6)
    ax2[1, 2].annotate(txt_annotate, (0.6, 0.45), xycoords="axes fraction", fontsize=6)

    # IOPs
    data_st2 = pandas.read_csv("data/super_recu_bdc_fit_final/eudos_iops.csv")
    data_st3 = pandas.read_csv("data/bdc_3_fit/eudos_iops.csv")

    g_st2 = np.ones(data_st2["depths"][1:].shape[0]) * 0.99
    g_st2[data_st2["depths"][1:] < 0.20] = 0.85
    g_st2[0.80 <= data_st2["depths"][1:]] = 0.97

    g_st3 = np.ones(data_st3["depths"][1:].shape[0]) * 0.99
    g_st3[data_st3["depths"][1:] < 0.20] = 0.85
    g_st3[0.80 <= data_st3["depths"][1:]] = 0.97

    fig3, ax3 = plt.subplots(1, 3, sharey=True, figsize=(6.6929, 6.6929 * 0.5))
    colo = ["#d55e00", "#009e73", "#0072b2"]

    ax3[0].plot(data_st2["a_600.0"][1:], data_st2["depths"][1:] * 100, color=colo[0], label="600 nm", linewidth=0.9)
    ax3[0].plot(data_st2["a_540.0"][1:], data_st2["depths"][1:] * 100, color=colo[1], label="540 nm", linewidth=0.9)
    ax3[0].plot(data_st2["a_480.0"][1:], data_st2["depths"][1:] * 100, color=colo[2], label="480 nm", linewidth=0.9)

    ax3[0].plot(data_st3["a_600.0"][1:], data_st3["depths"][1:] * 100,  linestyle="-.", color=colo[0], linewidth=0.9)
    ax3[0].plot(data_st3["a_540.0"][1:], data_st3["depths"][1:] * 100,  linestyle="-.", color=colo[1], linewidth=0.9)
    ax3[0].plot(data_st3["a_480.0"][1:], data_st3["depths"][1:] * 100,  linestyle="-.", color=colo[2], linewidth=0.9)

    ax3[1].plot(data_st2["b_600.0"][1:], data_st2["depths"][1:] * 100, label="Station 2", linewidth=0.9, color="k")
    ax3[1].plot(data_st3["b_600.0"][1:], data_st3["depths"][1:] * 100, linestyle="-.", label="Station 3", linewidth=0.9, color="k")

    #ax3t = ax3[1].twiny()
    #ax3t.plot(1-g_st2, data_st2["depths"][1:] * 100, linewidth=0.9, color="grey")
    #ax3t.plot(1-g_st3, data_st3["depths"][1:] * 100, linewidth=0.9,  linestyle="-.", color="grey")

    ax3[2].plot(data_st2["b_600.0"][1:] * (1 - g_st2), data_st2["depths"][1:] * 100, label="Station 2", linewidth=0.9, color="k")
    ax3[2].plot(data_st3["b_600.0"][1:] * (1 - g_st3), data_st3["depths"][1:] * 100, linestyle="-.", label="Station 3", linewidth=0.9, color="k")

    ax3[0].set_xscale("log")
    ax3[1].set_xscale("log")
    ax3[2].set_xscale("log")

    ax3[0].set_ylim((-5, 105))
    ax3[0].invert_yaxis()
    ax3[0].set_xlabel("$a~[\mathrm{m^{-1}}]$")
    ax3[0].set_ylabel("Depth [cm]")
    ax3[0].legend(loc="best", frameon=False)

    ax3[1].set_xlabel("$b~[\mathrm{m^{-1}}]$")
    ax3[2].set_xlabel("$b(1-g)~[\mathrm{m^{-1}}]$")

    ax3[1].legend(loc="best")
    ax3[2].legend(loc="best")

    # Figure 4 -
    fig4, ax4 = graph_iops(rc_st2, ifb_st2)
    fig5, ax5 = graph_iops(rc_st3, ifb_st3)

    ax5[2].plot(data_st3["a_600.0"][1:], data_st3["depths"][1:] * 100, linestyle=":", color=colo[0])
    ax5[2].plot(data_st3["a_540.0"][1:], data_st3["depths"][1:] * 100, linestyle=":", color=colo[1])
    ax5[2].plot(data_st3["a_480.0"][1:], data_st3["depths"][1:] * 100, linestyle=":", color=colo[2])

    ax5[0].set_ylim((-5, 105))
    ax5[0].invert_yaxis()

    # Figure 6 - binning
    #layers_bdc = [0, 5, 18, 50, 72, 100]
    layers_bdc = [0, 5, 18, 35, 65, 72, 100]

    fig6, ax6 = plt.subplots(1, 2, sharey=True)
    mask_depths = rc_st2.mu_a["depth"] > 5.0
    [ax6[0].plot(rc_st2.mu_a[i][mask_depths], rc_st2.mu_a["depth"][mask_depths], marker=".", linestyle="-", color=i) for i in ["r", "g", "b"]]
    [ax6[0].axhspan(layers_bdc[d], layers_bdc[d+1], color="DodgerBlue", alpha=0.1 * d) for d in range(len(layers_bdc)-1)]

    a_bin_st2 = np.zeros((len(layers_bdc)-1, 3))
    for d in range(len(layers_bdc)-1):

        mask_layer = (rc_st2.mu_a["depth"] >= layers_bdc[d]) & (rc_st2.mu_a["depth"] < layers_bdc[d+1])

        a_bin_st2[d, 0] = np.median(rc_st2.mu_a[mask_layer]["r"])
        a_bin_st2[d, 1] = np.median(rc_st2.mu_a[mask_layer]["g"])
        a_bin_st2[d, 2] = np.median(rc_st2.mu_a[mask_layer]["b"])

    a_bin_st2 = np.clip(a_bin_st2, 0, None)

    ax6[0].invert_yaxis()
    ax6[0].set_xscale('log')

    depths_center = 0.5 * (np.array(layers_bdc)[0:-1] + np.array(layers_bdc)[1:])
    ax6[1].plot(a_bin_st2[:, 0], depths_center, color="r", linestyle="-", marker=".")
    ax6[1].plot(a_bin_st2[:, 1], depths_center, color="g", linestyle="-", marker=".")
    ax6[1].plot(a_bin_st2[:, 2], depths_center, color="b", linestyle="-", marker=".")
    [ax6[1].axhspan(layers_bdc[d], layers_bdc[d+1], color="DodgerBlue", alpha=0.1 * d) for d in range(len(layers_bdc)-1)]

    ax6[1].set_xscale('log')

    # Figure 7
    #wl_label = {"r": "603 nm", "g": "544 nm", "b": "484 nm"}
    wl_label = {"r": "600 nm", "g": "540 nm", "b": "480 nm"}
    col = ['grey', 'darkorchid', 'darkorange']

    ed_hl, eu_hl, eo_hl, edo_hl, euo_hl = rfct.construct_irradiance_profile_HL(rc_st2.ed["depth"][1:], path="data/super_recu_bdc_fit_final/eudos_iops.csv")
    # Mean cosine hl
    ud_hl = ed_hl / edo_hl
    uu_hl = eu_hl / euo_hl
    u_hl = (ed_hl - eu_hl) / eo_hl
    enet_hl = ed_hl - eu_hl

    z_hr = np.arange(0, 91.0, 1)
    ed_hl_a, eu_hl_a, eo_hl_a, _, _ = rfct.construct_irradiance_profile_HL(z_hr, path="data/super_recu_bdc_fit_final/eudos_iops.csv")
    a_hl = -np.gradient(ed_hl_a - eu_hl_a, z_hr/100, axis=0, edge_order=2) * (1/eo_hl_a)

    fig7, ax7 = plt.subplots(2, 3, sharey=True, figsize=(6.6929, 6.6929 * 0.9))

    for i, ri in enumerate(['b', 'g', 'r']):
        ax7[0, i].plot(rc_st2.ed[ri][1:], rc_st2.ed['depth'][1:], marker="o",  color=col[0], markeredgecolor=col[0], markerfacecolor='none', markersize=5, linestyle='none')
        ax7[0, i].plot(ed_hl[:, 2-i], rc_st2.ed["depth"][1:], color=col[0], linestyle='-', linewidth=2.0, label="$E_{d}$")
        ax7[0, i].plot(rc_st2.eu[ri][1:], rc_st2.eu['depth'][1:], marker="o", color=col[1], markeredgecolor=col[1], markerfacecolor='none', markersize=5,  linestyle='none')
        ax7[0, i].plot(eu_hl[:, 2-i], rc_st2.ed["depth"][1:], color=col[1], linestyle='-', linewidth=2.0, label="$E_{u}$")
        ax7[0, i].plot(rc_st2.eo[ri][1:], rc_st2.eo['depth'][1:], marker="o", color=col[2], markeredgecolor=col[2], markerfacecolor='none', markersize=5,  linestyle='none')
        ax7[0, i].plot(eo_hl[:, 2-i], rc_st2.ed["depth"][1:], color=col[2], linestyle='-', linewidth=2.0,  label="$E_{o}$")

        ax7[0, i].set_title(wl_label[ri], fontsize=10)
        ax7[0, i].set_xlabel('Irradiance [$\mathrm{W \cdot m^{-2} \cdot nm^{-1}}$]')
        ax7[0, i].legend(loc='best', frameon=False, fontsize=7)

        ax7[1, i].plot(rc_st2.u_d[ri][1:], rc_st2.u_d['depth'][1:], marker="o",  color=col[0], markeredgecolor=col[0], markerfacecolor='none', markersize=5, linestyle='none')
        ax7[1, i].plot(ud_hl[:, 2-i], rc_st2.ed["depth"][1:], color=col[0], linestyle='-', linewidth=2.0, label="$\mu_{d}$")
        ax7[1, i].plot(rc_st2.u_u[ri][1:], rc_st2.u_u['depth'][1:], marker="o", color=col[1], markeredgecolor=col[1], markerfacecolor='none', markersize=5,  linestyle='none')
        ax7[1, i].plot(uu_hl[:, 2-i], rc_st2.ed["depth"][1:], color=col[1], linestyle='-', linewidth=2.0, label="$\mu_{u}$")
        ax7[1, i].plot(rc_st2.u[ri][1:], rc_st2.u['depth'][1:], marker="o", color=col[2], markeredgecolor=col[2], markerfacecolor='none', markersize=5,  linestyle='none')
        ax7[1, i].plot(u_hl[:, 2-i], rc_st2.u["depth"][1:], color=col[2], linestyle='-', linewidth=2.0,  label="$\mu_{o}$")

        ax7[1, i].set_title(wl_label[ri], fontsize=10)
        ax7[1, i].set_xlabel('Average cosine [-]')
        ax7[1, i].legend(loc='best', frameon=False, fontsize=7)

    ax7[0, 0].text(-0.05, 1.05, "(" + string.ascii_lowercase[0] + ")", transform=ax7[0, 0].transAxes, size=10, weight='bold')
    ax7[0, 1].text(-0.05, 1.05, "(" + string.ascii_lowercase[1] + ")", transform=ax7[0, 1].transAxes, size=10, weight='bold')
    ax7[0, 2].text(-0.05, 1.05, "(" + string.ascii_lowercase[2] + ")", transform=ax7[0, 2].transAxes, size=10, weight='bold')

    ax7[1, 0].text(-0.05, 1.05, "(" + string.ascii_lowercase[3] + ")", transform=ax7[1, 0].transAxes, size=10, weight='bold')
    ax7[1, 1].text(-0.05, 1.05, "(" + string.ascii_lowercase[4] + ")", transform=ax7[1, 1].transAxes, size=10, weight='bold')
    ax7[1, 2].text(-0.05, 1.05, "(" + string.ascii_lowercase[5] + ")", transform=ax7[1, 2].transAxes, size=10, weight='bold')

    snow_thickness = 0  # cm
    ice_thickness = 72 # cm  # or 200 cm ?
    ice_freebord = 18 # cm

    [ax.axhspan(0, snow_thickness, facecolor="dodgerblue", alpha=0.00) for ax in ax7.ravel()]
    [ax.axhspan(snow_thickness, ice_freebord, facecolor="dodgerblue", alpha=0.1) for ax in ax7.ravel()]
    [ax.axhspan(ice_freebord, ice_thickness, facecolor="dodgerblue", alpha=0.2) for ax in ax7.ravel()]
    [ax.axhspan(ice_thickness, rc_st2.mu_a["depth"].max() + 100, facecolor="dodgerblue", alpha=0.5) for ax in ax7.ravel()]

    [ax.axhline(0, linewidth=0.4, color="gray", alpha=0.5) for ax in ax7.ravel()]
    [ax.axhline(snow_thickness, linewidth=0.4, color="gray", alpha=0.5) for ax in ax7.ravel()]
    [ax.axhline(ice_freebord, linewidth=0.4, color="gray", alpha=0.5) for ax in ax7.ravel()]
    [ax.axhline(ice_thickness, linewidth=0.4, color="gray", alpha=0.5) for ax in ax7.ravel()]

    #[ax.set_yticks((np.arange(-50, 400, 50))) for ax in ax2.ravel()]

    [ax.annotate("sea ice", (0.02, 0.88), xycoords="axes fraction", fontsize=5) for ax in ax7[:, 0]]
    [ax.annotate("water level", (0.02, 0.77), xycoords="axes fraction", fontsize=5) for ax in ax7[:, 0]]
    [ax.annotate("seawater", (0.4, 0.03), xycoords="axes fraction", fontsize=5) for ax in ax7[:, 0]]

    txt_annotate = "o Measurements\n- HL simulations"

    ax7[0, 0].annotate(txt_annotate, (0.6, 0.4), xycoords="axes fraction", fontsize=6)
    ax7[0, 1].annotate(txt_annotate, (0.6, 0.4), xycoords="axes fraction", fontsize=6)
    ax7[0, 2].annotate(txt_annotate, (0.6, 0.4), xycoords="axes fraction", fontsize=6)

    ax7[1, 0].annotate(txt_annotate, (0.05, 0.3), xycoords="axes fraction", fontsize=6)
    ax7[1, 1].annotate(txt_annotate, (0.05, 0.3), xycoords="axes fraction", fontsize=6)
    ax7[1, 2].annotate(txt_annotate, (0.05, 0.3), xycoords="axes fraction", fontsize=6)

    ax7[0, 0].set_ylim((-10, 105))

    ax7[0, 0].invert_yaxis()
    ax7[0, 0].set_xscale('log')
    ax7[0, 1].set_xscale('log')
    ax7[0, 2].set_xscale('log')

    ax7[0, 0].set_ylabel("Depth [cm]")
    ax7[1, 0].set_ylabel("Depth [cm]")

    fig8, ax8 = plt.subplots(3, 3, figsize=(6.6929, 6.6929 *0.9), sharey=True)
    hl_labels = {"r": "a_600.0", "g": "a_540.0", "b": "a_480.0"}
    hl_kd_labels = {"r": "Kd_600.0", "g": "Kd_540.0", "b": "Kd_480.0"}
    st_depth = 2
    CMAR = matplotlib.cm.get_cmap("Reds", 10 + 1)
    CMAG = matplotlib.cm.get_cmap("Greens", 10 + 1)
    CMAB = matplotlib.cm.get_cmap("Blues", 10 + 1)

    #col = ["#d55e00", "#009e73", "#0072b2"]
    col_wl = {'r': CMAR(7), 'g': CMAG(7), 'b': CMAB(7)}

    xlims_axe = np.empty((3, 2))

    for i, b in enumerate(["b", "g", "r"]):
        ax8[0, i].plot(rc_st2.ed[b][st_depth:] - rc_st2.eu[b][st_depth:], rc_st2.ed["depth"][st_depth:], marker="o", markersize=5, color=col_wl[b], linestyle='none', markerfacecolor='none', markeredgecolor=col_wl[b], label='Measurements')
        ax8[1, i].plot(rc_st2.K_d[b][st_depth:], rc_st2.K_d["depth"][st_depth:], marker="o", color=col_wl[b],  markersize=5, linestyle='none', markerfacecolor='none',  markeredgecolor=col_wl[b], label='Measurements')
        ax8[2, i].plot(rc_st2.mu_a[b][st_depth:], rc_st2.mu_a["depth"][st_depth:], marker="o", color=col_wl[b],  markersize=5, linestyle='none', markerfacecolor='none',  markeredgecolor=col_wl[b], label='Measurements')

        ax8[0, i].plot(enet_hl[:, i], rc_st2.ed["depth"][1:], linestyle="-", linewidth=2.0, color=col_wl[b], label='HL simulations')
        ax8[1, i].plot(data_st2[hl_kd_labels[b]][st_depth:], data_st2["depths"][st_depth:] * 100, linestyle="-", linewidth=2.0, color=col_wl[b], label='HL simulations')
        ax8[2, i].plot(a_hl[:, i], z_hr, linestyle="-", color=col_wl[b], linewidth=2.0, label='HL simulations')
        #ax8[2, i].plot(data_st2[hl_labels[b]][st_depth:], data_st2["depths"][st_depth:] * 100, linestyle="-", color=col_wl[b], linewidth=2.0, label='HL simulations')


        # ax2[2, 1].plot(kd_hl[:, i][1:-1], z_hr[1:-1], linestyle="--", linewidth=0.9, color=col[i], alpha=0.7)
        # ax2[2, 2].plot(rc.mu_a[b], rc.mu_a["depth"], marker="o", markersize=3, linewidth=0.9, color=col[i], label=wl_label[b])

    #ax8[i].text(-0.0, 1.05, "("+ string.ascii_lowercase[i] + ")", transform=ax8[i].transAxes, size=10, weight='bold')

        ax8[0, i].set_xscale("log")
        ax8[0, i].set_xlabel("$(E_{d} - E_{u})~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
        #ax8[0, i].legend(loc="lower right", frameon=False, fontsize=7)
        ax8[0, i].annotate(txt_annotate, (0.6, 0.05), xycoords="axes fraction", fontsize=6)

        ax8[1, i].set_xscale("log")
        ax8[1, i].set_xlabel("$K_{d}~[\mathrm{m^{-1}}]$")
        #ax8[1, i].legend(loc="lower right", frameon=False, fontsize=7)
        ax8[1, i].annotate(txt_annotate, (0.6, 0.05), xycoords="axes fraction", fontsize=6)

        ax8[2, i].set_xscale("log")
        ax8[2, i].set_xlabel("$a~[\mathrm{m^{-1}}]$")
        #ax8[2, i].legend(loc="lower right", frameon=False, fontsize=7)
        ax8[2, i].annotate(txt_annotate, (0.6, 0.05), xycoords="axes fraction", fontsize=6)

        ax8[0, i].set_title(wl_label[b], fontsize=10)

        if i == 0:
            xlims_axe[0, :] = ax8[0, i].get_xlim()
            xlims_axe[1, :] = ax8[1, i].get_xlim()
            xlims_axe[2, :] = ax8[2, i].get_xlim()
        else:
            xlims_axe[0, :] = np.minimum(xlims_axe[0, 0], ax8[0, i].get_xlim()[0]), np.maximum(xlims_axe[0, 1], ax8[0, i].get_xlim()[1])
            xlims_axe[1, :] = np.minimum(xlims_axe[1, 0], ax8[1, i].get_xlim()[0]), np.maximum(xlims_axe[1, 1], ax8[1, i].get_xlim()[1])
            xlims_axe[2, :] = np.minimum(xlims_axe[2, 0], ax8[2, i].get_xlim()[0]), np.maximum(xlims_axe[2, 1], ax8[2, i].get_xlim()[1])

    for i in range(3):
        ax8[0, i].set_xlim(xlims_axe[0, :])
        ax8[1, i].set_xlim(xlims_axe[1, :])
        ax8[2, i].set_xlim(xlims_axe[2, :])

    ax8[0, 0].set_ylabel("Depth [cm]")
    ax8[1, 0].set_ylabel("Depth [cm]")
    ax8[2, 0].set_ylabel("Depth [cm]")

    ax8 = draw_layers(ax8, snow_thickness, ice_freebord, ice_thickness)

    ax8[0, 0].set_ylim((-10, 105))
    ax8[0, 0].invert_yaxis()
    #ax8[0].set_xscale("log")
    #ax8[0].set_ylabel("Depth [cm]")


    [axe.text(-0.05, 1.05, "(" + string.ascii_lowercase[i] + ")", transform=axe.transAxes, size=10, weight='bold') for i, axe in enumerate(ax8.ravel())]


    # Figure 5
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()
    fig7.tight_layout()
    fig8.tight_layout()

    # Save figures
    fig1.savefig("figures/bdc_aops.pdf", format="pdf", dpi=300)
    fig1.savefig("figures/bdc_aops.png", format="png", dpi=300)

    fig7.savefig("figures/FigS1.pdf", format="pdf", dpi=300)
    fig7.savefig("figures/FigS1.png", format="png", dpi=300)

    fig8.savefig("figures/FigS2.pdf", format="pdf", dpi=300)
    fig8.savefig("figures/FigS2.png", format="png", dpi=300)

    plt.show()
