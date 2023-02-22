# -*- coding: utf-8 -*-
"""

"""
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
        p = "data/bdc_2_fit_fluo/eudos_iops.csv"
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
    plt.style.use("../../figurestyle.mplstyle")
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
        ax[0, 0].plot(radcl.ed[b][mask_d], radcl.ed["depth"][mask_d], linewidth=0.9, color=col[i], label=wl_label[b])
        ax[0, 1].plot(radcl.eu[b][mask_d], radcl.eu["depth"][mask_d], linewidth=0.9, color=col[i], label=wl_label[b])
        ax[0, 2].plot(radcl.eo[b][mask_d], radcl.eo["depth"][mask_d], linewidth=0.9, color=col[i], label=wl_label[b])

        #ax[0, 0].plot(radcl.ed[b], radcl.ed["depth"], linestyle="none", markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])
        #ax[0, 1].plot(radcl.eu[b], radcl.eu["depth"], linestyle="none",  markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])
        #ax[0, 2].plot(radcl.eo[b], radcl.eo["depth"], linestyle="none",  markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])

        # Hydrolight
        ax[0, 0].plot(ed_hl[:, i], z_depth, linewidth=0.9,  linestyle="--",  color=col[i])
        ax[0, 1].plot(eu_hl[:, i], z_depth, linewidth=0.9,  linestyle="--",  color=col[i])
        ax[0, 2].plot(eo_hl[:, i], z_depth, linewidth=0.9,  linestyle="--", color=col[i])

        # Average cosines
        ax[1, 0].plot(radcl.u_d[b][mask_d], radcl.u_d["depth"][mask_d], linewidth=0.9, color=col[i], label=wl_label[b])
        ax[1, 1].plot(radcl.u_u[b][mask_d], radcl.u_u["depth"][mask_d], linewidth=0.9, color=col[i], label=wl_label[b])
        ax[1, 2].plot(radcl.u[b][mask_d], radcl.u["depth"][mask_d], linewidth=0.9, color=col[i], label=wl_label[b])

        #ax[1, 0].plot(radcl.u_d[b], radcl.u_d["depth"], linestyle="none", markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])
        #ax[1, 1].plot(radcl.u_u[b], radcl.u_u["depth"], linestyle="none", markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])
        #ax[1, 2].plot(radcl.u[b], radcl.u["depth"], linestyle="none", markersize=3, marker="o", markerfacecolor="none", markeredgecolor=col[i], label=wl_label[b])

        ax[1, 0].plot(ud_hl[:, i], z_depth, linewidth=0.8, linestyle="--", color=col[i])
        ax[1, 1].plot(uu_hl[:, i], z_depth, linewidth=0.8, linestyle="--", color=col[i])
        ax[1, 2].plot(u_hl[:, i], z_depth, linewidth=0.8, linestyle="--", color=col[i])

        ax[2, 0].plot(radcl.ed[b][mask_d] - radcl.eu[b][mask_d], radcl.ed["depth"][mask_d], linewidth=0.9, color=col[i], label=wl_label[b])
        ax[2, 0].plot(enet_hl[:, i], z_depth, linestyle="--", linewidth=0.9, color=col[i])
        ax[2, 1].plot(kd_hl[:, i], z_depth, linestyle="--", linewidth=0.9, color=col[i])

        ax[2, 1].plot(radcl.K_d[b][mask_d], radcl.K_d["depth"][mask_d], linewidth=0.9, color=col[i], label=wl_label[b])
        ax[2, 2].plot(a_hl[:, i], z_depth, linestyle="--", linewidth=0.9, color=col[i])

        ax[2, 2].plot(radcl.mu_a[b][mask_d], radcl.mu_a["depth"][mask_d], linewidth=0.9, color=col[i], label=wl_label[b])

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
        p = "data/bdc_2_fit_fluo/eudos_iops.csv"
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


if __name__ == "__main__":

    ifb_st2 = get_ice_freeboard("data/station_2_data.txt")
    ifb_st3 = get_ice_freeboard("data/station_3_data.txt")

    label_st2 = np.array(list(create_label("data/station_2_data.txt").keys()))
    label_st3 = np.array(list(create_label("data/station_3_data.txt").keys()))

    #rc_st2 = rfct.RadClass(data_path="data/baiedeschaleurs-03232022.h5", station="station_2", data_type="camera", freeboard=0.0)
    rc_st2 = rfct.RadClass(data_path="data/baiedeschaleurs-03232022-fluo.h5", station="station_2", data_type="camera", freeboard=0.0)
    #rc_st2 = rfct.RadClass(data_path="data/baiedeschaleurs-03232022.h5", station="station_2", data_type="camera", freeboard=0.11)
    rc_st3 = rfct.RadClass(data_path="data/baiedeschaleurs-03232022.h5", station="station_3", data_type="camera", freeboard=0.11)

    # Figures
    plt.style.use("../../figurestyle.mplstyle")
    #fig1, ax1 = graph_aops(rc_st2, 0.1)
    fig1, ax1 = graph_aops(rc_st2, ifb_st2)
    #fig2, ax2 = graph_aops(rc_st3, 0.1)
    #fig2, ax2 = graph_aops(rc_st3, ifb_st3)
    fig2, ax2 = plt.subplots(3, 3, sharey=True, figsize=(6.6929, 6.6929 * 0.9))

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
    data_st2 = pandas.read_csv("data/bdc_2_fit/eudos_iops.csv")
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

    # Figure 5
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()

    # Save figures
    fig1.savefig("figures/bdc_aops.pdf", format="pdf", dpi=300)
    fig1.savefig("figures/bdc_aops.png", format="png", dpi=300)

    plt.show()
