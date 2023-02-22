# -*- coding: utf-8 -*-
"""
Comparisons between mean cosines of the measurement vs. simulation.
"""
import pandas
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from source.radiance import RadClass


# Classes and functions
def get_Eudos_at_depth(pdf, depth, wavelength):
    """

    :param pdf:
    :param depth:
    :param wavelength:
    :return:
    """
    dlist = list(pdf["depths"])
    dlist = [round(num, 4) for num in dlist]
    i_depth = dlist.index(round(depth, 4)) + 1  # 0 is above the interface,

    Eu = pdf[f'Eu_{wavelength:.1f}'][i_depth]
    Ed = pdf[f'Ed_{wavelength:.1f}'][i_depth]
    Eo = pdf[f'Eo_{wavelength:.1f}'][i_depth]
    Eou = pdf[f'Eou_{wavelength:.1f}'][i_depth]
    Eod = pdf[f'Eod_{wavelength:.1f}'][i_depth]
    return Ed, Eu, Eo, Eod, Eou


def construct_irradiance_profile_HL(depths, path=r"C:\Users\Raphaël Larouche\PycharmProjects\HE60-PyMagister\to_raph\data\manual_opt\eudos_iops.csv"):
    """

    :param depths:
    :param path:
    :return:
    """

    # Pandas dataframe
    d_hl = pandas.read_csv(path)

    # Construct irradiance HL
    ed = np.zeros((depths.shape[0], 3))
    eo = ed.copy()
    eu = ed.copy()
    edo = ed.copy()
    euo = ed.copy()

    for i, d in enumerate(depths):
        for j, b in enumerate([600, 540, 480]):
            ed[i, j], eu[i, j], \
            eo[i, j], edo[i, j], euo[i, j] = get_Eudos_at_depth(d_hl, depth=d/100, wavelength=b)

    return ed, eu, eo, edo, euo


def dw_diffuse_attenuation_coefficients(depths, ed):
    """

    :param depths:
    :param ed:
    :return:
    """
    dz = np.diff(depths/100)
    return np.log(ed[:-1]/ed[1:])/dz


def kd_finite_difference(depths, ed):
    """

    :param depths:
    :param ed:
    :return:
    """
    dz = np.diff(depths/100)
    diff = ed[1:] - ed[:-1]
    sum = ed[1:] + ed[:-1]
    return (-2/sum) * (diff/dz), 0.5 * (depths[1:] + depths[:-1])


if __name__ == "__main__":

    # In situ
    #rc = RadClass(data_path="data/oden-08312018.h5")
    rc = RadClass(data_path="data/oden-08312018-fluo.h5")

    # Save gershun's law absorption coefficient
    a_oden_df = pandas.DataFrame(rc.mu_a.copy())
    a_oden_df = a_oden_df.set_index("depth")
    a_oden_df.to_csv("data/a_oden.csv")

    # Simul dort
    rc_sim = RadClass(data_path="data/dort-simulation.h5")

    # Construct irradiance HL
    ed_hl, eu_hl, eo_hl, edo_hl, euo_hl = construct_irradiance_profile_HL(rc.u_d["depth"], path="data/oden_fit_fluo/eudos_iops.csv")

    d_df = pandas.read_csv("data/oden_fit_fluo/eudos_iops.csv")
    g = np.ones(d_df["depths"][1:].shape[0]) * 0.99
    g[d_df["depths"][1:] < 0.20] = 0.85
    g[2.0 <= d_df["depths"][1:]] = 0.90

    # Mean cosines HL
    ud_hl = ed_hl / edo_hl
    uu_hl = eu_hl / euo_hl
    u_hl = (ed_hl - eu_hl) / eo_hl

    # enet, kd, gershun's law HL
    z_hr = np.arange(0, 201, 1)
    ed_hl_hr, eu_hl_hr, eo_hl_hr, _, _ = construct_irradiance_profile_HL(z_hr, path="data/oden_fit_fluo/eudos_iops.csv")
    enet_hl = ed_hl_hr - eu_hl_hr
    a_hl = -np.gradient(enet_hl, z_hr/100, axis=0, edge_order=2) * (1/eo_hl_hr)
    kd_hl = -np.gradient(ed_hl_hr, z_hr/100, axis=0, edge_order=2) * (1/ed_hl_hr)

    # Figure
    plt.style.use("../../figurestyle.mplstyle")
    fig1, ax1 = plt.subplots(1, 3, sharey=True, figsize=(6.136, 3.784))

    CMAR = matplotlib.cm.get_cmap("Reds", 10 + 1)
    CMAG = matplotlib.cm.get_cmap("Greens", 10 + 1)
    CMAB = matplotlib.cm.get_cmap("Blues", 10 + 1)

    #col = ["#d55e00", "#009e73", "#0072b2"]
    col = [CMAR(7), CMAG(7), CMAB(7)]

    for i, b in enumerate(["r", "g", "b"]):

        ax1[0].plot(rc.u_d[b], rc.u_d["depth"], linewidth=0.9, color=col[i])
        ax1[1].plot(rc.u_u[b], rc.u_u["depth"], linewidth=0.9, color=col[i])
        ax1[2].plot(rc.u[b], rc.u["depth"], linewidth=0.9, color=col[i])

        ax1[0].plot(rc_sim.u_d[b], rc_sim.u_d["depth"], linewidth=0.8, linestyle=":", color=col[i])
        ax1[1].plot(rc_sim.u_u[b], rc_sim.u_u["depth"], linewidth=0.8, linestyle=":", color=col[i])
        ax1[2].plot(rc_sim.u[b], rc_sim.u["depth"], linewidth=0.8, linestyle=":", color=col[i])

        ax1[0].plot(ud_hl[:, i], rc.u["depth"], linewidth=0.8, linestyle="--", color=col[i])
        ax1[1].plot(uu_hl[:, i], rc.u["depth"], linewidth=0.8, linestyle="--", color=col[i])
        ax1[2].plot(u_hl[:, i], rc.u["depth"], linewidth=0.8, linestyle="--", color=col[i])

    ax1[0].invert_yaxis()

    ax1[0].set_xlabel("$\overline{\mu_{d}}$")
    ax1[1].set_xlabel("$\overline{\mu_{u}}$")
    ax1[2].set_xlabel("$\overline{\mu}$")

    txt_annotate = "- camera\n: dort2002\n-- HL"
    ax1[0].annotate(txt_annotate, (0.6, 0.7), xycoords="axes fraction", fontsize=6)
    ax1[1].annotate(txt_annotate, (0.1, 0.7), xycoords="axes fraction", fontsize=6)
    ax1[2].annotate(txt_annotate, (0.6, 0.7), xycoords="axes fraction", fontsize=6)

    ax1[0].set_ylabel("Depth [cm]")
    fig1.tight_layout()

    # Figure 2
    fig2, ax2 = plt.subplots(3, 3, sharey=True, figsize=(6.6929, 6.6929 * 0.9))

    #wl_label = {"r": "603 nm", "g": "544 nm", "b": "484 nm"}
    wl_label = {"r": "600 nm", "g": "540 nm", "b": "480 nm"}
    hl_labels = {"r": "a_600.0", "g": "a_540.0", "b": "a_480.0"}

    txt_annotate = "- measurements\n-- simulations"

    for i, b in enumerate(["r", "g", "b"]):

        # Irradiances
        ax2[0, 0].plot(rc.ed[b], rc.ed["depth"], color=col[i], linewidth=0.9, label=wl_label[b])
        ax2[0, 1].plot(rc.eu[b], rc.eu["depth"], color=col[i], linewidth=0.9)
        ax2[0, 2].plot(rc.eo[b], rc.eo["depth"], color=col[i], linewidth=0.9)

        #ax2[0, 0].plot(rc.ed[b], rc.ed["depth"], color=col[i], linewidth=0.9, marker="o", markersize=2, linestyle="-", markeredgecolor=col[i], markerfacecolor="none", label=wl_label[b])
        #ax2[0, 1].plot(rc.eu[b], rc.eu["depth"], color=col[i], linewidth=0.9, marker="o", markersize=2, linestyle="-", markeredgecolor=col[i], markerfacecolor="none", label=wl_label[b])
        #ax2[0, 2].plot(rc.eo[b], rc.eo["depth"], color=col[i], linewidth=0.9, marker="o", markersize=2, linestyle="-", markeredgecolor=col[i], markerfacecolor="none", label=wl_label[b])

        ax2[0, 0].plot(ed_hl[:, i], rc.u_d["depth"], linewidth=1,  linestyle="--",  color=col[i])
        ax2[0, 1].plot(eu_hl[:, i], rc.u_d["depth"], linewidth=1,  linestyle="--",  color=col[i])
        ax2[0, 2].plot(eo_hl[:, i], rc.u_d["depth"], linewidth=1,  linestyle="--", color=col[i])

        ax2[0, i].text(-0.0, 1.05, "(" + string.ascii_lowercase[i] + ")", transform=ax2[0, i].transAxes, size=10, weight='bold')

        # Average cosines
        ax2[1, 0].plot(rc.u_d[b], rc.u_d["depth"], color=col[i], linewidth=0.9, label=wl_label[b])
        ax2[1, 1].plot(rc.u_u[b], rc.u_u["depth"],  color=col[i], linewidth=0.9)
        ax2[1, 2].plot(rc.u[b], rc.u["depth"], linewidth=0.9, color=col[i])

        ax2[1, 0].plot(ud_hl[:, i], rc.u["depth"], linewidth=0.8, linestyle="--", color=col[i])
        ax2[1, 1].plot(uu_hl[:, i], rc.u["depth"], linewidth=0.8, linestyle="--", color=col[i])
        ax2[1, 2].plot(u_hl[:, i], rc.u["depth"], linewidth=0.8, linestyle="--", color=col[i])

        ax2[1, i].text(-0.0, 1.05, "(" + string.ascii_lowercase[3 + i] + ")", transform=ax2[1, i].transAxes, size=10, weight='bold')

        # Enet, attenuation coefficient, gershun
        ax2[2, 0].plot(rc.ed[b] - rc.eu[b], rc.ed["depth"], linewidth=0.9, color=col[i], label=wl_label[b])
        ax2[2, 0].plot(enet_hl[:, i], z_hr, linestyle="--", linewidth=0.9, color=col[i])

        ax2[2, 1].plot(kd_hl[:, i][1:-1], z_hr[1:-1], linestyle="--", linewidth=0.9, color=col[i], alpha=0.7)
        ax2[2, 1].plot(rc.K_d[b], rc.K_d["depth"], linewidth=0.9, color=col[i], label=wl_label[b])

        ax2[2, 2].plot(rc.mu_a[b], rc.mu_a["depth"], linewidth=0.9, color=col[i], label=wl_label[b])
        ax2[2, 2].plot(d_df[hl_labels[b]][1:], d_df["depths"][1:] * 100, linestyle="--", color=col[i], linewidth=0.9)

        ax2[2, i].annotate(txt_annotate, (0.6, 0.5), xycoords="axes fraction", fontsize=6)
        ax2[2, i].text(-0.0, 1.05, "(" + string.ascii_lowercase[6+i] + ")", transform=ax2[2, i].transAxes, size=10, weight='bold')

    # First row param
    ax2[0, 0].set_xscale("log")
    ax2[0, 1].set_xscale("log")
    ax2[0, 2].set_xscale("log")

    ax2[0, 0].set_ylabel("Depth [cm]")
    ax2[0, 0].set_xlabel("$E_{d}[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax2[0, 1].set_xlabel("$E_{u}[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax2[0, 2].set_xlabel("$E_{o}[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")

    ax2[0, 0].legend(loc="best", frameon=False)

    ax2[0, 0].annotate(txt_annotate, (0.05, 0.8), xycoords="axes fraction", fontsize=6)
    ax2[0, 1].annotate(txt_annotate, (0.6, 0.05), xycoords="axes fraction", fontsize=6)
    ax2[0, 2].annotate(txt_annotate, (0.05, 0.8), xycoords="axes fraction", fontsize=6)

    # Second row param
    ax2[1, 2].set_xticks(np.arange(-0.1, 0.6, 0.1))
    ax2[1, 2].set_xlim((-0.01, 0.51))

    ax2[1, 0].set_ylabel("Depth [cm]")
    ax2[1, 0].set_xlabel("$\overline{\mu_{d}}$")
    ax2[1, 1].set_xlabel("$\overline{\mu_{u}}$")
    ax2[1, 2].set_xlabel("$\overline{\mu}$")

    ax2[1, 0].legend(loc="best", frameon=False)

    ax2[1, 0].annotate(txt_annotate, (0.6, 0.45), xycoords="axes fraction", fontsize=6)
    ax2[1, 1].annotate(txt_annotate, (0.1, 0.77), xycoords="axes fraction", fontsize=6)
    ax2[1, 2].annotate(txt_annotate, (0.6, 0.77), xycoords="axes fraction", fontsize=6)

    # Third row param
    ax2[2, 2].set_ylim((-5, 205))
    ax2[2, 2].invert_yaxis()
    ax2[2, 0].set_xscale("log")
    ax2[2, 0].set_ylabel("Depth [cm]")
    ax2[2, 0].set_xlabel("$(E_{d} - E_{u})~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")

    ax2[2, 1].set_xlabel("$K_{d}~[\mathrm{m^{-1}}]$")

    ax2[2, 2].set_xscale("log")
    ax2[2, 2].set_xlabel("$a~[\mathrm{m^{-1}}]$")
    ax2[2, 0].legend(loc="best", frameon=False)

    # IOPS
    fig3, ax3 = plt.subplots(1, 3, sharey=True, figsize=(6.6929, 6.6929 * 0.5))

    ax3[0].plot(d_df["a_600.0"][1:], d_df["depths"][1:] * 100, color=col[0], label="600 nm")
    ax3[0].plot(d_df["a_540.0"][1:], d_df["depths"][1:] * 100, color=col[1], label="540 nm")
    ax3[0].plot(d_df["a_480.0"][1:], d_df["depths"][1:] * 100, color=col[2], label="480 nm")

    ax3[1].plot(d_df["b_600.0"][1:], d_df["depths"][1:] * 100)

    ax3[2].plot(d_df["b_600.0"][1:] * (1 - g), d_df["depths"][1:] * 100)

    ax3[0].set_ylim((-5, 205))
    ax3[0].invert_yaxis()
    ax3[0].set_xlabel("$a~[\mathrm{m^{-1}}]$")
    ax3[0].set_ylabel("Depth [cm]")
    ax3[0].legend(loc="best", frameon=False)

    ax3[1].set_xlabel("$b~[\mathrm{m^{-1}}]$")
    ax3[2].set_xlabel("$b(1-g)~[\mathrm{m^{-1}}]$")

    # Figure 4
    fig4, ax4 = plt.subplots(1, 3, sharey=True, figsize=(6.6929, 6.6929 * 0.5))

    b = ["r", "g", "b"]
    for i in range(3):

        ax4[0].plot(rc.ed[b[i]] - rc.eu[b[i]], rc.ed["depth"], linewidth=0.9, color=col[i], label=list(wl_label.values())[i])
        ax4[0].plot(enet_hl[:, i], z_hr, linestyle="-.", linewidth=0.9, color=col[i])

        ax4[1].plot(kd_hl[:, i][1:-1], z_hr[1:-1], linestyle="-.", linewidth=0.9, color=col[i], alpha=0.7)
        ax4[1].plot(rc.K_d[b[i]], rc.K_d["depth"], linewidth=0.9, color=col[i], label=list(wl_label.values())[i])

        ax4[2].plot(a_hl[:, i][1:-1], z_hr[1:-1], linestyle="-.", linewidth=0.9, color=col[i], alpha=0.7)
        ax4[2].plot(rc.mu_a[b[i]], rc.mu_a["depth"], linewidth=0.9, color=col[i], label=list(wl_label.values())[i])

    ax4[2].plot(d_df["a_600.0"][1:], d_df["depths"][1:] * 100, linestyle=":", color=col[0], linewidth=0.9)
    ax4[2].plot(d_df["a_540.0"][1:], d_df["depths"][1:] * 100, linestyle=":", color=col[1], linewidth=0.9)
    ax4[2].plot(d_df["a_480.0"][1:], d_df["depths"][1:] * 100, linestyle=":", color=col[2], linewidth=0.9)

    ax4[2].set_ylim((-5, 205))
    ax4[0].invert_yaxis()
    ax4[0].set_xscale("log")
    ax4[0].set_ylabel("Depth [cm]")
    ax4[0].set_xlabel("$(E_{d} - E_{u})~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")

    ax4[1].set_xlabel("$K_{d}~[\mathrm{m^{-1}}]$")
    ax4[2].set_xscale("log")
    ax4[2].set_xlabel("$a~[\mathrm{m^{-1}}]$")

    ax4[0].annotate(txt_annotate, (0.6, 0.45), xycoords="axes fraction", fontsize=6)
    ax4[1].annotate(txt_annotate, (0.6, 0.45), xycoords="axes fraction", fontsize=6)
    ax4[2].annotate("- Gershun measurements\n-. Gershun simulations\n: $a$ simulations", (0.53, 0.13), xycoords="axes fraction", fontsize=6)

    ax4[0].legend(loc="best", frameon=False)
    ax4[1].legend(loc="best", frameon=False)
    ax4[2].legend(loc="best", frameon=False)

    # Errors
    all_ed = rc.ed.view((rc.ed.dtype[0], 4))[:, :3]
    all_eu = rc.eu.view((rc.eu.dtype[0], 4))[:, :3]
    all_eo = rc.eo.view((rc.eo.dtype[0], 4))[:, :3]

    err_ed = 100 * ((ed_hl - all_ed) / ed_hl)
    err_eu = 100 * ((eu_hl - all_eu) / eu_hl)
    err_eo = 100 * ((eo_hl - all_eo) / eo_hl)

    upd_ed = 200 * ((ed_hl - all_ed) / (ed_hl + all_ed))
    upd_eu = 200 * ((eu_hl - all_eu) / (eu_hl + all_eu))
    upd_eo = 100 * ((eo_hl - all_eo) / (eo_hl + all_eo))

    # Savefig
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()

    fig2.subplots_adjust(wspace=0.1, hspace=0.55)

    fig2.savefig("figures/oden_aops_fluo.pdf", format="pdf", dpi=300)
    fig2.savefig("figures/oden_aops_fluo.png", format="png", dpi=300)

    #fig3.savefig("figures/oden_hl_iops.png", format="png", dpi=300)
    #fig4.savefig("figures/oden_gershun.png", format="png", dpi=300)

    plt.show()
