# -*- coding: utf-8 -*-
"""
Figure angular radiance distributions in TC paper.
"""
# Module importation
import string
import numpy as np
import matplotlib.pyplot as plt

import field.oden2018.oden_dort_vs_hl as rad_func
from source.radiance import RadClass
from baiedeschaleurs2022.bdc_process_stations import create_label, get_ice_freeboard


# Functions
def graph_radiance_pts_vs_fit(rc_obj, hl_data, depths, fig_ax=None):
    """

    :return:
    """

    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(6.6929, 6.6929 * 0.75))

    # MUPD
    mre_profile = np.empty((depths.shape[0], 3))
    all_rad_cam = np.zeros((depths.shape[0], 181, 3))
    all_rad_sim = np.zeros((depths.shape[0], 181, 3))

    wl_hl = np.array([480, 540, 600])
    #wl_cam = np.array([484, 544, 603])
    wl_cam = np.array([480, 540, 600])

    for i, de in enumerate(depths):

        for b, wave in enumerate(zip(wl_hl, wl_cam)):

            wave_hl, wave_cam = wave
            zen_sim, rad_sim = rad_func.get_zenith_radiance_profile_at_depth(hl_data, depth=de/100, wavelength=wave_hl, interpolate=True)
            zen_cam, rad_cam = rc_obj.get_radiance_avg_at_depth_wl(depth=de, wl=wave_cam, smooth=False)  # Change between smooth and raw

            # MUPD
            ref_values = 0.5 * (rad_sim + rad_cam)
            rel_err = (rad_sim - rad_cam) / ref_values
            mre_profile[i, b] = np.nanmean(rel_err)

            all_rad_cam[i, :, b] = rad_cam
            all_rad_sim[i, :, b] = rad_sim

            if i == 0:
                ax[b].plot(zen_cam, rad_cam, linestyle=":", color="grey", alpha=1, label="Measurements")
                #ax[b].plot(zen_cam[::5], rad_cam[::5], marker="o", markersize=2, markerfacecolor="none", markeredgecolor="grey", linestyle="none", color="grey", alpha=1, label="Measurements")
                ax[b].plot(zen_sim, rad_sim, linestyle="-", color="grey", alpha=1, label="HL simulations")
            else:
                ax[b].plot(zen_cam, rad_cam, linestyle=":", color="grey", alpha=1)
                #ax[b].plot(zen_cam[::5], rad_cam[::5], marker="o", markersize=2, markerfacecolor="none", markeredgecolor="grey", linestyle="none", color="grey", alpha=1)
                ax[b].plot(zen_sim, rad_sim, linestyle="-", color="grey", alpha=1)

            ax[b].set_yscale("log")
            ax[b].set_xlim((20, 160))
            ax[b].set_yscale("log")

    for i, de in enumerate(depths):

        zen_sim, rad_sim = rad_func.get_zenith_radiance_profile_at_depth(hl_data, depth=de / 100, wavelength=600.0, interpolate=True)

        a_angle_txt = np.argwhere(np.round(zen_sim, 3) == 160.0)

        ylims = ax[2].get_ylim()
        if i == depths.shape[0] - 1:
            coordy_txt = rad_sim[a_angle_txt] * (1 - 0.2)
            lcoord = (np.log10(coordy_txt) - np.log10(ylims[0])) / (np.log10(ylims[1]) - np.log10(ylims[0]))
            ax[2].text(1.01, lcoord, f"{de} cm", transform=ax[2].transAxes, size=5, weight='bold')
        else:
            coordy_txt = rad_sim[a_angle_txt] * (1 + 0.01)
            lcoord = (np.log10(coordy_txt) - np.log10(ylims[0])) / (np.log10(ylims[1]) - np.log10(ylims[0]))
            ax[2].text(1.01, lcoord, f"{de} cm", transform=ax[2].transAxes, size=5, weight='bold')

    # Average mre
    mupd = mre_profile.mean(axis=0) * 100
    rmse_tot = rad_func.rad_rmse(all_rad_cam, all_rad_sim)

    ax[0].set_xticks(np.arange(0, 220, 40))
    ax[0].set_xlim((17, 163))

    ax[0].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")

    ax[1].legend(loc=3, fontsize=7, frameon=False)

    ax[0].annotate("MUAPD = {0:.2f} %\nRMSE = {1:.2f}%".format(mupd[0], rmse_tot[0]), (0.45, 0.9), xycoords="axes fraction", fontsize=6)
    ax[1].annotate("MUAPD = {0:.2f} %\nRMSE = {1:.2f}%".format(mupd[1], rmse_tot[1]), (0.45, 0.9), xycoords="axes fraction", fontsize=6)
    ax[2].annotate("MUAPD = {0:.2f} %\nRMSE = {1:.2f}%".format(mupd[2], rmse_tot[2]), (0.45, 0.9), xycoords="axes fraction", fontsize=6)

    ax[0].annotate(f"{wl_cam[0]} nm", (0.20, 0.2), xycoords="axes fraction", fontsize=6)
    ax[1].annotate(f"{wl_cam[1]} nm", (0.20, 0.2), xycoords="axes fraction", fontsize=6)
    ax[2].annotate(f"{wl_cam[2]} nm", (0.20, 0.2), xycoords="axes fraction", fontsize=6)

    return fig, ax


if __name__ == "__main__":

    # Freeboard
    ifb_st2 = get_ice_freeboard("baiedeschaleurs2022/data/station_2_data.txt")
    label_st2 = np.array(list(create_label("baiedeschaleurs2022/data/station_2_data.txt").keys()))

    # BDC
    rc_st2 = RadClass(data_path="baiedeschaleurs2022/data/baiedeschaleurs-03232022-fluo.h5", station="station_2", data_type="camera", freeboard=ifb_st2)
    #zd_st2 = rad_func.load_zenith_radiance(path=r"baiedeschaleurs2022/data/bdc_2_fit")
    zd_st2 = rad_func.load_zenith_radiance(path=r"baiedeschaleurs2022/data/bdc_2_fit_fluo")

    # ODEN
    rc = RadClass(data_path="oden2018/data/oden-08312018-fluo.h5")
    #zd = rad_func.load_zenith_radiance(path=r"oden2018/data/oden_fit")
    zd = rad_func.load_zenith_radiance(path=r"oden2018/data/oden_fit_fluo")
    d_oden = np.arange(20, 180, 20).astype(float)
    #d_oden = np.arange(0, 220, 20).astype(float)
    #d_oden = np.delete(d_oden, np.where(d_oden == 180))

    # Mask
    mask_st2 = (label_st2 >= ifb_st2) & (label_st2 < 90.0)

    # Figures
    fig1, ax1 = graph_radiance_pts_vs_fit(rc_st2, zd_st2, label_st2[mask_st2])

    # Figure 2
    fig2, ax2 = plt.subplots(2, 3, sharey="row", sharex=True, figsize=(6.6929 * 0.9, 6.6929 * 0.85))

    fig2, ax2[1, :] = graph_radiance_pts_vs_fit(rc_st2, zd_st2, label_st2[mask_st2], fig_ax=(fig2, ax2[1, :]))
    fig2, ax2[0, :] = graph_radiance_pts_vs_fit(rc, zd, d_oden, fig_ax=(fig2, ax2[0, :]))

    ax2[1, 0].set_xlabel("Zenith [˚]")
    ax2[1, 1].set_xlabel("Zenith [˚]")
    ax2[1, 2].set_xlabel("Zenith [˚]")

    ax2[0, 0].text(-0.0, 1.02, "(" + string.ascii_lowercase[0] + ")", transform=ax2[0, 0].transAxes, size=10, weight='bold')
    ax2[0, 1].text(-0.0, 1.02, "(" + string.ascii_lowercase[1] + ")", transform=ax2[0, 1].transAxes, size=10, weight='bold')
    ax2[0, 2].text(-0.0, 1.02, "(" + string.ascii_lowercase[2] + ")", transform=ax2[0, 2].transAxes, size=10, weight='bold')

    ax2[1, 0].text(-0.0, 1.02, "(" + string.ascii_lowercase[3] + ")", transform=ax2[1, 0].transAxes, size=10, weight='bold')
    ax2[1, 1].text(-0.0, 1.02, "(" + string.ascii_lowercase[4] + ")", transform=ax2[1, 1].transAxes, size=10, weight='bold')
    ax2[1, 2].text(-0.0, 1.02, "(" + string.ascii_lowercase[5] + ")", transform=ax2[1, 2].transAxes, size=10, weight='bold')

    fig2.tight_layout()

    fig2.savefig("baiedeschaleurs2022/figures/rad_both_field_fluo.pdf", format="pdf", dpi=300)
    fig2.savefig("baiedeschaleurs2022/figures/rad_both_field_fluo.png", format="png", dpi=300)


    plt.show()