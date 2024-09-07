# -*- coding: utf-8 -*-
"""

"""
import os
import string
import numpy as np
import matplotlib.pyplot as plt

import field.oden2018.oden_dort_vs_hl as rad_func
from bdc_process_stations import create_label, get_ice_freeboard


# Function and classes
def graph_radiance_pts_vs_fit(rc_obj, hl_data, depths, wl_cam):
    """

    :return:
    """

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(6.6929, 6.6929 * 0.75))

    # MUPD
    mre_profile = np.empty((depths.shape[0], 3))
    all_rad_cam = np.zeros((depths.shape[0], 181, 3))
    all_rad_sim = np.zeros((depths.shape[0], 181, 3))

    wl_hl = np.array([480, 540, 600])
    #wl_cam = np.array([484, 544, 603])
    #wl_cam = np.array([480, 540, 600])

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
                ax[b].plot(zen_cam[::5], rad_cam[::5], marker="o", markersize=2, markerfacecolor="none", markeredgecolor="grey", linestyle="none", color="grey", alpha=1, label="Measurements")
                ax[b].plot(zen_sim, rad_sim, linestyle="-", color="grey", alpha=1, label="HL simulations")
            else:

                ax[b].plot(zen_cam[::5], rad_cam[::5], marker="o", markersize=2, markerfacecolor="none", markeredgecolor="grey", linestyle="none", color="grey", alpha=1)
                ax[b].plot(zen_sim, rad_sim, linestyle="-", color="grey", alpha=1)

            angle_txt = 125.0  # degree
            #angle_txt = 180.0 # degree
            a_angle_txt = np.argwhere(np.round(zen_cam, 3) == angle_txt)

            if i == depths.shape[0] - 1:
                coordy_txt = rad_cam[a_angle_txt] * (1 - 0.35)
                ax[b].annotate(f"{de} cm", (angle_txt, coordy_txt), fontsize=5, weight="bold")
            else:
                coordy_txt = rad_cam[a_angle_txt] * (1 + 0.05)
                ax[b].annotate(f"{de} cm", (angle_txt, coordy_txt), fontsize=5, weight="bold")

            ax[b].set_yscale("log")
            ax[b].set_xlim((20, 160))
            ax[b].set_yscale("log")

    # Average mre
    mupd = mre_profile.mean(axis=0) * 100
    rmse_tot = rad_func.rad_rmse(all_rad_cam, all_rad_sim)

    ax[0].set_xticks(np.arange(0, 220, 40))
    ax[0].set_xlim((17, 163))

    ax[0].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")

    ax[0].set_xlabel("Zenith [˚]")
    ax[1].set_xlabel("Zenith [˚]")
    ax[2].set_xlabel("Zenith [˚]")

    ax[0].legend(loc=1, fontsize=7, frameon=False)

    ax[0].annotate("MUAPD = {0:.2f} %\nRMSE = {1:.2f}%".format(mupd[0], rmse_tot[0]), (0.05, 0.1), xycoords="axes fraction", fontsize=6)
    ax[1].annotate("MUAPD = {0:.2f} %\nRMSE = {1:.2f}%".format(mupd[1], rmse_tot[1]), (0.05, 0.1), xycoords="axes fraction", fontsize=6)
    ax[2].annotate("MUAPD = {0:.2f} %\nRMSE = {1:.2f}%".format(mupd[2], rmse_tot[2]), (0.05, 0.1), xycoords="axes fraction", fontsize=6)

    return fig, ax


if __name__ == "__main__":

    # Gen path
    gen_path = os.path.dirname(__file__)

    # Freeboard
    ifb_st2 = get_ice_freeboard("data/station_2_data.txt")
    ifb_st3 = get_ice_freeboard("data/station_3_data.txt")

    label_st2 = np.array(list(create_label("data/station_2_data.txt").keys()))
    label_st3 = np.array(list(create_label("data/station_3_data.txt").keys()))

    zd_st2 = rad_func.load_zenith_radiance(path=r"data/super_recu_bdc_fit_final")
    zd_st3 = rad_func.load_zenith_radiance(path=r"data/bdc_3_fit")

    rc_st2 = rad_func.RadClass(data_path=r"data/baiedeschaleurs-03232022-imf-fluo.h5", station="station_2", data_type="camera", freeboard=ifb_st2, wl_dct={484: 2, 544: 1, 603:0})
    rc_st3 = rad_func.RadClass(data_path=r"data/baiedeschaleurs-03232022.h5", station="station_3", data_type="camera", freeboard=ifb_st3, wl_dct={484: 2, 544: 1, 603:0})

    mask_st2 = label_st2 >= ifb_st2
    mask_st3 = label_st3 >= ifb_st3

    fig1, ax1, _, _ = rad_func.graph_radiance_cam_vs_simulations(rc_st2, zd_st2, label_st2[mask_st2], wl_cam=np.array([484, 544, 603]))
    fig2, ax2, _, _ = rad_func.graph_radiance_cam_vs_simulations(rc_st3, zd_st3, label_st3[mask_st3], wl_cam=np.array([484, 544, 603]))

    fig3, ax3 = graph_radiance_pts_vs_fit(rc_st2, zd_st2, label_st2[mask_st2], wl_cam=np.array([484, 544, 603]))

    fig3.subplots_adjust(wspace=0.07)
    ax3[0].text(-0.0, 1.02, "(" + string.ascii_lowercase[0] + ")", transform=ax3[0].transAxes, size=10, weight='bold')
    ax3[1].text(-0.0, 1.02, "(" + string.ascii_lowercase[1] + ")", transform=ax3[1].transAxes, size=10, weight='bold')
    ax3[2].text(-0.0, 1.02, "(" + string.ascii_lowercase[2] + ")", transform=ax3[2].transAxes, size=10, weight='bold')

    fig1.tight_layout()
    fig2.tight_layout()
    #fig3.tight_layout()

    rc_st2_fluo = rad_func.RadClass(data_path="data/baiedeschaleurs-03232022-imf-fluo.h5", station="station_2", data_type="camera", freeboard=ifb_st2, wl_dct={480: 2, 540: 1, 600: 0})
    zd_st2_fluo = rad_func.load_zenith_radiance(path=r"data/super_recu_bdc_fit_final")

    fig4, ax4 = graph_radiance_pts_vs_fit(rc_st2_fluo, zd_st2_fluo, label_st2[mask_st2][:-1], wl_cam=np.array([480, 540, 600]))

    fig3.savefig("figures/st2_radiance_profile.png", dpi=300, format="png")
    fig3.savefig("figures/st2_radiance_profile.pdf", dpi=300, format="pdf")

