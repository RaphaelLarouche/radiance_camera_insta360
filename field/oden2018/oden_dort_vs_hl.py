# -*- coding: utf-8 -*-
"""
Compare fit results from DORT20002 and HL.
"""
import pickle
import numpy as np
import string
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


from source.radiance import RadClass
from source.processing import FigureFunctions
from field.oden2018.script_bastian import colorbardepth, build_cmap_2cond_color, graph_cam_vs_simulations, open_radiance_data


# Classes and functions
def get_zenith_radiance_profile_at_depth(zen_data, depth, wavelength, interpolate=True):
    """

    :param depth:
    :param wavelength:
    :param interpolate:
    :return:
    """
    zenith_radiance, depths, run_bands = zen_data
    i_wavelength = list(run_bands).index(wavelength)
    try:
        #i_depth = list(depths).index(depth)
        i_depth = int(depth * 100)   # Ok but if depth is other than 0.01 increment, it will not work.
    except:
        if depth == -1.:  # Depth -1 is the incoming radiation, just above the interface
            i_depth = -1
        else:
            print(f"Warning: Could not find resquested depth ({depth}) in: get_zenith_radiance_profile_at_depth")
    #phi_angles = [0., 10., 20., 30., 40, 50., 60., 70., 80., 87.5,
    #              92.5, 100., 110., 120., 130., 140., 150., 160., 170., 180.]  # Angles for which radiance is known
    phi_angles = [0., 10., 20., 30., 40, 50., 60., 70., 80.,
                  90., 100., 110., 120., 130., 140., 150., 160., 170., 180.]  # Angles for which radiance is known
    zenith_radiance = zenith_radiance[i_depth + 1, :, i_wavelength] #/ 1.355 ** 2  # TODO: Check to remove that ?

    if interpolate:  # cubic interpolation
        f = interp1d(phi_angles, zenith_radiance, kind='cubic')
        x_new_angles = np.arange(181)
        y_new_radiance = f(x_new_angles)
        return x_new_angles, y_new_radiance
    else:
        return phi_angles, zenith_radiance


def load_zenith_radiance(path):
    """

    :param path:
    :return:
    """
    with open(path + "\hermes.pickle", 'rb') as handle:
        hermes = pickle.load(handle)

    depths = hermes['zetanom']
    run_bands = hermes['run_bands']

    zenith_radiance = np.zeros((len(depths) + 1, 19, len(run_bands)))  # 3D array to store [depth, zenith angle, wvlgth]
    raw_zenith_radiance = np.loadtxt(path + r"\zenith_profiles.txt")

    for i, wavelength in enumerate(run_bands):
        to_be_reshaped = raw_zenith_radiance[raw_zenith_radiance[:, 2] == wavelength, :]
        total_radiance_image = to_be_reshaped[:, 3].reshape(len(depths) + 1, 19)  # third row = total radiance
        zenith_radiance[:, :, i] = total_radiance_image

    return zenith_radiance, depths, run_bands


def graph_radiance_cam_vs_simulations_pts(radclass_obj, hl_data, depths):

    # Figure creation
    fig_func = FigureFunctions()
    fig = plt.figure(figsize=(6.6929, 5.74))

    #a00 = fig.add_subplot(3, 3, 1)
    #a01 = fig.add_subplot(3, 3, 2, sharex=a00, sharey=a00)
    #a02 = fig.add_subplot(3, 3, 3, sharex=a00)

    #a10 = fig.add_subplot(3, 3, 4, sharex=a00, sharey=a00)
    #a10 = fig.add_subplot(3, 3, 4, sharex=a00)
    #a11 = fig.add_subplot(3, 3, 5, sharex=a00, sharey=a00)
    #a12 = fig.add_subplot(3, 3, 6, sharex=a00, sharey=a02)

    #a20 = fig.add_subplot(3, 3, 7, sharex=a00)
    #a20 = fig.add_subplot(3, 3, 7, sharex=a00, sharey=a00)
    #a21 = fig.add_subplot(3, 3, 8, sharex=a00, sharey=a00)
    #a22 = fig.add_subplot(3, 3, 9, sharex=a00, sharey=a02)

    #ax = np.array([[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]])

    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(6.6929, 5.74))

    # Build colorbar
    depth_color = depths.copy().astype(int)

    colo_reds = build_cmap_2cond_color("Reds", depth_color)
    colo_greens = build_cmap_2cond_color("Greens", depth_color)
    colo_blues = build_cmap_2cond_color("Blues", depth_color)

    cm_it_r = iter(colo_reds(np.arange(0, colo_reds.N)))
    cm_it_g = iter(colo_greens(np.arange(0, colo_greens.N)))
    cm_it_b = iter(colo_blues(np.arange(0, colo_blues.N)))

    colorbardepth(fig, ax[0], colo_reds, depth_color)
    colorbardepth(fig, ax[1], colo_greens, depth_color)
    colorbardepth(fig, ax[2], colo_blues, depth_color)

    # MUPD
    mre_profile = np.empty((depths.shape[0], 3))
    all_rad_cam = np.zeros((depths.shape[0], 181, 3))
    all_rad_sim = np.zeros((depths.shape[0], 181, 3))
    #rmse_profile = mre_profile.copy()

    wl_hl = np.array([600, 540, 480])
    wl_cam = np.array([603, 544, 484])

    for i, de in enumerate(depths):

        # Radiances

        # Color increment
        col_r = next(cm_it_r)
        col_g = next(cm_it_g)
        col_b = next(cm_it_b)

        for b, wave in enumerate(zip(wl_hl, wl_cam)):

            wave_hl, wave_cam = wave
            zen_sim, rad_sim = get_zenith_radiance_profile_at_depth(hl_data, depth=de/100,
                                                                    wavelength=wave_hl, interpolate=True)
            zen_cam, rad_cam = radclass_obj.get_radiance_avg_at_depth_wl(depth=de, wl=wave_cam, smooth=False)  # Change between smooth and raw

            # MUPD
            ref_values = 0.5 * (rad_sim + rad_cam)
            rel_err = (rad_sim - rad_cam) / ref_values
            mre_profile[i, b] = np.nanmean(rel_err)

            all_rad_cam[i, :, b] = rad_cam
            all_rad_sim[i, :, b] = rad_sim

            if b == 0:
                curr_col = col_r
            elif b == 1:
                curr_col = col_g
            else:
                curr_col = col_b

            ax[b].plot(zen_cam[::5], rad_cam[::5], marker="o", markersize=3, markerfacecolor="none", markeredgecolor=curr_col, linestyle="none", color=curr_col)
            ax[b].plot(zen_sim, rad_sim, linestyle="-", color=curr_col)
            #ax[b, 0].plot(zen_cam, rad_cam, linestyle="-", color=curr_col)
            #ax[b, 1].plot(zen_sim, rad_sim, linestyle="-.", color=curr_col)
            #ax[b, 2].plot(zen_sim, rel_err * 100, color=curr_col)

            ax[b].set_yscale("log")
            ax[b].set_xlim((20, 160))
            ax[b].set_yscale("log")

            #ax[b, 0].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
            #ax[b, 1].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
            #ax[b, 2].set_ylabel("relative error [%]")

    # Average mre
    mupd = mre_profile.mean(axis=0) * 100
    rmse_tot = rad_rmse(all_rad_cam, all_rad_sim)

    ax[0].set_xlabel("Zenith [˚]")
    ax[1].set_xlabel("Zenith [˚]")
    ax[2].set_xlabel("Zenith [˚]")

    ax[0].text(25, np.round(np.max(mre_profile * 100)) + 7, "MUAPD = {0:.2f} %\n RMSE = {1:.2f}%".format(mupd[0], rmse_tot[0]), fontsize=6)
    ax[1].text(25, np.round(np.max(mre_profile * 100)) + 7, "MUAPD = {0:.2f} %\n RMSE = {1:.2f}%".format(mupd[1], rmse_tot[1]), fontsize=6)
    ax[2].text(25, np.round(np.max(mre_profile * 100)) + 7, "MUAPD = {0:.2f} %\n RMSE = {1:.2f}%".format(mupd[2], rmse_tot[2]), fontsize=6)

    fig.tight_layout()

    return fig, ax, all_rad_cam, all_rad_sim


def graph_radiance_cam_vs_simulations(radclass_obj, hl_data, depths, sm=False):

    # Figure creation
    fig = plt.figure(figsize=(6.6929, 5.74))

    a00 = fig.add_subplot(3, 3, 1)
    a01 = fig.add_subplot(3, 3, 2, sharex=a00, sharey=a00)
    a02 = fig.add_subplot(3, 3, 3, sharex=a00)

    a10 = fig.add_subplot(3, 3, 4, sharex=a00, sharey=a00)
    a11 = fig.add_subplot(3, 3, 5, sharex=a00, sharey=a00)
    a12 = fig.add_subplot(3, 3, 6, sharex=a00, sharey=a02)

    a20 = fig.add_subplot(3, 3, 7, sharex=a00, sharey=a00)
    a21 = fig.add_subplot(3, 3, 8, sharex=a00, sharey=a00)
    a22 = fig.add_subplot(3, 3, 9, sharex=a00, sharey=a02)

    ax = np.array([[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]])

    # Build colorbar
    depth_color = depths.copy().astype(int)

    colo_reds = build_cmap_2cond_color("Reds", depth_color)
    colo_greens = build_cmap_2cond_color("Greens", depth_color)
    colo_blues = build_cmap_2cond_color("Blues", depth_color)

    cm_it_r = iter(colo_reds(np.arange(0, colo_reds.N)))
    cm_it_g = iter(colo_greens(np.arange(0, colo_greens.N)))
    cm_it_b = iter(colo_blues(np.arange(0, colo_blues.N)))

    colorbardepth(fig, ax[0, 2], colo_reds, depth_color)
    colorbardepth(fig, ax[1, 2], colo_greens, depth_color)
    colorbardepth(fig, ax[2, 2], colo_blues, depth_color)

    # MUPD
    mre_profile = np.empty((depths.shape[0], 3))
    all_rad_cam = np.zeros((depths.shape[0], 181, 3))
    all_rad_sim = np.zeros((depths.shape[0], 181, 3))

    wl_hl = np.array([600, 540, 480])
    #wl_cam = np.array([603, 544, 484])
    wl_cam = np.array([600, 540, 480])

    for i, de in enumerate(depths):

        # Radiances

        # Color increment
        col_r = next(cm_it_r)
        col_g = next(cm_it_g)
        col_b = next(cm_it_b)

        for b, wave in enumerate(zip(wl_hl, wl_cam)):

            wave_hl, wave_cam = wave
            zen_sim, rad_sim = get_zenith_radiance_profile_at_depth(hl_data, depth=de/100, wavelength=wave_hl, interpolate=True)
            zen_cam, rad_cam = radclass_obj.get_radiance_avg_at_depth_wl(depth=de, wl=wave_cam, smooth=sm)  # Change between smooth and raw

            # MUPD
            ref_values = 0.5 * (rad_sim + rad_cam)
            rel_err = (rad_sim - rad_cam) / ref_values
            mre_profile[i, b] = np.nanmean(rel_err)

            all_rad_cam[i, :, b] = rad_cam
            all_rad_sim[i, :, b] = rad_sim

            if b == 0:
                curr_col = col_r
            elif b == 1:
                curr_col = col_g
            else:
                curr_col = col_b

            ax[b, 0].plot(zen_cam, rad_cam, linestyle="-", color=curr_col)
            ax[b, 1].plot(zen_sim, rad_sim, linestyle="-.", color=curr_col)
            ax[b, 2].plot(zen_sim, rel_err * 100, color=curr_col)

            ax[b, 0].set_yscale("log")
            ax[b, 1].set_yscale("log")

            ax[b, 0].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
            ax[b, 1].set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
            ax[b, 2].set_ylabel("relative error [%]")

    # Average mre
    mupd = mre_profile.mean(axis=0) * 100
    rmse_tot = rad_rmse(all_rad_cam, all_rad_sim)

    ax[2, 0].set_xticks(np.arange(20, 220, 40))
    ax[2, 0].set_xlim((19, 161))

    ax[2, 0].set_xlabel("Zenith [˚]")
    ax[2, 1].set_xlabel("Zenith [˚]")
    ax[2, 2].set_xlabel("Zenith [˚]")

    ax[0, 2].text(25, np.round(np.max(mre_profile * 100)) + 7, "MUPD = {0:.2f} %\n RMSE = {1:.2f}%".format(mupd[0], rmse_tot[0]), fontsize=6)
    ax[1, 2].text(25, np.round(np.max(mre_profile * 100)) + 7, "MUPD = {0:.2f} %\n RMSE = {1:.2f}%".format(mupd[1], rmse_tot[1]), fontsize=6)
    ax[2, 2].text(25, np.round(np.max(mre_profile * 100)) + 7, "MUPD = {0:.2f} %\n RMSE = {1:.2f}%".format(mupd[2], rmse_tot[2]), fontsize=6)

    # Add letters
    a00.text(-0.05, 1.02, "(" + string.ascii_lowercase[0] + ")", transform=a00.transAxes, size=9, weight='bold')
    a01.text(-0.05, 1.02, "(" + string.ascii_lowercase[1] + ")", transform=a01.transAxes, size=9, weight='bold')
    a02.text(-0.05, 1.02, "(" + string.ascii_lowercase[2] + ")", transform=a02.transAxes, size=9, weight='bold')

    a10.text(-0.05, 1.02, "(" + string.ascii_lowercase[3] + ")", transform=a10.transAxes, size=9, weight='bold')
    a11.text(-0.05, 1.02, "(" + string.ascii_lowercase[4] + ")", transform=a11.transAxes, size=9, weight='bold')
    a12.text(-0.05, 1.02, "(" + string.ascii_lowercase[5] + ")", transform=a12.transAxes, size=9, weight='bold')

    a20.text(-0.05, 1.02, "(" + string.ascii_lowercase[6] + ")", transform=a20.transAxes, size=9, weight='bold')
    a21.text(-0.05, 1.02, "(" + string.ascii_lowercase[7] + ")", transform=a21.transAxes, size=9, weight='bold')
    a22.text(-0.05, 1.02, "(" + string.ascii_lowercase[8] + ")", transform=a22.transAxes, size=9, weight='bold')

    fig.tight_layout()

    return fig, ax, all_rad_cam, all_rad_sim


def rad_rmse(rad_cam, rad_sim):
    """
    ROOT MEAN SQUARE ERROR.
    :param rad_cam: radiance from camera
    :param rad_sim: radiance from simulations
    :return:
    """
    rmse = []
    for i in range(rad_cam.shape[2]):
        rmse.append(np.sqrt(np.nanmean(np.square((rad_cam[:, :, i] - rad_sim[:, :, i]) / rad_sim[:, :, i]))) * 100)

    return rmse


def bulk_kd(ed, z1=0.0, z2=200.0, verbose=True):
    """

    :param z:
    :param kd:
    :param z1:
    :param z2:
    :return:
    """

    keys_rgb = ed.dtype.names[:3]
    zdepth = ed["depth"]
    mask_z1 = np.where(zdepth == z1)
    mask_z2 = np.where(zdepth == z2)

    bkd_ls = []

    for k in keys_rgb:
        delta_z = z2 - z1
        delta_z /= 100.0
        bkd = np.log(ed[k][mask_z1][0]/ed[k][mask_z2][0]) / delta_z
        bkd_ls.append(bkd)
        if verbose:
            print(f"band {k}: kd = {bkd:.4f} m-1")

    return np.array(bkd_ls)


def similarity(a, b, g):

    return (1 + (b * (1-g) / a)) ** (-1/2)


if __name__ == "__main__":

    # Cam data
    rc = RadClass(data_path="data/oden-08312018-fluo.h5")
    #zen_cam, rad_cam = rc.get_radiance_avg_at_depth_wl(depth=40.0, wl=484, smooth=True)
    zen_cam, rad_cam = rc.get_radiance_avg_at_depth_wl(depth=40.0, wl=480, smooth=True)

    # Load radiance data
    #zd = load_zenith_radiance(r"C:\Users\Raphaël Larouche\PycharmProjects\HE60-PyMagister\to_raph\data\manual_opt")

    #zd = load_zenith_radiance(path=r"data/oden_fit")
    zd = load_zenith_radiance(path=r"data\oden_pf_experiment\oden_pf_comp_tthg")
    #zd = load_zenith_radiance(path=r"data\oden_pf_experiment\oden_malinka_fit\oden_malinka_fit")
    zd_hg = load_zenith_radiance(path=r"data\oden_pf_experiment\oden_pf_comp_hg")
    zd_mal = load_zenith_radiance(path=r"data\oden_pf_experiment\oden_pf_comp_malinka")
    zd_hg_odenfluo = load_zenith_radiance(path=r"data\oden_fit_fluo")

    zen_oden, azi_oden, rad_oden = open_radiance_data(path="data/oden-08312018.h5")  # Path à changer
    zen_dort, azi_dort, rad_dort = open_radiance_data(path="data/dort-simulation.h5")  # Path à changer

    # Figures
    fig1, ax1, aradcam, aradsim = graph_radiance_cam_vs_simulations(rc, zd, np.arange(20, 180, 20).astype(float), sm=False)
    fig3, ax3, _, _ = graph_radiance_cam_vs_simulations(rc, zd_hg, np.arange(20, 180, 20).astype(float),
                                                                    sm=False)
    fig2, ax2, _, _ = graph_radiance_cam_vs_simulations(rc, zd_hg_odenfluo, np.arange(20, 180, 20).astype(float),
                                                                    sm=False)
    fig4, ax4, _, _ = graph_radiance_cam_vs_simulations(rc, zd_mal, np.arange(20, 180, 20).astype(float),
                                                                    sm=False)
    fig1.suptitle("TTHG")
    fig1.tight_layout()
    fig2.suptitle("HG avant")
    fig2.tight_layout()
    fig3.suptitle("HG après")
    fig3.tight_layout()
    fig4.suptitle("MAL")
    fig4.tight_layout()

    #fig2, ax2 = graph_cam_vs_simulations(rad_oden, zen_oden, rad_dort, zen_dort, ["20 cm (in water)",
                                                                                      #"40 cm",
                                                                                      #"60 cm",
                                                                                      #"80 cm",
                                                                                      #"100 cm",
                                                                                      #"120 cm",
                                                                                     # "140 cm",
                                                                                     # "160 cm"], ["20 cm",
                                                                                    #  "40 cm",
                                                                                     # "60 cm",
                                                                                     # "80 cm",
                                                                                    #  "100 cm",
                                                                                     # "120 cm",
                                                                                     # "140 cm",
                                                                                     # "160 cm"])
    #fig2.suptitle("DORT2002")
    #fig2.tight_layout()

    # Savefig
    #fig1.savefig("figures/radiance_profiles_oden.png", format="png", dpi=300)
    #fig1.savefig("figures/radiance_profiles_oden.pdf", format="pdf", dpi=300)

    plt.show()

