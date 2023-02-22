# -*- coding: utf-8 -*-

# Module importation
import string
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Other modules
from source.radiance import RadClass


def test_1():

    # Create figure
    fig1 = plt.figure(figsize=(6.6929, 6.6929 * 0.7))
    ax1 = np.empty((2, 3), dtype="object")

    for i in range(2):
        for j in range(3):
            if i == 0:
                ax1[i, j] = fig1.add_subplot(2, 3, int((i+1) * (j+1)))
            else:
                ax1[i, j] = fig1.add_subplot(2, 3, int((i+1) * (j+1) + (2 - j)), projection="polar")

    # Loop param
    #wl_ = [603, 544, 484]
    wl_ = [600, 540, 480]
    dpth = 120.

    rad_raw_d = rc.radiance_profile[rc.keys_from_depth[dpth]]
    rad_raw_d[rad_raw_d == 0] = np.nan

    zeni = rc.zenith_meshgrid.copy()
    azi = rc.azimuth_meshgrid.copy() * np.pi / 180

    for i, w in enumerate(wl_):

        th_sm_d, rad_sm_d = rc.get_radiance_avg_at_depth_wl(depth=dpth, wl=w, smooth=True)
        th_d, rad_d = rc.get_radiance_avg_at_depth_wl(depth=dpth, wl=w, smooth=False)

        # Residuals
        res = np.absolute(rad_raw_d[:, :, i].copy() - np.tile(rad_sm_d.reshape(-1, 1), (1, 361)))
        #res[np.isnan(res)] = 0.0
        # Fit figure
        ax1[0, 0].plot(th_d, rad_d, alpha=0.5, markeredgecolor="grey", markerfacecolor="none", markersize=4, marker="o", linestyle="none")
        ax1[0, 0].plot(th_d, rad_sm_d, color="k", linewidth=1,  linestyle="-.")

        cax = ax1[1, i].contourf(azi, zeni, res, 20, cmap="coolwarm")

        # Tick parameters
        ax1[1, i].tick_params(axis='x', which='major', labelsize=7.5, pad=-3)
        # ytick
        ytik = np.arange(0, 200, 40)
        ax1[1, i].set_yticks(ytik)
        ax1[1, i].set_yticklabels(["{}˚".format(i) for i in ytik], fontsize=7)

        ax1[1, i].grid(linestyle="-.")
        cl = fig1.colorbar(cax, ax=ax1[1, i], orientation="horizontal", format='%.1e', pad=0.17)
        cl.ax.set_title("Residuals", fontsize=6.5)
        ls_xtick = cl.ax.get_xticks()
        cl.ax.set_xticks(ls_xtick)
        cl.ax.set_xticklabels([format(xt, '.1e') for xt in ls_xtick], rotation=30, fontsize=7)

    fig1.tight_layout()
    plt.show()

    return


def plot_contour(f, a, x, y, z, nc=20, clabel="Residuals [%]", form=".1e"):
    """

    :param f:
    :param a:
    :param x:
    :param y:
    :param z:
    :param nc:
    :param clabel:
    :return:
    """
    cax = a.contourf(x, y, z, nc, cmap="coolwarm")

    # Tick parameters
    a.tick_params(axis='x', which='major', labelsize=7.5, pad=-3)
    # ytick
    ytik = np.arange(0, 200, 40)
    a.set_yticks(ytik)
    a.set_yticklabels(["{}˚".format(i) for i in ytik], fontsize=7)

    a.grid(linestyle="-.")
    #cl = f.colorbar(cax, ax=a, orientation="vertical", format='%.1e', pad=0.17)

    cl = f.colorbar(cax, ax=a, location="bottom", orientation="horizontal", shrink=0.9, format=form, pad=0.2)
    cl.ax.set_title(clabel, fontsize=7.5)
    ls_xtick = cl.ax.get_xticks()
    cl.ax.set_xticks(ls_xtick)
    cl.ax.set_xticklabels([format(xt, form) for xt in ls_xtick], rotation=40, fontsize=7.5)

    return f, a


def plot_all_contourf(fi, ax1, ax2, ax3, depth=20.0, wave=544):
    """

    :param fi:
    :param ax1:
    :param ax2:
    :param ax3:
    :param depth:
    :param wave:
    :return:
    """

    #rad_c = RadClass(data_path="data/oden-08312018.h5")
    rad_c = RadClass(data_path="data/oden-08312018-fluo.h5")

    # Angular coordinates
    ze = rad_c.zenith_meshgrid.copy()
    az = rad_c.azimuth_meshgrid.copy() * np.pi / 180

    # Raw radiance
    #wl_correspond = {484: 2, 544: 1, 603: 0}
    wl_correspond = {480: 2, 540: 1, 600: 0}
    rad_dist_raw = rad_c.radiance_profile[rad_c.keys_from_depth[depth]]
    rad_dist_raw[rad_dist_raw == 0] = np.nan

    rad_raw_dist_c = rad_dist_raw[:, :, wl_correspond[wave]].copy()

    # Smoothed radiance
    _, rad_sm_d = rc.get_radiance_avg_at_depth_wl(depth=depth, wl=wave, smooth=True)
    rad_sm_dist_c = np.tile(rad_sm_d.reshape(-1, 1), (1, 361))

    # Absolute percent errors
    err = 100 * (np.absolute(rad_raw_dist_c - rad_sm_dist_c) / rad_sm_dist_c)

    # Plot
    c_array = np.linspace(np.nanmin(rad_raw_dist_c), np.nanmax(rad_raw_dist_c), 20)
    fi, ax1 = plot_contour(fi, ax1, az, ze, rad_sm_dist_c, nc=c_array, clabel=f"$\overline{{L}}_{{{wave} \mathrm{{nm, smooth}}}}$({depth:.0f} cm)", form='.1e')
    fi, ax2 = plot_contour(fi, ax2, az, ze, rad_raw_dist_c, nc=c_array, clabel=f"$\overline{{L}}_{{{wave} \mathrm{{nm, raw}}}}$({depth:.0f} cm)", form='.1e')
    fi, ax3 = plot_contour(fi, ax3, az, ze, err, nc=20, form=".1f", clabel=r"$\left| e \right|$ [%]")

    return fi, ax1, ax2, ax3


def plot_azavg_radiance(ax, depth=20.0, wave=544):
    """

    :param f:
    :param ax:
    :param depth:
    :param wave:
    :return:
    """

    #rad_c = RadClass(data_path="data/oden-08312018.h5")
    rad_c = RadClass(data_path="data/oden-08312018-fluo.h5")

    th_smooth, rad_smooth = rad_c.get_radiance_avg_at_depth_wl(depth=depth, wl=wave, smooth=True)  # Smooth
    th_raw, rad_raw = rad_c.get_radiance_avg_at_depth_wl(depth=depth, wl=wave, smooth=False)  # Raw

    # legendre coeff
    #wl_correspond = {484: 2, 544: 1, 603: 0}
    wl_correspond = {480: 2, 540: 1, 600: 0}
    lc_coeff = rc.legendre_coeff[rc.keys_from_depth[depth]][:, wl_correspond[wave]]
    lc_stri = "".join([f"$c_{{{b}}}={lc_coeff[b]:+.1e}$\n" for b in range(lc_coeff.shape[0])])

    # PLot
    ax.plot(th_smooth, rad_smooth, color="k", linewidth=1, linestyle="-.", label="Legendre fit")
    ax.plot(th_raw, rad_raw, markersize=4, marker="o", markeredgecolor="grey", markerfacecolor="none", linestyle="none", alpha=0.5, label=f"Raw data")

    ax.text(0.01, 0.005, lc_stri, transform=ax.transAxes, fontsize=8)
    ax.set_title(f"{depth} cm, $\lambda={wave}$ nm", fontsize=9)

    #ax.set_yscale("log")
    ax.set_ylabel("$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
    ax.set_xlabel("Zenith [˚]")

    ax.legend(loc="best", frameon=False)

    return ax


if __name__ == "__main__":

    # Radiance class
    #rc = RadClass(data_path="data/oden-08312018.h5")
    rc = RadClass(data_path="data/oden-08312018-fluo.h5")

    # Create figure
    plt.style.use("../../figurestyle.mplstyle")
    #plt.style.use("seaborn")

    fig1 = plt.figure(figsize=(6.6929, 6.6929 * 1.1))

    gs = fig1.add_gridspec(3, 6)

    ax1_0 = fig1.add_subplot(gs[0, :3])
    ax1_1 = fig1.add_subplot(gs[0, 3:])
    ax2 = fig1.add_subplot(gs[1, :2], projection="polar")
    ax3 = fig1.add_subplot(gs[1, 2:4], projection="polar")
    ax4 = fig1.add_subplot(gs[1, 4:], projection="polar")
    ax5 = fig1.add_subplot(gs[2, :2], projection="polar")
    ax6 = fig1.add_subplot(gs[2, 2:4], projection="polar")
    ax7 = fig1.add_subplot(gs[2, 4:], projection="polar")

    # Wanted depth and wavelength
    dpth = 120
    w = 540

    ax1_0 = plot_azavg_radiance(ax1_0, depth=40, wave=480)
    ax1_1 = plot_azavg_radiance(ax1_1, depth=dpth, wave=w)
    ax1_1.yaxis.tick_right()
    ax1_1.yaxis.set_label_position("right")
    ax1_1.set_ylabel('')

    fig1, ax3, ax2, ax4 = plot_all_contourf(fig1, ax3, ax2, ax4, depth=40, wave=480)
    fig1, ax6, ax5, ax7 = plot_all_contourf(fig1, ax6, ax5, ax7, depth=dpth, wave=w)

    # Loop param
    #wl_ = [603, 544, 484]
    wl_ = [600, 540, 480]
    dpth_list = [20, 40, 60, 80, 100, 120, 140, 160]

    zeni = rc.zenith_meshgrid.copy()
    azi = rc.azimuth_meshgrid.copy() * np.pi / 180

    df_err_tot = pandas.DataFrame()

    for l, d in enumerate(dpth_list):
        mean_r = np.zeros(3)
        std_r = np.zeros(3)
        min_r = np.zeros(3)
        max_r = np.zeros(3)

        df_err = pandas.DataFrame()

        for i, w in enumerate(wl_):

            th_sm_d, rad_sm_d = rc.get_radiance_avg_at_depth_wl(depth=d, wl=w, smooth=True)
            th_d, rad_d = rc.get_radiance_avg_at_depth_wl(depth=d, wl=w, smooth=False)

            # Residuals
            rad_raw_d = rc.radiance_profile[rc.keys_from_depth[d]]
            rad_raw_d[rad_raw_d == 0] = np.nan
            r_raw_c = rad_raw_d[:, :, i].copy()
            r_sm_c = np.tile(rad_sm_d.reshape(-1, 1), (1, 361))

            # res[np.isnan(res)] = 0.0
            res_p = 100 * (np.absolute(r_raw_c - r_sm_c) / r_sm_c)  # residuals in percent
            mean_r[i] = np.nanmean(res_p)
            std_r[i] = np.nanstd(res_p)
            min_r[i] = np.nanmin(res_p)
            max_r[i] = np.nanmax(res_p)

            # Dataframe
            df_err["Zen"] = zeni[~np.isnan(res_p)]
            df_err["Azi"] = azi[~np.isnan(res_p)]
            df_err["Depth [cm]"] = np.ones(res_p[~np.isnan(res_p)].shape[0]) * d
            df_err["Wavelength [nm]"] = np.ones(res_p[~np.isnan(res_p)].shape[0]) * w
            df_err["Errors [%]"] = res_p[~np.isnan(res_p)]

            df_err_tot = pandas.concat([df_err_tot, df_err], axis=0)
        print(f"z={d} cm, Rm 480 nm = ({mean_r[2]:.2f} +/- {std_r[2]:.2f}), "
              f"Rm 540 nm = ({mean_r[1]:.2f} +/- {std_r[1]:.2f}), "
              f"Rm 600 nm = ({mean_r[0]:.2f} +/- {std_r[0]:.2f}), "
              f"R min (480, 540, 600) = ({min_r[2]:.2e}, {min_r[1]:.2e}, {min_r[0]:.2e}), "
              f"R max (480, 540, 600) = ({max_r[2]:.2f}, {max_r[1]:.2f}, {max_r[0]:.2f})")

    fig1.subplots_adjust(left=0.13,
                        bottom=0.08,
                        right=0.87,
                        top=0.92,
                        wspace=0.15,
                        hspace=0.35)


    # Add letters on fig1
    ax1_0.text(-0.05, 1.02, "(" + string.ascii_lowercase[0] + ")", transform=ax1_0.transAxes, size=9, weight='bold')
    ax1_1.text(-0.05, 1.02, "(" + string.ascii_lowercase[1] + ")", transform=ax1_1.transAxes, size=9, weight='bold')

    ax2.text(-0.05, 1.02, "(" + string.ascii_lowercase[2] + ")", transform=ax2.transAxes, size=9, weight='bold')
    ax3.text(-0.05, 1.02, "(" + string.ascii_lowercase[3] + ")", transform=ax3.transAxes, size=9, weight='bold')
    ax4.text(-0.05, 1.02, "(" + string.ascii_lowercase[4] + ")", transform=ax4.transAxes, size=9, weight='bold')

    ax5.text(-0.05, 1.02, "(" + string.ascii_lowercase[5] + ")", transform=ax5.transAxes, size=9, weight='bold')
    ax6.text(-0.05, 1.02, "(" + string.ascii_lowercase[6] + ")", transform=ax6.transAxes, size=9, weight='bold')
    ax7.text(-0.05, 1.02, "(" + string.ascii_lowercase[7] + ")", transform=ax7.transAxes, size=9, weight='bold')

    # Print results
    print(df_err_tot.groupby(["Depth [cm]", "Wavelength [nm]"])["Errors [%]"].describe().to_string())

    # Figure 2
    fig2, ax2 = plt.subplots(1, 1)

    palette = sns.color_palette(n_colors=3)
    npalette = palette.copy()
    npalette[1:] = palette[:0:-1]
    sns.boxplot(x="Depth [cm]", y="Errors [%]", data=df_err_tot, hue="Wavelength [nm]", ax=ax2, palette=npalette)
    sns.despine(right=True)

    # Savefig
    fig2.tight_layout()
    fig1.savefig("figures/leg_fit.pdf", format="pdf", dpi=300)
    fig1.savefig("figures/leg_fit.png", format="png", dpi=300)


    plt.show()
