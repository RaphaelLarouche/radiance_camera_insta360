# -*- coding: utf-8 -*-
"""
Oden icebreaker A02018 mission, cam optic vertical stack of radiance angular distribution.
"""

# Module importation
import os
import string
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

# Other module
from source.processing import FigureFunctions, ProcessImage
from source.radiance import RadClass


# Function and classes
def polar_plot_contourf(zenith_mesh, azimuth_mesh, mappedradiance, fig, ax, ncontour, depth_label, wl_order=None):
    """
    Function that plot contour filled figure.
    :param ncontour: number of levels
    :return: (figure, axe) - tuple
    """

    if wl_order is None:
        wl_order = ["red", "green", "blue"]
    if len(mappedradiance.shape) == 3:
        #wl_lab = {"red": "603 nm", "green": "544 nm", "blue": "484 nm"}
        wl_lab = {"red": "600 nm", "green": "540 nm", "blue": "480 nm"}
        wl_corr = {"red": 0, "green": 1, "blue": 2}

        for n, b in enumerate(wl_order):
            zeni = zenith_mesh.copy() * 180 / np.pi
            azi = azimuth_mesh.copy()

            im = mappedradiance[:, :, wl_corr[b]].copy()
            insideFOV = np.where(mappedradiance > 0)
            mini, maxi = np.nanmin(mappedradiance[insideFOV]), np.nanmax(mappedradiance[insideFOV])

            cax = ax[n].contourf(azi, zeni, np.clip(im, 0, None), np.linspace(mini, maxi, ncontour), cmap="coolwarm")

            # Tick parameters
            ax[n].tick_params(axis='x', which='major', labelsize=7.5, pad=-3)
            # ytick
            ytik = np.arange(0, 200, 40)
            ax[n].set_yticks(ytik)
            ax[n].set_yticklabels(["{}˚".format(i) for i in ytik], fontsize=7)

            # Ticks azimuth
            ax[n].grid(linestyle="-.")
            cl = fig.colorbar(cax, ax=ax[n], orientation="horizontal", format='%.1e', pad=0.19)
            cl.ax.set_title("$L_{{{0}}}({1})$ [$\mathrm{{W \cdot sr^{{-1}} \cdot m^{{-2}} \cdot nm^{{-1}}}}$]".format(wl_lab[b], depth_label), fontsize=6.5)
            ls_xtick = cl.ax.get_xticks()
            cl.ax.set_xticks(ls_xtick)
            cl.ax.set_xticklabels([format(xt, '.2e') for xt in ls_xtick], rotation=20, fontsize=7)

            # Letter at corner of image
            ax[n].text(-0.1, 1.1, "(" + string.ascii_lowercase[n] + ")", transform=ax[n].transAxes, size=10, weight='bold')

        fig.tight_layout()
        return fig, ax
    else:
        raise ValueError("Radiance map should be build before.")


def vertical_stack_radiance(data_keys, dkeys_polar, ncontour):
    """
    3D contourf plot of radiance angular distributions.
    :param data_keys:
    :return:
    """

    # Figure Function instance
    #ff = FigureFunctions()

    # Object ProcessImage
    process = ProcessImage()

    # Open spectral radiance angular distribution
    zenith_m, azimuth_m, profile_radiance = process.open_radiance_data(path="data/oden-08312018.h5")
    x, y = zenith_m * np.cos(azimuth_m * np.pi / 180), zenith_m * np.sin(azimuth_m * np.pi / 180)

    # Figure initialization
    f1 = plt.figure(figsize=(6.6929, 6.6929 * 0.7))
    a1 = np.empty((2, 3), dtype="object")

    for i in range(2):
        for j in range(3):
            if i == 0:
                a1[i, j] = f1.add_subplot(2, 3, int((i+1) * (j+1)), projection="polar")
            else:
                a1[i, j] = f1.add_subplot(2, 3, int((i+1) * (j+1) + (2 - j)), projection="3d")
                a1[i, j].view_init(elev=22.)
                a1[i, j].set_box_aspect(aspect=(1, 1, 1.7))

    d = np.array([])
    bands = ["r", "g", "b"]
    #wl_lab = {"r": "603 nm", "g": "544 nm", "b": "484 nm"}
    wl_lab = {"r": "600 nm", "g": "540 nm", "b": "480 nm"}

    for k in data_keys:

        curr_rp = profile_radiance[k].copy()
        curr_rp[curr_rp == 0] = np.nan

        dept = float(k.split()[0])
        d = np.append(dept, d)

        for n in range(curr_rp.shape[2]):

            curr_rp_norm = 100 * (curr_rp[:, :, n] / np.nanmax(curr_rp[:, :, n]))  # Normalization
            cf = a1[1, n].contourf(x, y, curr_rp_norm, ncontour, zdir='z', offset=dept, vmin=0, vmax=100, cmap="coolwarm")

    # Adding polar plot contour
    f1, a1[0, :] = polar_plot_contourf(zenith_m * np.pi / 180, azimuth_m * np.pi / 180, profile_radiance[dkeys_polar].copy(), f1, a1[0, :], ncontour, dkeys_polar)

    # Axes parameters
    for n in range(3):

        a1[1, n].text2D(-0.1, 0.9, "(" + string.ascii_lowercase[n] + ")", transform=a1[1, n].transAxes, size=10, weight='bold')

        a1[1, n].set_zticks(np.arange(0, np.max(d) + 40, 20))
        a1[1, n].set_zlim(0, np.max(d) + 20)
        a1[1, n].invert_zaxis()

        a1[1, n].set_ylim(-120, 120)
        a1[1, n].set_xlim(-120, 120)

        a1[1, n].axes.xaxis.set_ticklabels([])
        a1[1, n].axes.yaxis.set_ticklabels([])

        a1[1, n].grid(linestyle="-.")
        a1[1, n].xaxis.pane.fill = False
        a1[1, n].yaxis.pane.fill = False
        a1[1, n].zaxis.pane.fill = False

        a1[1, n].set_zlabel("Depth [cm]")

        cbl = f1.colorbar(cf, ax=a1[1, n], orientation="horizontal", pad=0.1)
        cbl.ax.set_title("$L_{{{0}}}$ [%]".format(wl_lab[bands[n]]), fontsize=7)
        ls_xtick = cbl.ax.get_xticks()
        cbl.ax.set_xticks(ls_xtick)
        cbl.ax.set_xticklabels([format(xt, '.1f') for xt in ls_xtick], rotation=20, fontsize=7)

    return f1, a1


def vertical_stack_radiance_v2(data_keys, dkeys_polar, ncontour):
    """
    3D contourf plot of radiance angular distributions.
    :param data_keys:
    :return:
    """

    # Open spectral radiance angular distribution
    #rc = RadClass(data_path="data/oden-08312018.h5")
    rc = RadClass(data_path="data/oden-08312018-imf-fluo.h5")
    x, y = rc.zenith_meshgrid * np.cos(rc.azimuth_meshgrid * np.pi / 180), rc.zenith_meshgrid * np.sin(rc.azimuth_meshgrid * np.pi / 180)

    # Figure initialization
    f1 = plt.figure(figsize=(6.6929, 6.6929 * 0.7))
    a1 = np.empty((2, 3), dtype="object")

    for i in range(2):
        for j in range(3):
            if i == 0:
                a1[i, j] = f1.add_subplot(2, 3, int((i+1) * (j+1)), projection="polar")
            else:
                a1[i, j] = f1.add_subplot(2, 3, int((i+1) * (j+1) + (2 - j)), projection="3d")
                a1[i, j].view_init(elev=22.)
                a1[i, j].set_box_aspect(aspect=(1, 1, 1.7))

    d = np.array([])
    bands = ["r", "g", "b"]
    #wl_lab = {"r": "603 nm", "g": "544 nm", "b": "484 nm"}
    wl_lab = {"r": "600 nm", "g": "540 nm", "b": "480 nm"}
    #wl_float = {"r": 603, "g": 544, "b": 484}
    wl_float = {"r": 600, "g": 540, "b": 480}

    # Vertical stack
    for k in data_keys:

        dept = float(k.split()[0])
        d = np.append(dept, d)

        for n, b in enumerate(bands[::-1]):
            c_rad = rc.get_radiance_dist_at_depth_wl(depth=dept, wl=wl_float[b])
            c_rad[c_rad == 0] = np.nan
            c_rad_norm = 100 * (c_rad / np.nanmax(c_rad))
            cf = a1[1, n].contourf(x, y, c_rad_norm, ncontour, zdir='z', offset=dept, vmin=0, vmax=100, cmap="coolwarm")

    # Adding polar plot contour
    f1, a1[0, :] = polar_plot_contourf(rc.zenith_meshgrid * np.pi / 180, rc.azimuth_meshgrid * np.pi / 180,
                                       rc.radiance_profile[dkeys_polar].copy(), f1, a1[0, :], ncontour, dkeys_polar,
                                       wl_order=["blue", "green", "red"])

    # Axes parameters
    for n, b in enumerate(bands[::-1]):
        a1[1, n].text2D(-0.1, 0.9, "(" + string.ascii_lowercase[n+3] + ")", transform=a1[1, n].transAxes, size=10, weight='bold')

        a1[1, n].set_zticks(np.arange(0, np.max(d) + 40, 20))
        a1[1, n].set_zlim(0, np.max(d) + 20)
        a1[1, n].invert_zaxis()

        a1[1, n].set_ylim(-120, 120)
        a1[1, n].set_xlim(-120, 120)

        a1[1, n].axes.xaxis.set_ticklabels([])
        a1[1, n].axes.yaxis.set_ticklabels([])

        a1[1, n].grid(linestyle="-.")
        a1[1, n].xaxis.pane.fill = False
        a1[1, n].yaxis.pane.fill = False
        a1[1, n].zaxis.pane.fill = False

        a1[1, n].set_zlabel("Depth [cm]")

        cbl = f1.colorbar(cf, ax=a1[1, n], orientation="horizontal", pad=0.1)
        cbl.ax.set_title("$L_{{{0}}}$ [%]".format(wl_lab[b]), fontsize=7)
        ls_xtick = cbl.ax.get_xticks()
        cbl.ax.set_xticks(ls_xtick)
        cbl.ax.set_xticklabels([format(xt, '.1f') for xt in ls_xtick], rotation=20, fontsize=7)

    return f1, a1


if __name__ == "__main__":

    plt.style.use(os.path.abspath(os.path.join(__file__, "../../..")) + "/figurestyle.mplstyle")

    # Figure 1
    #fig1, ax1 = vertical_stack_radiance(["40 cm", "80 cm", "120 cm", "160 cm"], "40 cm", 15)
    fig1, ax1 = vertical_stack_radiance_v2(["40.0 cm", "80.0 cm", "120.0 cm", "160.0 cm"], "40.0 cm", 15)

    fig1.savefig("figures/vertical_stack.pdf", format="pdf", dpi=600)
    fig1.savefig("figures/vertical_stack.png", format="png", dpi=600)

    plt.show()
