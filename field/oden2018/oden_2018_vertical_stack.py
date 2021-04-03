# -*- coding: utf-8 -*-
"""
Oden icebreaker A02018 mission, cam optic vertical stack of radiance angular distribution.
"""

# Module importation
import string
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Other module
from source.processing import FigureFunctions, ProcessImage


# Function and classes
def polar_plot_contourf(zenith_mesh, azimuth_mesh, mappedradiance, fig, ax, ncontour):
    """
    Function that plot contour filled figure.
    :param ncontour: number of levels
    :return: (figure, axe) - tuple
    """

    if len(mappedradiance.shape) == 3:
        lab = ["red", "green", "blue"]

        for n, a in enumerate(ax):
            zeni = zenith_mesh.copy() * 180 / np.pi
            azi = azimuth_mesh.copy()

            im = mappedradiance[:, :, n].copy()
            insideFOV = np.where(im > 0)
            mini, maxi = np.nanmin(im[insideFOV]), np.nanmax(im[insideFOV])

            cax = a.contourf(azi, zeni, np.clip(im, 0, None), np.linspace(mini, maxi, ncontour), cmap="coolwarm")

            ytik = np.arange(0, 200, 40)
            a.set_yticks(ytik)
            a.set_yticklabels(["{}˚".format(i) for i in ytik], fontsize=7)
            a.tick_params(axis='x', which='major', labelsize=7)
            if (n == 1) or (n == 2):
                a.set_xticklabels([])

            #a.grid(linestyle="-.")
            cl = fig.colorbar(cax, ax=a, orientation="horizontal", format='%.1e')
            cl.ax.set_title("$L_{0}$".format(lab[n][0]), fontsize=7)
            cl.ax.set_xticklabels(cl.ax.get_xticklabels(), rotation=45, fontsize=8)
            a.text(-0.2, 1.2, "(" + string.ascii_lowercase[n] + ")", transform=a.transAxes, size=11, weight='bold')

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
    ff = FigureFunctions()

    # Object ProcessImage
    process = ProcessImage()

    # Open spectral radiance angular distribution
    zenith_m, azimuth_m, profile_radiance = process.open_radiance_data(path="data/oden-08312018.h5")
    x, y = zenith_m * np.cos(azimuth_m * np.pi / 180), zenith_m * np.sin(azimuth_m * np.pi / 180)

    # Figure initialization
    f1 = plt.figure(figsize=(ff.set_size(443.86319)[0], ff.set_size(443.86319)[1] * 1.8))
    a1 = np.empty((2, 3), dtype="object")
    for i in range(2):
        for j in range(3):
            if i == 0:
                a1[i, j] = f1.add_subplot(2, 3, int((i+1) * (j+1)), projection="polar")
            else:
                a1[i, j] = f1.add_subplot(2, 3, int((i+1) * (j+1) + (2 - j)), projection="3d")
    d = np.array([])
    bands = ["r", "g", "b"]

    for k in data_keys:

        curr_rp = profile_radiance[k].copy()
        curr_rp[curr_rp == 0] = np.nan

        dept = float(k.split()[0])
        d = np.append(dept, d)
        for n in range(curr_rp.shape[2]):

            curr_rp_norm = curr_rp[:, :, n] / np.nanmax(curr_rp[:, :, n]) # Normalization
            cf = a1[1, n].contourf(x, y, curr_rp_norm, ncontour, zdir='z', offset=dept, vmin=0, vmax=1, cmap="coolwarm")
            #a1[1, n].contour(x, y, gaussian_filter(curr_rp_norm, 5), levels=np.array([0.5, 0.75, 0.9]), zdir='z', offset=dept, linewidths=2, linestyles="dashed", colors='k')

    f1, a1[0, :] = polar_plot_contourf(zenith_m * np.pi / 180, azimuth_m * np.pi / 180, profile_radiance[dkeys_polar].copy(), f1, a1[0, :], ncontour)

    for n in range(3):

        a1[1, n].set_zticks(np.arange(0, np.max(d) + 40, 20))
        a1[1, n].set_zlim(0, np.max(d) + 20)
        a1[1, n].invert_zaxis()

        a1[1, n].set_ylim(-150, 150)
        a1[1, n].set_xlim(-150, 150)

        a1[1, n].axes.xaxis.set_ticklabels([])
        a1[1, n].axes.yaxis.set_ticklabels([])

        cbl = f1.colorbar(cf, ax=a1[1, n], orientation="horizontal")
        cbl.ax.set_title("$Lnorm_{0}$".format(bands[n]), fontsize=7)
        cbl.ax.set_xticklabels(cbl.ax.get_xticklabels(), rotation=45, fontsize=8)

    a1[1, 0].set_zlabel("Depth [cm]")

    return f1, a1


if __name__ == "__main__":

    plt.style.use("../../figurestyle.mplstyle")

    # Figure 1
    fig1, ax1 = vertical_stack_radiance(["40 cm", "80 cm", "120 cm", "160 cm"], "40 cm", 15)
    #fig1.tight_layout()

    fig1.savefig("figures/vertical_stack.pdf", format="pdf", dpi=600)
    plt.show()
