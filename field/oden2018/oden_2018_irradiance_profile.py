# -*- coding: utf-8 -*-
"""
Oden icebreaker A02018 mission, irradiance profiles.
"""

# Module importation
import glob
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Other modules
from source.processing import ProcessImage, FigureFunctions
import source.radiance as r


# Function and classes
def sigmoid_5params(x, a, b, c, d, g):
    """

    :return:
    """

    return d + ((a - d) / (1 + (x / c) ** b) ** g)


def third_order_poly(x, a, b, c, d):
    """

    :param x:
    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """
    return a + b * x + c * x ** 2 + d * x ** 3


def general_gaussian(x, a, b, c):
    """
    
    :param x: 
    :param a: 
    :param b: 
    :param c: 
    :param d: 
    :return: 
    """""
    return np.exp(-(x * a - b) ** 2) + c


def interpolation(zenith_meshgrid, angular_radiance_distribution):
    """
    Interpolation of missing angles (due reduced FOV due to water refractive index) using a gaussian function.

    :param zenith_meshgrid: zenith meshgrid in degrees (array)
    :param angular_radiance_distribution: current radiance angular distribution (array)
    :return: interpolated radiance angular distribution (array)
    """

    ard = angular_radiance_distribution.copy()
    rad_zen = r.azimuthal_average(ard)

    for b in range(rad_zen.shape[1]):
        co = ~np.isnan(rad_zen[:, b])

        norm_val = np.mean(rad_zen[:, b][co][:5])  # 5 first values
        rad_zen_norm = rad_zen[:, b][co] / norm_val
        popt, pcov = curve_fit(general_gaussian, zenith_meshgrid[:, 0][co] * np.pi / 180, rad_zen_norm, p0=[-0.7, 0, 0.1])

        ard_c = ard[:, :, b].copy()
        ard_c[ard_c==0] = general_gaussian(zenith_meshgrid[ard_c==0] * np.pi / 180, *popt) * norm_val
        ard[:, :, b] = ard_c

    return ard

# TODO: normalization by surface data ??


if __name__ == "__main__":

    # Object FigureFunction
    ff = FigureFunctions()

    # Object ProcessImage
    process_im = ProcessImage()

    # Oden2018 image radiance data
    zen, azi, rad = process_im.open_radiance_data(path="data/oden-08312018.h5")

    wanted_depth = ["zero minus", "20 cm (in water)", "40 cm", "60 cm", "80 cm", "100 cm", "120 cm", "140 cm", "160 cm",
                    "180 cm", "200 cm"]

    depths = np.arange(0, 220, 20)

    # Pre-allocation
    Ed = np.zeros(len(wanted_depth), dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4')]))
    Eu, Eo, Edo, Euo = Ed.copy(), Ed.copy(), Ed.copy(), Ed.copy()

    figrad, axrad = plt.subplots(1, 3, sharex=True, sharey=True, figsize=ff.set_size(subplots=(1, 3)))

    # Colormap
    colornormdict = dict(zip(wanted_depth, np.arange(0, 220, 20)))
    col = matplotlib.cm.get_cmap("viridis", len(colornormdict.values()))
    cmit = iter(col.colors)

    # Loop
    for i, k in enumerate(wanted_depth):

        rad_interpo = interpolation(zen , rad[k])

        # Down-welling irradiance
        Ed[i] = tuple(r.irradiance(zen, azi, rad_interpo, 0, 90, planar=True))
        Edo[i] = tuple(r.irradiance(zen, azi, rad_interpo, 0, 90, planar=False))

        # Up-welling irradiance
        Eu[i] = tuple(r.irradiance(zen, azi, rad_interpo, 90, 180, planar=True))
        Euo[i] = tuple(r.irradiance(zen, azi, rad_interpo, 90, 180, planar=False))

        # Scalar irradiance
        Eo[i] = tuple(r.irradiance(zen, azi, rad_interpo, 0, 180, planar=False))

        # Azimuthal average and plot
        cl = next(cmit)
        #azimuthal_average = r.azimuthal_average(rad[k])
        azimuthal_average = r.azimuthal_average(rad_interpo)

        axrad[0].plot(zen[:, 0], azimuthal_average[:, 0], linewidth=2, color=cl, label=k)
        axrad[1].plot(zen[:, 0], azimuthal_average[:, 1], linewidth=2, color=cl, label=k)
        axrad[2].plot(zen[:, 0], azimuthal_average[:, 2], linewidth=2, color=cl, label=k)

    # Gershun law estimation of absorption coefficient
    absorption = np.zeros(len(wanted_depth), dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4')]))
    absorption["r"] = r.attenuation_coefficient((Ed["r"] - Eu["r"]), depths) * ((Ed["r"] - Eu["r"]) / Eo["r"])
    absorption["g"] = r.attenuation_coefficient((Ed["g"] - Eu["g"]), depths) * ((Ed["g"] - Eu["g"]) / Eo["g"])
    absorption["b"] = r.attenuation_coefficient((Ed["b"] - Eu["b"]), depths) * ((Ed["b"] - Eu["b"]) / Eo["b"])

    # Transmittance
    T = np.array(list(Ed[-3])) / np.array(list(Ed[1]))  # between 20 cm and 160 cm - 180 cm uncertain if the stick was properly lowered

    print(T)

    # Figure
    fs = ff.set_size(subplots=(2, 2))
    fig1, ax1 = plt.subplots(1, 3, sharey=True, sharex=True, figsize=ff.set_size(subplots=(1, 3)))
    #fig2, ax2 = plt.subplots(2, 2, sharey=True, figsize=(fs[0], fs[1] * 1.5))
    fig2, ax2 = plt.subplots(1, 3, sharey=True, figsize=ff.set_size(subplots=(1, 3)))
    ax2 = ax2.ravel()
    #fig3, ax3 = plt.subplots(1, 2, sharey=True, figsize=ff.set_size(subplots=(1, 2)))
    fig3, ax3 = plt.subplots(1, 1, sharey=True, figsize=ff.set_size())
    fig4, ax4 = plt.subplots(1, 1, sharey=True, figsize=ff.set_size())

    dicband = {0: "r", 1: "g", 2: "b"}
    ls = {"r": "-", "g": "-.", "b": ":"}
    colo = {"r": "r", "g": "g", "b": "b"}
    labe = {"r": "red", "g": "green", "b": "blue"}

    for i, k in enumerate(dicband.values()):

        # Axe 1
        ax1[i].plot(Ed[k], depths, color=colo[k], linestyle="-", marker=".", label="$E_{d}$")
        ax1[i].plot(Eu[k], depths, color=colo[k], linestyle="-.", marker=".", label="$E_{u}$")
        ax1[i].plot(Eo[k], depths, color=colo[k], linestyle=":", marker=".", label="$E_{0}$")

        ax1[i].set_xscale("log")
        ax1[i].invert_yaxis()

        ax1[i].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
        ax1[i].text(-0.05, 1.05, "(" + string.ascii_lowercase[i] + ")", transform=ax1[i].transAxes, size=11, weight='bold')

        ax1[i].legend(loc="best")

        # Axe 2
        ax2[0].plot(Ed[k] / Edo[k], depths, color=colo[k], linestyle=ls[k], label=labe[k])
        ax2[1].plot(Eu[k] / Euo[k], depths, color=colo[k], linestyle=ls[k], label=labe[k])
        ax2[2].plot((Ed[k] - Eu[k]) / Eo[k], depths, color=colo[k], linestyle=ls[k], label=labe[k])

        ax2[i].text(-0.05, 1.05, "(" + string.ascii_lowercase[i] + ")", transform=ax2[i].transAxes, size=11, weight='bold')

        ax3.plot(r.attenuation_coefficient(Ed[k], depths), depths, linestyle=ls[k], color=colo[k], label=labe[k])
        ax4.plot(absorption[k], depths, color=colo[k], linestyle=ls[k], marker=".", label=labe[k])

    # Figure parameters
    # Figure 1
    ax1[0].set_ylabel("Depth [cm]")

    # Figure 2
    ax2[0].invert_yaxis()
    ax2[0].set_ylabel("Depth [cm]")
    ax2[0].legend(loc="best", fontsize=8)
    ax2[0].set_xlabel("$\mu_{d}$")
    ax2[1].set_xlabel("$\mu_{u}$")
    ax2[2].set_xlabel("$\mu$")

    # Figure rad
    axrad[0].set_yscale("log")
    axrad[0].set_xlabel("Zenith angle [˚]")
    axrad[1].set_xlabel("Zenith angle [˚]")
    axrad[2].set_xlabel("Zenith angle [˚]")
    axrad[0].set_ylabel(r"$\overline{{L}}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")

    # Figure 3
    ax3.set_xticks(np.arange(-1, 10, 1))
    ax3.invert_yaxis()
    ax3.set_xlim((-0.1, 9))
    ax3.set_ylabel("Depth [cm]")
    ax3.set_xlabel("$\kappa_{d}~[\mathrm{m^{-1}}]$")

    ax3.legend(loc="best")

    # ax4.set_xticks(np.arange(-0.5, 3.5, 0.5))
    # ax4.set_xlim((-0.1, 3.1))
    ax4.invert_yaxis()
    ax4.set_xscale("log")
    ax4.set_ylabel("Depth [cm]")
    ax4.set_xlabel("$a~[\mathrm{m^{-1}}]$")
    ax4.legend(loc="best")

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    figrad.tight_layout()

    # Saving figures
    fig1.savefig("figures/irradiance_profile.pdf", format="pdf", dpi=600)
    fig2.savefig("figures/aops.pdf", format="pdf", dpi=600)
    fig3.savefig("figures/diffuse_attenuation_downwelling.pdf", format="pdf", dpi=600)
    fig4.savefig("figures/absorption_coefficient.pdf", format="pdf", dpi=600)

    plt.show()
