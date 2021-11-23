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
import source.radiance as r
from source.processing import ProcessImage, FigureFunctions


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


def extrapolation(zenith_meshgrid, angular_radiance_distribution):
    """
    Interpolation of missing angles (due reduced FOV due to water refractive index) using a gaussian function.

    :param zenith_meshgrid: zenith meshgrid in degrees (array)
    :param angular_radiance_distribution: current radiance angular distribution (array)
    :return: interpolated radiance angular distribution (array)
    """

    ard = angular_radiance_distribution.copy()  # Angular radiance distribution
    rad_zen = r.azimuthal_average(ard)  # Perform azimuthal average

    for b in range(rad_zen.shape[1]):

        # Condition for non-nan data
        co = ~np.isnan(rad_zen[:, b])

        # Normalization
        norm_val = np.mean(rad_zen[:, b][co][:5])  # 5 first values
        rad_zen_norm = rad_zen[:, b][co] / norm_val

        # Fit ()
        popt, pcov = curve_fit(general_gaussian, zenith_meshgrid[:, 0][co] * np.pi / 180, rad_zen_norm, p0=[-0.7, 0, 0.1])

        ard_c = ard[:, :, b].copy()

        ard_c[ard_c == 0] = general_gaussian(zenith_meshgrid[ard_c == 0] * np.pi / 180, *popt) * norm_val
        ard[:, :, b] = ard_c

    return ard


def extrapolation_legendre_polynomials():

    return

# TODO: normalization by surface data ??  - AOPS normally not influenced by surface values
# TODO: OPEN radiometer data
# TODO: Try legendre fit
# TODO: Forward differentiation


def forward_gradient_dxdy(depths, values):
    """

    :param depths:
    :param values:
    :return:
    """
    nd = (depths[1:] + depths[:-1]) * 0.5
    df = (values[1:] - values[:-1]) * 1/np.diff(depths)
    return nd, df


if __name__ == "__main__":

    # Object FigureFunction
    ff = FigureFunctions()
    plt.style.use("../../figurestyle.mplstyle")

    # Object ProcessImage
    process_im = ProcessImage()

    # Opening radiometer data

    # Oden2018 image radiance data
    zen, azi, rad = process_im.open_radiance_data(path="data/oden-08312018.h5")

    # rad - radiance angular distribution

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

        # Extrapolation
        rad_interpo = extrapolation(zen, rad[k])

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

    # Using forward differentiation
    nd, grad_g = forward_gradient_dxdy(depths, Ed["g"] - Eu["g"])
    abs_g = -1 * grad_g * 100 * 1/np.interp(nd, depths, Eo["g"])

    # Transmittance
    T = np.array(list(Ed[-3])) / np.array(list(Ed[1]))  # between 20 cm and 160 cm - 180 cm uncertain if the stick was properly lowered

    print(T)

    # Figure ********************
    fs = ff.set_size(subplots=(2, 2))
    fig1, ax1 = plt.subplots(2, 3, sharey=True, figsize=(ff.set_size(subplots=(1, 3))[0], ff.set_size(subplots=(1, 3))[1] * 1.6))

    fig2, ax2 = plt.subplots(1, 3, sharey=True, figsize=ff.set_size(subplots=(1, 3)))
    ax2 = ax2.ravel()

    fig3, ax3 = plt.subplots(1, 3, sharey=True, figsize=ff.set_size(subplots=(1, 2)))

    #fig3, ax3 = plt.subplots(1, 1, sharey=True, figsize=ff.set_size())
    #fig4, ax4 = plt.subplots(1, 1, sharey=True, figsize=ff.set_size())

    dicband = {0: "r", 1: "g", 2: "b"}
    ls = {"r": "-", "g": "-.", "b": ":"}
    colo = {"r": "r", "g": "g", "b": "b"}
    labe = {"r": "red", "g": "green", "b": "blue"}

    labe_nm = {"r": "602 nm", "g": "544 nm", "b": "484 nm"}

    for i, k in enumerate(dicband.values()):

        # Axe 1
        ax1[0, i].plot(Ed[k], depths, color=colo[k], markeredgecolor=colo[k], markerfacecolor="none", linestyle="-", marker="o", markersize=4, label="$E_{d}$")
        ax1[0, i].plot(Eu[k], depths, color=colo[k], markeredgecolor=colo[k], markerfacecolor="none", linestyle="-.", marker="o", markersize=4, label="$E_{u}$")
        ax1[0, i].plot(Eo[k], depths, color=colo[k], markeredgecolor=colo[k], markerfacecolor="none", linestyle=":", marker="o", markersize=4, label="$E_{0}$")

        ax1[0, i].set_xscale("log")

        ax1[0, i].set_xlabel("Irradiance $[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
        ax1[0, i].text(-0.05, 1.05, "(" + string.ascii_lowercase[i] + ")", transform=ax1[0, i].transAxes, size=11, weight='bold')

        ax1[0, i].legend(loc="best")

        # Axe 1
        ax1[1, 0].plot(Ed[k] / Edo[k], depths, color=colo[k], markeredgecolor=colo[k], markerfacecolor="none", linestyle=ls[k], marker="o", markersize=4, label=labe_nm[k])
        ax1[1, 1].plot(Eu[k] / Euo[k], depths, color=colo[k], markeredgecolor=colo[k], markerfacecolor="none", linestyle=ls[k], marker="o", markersize=4, label=labe_nm[k])
        ax1[1, 2].plot((Ed[k] - Eu[k]) / Eo[k], depths, color=colo[k], markeredgecolor=colo[k], markerfacecolor="none", linestyle=ls[k], marker="o", markersize=4, label=labe_nm[k])

        ax1[1, i].text(-0.05, 1.05, "(" + string.ascii_lowercase[3 + i] + ")", transform=ax1[1, i].transAxes, size=11, weight='bold')

        ax3[0].plot(r.attenuation_coefficient(Ed[k], depths), depths, linestyle=ls[k], color=colo[k],  markeredgecolor=colo[k], markerfacecolor="none", marker="o", markersize=4, label=labe_nm[k])
        ax3[1].plot((Ed[k] - Eu[k]), depths, linestyle=ls[k], color=colo[k],  markeredgecolor=colo[k], markerfacecolor="none", marker="o", markersize=4, label=labe_nm[k])
        ax3[2].plot(absorption[k], depths, color=colo[k],  markeredgecolor=colo[k], markerfacecolor="none", linestyle=ls[k], marker="o", markersize=4, label=labe_nm[k])

    # Figure parameters
    # Figure 1
    ax1[0, 0].set_ylabel("Depth [cm]")

    # Figure 2
    ax1[1, 0].invert_yaxis()
    ax1[1, 0].set_ylabel("Depth [cm]")
    ax1[1, 0].legend(loc="best", fontsize=8)
    ax1[1, 0].set_xlabel("$\mu_{d}$ [a.u.]")
    ax1[1, 1].set_xlabel("$\mu_{u}$ [a.u.]")
    ax1[1, 2].set_xlabel("$\mu$ [a.u.]")

    # Figure rad
    axrad[0].set_yscale("log")
    axrad[0].set_xlabel("Zenith angle [˚]")
    axrad[1].set_xlabel("Zenith angle [˚]")
    axrad[2].set_xlabel("Zenith angle [˚]")
    axrad[0].set_ylabel(r"$\overline{{L}}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")

    # Figure 3
    ax3[0].text(-0.05, 1.05, "(" + string.ascii_lowercase[0] + ")", transform=ax3[0].transAxes, size=11,weight='bold')
    ax3[0].set_xticks(np.arange(-1, 7, 1))
    ax3[0].invert_yaxis()
    ax3[0].set_xlim((-0.1, 6))
    ax3[0].set_ylabel("Depth [cm]")
    ax3[0].set_xlabel("$\kappa_{d}~[\mathrm{m^{-1}}]$")
    ax3[0].legend(loc="best")

    ax3[1].text(-0.05, 1.05, "(" + string.ascii_lowercase[1] + ")", transform=ax3[1].transAxes, size=11,weight='bold')
    ax3[1].set_xscale("log")
    ax3[1].set_ylabel("Depth [cm]")
    ax3[1].set_xlabel("$E_{d} - E_{u}~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax3[1].legend(loc="best")

    ax3[2].text(-0.05, 1.05, "(" + string.ascii_lowercase[2] + ")", transform=ax3[2].transAxes, size=11,weight='bold')
    ax3[2].set_xscale("log")
    ax3[2].set_ylabel("Depth [cm]")
    ax3[2].set_xlabel("$a~[\mathrm{m^{-1}}]$")
    ax3[2].legend(loc="best")

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    figrad.tight_layout()

    # Saving figures
    fig1.savefig("figures/irradiance_profile_aops.pdf", format="pdf", dpi=600)
    fig1.savefig("figures/irradiance_profile_aops.png", format="png", dpi=600)
    fig2.savefig("figures/aops.pdf", format="pdf", dpi=600)

    fig3.savefig("figures/diffuse_attenuation_downwelling.pdf", format="pdf", dpi=600)
    fig3.savefig("figures/attenuation.png", format="png", dpi=600)

    plt.show()
