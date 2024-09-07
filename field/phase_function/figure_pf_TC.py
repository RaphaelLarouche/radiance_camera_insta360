# -*- coding: utf-8 -*-
"""
Phase functions for HL simulations.
"""

# Module importation
import os
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

import malinka_2017 as mlk
from source.processing import FigureFunctions


# Functions
def pf_brines_mlk_2017(angles_rad):
    """
    Phase function for optically soft and large particles according to Malinka et al (2017).
    :param angles_rad: scattering angles [radians] - array
    :return: phase function normalized to 1, asymmetry param - tuple
    """
    # Angles
    mu = np.cos(angles_rad)

    # Size parameter
    nb = 1.024
    x = 1 / (nb - 1)
    pf = (2 * x ** 2 * (1 + mu ** 2)) / (1 + 2 * (x ** 2) * (1 - mu)) ** 2

    # Asymmetry parameter
    g = 1 - (np.log(2 * x) - 1) / x ** 2

    return pf, g


def hg(angles, g):
    """
    Henyey-Greenstein phase function.

    :param angles: Scattering angles in radians - array
    :param g: asymmetry parameter - float
    :return:
    """
    mu = np.cos(angles)
    return (1 / (4 * np.pi)) * (1 - g ** 2) / (1 + g ** 2 - 2 * g * mu) ** (3 / 2)


def hg_mu(mu, g):
    """
    Henyey-Greenstein phase function.

    :param angles: Scattering angles in degrees - array
    :param g: asymmetry parameter - float
    :return:
    """

    return (1 / (4 * np.pi)) * (1 - g ** 2) / (1 + g ** 2 - 2 * g * mu) ** (3 / 2)


def optthg(angles, g):
    """

    :param angles: rad
    :param g:
    :return:
    """
    h = -0.3061446 + 1.000568 * g - 0.01826332 * g ** 2 + 0.03643748 * g ** 3
    alpha = (h * (1 + h)) / ((g + h) * (1 + h - g))

    return alpha * hg(angles, g) + (1 - alpha) * hg(angles, -h), (g, h, alpha, alpha * (g + h) - h)


if __name__ == "__main__":

    # Phase functions
    a = np.linspace(0, 180, 1000)
    a_rad = a * np.pi / 180

    # Malinka et al. 2017 PF
    fr = 0.025  # fraction ratio of air volume concentration over brine volume concentration
    #fr = 0.0
    pf_mlk, g_mlk = mlk.sea_ice_pf_malinka_2017(angles=a, fraction_ratio=fr)

    norm_pf_brines = quad(lambda x: 2 * np.pi * pf_brines_mlk_2017(x)[0] * np.sin(x), 0, np.pi)[0]
    g_pf_brines = quad(lambda x: 2 * np.pi * (pf_brines_mlk_2017(x)[0]/norm_pf_brines) * np.cos(x) * np.sin(x), 0, np.pi)[0]

    # Henyey-Greenstein (one-term)
    pf_hg = hg(a_rad, 0.99)
    g_hg = quad(lambda x: 2 * np.pi * hg(x, 0.99) * np.cos(x) * np.sin(x), 0, np.pi)[0]

    # Two-terms Henyey-Greenstein
    g1 = 0.996
    pf_tthg, tthg_param = optthg(a_rad, g1)
    g_tthg = quad(lambda x: 2 * np.pi * optthg(x, g1)[0] * np.cos(x) * np.sin(x), 0, np.pi)[0]
    print(g_tthg)

    # Figure
    plt.style.use(os.path.abspath(os.path.join(__file__, "../../..")) + "/figurestyle.mplstyle")
    ff = FigureFunctions()

    #fig1, ax1 = plt.subplots(1, 1, figsize=ff.set_size(fraction=0.78))
    fig1, ax1 = plt.subplots(1, 1, figsize=(11 / 2.54, 11 / (1.33*2.54)))

    ax1.plot(a, pf_mlk, linestyle="-", color="k", label=f"Malinka et al. (2018), $g={g_mlk:.3f}$, $C_{{a}}^{{v}} / C_{{b}}^{{v}}={fr:.3f}$")
    ax1.plot(a, pf_hg, linestyle="--", color="k", label=f"Henyey-Greenstein, $g={g_hg:.3f}$")
    ax1.plot(a, pf_tthg, linestyle="-.", color="k", label=fr"TTHG, $g={tthg_param[3]:.3f}$, $g_{{1}}={tthg_param[0]:.3f}$, $g_{{2}}={tthg_param[1]:.3f}$, $\alpha={tthg_param[2]:.3f}$")

    ax1.set_yscale("log")
    ax1.set_xlabel(r"Scattering angle $\theta$ [°]")
    ax1.set_ylabel(r"$p(\theta)$ [$\mathrm{sr^{-1}}$]")

    ax1.legend(loc="best")

    # Save figure
    fig1.tight_layout()
    #fig1.savefig("figures/pf.pdf", format="pdf", dpi=300)
    #fig1.savefig("figures/pf.png", format="png", dpi=300)

    plt.show()
