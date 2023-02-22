# -*- coding: utf-8 -*-
"""

"""
# Module importation
import numpy as np
import miepython
import pandas
from scipy.integrate import simps
import matplotlib.pyplot as plt
import pandas as pd


# Classes and functions
def brines_psd(lengths):
    """
    Size distribution of brines fitted by Light et al. (2003)
    :param lengths:
    :return:
    """
    return 0.28 * lengths ** (-1.96)


def particulate_size_distribution(lengths, amplitude, slope):
    """

    :param lengths:
    :param amplitude:
    :param slope:
    :return:
    """

    return amplitude * lengths ** (-slope)


def brines_aspect_ratio(lengths):
    """
    Aspect ratio vertical to horizontal for brines channels from Light et al. (2003).

    :param lengths: brines length [mm]
    :return: aspect ratio
    """
    ar = 10.3 * (lengths ** 0.67)
    ar[np.argwhere(lengths < 0.03)] = 1
    return ar


def equivalent_radius(volume):
    """
    Given a volume, the corresponding radius considering a sphere.

    :param volume:
    :return:
    """
    return ((3 * volume) / (4 * np.pi)) ** (1 / 3)


def volume_prolate_ellipsoid(l, asp):
    """
    Volume of a prolate ellipsoid given the major axis l and the vertical-to-horizontal aspect ratio.
    :param l:
    :param asp:
    :return:
    """
    return ((4 * np.pi) / 3) * (l ** 3 / asp ** 2)


def volume_cylinder(l, asp):
    """
    Volume of a cylinder given the height and the vertical-to-horizontal aspect ratio.
    :param l:
    :param asp:
    :return:
    """

    return (np.pi / 4) * (l ** 3 / asp ** 2)


def get_brines_equivalent_radius(brines_l):
    """

    :param brines_l:
    :return:
    """
    # Initialize volume array
    vol = np.zeros(brines_l.shape)

    aspect_ratio = brines_aspect_ratio(brines_l)  # Aspect ratio

    mask_ell = brines_l <= 0.5
    vol[mask_ell] = volume_prolate_ellipsoid(brines_l[mask_ell], aspect_ratio[mask_ell])
    vol[~mask_ell] = volume_cylinder(brines_l[~mask_ell], aspect_ratio[~mask_ell])

    return equivalent_radius(vol)


def mie_scattering_all_radius(radius, wl, m, number_a):
    """
    
    :param radius: 
    :param wl: 
    :param m: 
    :param number_a: 
    :return: 
    """

    a = np.linspace(0.00001, 179.99999, number_a)
    arad = a * np.pi / 180
    mu = np.cos(arad)

    # Pre-allocation
    pf = np.zeros((a.shape[0], radius.shape[0]))
    g_all = np.array([])

    for i, r in enumerate(radius):

        x = 2 * np.pi * r / wl
        s1, s2 = miepython.mie_S1_S2(m, x, mu)
        qext, qsca, qback, g = miepython.mie(m, x)
        g_all = np.append(g_all, g)
        pf[:, i] = 0.5 * (abs(s1) ** 2 + abs(s2) ** 2)

        #total = 2 * np.pi * np.trapz(pf[:, i][::-1], mu[::-1])
        #total = 2 * np.pi * np.trapz(pf[:, i] * np.sin(arad), arad)

        total = 2 * np.pi * simps(pf[:, i] * np.sin(arad), x=arad)  # normalization Best using simpson methods
        pf[:, i] /= total  # normalization

    return a, pf, g_all


def average_phase_function(pf, l):
    """

    :param pf:
    :param lengths:
    :return:
    """

    new_pf = np.zeros(pf.shape[0])
    psd = particulate_size_distribution(l, 0.28, 1.96)
    for r in range(pf.shape[0]):

        new_pf[r] = simps(pf[r, :] * psd, x=l) / simps(psd, x=l)

    return new_pf


def calculate_asymmetry_g(theta, pf):
    """

    :param theta: angles [degrees] - array
    :param pf:
    :return:
    """
    theta_rad = theta * np.pi / 180

    return 2 * np.pi * simps(pf * np.cos(theta_rad) * np.sin(theta_rad), x=theta_rad)


def hg(g, costheta):
    """
    Henyey-Greenstein
    :param g:
    :param costheta:
    :return:
    """
    return (1/4/np.pi)*(1-g**2)/(1+g**2-2*g*costheta)**1.5


if __name__ == "__main__":

    # ***** Phase function for brines *****
    br_lengths = np.logspace(-2, 1, 100, endpoint=True)  # mm
    br_ar = brines_aspect_ratio(br_lengths)
    br_eq_radius = get_brines_equivalent_radius(br_lengths)  # equivalent radius with (volume constant) mm

    # MIE simulations
    #num_angle = 500
    num_angle = 1000
    br_eq_radius_microns = br_eq_radius * 1000  # microns

    # Refractive indexes
    n_brines = 1.344 - 0j
    n_ice = 1.311
    n_rel = n_brines / n_ice
    #n_rel = 1.03 - 0j

    df_ice_refr = pandas.read_excel("Warren_1984.xlsx")
    mask_df = (df_ice_refr["λ (μm)"] >= 0.4) & (df_ice_refr["λ (μm)"] <= 0.8)
    df_ice_refr = df_ice_refr[mask_df]

    wle = 0.484  # microns
    wle_rel = wle / n_ice  # microns

    ang, br_pf, br_g = mie_scattering_all_radius(br_eq_radius_microns, wle_rel, n_rel, num_angle)

    # Verification normalization
    total = 2 * np.pi * simps(br_pf[:, 10] * np.sin(ang * np.pi / 180), x=ang * np.pi / 180)

    # Calculate average phase function with size distribution
    br_pf_avg = average_phase_function(br_pf, br_lengths)  # Phase function for
    g_br_pf = calculate_asymmetry_g(ang, br_pf_avg)

    # Henyey-Greenstein
    henyey = hg(0.98, np.cos(ang * np.pi / 180))
    g_henyey = calculate_asymmetry_g(ang, henyey)

    # Reduced scattering coefficient
    b_oden = np.array([2323, 308, 81])
    b_scaled = b_oden * (1 - 0.98) / (1 - g_br_pf)

    # Save data
    #df = pd.DataFrame({"Angle [deg]": ang, "PF [1/sr]": br_pf_avg})
    #df.to_csv(r"brine_{0:.0f}nm_196e-2.txt".format(wle * 1000))

    # Figures
    # Figure 2
    fig2, ax2 = plt.subplots(3, 1, figsize=(6.4, 6), sharex=True)

    ax2[0].plot(br_lengths, brines_psd(br_lengths), linestyle="-", marker=".")

    ax2[0].set_yscale("log")
    ax2[0].set_xscale("log")
    ax2[0].set_ylabel("PSD [# / $\mathrm{mm^{3}}$ / unit length]", fontsize=10)

    ax2[1].plot(br_lengths, br_ar, linestyle="-", marker=".")

    ax2[1].set_yscale("log")
    ax2[1].set_xscale("log")
    ax2[1].set_ylabel("Aspect ratio")

    ax2[2].plot(br_lengths, br_eq_radius, linestyle="-", marker=".", label="Constant volume")

    ax2[2].legend(loc="best")
    ax2[2].set_xscale("log")
    ax2[2].set_xlabel("Brines inclusion length [mm]")
    ax2[2].set_ylabel("Equivalent radius [mm]")

    # Figure 3 - phase function
    fig3, ax3 = plt.subplots(1, 1)

    ax3.plot(ang, br_pf[:, 0], label="$r = {0:.5f}$ mm".format(br_eq_radius[0]))
    ax3.plot(ang, br_pf[:, 50], label="$r = {0:.5f}$ mm".format(br_eq_radius[50]))
    ax3.plot(ang, br_pf[:, -1], label="$r = {0:.5f}$ mm".format(br_eq_radius[-1]))
    ax3.plot(ang, br_pf_avg, label="PSD scaled Mie")
    ax3.plot(ang, henyey, label="Henyey-Greenstein $g={0:.3f}$".format(0.98))

    ax3.legend(loc="best")
    ax3.set_yscale("log")
    ax3.set_xlabel("Scattering angle [degrees]")
    ax3.set_ylabel("Phase function [1/sr]")

    # Figure 4 - phase function
    fig4, ax4 = plt.subplots(1, 1)

    ax4.set_title(r"$n_{{rel}}$ = $n_{{brines}}$ \ $n_{{ice}}$ = ${0:.3f}$, $\lambda = {1:.1f}$ nm".format(n_rel, wle * 1000), fontsize=8)
    ax4.plot(ang, br_pf_avg, label="PSD scaled Mie phase function")
    ax4.plot(ang, henyey, label="Henyey-Greenstein $g={0:.3f}$".format(0.98))

    ax4.legend(loc="best")
    ax4.set_yscale("log")
    ax4.set_xlabel("Scattering angle [degrees]")
    ax4.set_ylabel("Phase function [1/sr]")

    fig2.tight_layout()

    plt.show()
