# -*- coding: utf-8 -*-
"""
Phase function for large inclusions inside sea ice (air bubbles and brines) according to their volume fraction ratio.
"""

# Module importation
import miepython
import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.integrate import simps
from scipy.integrate import quad

# Other modules
from mie_phase_function import calculate_asymmetry_g, hg


# Function and classes
def pf_brines_malinka_2017(angles):
    """
    Phase function for optically soft and large particles according to Malinka et al (2017).
    :param angles: scattering angles [degrees] - array
    :return: phase function normalized to 1, asymmetry param - tuple
    """
    # Angles
    arad = angles * np.pi / 180
    mu = np.cos(arad)

    # Size parameter
    nb = 1.024
    x = 1 / (nb - 1)
    pf = (2 * x ** 2 * (1 + mu ** 2)) / (1 + 2 * (x ** 2) * (1 - mu)) ** 2

    # Calculate normalization param
    norm = 2 * np.pi * simps(pf * np.sin(arad), x=arad)  # Should be close to 4pi
    print(norm / (4 * np.pi))

    # Asymmetry parameter
    g = 1 - (np.log(2 * x) - 1) / x ** 2

    return pf / norm, g


def pf_bubbles_malinka_2017_psd(angles, verbose=False):
    """
    Mie simulation for air bubbles in sea ice with size distribution given by Light et al. 2010 (n(r) = r **-1.5, for
    r between 4 um and 70 um. The wavelength for calculation is 0.650 um and the relative refractive index 0.763.

    :param angles: scattering angles [degrees] - array
    :return: phase function normalized to 1, asymmetry param - tuple
    """
    # Air bubbles effective radius
    ra = np.linspace(4, 70, 500)  # microns, r_avg = 42.55 um

    ang_rad = angles * np.pi / 180
    mu = np.cos(ang_rad)

    # Refractive indexes # TODO: Update will refractive indexes of ice in litterature
    #m = 0.763
    m = 1 / 1.329
    n_ice = 1.0 / m

    # Wavelength
    wave_rel = 0.650 / n_ice  # 650 nm

    # LOOP over all bubbles sizes
    pf = np.zeros((angles.shape[0], ra.shape[0]))
    qsc_mesh = pf.copy()
    g_arr = np.zeros(ra.shape[0])

    for i, r in enumerate(ra):

        # Size parameter
        x = 2 * np.pi * r / wave_rel

        # Mie calculation
        s1, s2 = miepython.mie_S1_S2(m, x, mu)
        qext, qsca, qback, g = miepython.mie(m, x)

        pf_unscaled = 0.5 * (abs(s1) ** 2 + abs(s2) ** 2) # should be normalized to albedo, in this case a = Qsca/Qext = 1

        pf[:, i] = pf_unscaled * qext / qsca
        qsc_mesh[:, i] = qsca
        g_arr[i] = g

        if verbose:
            print("Qext:{0}, Qsca = {1}, Qback = {2}, g = {3}".format(qext, qsca, qback, g))

    # Scaled by size distribution
    n = ra ** (-1.5)  # Size distribution - comes from Light et al. 2010
    n_mesh = np.tile(n, (angles.shape[0], 1))
    r_mesh = np.tile(ra, (angles.shape[0], 1))

    pf_avg = simps(np.pi * r_mesh ** 2 * qsc_mesh * pf * n_mesh, x=r_mesh, axis=1) / \
             simps(np.pi * r_mesh ** 2 * qsc_mesh * n_mesh, x=r_mesh, axis=1)

    g_avg = simps(np.pi * r_mesh[0, :] ** 2 * qsc_mesh[0, :] * g_arr * n_mesh[0, :], x=r_mesh[0, :]) / \
            simps(np.pi * r_mesh[0, :] ** 2 * qsc_mesh[0, :] * n_mesh[0, :], x=r_mesh[0, :])

    # Calculate normalization to 1
    norm = 2 * np.pi * simps(pf_avg * np.sin(ang_rad), x=ang_rad)

    # Effective radius
    r_eff_a = np.trapz((ra ** 3) * n) / np.trapz((ra ** 2) * n)

    return pf_avg / norm, g_avg, r_eff_a


def sea_ice_pf_malinka_2017(angles, fraction_ratio=6e-3):
    """
    Phase function considering both air bubbles and brines as a function of the fraction ratio of volume concentration
    of air over brines.

    :param angles: scattering angles [degrees] - array
    :param fraction_ratio: fraction ratio of volume concentration - float
    :return: total phase function [1/sr]- array
    """
    ph_func_b, gb = pf_brines_malinka_2017(angles)
    ph_func_a, ga, ra = pf_bubbles_malinka_2017_psd(angles)

    rb = 100
    #ra = 42.55
    print(ra)
    sa_over_sb = (rb / ra) * fraction_ratio

    sb_over_s = 1 / (1 + sa_over_sb)
    sa_over_s = 1 - sb_over_s

    ph_tot = sb_over_s * ph_func_b + sa_over_s * ph_func_a
    g_tot = sb_over_s * gb + sa_over_s * ga

    return ph_tot, g_tot


# def calculate_meantheta(fr):
#    """
#
#    :param theta: angles [degrees] - array
#    :param pf:
#    :return:
#    """
    #theta_rad = theta * np.pi / 180

#    pt = lambda x: sea_ice_pf_malinka_2017(x * 180/np.pi, fraction_ratio=fr) * np.cos(x) * np.sin(x)
#
#    return 2 * np.pi * quad(pt, 0, np.pi)


if __name__ == "__main__":

    # Estimation of effective radius of size distribution of bubbles

    r_bubbles = np.linspace(4, 70, 500)  # microns
    N_bubbles = r_bubbles ** (-1.5)

    r_eff_bubbles = np.trapz((r_bubbles ** 3) * N_bubbles) / np.trapz((r_bubbles ** 2) * N_bubbles)  # effective radius - should be 42.55 microns

    # Brines PF
    a = np.linspace(0, 180, 1000)  # angles in degrees
    a_radians = a * np.pi / 180  # angles in radians

    pf_b, g_b = pf_brines_malinka_2017(a)
    g_b_calcu = calculate_asymmetry_g(a, pf_b)

    # Bubbles PF
    pf_a, g_a, r_a = pf_bubbles_malinka_2017_psd(a)

    # H-G
    pf_hg = hg(0.98, np.cos(a_radians))

    # Figure
    fig1, ax1 = plt.subplots(1, 1)

    ax1.plot(a, pf_b, label="Brines, Malinka et al. 2017")
    ax1.plot(a, pf_a, label="Bubbles, Mie")
    ax1.plot(a, pf_hg, label="Henyey-Greenstein, $g=$0.98")

    ax1.set_yscale("log")
    ax1.legend(loc="best")
    ax1.set_xlabel(r"Scattering angle $\theta$ [°]")
    ax1.set_ylabel(r"$p(\theta)$ [1/sr]")

    # Sea ice tot
    fig2, ax2 = plt.subplots(1, 1)
    fr = [0, 6e-3, 6e-2, 6e-1, np.inf]  # fraction ratio to loop

    for frac in fr:

        pf_tot, g_tot_c = sea_ice_pf_malinka_2017(a, fraction_ratio=frac)
        gtot = calculate_asymmetry_g(a, pf_tot)
        ax2.plot(a, pf_tot * 4*np.pi, label="$C_{{a}}^{{v}} / C_{{b}}^{{v}}$={0}, $g = {1:.5f}$".format(frac, gtot))  # in Paper PF are normalized to 4pi

        print(f"g calculated: {gtot}, g theory {g_tot_c}")
        #print(2 * np.pi * simps(pf_tot * np.sin(a_radians), x=a_radians))
        #print(calculate_asymmetry_g(a, pf_tot))

    ax2.set_ylim((0.0001, 10**6))
    ax2.set_xlim((0, 180))
    ax2.set_yscale("log")
    ax2.legend(loc="best")
    ax2.set_xlabel(r"Scattering angle $\theta$ [°]")
    ax2.set_ylabel(r"$p(\theta)$ [1/sr]")

    a_bb = np.linspace(0, 180, 1000)
    pf_tot_085, gtot_085 = sea_ice_pf_malinka_2017(a_bb, fraction_ratio=np.inf)
    g_cal = calculate_asymmetry_g(a_bb, pf_tot_085)

    print(f"g calculated: {g_cal}, g theory {gtot_085}")

    #pdf = pandas.DataFrame({"Angle [deg]": a_bb, "PF [1/sr]": pf_tot_085})
    #pdf.to_csv("malinka_085.csv")

    dat_tthg085 = pandas.read_csv("dpf_TwoTHG_g10_855_g2-0_750_a0_997.csv")

    fig_pf, ax_pf = plt.subplots()

    ax_pf.plot(a_bb, pf_tot_085)
    ax_pf.plot(a_bb, hg(0.85, np.cos(np.radians(a_bb))))
    ax_pf.plot(dat_tthg085["psi[deg]"], dat_tthg085["betatilde[1/sr]"])

    ax_pf.set_yscale("log")

    plt.show()
