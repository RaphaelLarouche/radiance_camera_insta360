"""
Script to fit the Legendre Polynomial moments of the Fournier-Forand phase function.
"""


# Module importation
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre


# Function and classes
def BHG(g):
    """
    Backscattering ratio. Analytic expression of the integration of the Henyey-Greenstein phase function between angles
    of pi/2 and pi.

    :param g: Asymmetry factor
    :return: Backscattering ratio
    """
    return ((1 - g) / (2 * g)) * (((1 + g) / (1 + g**2)**(1/2)) - 1)


def delta_FF(n, scatt_angle):
    """
    Computation of delta in Fournier-Forand phase function equation.
    :param n:
    :param scatt_angle: scattering angle in degrees
    :return:
    """

    s_angle_rad = scatt_angle * np.pi / 180
    term_front = 4 / (3 * (n - 1) ** 2)
    si = np.sin(s_angle_rad / 2) ** 2

    return term_front * si


def v_FF(u):
    """
    Computation of v parameters in Fournier-Forand phase function equation.
    :param u:
    :return:
    """

    return (3 - u) / 2


def inversion_u(n, backscatter_fraction):
    """
    From Eq.(3) of the paper: Phase function effects on oceanic light fields, Mobley et al. (2002)

    :param n:
    :param backscatter_fraction:
    :return:
    """
    d90 = delta_FF(n, 90)
    A = (1 - backscatter_fraction) * (1 - d90)
    term = 0.5 / (A - 0.5 + d90)
    v = np.log(term) / np.log(d90)

    return 3 - v * 2


def fournier_forand_pf(n, u, scattering_angles):
    """

    :param n:
    :param u:
    :param scattering_angles: scattering angle in degrees
    :return:
    """
    d = delta_FF(n, scattering_angles)
    d180 = delta_FF(n, 180)
    v = (3 - u) / 2

    scattering_angles_rad = scattering_angles * np.pi / 180

    A = 1 / (4 * np.pi * d ** v * (1 - d) ** 2)
    B = v * (1 - d) - (1 - d ** v)
    C = (d * (1 - d ** v) - v * (1 - d)) * np.sin(scattering_angles_rad / 2) ** (-2)
    D = ((1 - d180 ** v) / (16 * np.pi * d180 ** v * (d180 - 1))) * ((3 * np.cos(scattering_angles_rad) ** 2) - 1)

    # A = 1 / (2 * d ** v * (1 - d) ** 2)
    # B = v * (1 - d) - (1 - d ** v)
    # C = (d * (1 - d ** v) - v * (1 - d)) * np.sin(scattering_angles_rad / 2) ** (-2)
    # D = ((1 - d180 ** v) / (8 * d180 ** v * (d180 - 1))) * (3 * np.cos(scattering_angles_rad) ** 2 - 1)

    return (A * (B + C)) + D


def henyey_greenstein_p(scattering_angles, assymetry_param):
    """

    :param scattering_angles:
    :param assymetry_param:
    :return:
    """

    s_an_rad = scattering_angles * np.pi / 180

    nom = 1 - assymetry_param ** 2
    denom = (1 + assymetry_param ** 2 - 2 * assymetry_param * np.cos(s_an_rad)) ** (3 / 2)

    return (1 / (4 * np.pi)) * nom / denom


def calculate_moments(order, accurate_phase_function, costheta):
    """

    :param order:
    :param accurate_phase_function:
    :param costheta:
    :return:
    """
    # First estimation
    moments = np.array([])
    for n in range(order):
        p = legendre(n)
        xl = 2 * np.pi * integrate.simps(p(costheta)[::-1] * accurate_phase_function[::-1], x=costheta[::-1])
        moments = np.append(moments, xl)

    # Delta M method
    f = moments[-1]
    moments_prim = (moments - f) / (1 - f)

    return moments_prim * (1 - f), f


def calculate_moments_orthogonality(order, accurate_pf, theta):
    """

    :param order:
    :param accurate_pf:
    :param theta: angle in degrees
    :return:
    """

    a = np.array([])
    t_rad = theta * np.pi / 180
    cos_t = np.cos(t_rad)
    sin_t = np.sin(t_rad)
    for n in range(order):
        Pn = legendre(n)
        An = 2 * np.pi * integrate.simps(accurate_pf[::-1] * Pn(cos_t)[::-1] * sin_t[::-1], x=t_rad[::-1])
        a = np.append(a, An)

    return a


def numeric_calculation_mean_cosine(phase_function, angles):
    """

    :param phase_function:
    :param angles:
    :return:
    """
    angles_rad = angles * np.pi / 180
    costheta = np.cos(angles_rad)

    return 2 * np.pi * integrate.simps(phase_function * np.cos(angles_rad) * np.sin(angles_rad), x=angles_rad)
    #return 2 * np.pi * np.trapz(phase_function[::-1] * costheta[::-1], costheta[::-1])


def henyey_greenstein_legendre(cos_theta, g, order):
    """

    :param cos_theta:
    :param g:
    :param order:
    :return:
    """
    p_tot = np.empty((order, cos_theta.shape[0]))

    for l in range(order):

        pl = legendre(l)
        p_tot[l, :] = (2 * l + 1) * (g ** l) * pl(cos_theta)

    return p_tot


def return_moments_HG(g, order):
    """

    :param order:
    :return:
    """
    m = np.array([])
    for l in range(order):
        m = np.append(m, g ** l)

    return m


def normalisation_verification(pf, angles_deg):
    """

    :param pf:
    :param angles_deg:
    :return:
    """

    angles_rad = angles_deg * np.pi / 180
    costheta = np.cos(angles_rad)
    #return 2 * np.pi * integrate.simps(pf * np.sin(angles_rad), x=angles_rad)
    return 2 * np.pi * np.trapz(pf[::-1], costheta[::-1])


if __name__ == "__main__":

    # Parameters of FF phase function
    g = 0.98
    b_fraction = BHG(g)   # backscattering function from Henyey-Greenstein asymmetry param
    refractive_index = 1.33
    u_param = inversion_u(refractive_index, b_fraction)  # From Eq.(3) of the paper: Phase function effects on oceanic light fields, Mobley et al. (2002)

    # Angles
    s_angle = np.linspace(0.01, 180, 361)
    s_angle_cos = np.cos(s_angle * np.pi/180)

    b_ff = fournier_forand_pf(refractive_index, u_param, s_angle)
    #b_ff = fournier_forand_pf(1.35, 2.4, s_angle)
    #b_ff = henyey_greenstein_p(s_angle, g)
    #b_ff/=normalisation_verification(b_ff, s_angle)
    print(numeric_calculation_mean_cosine(b_ff, s_angle))

    # Legendre polynomial fit
    N = 30
    n_streams = 2 * N + 1

    # Legendre polynomial fit
    moments_xl, f = calculate_moments(n_streams, b_ff, s_angle_cos)
    p_prim = np.empty((n_streams-1, s_angle_cos.shape[0]))
    p_prim_deltaM = np.empty((n_streams-1, s_angle_cos.shape[0]))
    front_param = np.empty(moments_xl.shape)

    #moments_xl = return_moments_HG(0.98, n_streams)
    #moments_xl = moments_xl / moments_xl[0]
    #moments_xl = calculate_moments_orthogonality(n_streams, b_ff, s_angle)
    for i in range(n_streams-1):
        P = legendre(i)
        p_prim_deltaM[i, :] = (1 / (4 * np.pi)) * (2 * i + 1) * moments_xl[i] * P(s_angle_cos)

    # cond_diract = 1 - s_angle_cos <= 1e-5
    # delta_m_recomputed_bff = 2 * f * cond_diract + np.sum(p_prim_deltaM, axis=0)

    delta_m_recomputed_bff = np.sum(p_prim_deltaM, axis=0)
    print(numeric_calculation_mean_cosine(delta_m_recomputed_bff, s_angle))

    # Loop to test different amounts of backscatter
    backscatter_f = BHG(np.arange(0.6, 1.0, 0.05))
    fig3, ax3 = plt.subplots(1, 2)
    for bfraction in backscatter_f:
        u = inversion_u(refractive_index, bfraction)
        b_ff_loop = fournier_forand_pf(refractive_index, u, s_angle)

        ax3[0].plot(s_angle, b_ff_loop, label="{0:.5f}".format(bfraction))
        ax3[1].plot(s_angle, b_ff_loop / b_ff_loop[0], label="B = {0:.5f}".format(bfraction))

    # Fit Christophe
    # eng = matlab.engine.start_matlab()
    # b_ff_matlab = matlab.double(b_ff.tolist())
    # s_angle_matlab = matlab.double(s_angle.tolist())
    #
    # eng.pf_fitter_function(b_ff_matlab, s_angle_matlab, nargout=0)

    # Viz
    # Figure 1
    fig1, ax1 = plt.subplots(1, 2, sharey=True)

    ax1[0].plot(s_angle, b_ff, label="Analytic Fournier-Forand", color="k")
    ax1[0].plot(s_angle, henyey_greenstein_p(s_angle, 0.98), label="Henyey-Greenstein ($g={0:.3f}$)".format(g))
    ax1[0].plot(s_angle, delta_m_recomputed_bff, "-.", label="Fit ($n={0}$)".format(n_streams))
    #ax1[0].plot(s_angle, recomputed_bff_2, "-.")

    ax1[0].set_xscale("log")
    ax1[0].set_yscale("log")
    ax1[0].set_xlabel(r"scattering angle $\theta$ [˚]")
    ax1[0].set_ylabel("phase function [1/sr]")

    ax1[0].legend(loc="best", fontsize=8)

    ax1[1].plot(s_angle_cos, b_ff, color="k", label="Analytic Fournier-Forand")
    ax1[1].plot(s_angle_cos, henyey_greenstein_p(s_angle, 0.98), label="Henyey-Greenstein ($g={0:.3f}$)".format(g))
    ax1[1].plot(s_angle_cos, delta_m_recomputed_bff, '-.', label="Fit ($n={0}$)".format(n_streams))
    ax1[1].set_yscale("log")

    ax1[1].legend(loc="best", fontsize=8)
    ax1[1].set_xlabel(r"$\cos(\theta)$")

    # Figure 2
    fig2, ax2 = plt.subplots(1, 1)

    ax2.plot(s_angle_cos, delta_m_recomputed_bff/b_ff - 1)

    # Figure 3
    ax3[0].set_xscale("log")
    ax3[0].set_yscale("log")
    ax3[0].set_xlabel(r"scattering angle $\theta$ [˚]")
    ax3[0].set_ylabel("phase function [1/sr]")

    ax3[0].legend(loc="best")

    ax3[1].set_xscale("log")
    ax3[1].set_yscale("log")
    ax3[1].set_xlabel(r"scattering angle $\theta$ [˚]")
    ax3[1].set_ylabel("phase function normalized")

    ax3[1].legend(loc="best")

    fig1.tight_layout()
    fig3.tight_layout()

    plt.show()