"""

"""

# Module importation
import h5py
import string
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
import pandas as pd

plt.rcParams.update({'font.size': 14})

# Functions and classes
def azimuthal_average(rad):
    """
    Average of radiance in azimuth direction.

    :return:
    """
    condzero = rad == 0
    rad2 = rad.copy()
    rad2[condzero] = np.nan
    return np.nanmean(rad2, axis=1)


def irradiance(zeni, azi, radm, zenimin, zenimax, planar=True):
    """
    Estimate irradiance from the radiance angular distribution. By default, it calculates the planar irradiance.
    By setting the parameter planar to false, the scalar irradiance is computed. Zenimin = 0˚ and Zenimax = 90˚ gives
    the downwelling irradiance, while Zenimin = 90° and Zenimax = 180˚ gives the upwelling irradiance.

    :param zeni: zenith meshgrid in degrees
    :param azi: azimuth meshgrid in degrees
    :param radm: radiance angular distribution
    :param zenimin: min zenith in degrees
    :param zenimax: max zenith in degrees
    :param planar: if True - planar radiance, if false - scalar (bool)
    :return:
    """

    mask = (zenimin <= zeni) & (zeni <= zenimax)
    irr = np.array([])
    zeni_rad = zeni * np.pi / 180
    azi_rad = azi * np.pi / 180
    for b in range(radm.shape[2]):

        # Integrand
        if planar:
            integrand = radm[:, :, b][mask] * np.absolute(np.cos(zeni_rad[mask])) * np.sin(zeni_rad[mask])
        else:
            integrand = radm[:, :, b][mask] * np.sin(zeni_rad[mask])

        azimuth_inte = integrate.simps(integrand.reshape((-1, azi_rad.shape[1])), azi_rad[mask].reshape((-1, azi_rad.shape[1])), axis=1)
        e = integrate.simps(azimuth_inte, zeni_rad[mask].reshape((-1, azi_rad.shape[1]))[:, 0], axis=0)

        irr = np.append(irr, e)

    return irr


def open_radiance_data(path="data/oden-08312018.h5"):
    """
    Function to open data stored in hdf5 file.

    :param path: relative or absolute path to file
    :return: (zenith meshgrid, azimuth meshgrid, radiance) (dct)
    """

    radiance_profile = {}
    with h5py.File(path) as hf:
        data = hf
        for k in data.keys():
            if k not in ["azimuth", "zenith"]:
                radiance_profile[k] = data[k][:]

        zenith_mesh = data["zenith"][:]
        azimuth_mesh = data["azimuth"][:]

    return zenith_mesh, azimuth_mesh, radiance_profile


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
    Extrapolation of missing angles (due reduced FOV due to water refractive index) using a gaussian function.

    :param zenith_meshgrid: zenith meshgrid in degrees (array)
    :param angular_radiance_distribution: current radiance angular distribution (array)
    :return: interpolated radiance angular distribution (array)
    """

    ard = angular_radiance_distribution.copy()  # Angular radiance distribution
    rad_zen = azimuthal_average(ard)  # Perform azimuthal average

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


def create_irradiance_data(zenith_mesh, azimuth_mesh, radiance_mesh, keys_ordered, oden=True):
    """
    Function that output irradiance data from radiance simulations using DORT2002.
    :param zenith_mesh:
    :param azimuth_mesh:
    :param radiance_mesh:
    :return:
    """
    ed = np.zeros(len(keys_ordered), dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4')]))
    eu, eo = ed.copy(), ed.copy()

    # LOOP
    for i, ke in enumerate(keys_ordered):

        print(ke)
        dort_rad = radiance_mesh[ke]

        if oden:
            dort_rad = extrapolation(zenith_mesh, radiance_mesh[ke])

        ed[i] = tuple(irradiance(zenith_mesh, azimuth_mesh, dort_rad, 0, 90))
        eu[i] = tuple(irradiance(zenith_mesh, azimuth_mesh, dort_rad, 90, 180))
        eo[i] = tuple(irradiance(zenith_mesh, azimuth_mesh, dort_rad, 0, 180, planar=False))

    return ed, eu, eo


def graph_dort_vs_measurements(depths, irradiance_dort, irradiance_meas, HE60_df):
    """

    :param irradiance_dort:
    :param irradiance_meas:
    :return:
    """

    i_d, i_u, i_o = irradiance_meas
    i_d_dort, i_u_dort, i_o_dort = irradiance_dort

    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(13,8))

    band_name = ["r", "g", "b"]

    HE_60_bands = ['600.0', '540.0', '480.0']
    depths_HE60 = HE60_df['depths']

    for b in range(3):

        ax[b].plot(i_d[band_name[b]], depths, linewidth=1.4, color="#003f5c", linestyle="-", label="$E_{d}$")
        ax[b].plot(i_u[band_name[b]], depths, linewidth=1.4, color="#bc5090", linestyle="-", label="$E_{u}$")
        ax[b].plot(i_o[band_name[b]], depths, linewidth=1.4, color="#ffa600", linestyle="-", label="$E_{0}$")

        HE60_df[f'Ed_{HE_60_bands[b]}'][1:] = HE60_df[f'Ed_{HE_60_bands[b]}'][1:]/1.355**2
        HE60_df[f'Eu_{HE_60_bands[b]}'][1:] = HE60_df[f'Eu_{HE_60_bands[b]}'][1:]/1.355**2
        HE60_df[f'Eo_{HE_60_bands[b]}'][1:] = HE60_df[f'Eo_{HE_60_bands[b]}'][1:]/1.355**2

        ax[b].plot(HE60_df[f'Ed_{HE_60_bands[b]}'], depths_HE60*100, linewidth=2.0, color="#003f5c", linestyle="dotted", label=r'$E_u$ HE')
        ax[b].plot(HE60_df[f'Eu_{HE_60_bands[b]}'], depths_HE60*100, linewidth=2.0, color="#bc5090", linestyle="dotted", label=r'$E_d$ HE')
        ax[b].plot(HE60_df[f'Eo_{HE_60_bands[b]}'], depths_HE60*100, linewidth=2.0, color="#ffa600", linestyle="dotted", label=r'$E_o$ HE')

        ax[b].plot(i_d_dort[band_name[b]], depths, linewidth=2.0, color="#003f5c", linestyle="--", label=r'$E_u$ DORT')
        ax[b].plot(i_u_dort[band_name[b]], depths, linewidth=2.0, color="#bc5090", linestyle="--", label=r'$E_d$ DORT')
        ax[b].plot(i_o_dort[band_name[b]], depths, linewidth=2.0, color="#ffa600", linestyle="--", label=r'$E_o$ DORT')


        ax[b].set_ylim([0, 200])
        ax[b].set_xscale("log")
        ax[b].invert_yaxis()

        ax[b].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
        ax[b].text(-0.05, 1.05, "(" + string.ascii_lowercase[b] + ")", transform=ax[b].transAxes, size=11, weight='bold')

        # ax[b].legend(loc="best", frameon=False, fontsize=6)
        # ax[b].annotate("- : measurements\n-- : DORT\n·· : HE  ", (0.04, 0.7), xycoords="axes fraction", fontsize=6)
    ax[2].legend()
    ax[0].set_ylabel("Depth [cm]")

    fig.tight_layout()

    return fig, ax


def azimuthal_average(mapped_radiance):
    """
    Average of radiance in azimuth direction.

    :return:
    """
    if len(mapped_radiance.shape) > 1:
        map_rad = mapped_radiance.copy()
        mask_zero = mapped_radiance == 0

        map_rad[mask_zero] = np.nan

        return np.nanmean(map_rad, axis=1)
    else:
        raise ValueError("Build radiance map before any integration.")


def colorbardepth(f, a, cmapp, valdum):
    """
    Function to add colorbar to the profile figure.

    :param f:
    :param a:
    :param cmapp:
    :param diction:
    :return:
    """

    dcax = a.scatter(valdum, valdum, c=np.arange(1, cmapp.N+1), cmap=cmapp)

    a.cla()
    cb = f.colorbar(dcax, ax=a, orientation="vertical")

    cb.ax.locator_params(nbins=cmapp.N)
    cb.ax.set_yticklabels(["{0:.0f}".format(j) for j in valdum])
    cb.ax.set_title("depth [cm]", fontsize=8)
    cb.ax.invert_yaxis()

    return f, a


if __name__ == "__main__":

    # Open measurements
    zen_oden, azi_oden, rad_oden = open_radiance_data(path="data/oden-08312018.h5")  # Path à changer

    ed_oden, eu_oden, eo_oden = create_irradiance_data(zen_oden, azi_oden, rad_oden, ["zero minus",
                                                                                      "20 cm (in water)",
                                                                                      "40 cm",
                                                                                      "60 cm",
                                                                                      "80 cm",
                                                                                      "100 cm",
                                                                                      "120 cm",
                                                                                      "140 cm",
                                                                                      "160 cm",
                                                                                      "180 cm",
                                                                                      "200 cm"], oden=True)

    # Open dort simulation
    zen_dort, azi_dort, rad_dort = open_radiance_data(path="data/dort-simulation.h5")  # Path à changer
    ed_dort, eu_dort, eo_dort = create_irradiance_data(zen_dort, azi_dort, rad_dort, ["0 cm",
                                                                                      "20 cm",
                                                                                      "40 cm",
                                                                                      "60 cm",
                                                                                      "80 cm",
                                                                                      "100 cm",
                                                                                      "120 cm",
                                                                                      "140 cm",
                                                                                      "160 cm",
                                                                                      "180 cm",
                                                                                      "200 cm"])
    # path = '/Users/braulier/Documents/Maitrise/Raphael/radiance_camera_insta360/source/data/HE60_results'
    path = '/Users/braulier/Documents/Maitrise/Raphael/radiance_camera_insta360/source/data/HE60_results.txt'
    HE60_dataframe = pd.read_csv(path)
    fig1, ax1 = graph_dort_vs_measurements(np.arange(0, 220, 20), (ed_dort, eu_dort, eo_dort), (ed_oden, eu_oden, eo_oden), HE60_dataframe)
    # plt.savefig('data/figures/HE_results_comp_dort_zoom.png')
    plt.show()
