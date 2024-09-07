# -*- coding: utf-8 -*-
"""
Baie des Chaleurs by Bastien.
"""

# Module importation
import os
import h5py
import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Other modules
from source.radiance import attenuation_coefficient
from bdc_process_stations import create_label, get_ice_freeboard
from source.radiance import RadClass


# Function and classes
def load_dict_from_hdf5(filename="data/baiedeschaleurs-03232022-imf-fluo.h5"):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[:]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


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


def create_irradiance_data(zenith_mesh, azimuth_mesh, radiance_mesh, keys_ordered, ice_freeboard):
    """
    Function that output irradiance data from radiance simulations using DORT2002.
    :param zenith_mesh:
    :param azimuth_mesh:
    :param radiance_mesh:
    :return:
    """
    ed = np.zeros(len(keys_ordered), dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4'), ('depth', 'f4')]))
    eu, eo = ed.copy(), ed.copy()

    # LOOP
    for i, ke in enumerate(keys_ordered):

        de = "{0} cm".format(ke)
        print(de)

        dort_rad = radiance_mesh[de]

        if ke >= ice_freeboard:
            dort_rad = extrapolation(zenith_mesh, radiance_mesh[de])

        ed[i] = tuple(irradiance(zenith_mesh, azimuth_mesh, dort_rad, 0, 90)) + (ke, )
        eu[i] = tuple(irradiance(zenith_mesh, azimuth_mesh, dort_rad, 90, 180)) + (ke, )
        eo[i] = tuple(irradiance(zenith_mesh, azimuth_mesh, dort_rad, 0, 180, planar=False)) + (ke, )

    return ed, eu, eo


def generate_graph(ed_tuple, eu_tuple, eo_tuple, stations_list):
    """
    All curves in one graph.

    :param ed_tuple:
    :param eu_tuple:
    :param eo_tuple:
    :param stations_list:
    :return:
    """

    fig, ax = plt.subplots(1, 3, figsize=(8.448, 4.872), sharey=True, sharex=True)

    bands = ["r", "g", "b"]
    lstyle = ["-", "--", "-.", ":"]
    color_dct = {"r": "red", "g": "green", "b": "blue"}
    for k, ir in enumerate(zip(ed_tuple, eu_tuple, eo_tuple)):

        edown, eup, escal = ir

        st = stations_list[k]

        for i, a in enumerate(bands):
            legend_label = st + " {0}".format(color_dct[a])
            ax[0].plot(edown[a], edown["depth"], color=a, linestyle=lstyle[k], linewidth=0.8, label=legend_label)
            ax[1].plot(eup[a], edown["depth"],  color=a, linestyle=lstyle[k], linewidth=0.8, label=legend_label)
            ax[2].plot(escal[a], edown["depth"],  color=a, linestyle=lstyle[k], linewidth=0.8, label=legend_label)

    ax[0].set_ylabel("Depth [cm]")
    ax[0].invert_yaxis()

    ax[0].set_xscale("log")

    ax[0].set_xlabel("$E_{d}~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax[1].set_xlabel("$E_{u}~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
    ax[2].set_xlabel("$E_{0}~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")

    ax[0].legend(loc="best", frameon=False, fontsize=6)
    ax[1].legend(loc="best", frameon=False, fontsize=6)
    ax[2].legend(loc="best", frameon=False, fontsize=6)

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":

    # Load data
    data = load_dict_from_hdf5(filename="data/baiedeschaleurs-03232022-imf-fluo.h5")

    # Create irradiance data
    label_st1 = list(create_label("data/station_1_data.txt").keys())
    label_st2 = list(create_label("data/station_2_data.txt").keys())
    label_st3 = list(create_label("data/station_3_data.txt").keys())
    label_st4 = list(create_label("data/station_4_data.txt").keys())

    ifb_st1 = get_ice_freeboard("data/station_1_data.txt")
    ifb_st2 = get_ice_freeboard("data/station_2_data.txt")
    ifb_st3 = get_ice_freeboard("data/station_3_data.txt")
    ifb_st4 = get_ice_freeboard("data/station_4_data.txt")

    ed_st1, eu_st1, eo_st1 = create_irradiance_data(data["station_1"]["zenith"] * 180/np.pi,
                                                    data["station_1"]["azimuth"] * 180/np.pi,
                                                    data["station_1"], label_st1, ifb_st1)

    ed_st2, eu_st2, eo_st2 = create_irradiance_data(data["station_2"]["zenith"] * 180 / np.pi,
                                                    data["station_2"]["azimuth"] * 180 / np.pi,
                                                    data["station_2"], label_st2, ifb_st2)

    ed_st3, eu_st3, eo_st3 = create_irradiance_data(data["station_3"]["zenith"] * 180 / np.pi,
                                                    data["station_3"]["azimuth"] * 180 / np.pi,
                                                    data["station_3"], label_st3, ifb_st3)

    ed_st4, eu_st4, eo_st4 = create_irradiance_data(data["station_4"]["zenith"] * 180 / np.pi,
                                                    data["station_4"]["azimuth"] * 180 / np.pi,
                                                    data["station_4"], label_st4, ifb_st4)

    ed_tup = (ed_st1, ed_st2, ed_st3, ed_st4)
    eu_tup = (eu_st1, eu_st2, eu_st3, eu_st4)
    eo_tup = (eo_st1, eo_st2, eo_st3, eo_st4)

    # Gershun law estimation of absorption coefficient
    absorption_st1 = np.zeros(ed_st1.shape[0], dtype=([('r', 'f4'), ('g', 'f4'), ('b', 'f4')]))

    absorption_st1["r"] = attenuation_coefficient((ed_st1["r"] - eu_st1["r"]), ed_st1["depth"]) * ((ed_st1["r"] - eu_st1["r"]) / eo_st1["r"])
    absorption_st1["g"] = attenuation_coefficient((ed_st1["g"] - eu_st1["g"]), ed_st1["depth"]) * ((ed_st1["g"] - eu_st1["g"]) / eo_st1["g"])
    absorption_st1["b"] = attenuation_coefficient((ed_st1["b"] - eu_st1["b"]), ed_st1["depth"]) * ((ed_st1["b"] - eu_st1["b"]) / eo_st1["b"])

    # Transmittance
    tst1 = np.array([ed_st1[-1]["r"]/ed_st1[0]["r"], ed_st1[-1]["g"]/ed_st1[0]["g"], ed_st1[-1]["b"]/ed_st1[0]["b"]])
    tst2 = np.array([ed_st2[-1]["r"] / ed_st2[0]["r"], ed_st2[-1]["g"] / ed_st2[0]["g"], ed_st2[-1]["b"] / ed_st2[0]["b"]])
    tst3 = np.array([ed_st3[-1]["r"] / ed_st3[0]["r"], ed_st3[-1]["g"] / ed_st3[0]["g"], ed_st3[-1]["b"] / ed_st3[0]["b"]])
    tst4 = np.array([ed_st4[-1]["r"] / ed_st4[0]["r"], ed_st4[-1]["g"] / ed_st4[0]["g"], ed_st4[-1]["b"] / ed_st4[0]["b"]])

    # Figures
    # Figure 1
    fig1, ax1 = generate_graph(ed_tup, eu_tup, eo_tup, ["station 1", "station 2", "station 3", "station 4"])

    # Figure 2
    fig2, ax2 = plt.subplots(1, 1)

    ax2.plot(absorption_st1["r"][2:], ed_st1["depth"][2:], color="r")
    ax2.plot(absorption_st1["g"][2:], ed_st1["depth"][2:], color="g")
    ax2.plot(absorption_st1["b"][2:], ed_st1["depth"][2:], color="b")

    ax2.invert_yaxis()

    # Save fig
    fig1.savefig("figures/bdc_allcurves.png", dpi=600, format="png")

    plt.show()
