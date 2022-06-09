# -*- coding: utf-8 -*-
"""

"""

# Module importation
import os
import h5py
import pandas
import matplotlib
import numpy as np
from scipy.optimize import minimize
from scipy import integrate
import matplotlib.pyplot as plt

# Other module
import radiance as r
from processing import ProcessImage
from source.geometric_rolloff import OpenMatlabFiles

from script_bastian_14_04_2022 import extrapolation, open_radiance_data, create_irradiance_data
from radclass import RadClass
from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer
from HE60PY.phasefunctions import *


# Function and classes
def minimization(init, matlab_engine, radclass, trios):
    """

    :return:
    """
    # Radiance measurements
    a_init, b_init, pf_ice = init
    print(list(b_init))

    root_name = "Eo_fit_multilayer2"
    HE_simulation = SeaIceSimulation(run_title=root_name,
                                     root_name=root_name,
                                     refr=1.00,
                                     mode='HE60DORT',
                                     wavelength_list=[484],
                                     IrradDataFile="HE60BDC_irrad_cops_station_1")
    HE_simulation.set_z_grid(z_max=1.00)
    for i, b in enumerate(b_init):
        top, bot = i * 0.10, (i + 1) * 0.10
        if i < 8:
            HE_simulation.add_layer(z1=top, z2=bot, abs={'484': a_init[i], '544': 0.0683, '602': 0.12}, scat=b, dpf=pf_ice[i]) # bb arg is not relevent since we use a discretized phase function in a file indepêdnant of depth (g=0.98)
        elif i == 8:
            HE_simulation.add_layer(z1=0.80, z2=1.01, abs={'484': 1.36e-2, '544': 5.11e-2, '602': 2.224e-1}, scat=b,
                                    dpf='dpf_OTHG_0_90.txt')
    HE_simulation.run_simulation(printoutput=True)
    HE_simulation.parse_results()
    HE_analyze = DataViewer(root_name=root_name)



    depth_keys_order = ['0.0 cm',
                        '0.1 cm',
                        '5.0 cm',
                        '10.0 cm',
                        '15.0 cm',
                        '20.0 cm',
                        '25.0 cm',
                        '30.0 cm',
                        '40.0 cm',
                        '50.0 cm',
                        '60.0 cm']
    # Open measurements
    zen_oden, azi_oden, rad_oden = open_radiance_data(path="../oden-08312018.h5")  # Path à changer

    band_name = ["r", "g", "b"] # For measurments

    # For HE60 results handling
    depth_list = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    HE_eo = []
    HE_eu = []
    HE_ed = []
    Oden_eo = []
    Oden_eu = []
    Oden_ed = []


    # mre_profile = np.empty((len(depth_keys_order), 3))
    quad_loss = np.empty((len(depth_keys_order)))
    for i, d_keys in enumerate(depth_keys_order):
        # current_rad_map = oden_data[d_keys]
        #
        # # Oden azimuthal average Measurements
        # az_average = r.azimuthal_average(current_rad_map)[19:159, :]
        # Oden eudos
        ed, eu, eo, edo, euo = radclass.create_irradiance_data()
        ed_oden_r, eu_oden_r, eo_oden_r = ed["r"], eu["r"], eo["r"]      # Red
        ed_oden_g, eu_oden_g, eo_oden_g = ed["g"], eu["g"], eo["g"]        # Green
        ed_oden_b, eu_oden_b, eo_oden_b = ed["b"], eu["b"], eo["b"]        # Blue

        # Getting HE60 radiances
        # x_red, zenith_red = HE_analyze.get_zenith_radiance_profile_at_depth(depth_list[i], wavelengths[0])
        # x_green, zenith_green = HE_analyze.get_zenith_radiance_profile_at_depth(depth_list[i], wavelengths[1])
        # x_blue, zenith_blue = HE_analyze.get_zenith_radiance_profile_at_depth(depth_list[i], wavelengths[2])
        # az_average_he60_interp = np.stack((zenith_red, zenith_green, zenith_blue), axis=1)
        # az_average_he60_interp = az_average_he60_interp[19:159, :]
        # Getting HE60 Eudos
        # eu_he_r, eu_he_r, eo_he_r = HE_analyze.get_Eudos_at_depth(depth_list[i], 602) # Red
        # eu_he_g, eu_he_g, eo_he_g = HE_analyze.get_Eudos_at_depth(depth_list[i], 544) # green
        eu_he_b, ed_he_b, eo_he_b = HE_analyze.get_Eudos_at_depth(depth_list[i], 484) # blue

        # Quad loss on Eo blue only
        # rel_err_blue = 100*(eo_oden_b - eo_he_b)/eo_oden_b
        # quad_loss[i] = rel_err_blue**2
        HE_eo.append(eo_he_b)
        HE_eu.append(eu_he_b)
        HE_ed.append(ed_he_b)
        Oden_eo.append(eo_oden_b)
        Oden_eu.append(eu_oden_b)
        Oden_ed.append(ed_oden_b)

        # Mean relative error (MUPD)
        # ref_values = 0.5 * (az_average_he60_interp + az_average)
        # rel_err = np.abs(az_average - az_average_he60_interp) / ref_values

        # mre_profile[i, :] = np.nanmean(rel_err[19:159])
    # mupd = mre_profile.mean(axis=0) * 100
    # funval = np.mean(mupd)
    fig, ax = plt.subplots(1,3)
    ax[0].semilogx(np.array(HE_eo), depth_list, label="Eo HE")
    ax[0].semilogx(Oden_eo, depth_list, label="Eo Oden")
    ax[0].title.set_text('Eo')
    ax[1].semilogx(np.array(HE_eu), depth_list, label="Eu HE")
    ax[1].semilogx(Oden_eu, depth_list, label="Eu Oden")
    ax[1].title.set_text('Eu')
    ax[2].semilogx(np.array(HE_ed), depth_list, label="Ed HE")
    ax[2].semilogx(Oden_ed, depth_list, label="Ed Oden")
    ax[2].title.set_text('Ed')
    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[2].set_xscale("log")
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[2].invert_yaxis()
    plt.legend()
    plt.show()
    print(np.sum(quad_loss))
    total_error = np.sum(quad_loss)
    return total_error


def open_TriOS_data(min_date="2018-08-31 8:00", max_date="2018-08-31 16:00", path_trios="data/2018R4_300025060010720.mat"):
    """

    :param min_date:
    :param max_date:
    :param path_trios:
    :return:
    """
    openmatlab = OpenMatlabFiles()
    radiometer = openmatlab.loadmat(path_trios)
    radiometer = radiometer["raw"]

    timestamps = pandas.to_datetime(radiometer["irrad_t_1"] - 719529, unit='D')   # 719529 = 1 january 2000 Matlab
    df_time = pandas.DataFrame({"Date": timestamps.to_period('H')})
    mask = (df_time["Date"] <= max_date) & (df_time["Date"] >= min_date)
    inx = df_time["Date"][mask].index

    timestamps = timestamps.strftime("%B %d %Y, %H:%M:%S")
    irradiance_incom = radiometer["incom_int_1"][inx].T / 1000  # mW to W
    irradiance_below = radiometer["irrad_int_1"][inx].T / 1000  # mW to W

    return timestamps, radiometer["wl"], irradiance_incom, irradiance_below


if __name__ == "__main__":

    # Object ProcessImage
    pim = ProcessImage()
                        #
    b_start = np.array([200, # 0 - 10 cm
                        350, # 10 - 20 cm
                        350, # 20 - 30 cm
                        350,  # 30 - 40 cm
                        250,  # 40 - 50 cm
                        110,  # 50 - 60 cm
                        100,  # 60 - 70 cm
                        60,  # 70 - 80 cm
                        60])  # 80 - 100 cm


    a_start = np.array([0.01, # 0 - 10 cm
                        0.01, # 10 - 20 cm
                        0.01, # 20 - 30 cm
                        0.01,  # 30 - 40 cm
                        0.01,  # 40 - 50 cm
                        0.01,  # 50 - 60 cm
                        0.01,  # 60 - 70 cm
                        0.01,  # 70 - 80 cm
                        0.01])  # 80 - 100 cm

    pf = np.array([OTHG(0.85),  # 0 - 10 cm
                   OTHG(0.99),  # 10 - 20 cm
                   OTHG(0.99),  # 20 - 30 cm
                   OTHG(0.99),   # 30 - 40 cm
                   OTHG(0.99),   # 40 - 50 cm
                   OTHG(0.99),   # 50 - 60 cm
                   OTHG(0.99),   # 60 - 70 cm
                   OTHG(0.99),  # 70 - 80 cm
                   OTHG(0.90)])  # 80 - 100 cm

    measurements = RadClass(data_path="../baiedeschaleurs-03232022.h5", station="station_1", freeboard=20)
    #
    # Stations 1 et 2
    # thickness 80, 72
    # draft 20, 18

    minimization(init=(a_start, b_start, pf), matlab_engine=None, radclass=measurements, trios=None)