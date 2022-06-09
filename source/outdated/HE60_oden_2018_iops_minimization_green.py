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

from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.phasefunctions import *

from HE60PY.dataviewer import DataViewer


# Function and classes
def minimization(init, matlab_engine, measured_radiance_profile, trios):
    """

    :return:
    """
    # Radiance measurements
    a_init, b_init, pf_ice = init
    print(list(b_init))
    zen_data, azi_data, oden_data = measured_radiance_profile
    root_name = "Eo_fit_multilayer2"
    HE_simulation = SeaIceSimulation(run_title=root_name, root_name=root_name, mode='HE60DORT', wavelength_list=[544])
    HE_simulation.set_z_grid(z_max=3.0)
    for i, b in enumerate(b_init):
        top, bot = i * 0.20, (i + 1) * 0.20
        if i < 10:
            HE_simulation.add_layer(z1=top, z2=bot, abs={'484': 0.012, '544': a_init[i], '602': 0.12}, scat=b, dpf=pf_ice[i]) # bb arg is not relevent since we use a discretized phase function in a file indepêdnant of depth (g=0.98)
        elif i == 10:
            HE_simulation.add_layer(z1=2.0, z2=3.01, abs={'484': 1.36e-2, '544': a_init[i], '602': 2.224e-1}, scat=b,
                                    dpf='dpf_OTHG_0_90.txt')
    HE_simulation.run_simulation(printoutput=True)
    HE_simulation.parse_results()
    HE_analyze = DataViewer(root_name=root_name)



    depth_keys_order = ["zero minus",
                        "20 cm (in water)",
                        "40 cm",
                        "60 cm",
                        "80 cm",
                        "100 cm",
                        "120 cm",
                        "140 cm",
                        "160 cm",
                        "180 cm",
                        "200 cm"]
    # Open measurements
    zen_oden, azi_oden, rad_oden = open_radiance_data(path="../oden-08312018.h5")  # Path à changer

    band_name = ["r", "g", "b"] # For measurments

    # For HE60 results handling
    depth_list = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.41, 1.60, 1.80, 2.00]

    HE_eo = []
    HE_eu = []
    HE_ed = []
    Oden_eo = []
    Oden_eu = []
    Oden_ed = []


    # mre_profile = np.empty((len(depth_keys_order), 3))
    quad_loss = np.empty((len(depth_keys_order)))
    for i, d_keys in enumerate(depth_keys_order):
        current_rad_map = oden_data[d_keys]

        # Oden azimuthal average Measurements
        az_average = r.azimuthal_average(current_rad_map)[19:159, :]
        # Oden eudos
        ed_oden, eu_oden, eo_oden = create_irradiance_data(zen_oden, azi_oden, rad_oden, [d_keys], oden=True)
        ed_oden_r, eu_oden_r, eo_oden_r = ed_oden["r"], eu_oden["r"], eo_oden["r"]      # Red
        ed_oden_g, eu_oden_g, eo_oden_g = ed_oden["g"], eu_oden["g"], eo_oden["g"]        # Green
        ed_oden_b, eu_oden_b, eo_oden_b = ed_oden["b"], eu_oden["b"], eo_oden["b"]        # Blue

        # Getting HE60 radiances
        # x_red, zenith_red = HE_analyze.get_zenith_radiance_profile_at_depth(depth_list[i], wavelengths[0])
        # x_green, zenith_green = HE_analyze.get_zenith_radiance_profile_at_depth(depth_list[i], wavelengths[1])
        # x_blue, zenith_blue = HE_analyze.get_zenith_radiance_profile_at_depth(depth_list[i], wavelengths[2])
        # az_average_he60_interp = np.stack((zenith_red, zenith_green, zenith_blue), axis=1)
        # az_average_he60_interp = az_average_he60_interp[19:159, :]
        # Getting HE60 Eudos
        # eu_he_r, eu_he_r, eo_he_r = HE_analyze.get_Eudos_at_depth(depth_list[i], 600) # Red
        eu_he_g, ed_he_g, eo_he_g = HE_analyze.get_Eudos_at_depth(depth_list[i], 544) #green

        # Quad loss on Eo blue only
        rel_err_blue = 100*(eo_oden_g - eo_he_g)/eo_oden_g
        quad_loss[i] = rel_err_blue**2
        HE_eo.append(eo_he_g)
        HE_eu.append(eu_he_g)
        HE_ed.append(ed_he_g)
        Oden_eo.append(eo_oden_g)
        Oden_eu.append(eu_oden_g)
        Oden_ed.append(ed_oden_g)

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


def compute_results_dort(p, r, wdepth):
    """

    :param p:
    :param r:
    :param wdepth:
    :return:
    """
    polar = np.arccos(p["mu_interpol"]) * 180 / np.pi
    zenith = np.concatenate((polar[::-1], 180 - polar), axis=0)
    azimuth = np.array(p["phi_interpol"]) * 180 / np.pi

    az_mesh, zen_mesh = np.meshgrid(azimuth, zenith)

    # Pre-allocation
    rad_dist = {}

    for i, de in enumerate(p["depth"]):

        # Depth
        decm = round((float(np.array(de)) - 100) * 100)

        if decm in wdepth:

            radiance_d = np.array(r["I_ac_" + str(i+1)])
            radiance_d = radiance_d[::-1, :]
            radiance_d[45:, :] = radiance_d[45:, :][::-1, :]

            rad_dist[str(decm)] = radiance_d

    return rad_dist, zen_mesh, az_mesh

if __name__ == "__main__":

    # Object ProcessImage
    pim = ProcessImage()
                        #
    # b_start = np.array([1100, # 0-20 cm
    #                     350, # 20 - 40 cm
    #                     400, # 40 - 60 cm
    #                     400,  # 60 - 80 cm
    #                     300,  # 80 - 100 cm
    #                     125,  # 100 - 120 cm
    #                     125,  # 120 - 140 cm
    #                     80,  # 140 - 160 cm
    #                     60,  # 160 - 180 cm
    #                     60,  # 180 - 200 cm
    #                     0.65]) # 200 - 300 cm

    b_start = np.array([200, # 0-20 cm
                        350, # 20 - 40 cm
                        350, # 40 - 60 cm
                        350,  # 60 - 80 cm
                        250,  # 80 - 100 cm
                        110,  # 100 - 120 cm
                        100,  # 120 - 140 cm
                        60,  # 140 - 160 cm
                        60,  # 160 - 180 cm
                        120,  # 180 - 200 cm
                        0.25]) # 200 - 300 cm

    a_start = np.array([0.09, # 0-20 cm
                        0.09, # 20 - 40 cm
                       0.09, # 40 - 60 cm
                        0.09,  # 60 - 80 cm
                        0.09,  # 80 - 100 cm
                        0.09, # 100 - 120 cm
                        0.09,  # 120 - 140 cm
                        0.09,  # 140 - 160 cm
                        0.09,  # 160 - 180 cm
                        0.09, # 180 - 200 cm
                        0.05]) # 200 - 300 cm
    a_start[0:10] = 0.065

    pf = np.array([OTHG(0.85), # 0-20 cm
                   OTHG(0.99), # 20 - 40 cm
                   OTHG(0.99), # 40 - 60 cm
                   OTHG(0.99),  # 60 - 80 cm
                   OTHG(0.99),  # 80 - 100 cm
                   OTHG(0.99),  # 100 - 120 cm
                   OTHG(0.99),  # 120 - 140 cm
                   OTHG(0.99), # 140 - 160 cm
                   OTHG(0.99),  # 160 - 180 cm
                   OTHG(0.99),  # 180 - 200 cm
                   OTHG(0.90)]) # 200 - 300 cm
    # b_start = np.array([2000, 2000, 2000, 2000,
    #        7.48331506e+02, 4.12127634e+02, 3.27546370e+02, 3.12779500e+02,
    #        1.82968258e+02, 9.01500376e+01, 1.02889845e+00])
    # a_start = a_start*1.48



    # Table of records!
    # 1.  with array([3051.62098427,  203.87618757,   69.03649511])
    od = pim.open_radiance_data(path="../oden-08312018.h5")
    tr = None

    eng=None
    minimization(init=(a_start, b_start, pf), matlab_engine=None, measured_radiance_profile=od, trios=None)
    # res = minimize(minimization, b_start, method="Nelder-Mead", args=(eng, od, tr), options={'maxfev': 100, 'disp': True})
    # print(res)

