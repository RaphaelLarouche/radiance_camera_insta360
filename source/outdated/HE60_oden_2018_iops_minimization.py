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

from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer


# Function and classes
def minimization(b_init, matlab_engine, measured_radiance_profile, trios):
    """

    :return:
    """
    # Radiance measurements
    zen_data, azi_data, oden_data = measured_radiance_profile

    b_ssl, b_dl, b_il = b_init
    print(b_init)

    new_mode = SeaIceSimulation(run_title='HE60DORT', root_name='HE60DORT', mode='HE60DORT', windspd=15.0)
    new_mode.set_z_grid(z_max=3.00, wavelength_list=[484, 544, 602])
    new_mode.add_layer(z1=0.0, z2=0.20, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=b_ssl, dpf='dpf_OTHG_0_98.txt')
    new_mode.add_layer(z1=0.20, z2=0.80, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=b_dl, dpf='dpf_OTHG_0_98.txt')
    new_mode.add_layer(z1=0.80, z2=2.00, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=b_il, dpf='dpf_OTHG_0_98.txt')
    new_mode.add_layer(z1=2.0, z2=3.01, abs={'484': 1.36e-2, '544': 5.11e-2, '602': 2.224e-1}, scat=0.1, dpf='dpf_OTHG_0_98.txt')
    new_mode.run_simulation(printoutput=False)
    new_mode.parse_results()
    analyzer = DataViewer(root_name="HE60DORT")



    depth_keys_order = ["20 cm (in water)",
                        "40 cm",
                        "60 cm",
                        "80 cm",
                        "100 cm",
                        "120 cm",
                        "140 cm",
                        "160 cm"]

    depth_list = [0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.41, 1.60]
    wavelengths = [600, 540, 480]

    mre_profile = np.empty((len(depth_keys_order), 3))

    for i, d_keys in enumerate(depth_keys_order):
        current_rad_map = oden_data[d_keys]

        # Azimuthal average Measurements
        az_average = r.azimuthal_average(current_rad_map)[19:159, :]
        x_red, zenith_red = analyzer.get_zenith_radiance_profile_at_depth(depth_list[i], wavelengths[0])
        x_green, zenith_green = analyzer.get_zenith_radiance_profile_at_depth(depth_list[i], wavelengths[1])
        x_blue, zenith_blue = analyzer.get_zenith_radiance_profile_at_depth(depth_list[i], wavelengths[2])

        az_average_he60_interp = np.stack((zenith_red, zenith_green, zenith_blue), axis=1)

        az_average_he60_interp = az_average_he60_interp[19:159, :]
        # Mean relative error (MUPD)
        ref_values = 0.5 * (az_average_he60_interp + az_average)
        rel_err = np.abs(az_average - az_average_he60_interp) / ref_values

        mre_profile[i, :] = np.nanmean(rel_err[19:159])

    mupd = mre_profile.mean(axis=0) * 100
    funval = np.mean(mupd)
    print(funval)
    return funval


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

    b_start = np.array([3051.62098427,  203.87618757,   69.03649511])
    # Table of records!
    # 1.  with array([3051.62098427,  203.87618757,   69.03649511])
    od = pim.open_radiance_data(path="../oden-08312018.h5")
    tr = None

    eng=None
    res = minimize(minimization, b_start, method="Nelder-Mead", args=(eng, od, tr), options={'maxfev': 100, 'disp': True})
    print(res)

    plt.show()

