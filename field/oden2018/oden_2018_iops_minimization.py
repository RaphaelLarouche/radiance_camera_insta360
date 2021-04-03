# -*- coding: utf-8 -*-
"""

"""

# Module importation
import os
import h5py
import pandas
import matplotlib
import numpy as np
import matlab.engine
from scipy.optimize import minimize
from scipy import integrate
import matplotlib.pyplot as plt

# Other module
import source.radiance as r
from source.processing import ProcessImage
from source.geometric_rolloff import OpenMatlabFiles


# Function and classes
def minimization(b_init, matlab_engine, measured_radiance_profile, trios):
    """

    :return:
    """

    # TRIOS
    time_stamp, radiometer_wl, irr_incom, irr_below = trios

    eff_wl = np.array([602, 544, 484])

    # Radiance measurements
    zen_data, azi_data, oden_data = measured_radiance_profile

    # DORT simul
    nstreams = 30
    number_layer = 5.0
    layer_thickness = matlab.double([100, 0.1, 0.7, 1.2, 10.0])  # in meters
    depth = matlab.double(list(np.arange(100, 102.1, 0.1)))

    # IOPs
    b_constr = np.array([0, 0, 0, 0, 0.1])
    b_constr[1:-1] = b_init
    #print(b_constr)
    b = matlab.double(b_constr.tolist())  # Scattering coefficient matlab format

    g_all = 0.98  # Assymetry parameter
    phase_type = matlab.double([1, 1, 1, 1, 1])
    g = eng.cell(1, 5)
    g[0], g[4] = matlab.double([0.]), matlab.double([0.9])
    g[1], g[2], g[3] = matlab.double([g_all]), matlab.double([g_all]), matlab.double([g_all])

    # Refractive index
    n = matlab.double([1.00, 1.00, 1.30, 1.30, 1.33])

    # Simulations
    rad_dort_band = {}
    for ii in range(3):

        arg_wl_dort = np.where(eff_wl[ii] == radiometer_wl)
        rad_inc = irr_incom.mean(axis=1)[arg_wl_dort]/np.pi

        # DORT simulation paramters (absorption coefficient change according to wavelength)
        if ii == 0:
           a = matlab.double([0, 0.12, 0.12, 0.12, 0.12])  # absorption coefficient red
        elif ii == 1:
           a = matlab.double([0, 0.0683, 0.0683, 0.0683, 0.0683])  # absorption coefficient green
        else:
           a = matlab.double([0, 0.0430, 0.0430, 0.0430, 0.0430])  # absorption coefficient blue

        # Simulations
        matlab_engine.cd(os.path.dirname(__file__))
        dort_p, dort_r = matlab_engine.dort_simulation_oden(matlab.double([nstreams]), number_layer, layer_thickness, b, a, phase_type, g, n, depth, matlab.double(list(rad_inc)), nargout=2)

        # Results
        bounds = np.arange(0, 220, 20)
        rad_dist_dort, zen_mesh_dort, azi_mesh_dort = compute_results_dort(dort_p, dort_r, bounds)

        rad_dort_band[str(ii)] = rad_dist_dort

    cdict = dict(zip(["zero minus", "20 cm (in water)", "40 cm", "60 cm", "80 cm", "100 cm", "120 cm", "140 cm", "160 cm", "180 cm", "200 cm"], list(rad_dist_dort.keys())))

    depth_keys_order = ["20 cm (in water)",
                        "40 cm",
                        "60 cm",
                        "80 cm",
                        "100 cm",
                        "120 cm",
                        "140 cm",
                        "160 cm"]

    mre_profile = np.empty((len(depth_keys_order), 3))

    for i, d_keys in enumerate(depth_keys_order):
        current_rad_map = oden_data[d_keys]

        # Azimuthal average Measurements
        az_average = r.azimuthal_average(current_rad_map)

        # DORT simluation
        current_dort_rad_map = np.stack((rad_dort_band["0"][cdict[d_keys]], rad_dort_band["1"][cdict[d_keys]],
                                         rad_dort_band["2"][cdict[d_keys]]), axis=2)

        # Azimuthal average DORT
        az_average_dort = r.azimuthal_average(current_dort_rad_map)
        az_average_dort_interp = np.empty((zen_data.shape[0], 3))

        for band in range(current_rad_map.shape[2]):
            az_average_dort_interp[:, band] = np.interp(zen_data[:, 0], zen_mesh_dort[:, 0], az_average_dort[:, band])

        # Mean relative error (MUPD)
        ref_values = 0.5 * (az_average_dort_interp + az_average)
        rel_err = np.abs(az_average - az_average_dort_interp) / ref_values

        mre_profile[i, :] = np.nanmean(rel_err[19:89])

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

    b_start = np.array([2300, 300, 80])
    eng = matlab.engine.start_matlab()
    od = pim.open_radiance_data(path="data/oden-08312018.h5")
    tr = open_TriOS_data()

    res = minimize(minimization, b_start, method="Nelder-Mead", args=(eng, od, tr), options={'maxfev': 30, 'disp': True})
    print(res)

    plt.show()

