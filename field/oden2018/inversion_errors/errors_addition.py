# -*- coding : utf-8 -*-

# Module importation
import os
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
from scipy.integrate import simps
from scipy.interpolate import interp1d

# Other modules
import noise_simulation as ns
from source.radiance import RadClass
from source.radiance import irradiance
from field.oden2018.oden_dort_vs_hl import load_zenith_radiance


# Classes and functions
def rmse_calculation(noisy_rad, original_rad):
    return np.sqrt(np.nanmean(np.square((noisy_rad - original_rad) / original_rad))) * 100


def set_Eudos_at_depth(pdf, depth, wavelength, irradiance_data):
    """

    :param pdf:
    :param depth:
    :param wavelength:
    :return:
    """
    dlist = list(pdf["depths"])
    dlist = [round(num, 4) for num in dlist]
    #i_depth = dlist.index(round(depth, 4)) + 1  # 0 is above the interface,
    #i_depth = dlist.index(round(depth, 4))

    try:

        i_depth = list(dlist).index(round(depth, 4))
        if depth == 0.0:
            i_depth = 1

        pdf.loc[i_depth, f'Eu_{wavelength:.1f}'] = irradiance_data[0]
        pdf.loc[i_depth, f'Ed_{wavelength:.1f}'] = irradiance_data[1]
        pdf.loc[i_depth, f'Eo_{wavelength:.1f}'] = irradiance_data[2]
        pdf.loc[i_depth, f'Eou_{wavelength:.1f}'] = irradiance_data[3]
        pdf.loc[i_depth, f'Eod_{wavelength:.1f}'] = irradiance_data[4]

    except ValueError:
        if depth == -1.:  # Depth -1 is the incoming radiation, just above the interface
            i_depth = 0

            pdf.loc[i_depth, f'Eu_{wavelength:.1f}'] = irradiance_data[0]
            pdf.loc[i_depth, f'Ed_{wavelength:.1f}'] = irradiance_data[1]
            pdf.loc[i_depth, f'Eo_{wavelength:.1f}'] = irradiance_data[2]
            pdf.loc[i_depth, f'Eou_{wavelength:.1f}'] = irradiance_data[3]
            pdf.loc[i_depth, f'Eod_{wavelength:.1f}'] = irradiance_data[4]
        else:
            print(f"Warning: Could not find resquested depth ({depth}) in: get_zenith_radiance_profile_at_depth")


    return pdf


def irradiance_calculation(zeni, azi, radm, zenimin, zenimax, planar=True):
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
    zeni_rad = zeni * np.pi / 180
    azi_rad = azi * np.pi / 180

    # Integrand
    if planar:
        integrand = radm[mask] * np.absolute(np.cos(zeni_rad[mask])) * np.sin(zeni_rad[mask])
    else:
        integrand = radm[mask] * np.sin(zeni_rad[mask])

    # Azimuthal integration
    azimuth_inte = simps(integrand.reshape((-1, azi_rad.shape[1])), azi_rad[mask].reshape((-1, azi_rad.shape[1])), axis=1)

    # Zenith integration
    e = simps(azimuth_inte, zeni_rad[mask].reshape((-1, azi_rad.shape[1]))[:, 0], axis=0)

    return e


def calculate_irradiance(zenith_meshgrid, azimuth_meshgrid, radiance):
    ed = irradiance_calculation(zenith_meshgrid, azimuth_meshgrid, radiance, 0, 90)
    edo = irradiance_calculation(zenith_meshgrid, azimuth_meshgrid, radiance, 0, 90, planar=False)
    eu = irradiance_calculation(zenith_meshgrid, azimuth_meshgrid, radiance, 90, 180)
    euo = irradiance_calculation(zenith_meshgrid, azimuth_meshgrid, radiance, 90, 180, planar=False)
    eo = irradiance_calculation(zenith_meshgrid, azimuth_meshgrid, radiance, 0, 180, planar=False)

    return eu, ed, eo, euo, edo


def get_zenith_radiance_profile_at_depth(zen_data, depth, wavelength, interpolate=True):
    """

    :param depth:
    :param wavelength:
    :param interpolate:
    :return:
    """
    zenith_radiance, depths, run_bands = zen_data
    i_wavelength = list(run_bands).index(wavelength)
    try:

        i_depth = list(depths).index(depth)
        if depth == 0.0:
            i_depth = 1

        zenith_radiance = zenith_radiance[i_depth, :, i_wavelength]

    except ValueError:
        if depth == -1.:  # Depth -1 is the incoming radiation, just above the interface
            i_depth = -1
            zenith_radiance = zenith_radiance[i_depth + 1, :, i_wavelength]
        else:
            print(f"Warning: Could not find resquested depth ({depth}) in: get_zenith_radiance_profile_at_depth")
    phi_angles = [0., 10., 20., 30., 40, 50., 60., 70., 80.,
                  90., 100., 110., 120., 130., 140., 150., 160., 170., 180.]  # Angles for which radiance is known

    if interpolate:  # cubic interpolation
        f = interp1d(phi_angles, zenith_radiance, kind='cubic')
        x_new_angles = np.arange(181)
        y_new_radiance = f(x_new_angles)
        return x_new_angles, y_new_radiance
    else:
        return phi_angles, zenith_radiance


# Copy existing files in some directory
def copy_files(path_source, path_destination):
    """
    Copy files from source to destination.

    :param path_source: absolute path to source (str)
    :param path_destination: absolute path to destination (str)
    :return:
    """
    if not os.path.exists(path_destination):
        os.makedirs(path_destination)
    for file in os.listdir(path_source):
        if file.endswith(".txt") or file.endswith(".pickle"):
            shutil.copy(os.path.join(path_source, file), os.path.join(path_destination, file))


# Function to loop over all radiance profiles and add noise
def add_noise_to_profile(path_source, path_destination):
    """
    Add noise over all radiance in zenith_profiles.text file.
    :param path_source:
    :return:
    """
    # Load radiance profiles
    rc = RadClass("../data/oden-08312018.h5")
    zr = load_zenith_radiance(path=path_source)
    zr_oden = load_zenith_radiance(path=r"../data/oden_fit")

    # Load csv file
    ori_df = pandas.read_csv(path_source + "/eudos_iops.csv")
    new_df = ori_df.copy()

    # Fourier errors information
    fr_top_ampl, fr_top_phase = ns.get_fourier_error(rc, zr_oden, np.arange(20, 100, 20).astype(float), show=False)
    fr_bot_ampl, fr_bot_phase = ns.get_fourier_error(rc, zr_oden, np.arange(100, 160, 20).astype(float), show=False)

    # Figure parameters
    fig1, ax1 = plt.subplots(1, 2, figsize=(6.6929, 6.6929 * 3 / 4), sharey=True)
    ax1[0].set_yscale("log")

    dloop = zr[1] # depths in meters
    CMAR = matplotlib.cm.get_cmap("Reds", len(dloop))
    wave = [480.0, 540.0, 600.0]
    all_rsme = []
    for i, d in enumerate(dloop):
        for nw, w in enumerate(wave):

            theta, original_rad = get_zenith_radiance_profile_at_depth(zr, d, wavelength=w, interpolate=True)

            # Seed number
            #seed_n = np.random.randint(0, 100000)  # cannot be randomized
            seed_n = 3 * i + nw

            # if depth smaller than 100
            #if d <= 1.0:
                # Add noise to radiance
            #    noisy_rad = ns.get_noisy_radiance_curve(zr, fr_top_ampl, fr_top_phase, seed=seed_n, wave=w, depth=d * 100, show=False)
            #else:
                # Add noise to radiance
            noisy_rad = ns.get_noisy_radiance_curve(zr, fr_bot_ampl, fr_bot_phase, seed=seed_n, wave=w, depth=d * 100, show=False)
            #noisy_rad = ns.get_noisy_radiance_curve(zr, fr_top_ampl, fr_top_phase, seed=seed_n, wave=w, depth=d * 100, show=False)

            # Calculate irradiance
            azi, zeni = np.meshgrid(np.arange(0, 361, 1), theta)
            rad_distribution = np.tile(noisy_rad.reshape(-1, 1), (1, 361))  # Propagation of radiance

            all_irrad = calculate_irradiance(zeni, azi, rad_distribution)
            new_df = set_Eudos_at_depth(new_df, depth=d, wavelength=w, irradiance_data=all_irrad)  # Set irradiance at depth in new df (pandas)

            if d <= 2.0:
                all_rsme.append(rmse_calculation(noisy_rad, original_rad))

        ax1[0].plot(theta, original_rad, color=CMAR(i))
        ax1[1].plot(theta, noisy_rad, color=CMAR(i))

    print(np.array(all_rsme).mean())

    # Save csv
    #new_df.to_csv(path_destination + "/eudos_iops.csv", index=False)
    return ori_df, new_df


if __name__ == "__main__":

    original_path = "C:\\Users\\Raphaël Larouche\\PycharmProjects\\radiance_camera_insta360\\field\\oden2018\\data\\inversion_errors\\fit_errors_plus_noise_2\\original_files"
    destination_path = "C:\\Users\\Raphaël Larouche\\PycharmProjects\\radiance_camera_insta360\\field\\oden2018\\data\\inversion_errors\\fit_errors_plus_noise_2\\new_files"

    copy_files(original_path, destination_path)

    dat, dat_modif = add_noise_to_profile(original_path, destination_path)

    fig1, ax1 = plt.subplots(1, 1)

    ax1.plot(dat["Eu_480.0"], dat["depths"], label="480 nm original")
    ax1.plot(dat_modif["Eu_480.0"], dat_modif["depths"], label="480 nm noisy")

    ax1.plot(dat["Eu_600.0"], dat["depths"], label="600 nm original")
    ax1.plot(dat_modif["Eu_600.0"], dat_modif["depths"], label="600 nm noisy")

    ax1.plot(dat["Eu_540.0"], dat["depths"], label="540 nm original")
    ax1.plot(dat_modif["Eu_540.0"], dat_modif["depths"], label="540 nm noisy")

    ax1.invert_yaxis()
    ax1.set_xscale("log")

    ax1.legend(loc="best")


