"""
Photon transfer curves.
"""

# Module importation
import os
import h5py
import numpy as np
import scipy.stats
import natsort
import scipy.optimize
import matplotlib.pyplot as plt

from characterization_darkframe import create_dict
from source.processing import ProcessImage, FigureFunctions
from source.geometric_rolloff import MatlabGeometricMengine


def format_geometric_calibration(calibration):
    """
    Function to format correctly all the calibration bands (red, green, blue) object (MatlabGeometricMengine)
    inside a dictionary.
    :return:
    """
    gvar = {}
    for a in calibration["fp"].keys():
        gvar[a] = MatlabGeometricMengine(calibration["fp"][a], calibration["ierror"][a])
    return gvar


def fpn_fit(counts, fpn):
    """

    :param counts:
    :param fpn:
    :return:
    """
    return fpn * counts


def total_noise_eq(counts, rn, fpn, k):
    """

    :param counts:
    :param rn:
    :param fpn:
    :param k:
    :return:
    """
    return (rn ** 2 + (counts/k) + (fpn * counts)**2) ** (0.5)


def signal_for_SNR(snr, noise_parameters):
    """
    Function that return the real (and positive) roots that gives a specific SNR given the total noise equation.

    :param snr: wanted SNR - float
    :param noise_parameters: tuple of (reading noise, FPN, gain) - (floats, floats, floats)
    :return: positive and real root
    """

    p = np.array([noise_parameters[1] ** 2 - (1 / snr ** 2), 1 / noise_parameters[2], noise_parameters[0] ** 2])
    r = np.roots(p)
    r = r[np.isreal(r)]
    return r[r >= 0]


if __name__ == "__main__":

    # Instance figure functions
    ff = FigureFunctions()

    # Instance of of class ProcessImage
    processimage = ProcessImage()

    # if windows:
    volume_path = processimage.folder_choice()
    #filepath_exp = volume_path + "data-i360/calibrations/linearity/integration-time/"
    filepath_exp = volume_path + "data-i360-tests/calibrations/linearity/integration-time/09102020/"
    #filepath_dark_1sec = volume_path + "data-i360/calibrations/darkframe/integration-time/1_1s/"

    # Geometric mask
    #mask_zenith = 5.5
    mask_zenith = 7.0

    geo_air = h5py.File("../geometric-calibration/calibrationfiles/geometric-calibration-air.h5")
    geo_air_close = geo_air["lens-close"]["20190104_192404"]
    geo = format_geometric_calibration(geo_air_close)
    band_corr = {0: "red", 1: "green", 2: "blue"}

    # Imlist (MYBOOK)
    #imlist_light = processimage.imageslist(filepath_exp + "lensclose")[4:-6:2]
    #imlist_dark = processimage.imageslist_dark(filepath_exp + "lensclose", prefix="DARK") #+ [processimage.imageslist(filepath_dark_1sec)[0]]

    imlist_light = processimage.imageslist(filepath_exp + "lensclose")[:20][::2]
    imlist_dark = processimage.imageslist_dark(filepath_exp + "lensclose", prefix="AMB")[:20][::4]

    # Dark
    imstack_dark, exp_bl, iso_bl, _ = processimage.imagestack(imlist_dark, "close")
    # Light
    imstack_light, exposure_time, iso_gain, black_level = processimage.imagestack(imlist_light, "close")

    im1_average = np.zeros((exp_bl.shape[0], 3))
    im1_deviation = im1_average.copy()
    im2_average = im1_average.copy()
    im2_deviation = im1_average.copy()

    shotread = im1_average.copy()

    for k in range(imstack_light.shape[2] // 2):

        # Bias removal
        idx_im1 = int(2 * k)
        idx_im2 = idx_im1 + 1

        im_1_dark_rm = imstack_light[:, :, idx_im1] - imstack_dark[:, :, k]
        im_2_dark_rm = imstack_light[:, :, idx_im2] - imstack_dark[:, :, k]

        # Pair difference
        im_diff = im_1_dark_rm - im_2_dark_rm
        #im_diff = imstack_light[:, :, idx_im1] - imstack_light[:, :, idx_im2]

        # Down-sampling
        im_1_dws = processimage.dwnsampling(im_1_dark_rm, "RGGB")
        im_2_dws = processimage.dwnsampling(im_2_dark_rm, "RGGB")
        im_dff_dws = processimage.dwnsampling(im_diff, "RGGB")

        for b in range(im_1_dws.shape[2]):

            # Geometric calibration
            _, zen, _ = geo[band_corr[b]].angular_coordinates()
            mask = zen <= mask_zenith

            # Inside mask
            im1_average[k, b] = im_1_dws[mask, b].mean()
            im1_deviation[k, b] = im_1_dws[mask, b].std()

            im2_average[k, b] = im_2_dws[mask, b].mean()
            im2_deviation[k, b] = im_2_dws[mask, b].std()

            shotread[k, b] = im_dff_dws[mask, b].std() / np.sqrt(2)

    # Apparent read-noise analysis
    filepath_dark = volume_path + "data-i360/calibrations/darkframe/integration-time"
    dict_images_dark_ex = create_dict(filepath_dark)

    num_pixel_xy = 10

    read_noise_mean_exp = np.zeros((len(dict_images_dark_ex.keys()), 3))
    read_noise_std_exp = read_noise_mean_exp.copy()

    exposures_dct = {"0_2s": 2.0, "1_1000s": 1/1000, "1_15s": 1/15, "1_1s": 1.0, "1_240s":1/240, "1_4000s": 1/4000,
                     "1_5s": 1/5, "1_60s": 1/60}
    df_exposures = np.array([])

    for i, ke in enumerate(natsort.natsorted(dict_images_dark_ex.keys())[::-1]):

        print(ke)
        df_exposures = np.append(df_exposures, exposures_dct[ke])

        # Image list
        df, exp_df, iso_df, _ = processimage.imagestack(dict_images_dark_ex[ke][:-1], "close")

        # Diff pairs
        df_p1 = df[:, :, 0::2]
        df_p2 = df[:, :, 1::2]
        df_diff = df_p1 - df_p2

        # Pre-allocation
        read_noise_eachpair = np.zeros((df_diff.shape[2], 3))

        for fr in range(df_diff.shape[2]):

            # Downsampling
            df_diff_dws = processimage.dwnsampling(df_diff[:, :, fr], "RGGB")

            for ba in range(df_diff_dws.shape[2]):

                c = geo[band_corr[ba]].center
                read_noise_eachpair[fr, ba] = df_diff_dws[int(c[1]) - num_pixel_xy:
                                                          int(c[1]) + num_pixel_xy + 1,
                                                          int(c[0]) - num_pixel_xy:
                                                          int(c[0]) + num_pixel_xy + 1, ba].std() / np.sqrt(2)

        read_noise_mean_exp[i, :] = read_noise_eachpair.mean(axis=0)
        read_noise_std_exp[i, :] = read_noise_eachpair.std(axis=0)

    # Average signal and noises
    avg_counts = np.mean(np.stack((im1_average, im2_average), axis=2), axis=2)
    noise = np.mean(np.stack((im1_deviation, im2_deviation), axis=2), axis=2)

    fpn = np.sqrt(noise ** 2 - shotread ** 2)
    read_noise = read_noise_mean_exp.mean(axis=0)
    read_noise_array = np.tile(read_noise, (shotread.shape[0], 1))
    shot = np.sqrt(shotread ** 2 - read_noise_array ** 2)

    # Fit
    # FPN
    fpn_fit_r = scipy.optimize.curve_fit(fpn_fit, avg_counts[:, 0], fpn[:, 0])
    fpn_fit_g = scipy.optimize.curve_fit(fpn_fit, avg_counts[:, 1], fpn[:, 1])
    fpn_fit_b = scipy.optimize.curve_fit(fpn_fit, avg_counts[:, 2], fpn[:, 2])
    s = np.logspace(1, 4, 100)

    # Fit conversion gain knowning fpn and reading noises
    gain_popt_r, _ = scipy.optimize.curve_fit(lambda x, a: total_noise_eq(x, read_noise[0], fpn_fit_r[0], a), avg_counts[:, 0], noise[:, 0])
    gain_popt_g, _ = scipy.optimize.curve_fit(lambda x, a: total_noise_eq(x, read_noise[1], fpn_fit_g[0], a), avg_counts[:, 1], noise[:, 1])
    gain_popt_b, _ = scipy.optimize.curve_fit(lambda x, a: total_noise_eq(x, read_noise[2], fpn_fit_b[0], a), avg_counts[:, 2], noise[:, 2])

    # Fit every noise contributions
    #popt_fit_all_r, _ = scipy.optimize.curve_fit(total_noise_eq, avg_counts[:, 0], noise[:, 0], p0=np.array([read_noise[0], fpn_fit_r[0][0], 0.1]))
    #popt_fit_all_g, _ = scipy.optimize.curve_fit(total_noise_eq, avg_counts[:, 1], noise[:, 1], p0=np.array([read_noise[1], fpn_fit_g[0][0], 0.2]))
    #popt_fit_all_b, _ = scipy.optimize.curve_fit(total_noise_eq, avg_counts[:, 2], noise[:, 2], p0=np.array([read_noise[2], fpn_fit_b[0][0], 0.1]))

    popt_fit_all_r = np.array([read_noise[0], fpn_fit_r[0][0], gain_popt_r[0]])
    popt_fit_all_g = np.array([read_noise[1], fpn_fit_g[0][0], gain_popt_g[0]])
    popt_fit_all_b = np.array([read_noise[2], fpn_fit_b[0][0], gain_popt_b[0]])

    # Print fit results
    str_print = "Gain: {0:.3f} e/ADU, reading noise: {1:.3f} ADU, FPN: {2:.3f} %"

    print("red band - " + str_print.format(popt_fit_all_r[2], popt_fit_all_r[0], popt_fit_all_r[1] * 100))
    print("green band - " +str_print.format(popt_fit_all_g[2], popt_fit_all_g[0], popt_fit_all_g[1] * 100))
    print("blue band - " +str_print.format(popt_fit_all_b[2], popt_fit_all_b[0], popt_fit_all_b[1] * 100))

    # NER - noise equivalent radiance
    #ptf = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/calibrations/absolute-spectral-radiance/calibrationfiles/absolute_radiance.h5"
    #tag = "lens-close/20200909/cal-coefficients"
    ptf = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/calibrations/absolute-spectral-radiance/calibrationfiles/absolute_radiance_fluorolog.h5"
    tag = "lens-close/20200908/cal-coefficients"
    with h5py.File(ptf) as hfrel:
        cal = hfrel[tag][:]

    t_int = 1.0  # secs
    iso = iso_gain[0] * 0.01

    NER = np.array([(signal_for_SNR(1, np.array([read_noise[0], fpn_fit_r[0][0], gain_popt_r[0]])) / (t_int * iso)) * cal[0],
                    (signal_for_SNR(1, popt_fit_all_g) / (t_int * iso)) * cal[1],
                    (signal_for_SNR(1, popt_fit_all_b) / (t_int * iso)) * cal[2]])

    # Figures
    # Fig 1 - total noise vs. count
    fig1, ax1 = plt.subplots(1, 3, sharey=True, sharex=True)

    # Total noise
    ax1[0].plot(avg_counts[:, 0], noise[:, 0], color="k", linestyle="none", marker=".", label="Total")
    ax1[1].plot(avg_counts[:, 1], noise[:, 1], color="k", linestyle="none", marker=".", label="Total")
    ax1[2].plot(avg_counts[:, 2], noise[:, 2], color="k", linestyle="none", marker=".", label="Total")

    # Total noise fit
    #ax1[0].plot(s, total_noise_eq(s, read_noise[0], fpn_fit_r[0], gain_popt_r), color="k")
    #ax1[1].plot(s, total_noise_eq(s, read_noise[1], fpn_fit_g[0], gain_popt_g), color="k")
    #ax1[2].plot(s, total_noise_eq(s, read_noise[2], fpn_fit_b[0], gain_popt_b), color="k")
    ax1[0].plot(s, total_noise_eq(s, *popt_fit_all_r), color="k")
    ax1[1].plot(s, total_noise_eq(s, *popt_fit_all_g), color="k")
    ax1[2].plot(s, total_noise_eq(s, *popt_fit_all_b), color="k")

    # Fix pattern noise
    ax1[0].plot(avg_counts[:, 0], fpn[:, 0], marker=".", color="y", linestyle="none", label="Fixed pattern")
    ax1[1].plot(avg_counts[:, 1], fpn[:, 1], marker=".", color="y", linestyle="none", label="Fixed pattern")
    ax1[2].plot(avg_counts[:, 2], fpn[:, 2], marker=".", color="y", linestyle="none", label="Fixed pattern")

    #ax1[0].plot(s, fpn_fit_r[0] * s, color="y")
    #ax1[1].plot(s, fpn_fit_g[0] * s, color="y")
    #ax1[2].plot(s, fpn_fit_b[0] * s, color="y")
    ax1[0].plot(s, popt_fit_all_r[1] * s, color="y")
    ax1[1].plot(s, popt_fit_all_g[1] * s, color="y")
    ax1[2].plot(s, popt_fit_all_b[1] * s, color="y")

    ax1[0].plot(avg_counts[:, 0], shot[:, 0], color="b",marker=".", linestyle="none", label="Shot noise")
    ax1[1].plot(avg_counts[:, 1], shot[:, 1], color="b", marker=".", linestyle="none", label="Shot noise")
    ax1[2].plot(avg_counts[:, 2], shot[:, 2], color="b", marker=".", linestyle="none", label="Shot noise")

    ax1[0].plot(s, (s / popt_fit_all_r[2]) ** 0.5, color="b")
    ax1[1].plot(s, (s / popt_fit_all_g[2]) ** 0.5, color="b")
    ax1[2].plot(s, (s / popt_fit_all_b[2]) ** 0.5, color="b")

    ax1[0].set_yscale("log")
    ax1[0].set_xscale("log")
    ax1[1].set_xscale("log")
    ax1[2].set_xscale("log")

    ax1[0].set_ylabel("Noise level [ADU]")
    ax1[0].set_xlabel("Counts [ADU]")
    ax1[1].set_xlabel("Counts [ADU]")
    ax1[2].set_xlabel("Counts [ADU]")

    ax1[0].legend(loc="best", frameon=False)
    ax1[1].legend(loc="best", frameon=False)
    ax1[2].legend(loc="best", frameon=False)

    # Fig2 - total noise vs. counts - all curves same graph
    fig2, ax2 = plt.subplots(1, 1)

    ax2.plot(avg_counts[:, 0], noise[:, 0], marker=".", label="Red")
    ax2.plot(avg_counts[:, 1], noise[:, 1], marker=".", label="Green")
    ax2.plot(avg_counts[:, 2], noise[:, 2], marker=".", label="Blue")

    ax2.set_xscale("log")
    ax2.set_ylabel("Noise level [ADU]")
    ax2.set_xlabel("Counts [ADU]")
    ax2.legend(loc="best", frameon=False)

    # Fig3 - SNR vs. counts - all curves same graph
    fig3, ax3 = plt.subplots(1, 1)

    ax3.plot(avg_counts[:, 0], avg_counts[:, 0]/noise[:, 0], marker=".", label="Red")
    ax3.plot(avg_counts[:, 1], avg_counts[:, 1]/noise[:, 1], marker=".", label="Green")
    ax3.plot(avg_counts[:, 2], avg_counts[:, 2]/noise[:, 2], marker=".", label="Blue")

    ax3.set_xscale("log")
    ax3.set_ylabel("SNR")
    ax3.set_xlabel("Counts [ADU]")
    ax3.legend(loc="best", frameon=False)

    # Figure 4 - apparent reading noise vs.
    fig4, ax4 = plt.subplots()

    ax4.plot(df_exposures, read_noise_mean_exp[:, 0], label="Red")
    ax4.plot(df_exposures, read_noise_mean_exp[:, 1], label="Green")
    ax4.plot(df_exposures, read_noise_mean_exp[:, 2], label="Blue")

    ax4.set_xscale("log")
    ax4.set_xlabel("Exposure time [s]")
    ax4.set_ylabel("Reading noise [ADU]")

    ax4.legend(loc="best", frameon=False)

    plt.show()