"""
Script to analyze data from Nanolog.
"""

# Module importation
import glob
import pandas
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


import source.processing as processing
import rsr as rsr_fct


# Classes and functions
def open_datfiles(path):
    """
    Function to quickly open data in .dat files.

    :param path:
    :return:
    """
    with open(path) as datfile:
        all_lines = datfile.readlines()
    datfile.close()
    fr, sr = [], []
    for i, l in enumerate(all_lines):
        if i != 0:
            tlines = l.strip().split("\t")
            fr.append(int(tlines[0]))
            sr.append(int(tlines[1]))
    return np.array(fr), np.array(sr)


if __name__ == "__main__":

    # Open beam-splitter data
    bs5050_NIR_r = pandas.read_csv("calibrationfiles/NPBS_NIR_5050_RawData_E1_R.csv")
    bs5050_NIR_t = pandas.read_csv("calibrationfiles/NPBS_NIR_5050_RawData_E1_T.csv")

    # Open reference data
    ref_data = open_datfiles("calibrationfiles/One 080322-OK.dat")

    waves = np.arange(400, 710, 10)  # wavelengths
    waves = np.insert(waves, 0, 610)
    waves = np.sort(waves)

    # Find the peaks in the signal
    peaks = scipy.signal.find_peaks(np.diff(ref_data[1]), height=1000, distance=140)
    peaks_negative = scipy.signal.find_peaks(-np.diff(ref_data[1]), height=1000, distance=140)

    all_peaks = np.append(peaks[0], peaks_negative[0])
    all_peaks = np.sort(all_peaks)
    all_peaks = np.insert(all_peaks, 0, 0)

    avg_step_size = int(np.round(np.diff(all_peaks[:10]).mean()))
    central_positions = np.arange(0, waves.shape[0], 1) * avg_step_size + avg_step_size//2

    # Average nanolog signal
    num_val = 11
    signal_avg = []
    signal_std = []
    for n, w in enumerate(waves):
        curr_data = ref_data[1][central_positions[n]-num_val:central_positions[n]+(num_val+1):1]
        signal_avg.append(curr_data.mean())
        signal_std.append(curr_data.std())

    signal_avg = np.array(signal_avg)
    signal_std = np.array(signal_std)

    # Images
    wlens = "close"
    process_im = processing.ProcessImage()
    generalpath = process_im.folder_choice(r"D:\data-i360-tests\calibrations\relative-spectral-response")

    # Dark
    amb_list = glob.glob(generalpath + "/AMB*.dng")
    stack_ambiance = process_im.imagestack(amb_list, wlens)[0]
    image_ambiance = stack_ambiance.mean(axis=2)

    # Stack all images
    beam_im_list = glob.glob(generalpath + "/IMG*.dng")
    im_list_sep = rsr_fct.chunck_imagelist(beam_im_list, 4)

    stack, exptime = rsr_fct.avg_image(im_list_sep, image_ambiance, wlens)

    # Find centroid
    _, centro = rsr_fct.find_centroid(stack[:, :, 15])

    # Data of DN around centro
    #data = rsr_fct.stack_roidata(stack, 2, centro, exptime=exptime)
    data = rsr_fct.stack_roidata(stack, 5, centro, exptime=exptime)

    # Normalization by light spectral composition
    idx_rmv = np.where(waves == 610)[0][0]
    waves_g = np.delete(waves, idx_rmv)
    signal_avg_g = np.delete(signal_avg, idx_rmv)
    signal_std_g = np.delete(signal_std, idx_rmv)

    # Figures
    fig1, ax1 = plt.subplots(2, 1, sharex=True)

    ax1[0].plot(np.diff(ref_data[1]))
    ax1[0].scatter(peaks[0], peaks[1]["peak_heights"], facecolor='none', s=20, color="r", label="Positive peaks")
    ax1[0].scatter(peaks_negative[0], -peaks_negative[1]["peak_heights"], facecolor='none', s=20, color="k", label="Negative peaks")

    ax1[1].plot(ref_data[0], ref_data[1])
    ax1[1].scatter(peaks[0], ref_data[1][peaks[0]], facecolor='none', s=20, color="r", label="Positive peaks")
    ax1[1].scatter(peaks_negative[0], ref_data[1][peaks_negative[0]], facecolor='none', s=20, color="k", label="Negative peaks")
    ax1[1].scatter(central_positions, ref_data[1][central_positions], s=20, color="y", label="Central positions")

    ax1[0].set_ylabel("Peaks")
    ax1[1].set_ylabel("Beam intensity [a.u.]")
    ax1[1].set_xlabel("Measurements number")

    ax1[0].legend(loc="best")
    ax1[1].legend(loc="best")

    # Figure 2
    fig2, ax2 = plt.subplots(1, 1)

    ax2.errorbar(waves, signal_avg, yerr=signal_std, capsize=3)

    ax2.set_xlabel("Wavelengths [nm]")
    ax2.set_ylabel("Beam intensity [a.u.]")

    # Figure 3
    fig3, ax3 = plt.subplots(1, 1)

    ax3.plot(bs5050_NIR_r["Wavelength (nm)"], 0.5 * (bs5050_NIR_r["P-Polarized"] + bs5050_NIR_r["S-Polarized"]), label="Reflectance")
    ax3.plot(bs5050_NIR_t["Wavelength (nm)"], 0.5 * (bs5050_NIR_t["P-Polarized"] + bs5050_NIR_t["S-Polarized"]), label="Transmittance")

    ax3.legend(loc="best")

    fig1.tight_layout()
    fig2.tight_layout()

    plt.show()
