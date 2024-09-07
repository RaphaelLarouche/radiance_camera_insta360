"""
Script to analyze data from Fluorolog
"""

# Module importation
import os
import time
import glob
import h5py
import pandas
import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from source.geometric_rolloff import MatlabGeometricMengine
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
            sr.append(float(tlines[1]))
    return np.array(fr), np.array(sr)


def process_all_datfiles(wl, dirpath, show=True):
    """

    :param dirpath:
    :return:
    """
    if show:
        fall, axall = plt.subplots(1, 1)
    sig_avg = np.zeros(wl.shape[0])
    sig_std = sig_avg.copy()

    for n, wl in enumerate(wl):

        d = open_datfiles(dirpath + f"/tabfiles/R_{str(int(wl))}.dat")
        sig_avg[n] = d[1].mean()
        sig_std[n] = d[1].std()

        if show:
            axall.plot(d[0], d[1])

    return sig_avg, sig_std


def stack_roidata_mask(imstack, mask, exptime=False):
    """

    :param imstack:
    :param nbpixel:
    :param centroid:
    :param exptime:
    :return:
    """

    # Instance processimage
    pim = processing.ProcessImage()

    data = np.empty((imstack.shape[2], 3), dtype=[('dn_avg', np.float32), ('dn_std', np.float32)])
    data.fill(np.nan)

    # Centroid
    #y, x = centroid

    for n in range(imstack.shape[2]):

        # Downsampling
        im_dws = pim.dwnsampling(imstack[:, :, n], "RGGB")

        for i in range(im_dws.shape[2]):
            val = im_dws[:, :, i]
            if isinstance(exptime, bool):
                #roi = val[y - nbpixel // 2:y + nbpixel // 2 + 1, x - nbpixel // 2:x + nbpixel // 2 + 1]
                roi = val[mask[:, :, i]]
            else:
                #roi = val[y-nbpixel//2:y+nbpixel//2+1, x-nbpixel//2:x+nbpixel//2+1] / exptime[n]
                roi = val[mask[:, :, i]] / exptime[n]

            data["dn_avg"][n, i] = roi.mean()
            data["dn_std"][n, i] = roi.std()

    return data


def spheri2cart(zeni, azi):

    z = zeni * np.pi / 180
    az = azi * np.pi / 180

    X = np.sin(z) * np.cos(az)
    Y = np.sin(z) * np.sin(az)
    Z = np.cos(z)

    return X, Y, Z


def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])


def Ry(theta):
    return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])


def Rz(theta):
    return np.matrix([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])


def coordinates_rotation(zenith, azimuth, zeni_ro, azi_ro):

    X, Y, Z = spheri2cart(zenith, azimuth)
    XYZ = np.stack((X.ravel(), Y.ravel(), Z.ravel()))
    Rot = Ry(-zeni_ro * np.pi / 180) * Rz(-azi_ro * np.pi / 180)

    newXYZ = np.matmul(Rot, XYZ)
    nX = np.array(newXYZ[0, :].reshape(X.shape))
    nY = np.array(newXYZ[1, :].reshape(Y.shape))
    nZ = np.array(newXYZ[2, :].reshape(Z.shape))

    return np.arctan2(np.sqrt(nX ** 2 + nY ** 2), nZ) * 180/np.pi


if __name__ == "__main__":

    # Images
    process_im = processing.ProcessImage()
    generalpath = process_im.folder_choice(r"D:\data-i360-tests\calibrations\relative-spectral-response")

    # Geometric
    path_i360 = os.path.dirname(os.path.dirname(__file__))
    geocalib = h5py.File(path_i360 + "/geometric-calibration/calibrationfiles/geometric-calibration-air.h5")

    if ("lens1_c" or "lensclose" or "close") in generalpath:
        wlens = "close"
        geocalib = geocalib["/lens-close/20190104_192404/"]
    elif ("lens2_f" or "lensfar" or "far") in generalpath:
        wlens = "far"
        geocalib = geocalib["/lens-far/20190104_214037/"]
    else:
        wlens = None
        raise ValueError("Cannot find which lens...")

    # Geometric
    geo = {}
    for i in geocalib["fp"].keys():
        geo[i] = MatlabGeometricMengine(geocalib["fp"][i], geocalib["ierror"][i])

    amb_list = glob.glob(generalpath + "/AMB*.dng")
    stack_ambiance = process_im.imagestack(amb_list, wlens)[0]
    image_ambiance = stack_ambiance.mean(axis=2)

    # Fluorolog reference sensor
    wavel = np.arange(400, 710, 10)
    si_avg, si_std = process_all_datfiles(wavel, generalpath, show=True)
    si_rel_unc = si_std / si_avg

    # Stack all images
    beam_im_list = glob.glob(generalpath + "/IMG*.dng")
    im_list_sep = rsr_fct.chunck_imagelist(beam_im_list, 4)

    stack, exptime = rsr_fct.avg_image(im_list_sep, image_ambiance, wlens)
    _, centro = rsr_fct.find_centroid(stack[:, :, 15])
    y, x = centro

    # ROI
    # Squared ROI
    pixel_roi = 11  # 5 x 5
    stack_size = stack.shape
    m_square_rgb = np.zeros((stack_size[0] // 2, stack_size[1] // 2, 3)).astype(float)
    m_square_rgb[y - pixel_roi // 2:y + pixel_roi // 2 + 1, x - pixel_roi // 2:x + pixel_roi // 2 + 1, :] = 1.0
    m_square_rgb = m_square_rgb.astype(bool)

    # Angular ROI
    zen_red, zen_green, zen_blue = geo["red"].angular_coordinates()[1], geo["green"].angular_coordinates()[1], geo["blue"].angular_coordinates()[1]
    az_red, az_green, az_blue = geo["red"].angular_coordinates()[2], geo["green"].angular_coordinates()[2], geo["blue"].angular_coordinates()[2]

    nzen_r = coordinates_rotation(zen_red, az_red, zen_red[y, x], az_red[y, x])
    nzen_g = coordinates_rotation(zen_green, az_green, zen_green[y, x], az_green[y, x])
    nzen_b = coordinates_rotation(zen_blue, az_blue, zen_blue[y, x], az_blue[y, x])

    mask_ang = np.stack([nzen_r, nzen_b, nzen_b], axis=2)
    mask_ang = mask_ang < 1.0

    #data = rsr_fct.stack_roidata(stack, pixel_roi, np.array([873, 893]), exptime=exptime)
    #data = rsr_fct.stack_roidata(stack, pixel_roi, np.array([873, 893]), exptime=exptime)
    #data = rsr_fct.stack_roidata(stack, pixel_roi, np.array([int(geo["green"].center[1]), int(geo["green"].center[0])]), exptime=exptime)
    #data = rsr_fct.stack_roidata(stack, pixel_roi, centro, exptime=exptime)

    #data = stack_roidata_mask(stack, m_square_rgb, exptime=exptime)
    data = stack_roidata_mask(stack, mask_ang, exptime=exptime)

    data_rel_unc = data["dn_std"] / data["dn_avg"]

    # Spectral response
    sr = data["dn_avg"] / si_avg[:, None]
    sr_unc = data["dn_std"] / si_avg[:, None]
    sr_rel_unc = rsr_fct.relative_uncertainty(data_rel_unc, si_rel_unc[:, None])

    # Relative spectral response - normalization by maximum
    rsr = sr / sr.max(axis=0)
    sr_indx_max = np.argmax(sr, axis=0)
    rsr_unc = sr_unc / np.diagonal(sr[sr_indx_max[None, :]][0])
    rsr_rel_unc = rsr_fct.relative_uncertainty(sr_rel_unc, np.diagonal(sr_rel_unc[sr_indx_max[None, :]][0]))

    # Print stats
    rsr_fct.rsr_statistics(wavel, rsr)
    print("--Rel. uncertainties over 10% --")
    print("Average red: {0:.3f}\n"
          "Average green: {1:.3f}\n"
          "Average blue: {2:.3f}".format(rsr_rel_unc[rsr[:, 0] > 0.1, 0].mean() * 100,
                                        rsr_rel_unc[rsr[:, 1] > 0.1, 1].mean() * 100,
                                        rsr_rel_unc[rsr[:, 2] > 0.1, 2].mean() * 100))

    # Create table
    new_rsr, new_rsr_rel_unc = rsr_fct.round_signi(rsr, rsr_rel_unc * rsr)
    df_rsr = pandas.DataFrame(np.concatenate((wavel.reshape(-1, 1), new_rsr * 100, new_rsr_rel_unc * 100), axis=1),
                              columns=("Wl [nm]", "rsr red", "rsr green", "rsr blue", "unc red", "unc green", "unc blue"))
    print(df_rsr)

    # Saving data
    filename = "rsr_fluorolog_" + time.strftime("%Y%m%d", time.localtime(os.stat(beam_im_list[0])[-2])) + ".h5"
    pathname = "calibrationfiles/" + filename

    saved_answer = process_im.save_results()
    if saved_answer == "y":
        if wlens == "close":
            rsr_fct.create_hdf5_dataset(pathname, "lens-close", "rsr_peak_norm", rsr)
            rsr_fct.create_hdf5_dataset(pathname, "lens-close", "rsr_relative_unc", rsr_rel_unc)
            rsr_fct.create_hdf5_dataset(pathname, "lens-close", "wavelength", wavel)
        else:
            rsr_fct.create_hdf5_dataset(pathname, "lens-far", "rsr_peak_norm", rsr)
            rsr_fct.create_hdf5_dataset(pathname, "lens-far", "rsr_relative_unc", rsr_rel_unc)
            rsr_fct.create_hdf5_dataset(pathname, "lens-far", "wavelength", wavel)

    # Figure
    fig1, ax1 = plt.subplots(1, 1)

    ax1.errorbar(wavel, si_avg, yerr=si_std, capsize=3)

    ax1.set_xlabel("Wavelengths [nm]")
    ax1.set_ylabel("Beam intensity [a.u.]")

    # RSR
    fig2, ax2 = plt.subplots(1, 1)

    lstyle = ["-", "-.", ":"]
    m = ["o", "s", "d"]
    col = ['#d62728', '#2ca02c', '#1f77b4']
    lab = ["red band", "green band", "blue band"]

    for i in range(rsr.shape[1]):

        ax2.plot(wavel, rsr[:, i], marker=m[i], linestyle=lstyle[i], color=col[i], label=lab[i], markeredgecolor=col[i], markerfacecolor="none")
        ax2.fill_between(wavel, rsr[:, i]-rsr_unc[:, i], rsr[:, i]+rsr_unc[:, i], color="lightgrey", alpha=0.7)

    ax2.set_ylabel("Relative spectral response, $RSR_{i}$")
    ax2.set_xlabel("Wavelength [nm]")

    ax2.legend(loc="best")

    # Image of the beam
    fig3, ax3 = plt.subplots(1, 1)

    # Image
    imrgb = process_im.dwnsampling(stack[:, :, 15], "RGGB")

    ax3.imshow(imrgb[:, :, 1])
    ax3.imshow(mask_ang[:, :, 1], alpha=0.2)
    ax3.imshow(m_square_rgb[:, :, 1], alpha=0.5)
    ax3.plot(centro[1], centro[0], "r+")
    ax3.plot(geo["green"].center[0], geo["green"].center[1], "g.")

    plt.show()
