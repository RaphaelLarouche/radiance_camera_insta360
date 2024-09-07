# -*- coding: utf-8 -*-
"""

"""

# Module importation
import os
import time
import glob
import h5py
import pandas
import numpy as np
import deepdish
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from source.geometric_rolloff import MatlabGeometricMengine
import calibrations.calibrations_info
import source.processing as processing
import rsr as rsr_fct


# Classes and functions
def open_datfiles(path):
    """
    Function to quickly open data in .dat files.

    :param path: path to file
    :type path: str
    :return: tuple of first column and second column
    :rtype: ndarray, ndarray
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

    for n in range(imstack.shape[2]):

        print(f"Processing image number {n + 1} out of {imstack.shape[2]} ({100 * ((n + 1) / imstack.shape[2]):.1f} %)")
        # Downsampling
        t = time.time()
        im_dws = pim.dwnsampling_acc(imstack[:, :, n], pim.dws_pattern("GBRG"))
        #r, g, b = pim.dwnsampling(imstack[:, :, n], "GBRG", ave=False)
        #g = (g[0::2, :] + g[1::2, :]) / 2
        #rgb = r, g, b
        #rgb = pim.dwnsampling(imstack[:, :, n], "GBRG")
        print(time.time() - t)

        t2 = time.time()
        for i in range(im_dws.shape[2]):
        #for i in range(len(rgb)):
            val = im_dws[:, :, i]
            #val = rgb[i]

            if isinstance(exptime, bool):
                roi = val[mask[:, :, i]]
            elif isinstance(exptime, (np.ndarray, np.generic)):
                roi = val[mask[:, :, i]] / exptime[n]

            data["dn_avg"][n, i] = roi.mean()
            data["dn_std"][n, i] = roi.std()
        print(time.time() - t2)

    return data

def spheri2cart(zeni, azi):
    """

    :param zeni:
    :type zeni:
    :param azi:
    :type azi:
    :return:
    :rtype:
    """

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
    """

    :param zenith:
    :type zenith:
    :param azimuth:
    :type azimuth:
    :param zeni_ro:
    :type zeni_ro:
    :param azi_ro:
    :type azi_ro:
    :return:
    :rtype:
    """

    X, Y, Z = spheri2cart(zenith, azimuth)
    XYZ = np.stack((X.ravel(), Y.ravel(), Z.ravel()))
    Rot = Ry(-zeni_ro * np.pi / 180) * Rz(-azi_ro * np.pi / 180)

    newXYZ = np.matmul(Rot, XYZ)
    nX = np.array(newXYZ[0, :].reshape(X.shape))
    nY = np.array(newXYZ[1, :].reshape(Y.shape))
    nZ = np.array(newXYZ[2, :].reshape(Z.shape))

    return np.arctan2(np.sqrt(nX ** 2 + nY ** 2), nZ) * 180/np.pi


def avg_image_x3(imagelist, amb_image, which):
    """

    :param imagelist:
    :param which:
    :return:
    """
    pim = processing.ProcessImage()
    stack = np.empty((5984, 5984, len(imagelist)))
    exp = np.array([])

    for n, l in enumerate(imagelist):
        print(f"Process {os.path.basename(l)} nm")
        s = pim.imagestack(glob.glob(l + "/*.dng")[:1],  which, cam="x3")
        currim = s[0] - amb_image[:, :, None]
        stack[:, :, n] = np.clip(currim.mean(axis=2), 0, None)

    return stack, exp


def find_centroid(image):
    """
    Function to find spatial location centroid (x, y) of the light spot.

    :param image: 2D bayer image
    :type image: ndarray
    :return: tuple of the green image (ndarray) and centroid (ndarray)
    :rtype: tuple
    """
    pim = processing.ProcessImage()

    im_dws = pim.dwnsampling(image, "GBRG")

    _, regpro = pim.region_properties(im_dws[:, :, 1], 1000)
    centro = np.round(regpro[0].centroid).astype(int)

    return im_dws[:, :, 1], centro


def save_rsr_hdf5(filename, name_group, dataname, data):
    """

    :param filename:
    :type filename:
    :param name_group:
    :type name_group:
    :param dataname:
    :type dataname:
    :param data:
    :type data:
    :return:
    :rtype:
    """

    data_path = f"{name_group}/{dataname}"

    with h5py.File(filename, "a") as hf:
        if data_path in hf:
            d = hf[data_path]
            d[...] = data
        else:
            hf.create_dataset(data_path, data=data)


if __name__ == "__main__":

    # Images
    process_im = processing.ProcessImage()
    #generalpath = process_im.folder_choice("Users/RaphaelLarouche/MYBOOK")
    #generalpath = "/Volumes/MYBOOK/data-i360X3/calibrations/relative_spectral_response/2BW7X7/02222023/front"
    #generalpath = "/Volumes/MYBOOK/data-i360X3/calibrations/relative_spectral_response/2BW7X7/03132023/back"

    # Choosing lens
    cover = "nocover"
    sn = "2C9JCA"
    wlens = "back"
    date = "20230330"
    generalpath = f"/Volumes/MYBOOK/data-i360X3/calibrations/relative_spectral_response/{sn}/{cover}/{wlens}/{date}"

    # Geometric
    path_i360 = os.path.dirname(os.path.dirname(__file__))
    geocalib = h5py.File(f"../geometric-calibration/calibrationfiles/geometric-calibration-{sn}.h5")
    calib_id = calibrations.calibrations_info.geometric[f"{sn}"][f"{cover}"]["air"][f"{wlens}"]
    geocalib = geocalib[f"air/{cover}/{wlens}/{calib_id}"]

    # Geometric
    geo = {}
    for i in geocalib["fp"].keys():
        geo[i] = MatlabGeometricMengine(geocalib["fp"][i], geocalib["ierror"][i])

    amb_list = glob.glob(generalpath + "/data/amb/AMB*.dng")
    stack_ambiance = process_im.imagestack(amb_list, wlens, cam="x3")[0]
    image_ambiance = stack_ambiance.mean(axis=2)

    # Fluorolog reference sensor
    wavel = np.arange(400, 710, 10)
    si_avg, si_std = process_all_datfiles(wavel, generalpath, show=True)
    si_rel_unc = si_std / si_avg

    # Stack all images
    beam_im_list = glob.glob(generalpath + "/data/[0-9]*")

    # Find centroid
    #stack, exptime = avg_image_x3(beam_im_list, image_ambiance, wlens)
    stack_15 = process_im.imagestack(glob.glob(beam_im_list[15] + "/*.dng"), wlens, cam="x3")[0]
    stack_15 = stack_15 - image_ambiance[:, :, None]
    _, centro = find_centroid(np.clip(stack_15.mean(axis=2), 0, None))
    y, x = centro

    # ROI
    # Squared ROI
    pixel_roi = 11  # 5 x 5
    #stack_size = stack.shape
    #m_square_rgb = np.zeros((stack_size[0] // 2, stack_size[1] // 2, 3)).astype(float)
    #m_square_rgb[y - pixel_roi // 2:y + pixel_roi // 2 + 1, x - pixel_roi // 2:x + pixel_roi // 2 + 1, :] = 1.0
    #m_square_rgb = m_square_rgb.astype(bool)

    # Angular ROI
    zen_red, zen_green, zen_blue = geo["red"].angular_coordinates()[1], geo["green"].angular_coordinates()[1], geo["blue"].angular_coordinates()[1]
    az_red, az_green, az_blue = geo["red"].angular_coordinates()[2], geo["green"].angular_coordinates()[2], geo["blue"].angular_coordinates()[2]

    nzen_r = coordinates_rotation(zen_red, az_red, zen_red[y, x], az_red[y, x])
    nzen_g = coordinates_rotation(zen_green, az_green, zen_green[y, x], az_green[y, x])
    nzen_b = coordinates_rotation(zen_blue, az_blue, zen_blue[y, x], az_blue[y, x])

    mask_ang = np.stack([nzen_r, nzen_g, nzen_b], axis=2)
    #mask_ang = mask_ang < 1.0
    #mask_ang = mask_ang < 0.7
    mask_ang = mask_ang < 0.3

    #data = rsr_fct.stack_roidata(stack, pixel_roi, np.array([873, 893]), exptime=exptime)
    #data = rsr_fct.stack_roidata(stack, pixel_roi, np.array([873, 893]), exptime=exptime)
    #data = rsr_fct.stack_roidata(stack, pixel_roi, np.array([int(geo["green"].center[1]), int(geo["green"].center[0])]), exptime=exptime)
    #data = rsr_fct.stack_roidata(stack, pixel_roi, centro, exptime=exptime)

    #data = stack_roidata_mask(stack, m_square_rgb, exptime=exptime)
    #data = stack_roidata_mask(stack, mask_ang, exptime=False)

    # LOOP to extract data
    data = np.empty((len(beam_im_list), 3), dtype=[('dn_avg', np.float32), ('dn_std', np.float32)])
    data.fill(np.nan)

    for b, l in enumerate(beam_im_list):
        print(f"Process {os.path.basename(l)} nm")
        s = process_im.imagestack(glob.glob(l + "/*.dng"), wlens, cam="x3")
        currim = s[0] - image_ambiance[:, :, None]
        image = np.clip(currim.mean(axis=2), 0, None)

        # Downsampling
        t = time.time()
        im_dws = process_im.dwnsampling_acc(image, process_im.dws_pattern("GBRG"))
        print(time.time() - t)
        t2 = time.time()
        for i in range(im_dws.shape[2]):
            val = im_dws[:, :, i]
            roi = val[mask_ang[:, :, i]]
            data["dn_avg"][b, i] = roi.mean()
            data["dn_std"][b, i] = roi.std()
        print(time.time() - t2)

    # Relative spectral response calculations
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
    df_rsr = pandas.DataFrame(np.concatenate((wavel.reshape(-1, 1), new_rsr * 100, new_rsr_rel_unc * 100), axis=1), columns=("Wl [nm]", "rsr red", "rsr green", "rsr blue", "unc red", "unc green", "unc blue"))
    print(df_rsr)

    # Saving data
    filename = f"rsr-x3-{sn}.h5"
    #filename = "rsr_fluorolog_x3_" + time.strftime("%Y%m%d", time.localtime(os.stat(amb_list[0])[-2])) + ".h5"
    pathname = "calibrationfiles/" + filename
    group_name = f"{cover}/{wlens}/{date}"

    saved_answer = process_im.save_results()
    if saved_answer == "y":

        save_rsr_hdf5(pathname, group_name, "rsr [-]", rsr)
        save_rsr_hdf5(pathname, group_name, "uncertainties [%]", rsr_rel_unc * 100)
        save_rsr_hdf5(pathname, group_name, "wavelength [nm]", wavel)

        #if wlens == "front":
        #    rsr_fct.create_hdf5_dataset(pathname, "lens-front", "rsr_peak_norm", rsr)
        #    rsr_fct.create_hdf5_dataset(pathname, "lens-front", "rsr_relative_unc", rsr_rel_unc)
        #    rsr_fct.create_hdf5_dataset(pathname, "lens-front", "wavelength", wavel)
        #else:
        #    rsr_fct.create_hdf5_dataset(pathname, "lens-back", "rsr_peak_norm", rsr)
        #    rsr_fct.create_hdf5_dataset(pathname, "lens-back", "rsr_relative_unc", rsr_rel_unc)
        #    rsr_fct.create_hdf5_dataset(pathname, "lens-back", "wavelength", wavel)

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
    imrgb = stack_15 - image_ambiance[:, :, None]
    imrgb = process_im.dwnsampling(np.clip(imrgb.mean(axis=2), 0, None) , "GBRG")

    angles_green = nzen_g[mask_ang[:, :, 1]].ravel()
    intensities_green = imrgb[:, :, 1][mask_ang[:, :, 1]].ravel()
    #mean_stack_15 = imrgb[:, :, 1][mask_ang[:, :, 1]].mean()
    #absolute_diff_stack_15 = np.sort(np.absolute(imrgb[:, :, 1][mask_ang[:, :, 1]] - mean_stack_15))
    #max_variation_stack_15 =


    ax3.imshow(imrgb[:, :, 1])
    ax3.imshow(mask_ang[:, :, 1], alpha=0.2)
    #ax3.imshow(m_square_rgb[:, :, 1], alpha=0.5)
    ax3.plot(centro[1], centro[0], "r+")
    ax3.plot(geo["green"].center[0], geo["green"].center[1], "g.")

    plt.show()
