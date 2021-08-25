# -*- coding: utf-8 -*-
"""
Insta360 ONE camera relative spectral response rsr (characterization).
"""

# Module importation
import os
import re
import glob
import time
import h5py
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes

# Other module importation
import source.processing as processing
from source.geometric_rolloff import MatlabGeometricMengine


# Functions
def read_ascii(path):
    """

    :param path:
    :return:
    """
    start = False
    s, e = [], []
    regex = re.compile(r'\d+')
    with open(path, "r") as f:
        for line in f:
            if start:
                s.append(float(line.strip().split()[0]))
                e.append(float(line.strip().split()[1]))
            else:
                if line.strip() == "#DATA":
                    start = True
    return regex.findall(os.path.basename(path)), np.array(s), np.array(e)


def identify_lens(path):
    """
    Function to identify the optics of Insta360 ONE (close or far).

    :param path:
    :return:
    """
    if "lensclose" in path:
        which_lens = "close"
    elif "lensfar" in path:
        which_lens = "far"
    else:
        raise ValueError("Could not identify lens.")
    return which_lens


def find_centroid(image):
    """

    :param imagepath:
    :param whichlens:
    :return:
    """
    pim = processing.ProcessImage()

    im_dws = pim.dwnsampling(image, "RGGB")

    _, regpro = pim.region_properties(im_dws[:, :, 1], 1000)
    centro = np.round(regpro[0].centroid).astype(int)

    return im_dws[:, :, 1], centro


def chunck_imagelist(imagelist, n):
    """

    :param imagelist:
    :param n:
    :return:
    """
    newimlist = []
    for i in range(0, len(imagelist), n):
        newimlist.append(imagelist[i:i+n])
    return newimlist


def avg_image(imagelist, amb_image, which):
    """

    :param imagelist:
    :param which:
    :return:
    """
    pim = processing.ProcessImage()
    stack = np.empty((3456, 3456, len(imagelist)))
    exp = np.array([])

    for n, l in enumerate(imagelist):
        s = pim.imagestack(l, which)
        currim = s[0] - amb_image[:, :, None]
        stack[:, :, n] = np.clip(currim.mean(axis=2), 0, None)
        exp = np.append(exp, s[1][0])
    return stack, exp


def stack_roidata(imstack, nbpixel, centroid, exptime=False):
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
    y, x = centroid

    for n in range(imstack.shape[2]):

        # Downsampling
        im_dws = pim.dwnsampling(imstack[:, :, n], "RGGB")

        for i in range(im_dws.shape[2]):
            val = im_dws[:, :, i]
            if isinstance(exptime, bool):
                roi = val[y - nbpixel // 2:y + nbpixel // 2 + 1, x - nbpixel // 2:x + nbpixel // 2 + 1]
            else:
                roi = val[y-nbpixel//2:y+nbpixel//2+1, x-nbpixel//2:x+nbpixel//2+1] / exptime[n]
            data["dn_avg"][n, i] = roi.mean()
            data["dn_std"][n, i] = roi.std()

    return data


def rsr_statistics(wavelength, rsr):
    """
    Printing spectral response statistics.
    :param wavelength:
    :param rsr:
    :return:
    """

    for band in range(rsr.shape[1]):
        eff_bw = np.trapz(rsr[:, band], x=wavelength)
        eff_wl = np.trapz(rsr[:, band] * wavelength, x=wavelength) / eff_bw
        max_wl = wavelength[np.argmax(rsr[:, band])]

        print("Band no. {0} statistics".format(band))
        print("Effective bw: {0:.4f}, effective wl: {1:.4f}, maximum wl: {2:.4f}". format(eff_bw, eff_wl, max_wl))


def relative_uncertainty(relative_unc_x, relative_unc_y):
    """
    Calculate relative uncertainties for multiplication of division.

    :param relative_unc_x:
    :param relative_unc_y:
    :return:
    """
    return np.sqrt(relative_unc_x**2 + relative_unc_y**2)


def uncertainty_trapz(x, unc):
    """

    :param x:
    :param unc:
    :return:
    """
    diff = x[1:] - x[:-1]
    return np.sqrt(np.sum((unc[1:] + unc[:-1])**2 * (diff[:, None]/2) ** 2, axis=0))


def trapz_replica(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    diff = x[1:] - x[:-1]
    return np.sum((y[1:] + y[:-1]) * diff[:, None]/2, axis=0)


def round_signi(arr, unc):
    """

    :param arr:
    :param unc:

    :return:
    """
    if arr.shape == unc.shape:
        newunc = np.empty(unc.shape)
        newarr = np.empty(arr.shape)
        for j in range(unc.shape[1]):
            for i in range(unc.shape[0]):
                newunc[i, j] = np.round(unc[i, j], decimals=-np.floor(np.log10(abs(unc[i, j]))).astype(int))
                newarr[i, j] = np.round(arr[i, j], decimals=-np.floor(np.log10(abs(newunc[i, j]))).astype(int))
    else:
        raise ValueError("Values and uncertainties are not the same shape.")
    return newarr, newunc


def create_hdf5_dataset(path, group, dataname, dat):
    """

    :param path:
    :param group:
    :param dataname:
    :param dat:

    :return:
    """
    datapath = group + "/" + dataname
    with h5py.File(path) as hf:
        if datapath in hf:
            d = hf[datapath]  # load the data
            d[...] = dat
        else:
            hf.create_dataset(group + "/" + dataname, data=dat)


if __name__ == "__main__":

    # FigureFunction object
    ff = processing.FigureFunctions()

    # ProcessImage object
    processim = processing.ProcessImage()

    generalpath = processim.folder_choice("/Volumes/MYBOOK/data-i360/calibrations/relative-spectral-response")

    # Which lens ?
    wlens = identify_lens(generalpath)

    # Image lists
    amb_list = glob.glob(generalpath + "/images/AMB*.dng")

    im_list = processim.imageslist(generalpath + "/images")
    im_list_sep = chunck_imagelist(im_list, 5)

    # Avg image
    stack_ambiance = processim.imagestack(amb_list, wlens)[0]
    image_ambiance = stack_ambiance.mean(axis=2)

    stack, exptime = avg_image(im_list_sep, image_ambiance, wlens)

    # Centroid
    _, centro = find_centroid(stack[:, :, 15])

    # Loop mean
    nb_pixel = 11
    data = stack_roidata(stack, nb_pixel, centro, exptime=exptime)

    # Reading ascii data
    beamE2list = glob.glob(generalpath + "/beamE2/*.asc")

    # Fig 2
    fig1, ax1 = plt.subplots(1, 1)

    e2 = np.array([])
    e2_sigma = np.array([])
    wl = np.array([])
    for n, files in enumerate(beamE2list):
        lam, s, e = read_ascii(files)
        if lam:
            e2 = np.append(e2, e.mean())
            e2_sigma = np.append(e2_sigma, e.std())
            wl = np.append(wl, float(lam[0]))

            # Figure 2
            ax1.plot(s, e)

    # Relative uncertainty
    e2_relunc = (e2_sigma / e2)  # beam relative uncertainty
    data_relunc = data["dn_std"] / data["dn_avg"]  # data standard uncertainty (standard deviation)

    # Normalization by beam power
    dnorm = data["dn_avg"] / e2[:, None]
    dnorm_unc = data["dn_std"] / e2[:, None]
    dnorm_relunc = relative_uncertainty(data_relunc, e2_relunc[:, None])  # data normalized by beam relative uncertainty

    # Normalization by maximum
    dnorm_argmax = np.argmax(dnorm, axis=0)
    rsr = dnorm / np.diagonal(dnorm[dnorm_argmax[None, :]][0])
    rsr_unc = dnorm_unc / np.diagonal(dnorm[dnorm_argmax[None, :]][0])
    rsr_relunc = relative_uncertainty(dnorm_relunc, np.diagonal(dnorm_relunc[dnorm_argmax[None, :]][0]))  #rsr relative

    # Caracterizations
    rsr_statistics(wl, rsr)

    bw = trapz_replica(wl, rsr)  # bandwidth
    bw_abs_unc = uncertainty_trapz(wl, rsr_relunc * rsr)  # bandwidth uncertainty

    wl_eff = trapz_replica(wl, rsr * wl[:, None]) / bw  # effective wavelegnth

    wl_accuracy = np.ones(rsr.shape[0]) * 0.08
    # wavelength accuracy of 0.08 nm: http://mobile.labwrench.com/equipment/2891/perkinelmer/lambda-850-950-1050
    wl_relunc = wl_accuracy / wl

    nom_rel_unc = uncertainty_trapz(wl, relative_uncertainty(wl_relunc[:, None], rsr_relunc) * wl[:, None] * rsr) / trapz_replica(wl, rsr * wl[:, None])
    wl_eff_abs_unc = wl_eff * relative_uncertainty(nom_rel_unc, bw_abs_unc / bw)

    # Table using Pandas Dataframe
    newrsr, newrrsr_relunc = round_signi(rsr, rsr_relunc * rsr)
    rsrdf = pandas.DataFrame(np.concatenate((wl.reshape(-1, 1), newrsr, newrrsr_relunc), axis=1))
    print(rsrdf)

    # Figures _____
    plt.style.use("../../figurestyle.mplstyle")

    # Fig 1
    ax1.set_xlabel("time [seconds]")
    ax1.set_ylabel("Reference beam energy [a.d.u]")

    # Fig 2 - RSR
    fig2, ax2 = plt.subplots(1, 2, figsize=(12.8, 4.8))
    figrsr = plt.figure(figsize=ff.set_size(443.86319, fraction=0.7))  #relative spectral response only
    axrsr = figrsr.add_subplot(111)

    lstyle = ["-", "-.", ":"]
    m = ["o", "s", "d"]
    col = ['#d62728', '#2ca02c', '#1f77b4']
    lab = ["red band", "green band", "blue band"]

    for i in range(rsr.shape[1]):

        ax2[0].plot(wl, rsr[:, i], marker=m[i], linestyle=lstyle[i], color=col[i], label=lab[i], markeredgecolor=col[i], markerfacecolor="none")
        ax2[0].fill_between(wl, rsr[:, i]-rsr_unc[:, i], rsr[:, i]+rsr_unc[:, i], color="lightgrey", alpha=0.7)

        axrsr.plot(wl, rsr[:, i], marker=m[i], markersize=3, linestyle=lstyle[i], color=col[i], label=lab[i], markeredgecolor=col[i], markerfacecolor="none")
        axrsr.fill_between(wl, rsr[:, i] - rsr_unc[:, i], rsr[:, i] + rsr_unc[:, i], color="lightgrey", alpha=0.7)

    ax2[0].set_xticks(range(375, 725, 25))
    ax2[0].set_xlim((390, 685))
    ax2[0].set_ylabel("$RSR$")
    ax2[0].set_xlabel("Wavelength [nm]")

    axrsr.set_xticks(range(375, 725, 25))
    axrsr.set_xlim((390, 685))
    axrsr.set_ylabel("$RSR$")
    axrsr.set_xlabel("Wavelength [nm]")

    ax2[1].errorbar(wl, e2, color="k", yerr=e2_sigma, linewidth=1.5)
    ax2[1].tick_params(axis="y", labelcolor="k")

    ax2[1].set_xticks(range(375, 725, 25))
    ax2[1].set_xlim((390, 685))
    ax2[1].set_ylim((0.08, 0.82))

    ax2[1].set_xlabel("Wavelength [nm]")
    ax2[1].set_ylabel("Signal [a.d.u]")

    # Fig3 - standard uncertainty
    ax3 = ax2[1].twinx()
    ax3.plot(wl, e2_relunc * 100, color="gray", linestyle="--", linewidth=1.5)

    ax3.set_ylim((0.03, 0.21))

    ax3.tick_params(axis="y", labelcolor="gray")

    ax3.set_ylabel("Standard uncertainty [%]", color="gray")

    # Fig4 - figure of standard uncertainty
    fig4, ax4 = plt.subplots(1, 1)

    for i in range(data_relunc.shape[1]):
        ax4.plot(wl, data_relunc[:, i]*100, marker=".", linestyle="-", color=col[i], label="$DN_{beam}(\lambda) - DN_{dark}$" + " " + lab[i])

    ax4.plot(wl, e2_relunc * 100, marker=".", linestyle="--", label="$I(\lambda)$")
    ax4.set_yscale("log")

    ax4.set_xticks(range(375, 725, 25))
    ax4.set_xlim((390, 685))

    ax4.set_ylabel("Standard uncertainty [%]")
    ax4.set_xlabel("Wavelength [nm]")
    ax4.legend(loc="best", fontsize=9)

    # # Fig non-uniformity
    # fnum1, _ = spatial_uniformity(stack, wl, np.array([590, 600, 630]), "red", centro)
    # fnum2, _ = spatial_uniformity(stack, wl, np.array([510, 520, 530]), "green", centro)
    # fnum3, _ = spatial_uniformity(stack, wl, np.array([450, 460, 470]), "blue", centro)

    figrsr.tight_layout()

    # Saving file
    filename = "rsr_" + time.strftime("%Y%m%d", time.localtime(os.stat(im_list[0])[-2])) + ".h5"
    pathname = "calibrationfiles/" + filename

    saved_answer = processim.save_results()
    if saved_answer == "y":

        if wlens == "close":
            create_hdf5_dataset(pathname, "lens-close", "rsr_peak_norm", rsr)
            create_hdf5_dataset(pathname, "lens-close", "rsr_relative_unc", rsr_relunc)
            create_hdf5_dataset(pathname, "lens-close", "wavelength", wl)
        else:
            create_hdf5_dataset(pathname, "lens-far", "rsr_peak_norm", rsr)
            create_hdf5_dataset(pathname, "lens-far", "rsr_relative_unc", rsr_relunc)
            create_hdf5_dataset(pathname, "lens-far", "wavelength", wl)

        figrsr.savefig("figures/rsr_{0}.pdf".format(wlens), format="pdf", dpi=600)

    plt.show()
