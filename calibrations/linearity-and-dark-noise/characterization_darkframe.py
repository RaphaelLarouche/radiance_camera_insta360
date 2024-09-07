# -*- coding: utf-8 -*-
"""
Noise analysis of insta360 ONE camera.
"""

# Importation of standard modules
import os
import glob
import string
import natsort  # TODO: remove dependencies on Natsort
import numpy as np
import matplotlib.cm
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Importation of other modules
from source.processing import ProcessImage, FigureFunctions


# Functions
def create_dict(path):
    """

    :param path:
    :return:
    """
    subfolders = os.listdir(path)
    dictio = {}
    for s in subfolders:
        if not "." in s:
            ptot = path + "/" + s
            dictio[s] = glob.glob(ptot + "/*.dng")
    return dictio


def blacklevel_metadata(metadata):
    return float(str(metadata['Image Tag 0xC61A']))


def exposuretime_metadata(metadata):
    """

    :return:
    """
    exp = str(metadata["Image ExposureTime"]).split("/")
    if len(exp) > 1:
        return float(exp[0])/float(exp[1])
    else:
        return float(exp[0])


def Gauss(x, a, x0, sigma):
    """
    Gaussian function.
    :param x:
    :param a:
    :param x0:
    :param sigma:
    :return:
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def plot_mean_histogram_exp(fig, ax, imagestack_dictionary, nbin, range, label):
    """

    :param fig:
    :param ax:
    :param imagestack_dictionary:
    :param nbin:
    :param range:
    :param label:
    :return:
    """
    cm = matplotlib.cm.cividis(np.linspace(0, 1, len(imagestack_dictionary.keys())))  # color

    # dummy
    c = np.arange(1, len(imagestack_dictionary.keys()) + 1)
    val = np.array(list(label.values()))
    dcax = ax.scatter(val, val, c=c, cmap=matplotlib.cm.get_cmap('cividis', len(imagestack_dictionary.keys())))
    ax.cla()

    # Pre-allocation
    mu = np.array([])  # average
    sigma = np.array([])  # standard deviation
    real_mu = np.array([])
    real_sigma = np.array([])

    for i, k in enumerate(natsort.natsorted(imagestack_dictionary.keys())[::-1]):

        current_stack = imagestack_dictionary[k]
        stack_mean = np.mean(current_stack, axis=2)

        h = np.histogram(stack_mean, bins=int(nbin), range=range)
        mask_range = (range[0] <= stack_mean) & (stack_mean <= range[1])
        data_range = stack_mean[mask_range]
        bincenter = (h[1][:-1] + h[1][1:]) / 2
        print(bincenter[np.argmax(h[0])])
        popt, pcov = curve_fit(Gauss, bincenter, h[0]/h[0].max(), p0=[1, 800, 1])

        #plt.figure()
        #plt.plot(bincenter, h[0]/h[0].max(), linestyle="-")
        #plt.errorbar(popt[1], 0.66, xerr=popt[2])
        #plt.axvline(popt[1])
        #plt.plot(np.linspace(780, 820, 100), Gauss(np.linspace(780, 820, 100), *popt), linestyle="-")

        real_mu = np.append(real_mu, stack_mean.mean())
        real_sigma = np.append(real_sigma, stack_mean.std())
        mu = np.append(mu, popt[1])
        sigma = np.append(sigma, popt[2])

        #print("Fit avg, std: {0:.4f}, {1:.4f}".format(popt[1], popt[2]))
        #print("Avg centroid: {0:.4f}".format(np.sum(h[0]*bincenter)/np.sum(h[0])))

        leg = "{0:.1f} ms, $\mu = {1:.1f}$, $\sigma = {2:.2f}$".format(label[k] * 1000, data_range.mean(), data_range.std())
        ax.plot(bincenter, h[0], color=cm[i, :], linewidth=2.3, alpha=0.8, label=leg)
        ax.set_yscale("log")

    cb = fig.colorbar(dcax, ax=ax, orientation="vertical", fraction=0.05, pad=0.04)
    cb.ax.locator_params(nbins=len(list(label.values())))
    #cb.ax.yaxis.set_major_locator(mticker.FixedLocator(list(label.values())))
    #cb.set_ticks(np.array(list(label.values())))
    cb.ax.set_yticklabels(["{0:.2e}".format(a) for a in np.array(list(label.values()))])
    cb.ax.set_title("$t_{int}~\mathrm{[s]}$", fontsize=8)

    return ax, mu, sigma, real_mu, real_sigma


def plot_mean_histogram_iso(fig, ax, imagestack, nbin, range, label):
    """

    :param fig:
    :param ax:
    :param imagestack:
    :param nbin:
    :param range:
    :param label:
    :return:
    """
    cm = matplotlib.cm.cividis(np.linspace(0, 1, len(imagestack.keys())))   # color

    c = np.arange(1, len(imagestack.keys()) + 1)
    val = np.array(list(label.values()))
    dcax = ax.scatter(val, val, c=c, cmap=matplotlib.cm.get_cmap('cividis', len(imagestack.keys())))
    ax.cla()

    # Pre-allocation
    mu_i = np.array([])  # average
    sigma_i = np.array([])  # standard deviation
    real_mu_i = np.array([])  # average
    real_sigma_i = np.array([])  # standard deviation

    for i, k in enumerate(natsort.natsorted(imagestack.keys())):

        current_stack = imagestack[k]
        stack_mean = np.mean(current_stack, axis=2)

        if k == "ISO3200":
            h = np.histogram(stack_mean, bins=438, range=(600, 1100))
            mask_range = (600 <= stack_mean) & (stack_mean <= 1100)
            data_range = stack_mean[mask_range]
        #elif k == "ISO1600":
        #    h = np.histogram(stack_mean, bins=200, range=(700, 1000))
        #    mask_range = (700 <= stack_mean) & (stack_mean <= 1000)
        #    data_range = stack_mean[mask_range]
        else:
            h = np.histogram(stack_mean, bins=int(nbin), range=range)
            mask_range = (range[0] <= stack_mean) & (stack_mean <= range[1])
            data_range = stack_mean[mask_range]

        bincenter = (h[1][:-1] + h[1][1:]) / 2

        popt, pcov = curve_fit(Gauss, bincenter, h[0] / h[0].max(), p0=[1, 800, 1])

        real_mu_i = np.append(real_mu_i, stack_mean.mean())
        real_sigma_i = np.append(real_sigma_i, stack_mean.std())
        mu_i = np.append(mu_i, popt[1])
        sigma_i = np.append(sigma_i, popt[2])

        leg = "ISO {0:.0f}, $\mu = {1:.1f}$, $\sigma = {2:.2f}$".format(label[k], data_range.mean(), data_range.std())
        ax.plot(bincenter, h[0], color=cm[i, :], linewidth=2.3, alpha=0.8, label=leg)
        ax.set_yscale("log")

    cb = fig.colorbar(dcax, ax=ax, orientation="vertical", fraction=0.05, pad=0.04)
    cb.ax.set_yticklabels(["{0:.0f}".format(i) for i in np.sort(np.array(list(label.values())))])
    cb.ax.set_title("$S_{ISO}$", fontsize=8)

    return ax, mu_i, sigma_i, real_mu_i, real_sigma_i


def loop_imagestack(image_dict, which="close"):
    """

    :param image_dict:
    :param which:
    :return:
    """
    openimage = ProcessImage()

    imstack_dict = {}
    exposure = {}
    iso = {}

    # Loop for exposure
    for key in natsort.natsorted(image_dict.keys())[::-1]:
        print("Processing: {0}".format(key))

        imstack = np.empty((3456, 3456, len(image_dict[key])))

        for n, impath in enumerate(image_dict[key]):
            # Opening image
            im, met = openimage.readDNG_insta360_np(impath, which)
            imstack[:, :, n] = im

        imstack_dict[key] = imstack
        exposure[key] = exposuretime_metadata(met)
        iso[key] = float(str(met["Image ISOSpeedRatings"]))

    return imstack_dict, exposure, iso


if __name__ == "__main__":

    # Instance of figurefunctions
    ff = FigureFunctions()

    # if Mac:
    # General path MYBOOK
    #path_ex = "/Volumes/MYBOOK/data-i360/calibrations/darkframe/integration-time"
    #path_iso = "/Volumes/MYBOOK/data-i360/calibrations/darkframe/iso-gain"

    # if Windows:
    # General path MYBOOK
    gen_path = '/Volumes/MYBOOK'
    path_ex = gen_path + "/data-i360/calibrations/darkframe/integration-time"  # exposure path
    path_iso = gen_path + "/data-i360/calibrations/darkframe/iso-gain"  #iso gain

    dict_images_ex = create_dict(path_ex)
    dict_images_iso = create_dict(path_iso)

    # Looping over all image
    imstack_exp, exp_exp, iso_exp = loop_imagestack(dict_images_ex, which="close")
    imstack_iso, exp_iso, iso_iso = loop_imagestack(dict_images_iso, which="close")

    # Figures
    plt.style.use(os.path.abspath(os.path.join(__file__, "../../..")) + "/figurestyle.mplstyle")

    # Fig1 - histogram
    #fig1, ax1 = plt.subplots(1, 2, figsize=ff.set_size(height_ratio=0.45), sharey=True)
    fig1, ax1 = plt.subplots(2, 2, figsize=ff.set_size(height_ratio=0.8), sharey=False)

    ax1[0, 0], mu_exp, sigma_exp, real_mu_exp, real_sigma_exp = plot_mean_histogram_exp(fig1, ax1[0, 0], imstack_exp, 75, (790, 810), exp_exp)
    ax1[0, 0].text(798, 3000, "$S_{{ISO}} = {0}$".format(int(iso_exp["1_4000s"])), fontsize=7)

    #ax1[0, 0].set_xticks(np.arange(780, 820, 5))
    #ax1[0, 0].set_xlim((789.1466666666666, 810.8533333333334))
    ax1[0, 0].set_xlabel("Pixel-wise averaged $y_{DN}$ [ADU]")
    ax1[0, 0].set_ylabel("Counts")
    ax1[0, 0].set_aspect('auto')
    #ax1[0].legend(loc="best", fontsize=9)

    ax1[0, 1], mu_iso, sigma_iso, real_mu_iso, real_sigma_iso = plot_mean_histogram_iso(fig1, ax1[0, 1], imstack_iso, 142, (750, 900), iso_iso)
    ax1[0, 1].text(850, 1000000, r"$t_{{int}}={0:.1f} \mathrm{{s}}$".format(exp_iso["ISO3200"]), fontsize=7)

    #ax1[0, 1].set_xticks(np.arange(725, 1075, 50))
    #ax1[0, 1].set_xlim((725, 975))

    #ax1[0, 1].set_yticks(ax1[0, 0].get_yticks())
    #ax1[0, 1].set_ylim((0.7, 5*10**6))
    ax1[0, 1].set_xlabel("Pixel-wise averaged $y_{DN}$ [ADU]")
    ax1[0, 1].set_ylabel("Counts")
    ax1[0, 1].set_aspect('auto')
    #ax1[1].legend(loc="best", fontsize=9)

    # Figure exposure time vs. average and std
    marker = ["o", "s", "d"]
    ls = ["-", "-.", ":"]
    #col = '#d62728'
    col = "gray"

    ax1[1, 0].plot(exp_exp.values(), mu_exp, marker=marker[0], markersize=5, linestyle=ls[0], color="k", markerfacecolor="none", markeredgecolor="k")

    ax1[1, 0].set_xscale("log")
    ax1[1, 0].set_yticks(np.arange(800, 803, 0.25))
    ax1[1, 0].set_ylim((799.90, 802.1))
    ax1[1, 0].set_xticks(np.array([10**-4, 10**-3, 10**-2, 10**-1, 10**0]))
    ax1[1, 0].set_xlim((0.00015950898792959354, 3.134617195566509))
    ax1[1, 0].set_ylabel(r"$\mu_{y_{DN}}$ [ADU]")
    ax1[1, 0].set_xlabel("$t_{int}~\mathrm{[s]}$")

    ax2_2 = ax1[1, 0].twinx()
    ax2_2.plot(exp_exp.values(), sigma_exp, marker=marker[1], markersize=5, linestyle=ls[1], color=col, markerfacecolor="none", markeredgecolor=col)
    #ax2_2.set_ylim((-1, 26))
    ax2_2.set_ylabel(r'$\sigma_{y_{DN}}$ [ADU]', color=col)
    ax2_2.tick_params(axis='y', labelcolor=col)

    # Figure ISO gain vs. average and std

    ax1[1, 1].plot(np.array(list(iso_iso.values())[::-1]), mu_iso, marker=marker[0], markersize=5, linestyle=ls[0], color="k", markerfacecolor="none", markeredgecolor="k")

    ax1[1, 1].set_ylim((799, 826))
    ax1[1, 1].set_ylabel(r"$\mu_{y_{DN}}$ [ADU]")
    ax1[1, 1].set_xlabel("$S_{ISO}$")

    ax11_2 = ax1[1, 1].twinx()
    #ax11_2.set_yticks(np.arange(-20, 140, 20))
    ax11_2.set_ylim((-1, 41))
    ax11_2.plot(np.array(list(iso_iso.values())[::-1]), sigma_iso, markersize=5, marker=marker[1], linestyle=ls[1], color=col, markerfacecolor="none", markeredgecolor=col)
    ax11_2.set_ylabel(r'$\sigma_{y_{DN}}$ [ADU]', color=col)
    ax11_2.tick_params(axis='y', labelcolor=col)

    ax1[0, 0].text(0.02, 0.90, "(" + string.ascii_lowercase[0] + ")", transform=ax1[0, 0].transAxes, size=11, weight='bold')
    ax1[0, 1].text(0.02, 0.90, "(" + string.ascii_lowercase[1] + ")", transform=ax1[0, 1].transAxes, size=11, weight='bold')
    ax1[1, 0].text(0.02, 0.90, "(" + string.ascii_lowercase[2] + ")", transform=ax1[1, 0].transAxes, size=11, weight='bold')
    ax1[1, 1].text(0.02, 0.90, "(" + string.ascii_lowercase[3] + ")", transform=ax1[1, 1].transAxes, size=11, weight='bold')

    fig1.tight_layout(h_pad=1)

    # All Exposure time histograms
    for t in imstack_iso.keys():

        average_darkimages = imstack_iso[t].mean(axis=2)
        range_darkimages = int(np.floor(average_darkimages.min())), int(np.ceil(average_darkimages.max()))
        nbin = range_darkimages[1] - range_darkimages[0]
        histo = np.histogram(average_darkimages, bins=int(nbin+1), range=range_darkimages)
        bcenter = (histo[1][:-1] + histo[1][1:]) / 2
        print(bcenter[histo[0].argmax()])
        popt, pcov = curve_fit(Gauss, bcenter, histo[0]/histo[0].sum(), p0=[1, 800, 1])
        print(popt)
        min_, max_ = 750, 1000
        plt.figure()
        plt.plot((histo[1][:-1] + histo[1][1:]) / 2, histo[0]/histo[0].sum())
        plt.plot(np.linspace(min_, max_, 1000), Gauss(np.linspace(min_, max_, 1000), *popt))
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.gca().set_xlim((min_, max_))
        plt.gca().set_ylim((1e-7, 2.0))
        plt.gca().set_title(t)

    plt.figure()
    for k in (imstack_iso.keys()):
        a = []
        for i in range(imstack_iso[k].shape[2]):
            #print(imstack_iso[k][:, :, i].mean())
            #print(imstack_iso[k][:, :, i].max())
            #print(imstack_iso[k][:, :, i].min())
            a.append(np.median(imstack_iso[k][:, :, i]))

        plt.plot(np.arange(len(a)), a, linestyle="-", marker='.', label=k + f'avg-{np.mean(a):.3f} / {np.std(a):.3f} ')
    plt.gca().legend(loc='best')

    # Saving figure
    fig1.savefig("figures/dark-histograms.pdf", format="pdf", dpi=600)
    fig1.savefig("figures/dark-histograms.png", format="png", dpi=600)
    fig1.savefig("figures/dark-histograms.jpg", format="jpg", dpi=600)

    plt.show()
