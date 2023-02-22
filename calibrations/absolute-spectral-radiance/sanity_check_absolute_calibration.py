# -*- coding: utf-8 -*-
"""

"""

# Module importation
import h5py
import glob
import string
import datetime
from scipy.stats import t
import pandas
import numpy as np
from scipy.stats import linregress
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Other modules
from source.processing import ProcessImage, FigureFunctions
from source.radiance import ImageRadiancei360


# Functions
def UPD(X, Y):
    """
    Unbiased percent difference.

    :param X: Estimate X
    :param Y: Estimate Y
    :return: UPD
    """
    return 2 * np.abs(X - Y) / np.abs(X + Y)


def rmse(y_value, y_predicted):
    """
    Root means square error.

    :param y_value:
    :param y_predicted:
    :return:
    """
    diff_squared = (y_predicted - y_value) ** 2

    return np.sqrt(diff_squared.mean())


if __name__ == "__main__":

    # Path to data
    pp = ProcessImage()
    ff = FigureFunctions()

    # Folder choice
    #abs_path = pp.folder_choice() + "data-i360-tests/calibrations/absolute-radiance/validation/03182021"
    abs_path = pp.folder_choice() + "data-i360-tests/calibrations/absolute-radiance/validation/03182021"
    path_list = glob.glob(abs_path + "/IMG*.dng")
    path_list.sort()

    # C-OPS data
    dict_header = {380: 'LuZ380 (µW/(sr cm² nm))',
                   395: 'LuZ395 (µW/(sr cm² nm))',
                   412: 'LuZ412 (µW/(sr cm² nm))',
                   443: 'LuZ443 (µW/(sr cm² nm))',
                   465: 'LuZ465 (µW/(sr cm² nm))',
                   490: 'LuZ490 (µW/(sr cm² nm))',
                   510: 'LuZ510 (µW/(sr cm² nm))',
                   532: 'LuZ532 (µW/(sr cm² nm))',
                   555: 'LuZ555 (µW/(sr cm² nm))',
                   560: 'LuZ560 (µW/(sr cm² nm))',
                   589: 'LuZ589 (µW/(sr cm² nm))',
                   625: 'LuZ625 (µW/(sr cm² nm))',
                   665: 'LuZ665 (µW/(sr cm² nm))',
                   683: 'LuZ683 (µW/(sr cm² nm))',
                   694: 'LuZ694 (µW/(sr cm² nm))',
                   710: 'LuZ710 (µW/(sr cm² nm))',
                   765: 'LuZ765 (µW/(sr cm² nm))',
                   780: 'LuZ780 (µW/(sr cm² nm))',
                   875: 'LuZ875 (µW/(sr cm² nm))'}

    path_cops = glob.glob(abs_path + "/*.tsv")[0]
    dfcops = pandas.read_csv(path_cops, sep="\t", header=0, encoding="ISO-8859-1")

    #wl_cops = [625, 532, 490]
    #wl_cops = [625, 555, 490]
    wl_cops = [625, 555, 490]

    # Convert datetime to local datetime
    dfcops["DateTimeUTC"] = pandas.to_datetime(dfcops["DateTimeUTC"])
    dfcops["DateTimeUTC"] = dfcops["DateTimeUTC"].dt.tz_localize("utc").dt.tz_convert("America/Montreal")

    # Geometric parameters
    #mask_deg = 7  # 7 degrees corresponding to C-OPS half FOV (in-water though)
    mask_deg = 9  # 9 degrees half FOV (in-air) / 18 deg FAFOV
    #mask_deg = 15  # mask of 15 degrees give good correspondances

    # Loop
    cops_radiance = np.empty((len(path_list), 3))
    camera_radiance = np.empty((len(path_list), 3), dtype=[('avg', np.float32), ('std', np.float32)])
    timecam = np.array([], dtype=np.datetime64)

    for i, p in enumerate(path_list):
        print("Process image number {0}".format(i))

        im_rad = ImageRadiancei360(p, "air")
        im_rad.get_radiance()  # Compute spectral radiance

        # Time info
        tstamp = pandas.to_datetime(str(im_rad.metadata["Image DateTime"]), format="%Y:%m:%d %H:%M:%S") - pandas.Timedelta(hours=1)
        tstamp = tstamp.tz_localize("America/Montreal")
        timecam = np.append(timecam, tstamp.to_numpy())

        # C-ops data
        amin = np.argmin(np.abs(np.array(dfcops["DateTimeUTC"]).astype("datetime64[ns]") - tstamp.to_numpy()))

        zenith = im_rad.zen_c.copy()
        radiance_image = im_rad.getimage("close").copy()

        for n in range(radiance_image.shape[2]):
            maskzenith = zenith[:, :, n] <= mask_deg
            val = radiance_image[:, :, n][maskzenith]

            # Normalization

            camera_radiance["avg"][i, n] = val.mean()
            camera_radiance["std"][i, n] = val.std()
            cops_radiance[i, n] = float(dfcops.iloc[amin][dict_header[wl_cops[n]]]) / 100

    upd = np.abs((UPD(cops_radiance, camera_radiance['avg'])))
    MUPD = np.mean(upd, axis=0) * 100
    print(MUPD)

    # Uncertainties
    relunc_calib_coeff = np.tile(np.array([10.4, 10.6, 11.0]), (camera_radiance["avg"].shape[0], 1))
    relunc_rolloff = np.tile(np.array([3.0, 2.0, 4.0]), (camera_radiance["avg"].shape[0], 1))

    relunc_total = np.sqrt((camera_radiance["std"]/camera_radiance["avg"]) ** 2 +
                           (relunc_calib_coeff/100) ** 2 +
                           (relunc_rolloff/100) ** 2)

    # Figure 1
    plt.style.use("../../figurestyle.mplstyle")

    fig1 = plt.figure(figsize=ff.set_size(subplots=(2, 1)), constrained_layout=True)
    gs = fig1.add_gridspec(3, 9)
    ax1 = [fig1.add_subplot(gs[0, 5:]), fig1.add_subplot(gs[1, 5:]), fig1.add_subplot(gs[2, 5:])]

    #lambda_eff = [589, 544, 484]
    lambda_eff = [600, 540, 480]
    #lambda_cops = [589, 532, 490]
    #lambda_cops = [589, 555, 490]
    #txtstr = "$\lambda_{{cam, eff}}={0} nm$\n$\lambda_{{COPS}}={1} nm$\n$\mathrm{{MUPD}}={2:.1f}$%"
    txt_mupd = "$\mathrm{{MUPD}}={0:.1f}$%"

    marker = ["o", "s", "^"]
    colo = ['#d95f02', '#1b9e77', '#7570b3']
    lstyle = ["-", "-.", ":"]

    # Two-sided inverse Students t-distribution
    tinv = lambda p, df: abs(t.ppf(p / 2, df))

    for n, a in enumerate(ax1):

        # Linear regression
        res = linregress(cops_radiance[:, n], camera_radiance["avg"][:, n])

        # Print Results
        print(res.rvalue ** 2)
        ts = tinv(0.05,  len(cops_radiance[:, n]) - 2)
        print(f"slope (95%): {res.slope:.6f} +/- {ts * res.stderr:.6f}")
        print(f"intercept (95%): {res.intercept:.6f} +/- {ts * res.intercept_stderr:.6f}")

        # RMSE
        ypred = res.intercept + res.slope * cops_radiance[:, n]
        rootmeanse = rmse(camera_radiance["avg"][:, n], ypred)

        # Text results
        txt_eq = "$\overline{L}_{cam} = m \cdot L_{COPS} + b$\n"
        txt_reg = "$m = ({0:.1f} \pm {1:.1f})$\n".format(res.slope, ts * res.stderr) + "$b = ({0:.2f} \pm {1:.2f})$\n".format(res.intercept, ts * res.intercept_stderr)
        txt_res = txt_eq + txt_reg + "$R^2=${0:.3f}\n$RMSE=${1:.3f}\n".format(res.rvalue ** 2, rootmeanse) + txt_mupd.format(MUPD[n])

        # Plot result graph
        one_one = np.linspace(cops_radiance[:, n].min() * 0.9, cops_radiance[:, n].max() * 1.1, 1000)
        a.plot(one_one, one_one, "k--", linewidth=0.9, label="1:1")
        #a.scatter(cops_radiance[:, n], camera_radiance["avg"][:, n], s=5, c=colo[n], label="data")
        a.errorbar(cops_radiance[:, n], camera_radiance["avg"][:, n],
                   yerr=relunc_total[:, n] * camera_radiance["avg"][:, n], xerr=cops_radiance[:, n] * 2.5/100,
                   linestyle="none", marker=".", color=colo[n], markeredgecolor="k", alpha=0.8, label="data")
        a.plot(one_one, res.intercept + res.slope * one_one, c=colo[n], linewidth=1.5, label="linear fit")

        a.text(0.88, 0.9, "(" + string.ascii_lowercase[2 * n + 1] + ")", transform=a.transAxes, size=11, weight='bold')
        #a.text(0.01, 0.7, txtstr.format(lambda_eff[n], lambda_cops[n], MUPD[n]), transform=a.transAxes, fontsize=8)
        a.text(0.01, 0.55, txt_res, transform=a.transAxes, fontsize=7)

        #a.set_ylabel("$L_{cam}$ " + "[$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")
        #a.set_xlabel("$L_{COPS}$ " + "[$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")

        a.set_ylabel("$\overline{L}_{cam}$ [$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")
        a.set_xlabel("$L_{COPS}$ [$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")

        a.legend(loc=4)

    # Spectral composition of light
    spectra = np.zeros((19, 400))
    rad_spectra = spectra.copy()
    for i in range(spectra.shape[1]):
        for j, k in enumerate(dict_header.keys()):
            spectra[j, i] = k
            rad_spectra[j, i] = dfcops.iloc[i][dict_header[k]] / 100

    # Figure 2
    ax2 = [fig1.add_subplot(gs[0, :-4]), fig1.add_subplot(gs[1, :-4]), fig1.add_subplot(gs[2, :-4])]
    txt_eff = "$\lambda_{{cam, eff}}={0} nm$\n$\lambda_{{COPS}}={1} nm$"

    #ax1[0].set_yticks(np.arange(0, 0.175, 0.025))
    ax1[0].set_xscale("log")
    ax1[0].set_yscale("log")
    ax1[0].set_xticks(ax1[0].get_yticks())
    ax1[0].set_xlim(ax1[0].get_ylim())
    #ax1[0].tick_params(axis='x', labelrotation=30)

    #ax1[1].set_yticks(np.arange(0, 0.175, 0.025))
    ax1[1].set_xscale("log")
    ax1[1].set_yscale("log")
    #ax1[1].set_yticks(np.arange(0, 0.2, 0.025))
    ax1[1].set_xticks(ax1[1].get_yticks())
    ax1[1].set_xlim(ax1[1].get_ylim())
    #ax1[1].set_ylim((0.014906326953882986, 0.19247353881009444))
    #ax1[1].set_xlim((0.014906326953882986, 0.19247353881009444))
    #ax1[1].tick_params(axis='x', labelrotation=30)

    ax1[2].set_xscale("log")
    ax1[2].set_yscale("log")
    #ax1[2].set_yticks(np.arange(0, 0.4, 0.025))
    ax1[2].set_xticks(ax1[2].get_yticks())
    ax1[2].set_xlim(ax1[2].get_ylim())
    #ax1[2].set_ylim((0.024422888556137584, 0.2284817821166338))
    #ax1[2].set_xlim((0.024422888556137584, 0.2284817821166338))
    #ax1[2].tick_params(axis='x', labelrotation=30)

    #ax1[0].set_yticklabels(np.array([]))
    #ax1[1].set_yticklabels(np.array([]))
    #ax1[2].set_yticklabels(np.array([]))

    for i, a in enumerate(ax2):

        # Plot C-OPS and camera data
        # "Cam, $\lambda_{{cam, eff}}={0}$ nm"
        # "C-OPS, $\lambda_{{COPS}}={0} nm$"
        a.plot(dfcops["DateTimeUTC"], dfcops[dict_header[wl_cops[i]]]/100, label="C-OPS, $\lambda={0} nm$".format(wl_cops[i]), color="k")
        #a.plot(timecam, camera_radiance["avg"][:, i], color=colo[i], linestyle='none', marker=marker[i], markersize=4, markerfacecolor="none", label="Cam, $\lambda={0}$ nm".format(lambda_eff[i]))
        a.errorbar(timecam, camera_radiance["avg"][:, i], yerr=relunc_total[:, i] * camera_radiance["avg"][:, i],
                   color=colo[i], ecolor=colo[i], linestyle="none", marker=marker[i], ms=4, markerfacecolor="none",
                   capsize=2, label="Cam, $\lambda={0}$ nm".format(lambda_eff[i]), alpha=0.8)

        # Error
        #a.fill_between(timecam, camera_radiance["avg"][:, i] - relunc_total[:, i] * camera_radiance["avg"][:, i],
        #               camera_radiance["avg"][:, i] + relunc_total[:, i] * camera_radiance["avg"][:, i], color=colo[i], alpha=0.6)

        a.set_ylabel("$\overline{L}$ " + "[$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")
        #a.set_ylabel("Spectral radiance")

        #a.text(0.5, 0.7, txt_eff.format(lambda_eff[i], lambda_cops[i]), transform=a.transAxes, fontsize=8)

        myFmt = mdates.DateFormatter('%H:%M:%S', tz=dfcops["DateTimeUTC"][0].tz)
        a.xaxis.set_major_formatter(myFmt)
        a.tick_params(axis='x', labelrotation=45)
        a.text(0.02, 0.9, "(" + string.ascii_lowercase[2 * i] + ")", transform=a.transAxes, size=11, weight='bold')

        a.set_yscale("log")

        ax2[i].legend(loc="best")

    ax2[2].set_xlabel("Time of day on {0}".format(dfcops["DateTimeUTC"][0].strftime("%d %B %Y")))

    # Figure 3
    fig3, ax3 = plt.subplots(1, 1)

    ax3.plot(spectra[:, 0], rad_spectra.mean(axis=1), linestyle="-", marker=".")
    ax3.fill_between(spectra[:, 0], rad_spectra.mean(axis=1)-rad_spectra.std(axis=1), rad_spectra.mean(axis=1)+rad_spectra.std(axis=1), color="gray", alpha=0.8)
    ax3.set_xlabel("Wavelength [nm]")
    ax3.set_ylabel("$L$ " + " [$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")

    fig1.tight_layout()
    fig1.savefig("figures/validation_sky.pdf", format="pdf", dpi=600)
    fig1.savefig("figures/validation_sky.png", format="png", dpi=600)
    plt.show()
