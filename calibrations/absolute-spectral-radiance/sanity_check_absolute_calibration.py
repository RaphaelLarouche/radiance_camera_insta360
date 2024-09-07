# -*- coding: utf-8 -*-
"""

"""

# Module importation
import os
import h5py
import glob
import string
import datetime
from scipy.stats import t, pearsonr
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
    abs_path = "/Volumes/MYBOOK/data-i360-tests/calibrations/absolute-radiance/validation/03182021"
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
    immersion_factor = [0.5770, 0.5752, 0.5730]

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
    amin_array = np.array([])

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
        amin_array = np.append(amin_array, amin)

        zenith = im_rad.zen_c.copy()
        radiance_image = im_rad.getimage("close").copy()

        for n in range(radiance_image.shape[2]):
            maskzenith = zenith[:, :, n] <= mask_deg
            val = radiance_image[:, :, n][maskzenith]

            # Normalization
            camera_radiance["avg"][i, n] = val.mean()
            camera_radiance["std"][i, n] = val.std()

            cops_radiance[i, n] = float(dfcops.iloc[amin][dict_header[wl_cops[n]]]) / 100
            cops_radiance[i, n] *= immersion_factor[n]

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
    path_i360 = os.path.abspath(os.path.join(__file__, "../../.."))
    plt.style.use(path_i360 + "/figurestyle.mplstyle")

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

        print(f"{lambda_eff[n]} nm")
        # Linear regression
        res = linregress(cops_radiance[:, n], camera_radiance["avg"][:, n])

        # Pearson coefficient
        pearson_c_res = pearsonr(cops_radiance[:, n], camera_radiance["avg"][:, n])

        # Print Results
        print(res.rvalue ** 2)
        ts = tinv(0.05,  len(cops_radiance[:, n]) - 2)
        print(f"slope (95%): {res.slope:.6f} +/- {ts * res.stderr:.6f}")
        print(f"intercept (95%): {res.intercept:.6f} +/- {ts * res.intercept_stderr:.6f}")

        # RMSE
        ypred = res.intercept + res.slope * cops_radiance[:, n]
        rootmeanse = rmse(camera_radiance["avg"][:, n], ypred)
        rootmease_one_one = rmse(camera_radiance["avg"][:, n], cops_radiance[:, n])

        # Compare RMSE to uncertainties
        print("RMSE compared to uncertainties")
        print(f"RMSE 1:1 - {rootmease_one_one:.7f}")
        print(f"RMSE: {rootmeanse:.7f}")
        print("Min, max, mean, median uncertainties: ({0:.7f}, {1:.7f}, {2:.7f}, {3:.7f})".format(camera_radiance["std"][:, n].min(),
                                                                  camera_radiance["std"][:, n].max(),
                                                                  camera_radiance["std"][:, n].mean(),
                                                                  np.median(camera_radiance["std"][:, n])))

        # Text results
        txt_eq = "$\overline{L}_{cam} = m \cdot L_{COPS} + b$\n"
        txt_reg = "$m = ({0:.1f} \pm {1:.1f})$\n".format(res.slope, ts * res.stderr) + "$b = ({0:.2f} \pm {1:.2f})$\n".format(res.intercept, ts * res.intercept_stderr)
        txt_res = txt_eq + txt_reg + "$R^2=${0:.3f}\n$r=${1:.3f}\n$RMSE=${2:.3f}\n".format(res.rvalue ** 2, pearson_c_res.statistic, rootmeanse) + txt_mupd.format(MUPD[n])

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
    time_list_cops_rad = []
    #time_interval_mins = 4
    #each_n_frame = int(time_interval_mins * 60 * 6)
    #radiance_spectra_cops = np.zeros((dfcops.shape[0] // each_n_frame, 19))
    #wave_cops_array = np.zeros((dfcops.shape[0] // each_n_frame, 19))
    radiance_spectra_cops = np.zeros((amin_array.shape[0]//2, 19))
    wave_cops_array = radiance_spectra_cops.copy()
    #for i in range(radiance_spectra_cops.shape[0]):
    #    print(i*each_n_frame)
    #    for j, k in enumerate(dict_header.keys()):
    #        wave_cops_array[i, j] = k
    #        radiance_spectra_cops[i, j] = dfcops.iloc[i*each_n_frame][dict_header[k]] / 100
    #    time_list_cops_rad.append(dfcops.iloc[i*each_n_frame]["DateTimeUTC"].strftime('%H:%M:%S'))
    for i in range(radiance_spectra_cops.shape[0]):
        print(amin_array[2*i])
        for j, k in enumerate(dict_header.keys()):
            wave_cops_array[i, j] = k
            radiance_spectra_cops[i, j] = dfcops.iloc[int(amin_array[2*i])][dict_header[k]] / 100
        time_list_cops_rad.append(dfcops.iloc[int(amin_array[2*i])]["DateTimeUTC"].strftime('%H:%M:%S'))
    #spectra = np.zeros((19, 400))
    #rad_spectra = spectra.copy()
    #for i in range(spectra.shape[1]):
    #    for j, k in enumerate(dict_header.keys()):
    #        spectra[j, i] = k
    #        rad_spectra[j, i] = dfcops.iloc[i][dict_header[k]] / 100

    #spectra = np.zeros((19, dfcops.shape[0]))
    #rad_spectra = spectra.copy()
    #for i in range(spectra.shape[1]):
    #    for j, k in enumerate(dict_header.keys()):
    #        spectra[j, i] = k
    #        rad_spectra[j, i] = dfcops.iloc[i][dict_header[k]] / 100

    # Figure 2
    ax2 = [fig1.add_subplot(gs[0, :-4], sharey=ax1[0]), fig1.add_subplot(gs[1, :-4], sharey=ax1[1]), fig1.add_subplot(gs[2, :-4], sharey=ax1[2])]
    ax2[1].get_shared_x_axes().join(ax2[0], ax2[1])
    ax2[2].get_shared_x_axes().join(ax2[1], ax2[2])
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
        cops_radi = dfcops[dict_header[wl_cops[i]]] / 100
        cops_radi *= immersion_factor[i]
        a.plot(dfcops["DateTimeUTC"], cops_radi, label="C-OPS, $\lambda={0} nm$".format(wl_cops[i]), color="k")
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

    #ax2[2].set_xlabel("Time of day on {0}".format(dfcops["DateTimeUTC"][0].strftime("%d %B %Y")))
    ax2[2].set_xlabel("Local time of measurement")
    #ax2[0].set_xticklabels([])
    #ax2[1].set_xticklabels([])

    # Figure 3
    fig3, ax3 = plt.subplots(1, 1, figsize=ff.set_size(subplots=(1, 1)))
    vir_cmap = plt.get_cmap('viridis')
    colors_vir_rad_spectrum = vir_cmap(np.linspace(0, 1, radiance_spectra_cops.shape[0]))
    for n in range(radiance_spectra_cops.shape[0]):
        rscops_norm = radiance_spectra_cops[n, :] / radiance_spectra_cops[n, :][2]
        #rscops_norm = radiance_spectra_cops[n, :] / np.sum(radiance_spectra_cops[n, :])
        ax3.plot(wave_cops_array[n, :], rscops_norm, marker='.', color=colors_vir_rad_spectrum[n, :], label=time_list_cops_rad[n])
    #ax3.plot(spectra[:, 0], rad_spectra.mean(axis=1), linestyle="-", marker=".")
    #ax3.fill_between(spectra[:, 0], rad_spectra.mean(axis=1)-rad_spectra.std(axis=1), rad_spectra.mean(axis=1)+rad_spectra.std(axis=1), color="gray", alpha=0.8)
    ax3.legend(loc='best', ncol=2)
    ax3.set_xlabel("Wavelength [nm]")
    ax3.set_ylabel("$L_{COPS}(\lambda)/L_{COPS}(\mathrm{412 nm})$ " + " [-]")
    #ax3.set_ylabel("$L_{COPS}(\lambda)/L_{COPS}(412 nm)$ " + " [$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")

    fig3.tight_layout()
    #fig1.tight_layout()
    fig1.savefig("figures/validation_sky.pdf", format="pdf", dpi=600)
    fig1.savefig("figures/validation_sky.png", format="png", dpi=600)
    fig1.savefig("figures/validation_sky.jpeg", format="jpeg", dpi=600)
    fig3.savefig("figures/Fig.S4.pdf", format="pdf", dpi=600)
    fig3.savefig("figures/Fig.S4.jpeg", format="jpeg", dpi=600)

    plt.show()
