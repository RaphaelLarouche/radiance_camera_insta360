"""

"""

# Module importation
import h5py
import glob
import string
import datetime
import pandas
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Other modules
from source.processing import ProcessImage, FigureFunctions
from source.radiance import ImageRadiancei360


# Functions
def UPD(X, Y):
    """

    :param X:
    :param Y:
    :return:
    """
    return 2 * (X - Y) / (X + Y)


if __name__ == "__main__":

    # Path to data
    pp = ProcessImage()
    ff = FigureFunctions()

    abs_path = pp.folder_choice() + "data-i360-tests/calibrations/absolute-radiance/validation/03182021"  # Choose folder in which data are
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

    wl_cops = [625, 532, 490]

    # Convert datetime to local datetime
    dfcops["DateTimeUTC"] = pandas.to_datetime(dfcops["DateTimeUTC"])
    dfcops["DateTimeUTC"] = dfcops["DateTimeUTC"].dt.tz_localize("utc").dt.tz_convert("America/Montreal")

    # Geometric parameters
    mask_deg = 7  # 7 degrees corresponding to C-OPS FOV

    # Loop
    plt.style.use("../../figurestyle.mplstyle")

    fig1, ax1 = plt.subplots(1, 3, sharey=True, sharex=True)
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

        for n, a in enumerate(ax1):
            maskzenith = zenith[:, :, n] <= mask_deg
            val = radiance_image[:, :, n][maskzenith]

            # Normalization

            camera_radiance["avg"][i, n] = val.mean()
            camera_radiance["std"][i, n] = val.std()
            cops_radiance[i, n] = float(dfcops.iloc[amin][dict_header[wl_cops[n]]]) / 100

    upd = np.abs((UPD(cops_radiance, camera_radiance['avg'])))
    MUPD = np.mean(upd, axis=0) * 100
    print(MUPD)

    # Figure 1
    lambda_eff = [600, 542, 486]
    lambda_cops = [589, 532, 490]
    txtstr = "$\lambda_{{cam, eff}}={0} nm$\n$\lambda_{{COPS}}={1} nm$\n$\mathrm{{MUPD}}={2:.1f}$%"

    for n, a in enumerate(ax1):

        one_one = np.linspace(cops_radiance[:, n].min() * 0.5, cops_radiance[:, n].max() * 1.5, 1000)
        a.plot(one_one, one_one, "k--", label="1:1")
        a.scatter(cops_radiance[:, n], camera_radiance["avg"][:, n], s=5, label="data")

        a.text(0.01, 0.2, txtstr.format(lambda_eff[n], lambda_cops[n], MUPD[n]), fontsize=8)

        a.set_ylabel("$L_{cam}$ " + "[$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")
        a.set_xlabel("$L_{COPS}$ " + "[$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")

        a.legend(loc=4)

    # Figure 2
    fig2, ax2 = plt.subplots(3, 1, sharey=True, sharex=True, figsize=ff.set_size())

    for i, a in enumerate(ax2):

        a.plot(dfcops["DateTimeUTC"], dfcops[dict_header[wl_cops[i]]]/100, label="C-OPS")
        a.plot(timecam, camera_radiance["avg"][:, i], linestyle="-.", marker="o", markersize=3, markerfacecolor="none", label="Camera")

        #a.set_ylabel("$L$ " + "[$\mathrm{W \cdot sr^{-1} \cdot m^{-2} \cdot nm^{-1}}$]")
        a.set_ylabel("Spectral radiance")
        myFmt = mdates.DateFormatter('%H:%M:%S', tz=dfcops["DateTimeUTC"][0].tz)
        a.xaxis.set_major_formatter(myFmt)
        a.tick_params(axis='x', labelrotation=45)
        a.text(0.01, 0.87, "(" + string.ascii_lowercase[i] + ")", transform=a.transAxes, size=11, weight='bold')

    ax2[0].text(0.5, 0.5, dfcops["DateTimeUTC"][0].strftime("%d %B %Y"), transform=ax2[0].transAxes, size=9)
    ax2[0].legend(loc="best")
    fig2.tight_layout()

    fig1.tight_layout()
    #fig1.savefig("../figures/validation_sky.pdf", format="pdf", dpi=600)
    plt.show()
