# -*- coding: utf-8 -*-
"""
Baie des Chaleurs by Bastien. Irradiance per station.
"""

# Module importation
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Other modules
# import bdc_all_irradiance_curves as bdc_irradiance_fct


# Function and classes
def cops_irradiances(path=r"data/cops/BRML_220323_1511_C_data_001.tsv", wl=np.array([589, 555, 490])):
    """

    :param path:
    :return:
    """
    cops_d = pd.read_csv(path, sep="\t", header=0, encoding="ISO-8859-1")

    # C-OPS header
    dict_header = {380: 'Ed0380 (µW/(cm² nm))',
                   395: 'Ed0395 (µW/(cm² nm))',
                   412: 'Ed0412 (µW/(cm² nm))',
                   443: 'Ed0443 (µW/(cm² nm))',
                   465: 'Ed0465 (µW/(cm² nm))',
                   490: 'Ed0490 (µW/(cm² nm))',
                   510: 'Ed0510 (µW/(cm² nm))',
                   532: 'Ed0532 (µW/(cm² nm))',
                   555: 'Ed0555 (µW/(cm² nm))',
                   560: 'Ed0560 (µW/(cm² nm))',
                   589: 'Ed0589 (µW/(cm² nm))',
                   625: 'Ed0625 (µW/(cm² nm))',
                   665: 'Ed0665 (µW/(cm² nm))',
                   683: 'Ed0683 (µW/(cm² nm))',
                   694: 'Ed0694 (µW/(cm² nm))',
                   710: 'Ed0710 (µW/(cm² nm))',
                   765: 'Ed0765 (µW/(cm² nm))',
                   780: 'Ed0780 (µW/(cm² nm))',
                   875: 'Ed0875 (µW/(cm² nm))'}

    cops_r = cops_d[dict_header[wl[0]]] / 100
    cops_g = cops_d[dict_header[wl[1]]] / 100
    cops_b = cops_d[dict_header[wl[2]]] / 100

    cops_datetime = pd.to_datetime(cops_d["DateTimeUTC"])
    cops_datetime = cops_datetime.dt.tz_localize("utc").dt.tz_convert("America/Montreal")

    return cops_datetime, (cops_r, cops_g, cops_b)


if __name__ == "__main__":

    # Load data
    data = bdc_irradiance_fct.load_dict_from_hdf5(filename="data/baiedeschaleurs-03232022.h5")

    # Loop
    station_list = ["station_1", "station_2", "station_3", "station_4"]

    time_stamp_stations = np.array([pd.Timestamp('2022-03-23T12:19', tz='America/Montreal'),
                                    pd.Timestamp('2022-03-23T12:51', tz='America/Montreal'),
                                    pd.Timestamp('2022-03-23T13:07', tz='America/Montreal'),
                                    pd.Timestamp('2022-03-23T13:36', tz='America/Montreal')])

    ed_zeroplus = np.zeros((len(station_list), 3))

    for i, st in enumerate(station_list):

        # Label
        label_st = list(bdc_irradiance_fct.create_label("data/{0}_data.txt".format(st)).keys())

        # Freeboard
        ifb_st = bdc_irradiance_fct.get_ice_freeboard("data/{0}_data.txt".format(st))

        # Irradiance
        ed_st, eu_st, eo_st = bdc_irradiance_fct.create_irradiance_data(data[st]["zenith"] * 180 / np.pi,
                                                                        data[st]["azimuth"] * 180 / np.pi,
                                                                        data[st], label_st, ifb_st)

        ed_zeroplus[i, :] = np.array([ed_st["r"][0], ed_st["g"][0], ed_st["b"][0]])

        # Figures
        fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(6.136, 3.784))

        ax[0].axhline(y=ifb_st, color="k", linestyle="--", label="Freeboard")
        ax[1].axhline(y=ifb_st, color="k", linestyle="--", label="Freeboard")
        ax[2].axhline(y=ifb_st, color="k", linestyle="--", label="Freeboard")

        ax[0].plot(ed_st["r"], ed_st["depth"], linewidth=0.8, color="#a6cee3", linestyle="-", label="$E_{d}$")
        ax[0].plot(eu_st["r"], eu_st["depth"], linewidth=0.8, color="#1f78b4", linestyle="-", label="$E_{u}$")
        ax[0].plot(eo_st["r"], eo_st["depth"], linewidth=0.8, color="#b2df8a", linestyle="-", label="$E_{0}$")

        ax[1].plot(ed_st["g"], ed_st["depth"], linewidth=0.8, color="#a6cee3", linestyle="-", label="$E_{d}$")
        ax[1].plot(eu_st["g"], eu_st["depth"], linewidth=0.8, color="#1f78b4", linestyle="-", label="$E_{u}$")
        ax[1].plot(eo_st["g"], eo_st["depth"], linewidth=0.8, color="#b2df8a", linestyle="-", label="$E_{0}$")

        ax[2].plot(ed_st["b"], ed_st["depth"], linewidth=0.8, color="#a6cee3", linestyle="-", label="$E_{d}$")
        ax[2].plot(eu_st["b"], eu_st["depth"], linewidth=0.8, color="#1f78b4", linestyle="-", label="$E_{u}$")
        ax[2].plot(eo_st["b"], eo_st["depth"], linewidth=0.8, color="#b2df8a", linestyle="-", label="$E_{0}$")

        ax[0].set_xscale("log")
        ax[0].invert_yaxis()

        ax[0].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
        ax[1].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
        ax[2].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")

        ax[0].legend(loc="best", frameon=False, fontsize=6)
        ax[1].legend(loc="best", frameon=False, fontsize=6)
        ax[2].legend(loc="best", frameon=False, fontsize=6)

        ax[0].set_ylabel("Depth [cm]")
        fig.suptitle(st)
        fig.tight_layout()

    # C-OPS data
    c_datetime_001, cops_ed_001 = cops_irradiances(path=r"data/cops/BRML_220323_1511_C_data_001.tsv")
    c_datetime_002, cops_ed_002 = cops_irradiances(path=r"data/cops/BRML_220323_1511_C_data_002.tsv")
    c_datetime_003, cops_ed_003 = cops_irradiances(path=r"data/cops/BRML_220323_1511_C_data_003.tsv")
    c_datetime_004, cops_ed_004 = cops_irradiances(path=r"data/cops/BRML_220323_1511_C_data_004.tsv")
    c_datetime_005, cops_ed_005 = cops_irradiances(path=r"data/cops/BRML_220323_1511_C_data_005.tsv")

    cops_all_datetime = np.concatenate((c_datetime_001.to_numpy(),
                                         c_datetime_002.to_numpy(),
                                         c_datetime_003.to_numpy(),
                                         c_datetime_004.to_numpy(),
                                         c_datetime_005.to_numpy()))
    cops_ed_r = np.concatenate((cops_ed_001[0], cops_ed_002[0], cops_ed_003[0], cops_ed_004[0], cops_ed_005[0]))
    cops_ed_g = np.concatenate((cops_ed_001[1], cops_ed_002[1], cops_ed_003[1], cops_ed_004[1], cops_ed_005[1]))
    cops_ed_b = np.concatenate((cops_ed_001[2], cops_ed_002[2], cops_ed_003[2], cops_ed_004[2], cops_ed_005[2]))

    # Figure
    fig1, ax1 = plt.subplots(3, 1, sharex=True, figsize=(6.4, 6))

    ax1[0].plot(cops_all_datetime, cops_ed_r, linestyle="none", marker="o", markersize=4, label="C-OPS, $\lambda={0} nm$".format(589))
    ax1[1].plot(cops_all_datetime, cops_ed_g, linestyle="none", marker="o", markersize=4, label="C-OPS, $\lambda={0} nm$".format(555))
    ax1[2].plot(cops_all_datetime, cops_ed_b, linestyle="none", marker="o", markersize=4, label="C-OPS, $\lambda={0} nm$".format(490))

    ax1[0].plot(time_stamp_stations, ed_zeroplus[:, 0], linestyle="none", marker="s", markersize=4, markerfacecolor="none", label="Cam, $\lambda={0}$ nm".format(603))
    ax1[1].plot(time_stamp_stations, ed_zeroplus[:, 1], linestyle="none", marker="s", markersize=4, markerfacecolor="none", label="Cam, $\lambda={0}$ nm".format(544))
    ax1[2].plot(time_stamp_stations, ed_zeroplus[:, 2], linestyle="none", marker="s", markersize=4, markerfacecolor="none", label="Cam, $\lambda={0}$ nm".format(484))

    ax1[0].set_ylabel("$E_{d}$ [$\mathrm{W \cdot m^{-2} \cdot nm^{-1}}$]")
    ax1[1].set_ylabel("$E_{d}$ [$\mathrm{W \cdot m^{-2} \cdot nm^{-1}}$]")
    ax1[2].set_ylabel("$E_{d}$ [$\mathrm{W \cdot m^{-2} \cdot nm^{-1}}$]")

    ax1[0].legend(loc="best")
    ax1[1].legend(loc="best")
    ax1[2].legend(loc="best")

    myFmt = mdates.DateFormatter('%H:%M:%S', tz=cops_all_datetime[0].tz)
    ax1[2].xaxis.set_major_formatter(myFmt)
    ax1[2].tick_params(axis='x', labelrotation=30)
    ax1[2].set_xlabel("Time of day on {0}".format(cops_all_datetime[0].strftime("%d %B %Y")))

    fig1.tight_layout()

    plt.show()
