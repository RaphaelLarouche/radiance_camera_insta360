# -*- coding: utf-8 -*-
"""
Baie des Chaleurs by Bastien. AOPs and IOPS per station.
"""

# Module importation
import numpy as np
import pandas
from scipy import interpolate
import matplotlib.pyplot as plt

from source.radiance import RadClass
from bdc_process_stations import get_ice_freeboard

if __name__ == "__main__":

    # Freeboard
    ifb_st1 = get_ice_freeboard("data/station_1_data.txt")
    ifb_st2 = get_ice_freeboard("data/station_2_data.txt")
    ifb_st3 = get_ice_freeboard("data/station_3_data.txt")
    ifb_st4 = get_ice_freeboard("data/station_4_data.txt")

    # Station 1
    rc_st1 = RadClass(data_path="data/baiedeschaleurs-03232022.h5", station="station_1", data_type="camera", freeboard=ifb_st1)
    rc_st1.show_absorption_coefficient()
    rc_st1.show_mean_cosines()

    # Station 2
    rc_st2 = RadClass(data_path="data/baiedeschaleurs-03232022.h5", station="station_2", data_type="camera", freeboard=ifb_st2)
    fa2, axa2 = rc_st2.show_absorption_coefficient()
    rc_st2.show_mean_cosines()

    # ****** Test == - increasing vertical resolution  ******
    enet = np.array(rc_st2.ed.tolist())[:, :-1] - np.array(rc_st2.eu.tolist())[:, :-1]
    eo = np.array(rc_st2.eo.tolist())[:, :-1]
    agersh = -np.gradient(enet, rc_st2.ed["depth"]/100, axis=0) * 1/eo

    eo_f = interpolate.interp1d(rc_st2.ed["depth"][4:], eo[4:, 2], kind="cubic")
    enet_f = interpolate.interp1d(rc_st2.ed["depth"][4:], enet[4:, 2], kind="cubic")

    new_z = np.arange(20, 91, 1)

    agersh_int = -np.gradient(enet_f(new_z), new_z/100, axis=0) * 1/eo_f(new_z)

    fig_test, ax_test = plt.subplots(1, 2, sharey=True)

    ax_test[0].plot(enet[4:, 2], rc_st2.ed["depth"][4:], linestyle="none", marker=".", color="k")
    ax_test[0].plot(enet_f(new_z), new_z, linestyle="--", color="k")
    ax_test[0].plot(eo[4:, 2], rc_st2.ed["depth"][4:], linestyle="none", marker=".",  color="r")
    ax_test[0].plot(eo_f(new_z), new_z, linestyle="--", color="r")

    ax_test[0].set_xscale("log")

    ax_test[1].plot(agersh[4:, 2], rc_st2.ed["depth"][4:])
    ax_test[1].plot(agersh_int, new_z, linestyle="--")
    ax_test[1].invert_yaxis()
    # ******     ******

    # Station 3
    rc_st3 = RadClass(data_path="data/baiedeschaleurs-03232022.h5", station="station_3", data_type="camera", freeboard=ifb_st3)
    rc_st3.show_absorption_coefficient()
    rc_st3.show_mean_cosines()

    # Station 4
    rc_st4 = RadClass(data_path="data/baiedeschaleurs-03232022.h5", station="station_4", data_type="camera", freeboard=ifb_st4)
    rc_st4.show_absorption_coefficient()
    rc_st4.show_mean_cosines()

    # Save absorption coefficient of station 2
    a_st2_df = pandas.DataFrame(rc_st2.mu_a.copy())
    a_st2_df = a_st2_df.set_index("depth")
    a_st2_df.to_csv("data/a_bdc_st2.csv")

    plt.show()
