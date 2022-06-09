from processing import ProcessImage, ProcessImage
import radiance as r
from geometric_rolloff import OpenMatlabFiles
from processing import ProcessImage, FigureFunctions
import pandas

from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer
from HE60PY.phasefunctions import *

from script_bastian_14_04_2022 import draw_radiance_figure

import numpy as np

# Function and classes
def open_TriOS_data(min_date="2018-08-31 8:00", max_date="2018-08-31 16:00", path_trios="data/2018R4_300025060010720.mat"):
    """

    :param min_date:
    :param max_date:
    :param path_trios:
    :return:
    """
    openmatlab = OpenMatlabFiles()
    radiometer = openmatlab.loadmat(path_trios)
    radiometer = radiometer["raw"]

    timestamps = pandas.to_datetime(radiometer["irrad_t_1"] - 719529, unit='D')   # 719529 = 1 january 2000 Matlab
    df_time = pandas.DataFrame({"Date": timestamps.to_period('H')})
    mask = (df_time["Date"] <= max_date) & (df_time["Date"] >= min_date)
    inx = df_time["Date"][mask].index

    timestamps = timestamps.strftime("%B %d %Y, %H:%M:%S")
    irradiance_incom = radiometer["incom_int_1"][inx].T / 1000  # mW to W
    irradiance_below = radiometer["irrad_int_1"][inx].T / 1000  # mW to W

    return timestamps, radiometer["wl"], irradiance_incom, irradiance_below


if __name__ == "__main__":

    # b_s = np.array([2600, # 0-20 cm
    #                     525, # 20 - 40 cm
    #                    500, # 40 - 60 cm
    #                     550,  # 60 - 80 cm
    #                     275,  # 80 - 100 cm
    #                     135,  # 100 - 120 cm
    #                     80,  # 120 - 140 cm
    #                     80,  # 140 - 160 cm
    #                     50,  # 160 - 180 cm
    #                     120,  # 180 - 200 cm
    #                     1.0]) # 200 - 300 cm
    #
    # a_s = np.array([0.0435, # 0-20 cm
    #                     0.0435, # 20 - 40 cm
    #                    0.0435, # 40 - 60 cm
    #                     0.0435,  # 60 - 80 cm
    #                     0.0435,  # 80 - 100 cm
    #                     0.055,  # 100 - 120 cm
    #                     0.055,  # 120 - 140 cm
    #                     0.055,  # 140 - 160 cm
    #                     0.055,  # 160 - 180 cm
    #                     0.055,  # 180 - 200 cm
    #                     1.36e-2]) # 200 - 300 cm
    b_s = np.array([200, # 0-20 cm
                        350, # 20 - 40 cm
                        350, # 40 - 60 cm
                        350,  # 60 - 80 cm
                        250,  # 80 - 100 cm
                        110,  # 100 - 120 cm
                        110,  # 120 - 140 cm
                        80,  # 140 - 160 cm
                        80,  # 160 - 180 cm
                        120,  # 180 - 200 cm
                        0.25]) # 200 - 300 cm

    a_red = np.array([0.300, # 0-20 cm
                        0.300, # 20 - 40 cm
                       0.150, # 40 - 60 cm
                        0.140,  # 60 - 80 cm
                        0.105,  # 80 - 100 cm
                        0.105,  # 100 - 120 cm
                        0.105,  # 120 - 140 cm
                        0.105,  # 140 - 160 cm
                        0.105,  # 160 - 180 cm
                        0.05,  # 180 - 200 cm
                        0.2]) # 200 - 300 cm

    a_green = np.array([0.175, # 0-20 cm
                        0.175, # 20 - 40 cm
                       0.175, # 40 - 60 cm
                        0.115,  # 60 - 80 cm
                        0.115,  # 80 - 100 cm
                        0.095,  # 100 - 120 cm
                        0.095,  # 120 - 140 cm
                        0.095,  # 140 - 160 cm
                        0.095,  # 160 - 180 cm
                        0.095,  # 180 - 200 cm
                        0.05]) # 200 - 300 cm

    a_blue = np.array([0.055, # 0-20 cm
                        0.085, # 20 - 40 cm
                       0.095, # 40 - 60 cm
                        0.095,  # 60 - 80 cm
                        0.095,  # 80 - 100 cm
                        0.095,  # 100 - 120 cm
                        0.095,  # 120 - 140 cm
                        0.095,  # 140 - 160 cm
                        0.095,  # 160 - 180 cm
                        0.095,  # 180 - 200 cm
                        0.01]) # 200 - 300 cm

    a_red[0:10] = 0.133
    a_blue[0:10] = 0.04
    a_green[0:10] = 0.065
    pf_ice = np.array([OTHG(0.85), # 0-20 cm
                   OTHG(0.99), # 20 - 40 cm
                   OTHG(0.99), # 40 - 60 cm
                   OTHG(0.99),  # 60 - 80 cm
                   OTHG(0.99),  # 80 - 100 cm
                   OTHG(0.99),  # 100 - 120 cm
                   OTHG(0.99),  # 120 - 140 cm
                   OTHG(0.99), # 140 - 160 cm
                   OTHG(0.99),  # 160 - 180 cm
                   OTHG(0.99),  # 180 - 200 cm
                   OTHG(0.90)]) # 200 - 300 cm
    root_name = "manual_opt_no_interface"
    HE_simulation = SeaIceSimulation(run_title=root_name, root_name=root_name, mode='Oden', wavelength_list=[480, 540, 600])
    print(HE_simulation.wavelengths, HE_simulation.kwargs['bands'])
    HE_simulation.set_z_grid(z_max=3.0)
    for i, b  in enumerate(b_s):
        top, bot = i * 0.20, (i + 1) * 0.20
        if i < 10:
            HE_simulation.add_layer(z1=top, z2=bot, abs={'480': a_blue[i], '540': a_green[i], '600': a_red[i]}, scat=b, dpf=pf_ice[i]) # bb arg is not relevent since we use a discretized phase function in a file indepêdnant of depth (g=0.98)
        elif i == 10:
            HE_simulation.add_layer(z1=2.0, z2=3.01, abs={'480': a_blue[i], '540': a_green[i], '600': a_red[i]}, scat=b,
                                    dpf=pf_ice[i])
    HE_simulation.run_simulation(printoutput=True)
    HE_simulation.parse_results()
    HE_simulation.draw_figures()
    HE_view = DataViewer(root_name=root_name)
    # HE_view.draw_zenith_radiance_maps()
    # HE_view.draw_Eudos_profiles()
    # plt.show()
    draw_radiance_figure(rootname=root_name)
    plt.show()

    # root_name = "old_optimization"
    # pf_ice = 'dpf_OTHG_0_98.txt'
    #
    # oden_simulation = SeaIceSimulation(
    #                                              run_title=root_name,
    #                                              root_name=root_name,
    #                                              mode='HE60DORT')
    # oden_simulation.set_z_grid(z_max=3.00, wavelength_list=[484, 544, 602])
    # oden_simulation.add_layer(z1=0.0, z2=0.20, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=2277, dpf=pf_ice)  # 2277 # bb arg is not relevent since we use a discretized phase function in a file indepêdnant of depth (g=0.98)
    # oden_simulation.add_layer(z1=0.20, z2=0.80, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=303,dpf=pf_ice)  # 303
    # oden_simulation.add_layer(z1=0.80, z2=2.00, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=79, dpf=pf_ice)  # 79
    # oden_simulation.add_layer(z1=2.0, z2=3.01, abs={'484': 1.36e-2, '544': 5.11e-2, '602': 2.224e-1}, scat=0.1, dpf='dpf_OTHG_0_90.txt')  # 0.1
    # oden_simulation.run_simulation(printoutput=True)
    # oden_simulation.parse_results()
    # # oden_simulation.draw_figures()
    # draw_radiance_figure(rootname=root_name)
