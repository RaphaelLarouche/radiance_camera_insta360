from processing import ProcessImage, ProcessImage
import radiance as r
from source.geometric_rolloff import OpenMatlabFiles
from processing import ProcessImage, FigureFunctions
import pandas

from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer

from script_bastian_14_04_2022 import graph_cam_vs_HE60_simulations

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


# CoM Regions Science and Technology, 8 (1983) 119-127 119
# Elsevier Science Publishers B.V., Amsterdam - Printed in The Netherlands
# SCATTERING OF VISIBLE A N D NEAR I N F R A R E D R A D I A T I O N
# GLACIER ICE
if __name__ == "__main__":
    pf_list = ['dpf_OTHG2_g10_60_g2-0_55_b0_90.txt', 'dpf_OTHG2_g10_65_g2-0_15_b0_70.txt', 'dpf_OTHG2_g10_70_g2-0_25_b0_85.txt', 'dpf_OTHG2_g10_75_g2-0_15_b0_80.txt']
    for i, pf_ice in enumerate(pf_list):
        b_ssl, b_dl, b_il, = 500., 100., 30.
        a_red, a_green, a_blue =  10.0, 1.0, 0.1
        i = str(i)

        oden_simulation = SeaIceSimulation(run_title=f'He60_dort_G2_fit'+i, root_name=f'He60_dort_G2_fit'+i, mode='HE60DORT',)
        oden_simulation.set_z_grid(z_max=3.0, wavelength_list=[484, 544, 602])
        oden_simulation.add_layer(z1=0.0, z2=0.20, abs={'484': a_blue, '544': a_green, '602': a_red}, scat=b_ssl, dpf=pf_ice) # bb arg is not relevent since we use a discretized phase function in a file indepêdnant of depth (g=0.98)
        oden_simulation.add_layer(z1=0.20, z2=0.80, abs={'484': a_blue, '544': a_green, '602': a_red}, scat=b_dl, dpf=pf_ice)
        oden_simulation.add_layer(z1=0.80, z2=2.0, abs={'484': a_blue, '544': a_green, '602': a_red}, scat=b_il, dpf=pf_ice)
        oden_simulation.add_layer(z1=2.0, z2=3.01, abs={'484': 1.36e-2, '544': 5.11e-2, '602': 2.224e-1}, scat=0.1, dpf='dpf_OTHG_0_90.txt')
        oden_simulation.run_simulation(printoutput=True)
        oden_simulation.parse_results()
        oden_simulation.draw_figures()

        # graph_cam_vs_HE60_simulations(rad_oden, zen_oden, 'He60_dort_G2_fit'+i, ["20 cm (in water)",
        #                                                                                   "40 cm",
        #                                                                                   "60 cm",
        #                                                                                   "80 cm",
        #                                                                                   "100 cm",
        #                                                                                   "120 cm",
        #                                                                                   "140 cm",
        #                                                                                   "160 cm"], [0.20,
        #                                                                                               0.40,
        #                                                                                               0.60,
        #                                                                                               0.80,
        #                                                                                               1.00,
        #                                                                                               1.20,
        #                                                                                               1.41,
        #                                                                                               1.60])
        # plt.show()
