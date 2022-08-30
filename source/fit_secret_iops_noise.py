from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer
from HE60PY.phasefunctions import *

from comparator import HHComparator
# from script_bastian_14_04_2022 import draw_radiance_figure

import numpy as np



if __name__ == "__main__":

    # b_s = np.array([500, # 0-20 cm
    #                     350, # 20 - 40 cm
    #                     350, # 40 - 60 cm
    #                     350,  # 60 - 80 cm
    #                     250,  # 80 - 100 cm
    #                     110,  # 100 - 120 cm
    #                     110,  # 120 - 140 cm
    #                     80,  # 140 - 160 cm
    #                     80,  # 160 - 180 cm
    #                     120,  # 180 - 200 cm
    #                     0.25]) # 200 - 300 cm
    #
    # a_red = np.array([0.300, # 0-20 cm
    #                     0.300, # 20 - 40 cm
    #                    0.150, # 40 - 60 cm
    #                     0.140,  # 60 - 80 cm
    #                     0.105,  # 80 - 100 cm
    #                     0.105,  # 100 - 120 cm
    #                     0.105,  # 120 - 140 cm
    #                     0.105,  # 140 - 160 cm
    #                     0.105,  # 160 - 180 cm
    #                     0.05,  # 180 - 200 cm
    #                     0.2]) # 200 - 300 cm
    #
    # a_green = np.array([0.175, # 0-20 cm
    #                     0.175, # 20 - 40 cm
    #                    0.175, # 40 - 60 cm
    #                     0.115,  # 60 - 80 cm
    #                     0.115,  # 80 - 100 cm
    #                     0.095,  # 100 - 120 cm
    #                     0.095,  # 120 - 140 cm
    #                     0.095,  # 140 - 160 cm
    #                     0.095,  # 160 - 180 cm
    #                     0.095,  # 180 - 200 cm
    #                     0.05]) # 200 - 300 cm
    #
    # a_blue = np.array([0.055, # 0-20 cm
    #                     0.085, # 20 - 40 cm
    #                    0.095, # 40 - 60 cm
    #                     0.095,  # 60 - 80 cm
    #                     0.095,  # 80 - 100 cm
    #                     0.095,  # 100 - 120 cm
    #                     0.095,  # 120 - 140 cm
    #                     0.095,  # 140 - 160 cm
    #                     0.095,  # 160 - 180 cm
    #                     0.095,  # 180 - 200 cm
    #                     0.01]) # 200 - 300 cm
    #
    # pf_ice = np.array([OTHG(0.85), # 0-20 cm
    #                OTHG(0.99), # 20 - 40 cm
    #                OTHG(0.99), # 40 - 60 cm
    #                OTHG(0.99),  # 60 - 80 cm
    #                OTHG(0.99),  # 80 - 100 cm
    #                OTHG(0.99),  # 100 - 120 cm
    #                OTHG(0.99),  # 120 - 140 cm
    #                OTHG(0.99), # 140 - 160 cm
    #                OTHG(0.99),  # 160 - 180 cm
    #                OTHG(0.99),  # 180 - 200 cm
    #                OTHG(0.90)]) # 200 - 300 cm
    root_name = "fit_secret_iops_noise"
    # HE_simulation = SeaIceSimulation(run_title=root_name, root_name=root_name, mode='Oden', wavelength_list=[480])
    # print(HE_simulation.wavelengths, HE_simulation.kwargs['bands'])
    # HE_simulation.set_z_grid(z_max=3.0)
    # for i, b  in enumerate(b_s):
    #     top, bot = i * 0.20, (i + 1) * 0.20
    #     if i < 10:
    #         HE_simulation.add_layer(z1=top, z2=bot, abs={'480': a_blue[i], '540': a_green[i], '600': a_red[i]}, scat=b, dpf=pf_ice[i]) # bb arg is not relevent since we use a discretized phase function in a file indepêdnant of depth (g=0.98)
    #     elif i == 10:
    #         HE_simulation.add_layer(z1=2.0, z2=3.01, abs={'480': a_blue[i], '540': a_green[i], '600': a_red[i]}, scat=b,
    #                                 dpf=pf_ice[i])
    # HE_simulation.run_simulation(printoutput=True)
    # HE_simulation.parse_results()
    # HE_simulation.draw_figures()
    #

    secret_iops = SeaIceSimulation(root_name=root_name, run_title=root_name, mode="Oden",
                                   wavelength_list=[480])
    secret_iops.set_z_grid(z_max=3.0)  #
    secret_iops.add_layer(z1=0.00, z2=0.10, abs={'480': 0.14, '540': 0.15, '600': 0.2}, scat=925, dpf=OTHG(0.90))
    secret_iops.add_layer(z1=0.10, z2=0.12, abs={'480': 0.15, '540': 0.15, '600': 0.2}, scat=800, dpf=OTHG(0.90))
    secret_iops.add_layer(z1=0.12, z2=0.20, abs={'480': 0.145, '540': 0.17, '600': 0.2}, scat=190, dpf=OTHG(0.95))
    secret_iops.add_layer(z1=0.20, z2=1.00, abs={'480': 0.14, '540': 0.17, '600': 0.22}, scat=70, dpf=OTHG(0.99))
    secret_iops.add_layer(z1=1.00, z2=1.80, abs={'480': 0.19, '540': 0.24, '600': 0.33}, scat=40, dpf=OTHG(0.99))
    secret_iops.add_layer(z1=1.80, z2=2.10, abs={'480': 0.27, '540': 0.25, '600': 0.2}, scat=100, dpf=OTHG(0.99))
    secret_iops.add_layer(z1=2.10, z2=3.01, abs={'480': 0.06, '540': 0.11, '600': 0.2}, scat=0.35, dpf=OTHG(0.90))
    # secret_iops.run_simulation(printoutput=True)
    # secret_iops.parse_results()
    # secret_iops.draw_figures()
    comp = HHComparator(SimulationDataviewer=DataViewer(root_name="secret_iops_prenoise"),
                           FieldDataViewer=DataViewer(root_name='secret_iops_noise'))
    axes = comp.compare_irradiances(wavelengths=[480])
    # [ax.set_ylim([0, 0.40]) for ax in axes]
    # [ax.invert_yaxis() for ax in axes]

    plt.show()