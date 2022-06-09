from comparator import Comparator
from radclass import RadClass

from script_bastian_14_04_2022 import draw_radiance_figure

from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer
from HE60PY.phasefunctions import *

if __name__ == "__main__":
    # Simulation
    wavelength = 540
    a_ice_red, a_ice_green, a_ice_blue = 0.133, 0.065, 0.043
    a_sw_red, a_sw_green, a_sw_blue = 0.10, 0.05, 0.0475
    root_name = "bdc_1_fit"
    frontiers = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
    b = np.array([1000, # 0-20 cm
                        350, # 20 - 40 cm
                        350, # 40 - 60 cm
                        350,  # 60 - 80 cm
                        250]) # 80 - 100 cm

    pf = np.array([OTHG(0.85), # 0-20 cm
                   OTHG(0.99), # 20 - 40 cm
                   OTHG(0.99), # 40 - 60 cm
                   OTHG(0.99),  # 60 - 80 cm
                   OTHG(0.99)])  # 80 - 100 cm

    oden_simulation = SeaIceSimulation(root_name=root_name, run_title=root_name, mode="BaieDesChaleurs", wavelength_list=[wavelength], IrradDataFile="HE60BDC_irrad_cops_station_1")
    oden_simulation.set_z_grid(z_max=1.00)
    for i, top in enumerate(frontiers[:-1]):
        bot = frontiers[i + 1]
        if top < 0.80: # We are still in ice
            oden_simulation.add_layer(z1=top, z2=bot, abs={'480': a_ice_blue, '540': a_ice_green, '600': a_ice_red}, scat=b[i], dpf=pf[i])
        if top == 0.80: # Seawater layer
            oden_simulation.add_layer(z1=top, z2=bot, abs={'480': a_sw_blue, '540': a_sw_green, '600': a_sw_red}, scat=b[i], dpf=pf[i])
    oden_simulation.run_simulation(printoutput=True)
    oden_simulation.parse_results()

    comparison = Comparator(RadClass=RadClass(data_path="baiedeschaleurs-03232022.h5", station="station_1"),
                           HEDataViewer=DataViewer(root_name=root_name))
    comparison.compare_irradiances(wavelengths=[wavelength])
    plt.show()
    # draw_radiance_figure(rootname=root_name)
    # plt.show()