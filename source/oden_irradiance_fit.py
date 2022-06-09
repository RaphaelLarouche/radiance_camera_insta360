from comparator import HFComparator
from radclass import RadClass

from script_bastian_14_04_2022 import draw_radiance_figure

from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer
from HE60PY.phasefunctions import *

if __name__ == "__main__":
    # Simulation
    wavelength = 480
    a_ice_red, a_ice_green, a_ice_blue = 0.133, 0.065, 0.043
    a_sw_red, a_sw_green, a_sw_blue = 0.10, 0.05, 0.0475
    root_name = "oden_fit"
    frontiers = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 3.01]
    b = np.array([200, # 0-20 cm
                        350, # 20 - 40 cm
                        350, # 40 - 60 cm
                        350,  # 60 - 80 cm
                        250,  # 80 - 100 cm
                        110,  # 100 - 120 cm
                        110,  # 120 - 140 cm
                        80,  # 140 - 160 cm
                        80,  # 160 - 180 cm
                        150,  # 180 - 200 cm
                        0.50]) # 200 - 300 cm

    pf = np.array([OTHG(0.85), # 0-20 cm
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

    oden_simulation = SeaIceSimulation(root_name=root_name, run_title=root_name, mode="Oden", wavelength_list=[wavelength, 540, 600])
    oden_simulation.set_z_grid(z_max=3.0)
    for i, top in enumerate(frontiers[:-1]):
        bot = frontiers[i + 1]
        if top < 2.00: # We are still in ice
            oden_simulation.add_layer(z1=top, z2=bot, abs={'480': a_ice_blue, '540': a_ice_green, '600': a_ice_red}, scat=b[i], dpf=pf[i])
        if top == 2.00: # Seawater layer
            oden_simulation.add_layer(z1=top, z2=bot, abs={'480': a_sw_blue, '540': a_sw_green, '600': a_sw_red}, scat=b[i], dpf=pf[i])
    oden_simulation.run_simulation(printoutput=True)
    oden_simulation.parse_results()

    comparison = HFComparator(RadClass=RadClass(data_path="oden-08312018.h5"),
                           HEDataViewer=DataViewer(root_name=root_name))
    comparison.compare_irradiances(wavelengths=[wavelength])
    plt.show()
    draw_radiance_figure(rootname=root_name)
    plt.show()