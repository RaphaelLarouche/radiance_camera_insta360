import matplotlib.pyplot as plt
import pandas
import numpy as np

from HE60PY.ac9simulation import AC9Simulation
from HE60PY.data_manager import DataViewer


if __name__ == "__main__":
    path_to_user_files = r'/Users/braulier/Documents/HE60/run'
    oden_simulation = AC9Simulation(path=path_to_user_files,
                                                 run_title='low_iop_test',
                                                 root_name='low_iop_test',
                                                 mode='HE60DORT',
                                                wavelength_list=[484, 544, 602])

    oden_simulation.set_z_grid(z_max=3.00)
    oden_simulation.add_layer(z1=0.0, z2=0.20, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=2277, dpf='dpf_OTHG_0_98.txt')  # 2277 # bb arg is not relevent since we use a discretized phase function in a file indepêdnant of depth (g=0.98)
    oden_simulation.add_layer(z1=0.20, z2=0.80, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=302, dpf='dpf_OTHG_0_98.txt')  # 303
    oden_simulation.add_layer(z1=0.80, z2=2.00, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=79, dpf='dpf_OTHG_0_98.txt')  # 79
    oden_simulation.add_layer(z1=2.0, z2=3.01, abs={'484': 1.36e-2, '544': 5.11e-2, '602': 2.224e-1}, scat=0.1, dpf='dpf_OTHG_0_98.txt')  # 0.1
    oden_simulation.run_simulation(printoutput=True)
    oden_simulation.parse_results()

    analyze=





