import matplotlib.pyplot as plt
import pandas
import numpy as np

from HE60PY.ac9simulation import AC9Simulation
from HE60PY.data_manager import DataFinder


if __name__ == "__main__":
    path_to_user_files = r'/Users/braulier/Documents/HE60/run'
    oden_simulation = AC9Simulation(path=path_to_user_files,
                                                 run_title='low_iop_test',
                                                 root_name='low_iop_test',
                                                 mode='HE60DORT',
                                    zetanom=np.linspace(0.0, 3.00, 301))
    oden_simulation.set_z_grid(z_max=3.0, wavelength_list=[484, 544, 602])
    oden_simulation.add_layer(z1=0.0, z2=0.10, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=2277, bb=0.0)  # 2277 # bb arg is not relevent since we use a discretized phase function in a file indepêdnant of depth (g=0.98)
    oden_simulation.add_layer(z1=0.10, z2=0.80, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=303, bb=0.0)  # 303
    oden_simulation.add_layer(z1=0.80, z2=2.00, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=79, bb=0.0)  # 79
    oden_simulation.add_layer(z1=2.0, z2=3.01, abs={'484': 1.36e-2, '544': 5.11e-2, '602': 2.224e-1}, scat=0.1, bb=0.0)  # 0.1
    oden_simulation.run_simulation(printoutput=True)

    oden_analysis = DataFinder(oden_simulation.hermes)
    results = oden_analysis.get_Eudos_lambda()
    results.to_csv(f'data/normal_HE.txt')

    HE_60_bands = ['600.0', '540.0', '480.0']
    depths_HE60 = results['depths']
    fig, axes = plt.subplots(1, 3)
    ax = axes.ravel()
    for i, band in enumerate(HE_60_bands):
        ax[i].plot(results[f'Ed_{band}'], depths_HE60, color='#0078b0', label='Ed')
        ax[i].plot(results[f'Eu_{band}'], depths_HE60, color='#00ad91', label='Eu')
        ax[i].plot(results[f'Eo_{band}'], depths_HE60, color='#c9bc00', label='Eo')
        ax[i].set_xscale("log")
        ax[i].set_ylim([0, 2.0])
        ax[i].invert_yaxis()
        ax[i].legend()
        ax[i].set_ylabel('Depth [m]')
        ax[i].set_xlabel("$E~[\mathrm{W \cdot m^{-2} \cdot nm^{-1}}]$")
        ax[i].set_title(f'{band} nm')
    plt.tight_layout()
    # plt.savefig(f'data/figures/normal_HE.png')
    plt.show()




