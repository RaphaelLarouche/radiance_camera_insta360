from processing import ProcessImage, ProcessImage
import radiance as r
from source.geometric_rolloff import OpenMatlabFiles
from processing import ProcessImage, FigureFunctions
import pandas

from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer

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
    # process = ProcessImage()
    # ze_mesh, az_mesh, rad_profile = process.open_radiance_data(path="data/oden-08312018.h5")
    # print(az_mesh, '\n', ze_mesh)
    # time_stamp, radiometer_wl, irr_incom, irr_below = open_TriOS_data()
    # b_ssl, b_dl, b_il, = 2000., 400., 90.
    b_ssl, b_dl, b_il, = np.array([1e4, 1070., 281.])
    # pf_ice = 'dpf_OTHG_0_98.txt'
    pf_ice = "dpf_brine_1_96.txt"
    oden_simulation = SeaIceSimulation(run_title=f'He60_dort_oden_brine', root_name=f'he60_comp_dort_brine', mode='HE60DORT',)
    oden_simulation.set_z_grid(z_max=3.0, wavelength_list=[484, 544, 602])
    oden_simulation.add_layer(z1=0.0, z2=0.20, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=b_ssl, dpf=pf_ice) # bb arg is not relevent since we use a discretized phase function in a file indepêdnant of depth (g=0.98)
    oden_simulation.add_layer(z1=0.20, z2=0.80, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=b_dl, dpf=pf_ice)
    oden_simulation.add_layer(z1=0.80, z2=2.0, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=b_il, dpf=pf_ice)
    oden_simulation.add_layer(z1=2.0, z2=3.01, abs={'484': 1.36e-2, '544': 5.11e-2, '602': 2.224e-1}, scat=0.1, dpf='dpf_OTHG_0_90.txt')
    oden_simulation.run_simulation(printoutput=True)
    oden_simulation.parse_results()
    # oden_simulation.draw_figures()

    pf_ice = 'dpf_OTHG_0_98.txt'
    # pf_ice = "dpf_brine_1_96.txt"
    oden_simulation = SeaIceSimulation(run_title=f'He60_dort_oden_HG', root_name=f'he60_comp_dort_HG', mode='HE60DORT',)
    oden_simulation.set_z_grid(z_max=3.0, wavelength_list=[484, 544, 602])
    oden_simulation.add_layer(z1=0.0, z2=0.20, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=b_ssl, dpf=pf_ice) # bb arg is not relevent since we use a discretized phase function in a file indepêdnant of depth (g=0.98)
    oden_simulation.add_layer(z1=0.20, z2=0.80, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=b_dl, dpf=pf_ice)
    oden_simulation.add_layer(z1=0.80, z2=2.0, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=b_il, dpf=pf_ice)
    oden_simulation.add_layer(z1=2.0, z2=3.01, abs={'484': 1.36e-2, '544': 5.11e-2, '602': 2.224e-1}, scat=0.1, dpf='dpf_OTHG_0_90.txt')
    oden_simulation.run_simulation(printoutput=True)
    oden_simulation.parse_results()
