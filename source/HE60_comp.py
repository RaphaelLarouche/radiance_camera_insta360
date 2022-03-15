from processing import ProcessImage, ProcessImage
import radiance as r
from geometric_rolloff import OpenMatlabFiles
from processing import ProcessImage, FigureFunctions
import pandas

from HE60PY.ac9simulation import AC9Simulation



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
    time_stamp, radiometer_wl, irr_incom, irr_below = open_TriOS_data()


    path_to_user_files = r'/Users/braulier/Documents/HE60/run'
    wavelength_abs_built_sim = AC9Simulation(path=path_to_user_files, run_title='He60_dort_comp', root_name='he60 comparison to dort', mode='sea_ice')
    wavelength_abs_built_sim.set_z_grid(z_max=2.0, wavelength_list=[484, 544, 602])
    wavelength_abs_built_sim.add_layer(z1=0.0, z2=0.10, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=2277, bb=0.0109)
    wavelength_abs_built_sim.add_layer(z1=0.10, z2=0.80, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=303, bb=0.0042)
    wavelength_abs_built_sim.add_layer(z1=0.80, z2=2.01, abs={'484': 0.0430, '544': 0.0683, '602': 0.12}, scat=79, bb=0.0042)
    wavelength_abs_built_sim.run_simulation(printoutput=True)

