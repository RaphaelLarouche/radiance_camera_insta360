import pandas as pd
import numpy as np
from source.outdated.bdc_irradiance_stations import *

from HE60PY import environmentbuilder




if __name__ == "__main__":
    # C-OPS data
    wavelengths = np.array([380, 395, 412, 443, 465, 490, 510, 532, 555, 560, 589, 625, 665, 683, 694, 710, 765, 780, 875])
    for i in range(5):
        index = i + 1
        path = f"data/BRML_220323_1511_C_data_00{index}.tsv"
        path_to_HE60_file = f"/Applications/HE60.app/Contents/data/HE60BDC_irrad_cops_station_{index}"
        df = cops_d = pd.read_csv(path, sep="\t", header=0, encoding="ISO-8859-1")
        df = df.to_numpy()
        ed_lambda = np.mean(df[:, 5:], axis=0)/100
        print(np.array((wavelengths, ed_lambda)).T.shape)
        environmentbuilder.create_irrad_file(np.array((wavelengths, ed_lambda)).T, path_to_HE60_file)



