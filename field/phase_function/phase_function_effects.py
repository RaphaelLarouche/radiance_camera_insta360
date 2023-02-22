# -*- coding: utf-8 -*-
"""
PF comparisons from 2 december 2022
"""
# Module importation
import os
import numpy as np
import matplotlib.pyplot as plt

import field.oden2018.oden_dort_vs_hl as pf_cmpt
from source.radiance import RadClass

if __name__ == "__main__":

    rc = RadClass(data_path="../oden2018/data/oden-08312018-fluo.h5")

    d = os.listdir("data")

    for p in d:
        zd = pf_cmpt.load_zenith_radiance(path="data/" + p)

        fig, ax, _, _ = pf_cmpt.graph_radiance_cam_vs_simulations(rc, zd, np.arange(20, 180, 20).astype(float), sm=False)

        fig.suptitle(p)

    plt.show()