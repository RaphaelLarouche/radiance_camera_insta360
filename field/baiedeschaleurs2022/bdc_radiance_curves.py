# -*- coding: utf-8 -*-
"""
Baie des Chaleurs by Bastien. Radiance profiles.
"""

# Module importation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

# Other modules
from bdc_process_stations import create_label
from bdc_all_irradiance_curves import azimuthal_average, load_dict_from_hdf5, recursively_load_dict_contents_from_group


# Classes and functions
def colorbardepth(f, a, cmapp, valdum):
    """
    Function to add colorbar to the profile figure.

    :param f:
    :param a:
    :param cmapp:
    :param diction:
    :return:
    """

    dcax = a.scatter(valdum, valdum, c=np.arange(1, cmapp.N + 1), cmap=cmapp)

    a.cla()
    cb = f.colorbar(dcax, ax=a, orientation="vertical")

    cb.ax.locator_params(nbins=cmapp.N)
    cb.ax.set_yticklabels(["{0:.0f}".format(j) for j in valdum])
    cb.ax.set_title("depth [cm]", fontsize=8)
    cb.ax.invert_yaxis()

    return f, a


def build_cmap_2cond_color(cmap_name, d):
    """
    Building colormap for profile radiance zenith distribution.

    :param cmap_name:
    :return:
    """

    CMA = matplotlib.cm.get_cmap(cmap_name, len(d) + 1)
    colooor = CMA(np.arange(1, CMA.N))
    custom_cmap = matplotlib.colors.ListedColormap(colooor[::-1])
    return custom_cmap


def graph_radiance_curves(radiances, zenith, depth_keys_ordered):
    """

    :param radiances:
    :param zenith:
    :param depth_keys_ordered:
    :return:
    """
    # Figure
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(6.4 , 2.808))

    # Build colorbar
    colo_reds = build_cmap_2cond_color("Reds", depth_keys_ordered)
    colo_greens = build_cmap_2cond_color("Greens", depth_keys_ordered)
    colo_blues = build_cmap_2cond_color("Blues", depth_keys_ordered)

    cm_it_r = iter(colo_reds(np.arange(0, colo_reds.N)))
    cm_it_g = iter(colo_greens(np.arange(0, colo_greens.N)))
    cm_it_b = iter(colo_blues(np.arange(0, colo_blues.N)))

    colorbardepth(fig, ax[0], colo_reds, depth_keys_ordered)
    colorbardepth(fig, ax[1], colo_greens, depth_keys_ordered)
    colorbardepth(fig, ax[2], colo_blues, depth_keys_ordered)

    dc_color = {0: cm_it_r, 1: cm_it_g, 2: cm_it_b}

    for i, cam_k in enumerate(depth_keys_ordered):

        # Depth key
        de = "{0} cm".format(cam_k)
        rad_cam = radiances[de]

        # Azimuthal average
        rad_cam_avg = azimuthal_average(rad_cam)

        for b in range(rad_cam.shape[2]):

            current_col = next(dc_color[b])

            ax[b].plot(zenith[:, 0], rad_cam_avg[:, b], linestyle="-", color=current_col)

            ax[b].set_yscale("log")
            ax[b].set_xlim((20, 160))
            ax[b].set_yscale("log")
            ax[b].set_xlabel("Zenith [˚]")

    ax[0].set_ylabel(r"$\overline{L}$ [$\mathrm{{W \cdot m^{{-2}}  \cdot sr^{{-1}}\cdot nm^{{-1}}}}$]")
    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":

    # Load data
    data = load_dict_from_hdf5(filename="data/baiedeschaleurs-03232022-imf-fluo.h5")

    label_st1 = list(create_label("data/station_1_data.txt").keys())
    label_st2 = list(create_label("data/station_2_data.txt").keys())
    label_st3 = list(create_label("data/station_3_data.txt").keys())
    label_st4 = list(create_label("data/station_4_data.txt").keys())

    fig3, ax3 = graph_radiance_curves(data["station_1"], data["station_1"]["zenith"] * 180/np.pi, label_st1[4:])
    fig4, ax4 = graph_radiance_curves(data["station_2"], data["station_2"]["zenith"] * 180 / np.pi, label_st2[4:])
    fig5, ax5 = graph_radiance_curves(data["station_3"], data["station_3"]["zenith"] * 180 / np.pi, label_st3[4:])
    fig6, ax6 = graph_radiance_curves(data["station_4"], data["station_4"]["zenith"] * 180 / np.pi, label_st4[4:])

    fig3.savefig("figures/radiance_station1.png", dpi=600, format="png")
    fig4.savefig("figures/radiance_station2.png", dpi=600, format="png")
    fig5.savefig("figures/radiance_station3.png", dpi=600, format="png")
    fig6.savefig("figures/radiance_station4.png", dpi=600, format="png")

    plt.show()
