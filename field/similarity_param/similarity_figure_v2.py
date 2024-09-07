

# Module importation
import pandas
import os
import string
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from source.radiance import RadClass
import field.rgb_color_profile as sea_ice_color
from field.baiedeschaleurs2022.bdc_process_stations import get_ice_freeboard


def calculate_s(a, bprime):
    """

    :param a:
    :param b:
    :param g:
    :return:
    """
    return (1 + (bprime/a)) ** (-1/2)


def create_figure():
    """

    :return:
    :rtype:
    """
    f = plt.figure(figsize=(6.6929 * 0.8, 6.6929 * 0.8 * 1 / 1.618))
    gs0 = gridspec.GridSpec(1, 6, figure=f)
    a = []
    a.append(f.add_subplot(gs0[:-2]))
    a.append(f.add_subplot(gs0[-2]))
    a.append(f.add_subplot(gs0[-1]))

    return f, a


def similarity_fig_v2(data_dct1, data_dct2):
    """

    :param data:
    :type data:
    :return:
    :rtype:
    """

    # Fig params
    fig, ax = create_figure()

    ice_cond = {"FY blue ice": 0, "FY white ice": 1, "MY melting bare ice": 2, "FY snow covered ice": 3,
                "FY bare ice": 4, "MY snow covered ice": 5, "FY landfast ice": 6}
    ls_dict = {500: "-.", 540: "-", 633: "--"}

    cl_dct = {"SSL": "#d95f02", "DL": "#1b9e77", "II": "#7570b3"}
    auth = ["Ehn et al. 2008b", "Ehn et al. 2008b", "Light et al. 2008", 'Perron et al. 2021', 'Perron et al. 2021', "High Arctic site", "Chaleur Bay site"]

    for k in data_dct1.keys():

        color = cl_dct[data_dct1[k]['layer']]
        b_prime = data_dct1[k]['b_prime']
        la = k.split("_")[0]  # Layers name

        if len(b_prime) > 1:
            print("More than one b_prime value for this condition: ", k)

            min_s = calculate_s(data_dct1[k]["a"][0], b_prime[0])
            max_s = calculate_s(data_dct1[k]["a"][1], b_prime[1])

            if (data_dct1[k]["author"] == 'High Arctic site') or (data_dct1[k]["author"] == 'Chaleur Bay site'):

                maval = max([min_s, max_s])
                mival = min([min_s, max_s])

                ax[0].axhspan(mival, maval, color=color, alpha=0.4, zorder=1)
                ax[0].text(4.4, 0.5 * (mival + maval), data_dct1[k]["author"] + f" / {la}", fontsize=5)
            else:
                ax[0].bar(ice_cond[data_dct1[k]['cond']], bottom=max_s, height=(min_s - max_s),
                          linestyle=ls_dict[data_dct1[k]["wl"]], width=0.4,
                          alpha=1.0, color=color, edgecolor="k", zorder=2)
        else:
            if (data_dct1[k]["author"] == 'High Arctic site') or (data_dct1[k]["author"] == 'Chaleur Bay site'):
                print("Only one b_prime value for this condition: ", k)
                s = calculate_s(data_dct1[k]["a"][0], b_prime[0])
                ax[0].axhline(s, alpha=1.0, color=color, linewidth=0.8, linestyle="-", zorder=1)
                ax[0].text(4.4, s - 0.007, data_dct1[k]["author"] + f" / {la}", fontsize=5)

    for k in data_dct2.keys():

        color = cl_dct[data_dct2[k]['layer']]
        s = data_dct2[k]["s"]

        if len(s) > 1:
            print("More than one b_prime value for this condition: ", k)
            ax[0].bar(ice_cond[data_dct2[k]['cond']], bottom=s[0], height=(s[1] - s[0]), linestyle=ls_dict[data_dct2[k]["wl"]], width=0.4, alpha=1.0, color=color, edgecolor="k", zorder=2)

        else:
            print("Only one b_prime value for this condition: ", k)
            ax[0].bar(ice_cond[data_dct2[k]['cond']], bottom=s[0], height=0, width=0.4, alpha=1.0, color=color, linestyle=ls_dict[data_dct2[k]["wl"]], edgecolor=color, linewidth=2, zorder=2)

    ax[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color=cl_dct["SSL"], edgecolor="none", label="Surface scattering layer")
    ax[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color=cl_dct["DL"], edgecolor="none", label="Drained layer")
    ax[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color=cl_dct["II"], edgecolor="none", label="Interior ice")
    ax[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color="none", edgecolor="k", linestyle="-.", label="500 nm")
    ax[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color="none", edgecolor="k", linestyle="-", label="540 nm")

    xticktop = list(ice_cond.keys())
    xticktop[-1] = ""
    xticktop[-2] = ""

    xtickbot = auth
    xtickbot[-1] = ""
    xtickbot[-2] = ""

    ax_t = ax[0].secondary_xaxis('top')
    ax_t.set_xticks(list(ice_cond.values()), labels=xticktop, rotation=-15, fontsize=7, ha='right')

    ax[0].set_ylabel('Similarity parameter $S$')
    ax[0].set_ylim((0, 0.62))
    ax[0].set_xticks(list(ice_cond.values()), labels=xtickbot, rotation=-15, fontsize=7, ha='left')
    ax[0].legend(loc=3, fontsize=6)
    ax[0].invert_yaxis()

    return fig, ax


def show_ice_color(fig, ax):

    rc_oden = RadClass(data_path="../oden2018/data/oden-08312018-imf-fluo.h5")
    ifb_st2 = get_ice_freeboard("../baiedeschaleurs2022/data/station_2_data.txt")
    rc_st2 = RadClass(data_path="../baiedeschaleurs2022/data/baiedeschaleurs-03232022-imf-fluo.h5", station="station_2", data_type="camera", freeboard=ifb_st2)

    image_RGB_oden, oden_depths = sea_ice_color.process_rgb_images("../oden2018/jpeg_img", rc_oden, start=1)
    image_RGB_bdc, bdc_depths = sea_ice_color.process_rgb_images("../baiedeschaleurs2022/jpeg_img", rc_st2, start=0)


    ax[1].imshow(image_RGB_oden)
    ax[2].imshow(image_RGB_bdc)

    ax[1].yaxis.tick_right()
    ax[2].yaxis.tick_right()

    ax[1].set_xticks([])
    ax[1].set_yticks(np.arange(0,  oden_depths.shape[0], 1))
    ax[1].set_yticklabels(oden_depths.astype(int).astype(str).tolist())
    ax[1].set_ylabel("Depth [cm]")
    ax[1].set_title("High Arctic", fontsize=7)

    ax[2].set_xticks([])
    tickbdc = np.array([0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15])
    xticklabelbdc = ['5', '10', '15', '20', '25', '30', '40', '50', '60', '70', '80']
    ax[2].set_yticks(tickbdc)
    ax[2].set_yticklabels(xticklabelbdc)
    ax[2].set_ylabel("Depth [cm]")
    ax[2].set_title("Chaleur Bay", fontsize=7)

    ax[0].text(0.95, 1.1, "(" + string.ascii_lowercase[0] + ")", transform=ax[0].transAxes, size=10, weight='bold')
    ax[1].text(-0.0, 1.1, "(" + string.ascii_lowercase[1] + ")", transform=ax[1].transAxes, size=10, weight='bold')
    ax[2].text(-0.0, 1.1, "(" + string.ascii_lowercase[2] + ")", transform=ax[2].transAxes, size=10, weight='bold')

    return fig, ax


if __name__ == "__main__":

    plt.style.use(os.path.abspath(os.path.join(__file__, "../../..")) + "/figurestyle.mplstyle")
    wl = np.array([480, 540, 600])
    z_depth = np.arange(0, 220, 20)
    #z = 0.5 * (z_depth[1:] + z_depth[:-1])

    a_ha = np.ones((8, 3))
    a_ha[:, 0] *= 0.12
    a_ha[:, 1] *= 0.0683
    a_ha[:, 2] *= 0.043

    b_ha = np.array([4277.1, 410.44, 396.73, 291.53, 97.90, 48.40, 276.52, 0.89])
    b_ha = np.tile(b_ha[:, None], (1, 3))

    g_ha = np.array([0.85, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.90])
    g_ha = np.tile(g_ha[:, None], (1, 3))

    s_ha = calculate_s(a_ha, b_ha * (1 - g_ha))

    d_df = pandas.read_csv("../oden2018/data/super_recu_oden_fit_final/eudos_iops.csv")
    s_ha_v = calculate_s(d_df['a_540.0'], d_df['b_540.0'] * (1 - d_df['g']))


    #s_ha_b = np.array([0.0485, 0.1111, 0.1111, 0.1111, 0.1562, 0.2274, 0.2274, 0.2641, 0.2641, 0.1961])
    #s_ha_g = np.array([0.0550, 0.1261, 0.1261, 0.1261, 0.1769, 0.2565, 0.2565, 0.2971, 0.2971, 0.2216])
    #s_ha_r = np.array([0.0752, 0.1712, 0.1712, 0.1712, 0.2387, 0.3413, 0.3413, 0.3917, 0.3917, 0.2969])

    # --- First figure --- Reduced scattering coefficient
    data_dict = {'SSL_white_ice': {'layer': 'SSL', 'b_prime': [160], 'g': 0, 'author': 'Perovich 1990'},
                 'SSL_bare_ice':  {'layer': 'SSL', 'b_prime': [20, 150], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                 'DL_bare_ice_1': {'layer': 'DL', 'b_prime': [2.4, 12], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                 'DL_bare_ice_2': {'layer': 'DL', 'b_prime': [12.5], 'g': 0.95, 'a': [0.4], 'wl': 670, 'author': 'Mobley et al. 1998'},
                 'IL_cold_1': {'layer': 'II', 'b_prime': [3, 10], 'g': 0.5, 'author': 'Haines et al. 1997'},
                 'IL_cold_2': {'layer': 'II', 'b_prime': [4], 'g': 0.98, 'a': [0.4], 'wl': 670, 'author': 'Mobley et al. 1998'},
                 'IL_cold_3': {'layer': 'II', 'b_prime': [2.5], 'g': 0, 'author': 'Perovich 1990'},
                 'IL_cold_4': {'layer': 'II', 'b_prime': [1.5], 'g': 0.98, 'author': 'Pegau and Zaneveld 2000'},
                 'IL_cold_5': {'layer': 'II', 'b_prime': [0.5, 1.8], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                 'IL_snow_cover_1': {'layer': 'II', 'b_prime': [0.46, 4.4], 'a': [0.0683, 0.0683], 'wl': 540,  'author':"Perron et al. 2021"},
                 'IL_bare_cover_1': {'layer': 'II', 'b_prime': [0.15, 2.8], 'a': [0.0683, 0.0683], 'wl': 540,  'author':"Perron et al. 2021"},
                 'SSL_HA': {'layer': 'SSL', 'b_prime': [25.5], 'g': 0.85, 'a': [0.0775], 'wl': 540, 'author': "High Arctic site"},
                 'OII_HA': {'layer': 'II', 'b_prime': [2.4, 4.8], 'g': 0.99, 'a': [0.0775, 0.0775], 'wl': 540, 'author': "High Arctic site"},
                 'YII_HA': {'layer': 'II', 'b_prime': [0.8, 1.1], 'g': 0.99, 'a': [0.0775, 0.0775], 'wl': 540, 'author': "High Arctic site"},
                 'SSL_CB': {'layer': 'SSL', 'b_prime': [37.5], 'g': 0.85, 'a': [0.30], 'wl': 540, 'author': "Chaleur Bay site"},
                 'II_CB': {'layer': 'II', 'b_prime': [2.8, 6.4], 'g': 0.99, 'a': [0.30, 1.30], 'wl': 540, 'author': "Chaleur Bay site"}}

    # Ehn data
    data_dict_ehn_prior = {"SSL_blue_ice": {'cond': "FY blue ice", 'layer': 'DL', "s": [0.0355], "wl": 500, 'author': "Ehn et al. 2008b"},
                    "DL_blue_ice": {'cond': "FY blue ice", 'layer': 'II', "s": [0.0927, 0.1792], "wl": 500, 'author': "Ehn et al. 2008b"},
                    "SSL_white_ice": {'cond': "FY white ice", 'layer': 'SSL', "s": [0.0673], "wl": 500, 'author': "Ehn et al. 2008b"},
                    "DL_white_ice": {'cond': "FY white ice", 'layer': 'DL', "s": [0.0548], "wl": 500, 'author': "Ehn et al. 2008b"},
                    "IL_white_ice": {'cond': "FY white ice", 'layer': 'II', "s": [0.0914, 0.2768], "wl": 500, 'author': "Ehn et al. 2008b"}}

    data_dict_ehn = {"DL_blue_ice": {'cond': "FY blue ice", 'layer': 'DL', "s": [0.0355], "wl": 500, 'author': "Ehn et al. 2008b"},
                     "SSL_white_ice": {'cond': "FY white ice", 'layer': 'SSL', "s": [0.0673], "wl": 500, 'author': "Ehn et al. 2008b"},
                     "DL_white_ice": {'cond': "FY white ice", 'layer': 'DL', "s": [0.0548], "wl": 500, 'author': "Ehn et al. 2008b"},
                     "IL_white_ice": {'cond': "FY white ice", 'layer': 'II', "s": [0.0914, 0.2768], "wl": 500, 'author': "Ehn et al. 2008b"}}

    #
    data_dict_s_prior = {'SSL_bare_ice':  {'cond': "MY melting bare ice", 'layer': 'SSL', 'b_prime': [20, 150], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                   'DL_bare_ice_1': {'cond': "MY melting bare ice", 'layer': 'DL', 'b_prime': [2.4, 12], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                   'IL_cold_5': {'cond': "MY melting bare ice", 'layer': 'II', 'b_prime': [0.5, 1.8], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                   'IL_snow_cover_1': {'cond': "FY snow covered ice", 'layer': 'II', 'b_prime': [0.46, 4.4], 'a': [0.0683, 0.0683], 'wl': 540, 'author': "Perron et al. 2021"},
                   'IL_bare_cover_1': {'cond': "FY bare ice", 'layer': 'II', 'b_prime': [0.15, 2.8], 'a': [0.0683, 0.0683], 'wl': 540, 'author': "Perron et al. 2021"},
                   'SSL_HA': {'cond': "MY snow covered ice", 'layer': 'SSL', 'b_prime': [25.5], 'g': 0.85, 'a': [0.0775], 'wl': 540, 'author': "High Arctic site"},
                   'OII_HA': {'cond':"MY snow covered ice", 'layer': 'II', 'b_prime': [2.4, 4.8], 'g': 0.99, 'a': [0.0775, 0.0775], 'wl': 540, 'author': "High Arctic site"},
                   'YII_HA': {'cond': "MY snow covered ice", 'layer': 'II', 'b_prime': [0.8, 1.1], 'g': 0.99, 'a': [0.0775, 0.0775], 'wl': 540, 'author': "High Arctic site"},
                   'SSL_CB': {'cond': "FY landfast ice", 'layer': 'SSL', 'b_prime': [37.5], 'g': 0.85, 'a': [0.30], 'wl': 540, 'author': "Chaleur Bay site"},
                   'II_CB': {'cond': "FY landfast ice", 'layer': 'II', 'b_prime': [2.8, 6.4], 'g': 0.99, 'a': [0.30, 1.30], 'wl': 540, 'author': "Chaleur Bay site"}
                   }
    #

    data_dict_s = {'SSL_bare_ice':  {'cond': "MY melting bare ice", 'layer': 'SSL', 'b_prime': [20, 150], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                   'DL_bare_ice_1': {'cond': "MY melting bare ice", 'layer': 'DL', 'b_prime': [2.4, 12], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                   'IL_cold_5': {'cond': "MY melting bare ice", 'layer': 'II', 'b_prime': [0.5, 1.8], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                   'IL_snow_cover_1': {'cond': "FY snow covered ice", 'layer': 'II', 'b_prime': [0.15, 4.4], 'a': [0.0683, 0.0683], 'wl': 540, 'author': "Perron et al. 2021"},
                   'SSL_HA': {'cond': "MY snow covered ice", 'layer': 'SSL', 'b_prime': [641.57], 'g': 0.85, 'a': [0.0683], 'wl': 540, 'author': "High Arctic site"},
                   'DL_HA': {'cond': "MY snow covered ice", 'layer': 'DL', 'b_prime': [4.10], 'g': 0.99, 'a': [0.0683], 'wl': 540, 'author': "High Arctic site"},
                   'OII_HA': {'cond':"MY snow covered ice", 'layer': 'II', 'b_prime': [2.92, 3.96], 'g': 0.99, 'a': [0.0683, 0.0683], 'wl': 540, 'author': "High Arctic site"},
                   'YII_HA': {'cond': "MY snow covered ice", 'layer': 'II', 'b_prime': [0.48, 0.98], 'g': 0.99, 'a': [0.0683, 0.0683], 'wl': 540, 'author': "High Arctic site"},
                   'SSL_CB': {'cond': "FY landfast ice", 'layer': 'SSL', 'b_prime': [72.85], 'g': 0.85, 'a': [1.97], 'wl': 540, 'author': "Chaleur Bay site"},
                   'II_CB': {'cond': "FY landfast ice", 'layer': 'II', 'b_prime': [5.59, 7.79], 'g': 0.99, 'a': [1.80, 0.65], 'wl': 540, 'author': "Chaleur Bay site"}
                   }

    # Figure 1
    fig1, ax1 = create_figure()

    # Loop
    #cl_dct = {"SSL": "#a1dab4", "DL": "#41b6c4", "II": "#225ea8"}
    cl_dct = {"SSL": "#d95f02", "DL": "#1b9e77", "II": "#7570b3"}
    #ice_cond_bar = {"High Arctic site": 0, "Chaleur Bay site": 1 , "FY blue ice": 2, "FY white ice": 3, "MY melting bare ice": 4, "FY snow covered ice": 5,
    #                "FY bare ice": 6, "MY snow covered ice": 7, "FY landfast ice": 8}
    ice_cond_bar = {"MY snow covered ice": 0,  "FY landfast ice": 1, "FY blue ice": 2, "FY white ice": 2,
                    "MY melting bare ice": 3, "FY snow covered ice": 4}

    ls_dict = {500: "-.", 540: "-", 633: "--"}

    cycler_color = iter(matplotlib.rcParams["axes.prop_cycle"])

    for ke in data_dict_s:

        color = cl_dct[data_dict_s[ke]['layer']]
        b_prime = data_dict_s[ke]['b_prime']
        la = ke.split("_")[0]  # Layers name

        if len(b_prime) > 1:
            print("More than one b_prime value for this condition: ", ke)

            min_s = calculate_s(data_dict_s[ke]["a"][0], b_prime[0])
            max_s = calculate_s(data_dict_s[ke]["a"][1], b_prime[1])

            pos = ice_cond_bar[data_dict_s[ke]['cond']]
            if (data_dict_s[ke]["author"] == 'High Arctic site') or (data_dict_s[ke]["author"] == 'Chaleur Bay site'):
                #pos = ice_cond_bar[data_dict_s[ke]['author']]
                alph = 1.0
                ax1[0].axhspan(min_s, max_s, color=color, alpha=0.6, zorder=1, linewidth=0.5)
            else:

                alph = 1.0

            ax1[0].bar(pos, bottom=max_s, height=(min_s - max_s),
                       linestyle=ls_dict[data_dict_s[ke]["wl"]], width=0.4,
                       alpha=alph, color=color, edgecolor="k", zorder=2)
        else:
            s = calculate_s(data_dict_s[ke]["a"][0], b_prime[0])
            pos = ice_cond_bar[data_dict_s[ke]['cond']]
            if (data_dict_s[ke]["author"] == 'High Arctic site') or (data_dict_s[ke]["author"] == 'Chaleur Bay site'):
                #pos = ice_cond_bar[data_dict_s[ke]['author']]
                alph = 1.0
                ax1[0].axhline(s, alpha=0.6, color=color, linewidth=0.8, linestyle="-", zorder=1)
            else:
                #pos = ice_cond_bar[data_dict_s[ke]['cond']]
                alph = 1.0

            ax1[0].bar(pos, bottom=s, height=0, width=0.4, alpha=alph, color=color, linestyle=ls_dict[data_dict_s[ke]["wl"]], edgecolor=color, linewidth=2, zorder=2)

    for k in data_dict_ehn.keys():

        color = cl_dct[data_dict_ehn[k]['layer']]
        s = data_dict_ehn[k]["s"]

        alph=1.0

        if len(s) > 1:
            print("More than one b_prime value for this condition: ", k)
            ax1[0].bar(ice_cond_bar[data_dict_ehn[k]['cond']], bottom=s[0], height=(s[1] - s[0]), linestyle=ls_dict[data_dict_ehn[k]["wl"]], width=0.4, alpha=alph, color=color, edgecolor="k", zorder=2)

        else:
            print("Only one b_prime value for this condition: ", k)
            ax1[0].bar(ice_cond_bar[data_dict_ehn[k]['cond']], bottom=s[0], height=0, width=0.4, alpha=alph, color=color, linestyle=ls_dict[data_dict_ehn[k]["wl"]], edgecolor=color, linewidth=2, zorder=2)

    # Vertical line
    ax1[0].axvline(1.5, color="k", alpha=1.0, linestyle=":", linewidth=0.8)
    ax1[0].text(0.60, 1.05, "Literature values", fontsize=7, transform=ax1[0].transAxes)
    ax1[0].text(0.11, 1.05, "Inferred values", fontsize=7, transform=ax1[0].transAxes)


    # Dummy label
    ax1[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color=cl_dct["SSL"], edgecolor="none", label="Granular layer")
    ax1[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color=cl_dct["DL"], edgecolor="none", label="Drained layer")
    ax1[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color=cl_dct["II"], edgecolor="none", label="Interior ice")
    ax1[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color="none", edgecolor="k", linestyle="-.", label="500 nm")
    ax1[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color="none", edgecolor="k", linestyle="-", label="540 nm")

    authorss = ["High Arctic site", "Chaleur Bay site", "Ehn et al. 2008b", "Light et al. 2008", 'Perron et al. 2021']

    ax1[0].set_xticks(list(np.arange(5)), labels=authorss, rotation=-20, fontsize=7, ha='center')

    #inv_map_icond = {v: k for k, v in ice_cond_bar.items()}
    #xticktop = [inv_map_icond[i] for i in np.arange(0, len(ice_cond_bar))]
    #ax_t = ax1[0].secondary_xaxis('top')
    #ax_t.set_xticks(list(np.arange(5)), labels=xticktop, rotation=-20, fontsize=7, ha='center')

    ax1[0].set_ylabel('Similarity parameter $S$ [-]')
    ax1[0].set_ylim((0, 0.8))
    ax1[0].legend(loc=3, fontsize=6)
    ax1[0].invert_yaxis()

    show_ice_color(fig1, ax1)

    fig1.tight_layout()

    ## --- Second figure --- Similarity parameter histogram
    fig2, ax2 = similarity_fig_v2(data_dict_s_prior, data_dict_ehn_prior)

    ## --- Second figure --- Similarity parameter histogram
    fig2, ax2 = show_ice_color(fig2, ax2)

    fig2.tight_layout()


    # --- Third figure --- Similarity parameter

    fig3, ax3 = plt.subplots()

    z = np.array([1, 11, 38.5, 77.0, 117.0, 157.0, 181.0, 185.0])
    ax3.plot(s_ha_v[1:], d_df['depths'][1:], linestyle="-")
    #ax3.plot(s_ha_g, z, marker=".", linestyle="none")

    ax3.set_ylabel("Depth in ice [cm]")
    ax3.invert_yaxis()

    fig1.savefig("summary_fig_v3.pdf", format="pdf", dpi=300)
    fig1.savefig("summary_fig_v3.png", format="png", dpi=300)
    #fig2.savefig("summary_fig_v2.pdf", format="pdf", dpi=300)
    #fig2.savefig("summary_fig_v2.png", format="png", dpi=300)

    plt.show()

