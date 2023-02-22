

# Module importation
import string
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from source.radiance import RadClass
import field.rgb_color_profile as sea_ice_color
from field.baiedeschaleurs2022.bdc_process_stations import get_ice_freeboard


# Functions
def calculate_s(a, bprime):
    """

    :param a:
    :param b:
    :param g:
    :return:
    """
    return (1 + (bprime/a)) ** (-1/2)


if __name__ == "__main__":

    plt.style.use("../../figurestyle.mplstyle")
    wl = np.array([480, 540, 600])
    z_depth = np.arange(0, 220, 20)
    z = 0.5 * (z_depth[1:] + z_depth[:-1])

    s_ha_b = np.array([0.0485, 0.1111, 0.1111, 0.1111, 0.1562, 0.2274, 0.2274, 0.2641, 0.2641, 0.1961])
    s_ha_g = np.array([0.0550, 0.1261, 0.1261, 0.1261, 0.1769, 0.2565, 0.2565, 0.2971, 0.2971, 0.2216])
    s_ha_r = np.array([0.0752, 0.1712, 0.1712, 0.1712, 0.2387, 0.3413, 0.3413, 0.3917, 0.3917, 0.2969])

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
    data_dict_ehn = {"SSL_blue_ice": {'cond': "FY blue ice", 'layer': 'DL', "s": [0.0355], "wl": 500, 'author': "Ehn et al. 2008"},
                     "DL_blue_ice": {'cond': "FY blue ice", 'layer': 'II', "s": [0.0927, 0.1792], "wl": 500, 'author': "Ehn et al. 2008"},
                     "SSL_white_ice": {'cond': "FY white ice", 'layer': 'SSL', "s": [0.0673], "wl": 500, 'author': "Ehn et al. 2008"},
                     "DL_white_ice": {'cond': "FY white ice", 'layer': 'DL', "s": [0.0548], "wl": 500, 'author': "Ehn et al. 2008"},
                     "IL_white_ice": {'cond': "FY white ice", 'layer': 'II', "s": [0.0914, 0.2768], "wl": 500, 'author': "Ehn et al. 2008"}}

    #
    data_dict_s = {'SSL_bare_ice':  {'cond': "Melting MY bare ice", 'layer': 'SSL', 'b_prime': [20, 150], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                   'DL_bare_ice_1': {'cond': "Melting MY bare ice", 'layer': 'DL', 'b_prime': [2.4, 12], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                   'IL_cold_5': {'cond': "Melting MY bare ice", 'layer': 'II', 'b_prime': [0.5, 1.8], 'g': 0.94, 'a': [0.0683, 0.0683], 'wl': 540, 'author': 'Light et al. 2008'},
                   'IL_snow_cover_1': {'cond': "FY snow cover ice", 'layer': 'II', 'b_prime': [0.46, 4.4], 'a': [0.0683, 0.0683], 'wl': 540, 'author': "Perron et al. 2021"},
                   'IL_bare_cover_1': {'cond': "FY bare ice", 'layer': 'II', 'b_prime': [0.15, 2.8], 'a': [0.0683, 0.0683], 'wl': 540, 'author': "Perron et al. 2021"},
                   'SSL_HA': {'cond': "MY snow covered ice", 'layer': 'SSL', 'b_prime': [25.5], 'g': 0.85, 'a': [0.0775], 'wl': 540, 'author': "High Arctic site"},
                   'OII_HA': {'cond':"MY snow covered ice", 'layer': 'II', 'b_prime': [2.4, 4.8], 'g': 0.99, 'a': [0.0775, 0.0775], 'wl': 540, 'author': "High Arctic site"},
                   'YII_HA': {'cond': "MY snow covered ice", 'layer': 'II', 'b_prime': [0.8, 1.1], 'g': 0.99, 'a': [0.0775, 0.0775], 'wl': 540, 'author': "High Arctic site"},
                   'SSL_CB': {'cond': "FY landfast ice", 'layer': 'SSL', 'b_prime': [37.5], 'g': 0.85, 'a': [0.30], 'wl': 540, 'author': "Chaleur Bay site"},
                   'II_CB': {'cond': "FY landfast ice", 'layer': 'II', 'b_prime': [2.8, 6.4], 'g': 0.99, 'a': [0.30, 1.30], 'wl': 540, 'author': "Chaleur Bay site"}
                   }


    # LOOP
    fig1, ax1 = plt.subplots()
    colors = matplotlib.cm.get_cmap('tab10')
    keys_loop = ['SSL_white_ice', 'SSL_bare_ice', 'DL_bare_ice_1', 'DL_bare_ice_2',
                 'IL_cold_2', 'IL_cold_4', 'IL_cold_5', 'IL_snow_cover_1', 'IL_bare_cover_1',
                 'SSL_HA', 'OII_HA', 'YII_HA', 'SSL_CB', 'II_CB']
    xlabels = []
    ind = []
    x_pos = 0
    cl_dct = {"SSL": colors(0), "DL": colors(3), "II": colors(1)}
    for k in keys_loop:

        color = cl_dct[data_dict[k]['layer']]
        b_prime = data_dict[k]['b_prime']

        if len(b_prime) > 1:
            print("More than one b_prime value for this condition: ", k)
            min_b_prime, max_b_prime = b_prime[0], b_prime[1]
            if data_dict[k]['author'] == "High Arctic site":
                ax1.bar(9, bottom=min_b_prime, height=max_b_prime, width=0.4, alpha=0.5, color=color, edgecolor=color)
            elif data_dict[k]['author'] == "Chaleur Bay site":
                ax1.bar(10, bottom=min_b_prime, height=max_b_prime, width=0.4, alpha=0.5, color=color, edgecolor=color)
            else:
                ax1.bar(x_pos, bottom=min_b_prime, height=max_b_prime, width=0.4, alpha=0.5, color=color, edgecolor=color)

                xlabels.append(data_dict[k]['author'])
                ind.append(x_pos)
                x_pos += 1

        else:
            print("Only one b_prime value for this condition: ", k)
            if data_dict[k]['author'] == "High Arctic site":
                ax1.bar(9, bottom=b_prime[0], height=0, width=0.4, alpha=0.5, color=color, edgecolor=color)
            elif data_dict[k]['author'] == "Chaleur Bay site":
                ax1.bar(10, bottom=b_prime[0], height=0, width=0.4, alpha=0.5, color=color, edgecolor=color)
            else:
                ax1.bar(x_pos, bottom=b_prime[0], height=0, width=0.4, alpha=0.5, color="none", edgecolor=color)

                xlabels.append(data_dict[k]['author'])
                ind.append(x_pos)
                x_pos += 1

    xlabels.append("High Arctic site")
    xlabels.append("Chaleur Bay site")
    ind.append(9)
    ind.append(10)

    ax1.bar(x_pos, bottom=np.nan, height=np.nan, width=0.4, alpha=0.5, color=cl_dct["SSL"], edgecolor=cl_dct["SSL"], label="Surface scattering layer")
    ax1.bar(x_pos, bottom=np.nan, height=np.nan, width=0.4, alpha=0.5, color=cl_dct["DL"], edgecolor=cl_dct["DL"], label="Drained layer")
    ax1.bar(x_pos, bottom=np.nan, height=np.nan, width=0.4, alpha=0.5, color=cl_dct["II"], edgecolor=cl_dct["II"], label="Interior ice")

    ax1.set_ylim((0.1, 300))
    ax1.set_yscale("log")
    ax1.set_xticks(ind, labels=xlabels, rotation=-15, fontsize=6)
    ax1.set_ylabel('Reduced scattering coefficient [m$^{-1}$]')
    ax1.legend(loc=3, fontsize=9)

    # --- Second figure --- Similarity parameter histogram
    fig2 = plt.figure(figsize=(6.6929 * 0.8, 6.6929 * 0.8 * 1/1.618))

    gs0 = gridspec.GridSpec(1, 6, figure=fig2)
    #fig2, ax2 = plt.subplots(1, 3, figsize=(6.6929, 6.6929 * 3 / 4))
    ax2 = []
    ax2.append(fig2.add_subplot(gs0[:-2]))
    ax2.append(fig2.add_subplot(gs0[-2]))
    ax2.append(fig2.add_subplot(gs0[-1]))

    #bar_pos = {"Ehn et al. 2008\nblue ice": 0, "Ehn et al. 2008\nwhite ice": 1,
    #           "Light et al. 2008": 2, 'Perron et al. 2021\nsnow': 3, 'Perron et al. 2021\nbare': 4, "High Arctic site": 5, "Chaleur Bay site": 6}

    ice_cond = {"FY blue ice": 0, "FY white ice": 1, "Melting MY bare ice": 2, "FY snow cover ice": 3,
                "FY bare ice": 4, "MY snow covered ice": 5, "FY landfast ice": 6}
    ls_dict = {500: "-.", 540: "-", 633: "--"}
    k_loop_s = ['SSL_bare_ice', 'DL_bare_ice_1', 'IL_cold_5', 'IL_bare_cover_1', 'IL_snow_cover_1',
                'SSL_HA', 'OII_HA', 'YII_HA', 'SSL_CB', 'II_CB']

    auth = ["Ehn et al. 2008", "Ehn et al. 2008", "Light et al. 2008", 'Perron et al. 2021', 'Perron et al. 2021',
            "High Arctic site", "Chaleur Bay site"]

    for k in data_dict_s.keys():

        color = cl_dct[data_dict_s[k]['layer']]
        b_prime = data_dict_s[k]['b_prime']

        if len(b_prime) > 1:
            print("More than one b_prime value for this condition: ", k)
            min_b_prime, max_b_prime = b_prime[0], b_prime[1]

            min_s = calculate_s(data_dict_s[k]["a"][0], b_prime[0])
            max_s = calculate_s(data_dict_s[k]["a"][1], b_prime[1])

            ba = ax2[0].bar(ice_cond[data_dict_s[k]['cond']], bottom=max_s, height=(min_s - max_s), linestyle=ls_dict[data_dict_s[k]["wl"]], width=0.4, alpha=1.0, color=color, edgecolor="k", zorder=2)

            #xlabels.append(data_dict[k]['author'])
            #ind.append(x_pos)
            #x_pos += 1

            if (data_dict_s[k]["author"] == 'High Arctic site') or (data_dict_s[k]["author"] == 'Chaleur Bay site'):
                ax2[0].axhline(min_s, linestyle="--", linewidth=0.6, color="k", zorder=1)
                ax2[0].axhline(max_s, linestyle="--", linewidth=0.6, color="k", zorder=1)
                #ax2[2].bar_label(ba, label_type="edge", fmt="%.2f")

                maval = max([min_s, max_s])
                mival = min([min_s, max_s])

                ax2[0].text(ice_cond[data_dict_s[k]['cond']], maval - 0.015, f"{maval:.3f}", ha='center', fontsize=5)
                ax2[0].text(ice_cond[data_dict_s[k]['cond']], mival + 0.005, f"{mival:.3f}", ha='center', fontsize=5)

        else:
            print("Only one b_prime value for this condition: ", k)
            s = calculate_s(data_dict_s[k]["a"][0], b_prime[0])
            ax2[0].bar(ice_cond[data_dict_s[k]['cond']], bottom=s, height=0, width=0.4, alpha=1.0, color=color, linestyle=ls_dict[data_dict_s[k]["wl"]], edgecolor=color, zorder=2)

            #xlabels.append(data_dict[k]['author'])
            #ind.append(x_pos)
            #x_pos += 1

    for k in data_dict_ehn.keys():

        color = cl_dct[data_dict_ehn[k]['layer']]
        s = data_dict_ehn[k]["s"]

        if len(s) > 1:
            print("More than one b_prime value for this condition: ", k)
            ax2[0].bar(ice_cond[data_dict_ehn[k]['cond']], bottom=s[0], height=(s[1] - s[0]), linestyle=ls_dict[data_dict_ehn[k]["wl"]], width=0.4, alpha=1.0, color=color, edgecolor="k", zorder=2)

        else:
            print("Only one b_prime value for this condition: ", k)
            ax2[0].bar(ice_cond[data_dict_ehn[k]['cond']], bottom=s[0], height=0, width=0.4, alpha=1.0, color=color, linestyle=ls_dict[data_dict_ehn[k]["wl"]], edgecolor=color, zorder=2)

    ax2[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=0.5, color=cl_dct["SSL"], edgecolor="none", label="Surface scattering layer")
    ax2[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=0.5, color=cl_dct["DL"], edgecolor="none", label="Drained layer")
    ax2[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=0.5, color=cl_dct["II"], edgecolor="none", label="Interior ice")
    ax2[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color="none", edgecolor="k", linestyle="-.", label="500 nm")
    ax2[0].bar(0, bottom=np.nan, height=np.nan, width=0.4, alpha=1.0, color="none", edgecolor="k", linestyle="-", label="540 nm")

    ax_t = ax2[0].secondary_xaxis('top')
    ax_t.set_xticks(list(ice_cond.values()), labels=list(ice_cond.keys()), rotation=-15, fontsize=7)

    ax2[0].set_ylabel('Similarity parameter $S$')
    ax2[0].set_ylim((0, 0.62))
    ax2[0].set_xticks(list(ice_cond.values()), labels=auth, rotation=-15, fontsize=7)
    ax2[0].legend(loc=3, fontsize=6)
    ax2[0].invert_yaxis()

    # --- Second figure --- Similarity parameter histogram
    rc_oden = RadClass(data_path="../oden2018/data/oden-08312018-fluo.h5")
    ifb_st2 = get_ice_freeboard("../baiedeschaleurs2022/data/station_2_data.txt")
    rc_st2 = RadClass(data_path="../baiedeschaleurs2022/data/baiedeschaleurs-03232022-fluo.h5", station="station_2", data_type="camera", freeboard=ifb_st2)

    image_RGB_oden, oden_depths = sea_ice_color.process_rgb_images("../oden2018/jpeg_img", rc_oden, start=1)
    image_RGB_bdc, bdc_depths = sea_ice_color.process_rgb_images("../baiedeschaleurs2022/jpeg_img", rc_st2, start=0)

    ax2[1].imshow(image_RGB_oden)
    ax2[2].imshow(image_RGB_bdc)

    ax2[1].set_xticks([])
    ax2[1].set_yticks(np.arange(0,  oden_depths.shape[0], 1))
    ax2[1].set_yticklabels(oden_depths.astype(int).astype(str).tolist())
    ax2[1].set_ylabel("Depth [cm]")
    ax2[1].set_title("High Arctic", fontsize=7)

    ax2[2].set_xticks([])
    ax2[2].set_yticks(np.arange(0, bdc_depths.shape[0], 1))
    ax2[2].set_yticklabels(bdc_depths.astype(int).astype(str).tolist())
    ax2[2].set_ylabel("Depth [cm]")
    ax2[2].set_title("Chaleur Bay", fontsize=7)


    #ax2[0].text(-0.1, 1.1, "(" + string.ascii_lowercase[0] + ")", transform=ax2[0].transAxes, size=10, weight='bold')
    #ax2[1].text(-0.1, 1.1, "(" + string.ascii_lowercase[1] + ")", transform=ax2[1].transAxes, size=10, weight='bold')
    #ax2[2].text(-0.1, 1.1, "(" + string.ascii_lowercase[2] + ")", transform=ax2[2].transAxes, size=10, weight='bold')

    fig2.tight_layout()

    # --- Third figure --- Similarity parameter

    fig3, ax3 = plt.subplots()

    ax3.plot(s_ha_b, z, marker=".", linestyle="none")
    ax3.plot(s_ha_g, z, marker=".", linestyle="none")
    ax3.plot(s_ha_r, z, marker=".", linestyle="none")

    ax3.set_ylabel("Depth in ice [cm]")
    ax3.invert_yaxis()

    fig2.savefig("figures/summary_fig.pdf", format="pdf", dpi=300)
    fig2.savefig("figures/summary_fig.png", format="png", dpi=300)

    plt.show()

