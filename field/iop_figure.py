import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from meta_dict import meta_dict

conditions = meta_dict['conditions'].keys()
fig, ax = plt.subplots()
x_pos = 0
colors = matplotlib.cm.get_cmap('tab10')
for condition in conditions:
    color = colors(0.1*x_pos)
    delta_pos = 0.0
    for condition_tag in meta_dict['conditions'][condition]:
        b_prime = meta_dict[condition_tag]['b_prime']
        if len(b_prime) > 1:
            print("More than one b_prime value for this condition: ", condition_tag)
            min_b_prime, max_b_prime = b_prime[0], b_prime[1]
            ax.bar(x_pos+delta_pos, height=max_b_prime, width=0.55, bottom=min_b_prime, alpha=0.5, color=color)
        else:
            print("Only one b_prime value for this condition: ", condition_tag)
            ax.bar(x_pos+delta_pos, height=np.log10(b_prime[0])*2, width=0.55, bottom=b_prime[0], alpha=0.5, color=color)
        ax.text(x_pos+delta_pos, b_prime[0], meta_dict[condition_tag]['author'], fontsize='xx-small', ha='center', va='bottom')
        delta_pos += 0.05
    ax.bar(0, 0, label=condition, color=color, alpha=0.5)
    x_pos += 1
plt.ylabel('Scattering coefficient [m$^{-1}$]')
ax.set_xticks(np.arange(0, len(conditions)))
ax.set_xticklabels(conditions, rotation = -30   , ha='left')
plt.semilogy()
plt.legend()
plt.show()