"""
Inversion error with simulated noise.
"""

from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer
from HE60PY.phasefunctions import *


root_name = 'secret_iops_einv02'
secret_iops = SeaIceSimulation(root_name=root_name, run_title=root_name, mode="Oden", wavelength_list=[480, 540, 600])
secret_iops.set_z_grid(z_max=3.0) # tu as 3 mètres pour faire ce que tu veux, il faut juste que la dernière couche soit de l'eau pour qu'Hydro Light l'étire jusqu'à l'infini

# IOPS
# SSL
secret_iops.add_layer(z1=0.00, z2=0.06, abs={'480': 0.12, '540': 0.15, '600': 0.22}, scat=510, dpf=OTHG(0.78))
# DL
secret_iops.add_layer(z1=0.06, z2=0.15, abs={'480': 0.08, '540': 0.11, '600': 0.15}, scat=300, dpf=OTHG(0.90))
# II
secret_iops.add_layer(z1=0.15, z2=1.15, abs={'480': 0.06, '540': 0.09, '600': 0.13}, scat=103, dpf=OTHG(0.98))
secret_iops.add_layer(z1=1.15, z2=2.00, abs={'480': 0.06, '540': 0.09, '600': 0.13}, scat=30, dpf=OTHG(0.98))

# Water
secret_iops.add_layer(z1=2.00, z2=3.01, abs={'480': 0.04, '540': 0.06, '600': 0.12}, scat=0.17, dpf=OTHG(0.93))

secret_iops.run_simulation(printoutput=True)
secret_iops.parse_results()
# Tu peux faire le nombre de couches que tu veux, il faut juste que le dernier z2 soit 1 cm plus grand que zmax
# les couches doivent avoir une épaisseur supérieure à 1 cm mais peuvent être différentes les unes des autres.
# Amuses toi! Je vais supprimer le fichier après l'avoir executé!