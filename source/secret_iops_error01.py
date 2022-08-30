from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer
from HE60PY.phasefunctions import *


root_name = 'secret_iops_error01'
secret_iops = SeaIceSimulation(root_name=root_name, run_title=root_name, mode="Oden", wavelength_list=[480, 540, 600])
secret_iops.set_z_grid(z_max=3.0) # tu as 3 mètres pour faire ce que tu veux, il faut juste que la dernière couche soit de l'eau pour qu'Hydro Light l'étire jusqu'à l'infini

# IOPS
# SSL
secret_iops.add_layer(z1=0.00, z2=0.12, abs={'480': 0.10, '540': 0.12, '600': 0.16}, scat=700, dpf=OTHG(0.87))
# DL
secret_iops.add_layer(z1=0.12, z2=0.20, abs={'480': 0.15, '540': 0.18, '600': 0.25}, scat=180, dpf=OTHG(0.94))
# II
secret_iops.add_layer(z1=0.20, z2=1.00, abs={'480': 0.15, '540': 0.18, '600': 0.25}, scat=70, dpf=OTHG(0.99))
secret_iops.add_layer(z1=1.00, z2=1.80, abs={'480': 0.18, '540': 0.21, '600': 0.27}, scat=50, dpf=OTHG(0.99))
secret_iops.add_layer(z1=1.80, z2=2.10, abs={'480': 0.22, '540': 0.25, '600': 0.30}, scat=80, dpf=OTHG(0.99))

# Water
secret_iops.add_layer(z1=2.10, z2=3.01, abs={'480': 0.05, '540': 0.09, '600': 0.15}, scat=0.23, dpf=OTHG(0.88))

secret_iops.run_simulation(printoutput=True)
secret_iops.parse_results()
# Tu peux faire le nombre de couches que tu veux, il faut juste que le dernier z2 soit 1 cm plus grand que zmax
# les couches doivent avoir une épaisseur supérieure à 1 cm mais peuvent être différentes les unes des autres.
# Amuses toi! Je vais supprimer le fichier après l'avoir executé!