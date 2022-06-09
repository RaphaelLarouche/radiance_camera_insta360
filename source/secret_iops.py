from HE60PY.seaicesimulation import SeaIceSimulation
from HE60PY.dataparser import DataParser
from HE60PY.dataviewer import DataViewer
from HE60PY.phasefunctions import *


root_name = 'secret_iops'
secret_iops = SeaIceSimulation(root_name=root_name, run_title=root_name, mode="Oden", wavelength_list=[480, 540, 600])
secret_iops.set_z_grid(z_max=3.0) # tu as 3 mètres pour faire ce que tu veux, il faut juste que la dernière couche soit de l'eau pour qu'Hydro Light l'étire jusqu'à l'infini
secret_iops.add_layer(z1=0.00, z2=1.00, abs={'480': 666, '540': 666, '600': 666}, scat=999, dpf=OTHG(0.99))
secret_iops.add_layer(z1=1.00, z2=2.00, abs={'480': 666, '540': 666, '600': 666}, scat=999, dpf=OTHG(0.85))
secret_iops.add_layer(z1=2.00, z2=3.01, abs={'480': 666, '540': 666, '600': 666}, scat=999, dpf=OTHG(0.85))
secret_iops.run_simulation(printoutput=True)
secret_iops.parse_results()
# Tu peux faire le nombre de couches que tu veux, il faut juste que le dernier z2 soit 1 cm plus grand que zmax
# les couches doivent avoir une épaisseur supérieure à 1 cm mais peuvent être différentes les unes des autres.
# Amuses toi! Je vais supprimer le fichier après l'avoir executé!