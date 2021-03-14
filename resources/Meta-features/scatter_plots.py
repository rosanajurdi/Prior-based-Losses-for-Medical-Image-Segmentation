import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
root_dir = '/media/eljurros/Transcend/Decathlone/benchmark/Spleen/Fold_1'
size = pd.read_csv(os.path.join(root_dir, 'size.csv'))
dice = pd.read_csv(os.path.join(root_dir, 'results', 'surface', 'surface_clean.csv'))
d = dice[' dice'].reset_index().values
s = size.reset_index().values
plt.scatter(np.array(size[' cl1']), np.array(dice[' dice']))
plt.show()

