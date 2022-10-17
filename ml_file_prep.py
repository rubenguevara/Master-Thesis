import pandas as pd
import os
from EventIDs import IDs
import numpy as np
import uproot as up

save_dir = "../../../storage/racarcam/"
dm1 = save_dir + "DM1.root"
Run2_bkgs = save_dir + "Run2.root"


thing = up.open(Run2_bkgs)
print(thing.keys())

tree_a = thing['id_mc16a']
tree_d = thing['id_mc16d']
tree_e = thing['id_mc16e']

df1 = tree_a.arrays(library="pd")
df2 = tree_d.arrays(library="pd")
df3 = tree_e.arrays(library="pd")
print(df1)
print(df2)
print(df3)

dfs = [df1, df2, df3]
df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle and fix indices
print(df)

df['Label'] = np.isin(df['RunNumber'], IDs["all_bkg"]).astype(int)
print(df)


# # save_dir = "ML_Files"
# # try:
# #     os.makedirs(save_dir)

# # except FileExistsError:
# #     pass

# # df.to_csv(save_dir+'/Run2Bkgs.csv')
