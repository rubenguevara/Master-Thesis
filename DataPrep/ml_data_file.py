import pandas as pd
import numpy as np
import os, time, json
import uproot as up

save_dir = "/storage/racarcam/"
filename = "datafrfr.root"   # Change file if needed
data = save_dir+filename

thing = up.open(data)

tree_15 = thing['id_data15']
tree_16 = thing['id_data16']
tree_17 = thing['id_data17']
tree_18 = thing['id_data18']

df1 = tree_15.arrays(library="pd")
df2 = tree_16.arrays(library="pd")
df3 = tree_17.arrays(library="pd")
df4 = tree_18.arrays(library="pd")
print(df1)
print(df2)
print(df3)
print(df4)

dfs = [df1, df2, df3, df4]
df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle and fix indices
print(df)


df = df.drop(['Weight', 'CrossSection'], axis=1)
print(df)


df.to_hdf(save_dir+'datafrfr.h5', key='df')
