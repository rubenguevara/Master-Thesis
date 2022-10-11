import pandas as pd
import os
from EventIDs import IDs
import numpy as np
import uproot as up

file ='../EventSelector/ML_files_test/ParallelTest-mc16a.root'
file2 ='../EventSelector/ML_files_test/ParallelTest-mc16d.root'
# file2 ='../EventSelector/ML_files/ParallelTest-mc16e.root'

thing = up.open(file)
thing2 = up.open(file2)
# thing3 = up.open(file3)
tree = thing['id_mc16a']
tree2 = thing2['id_mc16d']
# tree3 = thing3['id_mc16e']
dic = {}
dic2 = {}
# dic3 = {}

for i in tree.keys():
    dic[i] = tree[i].array()
    dic2[i] = tree2[i].array()
    # dic3[i] = tree3[i].array()


df1 = pd.DataFrame(dic)
# print(df1)
df2 = pd.DataFrame(dic2)
# print(df2)
# df3 = pd.DataFrame(dic3)
# print(df3)

dfs = [df1, df2]#, df3]
df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle and fix indices
print(df)

# Labeling method 1
df['Label'] = np.isin(df['RunNumber'], IDs["all_bkg"])
print(df)

# df['Label'] = df['Label'].replace(True, 0)
# df['Label'] = df['Label'].replace(False, 1)
# print(df)

# # Labeling method 2
# df['Label'] = np.isin(df['RunNumber'], IDs["sig_AFII"])
# print(df)

# df['Label'] = df['Label'].astype(int)
# print(df)

# save_dir = "ML_Files"
# try:
#     os.makedirs(save_dir)

# except FileExistsError:
#     pass

# df.to_csv(save_dir+'/Run2Bkgs.csv')
