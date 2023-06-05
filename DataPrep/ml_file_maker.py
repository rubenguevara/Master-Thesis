import pandas as pd
import os, time, json
from EventIDs import IDs
import numpy as np
import uproot as up
from collections import OrderedDict

t0 = time.time()

save_dir = "/storage/racarcam/"


## Customize files here
dm1 = save_dir + "monoZp.root"
dm2 = save_dir + "SUSY.root"
Run2_bkgs = save_dir + "Run2_bkgs.root"
filename = 'FULL_DATASET.h5' 

thing = up.open(Run2_bkgs)

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

df['Label'] = np.isin(df['RunNumber'], IDs["dm_sig"]).astype(int)
print(df)

t = "{:.2f}".format(int( time.time()-t0 )/60.)
print( "---"*40)
print( "Time spent making bkg df: "+str(t)+" min")  
print( "---"*40)

t1 = time.time()

dm_thing = up.open(dm1)

dm_tree_a = dm_thing['id_mc16a']
dm_tree_d = dm_thing['id_mc16d']
dm_tree_e = dm_thing['id_mc16e']

dm_df1 = dm_tree_a.arrays(library="pd")
dm_df2 = dm_tree_d.arrays(library="pd")
dm_df3 = dm_tree_e.arrays(library="pd")
print(dm_df1)
print(dm_df2)
print(dm_df3)

dm_dfs = [dm_df1, dm_df2, dm_df3]
dm_df = pd.concat(dm_dfs).sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle and fix indices
print(dm_df)

dm_df['Label'] = np.isin(dm_df['RunNumber'], IDs["dm_sig"]).astype(int)
print(dm_df)

susy_thing = up.open(dm2)

susy_tree_a = susy_thing['id_mc16a']
susy_tree_d = susy_thing['id_mc16d']
susy_tree_e = susy_thing['id_mc16e']

susy_df1 = susy_tree_a.arrays(library="pd")
susy_df2 = susy_tree_d.arrays(library="pd")
susy_df3 = susy_tree_e.arrays(library="pd")
print(susy_df1)
print(susy_df2)
print(susy_df3)

susy_dfs = [susy_df1, susy_df2, susy_df3]
susy_df = pd.concat(susy_dfs).sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle and fix indices
print(susy_df)

susy_df['Label'] = np.isin(susy_df['RunNumber'], IDs["SUSY"]).astype(int)
print(susy_df)


t = "{:.2f}".format(int( time.time()-t1 )/60.)
print( "---"*40)
print( "Time spent making sig df: "+str(t)+" min")  
print( "---"*40)

t2 = time.time()

s_and_b = [df, dm_df, susy_df]
df_tot = pd.concat(s_and_b).sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle and fix indices
print(df_tot)

t = "{:.2f}".format(int( time.time()-t2 )/60.)
print( "---"*40)
print( "Time spent making mixed df: "+str(t)+" min")  
print( "---"*40)

print('REMOVING WEIGHT = 0 EVENTS')
df_weight_0 = df_tot[df_tot['Weight'] == 0].index
print(df_tot[df_tot['Weight'] == 0])
df_tot = df_tot.drop(df_weight_0).reset_index(drop=True)
print(df_tot)

print("---"*40)
print('Weight fixing')
print("---"*40)

sow_bkg_file = open('SOW_bkg.json')
SOW_bkg = json.load(sow_bkg_file)
sow_sig_file = open('SOW_sig_AFII.json')
SOW_sig = json.load(sow_sig_file)
sow_susy_file = open('SOW_SIG_SUSY.json')
SOW_SUSY = json.load(sow_susy_file)

SOW_a = OrderedDict(list(SOW_SUSY['mc16a'].items()) + list(SOW_sig['mc16a'].items()))
SOW_d = OrderedDict(list(SOW_SUSY['mc16d'].items()) + list(SOW_sig['mc16d'].items()))
SOW_e = OrderedDict(list(SOW_SUSY['mc16e'].items()) + list(SOW_sig['mc16e'].items()))

cols = df_tot.columns.tolist()

df_tot.rename(columns = {'Weight':'OldWeight'}, inplace = True)

new_weights = []
for w, dsid, mcrun, label in zip(df_tot['OldWeight'], df_tot['RunNumber'], df_tot['RunPeriod'], df_tot['Label']):
    if label == 0:
        sow = SOW_bkg[mcrun][str(dsid)]
    elif label == 1:
        if mcrun == 'mc16a':
            sow = SOW_a[str(dsid)]
        elif mcrun == 'mc16d':
            sow = SOW_d[str(dsid)]
        elif mcrun == 'mc16e':
            sow = SOW_e[str(dsid)]
    
    if mcrun == 'mc16a':
        lumi = 36.2
    elif mcrun == 'mc16d':
        lumi = 44.3        
    elif mcrun == 'mc16e':
        lumi = 58.5
    wgt = lumi*w/sow
    new_weights.append(wgt)
sow_bkg_file.close()
sow_sig_file.close()

print(df_tot['OldWeight'][df_tot["RunNumber"]==503057])
old = df_tot.pop('OldWeight')
df_tot['Weight'] = new_weights
df_tot = df_tot[cols]
print(df_tot['Weight'][df_tot["RunNumber"]==503057])
print(df_tot)
exit()
t = "{:.2f}".format(int( time.time()-t0 )/60.)
print( "---"*40)
print( "TOTAL time spent preparing df: "+str(t)+" min")  
print( "---"*40)

t3 = time.time()
df_tot.to_hdf(save_dir+filename, key='df_tot')

t = "{:.2f}".format(int( time.time()-t3 )/60.)
print( "---"*40)
print( "Time spent saving file: "+str(t)+" min")  
print( "---"*40)


t4 = time.time()
new_df = pd.read_hdf(save_dir+filename, key='df_tot')
print(new_df)

t = "{:.2f}".format(int( time.time()-t4 )/60.)
print( "---"*40)
print( "Time spent reading file: "+str(t)+" min")  
print( "---"*40)
