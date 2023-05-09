import pandas as pd
import os, time, json
from EventIDs import IDs
import numpy as np
import uproot as up

t0 = time.time()

save_dir = "../../../storage/racarcam/"


## Customize files here
Run2_bkgs = save_dir + "Run2FINAL.root"
filename = 'W_search.h5' 

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

df['Label'] = np.isin(df['RunNumber'], IDs["W"]).astype(int)
print(df)

t = "{:.2f}".format(int( time.time()-t0 )/60.)
print( "---"*40)
print( "Time spent making df: "+str(t)+" min")  
print( "---"*40)

t2 = time.time()

df_tot = df
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

cols = df_tot.columns.tolist()

df_tot.rename(columns = {'Weight':'OldWeight'}, inplace = True)

new_weights = []
for w, dsid, mcrun in zip(df_tot['OldWeight'], df_tot['RunNumber'], df_tot['RunPeriod']):
    sow = SOW_bkg[mcrun][str(dsid)]
    if mcrun == 'mc16a':
        lumi = 36.2
    elif mcrun == 'mc16d':
        lumi = 44.3        
    elif mcrun == 'mc16e':
        lumi = 58.5
    wgt = lumi*w/sow
    new_weights.append(wgt)
sow_bkg_file.close()

old = df_tot.pop('OldWeight')
df_tot['Weight'] = new_weights
df_tot = df_tot[cols]

print(df_tot)

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
