import pandas as pd
import numpy as np
import os, time, json
import uproot as up

save_dir = "../../../storage/racarcam/"
filename = 'FULL_Zp_FINAL.h5'            

df = pd.read_hdf(save_dir+filename, key='df_tot')

dm_df = df.loc[df['Label'] == 1]
df = df.drop(df[df['Label'] == 1].index)
print(dm_df)
print(df)
print("=="*40)

dsids = df['RunNumber'].unique()
df_dict = {dsid: df.loc[df['RunNumber'] == dsid] for dsid in dsids}

new_bkg = pd.concat(df_dict.values()).to_hdf(save_dir+'bkgs_final.h5', key='df_tot')

print("=="*40)
print('Making files for each DSID')
print("=="*40)

dm_dsids = dm_df['RunNumber'].unique()
dm_dict = {dm_dsid: dm_df.loc[dm_df['RunNumber'] == dm_dsid] for dm_dsid in dm_dsids}

for dm_dsid in dm_dsids:
    newfile = 'Zp_DMS/'+str(dm_dsid)+'.h5'
    print('Events in ', dm_dsid, " : ", np.shape(dm_dict[dm_dsid])[0])
    print('Making file '+newfile)
    dm_dict[dm_dsid].to_hdf(save_dir+newfile, key='df_tot')

print("=="*40)
print('Testing time')
print("=="*40)

t0 = time.time()

new_df = pd.read_hdf(save_dir+'bkgs_final.h5', key='df_tot')
new_dm = pd.read_hdf(save_dir+'Zp_DMS/'+str(dm_dsid)+'.h5', key='df_tot')

new_concat = pd.concat([new_df, new_dm]).sort_index()
print(new_concat)

t = "{:.2f}".format(int( time.time()-t0 )/60.)
print( "Time spent making reduced bkg + dsid dataframe: "+str(t)+" min")