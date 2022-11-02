import pandas as pd
import numpy as np
import os

save_dir = "../../../storage/racarcam/"
filename = "DM_Run2_50MET.h5"   # Change file if needed

df = pd.read_hdf(save_dir+filename, key='df_tot')
print(df)

dm_df = df.loc[df['Label'] == 1]
df = df.drop(df[df['Label'] == 1].index)
print(dm_df)
print(df)

print("=="*40)
print('Statistical reduction')
print("=="*40)

dsids = df['RunNumber'].unique()
df_dict = {dsid: df.loc[df['RunNumber'] == dsid] for dsid in dsids}

dm_dsids = dm_df['RunNumber'].unique()
dm_dict = {dm_dsid: dm_df.loc[dm_df['RunNumber'] == dm_dsid] for dm_dsid in dm_dsids}

for dsid in dsids:
#     xs = df_dict[dsid]['CrossSection'].iloc[0]
#     n_evnts = np.shape(df_dict[dsid])[0]
#     print('DSID: ',dsid,' n_evnts/xs = ', n_evnts,'/', xs,' = ', n_evnts/xs)
# exit()
    if np.shape(df_dict[dsid])[0] > 10000000:
        print('%g reduced to 4%%' %dsid)
        df_dict[dsid] = df_dict[dsid].sample(frac=0.04, random_state=42)
        
    elif np.shape(df_dict[dsid])[0] > 5000000:
        print('%g reduced to 5%%' %dsid)
        df_dict[dsid] = df_dict[dsid].sample(frac=0.05, random_state=42)
    
    elif np.shape(df_dict[dsid])[0] > 1000000:
        print('%g reduced to 8%%' %dsid)
        df_dict[dsid] = df_dict[dsid].sample(frac=0.08, random_state=42)
    
    print(dsid, np.shape(df_dict[dsid])[0], np.shape(df_dict[dsid])[1] )

new_bkg = pd.concat(df_dict.values())
print(new_bkg)

df_tot = pd.concat([new_bkg, dm_df]).sort_index()
print(df_tot)

newfile = "Stat_red_"+filename

if os.path.exists(save_dir+newfile):
    print('Rewriting h5 file')
    os.remove(save_dir+newfile)

df_tot.to_hdf(save_dir+newfile, key='df_tot')