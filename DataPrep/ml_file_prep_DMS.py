import pandas as pd
import numpy as np
import os, time

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

new_bkg = pd.concat(df_dict.values()).to_hdf(save_dir+'Stat_red_bkgs.h5', key='df_tot')
print(new_bkg)

print("=="*40)
print('Making files for each DSID')
print("=="*40)

for dm_dsid in dm_dsids:
    newfile = 'DMS/'+str(dm_dsid)+'.h5'
    print('Events in ', dm_dsid, " : ", np.shape(dm_dict[dm_dsid])[0])
    print('Making file '+newfile)
    dm_dict[dm_dsid].to_hdf(save_dir+newfile, key='df_tot')

print("=="*40)
print('Testing time')
print("=="*40)

t0 = time.time()

new_df = pd.read_hdf(save_dir+'Stat_red_bkgs.h5', key='df_tot')
new_dm = pd.read_hdf(save_dir+'DMS/'+str(dm_dsid)+'.h5', key='df_tot')

new_concat = pd.concat([new_df, new_dm]).sort_index()
print(new_concat)

t = "{:.2f}".format(int( time.time()-t0 )/60.)
print( "Time spent making reduced bkg + dsid dataframe: "+str(t)+" min")