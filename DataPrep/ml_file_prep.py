import pandas as pd
import numpy as np
import os, time, json

save_dir = "../../../storage/racarcam/"
filename = "FULL_DM_50MET.h5"   # Change file if needed

real_df = pd.read_hdf(save_dir+filename, key='df_tot')

df = real_df.copy()
dm_df = df.loc[df['Label'] == 1]
df = df.drop(df[df['Label'] == 1].index)

df.to_hdf(save_dir+'bkgs.h5', key='df_tot')

print(dm_df)
print(df)

print("=="*40)
print('Making files for different theories')
print("=="*40)

dm_dsids = dm_df['RunNumber'].unique()
dm_dict = {}
dm_files = {}
json_file = open('DM_DICT_Zp.json')

DM_file = json.load(json_file)

for k,v in DM_file.items():
    for x in v:
        dm_dict.setdefault(x,[]).append(k)
        dm_files[k]=[]
print(dm_files)
for dm_dsid in dm_dsids:
    print(dm_dsid)
    key = dm_dict[str(dm_dsid)][0]
    print(key)
    # dm_dict[key].append( dm_df.loc[dm_df['RunNumber'] == dm_dsid] )
exit()
for key in DM_keys:
    pd.concat(dm_dict[key]).to_hdf(save_dir+'/DM_Parts/'+key+'.h5', key='df_tot')
json_file.close()

print("=="*40)
print('Testing time')
print("=="*40)

t0 = time.time()

new_df = pd.read_hdf(save_dir+'bkgs.h5', key='df_tot')
new_dm = pd.read_hdf(save_dir+'/DM_Parts/'+key+'.h5', key='df_tot')

new_concat = pd.concat([new_df, new_dm]).sort_index()
print(new_concat)

t = "{:.2f}".format(int( time.time()-t0 )/60.)
print( "Time spent making reduced bkg + dsid dataframe: "+str(t)+" min")