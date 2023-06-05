import pandas as pd
import numpy as np
import os, time, json

save_dir = "/storage/racarcam/"
# filename = 'FULL_Zp_FINAL.h5'                                                       # Change file if needed
filename ='FINAL_FULL_DATASET.h5' 

df = pd.read_hdf(save_dir+filename, key='df_tot')

json_file2 = open('DM_DICT_SUSY_dsid.json')
SUSY_dict = json.load(json_file2)

model_dict = open('DM_DICT_Zp_models.json')
DM_DICT = json.load(model_dict)

dm_df = df.loc[df['Label'] == 1]
df = df.drop(df[df['Label'] == 1].index)
df.to_hdf(save_dir+'bkgs_frfr.h5', key='df_tot')

print(dm_df)
print(df)

print("=="*40)
print('Making files for each model')
print("=="*40)

dm_dsids = dm_df['RunNumber'].unique()
dm_models = {}
models = ['lv_lds', 'eft_lds', 'dh_lds', 'lv_hds', 'eft_hds', 'dh_hds', 'SlepSlep', 'Stop']
for m in models:
    dm_models[m] = []

    
dm_dict = {dm_dsid: dm_df.loc[dm_df['RunNumber'] == dm_dsid] for dm_dsid in dm_dsids}
for dm_dsid in dm_dsids:
    if str(dm_dsid) in DM_DICT['lv_lds']:
        dm_models['lv_lds'].append(dm_dict[dm_dsid])
    elif str(dm_dsid) in DM_DICT['eft_lds']:
        dm_models['eft_lds'].append(dm_dict[dm_dsid])
    elif str(dm_dsid) in DM_DICT['dh_lds']:
        dm_models['dh_lds'].append(dm_dict[dm_dsid])
    elif str(dm_dsid) in DM_DICT['lv_hds']:
        dm_models['lv_hds'].append(dm_dict[dm_dsid])
    elif str(dm_dsid) in DM_DICT['eft_hds']:
        dm_models['eft_hds'].append(dm_dict[dm_dsid])
    elif str(dm_dsid) in DM_DICT['dh_hds']:
        dm_models['dh_hds'].append(dm_dict[dm_dsid])
    elif str(dm_dsid) in SUSY_dict['SlepSlep']:
        dm_models['SlepSlep'].append(dm_dict[dm_dsid])
    elif str(dm_dsid) in SUSY_dict['Stop']:
        dm_models['Stop'].append(dm_dict[dm_dsid])

for m in models:
    if m != 'SlepSlep' or m != 'Stop':
        newfile = 'DM_Models/DM_Zp_'+str(m)+'.h5'
    else:
        newfile = 'DM_Models/'+str(m)+'.h5'
        
    print('Events in ', m, " : ", np.shape(pd.concat(dm_models[m]))[0])
    print('Making file '+newfile)
    pd.concat(dm_models[m]).to_hdf(save_dir+newfile, key='df_tot')

print("=="*40)
print('Testing time')
print("=="*40)

t0 = time.time()
models = ['SlepSlep', 'Stop']
new_df = pd.read_hdf(save_dir+'bkgs_frfr.h5', key='df_tot')
for m in models:
    new_dm = pd.read_hdf(save_dir+'DM_Models/DM_Zp_'+str(m)+'.h5', key='df_tot')
    print(new_dm)

new_concat = pd.concat([new_df, new_dm]).sort_index()
print(new_concat)

t = "{:.2f}".format(int( time.time()-t0 )/60.)
print( "Time spent making reduced bkg + dsid dataframe: "+str(t)+" min")