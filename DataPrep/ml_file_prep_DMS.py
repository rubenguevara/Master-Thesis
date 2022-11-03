import pandas as pd
import numpy as np
import os, time, json

save_dir = "../../../storage/racarcam/"
filename = "DM_Run2_50MET.h5"   # Change file if needed

sow_bkg_file = open('SOW_bkg.json')
SOW_bkg = json.load(sow_bkg_file)
sow_sig_file = open('SOW_sig_AFII.json')
SOW_sig = json.load(sow_sig_file)

real_df = pd.read_hdf(save_dir+filename, key='df_tot')

df = real_df.copy()
cols = df.columns.tolist()

print(df)
df.rename(columns = {'Weight':'OldWeight'},inplace = True)

new_weights = []
for w, dsid, mcrun, label in zip(df['OldWeight'], df['RunNumber'], df['RunPeriod'], df['Label']):
    if label == 0:
        sow = SOW_bkg[mcrun][str(dsid)]
    elif label == 1:
        sow = SOW_sig[mcrun][str(dsid)]
    
    if mcrun == 'mc16a':
        lumi = 36.2
    elif mcrun == 'mc16d':
        lumi = 44.3        
    elif mcrun == 'mc16e':
        lumi = 58.5
    wgt = lumi*w/sow
    new_weights.append(wgt)

old = df.pop('OldWeight')
df['Weight'] = new_weights
df = df[cols]
print(df)

sow_bkg_file.close()
sow_sig_file.close()

dm_df = df.loc[df['Label'] == 1]
df = df.drop(df[df['Label'] == 1].index)

df.to_hdf(save_dir+'bkgs.h5', key='df_tot')
print(dm_df)
print(df)

print("=="*40)
print('Making files for each DSID')
print("=="*40)

dm_dsids = dm_df['RunNumber'].unique()
dm_dict = {dm_dsid: dm_df.loc[dm_df['RunNumber'] == dm_dsid] for dm_dsid in dm_dsids}

for dm_dsid in dm_dsids:
    newfile = 'DMS/'+str(dm_dsid)+'.h5'
    print('Events in ', dm_dsid, " : ", np.shape(dm_dict[dm_dsid])[0])
    print('Making file '+newfile)
    dm_dict[dm_dsid].to_hdf(save_dir+newfile, key='df_tot')

print("=="*40)
print('Testing time')
print("=="*40)

t0 = time.time()

new_df = pd.read_hdf(save_dir+'bkgs.h5', key='df_tot')
new_dm = pd.read_hdf(save_dir+'DMS/'+str(dm_dsid)+'.h5', key='df_tot')

new_concat = pd.concat([new_df, new_dm]).sort_index()
print(new_concat)

t = "{:.2f}".format(int( time.time()-t0 )/60.)
print( "Time spent making reduced bkg + dsid dataframe: "+str(t)+" min")