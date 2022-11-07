import pandas as pd
import numpy as np
import os, json

save_dir = "../../../storage/racarcam/"
filename = "DM_Run2_50MET.h5"   # Change file if needed

df = pd.read_hdf(save_dir+filename, key='df_tot')
print(df)

print("=="*40)
print('Weight fixing')
print("=="*40)

sow_bkg_file = open('SOW_bkg.json')
SOW_bkg = json.load(sow_bkg_file)
sow_sig_file = open('SOW_sig_AFII.json')
SOW_sig = json.load(sow_sig_file)

cols = df.columns.tolist()

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

newfile = 'Full_DM_sig.h5'

if os.path.exists(save_dir+newfile):
    print('Rewriting h5 file')
    os.remove(save_dir+newfile)

df.to_hdf(save_dir+newfile, key='df_tot')