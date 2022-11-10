import pandas as pd
from sklearn.model_selection import train_test_split


save_dir = "../../../storage/racarcam/"
# filename = 'Full_DM_sig.h5'
filename = 'Find_Diboson.h5'

df = pd.read_hdf(save_dir+filename, key='df_tot')

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                             
df_dPhiLeps = df_features.pop('dPhiLeps')                                     

df_labels = df_features.pop('Label')

X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
X_train_w = X_train.pop('Weight')
X_test_w = X_test.pop('Weight')

pos = sum(Y_train)
tot = len(Y_train)
neg = tot - pos
print('Showcasing the unbalance between signal and background in', filename, 'dataset\n\n')
print('# sig events:',f"{pos:,}")
print('# bkg events:', f"{neg:,}")
print('# total events:', f"{tot:,}")
print('% of signal:', pos/tot*100, '% of background:', neg/tot*100 )

sow_sig_t = sum(X_train_w[Y_train==1])
sow_bkg_t = sum(X_train_w[Y_train==0])
sow_sig = sum(X_test_w[Y_test==1])
sow_bkg = sum(X_test_w[Y_test==0])

print('Training weight for signal:', f"{sow_sig_t:,}")
print(' for background:',f"{sow_bkg_t:,}")
print('Test weight for signal:', f"{sow_sig:,}")
print('for background:',f"{sow_bkg:,}")

print('Integrated Events*weights for train signal:', f"{pos*sow_sig_t:,}")
print('Integrated Events*weights for train background:', f"{neg*sow_bkg_t:,}")
print('Integrated Events*weights for test signal:', f"{sum(Y_test)*sow_sig:,}")
print('Integrated Events*weights for test background:', f"{(len(Y_test)-sum(Y_test))*sow_bkg:,}")
