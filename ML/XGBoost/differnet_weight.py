import os, time
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

print(xgb.__version__)

t0 = time.time()
start = time.asctime(time.localtime())
print('Started', start)

save_dir = "../../../../storage/racarcam/"
filename = "FULL_DM_50MET.h5"

df = pd.read_hdf(save_dir+filename, key='df_tot')

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                             # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')                                     # Bad variable

# df_features = df_features.loc[df_features['Weight'] > 0]  

df_labels = df_features.pop('Label')

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=test_size, random_state=42)

X_train_w = X_train.pop('Weight')
W_test = X_test.pop('Weight')

W_train = np.ones(len(Y_train))
W_train[Y_train==0] = sum(W_train[Y_train==1])/sum(W_train[Y_train==0])
W_train = pd.DataFrame(W_train, columns=['Weight'])   

# sow_sig = sum(X_train_w[Y_train==1])
# sow_bkg = sum(X_train_w[Y_train==0])

scaler = 1/test_size

model_dir = '../Models/XGB/WGTS/'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass

plot_dir = '../../Plots/TESTING/XGBoost/LVB/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass


def low_stat_Z(sig, bkg):
    Z = np.sqrt(2*( (sig + bkg)*np.log(1 + sig/bkg) - sig ))
    return Z

def stat_unc(prediction, bins, weights):
    binning = np.linspace(0,1,bins+1)
    histo_bins = np.digitize(prediction, binning)
    stat_unc_array = []
    for i in range(1,len(binning)):
        bin_wgt = weights[np.where(histo_bins==i)[0]]
        sow_bin = np.linalg.norm(bin_wgt,2)
        stat_unc_array.append(sow_bin)
    return np.asarray(stat_unc_array)


lamda = 1e-5                                                                # Define hyperparameter
n_est = 200
depth = 6
eta = 0.1

xgbclassifier = xgb.XGBClassifier(
                max_depth=depth, 
                use_label_encoder=False,
                n_estimators=int(n_est),
                learning_rate=0.1,
                reg_lambda = lamda,
                predictor = 'cpu_predictor',
                tree_method = 'hist',
                # scale_pos_weight=sow_bkg/sow_sig,
                objective='binary:logistic',
                eval_metric='auc',
                missing=10,
                random_state=42,
                verbosity = 1) 

xgbclassifier.fit(X_train, Y_train, sample_weight = W_train, verbose = True)
# xgbclassifier.fit(X_train, Y_train, verbose = True)

xgbclassifier.save_model(model_dir+'NN_wgt.txt')

print('==='*20)
t = "{:.2f}".format(int( time.time()-t0 )/60.)
finish = time.asctime(time.localtime())
print('Finished', finish)
print('Total time:', t)
print('==='*20)