import os, time
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print(xgb.__version__)

t0 = time.time()
start = time.asctime(time.localtime())
print('Started', start)

save_dir = "../../../../storage/racarcam/"
bkg_name = 'new_bkgs.h5'
sig_name = "DM_Models/DM_Zp_dh.h5"

df_bkg = pd.read_hdf(save_dir+bkg_name, key='df_tot')
df_sig = pd.read_hdf(save_dir+sig_name, key='df_tot')

df = pd.concat([df_bkg, df_sig])

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
# df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                             # Bad variable
# df_dPhiLeps = df_features.pop('dPhiLeps')                                     # Bad variable
# df_Weight = df_features.pop('Sample_Weight')

# variables = ['n_bjetPt20', 'n_bjetPt30', 'n_bjetPt40', 'n_bjetPt50', 'n_bjetPt60', 'n_ljetPt20', 
#             'n_ljetPt30', 'n_ljetPt40', 'n_ljetPt50', 'n_ljetPt60', 'jetEtaCentral', 'jetEtaForward']

variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50', 'dPhiCloseMet', 'dPhiLeps']

# variables = ['jet1Pt', 'jet1Eta', 'jet1Phi', 'jet2Pt', 'jet2Eta', 'jet2Phi']


df_features = df_features.drop(variables, axis=1)


df_features = df_features.loc[df_features['Weight'] > 0]  

df_labels = df_features.pop('Label')

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=test_size, random_state=42)

X_train_w = X_train.pop('Weight')
X_test_w = X_test.pop('Weight')

W_train = np.ones(len(Y_train))

# W_train[Y_train==0] = X_train_w[Y_train==0]
W_train = X_train_w
sow_sig = sum(W_train[Y_train==1])
sow_bkg = sum(W_train[Y_train==0])




model_dir = '../Models/XGB/WGTS_testing_DH/'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass

# plot_dir = '../../Plots/XGBoost/WEIGHT_TEST/'

# try:
#     os.makedirs(plot_dir)

# except FileExistsError:
#     pass


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
                scale_pos_weight=sow_bkg/sow_sig,
                objective='binary:logistic',
                eval_metric='auc',
                min_child_weight = 1,
                missing=-999,
                random_state=42,
                verbosity = 1) 

xgbclassifier.fit(X_train, Y_train, sample_weight = W_train, verbose = True)
# xgbclassifier.fit(X_train, Y_train, verbose = True)
# model_type = 'only_reweighting_background_SOW_of_MC.txt'
# model_type = 'only_reweighting_background_SOW_of_reweight.txt'
# model_type = 'reweighting_both_SOW_of_MC.txt'
model_type = 'reweighting_both_SOW_of_reweight.txt'

print('Doing', model_type)
xgbclassifier.save_model(model_dir+model_type)

print('==='*20)
t = "{:.2f}".format(int( time.time()-t0 )/60.)
finish = time.asctime(time.localtime())
print('Finished', finish)
print('Total time:', t)
print('==='*20)