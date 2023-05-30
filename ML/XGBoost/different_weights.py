import os, argparse
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Plot_maker import scaled_validation, ROC_curve, feature_importance, expected_significance
from EventIDs import IDs

print(xgb.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default="Pos_wgt", help="Normalization method")
args = parser.parse_args()

method = args.method

dm_model = 'DH_HDS'

save_dir = "/storage/racarcam/"
bkg_file = save_dir+'bkgs_final.h5'
sig_file = save_dir+'/DM_Models/DM_Zp_'+dm_model.lower()+'.h5'
datafile = save_dir+'dataFINAL.h5'

df_dat = pd.read_hdf(datafile, key='df')
df_bkg = pd.read_hdf(bkg_file, key='df_tot')
df_sig = pd.read_hdf(sig_file, key='df_tot')

df = pd.concat([df_bkg, df_sig])
df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
# df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')

variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50', 'dPhiCloseMet', 'dPhiLeps']
df_features = df_features.drop(variables, axis=1)

if method == 'Pos_wgt':
    df_features = df_features.loc[df_features['Weight'] > 0]  

df_labels = df_features.pop('Label')
test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=test_size, random_state=42)

X_train_w = X_train.pop('Weight')
X_test_w = X_test.pop('Weight')
DSID_train = X_train.pop('RunNumber')
DSID_test = X_test.pop('RunNumber')

if method == 'Abs_wgt':
    sclr = sum(X_train_w)/sum(abs(X_train_w))
    X_train_w = abs(X_train_w)*sclr

W_train = np.ones(len(Y_train))

W_train[Y_train==0] = X_train_w[Y_train==0]
sow_sig = sum(W_train[Y_train==1])
sow_bkg = sum(W_train[Y_train==0])


plot_dir = '../../Plots/XGBoost/Weighting_methods_full_data/'+method+'/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass


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

if method == 'No_wgt':
    xgbclassifier.fit(X_train, Y_train, verbose = True)
else:
    xgbclassifier.fit(X_train, Y_train, sample_weight = W_train, verbose = True)

print('Doing', method)

            
df_data = df_dat.copy()
df_EventID = df_data.pop('EventID')
df_RunNumber = df_data.pop('RunNumber')
df_Dileptons = df_data.pop('Dileptons')
df_RunPeriod = df_data.pop('RunPeriod')
df_data = df_data.drop(variables, axis=1)

data_test_size = 0.1
data_train, data_test = train_test_split(df_data, test_size=data_test_size, random_state=42)


IDs['DH_HDS_130'] = [514560,514561]
DH_HDS_130 = IDs["all_bkg"] + IDs['DH_HDS_130']
remove_non_DH_HDS_130 =  np.isin(DSID_test, DH_HDS_130)
X_test = X_test[remove_non_DH_HDS_130]
Y_test = Y_test[remove_non_DH_HDS_130]
DSIDs = DSID_test[remove_non_DH_HDS_130]
W_test = X_test_w[remove_non_DH_HDS_130]

y_pred_prob = xgbclassifier.predict_proba(X_test)
data_pred_prob = xgbclassifier.predict_proba(data_test)

pred = y_pred_prob[:,1]
data_pred = data_pred_prob[:,1]
    

[sig_pred, bkg_pred], [unc_sig, unc_bkg_sav], data_hist = scaled_validation([514560,514561], pred, W_test, Y_test, DSID_test, data_pred, plot_dir, dm_model='DH HDS')
# ROC_curve(Y_test, pred, [514560,514561], plot_dir)
# feature_importance(xgbclassifier, plot_dir)
# expected_significance(sig_pred, bkg_pred, "DH HDS $m_{Z'}$ 130 GeV", 'DH HDS', plot_dir)
