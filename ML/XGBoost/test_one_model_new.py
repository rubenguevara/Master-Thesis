import os, time, json, argparse
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Plot_maker import scaled_validation, unscaled_validation, expected_significance
from sklearn.metrics import roc_curve, auc

print(xgb.__version__)

t0 = time.time()
start = time.asctime(time.localtime())
print('Started', start)

"""
Choose which model and channel!
"""
parser = argparse.ArgumentParser()
parser.add_argument('--dm_model', type=str, default="DH_HDS", help="Dataset to test")
parser.add_argument('--channel', type=str, default="ee", help="Lepton channel to test")
args = parser.parse_args()

dm_model = args.dm_model
channel = args.channel 

# dm_model = 'LV_HDS'
# channel = 'uu'


N = 15
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.PuRd_r(np.linspace(0.1,0.95,N)))

model_dsids = []
json_file = open('DM_DICT_SUSY_dsid.json')
json_file2 = open('DM_DICT_SUSY_models.json')
dsids = json.load(json_file)
dsids = dsids[dm_model]
model_names = json.load(json_file2)
# plt.figure(figsize=[8,6])
# lw = 2

for i in dsids:
    save_as = str(model_names[i][0]).replace(' ', '_')+'/'
    if 'MET75' in save_as: continue
    if i != '508106': continue
    save_dir = "/storage/racarcam/"
    bkg_file = save_dir+'bkgs_frfr.h5'
    sig_file1 = save_dir+'/DMS_dsid/'+i+'.h5'
    data_file = save_dir+'datafrfr.h5'
    df_bkg = pd.read_hdf(bkg_file, key='df_tot')
    df_sig1 = pd.read_hdf(sig_file1, key='df_tot')
    df_dat = pd.read_hdf(data_file, key='df')
    df = pd.concat([df_bkg, df_sig1])


    extra_variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50', 'dPhiCloseMet', 'dPhiLeps', 'isOS']


    df_features = df.copy()
    df_EventID = df_features.pop('EventID')
    df_CrossSection = df_features.pop('CrossSection')
    # df_RunNumber = df_features.pop('RunNumber')
    # df_Dileptons = df_features.pop('Dileptons')
    df_RunPeriod = df_features.pop('RunPeriod')
    df_features = df_features.drop(extra_variables, axis=1)
                
    df_data = df_dat.copy()
    df_EventID = df_data.pop('EventID')
    df_RunNumber = df_data.pop('RunNumber')
    # df_Dileptons = df_data.pop('Dileptons')
    df_RunPeriod = df_data.pop('RunPeriod')
    df_data = df_data.drop(extra_variables, axis=1)


    # df_features = df_features.loc[df_features['mll'] > 110]                 
    # df_data = df_data.loc[df_data['mll'] > 110]                             
    print('Doing mll > 10 and met > 50 on '+save_as[:-5]+" model on",channel,'channel')
    df_labels = df_features.pop('Label')
    test_size = 0.2
    data_test_size = 0.1
    X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=test_size, random_state=42)
    data_train, data_test = train_test_split(df_data, test_size=data_test_size, random_state=42)
    W_train = X_train.pop('Weight')
    W_test = X_test.pop('Weight')
    
    DSID_train = X_train.pop('RunNumber')
    DSID_test = X_test.pop('RunNumber')
    Dilepton_train = X_train.pop('Dileptons')
    Dilepton_test = X_test.pop('Dileptons')
    dilepton_data = data_test.pop('Dileptons')

    scaler = 1/test_size
    data_scaler = 1/data_test_size

    model_dir = '../Models/XGB/Model_Dependent/'
    xgbclassifier = xgb.XGBClassifier()
    xgbclassifier.load_model(model_dir+'best_'+dm_model+'.txt')
    
    Y_test = Y_test[Dilepton_test==channel]
    W_test = W_test[Dilepton_test==channel]
    DSID_test = DSID_test[Dilepton_test==channel]
    X_test = X_test[Dilepton_test==channel]
    data_test = data_test[dilepton_data==channel]
    
    y_pred_prob = xgbclassifier.predict_proba(X_test)
    data_pred_prob = xgbclassifier.predict_proba(data_test)

    pred = y_pred_prob[:,1]
    data_pred = data_pred_prob[:,1]
    data_w = np.ones(len(data_pred))*10
    n_bins = 50
    if dm_model == 'Stop': 
        dm_model = '2HDM'
    plot_dir = '../../Plots/XGBoost/'+dm_model+'/'+save_as[:-5]+'/'

    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass
    
    
    np_dir = save_dir+'Data/'+dm_model+'_scores/'+save_as[:-5]+'/'

    try:
        os.makedirs(np_dir)

    except FileExistsError:
        pass
    [sig_pred, bkg_pred], [unc_sig, unc_bkg], data_prediction = scaled_validation([save_as[:-5]], pred, W_test, Y_test, DSID_test, data_pred, plot_dir, dm_model=dm_model, channel=channel)
    unscaled_validation(pred, Y_test, 50, [save_as[:-5]], plot_dir, channel=channel)
    plt.close('all')
    np.save(np_dir+'sig_pred_'+channel, sig_pred)
    np.save(np_dir+'bkg_pred_'+channel, bkg_pred)
    np.save(np_dir+'unc_sig_'+channel, unc_sig)
    np.save(np_dir+'unc_bkg_'+channel, unc_bkg)
    np.save(np_dir+'data_pred_'+channel, data_prediction)
    fpr, tpr, thresholds = roc_curve(Y_test, pred, pos_label=1)
    np.save(np_dir+'fpr_'+channel, fpr)
    np.save(np_dir+'tpr_'+channel, tpr)

    if dm_model == '2HDM': 
        dm_model = 'Stop'