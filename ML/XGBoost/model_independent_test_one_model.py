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
print('==='*20)
"""
Choose which model and channel!
"""
parser = argparse.ArgumentParser()
parser.add_argument('--met_reg', type=str, default="50-100", help="MET signal region")
parser.add_argument('--dm_model', type=str, default="DH_HDS", help="Dataset to test")
parser.add_argument('--channel', type=str, default="ee", help="Lepton channel to test")
args = parser.parse_args()

met_reg = args.met_reg
dm_model = args.dm_model
channel = args.channel 



N = 15
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.PuRd_r(np.linspace(0.1,0.95,N)))

model_dsids = []
json_file = open('DM_DICT_Zp_dsid.json')
DM_file = json.load(json_file)
for key in DM_file.keys():
    word = key.split('_')
    model_sec = word[0]+'_'+word[1]
    if model_sec == dm_model.lower():
        model_dsids.append(DM_file[key])

plt.figure(figsize=[8,6])
lw = 2
    

for i in range(len(model_dsids)):
    json_file2 = open('DM_DICT.json')
    model_names = json.load(json_file2)
    save_as = 'mZp_'+model_names[model_dsids[i][0]].split(' ')[-2]+'/'
    save_dir = "/storage/racarcam/"
    bkg_file = save_dir+'bkgs_final.h5'
    sig_file1 = save_dir+'/Zp_DMS/'+model_dsids[i][0]+'.h5'
    sig_file2 = save_dir+'/Zp_DMS/'+model_dsids[i][1]+'.h5'
    data_file = save_dir+'dataFINAL.h5'
    df_bkg = pd.read_hdf(bkg_file, key='df_tot')
    df_sig1 = pd.read_hdf(sig_file1, key='df_tot')
    df_sig2 = pd.read_hdf(sig_file2, key='df_tot')
    df_dat = pd.read_hdf(data_file, key='df')
    df = pd.concat([df_bkg, df_sig1, df_sig2])


    extra_variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50', 'dPhiCloseMet', 'dPhiLeps']


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


    df_features = df_features.loc[df_features['mll'] > 110]                 
    df_data = df_data.loc[df_data['mll'] > 110]                
    
    if met_reg == '50-100':
        df_features = df_features.loc[df_features['met'] < 100]    
        df_data = df_data.loc[df_data['met'] < 100]                     

    elif met_reg == '100-150':
        df_features = df_features.loc[df_features['met'] > 100]                     
        df_features = df_features.loc[df_features['met'] < 150]      
        df_data = df_data.loc[df_data['met'] > 100]    
        df_data = df_data.loc[df_data['met'] < 150]
        
    elif met_reg == '150':
        df_features = df_features.loc[df_features['met'] > 150]        
        df_data = df_data.loc[df_data['met'] > 150] 
    
    print('Doing mll > 110 and met reg '+met_reg+' on '+dm_model+" "+save_as[:-1]+" Z' model on",channel,'channel')
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

    model_dir = '../Models/XGB/Model_independent/'
    xgbclassifier = xgb.XGBClassifier()
    xgbclassifier.load_model(model_dir+met_reg+'.txt')
    
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
    plot_dir = '../../Plots/XGBoost/Model_independent/'+met_reg+'/'+dm_model+'/'+save_as

    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass
    
    
    
    np_dir = '../Data/XGB/'+met_reg+'/'+dm_model+'/'+save_as
    try:
        os.makedirs(np_dir)

    except FileExistsError:
        pass
    
    fpr, tpr, thresholds = roc_curve(Y_test, pred, pos_label=1)
    [sig_pred, bkg_pred], [unc_sig, unc_bkg], data_prediction = scaled_validation(model_dsids[i], pred, W_test, Y_test, DSID_test, data_pred, plot_dir, dm_model=dm_model, channel=channel, met_reg=met_reg)
    unscaled_validation(pred, Y_test, 50, model_dsids[i], plot_dir, channel=channel)
    plt.close('all')
    np.save(np_dir+'sig_pred_'+channel, sig_pred)
    np.save(np_dir+'bkg_pred_'+channel, bkg_pred)
    np.save(np_dir+'unc_sig_'+channel, unc_sig)
    np.save(np_dir+'unc_bkg_'+channel, unc_bkg)
    np.save(np_dir+'data_pred_'+channel, data_prediction)
    np.save(np_dir+'fpr_'+channel, fpr)
    np.save(np_dir+'tpr_'+channel, tpr)

print('==='*20)
t = "{:.2f}".format(int( time.time()-t0 )/60.)
finish = time.asctime(time.localtime())
print('Finished', finish)
print('Total time:', t)
print('==='*20)