import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os, json
from EventIDs import IDs
import multiprocessing as mp
from matplotlib import ticker as mticker

print(xgb.__version__)

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


save_dir = "/storage/racarcam/"
filename = "new_bkgs.h5"
datafile = 'new_data.h5'

"""Choose model and signal region here!"""
dm_model = 'DH'
# met_reg = '50-100'
# met_reg = '100-150'
met_reg = '150'
xgb_model = 'best_'+dm_model+'_'+met_reg+'_MET.txt'

variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50', 'dPhiCloseMet', 'dPhiLeps']

df_data = pd.read_hdf(save_dir+datafile, key='df')

dataID = df_data.pop('EventID') 
dataRN = df_data.pop('RunNumber') 
dataRP = df_data.pop('RunPeriod') 
dataDL = df_data.pop('Dileptons')
df_data = df_data.drop(variables, axis=1)

df_data = df_data.loc[df_data['mll'] > 120]                     

if met_reg == '50-100':
    df_data = df_data.loc[df_data['met'] < 100]                     

elif met_reg == '100-150':
    df_data = df_data.loc[df_data['met'] > 100]                     
    df_data = df_data.loc[df_data['met'] < 150]      
    
elif met_reg == '150':
    df_data = df_data.loc[df_data['met'] > 150]                     

data_train, data_test = train_test_split(df_data, test_size = 0.1, random_state = 42)


model_dir = '../Models/XGB/'+xgb_model

dm_dict_file = open('../DM_DICT.json')
DM_DICT = json.load(dm_dict_file)

df_bkgs = pd.read_hdf(save_dir+filename, key='df_tot')

def plot_maker(dsid_int):
    dsid1 = str(dsid_int[0])
    dsid2 = str(dsid_int[1])
    dsid_name = DM_DICT[dsid1].split(' ')
    dsid_title = dsid_name[0] +' '+ dsid_name[1] +' '+ dsid_name[2] +' '+ dsid_name[3]
    dsid_save = dsid_name[0] +'_'+ dsid_name[1] + '_mZp_' + dsid_name[3]
    
    print('Doing', dsid_save, 'on', dm_model, 'network with', met_reg ,'MET cut')

    df_dm = pd.read_hdf(save_dir+'/Zp_DMS/'+dsid1+'.h5', key='df_tot')
    df_dm2 = pd.read_hdf(save_dir+'/Zp_DMS/'+dsid2+'.h5', key='df_tot')
    
    df = pd.concat([df_bkgs, df_dm, df_dm2]).sort_index()
    
    df_features = df.copy()
    df_EventID = df_features.pop('EventID')
    df_Dileptons = df_features.pop('Dileptons')
    df_CrossSection = df_features.pop('CrossSection')
    df_RunPeriod = df_features.pop('RunPeriod')
    df_features = df_features.drop(variables, axis=1)
    
    df_features = df_features.loc[df_features['mll'] > 120]                             
    if met_reg == '50-100':
        df_features = df_features.loc[df_features['met'] < 100]                     

    elif met_reg == '100-150':
        df_features = df_features.loc[df_features['met'] > 100]                     
        df_features = df_features.loc[df_features['met'] < 150]      
        
    elif met_reg == '150':
        df_features = df_features.loc[df_features['met'] > 150]                     
    
    df_labels = df_features.pop('Label')
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size = test_size, random_state = 42)
    W_train = X_train.pop('Weight')
    W_test = X_test.pop('Weight')
    DSID_train = X_train.pop('RunNumber')
    DSID_test = X_test.pop('RunNumber')
    
    scaler = 1/test_size
    xgbclassifier = xgb.XGBClassifier()
    xgbclassifier.load_model(model_dir)
    
    y_pred_prob = xgbclassifier.predict_proba(X_test)
    data_pred_prob = xgbclassifier.predict_proba(data_test)
    
    pred = y_pred_prob[:,1]
    data_pred = data_pred_prob[:,1]
    data_w = np.ones(len(data_pred))*10
    n_bins = 50
    
    ### Plotting
    
    DY = []; ST = []; DB = []; W = []; TT = [];
    DY_w = []; ST_w = []; DB_w = []; W_w = []; TT_w = [];
    for DSID, output, w in zip(DSID_test, pred, W_test):
        if DSID in IDs["DY"]:
            DY.append(output)
            DY_w.append(w*scaler)
        elif DSID in IDs['Single_top']:
            ST.append(output)
            ST_w.append(w*scaler)
        elif DSID in IDs["Diboson"]:
            DB.append(output)
            DB_w.append(w*scaler)
        elif DSID in IDs["W"]:
            W.append(output)
            W_w.append(w*scaler)
        elif DSID in IDs["TTbar"]:
            TT.append(output)
            TT_w.append(w*scaler)

    hist = [W, DB, TT, ST, DY]
    hist_w = [W_w, DB_w, TT_w, ST_w, DY_w]
    colors = ['#218C8D', '#6CCECB', '#F9E559', '#EF7126', '#8EDC9D']
    labels = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan']
    
    plot_dir = '../../Plots/XGBoost/'+dm_model+'_'+met_reg+'_MET/'+dsid_save+'/'

    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass
    
    bkg_pred, bins, patches = plt.hist(pred[Y_test==0], weights = W_test[Y_test==0]*scaler, bins = n_bins)
    sig_pred, bins, patches = plt.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = n_bins)
    data_hist, bins, patches = plt.hist(data_pred, bins = n_bins)

    # unc_data = np.sqrt(data_hist)*10
    # data_hist = data_hist*10
    # # print(unc_data)
    # # print(data_hist)
    # # ratio_data = unc_data/data_hist
    # # print(ra)
    # # unc_data_plot =  []
    # # for k in len(data_hist):
    # #     if ratio_data[k] > 1:
    # #         unc_data_plot.append(data_hist[k])
    # #     else: unc_data_plot.append(unc_data[k])
    # # print(unc_data_plot)

    
    # stat_unc_bkgs = stat_unc(np.asarray(pred[Y_test==0]), 50, np.asarray(W_test[Y_test==0]*scaler))
    # syst_unc_bkgs = bkg_pred*0.3                                                    # Assuming 30% systematic uncertainty
    # unc_bkg = np.sqrt(stat_unc_bkgs**2 + syst_unc_bkgs**2)
    
    # np.seterr(divide='ignore', invalid='ignore')                                    # Remove true divide message
    # ratio = data_hist/bkg_pred
    # unc_ratio_stat = ratio*np.sqrt( (unc_data/data_hist)**2 + (stat_unc_bkgs/bkg_pred)**2)
    # unc_ratio = ratio*np.sqrt( (unc_data/data_hist)**2 + (unc_bkg/bkg_pred)**2)
    
    # plt.clf()

    # x_axis = np.linspace(0.5/n_bins, 1-0.5/n_bins, n_bins)
    # line = np.linspace(0, 1, 100)
    # width = x_axis[1]-x_axis[0]
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    # fig.subplots_adjust(hspace=0.04)
    # n, bins, patches = ax1.hist(hist, weights = hist_w, bins = n_bins, label = labels, histtype='barstacked', color=colors, zorder = 0)
    # n, bins, patches = ax1.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = n_bins, color='#F42069', label="Signal", zorder = 5, histtype='step')
    # ax1.bar(x_axis, 2*unc_bkg, bottom=bkg_pred-unc_bkg, fill=False, hatch='XXXXX', label='Stat. + Syst. Unc.', width = width, lw=0.0, alpha=0.3)
    # ax1.text(0.15, max(bkg_pred), '$\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, $m_{ll}>120$ GeV')
    # if met_reg == '50-100' :
    #     ax1.text(0.15, max(bkg_pred)/2.5, '100 GeV > $E_{T}^{miss}$ > 50 GeV')
    # if met_reg == '100-150' :
    #     ax1.text(0.15, max(bkg_pred)/2.5, '$150$ GeV > $E_{T}^{miss}$ > 100 GeV')
    # if met_reg == '150' :
    #     ax1.text(0.15, max(bkg_pred)/2.5, '$>150$ GeV $E_{T}^{miss}$')
    # ax1.errorbar(x_axis, data_hist, yerr = unc_data, fmt='o', color='black', label='Data', zorder = 10, ms=3, lw=1, capsize=2, lolims=0)
    # ax1.set_ylabel('Events')
    # ax1.set_yscale('log')
    # ax1.set_xlim([0,1])
    # ax1.set_ylim([2e-3, max(bkg_pred)*5])
    # ax1.legend(ncol=2)
    # ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    # ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    # ax1.tick_params(bottom=True, top=True, left=True, right=True, which='both')
    # ax2.set_ylabel('Events / Bkg')
    # ax2.errorbar(x_axis, ratio, yerr = unc_ratio_stat, fmt='o', color='black', ms=3, lw=1, lolims=0)
    # ax2.plot(line, np.ones(len(line)), linestyle='-', color='black', lw=2, alpha=0.3)
    # ax2.bar(x_axis, 2*unc_ratio, bottom=ratio-unc_ratio, color='grey', width = width, lw=0.0, alpha=0.3)
    # ax2.grid(axis='y')
    # ax2.set_xlim([0,1])
    # ax2.set_ylim([0.5, 1.5])
    # ax2.set_xlabel('XGBoost output')
    # fig.suptitle('XGBoost output, '+dsid_title+' dataset, validation data with 30 % syst. unc.\n Network trained on '+dm_model+' model with $m_{ll} > 120$ GeV and in '+met_reg+' MET signal region', fontsize='x-large')
    # plt.savefig(plot_dir+'VAL.pdf')
    # plt.clf()

    fpr, tpr, thresholds = roc_curve(Y_test, pred, pos_label=1)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.02])
    plt.ylim([-0.01, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for XGBoost on '+dsid_title+' dataset')
    plt.legend(loc="lower right")
    plt.savefig(plot_dir+'ROC.pdf')
    plt.clf()

    
    test_binning = [0, 1/50, 0.1, 0.3, 0.6, 1]
    plt.figure(figsize=[10,6])
    n, bins, patches = plt.hist(pred[Y_test==0], weights = W_test[Y_test==0]*scaler, bins = test_binning, facecolor='blue', alpha=0.2,label="Background")
    n, bins, patches = plt.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = test_binning, facecolor='red', alpha=0.2, label="Signal")
    plt.xlabel('XGBoost output')
    plt.ylabel('Events')
    plt.yscale('log')
    plt.xlim([0,1])
    plt.title('XGBoost output, '+dsid_title+' dataset, uscaled validation data')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir+'VAL_unscaled.pdf')
    plt.clf()    

    Y_axis = [low_stat_Z(sum(sig_pred[25:]), sum(bkg_pred[25:])),          
                low_stat_Z(sum(sig_pred[30:]), sum(bkg_pred[30:])), 
                low_stat_Z(sum(sig_pred[35:]), sum(bkg_pred[35:])),
                low_stat_Z(sum(sig_pred[40:]), sum(bkg_pred[40:])), 
                low_stat_Z(sum(sig_pred[45:]), sum(bkg_pred[45:])), 
                low_stat_Z(sum(sig_pred[49:]), sum(bkg_pred[49:]))]
    
    X_axis = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    
    plt.figure(figsize=[10,6])
    plt.plot(X_axis, Y_axis, linestyle='--')
    plt.scatter(X_axis, Y_axis, label = 'Output cut')
    plt.xlim([0,1])
    plt.grid(True)
    plt.legend()
    plt.ylabel('Expected significance [$\sigma$]')
    plt.title("Significance, trained network on "+dsid_title)
    plt.xlabel('XGBoost output')
    plt.savefig(plot_dir+'EXP_SIG.pdf')
    plt.clf()
    
    plt.close('all')
    """
    plot_dir = plot_dir+'feature_importance/'
    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass
    
    fig, ax = plt.subplots(1, 1, figsize = [10, 6])
    xgb.plot_importance(booster = xgbclassifier, ax=ax, show_values= False, xlabel='Number of times features appears in tree')
    plt.savefig(plot_dir+'weight.pdf')
    plt.clf()
    
    fig, ax = plt.subplots(1, 1, figsize = [10, 6])
    xgb.plot_importance(booster = xgbclassifier, ax=ax, show_values= False, xlabel='Average gain of splits which use the feature', importance_type = 'gain')
    plt.savefig(plot_dir+'gain.pdf')
    plt.clf()
    
    fig, ax = plt.subplots(1, 1, figsize = [10, 6])
    xgb.plot_importance(booster = xgbclassifier, ax=ax, show_values= False, xlabel='Total gain of splits which use the feature', importance_type = 'total_gain')
    plt.savefig(plot_dir+'total_gain.pdf')
    plt.clf()
    
    fig, ax = plt.subplots(1, 1, figsize = [10, 6])
    xgb.plot_importance(booster = xgbclassifier, ax=ax, show_values= False, xlabel='Average coverage of splits which use the feature', importance_type = 'cover')
    fig.suptitle('Coverage is defined as the number of samples affected by the split')
    plt.savefig(plot_dir+'cover.pdf')
    plt.clf()
    
    fig, ax = plt.subplots(1, 1, figsize = [10, 6])
    xgb.plot_importance(booster = xgbclassifier, ax=ax, show_values= False, xlabel='Total coverage of splits which use the feature', importance_type = 'total_cover')
    fig.suptitle('Coverage is defined as the number of samples affected by the split')
    plt.savefig(plot_dir+'total_cover.pdf')
    plt.close('all')
    """

DH_dsids = []
LV_dsids = []
EFT_dsids = []

keys = DM_DICT.keys()
for dsid in keys:
    if 'DH' in DM_DICT[dsid]:
        DH_dsids.append(dsid)
    if 'LV' in DM_DICT[dsid]:
        LV_dsids.append(dsid)
    if 'EFT' in DM_DICT[dsid]:
        EFT_dsids.append(dsid)

DH_dsids.sort()
LV_dsids.sort()
EFT_dsids.sort()

dsid_list = [[DH_dsids[i], DH_dsids[i+1]] for i in range(0, len(DH_dsids), 2)]
    
# with mp.Pool(processes=len(dsid_list)) as pool:
#     pool.map(plot_maker, dsid_list)
# pool.close()

# for dsids in dsid_list:
#     plot_maker(dsids)
plot_maker(dsid_list[0])