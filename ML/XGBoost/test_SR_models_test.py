import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os, json
import multiprocessing as mp
from Plot_maker import *

print(xgb.__version__)

file_dir = "/storage/racarcam/"
filename = "new_bkgs.h5"
datafile = 'new_data.h5'

"""Choose model and signal region here!"""
dm_model = 'DH'
# met_reg = '50-100'
# met_reg = '100-150'
# met_reg = '150'
# xgb_model = 'Re-weighted/best_'+dm_model+'_'+met_reg+'_MET_no_lightJets.txt'
# model_type = 'only_reweighting_background_SOW_of_MC.txt'
# model_type = 'only_reweighting_background_SOW_of_reweight.txt'
# model_type = 'reweighting_both_SOW_of_MC.txt'
model_type = 'reweighting_both_SOW_of_reweight.txt'
# model_type = 'only_MC_balance.txt'
xgb_model = 'WGTS_testing_DH/' + model_type


variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50', 'dPhiCloseMet', 'dPhiLeps', 'jetLight']

df_data = pd.read_hdf(file_dir+datafile, key='df')

dataID = df_data.pop('EventID') 
dataRN = df_data.pop('RunNumber') 
dataRP = df_data.pop('RunPeriod') 
dataDL = df_data.pop('Dileptons')
df_data = df_data.drop(variables, axis=1)

# df_data = df_data.loc[df_data['mll'] > 120]                     

# if met_reg == '50-100':
#     df_data = df_data.loc[df_data['met'] < 100]                     

# elif met_reg == '100-150':
#     df_data = df_data.loc[df_data['met'] > 100]                     
#     df_data = df_data.loc[df_data['met'] < 150]      
    
# elif met_reg == '150':
#     df_data = df_data.loc[df_data['met'] > 150]                     
data_test_size = 0.1
data_train, data_test = train_test_split(df_data, test_size = data_test_size, random_state = 42)
data_scaler = 1/data_test_size

model_dir = '../Models/XGB/'+xgb_model

dm_dict_file = open('../DM_DICT.json')
DM_DICT = json.load(dm_dict_file)

df_bkgs = pd.read_hdf(file_dir+filename, key='df_tot')


# binning = [0, 1/50, 0.2, 0.4, 0.6, 1]
# binning = [0, 1/50, 0.06, 0.15, 0.3, 1]

# def predictor(dsid_int, save_dir, met_region, dm_mod, varis):
def predictor(dsid_int, save_dir, dm_mod, varis):
    dsid1 = str(dsid_int[0])
    dsid2 = str(dsid_int[1])
    dsid_name = DM_DICT[dsid1].split(' ')
    dsid_title = dsid_name[0] +' '+ dsid_name[1] +' '+ dsid_name[2] +' '+ dsid_name[3]
    dsid_save = dsid_name[0] +'_'+ dsid_name[1] + '_mZp_' + dsid_name[3]
    
    print('Doing', dsid_save, 'on', dm_mod, 'network')# with', met_reg ,'MET cut')

    df_dm = pd.read_hdf(file_dir+'/Zp_DMS/'+dsid1+'.h5', key='df_tot')
    df_dm2 = pd.read_hdf(file_dir+'/Zp_DMS/'+dsid2+'.h5', key='df_tot')
    
    df = pd.concat([df_bkgs, df_dm, df_dm2]).sort_index()
    
    df_features = df.copy()
    df_EventID = df_features.pop('EventID')
    df_Dileptons = df_features.pop('Dileptons')
    df_CrossSection = df_features.pop('CrossSection')
    df_RunPeriod = df_features.pop('RunPeriod')
    df_features = df_features.drop(varis, axis=1)
    
    # df_features = df_features.loc[df_features['mll'] > 120]                             
    # if met_region == '50-100':
    #     df_features = df_features.loc[df_features['met'] < 100]                     

    # elif met_region == '100-150':
    #     df_features = df_features.loc[df_features['met'] > 100]                     
    #     df_features = df_features.loc[df_features['met'] < 150]      
        
    # elif met_region == '150':
    #     df_features = df_features.loc[df_features['met'] > 150]                     
    
    df_labels = df_features.pop('Label')
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size = test_size, random_state = 42)
    # X_test = df_features
    # Y_test = df_labels
    # W_train = X_train.pop('Weight')
    W_test = X_test.pop('Weight')
    # DSID_train = X_train.pop('RunNumber')
    DSID_test = X_test.pop('RunNumber')
    
    scaler = 1/test_size
    xgbclassifier = xgb.XGBClassifier()
    xgbclassifier.load_model(model_dir)
    
    y_pred_prob = xgbclassifier.predict_proba(X_test)
    data_pred_prob = xgbclassifier.predict_proba(data_test)
    # data_pred_prob = xgbclassifier.predict_proba(df_data)
    
    pred = y_pred_prob[:,1]
    data_pred = data_pred_prob[:,1]
    data_w = np.ones(len(data_pred))
    
    # wonky_bin_dir = '../Data/wonky-bin/'
    # try:
    #     os.makedirs(wonky_bin_dir)

    # except FileExistsError:
    #     pass
    # np.save(wonky_bin_dir+'data_pred', data_pred)
    # np.save(wonky_bin_dir+'pred', pred)
    # X_test.to_pickle(wonky_bin_dir+'X_test.pkl')
    # df_data.to_pickle(wonky_bin_dir+'df_data.pkl')
    # np.save(wonky_bin_dir+'Y_test', Y_test)
    # np.save(wonky_bin_dir+'W_test', W_test)
    # np.save(wonky_bin_dir+'DSID_test', DSID_test)
    
    # plot_dir = save_dir+'Re-weighted/'+dm_model+'_'+met_reg+'_MET_binned/' 
    plot_dir = save_dir+'New_wgts_DH/' +model_type+'/'
    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass
    # variables = MC_plot.columns
    
    # [sig_hist, bkg_hist], [sig_unc, bkg_unc] = scaled_validation(dsid_int, pred, W_test, Y_test, DSID_test, data_pred, plot_dir, 50, met_region, dm_mod, scaler = scaler, data_scaler=data_scaler)
    [sig_hist, bkg_hist], [sig_unc, bkg_unc] = scaled_validation(dsid_int, pred, W_test, Y_test, DSID_test, data_pred, plot_dir, 50, dm_model = dm_mod, scaler = scaler, data_scaler=data_scaler)
    exit()
    mc_bool = [i>0.3 for i in pred]
    data_bool = [i>0.3 for i in data_pred]
    MC_plot = X_test[mc_bool]
    MC_Y = Y_test[mc_bool]
    MC_plot = MC_plot[MC_Y==0]
    MC_dsid = DSID_test[mc_bool]
    MC_dsid = MC_dsid[MC_Y==0]
    MC_wgt = W_test[mc_bool]
    MC_wgt = MC_wgt[MC_Y==0]
    data_plot = df_data[data_bool]
    print(MC_plot)
    print(data_plot)
    
    plot_dir = save_dir+dm_model+'_'+met_reg+'_MET/TEST_1_binned/variables/' 
    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass
    variables = MC_plot.columns
    
    for k in variables:
        print('Plotting', k)
        distribution(MC_plot, MC_dsid, MC_wgt, data_plot, k, plot_dir)
    
    # data_bool = [i>0.3 for i in data_pred]
    
    exit()
    # np.save('../Data/MC_pred', pred[Y_test==0])
    # np.save('../Data/data_pred', data_pred)
    
    
    # pred_dir_d = sav_dir+dm_mod+'-'+met_region+'-MET/Data/'
    # pred_dir_mc = sav_dir+dm_mod+'-'+met_region+'-MET/MC/'
    # pred_dir_Y = sav_dir+dm_mod+'-'+met_region+'-MET/Label/'
    # pred_dir_ID = sav_dir+dm_mod+'-'+met_region+'-MET/DSID/'
    # pred_dir_wgt = sav_dir+dm_mod+'-'+met_region+'-MET/WGT/'
    
    plot_dir = save_dir+dm_model+'_'+met_reg+'_MET/TEST_1_binned/'
    
    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass
    
    # try:
    #     os.makedirs(pred_dir_d)

    # except FileExistsError:
    #     pass
    
    # try:
    #     os.makedirs(pred_dir_mc)

    # except FileExistsError:
    #     pass
    
    # try:
    #     os.makedirs(pred_dir_Y)

    # except FileExistsError:
    #     pass
    
    # try:
    #     os.makedirs(pred_dir_ID)

    # except FileExistsError:
    #     pass
    
    # try:
    #     os.makedirs(pred_dir_wgt)

    # except FileExistsError:
    #     pass
    
    # np.save(pred_dir_d+dsid_save, data_pred)
    # np.save(pred_dir_mc+dsid_save, pred)
    # np.save(pred_dir_Y+dsid_save, Y_test)
    # np.save(pred_dir_ID+dsid_save, DSID_test)
    # np.save(pred_dir_wgt+dsid_save, W_test)
    
    # return dsid_save
    [sig_hist, bkg_hist], [sig_unc, bkg_unc] = scaled_validation(dsid_int, pred, W_test, Y_test, DSID_test, data_pred, plot_dir, binning, met_region, dm_mod, scaler = scaler, data_scaler=1)
    # ROC_curve(Y_test, pred, dsid_int, plot_dir)
    unscaled_validation(pred, Y_test, binning, dsid_int, plot_dir)
    # feature_importance(xgbclassifier, plot_dir)
    return [sig_hist, bkg_hist], [sig_unc, bkg_unc]



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

# save_ids = []
directory = '../Data/predictions/'
directory = '../../Plots/XGBoost/'

# for k in range(len(dsid_list)):
#     ids = predictor(dsid_list[k], directory, met_reg, dm_model, variables)
# #     save_ids.append(ids)
# save_ids = np.load(directory+'save_ids.npy')
# print(save_ids)

# [sig_hist, bkg_hist], [sig_unc, bkg_unc] = predictor(dsid_list[0], directory, met_reg, dm_model, variables)
[sig_hist, bkg_hist], [sig_unc, bkg_unc] = predictor(dsid_list[0], directory, dm_model, variables)
# np.save(directory+'save_ids', save_ids)
exit()


def poop():
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
    
    plot_dir = '../../Plots/XGBoost/'+dm_model+'_'+met_reg+'_MET/'+dsid_save+'/ACC/'

    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass
    
    
    x_axis = []
    widths = []
    for i in range(len(binning) - 1):
        bin_width = binning[i+1] - binning[i]
        bin_center = binning[i] + bin_width/2
        widths.append(bin_width)
        x_axis.append(bin_center)
    
    bkg_pred, bins, patches = plt.hist(pred[Y_test==0], weights = W_test[Y_test==0]*scaler, bins = binning)
    sig_pred, bins, patches = plt.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = binning)
    data_hist, bins, patches = plt.hist(data_pred, weights = data_w, bins = binning)
    unc_data = np.sqrt(data_hist)*data_scaler
    data_hist = data_hist*data_scaler
    # data_hist = np.histogram(data_pred, bins = len(binning), weights = data_w*data_scaler)

    # unc_data = stat_unc(np.asarray(data_pred), binning, data_w, d_scaler=data_scaler)    
    stat_unc_bkgs = stat_unc(np.asarray(pred[Y_test==0]), binning, np.asarray(W_test[Y_test==0]*scaler))
    syst_unc_bkgs = bkg_pred*0.3                                                    # Assuming 30% systematic uncertainty
    unc_bkg = np.sqrt(stat_unc_bkgs**2 + syst_unc_bkgs**2)
    
    np.seterr(divide='ignore', invalid='ignore')                                    # Remove true divide message
    ratio = data_hist/bkg_pred
    unc_ratio_stat = ratio*np.sqrt( (unc_data/data_hist)**2 + (stat_unc_bkgs/bkg_pred)**2)
    unc_ratio = ratio*np.sqrt( (unc_data/data_hist)**2 + (unc_bkg/bkg_pred)**2)
    plt.clf()
    
    print('Data')
    print(data_hist)
    print('+-')
    print(unc_data)
    
    print('Bkg +- stat +- syst')
    print(bkg_pred)
    print('+-')
    print(stat_unc_bkgs)
    print('+-')
    print(syst_unc_bkgs)
    
    print('Ratio +- stat +- syst')
    print(ratio)
    print('+-')
    print(unc_ratio_stat)
    print('+-')
    print(unc_ratio)
    
    line = np.linspace(0, 1, 2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.04)
    n, bins, patches = ax1.hist(hist, weights = hist_w, bins = binning, label = labels, histtype='barstacked', color=colors, zorder = 0)
    n, bins, patches = ax1.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = binning, color='#F42069', label="Signal", zorder = 5, histtype='step')
    ax1.bar(x_axis, 2*stat_unc_bkgs, bottom=bkg_pred-stat_unc_bkgs, fill=False, hatch='XXXXX', label='Stat. Unc.', width = widths, lw=0.0, alpha=0.3, edgecolor='r')
    ax1.bar(x_axis, 2*unc_bkg, bottom=bkg_pred-unc_bkg, fill=False, hatch='XXXXX', label='Stat. + Syst. Unc.', width = widths, lw=0.0, alpha=0.3)
    ax1.text(0.15, max(bkg_pred), '$\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, $m_{ll}>120$ GeV')
    if met_reg == '50-100' :
        ax1.text(0.15, max(bkg_pred)/2.5, '100 GeV > $E_{T}^{miss}$ > 50 GeV')
    if met_reg == '100-150' :
        ax1.text(0.15, max(bkg_pred)/2.5, '$150$ GeV > $E_{T}^{miss}$ > 100 GeV')
    if met_reg == '150' :
        ax1.text(0.15, max(bkg_pred)/2.5, '$>150$ GeV $E_{T}^{miss}$')
    ax1.errorbar(x_axis, data_hist, yerr = unc_data, fmt='o', color='black', label='Data', zorder = 10, ms=3, lw=1, capsize=2, lolims=0)
    ax1.set_ylabel('Events')
    ax1.set_yscale('log')
    ax1.set_xlim([0,1])
    ax1.set_ylim([2e-3, max(bkg_pred)*5])
    ax1.legend(ncol=2)
    ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax1.tick_params(bottom=True, top=True, left=True, right=True, which='both')
    ax2.set_ylabel('Events / Bkg')
    ax2.errorbar(x_axis, ratio, yerr = unc_ratio_stat, fmt='o', color='black', ms=3, lw=1, lolims=0)
    ax2.plot(line, np.ones(len(line)), linestyle='-', color='black', lw=2, alpha=0.3)
    ax2.bar(x_axis, 2*unc_ratio, bottom=ratio-unc_ratio, color='grey', width = widths, lw=0.0, alpha=0.3)
    ax2.grid(axis='y')
    ax2.set_xlim([0,1])
    ax2.set_ylim([0.5, 1.5])
    ax2.set_xlabel('XGBoost output')
    fig.suptitle('XGBoost output, '+dsid_title+' dataset, validation data with 30 % syst. unc.\n Network trained on '+dm_model+' model with $m_{ll} > 120$ GeV and in '+met_reg+' MET signal region', fontsize='x-large')
    plt.savefig(plot_dir+'VAL.pdf')
    plt.clf()

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

    
    plt.figure(figsize=[10,6])
    n, bins, patches = plt.hist(pred[Y_test==0], bins = binning, facecolor='blue', alpha=0.2,label="Background")
    n, bins, patches = plt.hist(pred[Y_test==1], bins = binning, facecolor='red', alpha=0.2, label="Signal")
    plt.xlabel('XGBoost output')
    plt.ylabel('Events')
    plt.yscale('log')
    plt.xlim([0,1])
    plt.ylim([2e-3, 1e8])
    plt.title('XGBoost output, '+dsid_title+' dataset, uscaled validation data')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir+'VAL_unscaled.pdf')
    plt.clf()    

    # Y_axis = [low_stat_Z(sum(sig_pred[25:]), sum(bkg_pred[25:])),          
    #             low_stat_Z(sum(sig_pred[30:]), sum(bkg_pred[30:])), 
    #             low_stat_Z(sum(sig_pred[35:]), sum(bkg_pred[35:])),
    #             low_stat_Z(sum(sig_pred[40:]), sum(bkg_pred[40:])), 
    #             low_stat_Z(sum(sig_pred[45:]), sum(bkg_pred[45:])), 
    #             low_stat_Z(sum(sig_pred[49:]), sum(bkg_pred[49:]))]
    
    # X_axis = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    
    # plt.figure(figsize=[10,6])
    # plt.plot(X_axis, Y_axis, linestyle='--')
    # plt.scatter(X_axis, Y_axis, label = 'Output cut')
    # plt.xlim([0,1])
    # plt.grid(True)
    # plt.legend()
    # plt.ylabel('Expected significance [$\sigma$]')
    # plt.title("Significance, trained network on "+dsid_title)
    # plt.xlabel('XGBoost output')
    # plt.savefig(plot_dir+'EXP_SIG.pdf')
    # plt.clf()
    
    plt.close('all')

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