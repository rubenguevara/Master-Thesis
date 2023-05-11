import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import json
from EventIDs import IDs
from matplotlib import ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib.cm import RdPu, YlGnBu_r

def low_stat_Z(sig, bkg, sig_unc = None, bkg_unc = None):
    """
    Calcultes the expected significance of the signal. Can be done with and without uncertainties. Default is without
    
    """
    
    if sig_unc != None and bkg_unc != None:
        Z == 0
    else:   
        Z = np.sqrt(2*( (sig + bkg)*np.log(1 + sig/bkg) - sig ))
    return Z

def stat_unc(prediction, bins, weights, d_scaler = None):
    """
    Calculates statistical uncertainty of every bin by the formula sqrt( sum_i (w_i)^2 ) * scaler
    
    prediction: numpy array. ML prediction array
    bins: list. Binning of our plot
    weights: Weight of every MC sample
    d_scaler: Optional float. Scaler for data samples
    """
    binning = bins
    histo_bins = np.digitize(prediction, binning)
    stat_unc_array = []
    if d_scaler != None:
        for i in range(1,len(binning)):
            bin_wgt = weights[np.where(histo_bins==i)[0]]
            sow_bin = np.linalg.norm(bin_wgt,2)
            stat_unc_array.append(sow_bin*d_scaler)
    else:
        for i in range(1,len(binning)):
            bin_wgt = weights[np.where(histo_bins==i)[0]]
            sow_bin = np.linalg.norm(bin_wgt,2)
            stat_unc_array.append(sow_bin)
    
    return np.asarray(stat_unc_array)

dm_dict_file = open('DM_DICT.json')
DM_DICT = json.load(dm_dict_file)

def scaled_validation(dsid_int, MC_pred, MC_wgt, MC_label, MC_dsid, Data_pred, plot_dir, bins = 50, met_reg = None, dm_model = None, scaler = 5, data_scaler = 10, ML_type = 'XGBoost', channel = ''):
    """
    Makes validation plots of ML network prediction to further conduct Signal Region Search. Returns signal and background histrograms and their uncertainties.
    
    dsid_int: list of str. Data Set ID for signal we will test
    MC_pred: float array. Prediction of ML algorithm for MC samples, for XGB should be of the form pred_prob[:,1]
    MC_wgt: float array. The weights of the MC samples, used to re-weight events to expected events
    MC_label: int array. Label of every event on the MC samples
    MC_dsid: str array. Data Set ID of every MC sample, used to categorize different SM backgrounds
    Data_pred: float array Prediction of ML algorithm for data points, for XGB should be of the form pred_prob[:,1]
    plot_dir = str. Specify where you want to save plots 
    bins: float array. Array choosing our binning for the plots 
    met_reg: Optional str. MET SR being studied
    dm_model: Optional str. DM model being studied 
    scaler: float. Used to re-weight MC testing set to full luminosity. Default is 5
    data_scaler: float. Used to re-weight data testing set to full luminosity. Default is 10
    ML_type: Optional str. Used to write better axis. Default is 'XGBoost'
    channel: Optional str. labels plot on electron of muon final state. Default is empty
    
    Returns [sig_pred, bkg_pred], [unc_sig, unc_bkg], data_hist
    """
    
    dsid1 = str(dsid_int[0])
    dsid2 = str(dsid_int[1])
    dsid_name = DM_DICT[dsid1].split(' ')
    dsid_title = dsid_name[0] +' '+ dsid_name[1] +' '+ dsid_name[2] +' '+ dsid_name[3]
    dsid_save = 'mZp_' + dsid_name[3]

    pred = MC_pred
    W_test = MC_wgt
    Y_test = MC_label
    DSID_test = MC_dsid
    data_pred = Data_pred
    data_w = np.ones(len(data_pred))
    
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
    
    
    if isinstance(bins, (list, tuple, np.ndarray)):
        binning = bins
    else: binning = np.linspace(0,1, bins)
    
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
    
    stat_unc_bkgs = stat_unc(np.asarray(pred[Y_test==0]), binning, np.asarray(W_test[Y_test==0]*scaler))
    syst_unc_bkgs = bkg_pred*0.2                                                    # Assuming 20% systematic uncertainty
    unc_bkg_sav = np.sqrt(stat_unc_bkgs**2 + syst_unc_bkgs**2)                         
    unc_bkg = np.sqrt((stat_unc_bkgs/bkg_pred)**2 + 0.2**2)

    stat_unc_sig = stat_unc(np.asarray(pred[Y_test==1]), binning, np.asarray(W_test[Y_test==1]*scaler))
    syst_unc_sig = sig_pred*0.2                                                    # Assuming 20% systematic uncertainty
    unc_sig = np.sqrt(stat_unc_sig**2 + syst_unc_sig**2)
    
    np.seterr(divide='ignore', invalid='ignore')                                    # Remove true divide message
    ratio = data_hist/bkg_pred
    unc_ratio_stat = ratio*np.sqrt( (unc_data/data_hist)**2 + (stat_unc_bkgs/bkg_pred)**2)
    unc_ratio = ratio*np.sqrt( (unc_data/data_hist)**2 + (unc_bkg/bkg_pred)**2)
    plt.clf()
    
    line = np.linspace(0, 1, 2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.04)
    n, bins, patches = ax1.hist(hist, weights = hist_w, bins = binning, label = labels, histtype='barstacked', color=colors, zorder = 0)
    n, bins, patches = ax1.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = binning, color='#F42069', label="Signal", zorder = 5, histtype='step')
    ax1.bar(x_axis, 2*stat_unc_bkgs, bottom=bkg_pred-stat_unc_bkgs, fill=False, hatch='XXXXX', label='Stat. Unc.', width = widths, lw=0.0, alpha=0.3, edgecolor='r')
    ax1.bar(x_axis, 2*syst_unc_bkgs, bottom=bkg_pred-syst_unc_bkgs, fill=False, hatch='XXXXX', label='Syst. Unc.', width = widths, lw=0.0, alpha=0.3)
    
    ax1.text(0.15, max(bkg_pred), '$\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, $m_{ll}>110$ GeV, '+channel)
    
    if met_reg == '50-100' :
        ax1.text(0.15, max(bkg_pred)/2.5, '100 GeV > $E_{T}^{miss}$ > 50 GeV')
    elif met_reg == '100-150' :
        ax1.text(0.15, max(bkg_pred)/2.5, '$150$ GeV > $E_{T}^{miss}$ > 100 GeV')
    elif met_reg == '150' :
        ax1.text(0.15, max(bkg_pred)/2.5, '$>150$ GeV $E_{T}^{miss}$')
    else:
        ax1.text(0.15, max(bkg_pred)/2.5, '$>50$ GeV $E_{T}^{miss}$')
        
    ax1.errorbar(x_axis[:30], data_hist[:30], yerr = unc_data[:30], fmt='o', color='black', label='Data', zorder = 10, ms=3, lw=1, capsize=2, lolims=0)
    ax1.set_ylabel('Events')
    ax1.set_yscale('log')
    ax1.set_xlim([0,1])
    ax1.set_ylim([2e-3, max(bkg_pred)*5])
    ax1.legend(ncol=2)
    ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax1.tick_params(bottom=True, top=True, left=True, right=True, which='both')
    ax2.set_ylabel('Events / Bkg')
    ax2.errorbar(x_axis[:30], ratio[:30], yerr = unc_ratio_stat[:30], fmt='o', color='black', ms=3, lw=1, lolims=0)
    ax2.plot(line, np.ones(len(line)), linestyle='-', color='black', lw=2, alpha=0.3)
    # ax2.bar(x_axis, 2*unc_ratio, bottom=ratio-unc_ratio, color='grey', width = widths, lw=0.0, alpha=0.3)
    ax2.bar(x_axis, 2*unc_bkg, bottom=np.ones(len(x_axis))-unc_bkg, color='grey', width = widths, lw=0.0, alpha=0.3)
    ax2.grid(axis='y')
    ax2.set_xlim([0,1])
    ax2.set_ylim([0.01, 1.99])
    ax2.set_xlabel(ML_type+' output')
    if met_reg == None:
        fig.suptitle(ML_type+' output, '+dsid_title+' dataset, validation data with 20 % syst. unc.\n Network trained on '+dm_model+' model', fontsize='x-large')
    else: 
        fig.suptitle(ML_type+' output, '+dsid_title+' dataset, validation data with 20 % syst. unc.\n Network trained on all models with $m_{ll} > 110$ GeV and in '+met_reg+' MET signal region', fontsize='x-large')
    if channel == '':
        plt.savefig(plot_dir+'VAL.pdf')    
    else:
        plt.savefig(plot_dir+'VAL_'+channel+'.pdf')
    plt.clf()
    return [sig_pred, bkg_pred], [unc_sig, unc_bkg_sav], data_hist

def ROC_curve(Y_test, pred, dsid_int, plot_dir, ML_type = 'XGBoost'):
    dsid1 = str(dsid_int[0])
    dsid2 = str(dsid_int[1])
    dsid_name = DM_DICT[dsid1].split(' ')
    dsid_title = dsid_name[0] +' '+ dsid_name[1] +' '+ dsid_name[2] +' '+ dsid_name[3]
    dsid_save = dsid_name[0] +'_'+ dsid_name[1] + '_mZp_' + dsid_name[3]
            
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
    plt.title('ROC for '+ML_type+' on '+dsid_title+' dataset')
    plt.legend(loc="lower right")
    plt.savefig(plot_dir+'ROC.pdf')
    plt.clf()

def unscaled_validation(pred, Y_test, binning, dsid_int, plot_dir, ML_type = 'XGBoost', channel = ''):    
    dsid1 = str(dsid_int[0])
    dsid2 = str(dsid_int[1])
    dsid_name = DM_DICT[dsid1].split(' ')
    dsid_title = dsid_name[0] +' '+ dsid_name[1] +' '+ dsid_name[2] +' '+ dsid_name[3]
    dsid_save = dsid_name[0] +'_'+ dsid_name[1] + '_mZp_' + dsid_name[3]
    plt.figure(figsize=[10,6])
    n, bins, patches = plt.hist(pred[Y_test==0], bins = binning, facecolor='blue', alpha=0.2,label="Background")
    n, bins, patches = plt.hist(pred[Y_test==1], bins = binning, facecolor='red', alpha=0.2, label="Signal")
    plt.xlabel(ML_type+' output')
    plt.ylabel('Events')
    plt.yscale('log')
    plt.xlim([0,1])
    plt.title(ML_type+' output, '+dsid_title+' '+channel+' dataset, uscaled validation data')
    plt.grid(True)
    plt.legend()
    if channel == '':
        plt.savefig(plot_dir+'VAL_unscaled.pdf')
    else: 
        plt.savefig(plot_dir+'VAL_unscaled_'+channel+'.pdf')
    plt.clf()    

def expected_significance(sig_pred, bkg_pred, model, dsid_title, plot_dir, ML_type = 'XGBoost'):
    
    Y_axis = [low_stat_Z(sum(sig_pred[25:]), sum(bkg_pred[25:])),          
                low_stat_Z(sum(sig_pred[30:]), sum(bkg_pred[30:])), 
                low_stat_Z(sum(sig_pred[35:]), sum(bkg_pred[35:])),
                low_stat_Z(sum(sig_pred[40:]), sum(bkg_pred[40:])), 
                low_stat_Z(sum(sig_pred[45:]), sum(bkg_pred[45:])), 
                low_stat_Z(sig_pred[-1], bkg_pred[-1])]

    X_axis = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    plt.figure(figsize=[10,6])
    plt.plot(X_axis, Y_axis, linestyle='--')
    plt.scatter(X_axis, Y_axis, label = 'Prediction cut')
    plt.xlim([0,1])
    plt.grid(True)
    plt.legend()
    plt.ylabel('Expected significance [$\sigma$]')
    plt.title("Significance on "+model+", trained network on "+dsid_title)
    plt.xlabel(ML_type+' output')
    plt.savefig(plot_dir+'EXP_SIG.pdf')
    plt.clf()


def feature_importance(xgbclassifier, plot_dir):
    
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
    
def distribution(df_features, DSIDs, WGTs, df_data, variable, plot_dir):
    var_ax = {}

    var_ax['lep1Pt'] = '$p_{T}^{1}$ [GeV]'
    var_ax['lep2Pt'] = '$p_{T}^{2}$ [GeV]'
    var_ax['lep1Eta'] = '$\eta_{1}$'
    var_ax['lep2Eta'] = '$\eta_{2}$'
    var_ax['jet1Pt'] = 'jet $p_{T}^{1}$ [GeV]'
    var_ax['jet2Pt'] = 'jet $p_{T}^{2}$ [GeV]'
    var_ax['jet3Pt'] = 'jet $p_{T}^{3}$ [GeV]'
    var_ax['jet1Eta'] = 'jet $\eta_{1}$'
    var_ax['jet2Eta'] = 'jet $\eta_{2}$'
    var_ax['jet3Eta'] = 'jet $\eta_{3}$'
    var_ax['mll'] = '$m_{ll}$ [GeV]'
    var_ax['mjj'] = '$m_{jj}$ [GeV]'
    var_ax['met'] = '$E_{T}^{miss}$ [GeV]'
    var_ax['met_sig'] = '$E_{T}^{miss}/sigma$'
    var_ax['mt'] = '$m_{T}$ [GeV]'
    var_ax['rt'] = '$E_{T}^{miss}/H_T$'
    var_ax['ht'] = '$H_{T}$ [GeV]'
    var_ax['dPhiLeadMet'] = '$\Delta\phi(l_{lead}, E_{T}^{miss})$'
    var_ax['dPhiLLMet'] = '$\Delta\phi(ll, E_{T}^{miss})$'
    var_ax['mt2'] = '$m_{T2}$ [GeV]'
    var_ax['jetB'] = 'Number of B jets'
    var_ax['jetLight'] = 'Number of light jets'
    var_ax['jetTot'] = 'Total number of jets'
    var_ax['et'] = '$E_{T}$ [GeV]'
    var_ax['lep1Phi'] = '$\phi_{1}$'
    var_ax['lep2Phi'] = '$\phi_{2}$'
    var_ax['jet1Phi'] = 'jet $\phi_{1}$'
    var_ax['jet2Phi'] = 'jet $\phi_{2}$'
    var_ax['jet3Phi'] = 'jet $\phi_{3}$'
    var_ax['jetEtaForward50'] = '# jets with $|\eta| >2.5$ and $p_T > 50 GeV$'
    var_ax['jetEtaCentral'] = '# jets with $|\eta| <2.5$'
    var_ax['n_bjetPt20'] = '# b-jets with $p_T > 20 GeV$'
    var_ax['n_ljetPt40'] = '# l-jets with $p_T > 40 GeV$'

    # x_axis_gen = np.linspace(20, 3500, 74)
    # x_eta = np.linspace(-3, 3, 50);
    # x_met = np.linspace(20, 2500, 74)
    # x_mt2 = np.linspace(20, 1500, 74)
    # x_met_sig = np.linspace(0, 100, 74)
    # x_et = np.linspace(20, 3000, 74)
    # x_phi = np.linspace(-np.pi, np.pi, 50)
    # x_dphi = np.linspace(0, np.pi, 30) 
    # x_jets = np.linspace(0, 7, 8)
    # x_rt = np.linspace(0, 10, 50)
    if variable == 'mll':
        start = 10
    elif variable == 'met':
        start = 50
    else: 
        start = 0
    
    if int(max(df_features[variable])) > 1000:
        x_axis = np.linspace(start, 2000, 50 )
        
    elif 'Phi' in variable:
        x_axis = np.linspace(-np.pi, np.pi, 50 )     
        
    elif 'Eta' in variable:
        x_axis = np.linspace(-np.pi, np.pi, 50 ) 
        
    else:
        x_axis = np.linspace(start, 10, 11 )

    if 'Light' in variable:
        x_axis = np.linspace(0, 20, 21)
        
    if variable == 'rt':
        x_axis = np.linspace(0, 10, 11 )
                
    if variable == 'jet3Eta':
        x_axis = np.linspace(-np.pi, np.pi, 50)
        
    
    # if variable == 'met':
    #     x_axis = np.linspace(50, 100, 20)
    # x_axis = x_axis_gen
    # if 'Phi' in variable:
    #     x_axis = x_phi
    
    # if 'dPhi' in variable:
    #     x_axis = x_dphi
        
    # if 'Eta' in variable:
    #     x_axis = x_eta
    
    # if variable == 'met':
    #     x_axis = x_met
    
    # if variable =='mt2':
    #     x_axis = x_mt2
    
    # if variable =='met_sig':
    #     x_axis = x_met_sig
    
    # if variable == 'jetB' or variable == 'jetLight' or variable == 'jetTot':
    #     x_axis = x_jets
    
    # # if 'jet' in variable:
    # #     x_axis = x_jet_n
    
    # if variable == 'et':
    #     x_axis = x_et
        
    # if variable =='rt':
    #     x_axis = x_rt
    
    DY = []; ST = []; DB = []; W = []; TT = []; LV = []; DH = []; EFT = []
    DY_w = []; ST_w = []; DB_w = []; W_w = []; TT_w = []; LV_w = []; DH_w = []; EFT_w = []
    for DSID, var, w in zip(DSIDs, df_features[variable], WGTs):
        if DSID in IDs["DY"]:
            DY.append(var)
            DY_w.append(w)
        elif DSID in IDs['Single_top']:
            ST.append(var)
            ST_w.append(w)
        elif DSID in IDs["Diboson"]:
            DB.append(var)
            DB_w.append(w)
        elif DSID in IDs["W"]:
            W.append(var)
            W_w.append(w)
        elif DSID in IDs["TTbar"]:
            TT.append(var)
            TT_w.append(w)
        # elif DSID in dsid_DH_HDS_MZ_130:
        #     DH.append(var)
        #     DH_w.append(w)
        # elif DSID in dsid_LV_HDS_MZ_130:
        #     LV.append(var)
        #     LV_w.append(w)
        # elif DSID in dsid_EFT_HDS_MZ_130:
        #     EFT.append(var)
        #     EFT_w.append(w)

    hist = [W, DB, TT, ST, DY]
    hist_w = [W_w, DB_w, TT_w, ST_w, DY_w]
    colors = ['#218C8D', '#6CCECB', '#F9E559', '#EF7126', '#8EDC9D']
    labels = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan']
    d_te = plt.hist(df_data[variable], bins = x_axis)
    x_axis_data = []
    widths = []
    for i in range(len(x_axis) - 1):
        bin_width = x_axis[i+1] - x_axis[i]
        bin_center = x_axis[i] + bin_width/2
        widths.append(bin_width)
        x_axis_data.append(bin_center)
        
    bkg = df_features#.loc[df_features['Label']==0]
    bkg_hist, bins, patches = plt.hist(bkg[variable], weights = WGTs, bins = x_axis)
    data_hist, bins, patches = plt.hist(df_data[variable], bins = x_axis)
    
    unc_data = np.sqrt(data_hist)
    stat_unc_bkgs = stat_unc(np.asarray(bkg[variable]), x_axis, np.asarray(WGTs))
    syst_unc_bkgs = bkg_hist*0.3                                            # Assuming 30% systematic uncertainty
    unc_bkg = np.sqrt(stat_unc_bkgs**2 + syst_unc_bkgs**2)
    
    
    np.seterr(divide='ignore', invalid='ignore')                                    # Remove true divide message
    ratio = data_hist/bkg_hist
    unc_ratio_stat = ratio*np.sqrt( (unc_data/data_hist)**2 + (stat_unc_bkgs/bkg_hist)**2)
    unc_ratio = ratio*np.sqrt( (unc_data/data_hist)**2 + (unc_bkg/bkg_hist)**2)
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.04)
    ax1.hist(hist, weights=hist_w, bins = x_axis, stacked=True, color=colors, label=labels)
    ax1.bar(x_axis_data, 2*unc_bkg, bottom=bkg_hist-unc_bkg, fill=False, hatch='XXXXX', label='Stat. + Syst. Unc.', width = widths, lw=0.0, alpha=0.3)
    # ax1.hist(LV, weights=LV_w, bins = x_axis,  label='Light Vector', histtype='step', color='#F42069')
    # ax1.hist(DH, weights=DH_w, bins = x_axis,  label='Dark Higgs', histtype='step', color = 'pink')
    # ax1.hist(EFT, weights=EFT_w, bins = x_axis,  label='Effective Field Theory', histtype='step', color='red')
    ax1.errorbar(x_axis_data, d_te[0], yerr = unc_data, fmt='o', color='black', label='Data', zorder = 10)
    ax2.set_xlabel(var_ax[variable]); ax1.set_ylabel('Events'); 
    fig.suptitle("Distribution with $\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, \n $m_{ll}$ > 120 GeV and $E_{T}^{miss}$ > 150 GeV")
    # fig.suptitle("Signal search for Heavy Dark Sector and $m_{Z'}=130G$eV \n $\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, $>50GeV$ $E_{T}^{miss}$")
    ax1.set_yscale('log') 
    ax1.legend(ncol=2)
    ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax1.tick_params(bottom=True, top=True, left=True, right=True, which='both')
    ax1.set_xlim([x_axis[0], x_axis[-1]])
    ax1.set_ylim([1e-3, max(bkg_hist)*7])
    ax2.set_xlim([x_axis[0], x_axis[-1]])
    ax2.set_ylabel('Events / Bkg')
    ax2.errorbar(x_axis_data, ratio, yerr = unc_ratio_stat, fmt='o', color='black')
    ax2.plot([x_axis[0], x_axis[-1]], [1,1], 'k-', alpha = 0.3)
    
    unc_ratio= np.nan_to_num(unc_ratio)
    ax2.bar(x_axis_data, 2*unc_ratio, bottom=ratio-unc_ratio, color='grey', width = widths, lw=0.0, alpha=0.3)
    ax2.grid(axis='y')
    ax2.set_ylim([0.5,1.6])
    plt.savefig(plot_dir+variable+'.pdf')
    plt.show()


def scatter_plot(dataset, predictions, score, variable_1, variable_2, plot_dir, give_axis = False, do_weights = False, x_axis_1=None, x_axis_2=None, weights=None):
    var_ax = {}

    var_ax['lep1Pt'] = '$p_{T}^{1}$ [GeV]'
    var_ax['lep2Pt'] = '$p_{T}^{2}$ [GeV]'
    var_ax['lep1Eta'] = '$\eta_{1}$'
    var_ax['lep2Eta'] = '$\eta_{2}$'
    var_ax['jet1Pt'] = 'jet $p_{T}^{1}$ [GeV]'
    var_ax['jet2Pt'] = 'jet $p_{T}^{2}$ [GeV]'
    var_ax['jet3Pt'] = 'jet $p_{T}^{3}$ [GeV]'
    var_ax['jet1Eta'] = 'jet $\eta_{1}$'
    var_ax['jet2Eta'] = 'jet $\eta_{2}$'
    var_ax['jet3Eta'] = 'jet $\eta_{3}$'
    var_ax['mll'] = '$m_{ll}$ [GeV]'
    var_ax['mjj'] = '$m_{jj}$ [GeV]'
    var_ax['met'] = '$E_{T}^{miss}$ [GeV]'
    var_ax['met_sig'] = '$E_{T}^{miss}/sigma$'
    var_ax['mt'] = '$m_{T}$ [GeV]'
    var_ax['rt'] = '$E_{T}^{miss}/H_T$'
    var_ax['ht'] = '$H_{T}$ [GeV]'
    var_ax['dPhiLeadMet'] = '$\Delta\phi(l_{lead}, E_{T}^{miss})$'
    var_ax['dPhiLLMet'] = '$\Delta\phi(ll, E_{T}^{miss})$'
    var_ax['mt2'] = '$m_{T2}$ [GeV]'
    var_ax['jetB'] = 'Number of B jets'
    var_ax['jetLight'] = 'Number of light jets'
    var_ax['jetTot'] = 'Total number of jets'
    var_ax['et'] = '$E_{T}$ [GeV]'
    var_ax['lep1Phi'] = '$\phi_{1}$'
    var_ax['lep2Phi'] = '$\phi_{2}$'
    var_ax['jet1Phi'] = 'jet $\phi_{1}$'
    var_ax['jet2Phi'] = 'jet $\phi_{2}$'
    var_ax['jet3Phi'] = 'jet $\phi_{3}$'
    
    if give_axis == True:
        x_axis_1 = x_axis_1
        x_axis_2 = x_axis_2
    else:
        if variable_1 == 'mll':
            start_1 = 120
        elif variable_1 == 'met':
            start_1 = 150
            
        else: 
            start_1 = 0
        
        if int(max(dataset[variable_1])) > 1000:
            x_axis_1 = np.linspace(start_1, 2000, 26 )
            
        elif 'Phi' in variable_1:
            x_axis_1 = np.linspace(-np.pi, np.pi, 26 )     
            
        elif 'Eta' in variable_1:
            x_axis_1 = np.linspace(-np.pi, np.pi, 26 ) 
    
        elif variable_1 == 'mt2':
            x_axis_1 = np.linspace(0, 1000, 26 )
            
        else:
            x_axis_1 = np.linspace(start_1, 10, 11 )

        if 'Light' in variable_1:
            x_axis_1 = np.linspace(0, 20, 21)
            
        if variable_1 == 'rt':
            x_axis_1 = np.linspace(0, 10, 11 )
                    
        if variable_1 == 'jet3Eta':
            x_axis_1 = np.linspace(-np.pi, np.pi, 26)
            
        if variable_2 == 'mll':
            start_2 = 120
        elif variable_2 == 'met':
            start_2 = 150
            
        else: 
            start_2 = 0
        
        if int(max(dataset[variable_2])) > 1000:
            x_axis_2 = np.linspace(start_2, 2000, 26 )
            
        elif 'Phi' in variable_2:
            x_axis_2 = np.linspace(-np.pi, np.pi, 26 )     
            
        elif 'Eta' in variable_2:
            x_axis_2 = np.linspace(-np.pi, np.pi, 26 ) 
    
        elif variable_2 == 'mt2':
            x_axis_2 = np.linspace(0, 1000, 26 )
    
        else:
            x_axis_2 = np.linspace(start_2, 10, 11 )

        if 'Light' in variable_2:
            x_axis_2 = np.linspace(0, 20, 21)
            
        if variable_2 == 'rt':
            x_axis_2 = np.linspace(0, 10, 11 )
                    
        if variable_2 == 'jet3Eta':
            x_axis_2 = np.linspace(-np.pi, np.pi, 26)
    
    sig_bool = [i>score for i in predictions]
    bkg_bool = [i<score for i in predictions]
    
    x = dataset[variable_1][sig_bool] 
    y = dataset[variable_2][sig_bool]
    
    x_bkg = dataset[variable_1][bkg_bool] 
    y_bkg = dataset[variable_2][bkg_bool]
    
    
    # To make percentage contour
    if do_weights == False:
        title_set = 'data points'
        w_array = np.ones(len(x))
        wgts = w_array/sum(w_array)
        
        w_array_bkg = np.ones(len(x_bkg))
        wgts_bkg = w_array_bkg/sum(w_array_bkg) 
    
    else:
        title_set = 'SM backgrounds'
        w_array = weights[sig_bool]
        wgts = w_array/sum(w_array)
        
        w_array_bkg = weights[bkg_bool]
        wgts_bkg = w_array_bkg/sum(w_array_bkg) 
        
        
    
    fig, (ax_sig, ax_bkg) = plt.subplots(1, 2, figsize=(16, 6))
    image_sig = ax_sig.hist2d(x, y, weights = wgts, bins = (x_axis_1, x_axis_2), cmap = RdPu, norm=LogNorm(vmin=1e-5))
    image_bkg = ax_bkg.hist2d(x_bkg, y_bkg, weights = wgts_bkg, bins = (x_axis_1, x_axis_2), cmap = YlGnBu_r, norm=LogNorm(vmin=1e-5))
    cbar_sig = fig.colorbar(image_sig[3], ax = ax_sig, location = 'right')
    ax_sig.set_title('Prediction score greater than '+str(score))

    cbar_bkg = fig.colorbar(image_bkg[3], ax = ax_bkg, location = 'right')

    ax_bkg.set_title('Prediction score lesser than '+str(score))
    ax_sig.set_ylabel(var_ax[variable_2])
    ax_sig.set_xlabel(var_ax[variable_1])
    ax_bkg.set_ylabel(var_ax[variable_2])
    ax_bkg.set_xlabel(var_ax[variable_1])
    # ax_sig.grid()
    # ax_bkg.grid()
    
    if '[GeV]' in var_ax[variable_1] and '[GeV]' in var_ax[variable_2]:
        fig.suptitle('Scatter plot for '+var_ax[variable_1].split(' ')[0]+' and '+var_ax[variable_2].split(' ')[0]+' of network prediction on '+title_set, fontsize='x-large')
    
    elif '[GeV]' in var_ax[variable_1]:
        fig.suptitle('Scatter plot for '+var_ax[variable_1].split(' ')[0]+' and '+var_ax[variable_2]+' of network prediction', fontsize='x-large')

    elif '[GeV]' in var_ax[variable_2]:
        fig.suptitle('Scatter plot for '+var_ax[variable_1]+' and '+var_ax[variable_2].split(' ')[0]+' of network prediction', fontsize='x-large')

    else:
        fig.suptitle('Scatter plot for '+var_ax[variable_1]+' and '+var_ax[variable_2]+' of network prediction', fontsize='x-large')

    plt.savefig(plot_dir+variable_1+'_'+variable_2+'.pdf', bbox_inches='tight')
    plt.show()


def distribution_data(df_data, variable, plot_dir, score, pred):
    var_ax = {}

    var_ax['lep1Pt'] = '$p_{T}^{1}$ [GeV]'
    var_ax['lep2Pt'] = '$p_{T}^{2}$ [GeV]'
    var_ax['lep1Eta'] = '$\eta_{1}$'
    var_ax['lep2Eta'] = '$\eta_{2}$'
    var_ax['jet1Pt'] = 'jet $p_{T}^{1}$ [GeV]'
    var_ax['jet2Pt'] = 'jet $p_{T}^{2}$ [GeV]'
    var_ax['jet3Pt'] = 'jet $p_{T}^{3}$ [GeV]'
    var_ax['jet1Eta'] = 'jet $\eta_{1}$'
    var_ax['jet2Eta'] = 'jet $\eta_{2}$'
    var_ax['jet3Eta'] = 'jet $\eta_{3}$'
    var_ax['mll'] = '$m_{ll}$ [GeV]'
    var_ax['mjj'] = '$m_{jj}$ [GeV]'
    var_ax['met'] = '$E_{T}^{miss}$ [GeV]'
    var_ax['met_sig'] = '$E_{T}^{miss}/sigma$'
    var_ax['mt'] = '$m_{T}$ [GeV]'
    var_ax['rt'] = '$E_{T}^{miss}/H_T$'
    var_ax['ht'] = '$H_{T}$ [GeV]'
    var_ax['dPhiLeadMet'] = '$\Delta\phi(l_{lead}, E_{T}^{miss})$'
    var_ax['dPhiLLMet'] = '$\Delta\phi(ll, E_{T}^{miss})$'
    var_ax['mt2'] = '$m_{T2}$ [GeV]'
    var_ax['jetB'] = 'Number of B jets'
    var_ax['jetLight'] = 'Number of light jets'
    var_ax['jetTot'] = 'Total number of jets'
    var_ax['et'] = '$E_{T}$ [GeV]'
    var_ax['lep1Phi'] = '$\phi_{1}$'
    var_ax['lep2Phi'] = '$\phi_{2}$'
    var_ax['jet1Phi'] = 'jet $\phi_{1}$'
    var_ax['jet2Phi'] = 'jet $\phi_{2}$'
    var_ax['jet3Phi'] = 'jet $\phi_{3}$'
    var_ax['jetEtaForward50'] = '# jets with $|\eta| >2.5$ and $p_T > 50 GeV$'
    var_ax['jetEtaCentral'] = '# jets with $|\eta| <2.5$'
    var_ax['n_bjetPt20'] = '# b-jets with $p_T > 20 GeV$'
    var_ax['n_ljetPt40'] = '# l-jets with $p_T > 40 GeV$'

    if variable == 'mll':
        start = 120
    elif variable == 'met':
        start = 150
    else: 
        start = 0
    
    sig_bool = [i > score for i in pred]
    sig_data = df_data[sig_bool]
    bkg_bool = [i < score for i in pred]
    bkg_data = df_data[bkg_bool]
    
    if int(max(df_data[variable])) > 1000:
        x_axis = np.linspace(start, 2000, 50 )
        
    elif 'Phi' in variable:
        x_axis = np.linspace(-np.pi, np.pi, 50 )     
        
    elif 'Eta' in variable:
        x_axis = np.linspace(-np.pi, np.pi, 50 )  
    
    else:
        x_axis = np.linspace(start, 10, 11 )

    if variable == 'mt2':
        x_axis = np.linspace(0, 1000, 50 )
    
    if 'Light' in variable:
        x_axis = np.linspace(0, 20, 21)
        
    if variable == 'rt':
        x_axis = np.linspace(0, 10, 11 )
                
    if variable == 'jet3Eta':
        x_axis = np.linspace(-np.pi, np.pi, 50)
    
    sig_hist,_,__ = plt.hist(sig_data[variable], bins = x_axis)
    bkg_hist,_,__ = plt.hist(bkg_data[variable], bins = x_axis)
    plt.clf()
    x_axis_data = []
    widths = []
    for i in range(len(x_axis) - 1):
        bin_width = x_axis[i+1] - x_axis[i]
        bin_center = x_axis[i] + bin_width/2
        widths.append(bin_width)
        x_axis_data.append(bin_center)
    
    figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
    ax1.errorbar(x_axis_data, bkg_hist, yerr = np.sqrt(bkg_hist), fmt='o', color='black', label='Data', zorder = 10)
    figure.suptitle("Distribution with $\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, \n $m_{ll}$ > 120 GeV and $E_{T}^{miss}$ > 150 GeV")
    ax1.set_yscale('log') 
    ax1.legend()
    ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax1.tick_params(bottom=True, top=True, left=True, right=True, which='both')
    ax1.set_xlim([x_axis[0], x_axis[-1]])
    ax1.set_ylim([1e-1, max(bkg_hist)*7])
    
    
    ax1.set_title('Prediction score lesser than '+str(score))
    ax1.set_ylabel('Events')
    ax2.set_title('Prediction score greater than '+str(score))
    ax2.set_ylabel('Events')
    ax2.set_xlabel(var_ax[variable])
    
    ax2.errorbar(x_axis_data, sig_hist, yerr = np.sqrt(sig_hist), fmt='o', color='black', label='Data', zorder = 10)
    figure.suptitle("Data distribution with $\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, \n $m_{ll}$ > 120 GeV and $E_{T}^{miss}$ > 150 GeV")
    ax2.set_yscale('log') 
    ax2.legend()
    ax2.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax2.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax2.tick_params(bottom=True, top=True, left=True, right=True, which='both')
    ax2.set_xlim([x_axis[0], x_axis[-1]])
    ax2.set_ylim([1e-1, max(sig_hist)*7])
    plt.savefig(plot_dir+variable+'.pdf')
    plt.show()
    

def distribution_MC(df, wgts, dsid, variable, plot_dir, score, pred):
    var_ax = {}

    var_ax['lep1Pt'] = '$p_{T}^{1}$ [GeV]'
    var_ax['lep2Pt'] = '$p_{T}^{2}$ [GeV]'
    var_ax['lep1Eta'] = '$\eta_{1}$'
    var_ax['lep2Eta'] = '$\eta_{2}$'
    var_ax['jet1Pt'] = 'jet $p_{T}^{1}$ [GeV]'
    var_ax['jet2Pt'] = 'jet $p_{T}^{2}$ [GeV]'
    var_ax['jet3Pt'] = 'jet $p_{T}^{3}$ [GeV]'
    var_ax['jet1Eta'] = 'jet $\eta_{1}$'
    var_ax['jet2Eta'] = 'jet $\eta_{2}$'
    var_ax['jet3Eta'] = 'jet $\eta_{3}$'
    var_ax['mll'] = '$m_{ll}$ [GeV]'
    var_ax['mjj'] = '$m_{jj}$ [GeV]'
    var_ax['met'] = '$E_{T}^{miss}$ [GeV]'
    var_ax['met_sig'] = '$E_{T}^{miss}/sigma$'
    var_ax['mt'] = '$m_{T}$ [GeV]'
    var_ax['rt'] = '$E_{T}^{miss}/H_T$'
    var_ax['ht'] = '$H_{T}$ [GeV]'
    var_ax['dPhiLeadMet'] = '$\Delta\phi(l_{lead}, E_{T}^{miss})$'
    var_ax['dPhiLLMet'] = '$\Delta\phi(ll, E_{T}^{miss})$'
    var_ax['mt2'] = '$m_{T2}$ [GeV]'
    var_ax['jetB'] = 'Number of B jets'
    var_ax['jetLight'] = 'Number of light jets'
    var_ax['jetTot'] = 'Total number of jets'
    var_ax['et'] = '$E_{T}$ [GeV]'
    var_ax['lep1Phi'] = '$\phi_{1}$'
    var_ax['lep2Phi'] = '$\phi_{2}$'
    var_ax['jet1Phi'] = 'jet $\phi_{1}$'
    var_ax['jet2Phi'] = 'jet $\phi_{2}$'
    var_ax['jet3Phi'] = 'jet $\phi_{3}$'
    var_ax['jetEtaForward50'] = '# jets with $|\eta| >2.5$ and $p_T > 50 GeV$'
    var_ax['jetEtaCentral'] = '# jets with $|\eta| <2.5$'
    var_ax['n_bjetPt20'] = '# b-jets with $p_T > 20 GeV$'
    var_ax['n_ljetPt40'] = '# l-jets with $p_T > 40 GeV$'

    if variable == 'mll':
        start = 120
    elif variable == 'met':
        start = 150
    else: 
        start = 0
    
    sig_bool = [i > score for i in pred]
    sig_mc = df[sig_bool]
    sig_dsid = dsid[sig_bool]
    sig_wgts = wgts[sig_bool]
    
    bkg_bool = [i < score for i in pred]
    bkg_mc = df[bkg_bool]
    bkg_dsid = dsid[bkg_bool]
    bkg_wgts = wgts[bkg_bool]
    
    if int(max(df[variable])) > 1000:
        x_axis = np.linspace(start, 2000, 50 )
        
    elif 'Phi' in variable:
        x_axis = np.linspace(-np.pi, np.pi, 50 )     
        
    elif 'Eta' in variable:
        x_axis = np.linspace(-np.pi, np.pi, 50 ) 
    
    else:
        x_axis = np.linspace(start, 10, 11 )

    if variable == 'mt2':
        x_axis = np.linspace(0, 1000, 50 )
    
    if 'Light' in variable:
        x_axis = np.linspace(0, 20, 21)
        
    if variable == 'rt':
        x_axis = np.linspace(0, 10, 11 )
                
    if variable == 'jet3Eta':
        x_axis = np.linspace(-np.pi, np.pi, 50)
    
    
    sig_hist,_,__ = plt.hist(sig_mc[variable], weights=sig_wgts, bins = x_axis)
    bkg_hist,_,__ = plt.hist(bkg_mc[variable], weights=bkg_wgts, bins = x_axis)
    plt.clf()
    
    DY = []; ST = []; DB = []; W = []; TT = [];
    DY_w = []; ST_w = []; DB_w = []; W_w = []; TT_w = [];
    for DSID, var, w in zip(bkg_dsid, bkg_mc[variable], bkg_wgts):
        if DSID in IDs["DY"]:
            DY.append(var)
            DY_w.append(w)
        elif DSID in IDs['Single_top']:
            ST.append(var)
            ST_w.append(w)
        elif DSID in IDs["Diboson"]:
            DB.append(var)
            DB_w.append(w)
        elif DSID in IDs["W"]:
            W.append(var)
            W_w.append(w)
        elif DSID in IDs["TTbar"]:
            TT.append(var)
            TT_w.append(w)

    hist_bkg = [W, DB, TT, ST, DY]
    hist_bkg_w = [W_w, DB_w, TT_w, ST_w, DY_w]
    colors = ['#218C8D', '#6CCECB', '#F9E559', '#EF7126', '#8EDC9D']
    labels = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan']
    
    figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
    ax1.hist(hist_bkg, weights=hist_bkg_w, bins = x_axis, stacked=True, color=colors, label=labels)
    figure.suptitle("Distribution with $\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, \n $m_{ll}$ > 120 GeV and $E_{T}^{miss}$ > 150 GeV")
    ax1.set_yscale('log') 
    ax1.legend()
    ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax1.tick_params(bottom=True, top=True, left=True, right=True, which='both')
    ax1.set_xlim([x_axis[0], x_axis[-1]])
    ax1.set_ylim([1e-3, max(bkg_hist)*7])
    
    
    DY = []; ST = []; DB = []; W = []; TT = [];
    DY_w = []; ST_w = []; DB_w = []; W_w = []; TT_w = [];
    for DSID, var, w in zip(sig_dsid, sig_mc[variable], sig_wgts):
        if DSID in IDs["DY"]:
            DY.append(var)
            DY_w.append(w)
        elif DSID in IDs['Single_top']:
            ST.append(var)
            ST_w.append(w)
        elif DSID in IDs["Diboson"]:
            DB.append(var)
            DB_w.append(w)
        elif DSID in IDs["W"]:
            W.append(var)
            W_w.append(w)
        elif DSID in IDs["TTbar"]:
            TT.append(var)
            TT_w.append(w)

    hist_sig = [W, DB, TT, ST, DY]
    hist_sig_w = [W_w, DB_w, TT_w, ST_w, DY_w]
    
    ax1.set_title('Prediction score lesser than '+str(score))
    ax1.set_ylabel('Events')
    ax2.set_title('Prediction score greater than '+str(score))
    ax2.set_ylabel('Events')
    ax2.set_xlabel(var_ax[variable])
    
    ax2.hist(hist_sig, weights=hist_sig_w, bins = x_axis, stacked=True, color=colors, label=labels)
    ax2.set_yscale('log') 
    ax2.legend()
    ax2.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax2.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax2.tick_params(bottom=True, top=True, left=True, right=True, which='both')
    ax2.set_xlim([x_axis[0], x_axis[-1]])
    ax2.set_ylim([1e-5, max(sig_hist)*7])
    plt.savefig(plot_dir+variable+'.pdf')
    plt.show()


