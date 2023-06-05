import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import os, json
from EventIDs import IDs

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
print(tf.__version__)


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


save_dir = "../../../../storage/racarcam/"
filename = "new_bkgs.h5"
datafile = 'new_data.h5'

pad_variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50']

bad_variables = ['jet1Pt', 'jet1Eta', 'jet1Phi', 'jet2Pt', 'jet2Eta', 'jet2Phi', 'jet3Pt', 'jet3Eta', 'jet3Phi', 'mjj', 'dPhiCloseMet', 'dPhiLeps']


df_data = pd.read_hdf(save_dir+datafile, key='df')
dataID = df_data.pop('EventID') 
dataRN = df_data.pop('RunNumber') 
data_ll = df_data.pop('Dileptons')
dataRP = df_data.pop('RunPeriod') 
df_data = df_data.drop(bad_variables, axis=1)

# Comment out next line to include new variables!
# df_data = df_data.drop(pad_variables, axis=1)

data_train, data_test = train_test_split(df_data, test_size = 0.1, random_state = 42)

model_dir = '../Models/PAD/'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass

dm_dict_file = open('../DM_DICT.json')
DM_DICT = json.load(dm_dict_file)

df_bkgs = pd.read_hdf(save_dir+filename, key='df_tot')
already_done = os.listdir(model_dir)

dsid_test = [[514560, 514561]]#, 514560, 514561, 514564, 514565] 
for dsid_int in dsid_test:
    dsid1 = str(dsid_int[0])
    dsid2 = str(dsid_int[1])
    dsid_name = DM_DICT[dsid1].split(' ')
    dsid_title = dsid_name[0] +' '+ dsid_name[1] +' '+ dsid_name[2] +' '+ dsid_name[3]
    dsid_save = dsid_name[0] +'_'+ dsid_name[1] + '_mZp_' + dsid_name[3]
    
    print('Doing', dsid_save)

    df_dm = pd.read_hdf(save_dir+'/Zp_DMS/'+dsid1+'.h5', key='df_tot')
    df_dm2 = pd.read_hdf(save_dir+'/Zp_DMS/'+dsid2+'.h5', key='df_tot')
    
    df = pd.concat([df_bkgs, df_dm, df_dm2]).sort_index()

    df_features = df.copy()
    df_EventID = df_features.pop('EventID')
    df_CrossSection = df_features.pop('CrossSection')
    df_Dileptons = df_features.pop('Dileptons')
    df_RunPeriod = df_features.pop('RunPeriod')
    df_features = df_features.drop(bad_variables, axis=1)
    df_labels = df_features.pop('Label')
    
    # Comment out next line to include new variables!
    # df_features = df_features.drop(pad_variables, axis=1)
    
    
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size = test_size, random_state = 42)
    W_train = X_train.pop('Weight')
    W_test = X_test.pop('Weight')
    DSID_test = X_test.pop('RunNumber')
    DSID_train = X_train.pop('RunNumber')
    
    scaler = 1/test_size
    network = tf.keras.models.load_model(model_dir+'new_variables')
    # network = tf.keras.models.load_model(model_dir+'no_pad')
    
    pred = network.predict(X_test, batch_size = int(2**23), use_multiprocessing = True, verbose = 1).ravel()
    data_pred = network.predict(data_test, batch_size = int(2**23), use_multiprocessing = True, verbose = 1).ravel()
    data_w = np.ones(len(data_pred))*10
    n_bins = 100
    
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
    
    plot_dir = '../../Plots/NeuralNetwork/FULL/padding/new_variables/finer_binning/'
    # plot_dir = '../../Plots/NeuralNetwork/FULL/padding/no_pad/'

    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass
    
    bkg_pred, bins, patches = plt.hist(pred[Y_test==0], weights = W_test[Y_test==0]*scaler, bins = n_bins)
    sig_pred, bins, patches = plt.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = n_bins)
    data_hist, bins, patches = plt.hist(data_pred, bins = n_bins)

    unc_data = np.sqrt(data_hist)*10
    data_hist = data_hist*10
    
    stat_unc_bkgs = stat_unc(np.asarray(pred[Y_test==0]), 100, np.asarray(W_test[Y_test==0]*scaler))
    syst_unc_bkgs = bkg_pred*0.3                                            # Assuming 30% systematic uncertainty
    unc_bkg = np.sqrt(stat_unc_bkgs**2 + syst_unc_bkgs**2)
    
    
    ratio = data_hist/bkg_pred
    unc_ratio_stat = ratio*np.sqrt( (unc_data/data_hist)**2 + (stat_unc_bkgs/bkg_pred)**2)
    unc_ratio = ratio*np.sqrt( (unc_data/data_hist)**2 + (unc_bkg/bkg_pred)**2)
    
    plt.clf()

    x_axis = np.linspace(0.5/n_bins, 1-0.5/n_bins, n_bins)
    width = x_axis[1]-x_axis[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.04)
    n, bins, patches = ax1.hist(hist, weights = hist_w, bins = n_bins, label = labels, histtype='barstacked', color=colors, zorder = 0)
    n, bins, patches = ax1.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = n_bins, color='#F42069', label="Signal", zorder = 5, histtype='step')
    ax1.bar(x_axis, 2*unc_bkg, bottom=bkg_pred-unc_bkg, fill=False, hatch='XXXXX', label='Stat. + Syst. Unc.', width = width, lw=0.0, alpha=0.3)
    ax1.text(0.15, max(bkg_pred), '$\sqrt{s} = 13$ TeV, 139 fb$^{-1}$') 
    ax1.text(0.15, max(bkg_pred)/2.5, '$>50$ GeV $E_{T}^{miss}$')
    ax1.errorbar(x_axis, data_hist, yerr = unc_data, fmt='o', color='black', label='Data', zorder = 10, ms=3, lw=1, capsize=2 )
    ax1.set_ylabel('Events')
    ax1.set_yscale('log')
    ax1.set_xlim([0,1])
    ax1.set_ylim([2e-3,2e7])
    ax1.legend(ncol=2)
    ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax1.tick_params(bottom=True, top=True, left=True, right=True, which='both')
    ax2.set_ylabel('Events / Bkg')
    ax2.errorbar(x_axis, ratio, yerr = unc_ratio_stat, fmt='o', color='black', ms=3, lw=1)
    ax2.bar(x_axis, 2*unc_ratio, bottom=ratio-unc_ratio, color='grey', width = width, lw=0.0, alpha=0.3)
    ax2.grid(axis='y')
    ax2.set_xlim([0,1])
    ax2.set_ylim([0.5,1.5])
    ax2.set_xlabel('TensorFlow output')
    fig.suptitle('TensorFlow output, '+dsid_title+' dataset, validation data with 30 % syst. unc.', fontsize='x-large')
    plt.savefig(plot_dir+'VAL.pdf')
    plt.show()

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
    plt.title('ROC for TensorFlow on '+dsid_title+' dataset')
    plt.legend(loc="lower right")
    plt.savefig(plot_dir+'ROC.pdf')
    plt.show()

    plt.figure(figsize=[10,6])
    n, bins, patches = plt.hist(pred[Y_test==0], bins = n_bins, facecolor='blue', alpha=0.2,label="Background")
    n, bins, patches = plt.hist(pred[Y_test==1], bins = n_bins, facecolor='red', alpha=0.2, label="Signal")
    plt.xlabel('TensorFlow output')
    plt.ylabel('Events')
    plt.yscale('log')
    plt.xlim([0,1])
    plt.title('TensorFlow output, '+dsid_title+' dataset, uscaled validation data')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir+'VAL_unscaled.pdf')
    plt.show()    

    Y_axis = [low_stat_Z(sum(sig_pred[50:]), sum(bkg_pred[50:])),          
                low_stat_Z(sum(sig_pred[60:]), sum(bkg_pred[60:])), 
                low_stat_Z(sum(sig_pred[70:]), sum(bkg_pred[70:])),
                low_stat_Z(sum(sig_pred[80:]), sum(bkg_pred[80:])), 
                low_stat_Z(sum(sig_pred[90:]), sum(bkg_pred[90:])), 
                low_stat_Z(sum(sig_pred[99:]), sum(bkg_pred[99:]))]
    
    X_axis = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    
    plt.figure(figsize=[10,6])
    plt.plot(X_axis, Y_axis, linestyle='--')
    plt.scatter(X_axis, Y_axis, label = 'Output cut')
    plt.xlim([0,1])
    plt.grid(True)
    plt.legend()
    plt.ylabel('Expected significance [$\sigma$]')
    plt.title("Significance, trained network on "+dsid_title)
    plt.xlabel('TensorFlow output')
    plt.savefig(plot_dir+'EXP_SIG.pdf')
    plt.show()