import os, gc, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from EventIDs import IDs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import ticker as mticker
from sklearn.metrics import roc_curve, auc

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
print('TensorFlow v.', tf.__version__)

save_dir = "/storage/racarcam/"
filename = 'bkgs_final.h5'
datafile = 'dataFINAL.h5'

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


parser = argparse.ArgumentParser()
parser.add_argument('--try_pad', type=str, default="0", help="Try new variables to pad")
args = parser.parse_args()

try_pad = args.try_pad


df_data = pd.read_hdf(save_dir+datafile, key='df')
new_variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50']
paddable_variables = ['jet1Pt', 'jet1Phi', 'jet1Eta', 'jet2Pt', 'jet2Phi', 'jet2Eta', 'jet3Pt', 'jet3Phi', 'jet3Eta', 'mjj', 'dPhiCloseMet', 'dPhiLeps']

data_features = df_data.copy()
data_EventID = data_features.pop('EventID')
data_RunNumber = data_features.pop('RunNumber')
data_Dileptons = data_features.pop('Dileptons')
data_RunPeriod = data_features.pop('RunPeriod')
data_features = data_features.drop(paddable_variables, axis=1)

if try_pad == '0':
    data_features = data_features.drop(new_variables, axis = 1)

data_test_size = 0.1
data_train, data_test = train_test_split(data_features, test_size =data_test_size, random_state = 42)
data_scaler = 1/data_test_size

df_bkgs = pd.read_hdf(save_dir+filename, key='df_tot')
# sig_file1 = save_dir+'/Zp_DMS/514643.h5'
# sig_file2 = save_dir+'/Zp_DMS/514642.h5'
sig_file1 = save_dir+'/Zp_DMS/514560.h5'
sig_file2 = save_dir+'/Zp_DMS/514561.h5'
df_sig1 = pd.read_hdf(sig_file1, key='df_tot')
df_sig2 = pd.read_hdf(sig_file2, key='df_tot')
df = pd.concat([df_bkgs, df_sig1, df_sig2])


df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
# df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_features = df_features.drop(paddable_variables, axis=1)

if try_pad == '0':
    df_features = df_features.drop(new_variables, axis = 1)


df_labels = df_features.pop('Label')

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=test_size, random_state=42)
X_train_w = X_train.pop('Weight')
W_test = X_test.pop('Weight')
DSIDs_train = X_train.pop('RunNumber')
DSIDs = X_test.pop('RunNumber')
scaler = 1/test_size



model_dir = '../../Models/NN/No_pad/'

if try_pad == '0':
    model = tf.keras.models.load_model(model_dir+'Without_padding')
    plot_dir = '../../../Plots/NeuralNetwork/Padding/No_pad/'

else:
    model = tf.keras.models.load_model(model_dir+'With_padding')
    plot_dir = '../../../Plots/NeuralNetwork/Padding/New_pad/'




### Plotting
pred = model.predict(X_test, batch_size = int(2**20), use_multiprocessing = True, verbose = 2).ravel()
data_pred = model.predict(data_test, batch_size = int(2**20), use_multiprocessing = True, verbose = 2).ravel()
bins = 50

DY = []; ST = []; DB = []; W = []; TT = [];
DY_w = []; ST_w = []; DB_w = []; W_w = []; TT_w = [];
for DSID, output, w in zip(DSIDs, pred, W_test):
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
labels = ['W', "Diboson", 'TTbar', 'Single Top', 'Drell Yan']


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
    
plt.figure(figsize=[10,6])
plt.xlabel('TensorFlow output')
plt.ylabel('Events')
plt.yscale('log')
plt.xlim([0,1])
plt.title('TF output, DH HDS dataset, validation data')
plt.grid(True)
bkg_pred, bins, patches = plt.hist(pred[Y_test==0], weights = W_test[Y_test==0]*scaler, bins = binning, facecolor='blue', alpha=0.2,label="Background")
sig_pred, bins, patches = plt.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = binning, facecolor='red', alpha=0.2, label="Signal")
plt.legend()
plt.savefig(plot_dir+'VAL_pre.pdf')
plt.clf()   
data_hist, bins, patches = plt.hist(data_pred, bins = binning)
unc_data = np.sqrt(data_hist)*data_scaler
data_hist = data_hist*data_scaler

stat_unc_bkgs = stat_unc(np.asarray(pred[Y_test==0]), binning, np.asarray(W_test[Y_test==0]*scaler))
syst_unc_bkgs = bkg_pred*0.2                                                    # Assuming 20% systematic uncertainty
unc_bkg = np.sqrt((stat_unc_bkgs/bkg_pred)**2 + 0.2**2)
np.seterr(divide='ignore', invalid='ignore')                                    # Remove true divide message
ratio = data_hist/bkg_pred
unc_ratio = ratio*np.sqrt( (unc_data/data_hist)**2 + (stat_unc_bkgs/bkg_pred)**2)
plt.clf()

line = np.linspace(0, 1, 2)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
fig.subplots_adjust(hspace=0.04)
n, bins, patches = ax1.hist(hist, weights = hist_w, bins = binning, label = labels, histtype='barstacked', color=colors, zorder = 0)
n, bins, patches = ax1.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = binning, color='#F42069', label="$m_{Z'}$ 130", zorder = 5, histtype='step')
ax1.bar(x_axis, 2*stat_unc_bkgs, bottom=bkg_pred-stat_unc_bkgs, fill=False, hatch='XXXXX', label='Stat. Unc.', width = widths, lw=0.0, alpha=0.3, edgecolor='r')
ax1.bar(x_axis, 2*syst_unc_bkgs, bottom=bkg_pred-syst_unc_bkgs, fill=False, hatch='XXXXX', label='Syst. Unc.', width = widths, lw=0.0, alpha=0.3)
    
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
unc_ratio = np.nan_to_num(unc_ratio)
ratio = np.nan_to_num(ratio)
ax2.errorbar(x_axis[:30], ratio[:30], yerr = unc_ratio[:30], fmt='o', color='black', ms=3, lw=1)
ax2.plot(line, np.ones(len(line)), linestyle='-', color='black', lw=2, alpha=0.3)
ax2.bar(x_axis, 2*unc_bkg, bottom=np.ones(len(x_axis))-unc_bkg, color='grey', width = widths, lw=0.0, alpha=0.3)
ax2.grid(axis='y')
ax2.set_xlim([0,1])
ax2.set_ylim([0.5, 1.5])
ax2.set_xlabel('Tensorflow output')
fig.suptitle('TensorFlow output, DH HDS dataset, validation data with 20 % syst. unc.\n $\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, $>50$ GeV $E_{T}^{miss}$', fontsize='x-large')
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
plt.title('ROC for TensorFlow on DH HDS 130')
plt.legend(loc="lower right")
plt.savefig(plot_dir+'ROC.pdf')
plt.show()


plt.close('all')


def low_stat_Z(sig, bkg, sig_unc = None, bkg_unc = None):
    """
    Calcultes the expected significance of the signal. Can be done with and without uncertainties. Default is without
    
    """
    
    if sig_unc != None and bkg_unc != None:
        Z == 0
    else:   
        Z = np.sqrt(2*( (sig + bkg)*np.log(1 + sig/bkg) - sig ))
    return Z


def Z_score_array(sig_pred, bkg_pred):
    return [low_stat_Z(sum(sig_pred[25:]), sum(bkg_pred[25:])),          
                low_stat_Z(sum(sig_pred[30:]), sum(bkg_pred[30:])), 
                low_stat_Z(sum(sig_pred[35:]), sum(bkg_pred[35:])),
                low_stat_Z(sum(sig_pred[40:]), sum(bkg_pred[40:])), 
                low_stat_Z(sum(sig_pred[45:]), sum(bkg_pred[45:])), 
                low_stat_Z(sig_pred[-1], bkg_pred[-1])]

plt.figure(figsize=(11,8))
X_axis = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
Y_axis = Z_score_array(sig_pred, bkg_pred)

plt.figure(figsize=[10,6])
plt.plot(X_axis, Y_axis, linestyle='--')
plt.scatter(X_axis, Y_axis, label = "Signal")
plt.xlim([0,1])
plt.ylim([min(Y_axis)*0.9, max(Y_axis)*1.1])
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.ylabel('Expected significance [$\sigma$]')
plt.title("Significance on DH HDS 130")
plt.xlabel('TensorFlow output')
plt.savefig(plot_dir+'EXP_SIG.pdf')