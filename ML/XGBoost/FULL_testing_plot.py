import os, time, json, argparse
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from EventIDs import IDs
from Plot_maker import stat_unc, feature_importance
from matplotlib import ticker as mticker
from sklearn.metrics import roc_curve, auc

print(xgb.__version__)

t0 = time.time()
start = time.asctime(time.localtime())
print('Started', start)


parser = argparse.ArgumentParser()
parser.add_argument('--dm_model', type=str, default="DH_HDS", help="Dataset to test")
parser.add_argument('--channel', type=str, default="ee", help="Lepton channel to test")
args = parser.parse_args()

dm_model = args.dm_model
channel = args.channel 

np_dir = '../Data/'+dm_model+'_scores/'

mzp_130 = np.load(np_dir+dm_model+'_mZp_130/sig_pred_'+channel+'.npy')
mzp_200 = np.load(np_dir+dm_model+'_mZp_200/sig_pred_'+channel+'.npy')
mzp_400 = np.load(np_dir+dm_model+'_mZp_400/sig_pred_'+channel+'.npy')
mzp_600 = np.load(np_dir+dm_model+'_mZp_600/sig_pred_'+channel+'.npy')
mzp_800 = np.load(np_dir+dm_model+'_mZp_800/sig_pred_'+channel+'.npy')
mzp_900 = np.load(np_dir+dm_model+'_mZp_900/sig_pred_'+channel+'.npy')
mzp_1100 = np.load(np_dir+dm_model+'_mZp_1100/sig_pred_'+channel+'.npy')
mzp_1200 = np.load(np_dir+dm_model+'_mZp_1200/sig_pred_'+channel+'.npy')
mzp_1300 = np.load(np_dir+dm_model+'_mZp_1300/sig_pred_'+channel+'.npy')
mzp_1400 = np.load(np_dir+dm_model+'_mZp_1400/sig_pred_'+channel+'.npy')
mzp_1500 = np.load(np_dir+dm_model+'_mZp_1500/sig_pred_'+channel+'.npy')

N = 9
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.PuRd_r(np.linspace(0.1,0.95,N)))

model_dsids = []
json_file = open('DM_DICT_Zp_dsid.json')
DM_file = json.load(json_file)
for key in DM_file.keys():
    word = key.split('_')
    model_sec = word[0]+'_'+word[1]
    if model_sec == dm_model.lower():
        model_dsids.append(DM_file[key])

json_file2 = open('DM_DICT.json')
model_names = json.load(json_file2)
save_as = 'mZp_'+model_names[model_dsids[0][0]].split(' ')[-2]+'/'
save_dir = "/storage/racarcam/"
bkg_file = save_dir+'bkgs_final.h5'
sig_file1 = save_dir+'/Zp_DMS/'+model_dsids[0][0]+'.h5'
sig_file2 = save_dir+'/Zp_DMS/'+model_dsids[0][1]+'.h5'
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
print('Doing mll > 110 and met > 50 on '+dm_model+" "+save_as+" Z' model")
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


Y_test = Y_test[Dilepton_test==channel]
W_test = W_test[Dilepton_test==channel]
DSID_test = DSID_test[Dilepton_test==channel]
X_test = X_test[Dilepton_test==channel]
data_test = data_test[dilepton_data==channel]

model_dir = '../Models/XGB/Model_Dependent/'
xgbclassifier = xgb.XGBClassifier()
xgbclassifier.load_model(model_dir+'best_'+dm_model+'.txt')

y_pred_prob = xgbclassifier.predict_proba(X_test)
data_pred_prob = xgbclassifier.predict_proba(data_test)

pred = y_pred_prob[:,1]
data_pred = data_pred_prob[:,1]
data_w = np.ones(len(data_pred))*10
bins = 50

plot_dir = '../../Plots/XGBoost/'+dm_model+'/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass


np_dir = '../Data/'+dm_model+'/'

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

replot_axis = np.linspace(0, 1, 49)
n, bins, patches = ax1.hist(replot_axis, bins = 50, weights=mzp_130, label = "$m_{Z'}$ = 130 GeV", zorder = 5, histtype='step')
n, bins, patches = ax1.hist(replot_axis, bins = 50, weights=mzp_200, label = "$m_{Z'}$ = 200 GeV", zorder = 5, histtype='step')
n, bins, patches = ax1.hist(replot_axis, bins = 50, weights=mzp_400, label = "$m_{Z'}$ = 400 GeV", zorder = 5, histtype='step')
n, bins, patches = ax1.hist(replot_axis, bins = 50, weights=mzp_600, label = "$m_{Z'}$ = 600 GeV", zorder = 5, histtype='step')
n, bins, patches = ax1.hist(replot_axis, bins = 50, weights=mzp_800, label = "$m_{Z'}$ = 800 GeV", zorder = 5, histtype='step')
n, bins, patches = ax1.hist(replot_axis, bins = 50, weights=mzp_900, label = "$m_{Z'}$ = 900 GeV", zorder = 5, histtype='step')
n, bins, patches = ax1.hist(replot_axis, bins = 50, weights=mzp_1100, label = "$m_{Z'}$ = 1.1 TeV", zorder = 5, histtype='step')
n, bins, patches = ax1.hist(replot_axis, bins = 50, weights=mzp_1200, label = "$m_{Z'}$ = 1.2 TeV", zorder = 5, histtype='step')
n, bins, patches = ax1.hist(replot_axis, bins = 50, weights=mzp_1300, label = "$m_{Z'}$ = 1.3 TeV", zorder = 5, histtype='step')

ax1.bar(x_axis, 2*stat_unc_bkgs, bottom=bkg_pred-stat_unc_bkgs, fill=False, hatch='XXXXX', label='Stat. Unc.', width = widths, lw=0.0, alpha=0.3, edgecolor='r')
ax1.bar(x_axis, 2*syst_unc_bkgs, bottom=bkg_pred-syst_unc_bkgs, fill=False, hatch='XXXXX', label='Syst. Unc.', width = widths, lw=0.0, alpha=0.3)
ax1.text(0.08, max(bkg_pred), '$\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, $m_{ll}>110$ GeV')
if channel == 'uu': chnl = '$\mu\mu$'
if channel == 'ee': chnl = '$ee$'
ax1.text(0.08, max(bkg_pred)/2.5, '$>50$ GeV $E_{T}^{miss}$, '+chnl)
    
ax1.errorbar(x_axis[:30], data_hist[:30], yerr = unc_data[:30], fmt='o', color='black', label='Data', zorder = 10, ms=3, lw=1, capsize=2, lolims=0)
ax1.set_ylabel('Events')
ax1.set_yscale('log')
ax1.set_xlim([0,1])
ax1.set_ylim([2e-3, max(bkg_pred)*5])
ax1.legend(ncol=3)
ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
ax1.tick_params(bottom=True, top=True, left=True, right=True, which='both')
ax2.set_ylabel('Events / Bkg')
ax2.errorbar(x_axis[:30], ratio[:30], yerr = unc_ratio_stat[:30], fmt='o', color='black', ms=3, lw=1, lolims=0)
ax2.plot(line, np.ones(len(line)), linestyle='-', color='black', lw=2, alpha=0.3)
ax2.bar(x_axis, 2*unc_bkg, bottom=np.ones(len(x_axis))-unc_bkg, color='grey', width = widths, lw=0.0, alpha=0.3)
ax2.grid(axis='y')
ax2.set_xlim([0,1])
ax2.set_ylim([0.01, 1.99])
ax2.set_xlabel('XGBoost output')
fig.suptitle('XGBoost output, '+dm_model.split('_')[0]+' '+dm_model.split('_')[1]+' dataset, validation data with 20 % syst. unc.\n Network trained on '+dm_model.split('_')[0]+' '+dm_model.split('_')[1]+' model with $m_{ll} > 110$ GeV and $E_{T}^{miss} > 50$ GeV', fontsize='x-large')
plt.savefig(plot_dir+'VAL_'+channel+'.pdf')

# plot_dir = plot_dir+'feature_importance/'
# try:
#     os.makedirs(plot_dir)

# except FileExistsError:
#     pass


# feature_importance(xgbclassifier=xgbclassifier, plot_dir=plot_dir)