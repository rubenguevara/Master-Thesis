import os, json, argparse, random
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from EventIDs import IDs
from Plot_maker import stat_unc, feature_importance
from matplotlib import ticker as mticker
from sklearn.metrics import  auc

parser = argparse.ArgumentParser()
parser.add_argument('--met_reg', type=str, default="50-100", help="MET signal region")
parser.add_argument('--dm_model', type=str, default="SlepSlep", help="Dataset to test")
parser.add_argument('--channel', type=str, default="ee", help="Lepton channel to test")
args = parser.parse_args()

met_reg = args.met_reg
dm_model = args.dm_model
channel = args.channel 


save_dir = "/storage/racarcam/"
np_dir = save_dir+'Data/XGB_frfr/'+met_reg+'/'+dm_model+'/'

directories = [] 
for (dirpath, dirnames, filenames) in os.walk(np_dir):
    if filenames == []: continue
    if 'MET75' in dirpath: continue
    # if 'D10' in dirpath: continue
    directories.append(dirpath)
    continue

random.Random(69).shuffle(directories)

directories = directories[:5]

N = 15
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.PuRd_r(np.linspace(0.1,0.95,N)))


model_dsids = []
json_file  = open('DM_DICT_SUSY_dsid.json')
json_file2 = open('DM_DICT_SUSY_models.json')
dsids = json.load(json_file)
if '2HDM':
    dsids = dsids['Stop']
else:
    dsids = dsids[dm_model]
model_names = json.load(json_file2)

save_dir = "/storage/racarcam/"
bkg_file = save_dir+'bkgs_frfr.h5'
sig_file1 = save_dir+'/DMS_dsid/'+dsids[0]+'.h5'
data_file = save_dir+'datafrfr.h5'
df_bkg = pd.read_hdf(bkg_file, key='df_tot')
df_sig1 = pd.read_hdf(sig_file1, key='df_tot')
df_dat = pd.read_hdf(data_file, key='df')
df = pd.concat([df_bkg, df_sig1])


extra_variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50', 'dPhiCloseMet', 'dPhiLeps', 'isOS']



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

print('Doing mll > 110 and met reg '+met_reg+' on '+dm_model+" model on",channel,'channel')
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

model_dir = '../Models/XGB/Model_independent_frfr/'
xgbclassifier = xgb.XGBClassifier()
xgbclassifier.load_model(model_dir+met_reg+'.txt')

y_pred_prob = xgbclassifier.predict_proba(X_test)
data_pred_prob = xgbclassifier.predict_proba(data_test)

pred = y_pred_prob[:,1]
data_pred = data_pred_prob[:,1]
data_w = np.ones(len(data_pred))*10
bins = 50

plot_dir = '../../Plots/XGBoost/Model_independent_frfr/'+met_reg+'/'+dm_model+'/'

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

for dirpath in directories:
    siggy = np.load(dirpath+'/sig_pred_'+channel+'.npy')
    if dm_model =='SlepSlep':
        slep = dirpath.split('_')[-2][1:]
        neut = dirpath.split('_')[-1][1:]
        label = '($m_{\\tilde{\ell}},m_{\\tilde{\chi}_1^0}$)=('+slep+','+neut+')'
    if dm_model =='2HDM':
        slep = dirpath.split('_')[-2][1:]
        neut = dirpath.split('_')[-3][1:]
        tb = dirpath.split('_')[-1][2:]
        label = '($m_{a},m_{H},\\tan\\beta$)=('+slep+','+neut+','+tb+')'
    n, bins, patches = ax1.hist(replot_axis, bins = 50, weights=siggy, label = label, zorder = 5, histtype='step')

ax1.bar(x_axis, 2*stat_unc_bkgs, bottom=bkg_pred-stat_unc_bkgs, fill=False, hatch='XXXXX', label='Stat. Unc.', width = widths, lw=0.0, alpha=0.3, edgecolor='r')
ax1.bar(x_axis, 2*syst_unc_bkgs, bottom=bkg_pred-syst_unc_bkgs, fill=False, hatch='XXXXX', label='Syst. Unc.', width = widths, lw=0.0, alpha=0.3)
if channel == 'uu': chnl = '$\mu\mu$'
if channel == 'ee': chnl = '$ee$'
ax1.text(0.04, max(bkg_pred)*2.5, '$\sqrt{s} = 13$ TeV, 139 fb$^{-1}$,\n'+channel+' channel')

ax1.errorbar(x_axis[:30], data_hist[:30], yerr = unc_data[:30], fmt='o', color='black', label='Data', zorder = 10, ms=3, lw=1, capsize=2, lolims=0)
ax1.set_ylabel('Events')
ax1.set_yscale('log')
ax1.set_xlim([0,1])
ax1.set_ylim([2e-1, max(bkg_pred)*10])
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
if dm_model == 'SlepSlep':
    ax1.set_ylim([2e-1, max(bkg_pred)*10])
    fig.suptitle('XGBoost output, direct slepton production validation data with 20 % syst. unc.\n Network trained on full direct slepton dataset with $E_{T}^{miss} > 50$ GeV', fontsize='x-large')
if dm_model == '2HDM':
    ax1.set_ylim([2e-3, max(bkg_pred)*10])
    fig.suptitle('XGBoost output, Two Higgs Doublet Model validation data with 20 % syst. unc.\n Network trained on full 2HDM + a dataset with $E_{T}^{miss} > 50$ GeV', fontsize='x-large')
plt.savefig(plot_dir+'VAL_'+channel+'.pdf')



plt.figure(figsize=[8,6])
lw = 2
    
for dirpath in directories:
    if dm_model =='SlepSlep':
        slep = dirpath.split('_')[-2][1:]
        neut = dirpath.split('_')[-1][1:]
        label = '($m_{\\tilde{\ell}},m_{\\tilde{\chi}_1^0}$)=('+slep+','+neut+')'
    if dm_model =='2HDM':
        slep = dirpath.split('_')[-2][1:]
        neut = dirpath.split('_')[-3][1:]
        tb = dirpath.split('_')[-1][2:]
        label = '($m_{a},m_{H},\\tan\\beta$)=('+slep+','+neut+','+tb+')'
    fpr = np.load(dirpath+'/fpr_'+channel+'.npy')
    tpr = np.load(dirpath+'/tpr_'+channel+'.npy')
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, lw=lw, label=label+' (area = %0.2f)' % roc_auc)
    
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
if dm_model == 'SlepSlep':
    plt.title('ROC for direct slepton production dataset')
if dm_model == '2HDM':
    plt.title('ROC for 2HDM + a dataset')
plt.legend(loc="lower right",ncol=2)
plt.savefig(plot_dir+'ROC_'+channel+'.pdf')

plot_dir2 = plot_dir+'/feature_importance/'
try:
    os.makedirs(plot_dir2)

except FileExistsError:
    pass


feature_importance(xgbclassifier=xgbclassifier, plot_dir=plot_dir2)