import os, argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from XGBoost.Plot_maker import stat_unc
from EventIDs import IDs
from matplotlib import ticker as mticker


save_dir = "/storage/racarcam/"
file = save_dir+'FULL_Zp_FINAL.h5' 
datafile = 'dataFINAL.h5'

N = 9
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.PuRd_r(np.linspace(0.1,0.95,N)))


df_datas = pd.read_hdf(save_dir+datafile, key='df')
df = pd.read_hdf(file, key='df_tot')

df_datas = df_datas[df_datas['mll'] < 110]
df = df[df['mll'] < 110]

parser = argparse.ArgumentParser()
parser.add_argument('--vari', type=str, default="mll", help="Variable to plot")
parser.add_argument('--train', type=str, default='1', help="Train or test")
args = parser.parse_args()

vari = args.vari
train = args.train 

plot_dir = '../Plots/Data_Analysis/train_test_split/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
D_train, D_test = train_test_split(df_datas, test_size=0.2, random_state=42)

if train == '1':
    print('Plotting', vari, 'train dataset')
    df_features = X_train
    df_data = D_train
    DSIDs = X_train['RunNumber']
    WGTs = X_train['Weight']

    DH_HDS = pd.concat([X_train[[vari, 'Weight']][X_train['RunNumber'] == 514560], X_train[[vari, 'Weight']][X_train['RunNumber'] == 514561]])
    DH_LDS = pd.concat([X_train[[vari, 'Weight']][X_train['RunNumber'] == 514635], X_train[[vari, 'Weight']][X_train['RunNumber'] == 514634]])
    LV_HDS = pd.concat([X_train[[vari, 'Weight']][X_train['RunNumber'] == 514563], X_train[[vari, 'Weight']][X_train['RunNumber'] == 514562]])
    LV_LDS = pd.concat([X_train[[vari, 'Weight']][X_train['RunNumber'] == 514636], X_train[[vari, 'Weight']][X_train['RunNumber'] ==  514637]])
    EFT_HDS = pd.concat([X_train[[vari, 'Weight']][X_train['RunNumber'] == 514564], X_train[[vari, 'Weight']][X_train['RunNumber'] == 514565]])
    EFT_LDS = pd.concat([X_train[[vari, 'Weight']][X_train['RunNumber'] == 514638], X_train[[vari, 'Weight']][X_train['RunNumber'] == 514639]])

else:
    print('Plotting', vari, 'test dataset')
    df_features = X_test
    df_data = D_test
    DSIDs = X_test['RunNumber']
    WGTs = X_test['Weight']
    DH_HDS = pd.concat([X_test[[vari, 'Weight']][X_test['RunNumber'] == 514560], X_test[[vari, 'Weight']][X_test['RunNumber'] == 514561]])
    DH_LDS = pd.concat([X_test[[vari, 'Weight']][X_test['RunNumber'] == 514635], X_test[[vari, 'Weight']][X_test['RunNumber'] == 514634]])
    LV_HDS = pd.concat([X_test[[vari, 'Weight']][X_test['RunNumber'] == 514563], X_test[[vari, 'Weight']][X_test['RunNumber'] == 514562]])
    LV_LDS = pd.concat([X_test[[vari, 'Weight']][X_test['RunNumber'] == 514636], X_test[[vari, 'Weight']][X_test['RunNumber'] == 514637]])
    EFT_HDS = pd.concat([X_test[[vari, 'Weight']][X_test['RunNumber'] == 514564], X_test[[vari, 'Weight']][X_test['RunNumber'] == 514565]])
    EFT_LDS = pd.concat([X_test[[vari, 'Weight']][X_test['RunNumber'] == 514638], X_test[[vari, 'Weight']][X_test['RunNumber'] == 514639]])

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

if vari == 'mll':
    start = 10
elif vari == 'met':
    start = 50
else: 
    start = 0

if int(max(df_features[vari])) > 1000:
    x_axis = np.linspace(start, 2000, 50 )
    
elif 'Phi' in vari:
    x_axis = np.linspace(-np.pi, np.pi, 50 )     
    
elif 'Eta' in vari:
    x_axis = np.linspace(-np.pi, np.pi, 50 ) 
    
else:
    x_axis = np.linspace(start, 10, 11 )

if 'Light' in vari:
    x_axis = np.linspace(0, 20, 21)
    
if vari == 'rt':
    x_axis = np.linspace(0, 10, 11 )
            
if vari == 'jet3Eta':
    x_axis = np.linspace(-np.pi, np.pi, 50)

DY = []; ST = []; DB = []; W = []; TT = []; LV = []; DH = []; EFT = []
DY_w = []; ST_w = []; DB_w = []; W_w = []; TT_w = []; LV_w = []; DH_w = []; EFT_w = []
for DSID, var, w in zip(DSIDs, df_features[vari], WGTs):
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

hist = [W, DB, TT, ST, DY]
hist_w = [W_w, DB_w, TT_w, ST_w, DY_w]
colors = ['#218C8D', '#6CCECB', '#F9E559', '#EF7126', '#8EDC9D']
labels = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan']
d_te = plt.hist(df_data[vari], bins = x_axis)
x_axis_data = []
widths = []
for i in range(len(x_axis) - 1):
    bin_width = x_axis[i+1] - x_axis[i]
    bin_center = x_axis[i] + bin_width/2
    widths.append(bin_width)
    x_axis_data.append(bin_center)
    
bkg = df_features
bkg_hist, bins, patches = plt.hist(bkg[vari], weights = WGTs, bins = x_axis)
data_hist, bins, patches = plt.hist(df_data[vari], bins = x_axis)

unc_data = np.sqrt(data_hist)
stat_unc_bkgs = stat_unc(np.asarray(bkg[vari]), x_axis, np.asarray(WGTs))
syst_unc_bkgs = bkg_hist*0.2                                            
unc_bkg = np.sqrt((stat_unc_bkgs/bkg_hist)**2 + 0.2**2)

np.seterr(divide='ignore', invalid='ignore')                            
ratio = data_hist/bkg_hist
unc_ratio = ratio*np.sqrt( (unc_data/data_hist)**2 + (stat_unc_bkgs/bkg_hist)**2)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
fig.subplots_adjust(hspace=0.04)
ax1.hist(hist, weights=hist_w, bins = x_axis, stacked=True, color=colors, label=labels)
ax1.bar(x_axis_data, 2*stat_unc_bkgs, bottom=bkg_hist-stat_unc_bkgs, fill=False, hatch='XXXXX', label='Stat. Unc.', width = widths, lw=0.0, alpha=0.3, edgecolor='r')
ax1.bar(x_axis_data, 2*syst_unc_bkgs, bottom=bkg_hist-syst_unc_bkgs, fill=False, hatch='XXXXX', label='Syst. Unc.', width = widths, lw=0.0, alpha=0.3)
    
ax1.hist(DH_HDS[vari], weights = DH_HDS['Weight'], bins = x_axis, label="DH HDS $m_{Z'}$ 130", zorder = 5, histtype='step')
ax1.hist(DH_LDS[vari], weights = DH_LDS['Weight'], bins = x_axis, label="DH LDS $m_{Z'}$ 130", zorder = 5, histtype='step')
ax1.hist(LV_HDS[vari], weights = LV_HDS['Weight'], bins = x_axis, label="LV HDS $m_{Z'}$ 130", zorder = 5, histtype='step')
ax1.hist(LV_LDS[vari], weights = LV_LDS['Weight'], bins = x_axis, label="LV LDS $m_{Z'}$ 130", zorder = 5, histtype='step')
ax1.hist(EFT_HDS[vari], weights = EFT_HDS['Weight'], bins = x_axis, label="EFT HDS $m_{Z'}$ 130", zorder = 5, histtype='step')
ax1.hist(EFT_LDS[vari], weights = EFT_LDS['Weight'], bins = x_axis, label="EFT LDS $m_{Z'}$ 130", zorder = 5, histtype='step')

ax1.errorbar(x_axis_data, d_te[0], yerr = unc_data, fmt='o', color='black', label='Data', zorder = 10)
ax2.set_xlabel(var_ax[vari]); ax1.set_ylabel('Events'); 
if train == '1':
    fig.suptitle("Distribution with $\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, \n Training dataset in CR, 20% syst. uncertainty")
else: 
    fig.suptitle("Distribution with $\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, \n Testing dataset in CR, 20% syst. uncertainty")
ax1.set_yscale('log') 
ax1.legend(ncol=2)
ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
ax1.tick_params(bottom=True, top=True, left=True, right=True, which='both')
ax1.set_xlim([x_axis[0], x_axis[-1]])
ax1.set_ylim([1e-3, max(bkg_hist)*7])
ax2.set_xlim([x_axis[0], x_axis[-1]])
ax2.set_ylabel('Events / Bkg')
unc_ratio= np.nan_to_num(unc_ratio)
ax2.errorbar(x_axis_data, ratio, yerr = unc_ratio, fmt='o', color='black')
ax2.plot([x_axis[0], x_axis[-1]], [1,1], 'k-', alpha = 0.3)
ax2.bar(x_axis_data, 2*unc_bkg, bottom=np.ones(len(x_axis_data))-unc_bkg, color='grey', width = widths, lw=0.0, alpha=0.3)
ax2.grid(axis='y')
ax2.set_ylim([0.5,1.6])
if train == '1':
    plt.savefig(plot_dir+vari+'_train.pdf')
else:
    plt.savefig(plot_dir+vari+'_test.pdf')