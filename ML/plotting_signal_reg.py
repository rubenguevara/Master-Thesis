import os, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.cm import RdPu_r, YlGnBu_r

model_dir = 'Models/NN/'
save_dir = "../../../storage/racarcam/"
filename = 'Full_DM_sig.h5'

df = pd.read_hdf(save_dir+filename, key='df_tot')

dsid = ''                                                                   # Change to specific DSID if desired

"""Choose what variables to plot"""
variable_x = 'mll'
variable_y = 'met'

dsid_file = open('DM_DICT.json')

DM_DICT = json.load(dsid_file)

var_name = {}
var_name['lep1Pt'] = '$p_{T}^{1}$ [GeV]'
var_name['lep2Pt'] = '$p_{T}^{2}$ [GeV]'
var_name['lep1Eta'] = '$\eta_{1}$'
var_name['lep2Eta'] = '$\eta_{2}$'
var_name['jet1Pt'] = 'jet $p_{T}^{1}$ [GeV]'
var_name['jet2Pt'] = 'jet $p_{T}^{2}$ [GeV]'
var_name['jet1Eta'] = 'jet $\eta_{1}$'
var_name['jet2Eta'] = 'jet $\eta_{2}$'
var_name['mll'] = '$m_{ll}$ [GeV]'
var_name['met'] = '$E_{T}^{miss}$ [GeV]'
var_name['met_sig'] = '$E_{T}^{miss}/sigma$'
var_name['mt'] = '$m_{T}$ [GeV]'
var_name['ht'] = '$H_{T}$ [GeV]'
var_name['dPhiLeadMet'] = '$|\Delta\phi(l_{lead}, E_{T}^{miss})|$'
var_name['dPhiLLMet'] = '$|\Delta\phi(ll, E_{T}^{miss})|$'
var_name['mt2'] = '$m_{T2}$ [GeV]'
var_name['jetB'] = 'Number of B jets'
var_name['jetLight'] = 'Number of light jets'
var_name['jetTot'] = 'Total number of jets'
var_name['et'] = '$E_{T}$ [GeV]'
var_name['lep1Phi'] = '$\phi_{1}$'
var_name['lep2Phi'] = '$\phi_{2}$'
var_name['jet1Phi'] = 'jet $\phi_{1}$'
var_name['jet2Phi'] = 'jet $\phi_{2}$'


var_ax = {}
var_ax['lep1Pt'] = np.linspace(20, 3500, 74)
var_ax['lep2Pt'] = np.linspace(20, 3500, 74)
var_ax['lep1Eta'] = np.linspace(-3, 3, 50)
var_ax['lep2Eta'] = np.linspace(-3, 3, 50)
var_ax['jet1Pt'] = np.linspace(20, 3500, 74)
var_ax['jet2Pt'] = np.linspace(20, 3500, 74)
var_ax['jet1Eta'] = np.linspace(-3, 3, 50)
var_ax['jet2Eta'] = np.linspace(-3, 3, 50)
var_ax['mll'] = np.linspace(20, 3500, 74)
var_ax['met'] = np.linspace(20, 2500, 74)
var_ax['met_sig'] = np.linspace(0, 100, 74)
var_ax['mt'] = np.linspace(20, 1500, 74)
var_ax['ht'] = np.linspace(20, 3500, 74)
var_ax['dPhiLeadMet'] = np.linspace(0, np.pi, 30)
var_ax['dPhiLLMet'] = np.linspace(0, np.pi, 30)
var_ax['mt2'] = np.linspace(20, 1500, 74)
var_ax['jetB'] = np.linspace(0, 7, 8)
var_ax['jetLight'] = np.linspace(0, 7, 8)
var_ax['jetTot'] = np.linspace(0, 7, 8)
var_ax['et'] = np.linspace(20, 3000, 74)
var_ax['lep1Phi'] = np.linspace(-np.pi, np.pi, 50)
var_ax['lep2Phi'] = np.linspace(-np.pi, np.pi, 50)
var_ax['jet1Phi'] = np.linspace(-np.pi, np.pi, 50)
var_ax['jet2Phi'] = np.linspace(-np.pi, np.pi, 50)

if dsid == '':
    df_dm = df.loc[df['Label']==1]             
    data = 'full DM dataset'         
    plot_dir = '../Plots/Signal_Region/FULL/'

else:                       
    df_dm = df.loc[df['RunNumber']==int(dsid)]          
    data = DM_DICT[dsid]                 
    plot_dir = '../Plots/Signal_Region/'+dsid+'/'

dsid_file.close()

df = df.loc[df['Label']==0]

xaxis_x = var_ax[variable_x]                                                # Variable x
xaxis_y = var_ax[variable_y]                                                # Variable y


try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass


x_bkg = df[variable_x]
y_bkg = df[variable_y]
w_bkg = df['Weight']
sow_bkg = sum(w_bkg)
wgts_bkg = w_bkg/sow_bkg

x = df_dm[variable_x]
y = df_dm[variable_y]
w = df_dm['Weight']
sow = sum(w)
wgts = w/sow


fig, (ax_sig, ax_bkg) = plt.subplots(1, 2, figsize=(16, 6))
image_sig = ax_sig.hist2d(x, y, weights = wgts, bins = (xaxis_x, xaxis_y), cmap = RdPu_r, norm=LogNorm(vmin=1e-5))
image_bkg = ax_bkg.hist2d(x_bkg, y_bkg, weights = wgts_bkg, bins = (xaxis_x, xaxis_y), cmap = YlGnBu_r, norm=LogNorm(vmin=1e-5))
cbar_sig = fig.colorbar(image_sig[3], ax = ax_sig, location = 'right')
ax_sig.set_title('Signal')

cbar_bkg = fig.colorbar(image_bkg[3], ax = ax_bkg, location = 'right')

ax_bkg.set_title('Background')
ax_sig.set_ylabel(var_name[variable_y])
ax_sig.set_xlabel(var_name[variable_x])
ax_bkg.set_ylabel(var_name[variable_y])
ax_bkg.set_xlabel(var_name[variable_x])
ax_sig.grid()
ax_bkg.grid()

fig.suptitle('Signal region search for '+var_name[variable_x].split(' ')[0]+' and '+var_name[variable_y].split(' ')[0]+' on '+data, fontsize='x-large')

plt.savefig(plot_dir+variable_x+'_'+variable_y+'.pdf', bbox_inches='tight')
plt.show()
