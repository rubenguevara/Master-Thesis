import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from EventIDs import IDs
import multiprocessing as mp
from matplotlib import ticker as mticker

model_dir = 'Models/NN/'
save_dir = "../../../storage/racarcam/"
filename = "FULL_Zp_50MET_no_pad.h5"

df = pd.read_hdf(save_dir+filename, key='df_tot')
datafile = 'data_no_pad.h5'

df_data_unfiltered = pd.read_hdf(save_dir+datafile, key='df')

"""
Choose what you want to plot!
"""
dsid_LV_HDS_MZ_130 = [514562, 514563] 
dsid_DH_HDS_MZ_130 = [514560, 514561] 
dsid_EFT_HDS_MZ_130 = [514564, 514565] 

chnl = 'ee'

plot_dir = '../Plots/Data_Analysis/HDS_MZp_130_'+chnl+'/new_variables/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass


def low_stat_Z(sig, bkg):
    Z = np.sqrt(2*( (sig + bkg)*np.log(1 + sig/bkg) - sig ))
    return Z


def stat_unc(prediction, bins, weights):
    binning = bins
    histo_bins = np.digitize(prediction, binning)
    stat_unc_array = []
    for i in range(1,len(binning)):
        bin_wgt = weights[np.where(histo_bins==i)[0]]
        sow_bin = np.linalg.norm(bin_wgt,2)
        stat_unc_array.append(sow_bin)
    return np.asarray(stat_unc_array)

df_chnl = df.loc[df['Dileptons'] == chnl]
df_data = df_data_unfiltered.loc[df['Dileptons'] == chnl]


if chnl == 'uu':
    chnl = '$\mu\mu$'

df_features = df_chnl.copy()

# variables = df_features.columns[:-4]
variables = ['n_bjetPt20', 'n_bjetPt30', 'n_bjetPt40', 'n_bjetPt50', 'n_bjetPt60', 'n_ljetPt20', 
            'n_ljetPt30', 'n_ljetPt40', 'n_ljetPt50', 'n_ljetPt60', 'jetEtaCentral', 'jetEtaForward']



def plot_maker(variable):
    var_ax = {}

    var_ax['lep1Pt'] = '$p_{T}^{1}$ [GeV]'
    var_ax['lep2Pt'] = '$p_{T}^{2}$ [GeV]'
    var_ax['lep1Eta'] = '$\eta_{1}$'
    var_ax['lep2Eta'] = '$\eta_{2}$'
    var_ax['jet1Pt'] = 'jet $p_{T}^{1}$ [GeV]'
    var_ax['jet2Pt'] = 'jet $p_{T}^{2}$ [GeV]'
    var_ax['jet1Eta'] = 'jet $\eta_{1}$'
    var_ax['jet2Eta'] = 'jet $\eta_{2}$'
    var_ax['mll'] = '$m_{ll}$ [GeV]'
    var_ax['met'] = '$E_{T}^{miss}$ [GeV]'
    var_ax['met_sig'] = '$E_{T}^{miss}/sigma$'
    var_ax['mt'] = '$m_{T}$ [GeV]'
    var_ax['ht'] = '$H_{T}$ [GeV]'
    var_ax['dPhiLeadMet'] = '$|\Delta\phi(l_{lead}, E_{T}^{miss})|$'
    var_ax['dPhiLLMet'] = '$|\Delta\phi(ll, E_{T}^{miss})|$'
    var_ax['mt2'] = '$m_{T2}$ [GeV]'
    var_ax['jetB'] = 'Number of B jets'
    var_ax['jetLight'] = 'Number of light jets'
    var_ax['jetTot'] = 'Total number of jets'
    var_ax['et'] = '$E_{T}$ [GeV]'
    var_ax['lep1Phi'] = '$\phi_{1}$'
    var_ax['lep2Phi'] = '$\phi_{2}$'
    var_ax['jet1Phi'] = 'jet $\phi_{1}$'
    var_ax['jet2Phi'] = 'jet $\phi_{2}$'
    var_ax['jetEtaForward'] = '# jets with $|\eta| >2.5$'
    var_ax['jetEtaCentral'] = '# jets with $|\eta| <2.5$'
    var_ax['n_bjetPt20'] = '# b-jets with $p_T > 20 GeV$'
    var_ax['n_bjetPt30'] = '# b-jets with $p_T > 30 GeV$'
    var_ax['n_bjetPt40'] = '# b-jets with $p_T > 40 GeV$'
    var_ax['n_bjetPt50'] = '# b-jets with $p_T > 50 GeV$'
    var_ax['n_bjetPt60'] = '# b-jets with $p_T > 60 GeV$'
    var_ax['n_ljetPt20'] = '# l-jets with $p_T > 20 GeV$'
    var_ax['n_ljetPt30'] = '# l-jets with $p_T > 30 GeV$'
    var_ax['n_ljetPt40'] = '# l-jets with $p_T > 40 GeV$'
    var_ax['n_ljetPt50'] = '# l-jets with $p_T > 50 GeV$'
    var_ax['n_ljetPt60'] = '# l-jets with $p_T > 60 GeV$'

    x_axis_gen = np.linspace(20, 3500, 74)
    x_eta = np.linspace(-3, 3, 50);
    x_met = np.linspace(20, 2500, 74)
    x_mt2 = np.linspace(20, 1500, 74)
    x_met_sig = np.linspace(0, 100, 74)
    x_et = np.linspace(20, 3000, 74)
    x_phi = np.linspace(-np.pi, np.pi, 50)
    x_dphi = np.linspace(0, np.pi, 30) 
    x_jets = np.linspace(0, 7, 8)
    x_jet_n = np.linspace(0, max(df_features[variable]), max(df_features[variable])+1 )

    x_axis = x_axis_gen
    if 'Phi' in variable:
        x_axis = x_phi
    
    if 'dPhi' in variable:
        x_axis = x_dphi
        
    if 'Eta' in variable:
        x_axis = x_eta
    
    if variable == 'met':
        x_axis = x_met
    
    if variable =='mt2':
        x_axis = x_mt2
    
    if variable =='met_sig':
        x_axis = x_met_sig
    
    if variable == 'jetB' or variable == 'jetLight' or variable == 'jetTot':
        x_axis = x_jets
    
    if 'jet' in variable:
        x_axis = x_jet_n
    
    if variable == 'et':
        x_axis = x_et
    
    DY = []; ST = []; DB = []; W = []; TT = []; LV = []; DH = []; EFT = []
    DY_w = []; ST_w = []; DB_w = []; W_w = []; TT_w = []; LV_w = []; DH_w = []; EFT_w = []
    for DSID, var, w in zip(df_features['RunNumber'], df_features[variable], df_features['Weight']):
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
        elif DSID in dsid_DH_HDS_MZ_130:
            DH.append(var)
            DH_w.append(w)
        elif DSID in dsid_LV_HDS_MZ_130:
            LV.append(var)
            LV_w.append(w)
        elif DSID in dsid_EFT_HDS_MZ_130:
            EFT.append(var)
            EFT_w.append(w)

    hist = [W, DB, TT, ST, DY]
    hist_w = [W_w, DB_w, TT_w, ST_w, DY_w]
    colors = ['#218C8D', '#6CCECB', '#F9E559', '#EF7126', '#8EDC9D']
    labels = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan']
    d_te = np.histogram(df_data[variable], bins = max(df_features[variable]), range = (0,max(df_features[variable])))
    x_axis_data = np.linspace(0.5, max(df_features[variable])-0.5, max(df_features[variable]))
    
    bkg = df_features.loc[df_features['Label']==0]
    bkg_hist, bins, patches = plt.hist(bkg[variable], weights = bkg['Weight'], bins = x_axis)
    data_hist, bins, patches = plt.hist(df_data[variable], bins = x_axis)
    
    unc_data = np.sqrt(data_hist)
    stat_unc_bkgs = stat_unc(np.asarray(bkg[variable]), x_axis, np.asarray(bkg['Weight']))
    syst_unc_bkgs = bkg_hist*0.3                                            # Assuming 30% systematic uncertainty
    unc_bkg = np.sqrt(stat_unc_bkgs**2 + syst_unc_bkgs**2)
    
    ratio = data_hist/bkg_hist
    unc_ratio = ratio*np.sqrt( (unc_data/data_hist)**2 + (unc_bkg/bkg_hist)**2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.04)
    ax1.hist(hist, weights=hist_w, bins = x_axis, stacked=True, color=colors, label=labels)
    ax1.bar(x_axis_data, 2*unc_bkg, bottom=bkg_hist-unc_bkg, fill=False, hatch='XXXXX', label='Stat. + Syst. Unc.', width = x_axis[1]-x_axis[0], lw=0.0, alpha=0.3)
    ax1.hist(LV, weights=LV_w, bins = x_axis,  label='Light Vector', histtype='step', color='#F42069')
    ax1.hist(DH, weights=DH_w, bins = x_axis,  label='Dark Higgs', histtype='step', color = 'pink')
    ax1.hist(EFT, weights=EFT_w, bins = x_axis,  label='Effective Field Theory', histtype='step', color='red')
    ax1.errorbar(x_axis_data, d_te[0], yerr = unc_data, fmt='o', color='black', label='Data', zorder = 10, capsize = 5)
    ax2.set_xlabel(var_ax[variable]); ax1.set_ylabel('Events'); 
    fig.suptitle("Signal search for Heavy Dark Sector and $m_{Z'}=130G$eV \n $\sqrt{s} = 13$ TeV, 139 fb$^{-1}$, $>50GeV$ $E_{T}^{miss}$, "+chnl)
    ax1.set_yscale('log') 
    ax1.legend(ncol=2)
    ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax1.tick_params(bottom=True, top=True, left=True, right=True, which='both')
    ax1.set_xlim([x_axis[0], x_axis[-1]])
    ax1.set_ylim([1e-3, max(bkg_hist)*7])
    ax2.set_xlim([x_axis[0], x_axis[-1]])
    ax2.set_ylabel('Events / Bkg')
    ax2.errorbar(x_axis_data, ratio, yerr = unc_ratio, fmt='o', color='black', ecolor='grey')
    ax2.grid(axis='y')
    ax2.set_ylim([0.5,1.6])
    plt.savefig(plot_dir+variable+'.pdf')
    plt.show()
    
with mp.Pool(processes=len(variables)) as pool:
    pool.map(plot_maker, variables)
pool.close()