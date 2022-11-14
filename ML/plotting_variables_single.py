import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from EventIDs import IDs
import multiprocessing as mp

model_dir = 'Models/NN/'
save_dir = "../../../storage/racarcam/"
filename = "FULL_DM_50MET.h5"

df = pd.read_hdf(save_dir+filename, key='df_tot')


"""
Choose what you want to plot!
"""
dsid_LV_HDS_MZ_130 = [514562, 514563] 
dsid_DH_HDS_MZ_130 = [514560, 514561] 
dsid_EFT_HDS_MZ_130 = [514564, 514565] 

chnl = 'uu'

plot_dir = 'Plots_data_analysis/HDS_MZp_130_'+chnl+'/CUT/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

df_chnl = df.loc[df['Dileptons'] == chnl]

if chnl == 'uu':
    chnl = '$\mu\mu$'

df_features = df_chnl.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                           # "Bad" variable
df_dPhiLeps = df_features.pop('dPhiLeps')                                   # "Bad" variable

s_LV_u = sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_LV_HDS_MZ_130[0]])
s_LV_u += sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_LV_HDS_MZ_130[1]])
s_DH_u = sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_DH_HDS_MZ_130[0]])
s_DH_u += sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_DH_HDS_MZ_130[1]])
s_EFT_u = sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_EFT_HDS_MZ_130[0]])
s_EFT_u += sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_EFT_HDS_MZ_130[1]])
b_u = sum(df_features['Weight'].loc[df_features['Label'] == 0])

print('===='*12, 'Events before cuts', '===='*12)
print(int(s_LV_u), 'Light Vector events')
print(int(s_DH_u), 'Dark Higgs events')
print(int(s_EFT_u), ' EFT events')
print(int(b_u), 'background events')

def low_stat_Z(sig, bkg):
    Z = np.sqrt(2*( (sig + bkg)*np.log(1 + sig/bkg) - sig ))
    return Z

""" Make cuts here """
df_features = df_features.loc[df_features['met_sig'] > 10]                  
df_features = df_features.loc[df_features['mt'] > 160]                      
df_features = df_features.loc[df_features['mll'] > 120]                     
df_features = df_features.loc[df_features['jetB'] == 0]                               
df_features = df_features.loc[df_features['mt2'] > 110]        


variables = df_features.columns[:-4]

print('===='*12, 'Events after cuts ', '===='*12)
s_LV = sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_LV_HDS_MZ_130[0]])
s_LV += sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_LV_HDS_MZ_130[1]])
s_DH = sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_DH_HDS_MZ_130[0]])
s_DH += sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_DH_HDS_MZ_130[1]])
s_EFT = sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_EFT_HDS_MZ_130[0]])
s_EFT += sum(df_features['Weight'].loc[df_features['RunNumber'] == dsid_EFT_HDS_MZ_130[1]])
b = sum(df_features['Weight'].loc[df_features['Label'] == 0])


print(int(s_LV), 'Light Vector events')
print(int(s_DH), 'Dark Higgs events')
print(int(s_EFT), ' EFT events')
print(int(b), 'background events')


print('===='*29)
print('The expected significance for Light Vector is        %.3f sigma' %low_stat_Z(s_LV, b))
print('The expected significance for Dark Higgs is          %.3f sigma' %low_stat_Z(s_DH, b))
print('The expected significance for EFT is                 %.3f sigma' %low_stat_Z(s_EFT, b))



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

    x_axis_gen = np.linspace(20, 3500, 74)
    x_eta = np.linspace(-3, 3, 50);
    x_met = np.linspace(20, 2500, 74)
    x_mt2 = np.linspace(20, 1500, 74)
    x_met_sig = np.linspace(0, 100, 74)
    x_et = np.linspace(20, 3000, 74)
    x_phi = np.linspace(-np.pi, np.pi, 50)
    x_dphi = np.linspace(0, np.pi, 30) 
    x_jets = np.linspace(0, 7, 8)

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

    plt.figure(figsize=[12,7])
    plt.hist(hist, weights=hist_w, bins = x_axis, stacked=True, color=colors, label=labels)
    plt.hist(LV, weights=LV_w, bins = x_axis,  label='Light Vector', histtype='step', color='#F42069')
    plt.hist(DH, weights=DH_w, bins = x_axis,  label='Dark Higgs', histtype='step', color = 'pink')
    plt.hist(EFT, weights=EFT_w, bins = x_axis,  label='Effective Field Theory', histtype='step', color='red')
    plt.figtext(0.3, 0.82, 'ATLAS', fontstyle='italic', fontweight='bold')
    plt.figtext(0.3 + 0.045, 0.82, 'Preliminary')
    plt.figtext(0.3, 0.79, '$>50GeV$ $E_{T}^{miss}$, '+chnl)
    plt.xlabel(var_ax[variable]); plt.ylabel('Events'); 
    plt.title("Signal search for Heavy Dark Sector and $m_{Z'}=130G$eV")
    plt.yscale('log') 
    plt.legend(); 
    plt.xlim([x_axis[0], x_axis[-1]])
    plt.savefig(plot_dir+variable+'.pdf')
    plt.show()
    
with mp.Pool(processes=len(variables)) as pool:
    pool.map(plot_maker, variables)
pool.close()