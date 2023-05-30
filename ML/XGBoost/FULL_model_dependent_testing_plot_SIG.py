import os, json, argparse, random
import matplotlib.pyplot as plt
import numpy as np
from Plot_maker import low_stat_Z

parser = argparse.ArgumentParser()
parser.add_argument('--dm_model', type=str, default="DH_HDS", help="Dataset to test")
parser.add_argument('--channel', type=str, default="ee", help="Lepton channel to test")
parser.add_argument('--tanb', type=str, default="1", help="Lepton channel to test")
args = parser.parse_args()

dm_model = args.dm_model
channel = args.channel 
tanb = args.tanb 

def Z_score_array(sig_pred, bkg_pred, bkg_unc=None):
    np.seterr(divide='ignore', invalid='ignore')                                   
    if bkg_unc is None:
        Z = [low_stat_Z(sum(sig_pred[25:]), sum(bkg_pred[25:])),          
                low_stat_Z(sum(sig_pred[30:]), sum(bkg_pred[30:])), 
                low_stat_Z(sum(sig_pred[35:]), sum(bkg_pred[35:])),
                low_stat_Z(sum(sig_pred[40:]), sum(bkg_pred[40:])), 
                low_stat_Z(sum(sig_pred[45:]), sum(bkg_pred[45:])), 
                low_stat_Z(sig_pred[-1], bkg_pred[-1])]
    else:
        Z = [low_stat_Z(sum(sig_pred[25:]), sum(bkg_pred[25:]), np.linalg.norm(bkg_unc[25:],2)),          
                low_stat_Z(sum(sig_pred[30:]), sum(bkg_pred[30:]), np.linalg.norm(bkg_unc[30:],2)), 
                low_stat_Z(sum(sig_pred[35:]), sum(bkg_pred[35:]), np.linalg.norm(bkg_unc[35:],2)),
                low_stat_Z(sum(sig_pred[40:]), sum(bkg_pred[40:]), np.linalg.norm(bkg_unc[40:],2)), 
                low_stat_Z(sum(sig_pred[45:]), sum(bkg_pred[45:]), np.linalg.norm(bkg_unc[45:],2)), 
                low_stat_Z(sig_pred[-1], bkg_pred[-1], bkg_unc[-1])]
    
    return Z
    
save_dir = "/storage/racarcam/"
np_dir = save_dir+'Data/'+dm_model+'_scores/'

directories = [] 
for (dirpath, dirnames, filenames) in os.walk(np_dir):
    if filenames == []: continue
    if 'MET75' in dirpath: continue
    if 'D10' in dirpath: continue
    directories.append(dirpath)
    continue

random.Random(69).shuffle(directories)

directories_plot = directories[:10]

N = 15
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.PuRd_r(np.linspace(0.1,0.95,N)))
if channel != 'll':
    plot_dir = '../../Plots/XGBoost/'+dm_model+'/'
    plt.figure(figsize=(10,6))
    X_axis = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    for dirpath in directories_plot:
        siggy = np.load(dirpath+'/sig_pred_'+channel+'.npy')
        bkg = np.load(dirpath+'/bkg_pred_'+channel+'.npy')
        bkg_unc = np.load(dirpath+'/unc_bkg_'+channel+'.npy')
        Y_axis = Z_score_array(siggy, bkg, bkg_unc)
        if dm_model =='SlepSlep':
            slep = dirpath.split('_')[-2][1:]
            neut = dirpath.split('_')[-1][1:]
            label = '($m_{\\tilde{\ell}},m_{\\tilde{\chi}_1^0}$)=('+slep+','+neut+')'
        if dm_model =='2HDM':
            slep = dirpath.split('_')[-2][1:]
            neut = dirpath.split('_')[-3][1:]
            tb = dirpath.split('_')[-1][2:]
            label = '($m_{a},m_{H},\\tan\\beta$)=('+slep+','+neut+','+tb+')'
        plt.plot(X_axis, Y_axis, linestyle='--')
        plt.scatter(X_axis, Y_axis, label = label)

    plt.xlim([0,1])
    # plt.ylim([np.nanmin(Y_axis_600)*0.9, np.nanmax(Y_axis_130)*1.1])
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Expected significance [$\sigma$]')
    if dm_model == 'SlepSlep':
        plt.title("Significance on direct slepton model "+channel+" channel, trained network on full susy dataset")
    if dm_model == '2HDM':
        plt.title("Significance on 2 Higgs Doublet Model "+channel+" channel, trained network on full 2HDM + a dataset")
    plt.xlabel('XGBoost output')
    plt.savefig(plot_dir+'EXP_SIG_'+channel+'.pdf')

save_txt_path = '../../Plots/Limits/'+dm_model+'/'

try:
    os.makedirs(save_txt_path)

except FileExistsError:
    pass

try:
    if dm_model == '2HDM': 
        os.remove(save_txt_path+dm_model+'_'+channel+'_tb'+tanb+'.txt')
    else:
        os.remove(save_txt_path+dm_model+'_'+channel+".txt")
    print('Updating old file')
except:
    print('Nothing to update')

if dm_model == '2HDM': 
    text_file = open(save_txt_path+dm_model+'_'+channel+'_tb'+tanb+'.txt', 'w')
else:
    text_file = open(save_txt_path+dm_model+'_'+channel+'.txt', 'w')
text_file.write('m1 m2 sign \n')


if dm_model == 'SlepSlep':
    m_y = "'m_{ #tilde{#chi}_{1}^{0}} [GeV]'"

    if channel == 'ee':
        title = "'#tilde{e}#tilde{e} #rightarrow ee #tilde{#chi}_{1}^{0}#tilde{#chi}_{1}^{0}, 20% systematic uncertainty'"
        m_x = "'m_{ #tilde{e}} [GeV]'"
    if channel == 'uu':
        title = "'#tilde{#mu}#tilde{#mu} #rightarrow #mu#mu #tilde{#chi}_{1}^{0}#tilde{#chi}_{1}^{0}, 20% systematic uncertainty'"
        m_x = "'m_{ #tilde{#mu}} [GeV]'"
    if channel == 'll':
        m_x = "'m_{ #tilde{l}} [GeV]'"
        title = "'#tilde{l} #tilde{l} #rightarrow ll #tilde{#chi}_{1}^{0}#tilde{#chi}_{1}^{0}, 20% systematic uncertainty'"

if dm_model == '2HDM':
    m_y = "'m_{a} [GeV]'"
    title = "'tan#beta = "+tanb+"'"
    m_x= "'m_{H^-} [GeV]'"

if channel == 'll':
    for dirpath in directories:
        if dm_model == '2HDM':
            if dirpath.split('_')[-1][2:] != tanb: continue
        siggy_e = np.load(dirpath+'/sig_pred_ee.npy')
        bkg_e = np.load(dirpath+'/bkg_pred_ee.npy')
        bkg_unc_e = np.load(dirpath+'/unc_bkg_ee.npy')
        sig_e = low_stat_Z(siggy_e[-1], bkg_e[-1], bkg_unc_e[-1])

        siggy_u = np.load(dirpath+'/sig_pred_uu.npy')
        bkg_u = np.load(dirpath+'/bkg_pred_uu.npy')
        bkg_unc_u = np.load(dirpath+'/unc_bkg_uu.npy')
        sig_u = low_stat_Z(siggy_u[-1], bkg_u[-1], bkg_unc_u[-1])
        
        sig = np.sqrt(sig_e**2 +sig_u**2)
        if dm_model =='SlepSlep':
            slep = dirpath.split('_')[-2][1:]
            neut = dirpath.split('_')[-1][1:]
        if dm_model =='2HDM':
            slep = dirpath.split('_')[-2][1:]
            neut = dirpath.split('_')[-3][1:]
        text_file.write(slep+' '+neut+' '+str(sig)+'\n')

# exit()
else:
    for dirpath in directories:
        if dm_model == '2HDM':
            if dirpath.split('_')[-1][2:] != tanb: continue
        siggy = np.load(dirpath+'/sig_pred_'+channel+'.npy')
        bkg = np.load(dirpath+'/bkg_pred_'+channel+'.npy')
        bkg_unc = np.load(dirpath+'/unc_bkg_'+channel+'.npy')
        sig = low_stat_Z(siggy[-1], bkg[-1], bkg_unc[-1])
        if dm_model =='SlepSlep':
            slep = dirpath.split('_')[-2][1:]
            neut = dirpath.split('_')[-1][1:]
        if dm_model =='2HDM':
            slep = dirpath.split('_')[-2][1:]
            neut = dirpath.split('_')[-3][1:]
            tb = dirpath.split('_')[-1][2:]
        text_file.write(slep+' '+neut+' '+str(sig)+'\n')

if dm_model == '2HDM':
    print('../../software/Scripts/SUSYPheno/bin/munch.py -coord m1:'+m_x+',m2:'+m_y+' -resvars sign -title '+title+' -fn_table '+save_txt_path+dm_model+'_'+channel+'_tb'+tanb+'.txt -cont sign:Significance:1.645:2:2:3 --legend2bottomright')
else: 
    print('../../software/Scripts/SUSYPheno/bin/munch.py -coord m1:'+m_x+',m2:'+m_y+' -resvars sign -title '+title+' -fn_table '+save_txt_path+dm_model+'_'+channel+'.txt -cont sign:Significance:1.645:2:2:3 --legend2bottomright')
