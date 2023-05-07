import os, time, json, argparse
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from EventIDs import IDs
from Plot_maker import low_stat_Z

print(xgb.__version__)

t0 = time.time()
start = time.asctime(time.localtime())
print('Started', start)


parser = argparse.ArgumentParser()
parser.add_argument('--met_reg', type=str, default="50-100", help="MET signal region")
parser.add_argument('--dm_model', type=str, default="DH_HDS", help="Dataset to test")
parser.add_argument('--channel', type=str, default="ee", help="Lepton channel to test")
args = parser.parse_args()

met_reg = args.met_reg
dm_model = args.dm_model
channel = args.channel 



N = 9
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.PuRd_r(np.linspace(0.1,0.95,N)))

def Z_score_array(sig_pred, bkg_pred):
    np.seterr(divide='ignore', invalid='ignore')                                    # Remove true divide message
    return [low_stat_Z(sum(sig_pred[25:]), sum(bkg_pred[25:])),          
                low_stat_Z(sum(sig_pred[30:]), sum(bkg_pred[30:])), 
                low_stat_Z(sum(sig_pred[35:]), sum(bkg_pred[35:])),
                low_stat_Z(sum(sig_pred[40:]), sum(bkg_pred[40:])), 
                low_stat_Z(sum(sig_pred[45:]), sum(bkg_pred[45:])), 
                low_stat_Z(sig_pred[-1], bkg_pred[-1])]

np_dir = '../Data/XGB/'+met_reg+'/'+dm_model+'/'

sig_mzp_130 = np.load(np_dir+'mZp_130/sig_pred_'+channel+'.npy')
sig_mzp_200 = np.load(np_dir+'mZp_200/sig_pred_'+channel+'.npy')
sig_mzp_400 = np.load(np_dir+'mZp_400/sig_pred_'+channel+'.npy')
sig_mzp_600 = np.load(np_dir+'mZp_600/sig_pred_'+channel+'.npy')

bkg_mzp_130 = np.load(np_dir+'mZp_130/bkg_pred_'+channel+'.npy')
bkg_mzp_200 = np.load(np_dir+'mZp_200/bkg_pred_'+channel+'.npy')
bkg_mzp_400 = np.load(np_dir+'mZp_400/bkg_pred_'+channel+'.npy')
bkg_mzp_600 = np.load(np_dir+'mZp_600/bkg_pred_'+channel+'.npy')

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

plot_dir = '../../Plots/XGBoost/Model_independent/'+met_reg+'/'+dm_model+'/'

plt.figure(figsize=(11,8))
X_axis = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
Y_axis_130 = Z_score_array(sig_mzp_130, bkg_mzp_130)
Y_axis_200 = Z_score_array(sig_mzp_200, bkg_mzp_200)
Y_axis_400 = Z_score_array(sig_mzp_400, bkg_mzp_400)
Y_axis_600 = Z_score_array(sig_mzp_600, bkg_mzp_600)

plt.figure(figsize=[10,6])
plt.plot(X_axis, Y_axis_130, linestyle='--')
plt.scatter(X_axis, Y_axis_130, label = "$m_{Z'}$ 130 GeV")
plt.plot(X_axis, Y_axis_200, linestyle='--')
plt.scatter(X_axis, Y_axis_200, label = "$m_{Z'}$ 200 GeV")
plt.plot(X_axis, Y_axis_400, linestyle='--')
plt.scatter(X_axis, Y_axis_400, label = "$m_{Z'}$ 400 GeV")
plt.plot(X_axis, Y_axis_600, linestyle='--')
plt.scatter(X_axis, Y_axis_600, label = "$m_{Z'}$ 600 GeV")
plt.xlim([0,1])
plt.ylim([np.nanmin(Y_axis_600)*0.9, np.nanmax(Y_axis_130)*1.1])
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.ylabel('Expected significance [$\sigma$]')
if met_reg =='50-100':
    plt.title("Significance on "+dm_model.split('_')[0]+' '+dm_model.split('_')[1]+" "+channel+", trained network on SR1")
elif met_reg =='100-150':
    plt.title("Significance on "+dm_model.split('_')[0]+' '+dm_model.split('_')[1]+" "+channel+", trained network on SR2")
elif met_reg =='150':
    plt.title("Significance on "+dm_model.split('_')[0]+' '+dm_model.split('_')[1]+" "+channel+", trained network on SR3")
plt.xlabel('XGBoost output')
plt.savefig(plot_dir+'EXP_SIG_'+channel+'.pdf')

