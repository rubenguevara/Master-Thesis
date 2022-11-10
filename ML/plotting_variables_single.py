import os, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from EventIDs import IDs

model_dir = 'Models/NN/'
save_dir = "../../../storage/racarcam/"
filename = "bkgs.h5"
datafile = 'data.h5'

df_data = pd.read_hdf(save_dir+datafile, key='df')
df_bkgs = pd.read_hdf(save_dir+filename, key='df_tot')

dm_dict_file = open('DM_DICT.json')
DM_DICT = json.load(dm_dict_file)

"""
Choose what you want to plot!
"""
dsid = '514562'
# dsid = '514630'  
variable = 'met_sig'

plot_dir = 'Plots_NeuralNetwork/DSID/'+dsid+'/kinematics/uncut/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

df_dm = pd.read_hdf(save_dir+'DMS/'+dsid+'.h5', key='df_tot')

df = pd.concat([df_bkgs, df_dm]).sort_index()

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_labels = df_features.pop('Label')

X_train, X_test = train_test_split(df_features, test_size=0.2, random_state=42)
D_train, D_test = train_test_split(df_data, test_size=0.2, random_state=42)

x_axis = np.linspace(0, 100, 74)
data_train = np.histogram(D_train[variable], bins = 74, range = (0, 100))
data_test = np.histogram(D_test[variable], bins = 74, range = (0, 100))

DY = []; ST = []; DB = []; W = []; TT = []; DM = []
DY_w = []; ST_w = []; DB_w = []; W_w = []; TT_w = []; DM_w = []
for DSID, var, w in zip(X_train['RunNumber'], X_train[variable], X_train['Weight']):
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
    else:
        DM.append(var)
        DM_w.append(w)

hist = [W, DB, TT, ST, DY]
hist_w = [W_w, DB_w, TT_w, ST_w, DY_w]
colors = ['#218C8D', '#6CCECB', '#F9E559', '#EF7126', '#8EDC9D']
labels = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan']

plt.figure(1, figsize=[15,10])
plt.hist(hist, weights=hist_w ,bins=x_axis, log=True, stacked=True, color=colors, label=labels, zorder=0)
plt.hist(DM, weights=DM_w ,bins=x_axis, log=True, color='#F42069', label='Signal', zorder = 5)
plt.scatter(x_axis, data_train[0], color='black', label='Data', zorder = 10)
plt.xlabel('$E_{T}^{miss}/\sigma$'); plt.ylabel('Events'); plt.title('MET SIG training set for '+ DM_DICT[dsid]); plt.legend(); plt.ylim([1e-5,1e8])
plt.savefig(plot_dir+variable+'_train.pdf')
plt.show()


DY = []; ST = []; DB = []; W = []; TT = []; DM = []
DY_w = []; ST_w = []; DB_w = []; W_w = []; TT_w = []; DM_w = []
for DSID, var, w in zip(X_test['RunNumber'], X_test[variable], X_test['Weight']):
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
    else:
        DM.append(var)
        DM_w.append(w)

hist = [W, DB, TT, ST, DY]
hist_w = [W_w, DB_w, TT_w, ST_w, DY_w]
colors = ['#218C8D', '#6CCECB', '#F9E559', '#EF7126', '#8EDC9D']
labels = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan']

plt.figure(2, figsize=[15,10])
plt.hist(hist, weights=hist_w, bins=x_axis, log=True, stacked=True, color=colors, label=labels, zorder = 0)
plt.hist(DM, weights=DM_w ,bins=x_axis, log=True, color='#F42069', label='Signal', zorder = 5)
plt.scatter(x_axis, data_test[0], color='black', label='Data', zorder = 10)
plt.xlabel('$E_{T}^{miss}/\sigma$'); plt.ylabel('Events'); plt.title('MET SIG test set for '+ DM_DICT[dsid]); plt.legend(); plt.ylim([1e-5,1e8])
plt.savefig(plot_dir+variable+'_test.pdf')
plt.show()

dm_dict_file.close()