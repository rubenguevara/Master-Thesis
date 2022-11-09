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

DSIDS = os.listdir('Models/NN/')
dm_dict_file = open('DM_DICT.json')
DM_DICT = json.load(dm_dict_file)

"""
Choose what you want to plot!
"""
dsid = '514603'  
variable = 'mll'

plot_dir = 'Plots_NeuralNetwork/DSID/'+dsid+'_sample_splitting_not_red/'

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
print(df_features)
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
D_train, D_test = train_test_split(df_data, test_size=0.2, random_state=42)

x_axis = np.linspace(20, 3500, 74)
d_tr = np.histogram(D_train[variable], bins = 74, range = (20,3500))
d_te = np.histogram(D_test[variable], bins = 74, range = (20,3500))

DY = []; ST = []; DB = []; W = []; TT = []; DM = []
DY_w = []; ST_w = []; DB_w = []; W_w = []; TT_w = []; DM_w = []
for DSID, mll, w in zip(X_train['RunNumber'], X_train[variable], X_train['Weight']):
    if DSID in IDs["DY"]:
        DY.append(mll)
        DY_w.append(w)
    elif DSID in IDs['Single_top']:
        ST.append(mll)
        ST_w.append(w)
    elif DSID in IDs["Diboson"]:
        DB.append(mll)
        DB_w.append(w)
    elif DSID in IDs["W"]:
        W.append(mll)
        W_w.append(w)
    elif DSID in IDs["TTbar"]:
        TT.append(mll)
        TT_w.append(w)
    else:
        DM.append(mll)
        DM_w.append(w)

hist = [DM, TT, W, DB, ST, DY]
hist_w = [DM_w, TT_w, W_w, DB_w, ST_w, DY_w]
colors = ['pink', 'green', 'yellow', 'purple', 'blue', 'red']
labels = ['DM', 'TT', 'W', 'DB', 'ST', 'DY']

plt.figure(1)
plt.hist(hist ,bins=np.arange(20, 3500, 74), log=True, stacked=True, color=colors, label=labels)
plt.xlabel('$m_{ll}$ $[GeV]$'); plt.ylabel('Events'); plt.title('Invariant Mass unweighted training set'); plt.legend(); plt.ylim([1e-5,1e8])
plt.savefig(plot_dir+'mll_train_uw.pdf')
plt.show()

plt.figure(2)
plt.hist(hist, weights=hist_w ,bins=x_axis, log=True, stacked=True, color=colors, label=labels)
plt.scatter(x_axis, d_tr[0], color='black', label='Data')
plt.xlabel('$m_{ll}$ $[GeV]$'); plt.ylabel('Events'); plt.title('Invariant Mass weighted training set'); plt.legend(); plt.ylim([1e-5,1e8])
plt.savefig(plot_dir+'mll_train.pdf')
plt.show()


DY = []; ST = []; DB = []; W = []; TT = []; DM = []
DY_w = []; ST_w = []; DB_w = []; W_w = []; TT_w = []; DM_w = []
for DSID, mll, w in zip(X_test['RunNumber'], X_test[variable], X_test['Weight']):
    if DSID in IDs["DY"]:
        DY.append(mll)
        DY_w.append(w)
    elif DSID in IDs['Single_top']:
        ST.append(mll)
        ST_w.append(w)
    elif DSID in IDs["Diboson"]:
        DB.append(mll)
        DB_w.append(w)
    elif DSID in IDs["W"]:
        W.append(mll)
        W_w.append(w)
    elif DSID in IDs["TTbar"]:
        TT.append(mll)
        TT_w.append(w)
    else:
        DM.append(mll)
        DM_w.append(w)

hist = [DM, TT, W, DB, ST, DY]
hist_w = [DM_w, TT_w, W_w, DB_w, ST_w, DY_w]
colors = ['pink', 'green', 'yellow', 'purple', 'blue', 'red']
labels = ['DM', 'TT', 'W', 'DB', 'ST', 'DY']

plt.figure(3)
plt.hist(hist, bins=x_axis, log=True, stacked=True, color=colors, label=labels)
plt.xlabel('$m_{ll}$ $[GeV]$'); plt.ylabel('Events'); plt.title('Invariant Mass unweighted test set'); plt.legend(); plt.ylim([1e-5,1e8])
plt.savefig(plot_dir+'mll_test_uw.pdf')
plt.show()

plt.figure(4)
plt.hist(hist, weights=hist_w, bins=x_axis, log=True, stacked=True, color=colors, label=labels)
plt.scatter(x_axis, d_te[0], color='black', label='Data')
plt.xlabel('$m_{ll}$ $[GeV]$'); plt.ylabel('Events'); plt.title('Invariant Mass weighted test set'); plt.legend(); plt.ylim([1e-5,1e8])
plt.savefig(plot_dir+'mll_test.pdf')
plt.show()


dm_dict_file.close()
