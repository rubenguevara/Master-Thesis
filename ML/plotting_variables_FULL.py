import os, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from EventIDs import IDs

model_dir = 'Models/NN/'
save_dir = "../../../storage/racarcam/"
filename = 'Full_DM_sig.h5'
datafile = 'data.h5'

df_data = pd.read_hdf(save_dir+datafile, key='df')
df = pd.read_hdf(save_dir+filename, key='df_tot')


"""
Choose what you want to plot!
"""
variable = 'mll'

plot_dir = 'Plots_NeuralNetwork/ALL/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')   # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')           # Bad variable


X_train, X_test = train_test_split(df_features, test_size=0.2, random_state=42)
D_train, D_test = train_test_split(df_data, test_size=0.2, random_state=42)

x_axis = np.linspace(20, 3500, 74)
d_tr = np.histogram(D_train[variable], bins = 74, range = (20, 3500))
d_te = np.histogram(D_test[variable], bins = 74, range = (20, 3500))

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

hist = [W, DB, TT, ST, DY]
hist_w = [W_w, DB_w, TT_w, ST_w, DY_w]
colors = ['#218C8D', '#6CCECB', '#F9E559', '#EF7126', '#8EDC9D']
labels = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan']

plt.figure(1, figsize=[15,10])
plt.hist(hist, weights=hist_w ,bins=x_axis, log=True, stacked=True, color=colors, label=labels)
plt.hist(DM, weights=DM_w ,bins=x_axis, log=True, color='#F42069', label='Signal')
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

hist = [W, DB, TT, ST, DY]
hist_w = [W_w, DB_w, TT_w, ST_w, DY_w]
colors = ['#218C8D', '#6CCECB', '#F9E559', '#EF7126', '#8EDC9D']
labels = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan']

plt.figure(2, figsize=[15,10])
plt.hist(hist, weights=hist_w, bins=x_axis, log=True, stacked=True, color=colors, label=labels)
plt.hist(DM, weights=DM_w ,bins=x_axis, log=True, color='#F42069', label='Signal')
plt.scatter(x_axis, d_te[0], color='black', label='Data')
plt.xlabel('$m_{ll}$ $[GeV]$'); plt.ylabel('Events'); plt.title('Invariant Mass weighted test set'); plt.legend(); plt.ylim([1e-5,1e8])
plt.savefig(plot_dir+'mll_test.pdf')
plt.show()

