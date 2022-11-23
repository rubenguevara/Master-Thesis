import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from EventIDs import IDs

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)

print(tf.__version__)

channels = ["DY", 'Single_top', "Diboson", "W", "TTbar"]

model_dir = 'Models/NN/'
save_dir = "../../../storage/racarcam/"
filename = "FULL_DM_50MET.h5"

df = pd.read_hdf(save_dir+filename, key='df_tot')

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')   # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')           # Bad variable

df_labels = df_features.pop('Label')

X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
X_train_w = X_train.pop('Weight')
X_test_w = X_test.pop('Weight')

X_train_id = X_train.pop('RunNumber')
X_test_id = X_test.pop('RunNumber')

model_type = 'FULL_WEIGHTED'

network = tf.keras.models.load_model(model_dir+model_type)  
network_pred_label = network.predict(X_test, batch_size = 4096, use_multiprocessing = True, verbose = 1).ravel()

plot_dir = 'Plots_NeuralNetwork/FULL/WEIGHTED/'
try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

test = Y_test
pred = network_pred_label

pred_ch = {}
for ch in channels:
    pred_ch[ch] = pred[np.isin(X_test_id, IDs[ch])] 

plt.figure(1, figsize=[10, 6])
for ch in channels:
    n, bins, patches = plt.hist(pred_ch[ch], weights = X_test_w[np.isin(X_test_id, IDs[ch])]*1.8, bins = 100, histtype='step', label=ch)
n, bins, patches = plt.hist(pred[test==1], weights = X_test_w[test==1]*1.8, bins = 100, facecolor='red' , alpha=0.2, label="Signal")
plt.xlabel('TF output')
plt.xlim([0,1])
plt.ylabel('Events')
plt.yscale('log')
plt.title('Model output, Full DM dataset, validation data')
plt.grid(True)
plt.legend(loc='lower center')
plt.savefig(plot_dir+'VAL_CHNL.pdf')
plt.show()


ALL_DSIDS = X_test_id[np.isin(X_test_id, IDs['Diboson'])].unique()


plt.figure(2, figsize=[10, 6])
colormap = plt.cm.gist_rainbow
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(ALL_DSIDS)))))
for dsid in ALL_DSIDS:
    n, bins, patches = plt.hist(pred[X_test_id==dsid],  weights = X_test_w[X_test_id == dsid]*1.8, bins = 100, histtype='step', alpha = 0.5, label=str(dsid))
n, bins, patches = plt.hist(pred[test==1], weights = X_test_w[test==1]*1.8, bins = 100, facecolor='red' , alpha=0.2, label="Signal")
plt.xlabel('TF output')
plt.xlim([0,1])
plt.ylabel('Events')
plt.yscale('log')
plt.title('Model output, Full DM dataset, validation data')
plt.grid(True)
plt.legend(ncol=3, loc='lower center',
        bbox_to_anchor=[0.5, -0.05], 
        columnspacing=1.0, labelspacing=0.0,
        handletextpad=0.0, handlelength=1.5,
        fancybox=True, shadow=True)
plt.savefig(plot_dir+'VAL_DSID_Diboson.pdf')
plt.show()