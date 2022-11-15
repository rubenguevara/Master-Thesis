import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

ML = 'BDT'

if ML == 'NN':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(False)

    print(tf.__version__)

elif ML == 'BDT':
    print(xgb.__version__)
    
save_dir = "../../../storage/racarcam/"
filename = "FULL_DM_50MET.h5"

chnl = 'uu'
model = 'LV'

print('Doing', model, chnl, 'on', ML)

df = pd.read_hdf(save_dir+filename, key='df_tot')
df_chnl = df.loc[df['Dileptons'] == chnl]

df_features = df_chnl.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')   # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')           # Bad variable

df_labels = df_features.pop('Label')

X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
X_train = X_train.pop('RunNumber')

dsid_LV_HDS_MZ_130 = [514562, 514563] 
dsid_DH_HDS_MZ_130 = [514560, 514561] 
dsid_EFT_HDS_MZ_130 = [514564, 514565] 

X_test_LV_0 = X_test.loc[X_test['RunNumber'] == dsid_LV_HDS_MZ_130[0]]
X_test_LV_1 = X_test.loc[X_test['RunNumber'] == dsid_LV_HDS_MZ_130[1]]
X_test_DH_0 = X_test.loc[X_test['RunNumber'] == dsid_DH_HDS_MZ_130[0]]
X_test_DH_1 = X_test.loc[X_test['RunNumber'] == dsid_DH_HDS_MZ_130[1]]
X_test_EFT_0 = X_test.loc[X_test['RunNumber'] == dsid_EFT_HDS_MZ_130[0]]
X_test_EFT_1 = X_test.loc[X_test['RunNumber'] == dsid_EFT_HDS_MZ_130[1]] 

Y_test_LV_0 = Y_test.loc[X_test['RunNumber'] == dsid_LV_HDS_MZ_130[0]]
Y_test_LV_1 = Y_test.loc[X_test['RunNumber'] == dsid_LV_HDS_MZ_130[1]]
Y_test_DH_0 = Y_test.loc[X_test['RunNumber'] == dsid_DH_HDS_MZ_130[0]]
Y_test_DH_1 = Y_test.loc[X_test['RunNumber'] == dsid_DH_HDS_MZ_130[1]]
Y_test_EFT_0 = Y_test.loc[X_test['RunNumber'] == dsid_EFT_HDS_MZ_130[0]]
Y_test_EFT_1 = Y_test.loc[X_test['RunNumber'] == dsid_EFT_HDS_MZ_130[1]] 

X_test_bkg = X_test.loc[Y_test == 0] 
Y_test_bkg = Y_test.loc[Y_test == 0]

X_test_LV = pd.concat([X_test_bkg, X_test_LV_0, X_test_LV_1])
X_test_DH = pd.concat([X_test_bkg, X_test_DH_0, X_test_DH_1])
X_test_EFT = pd.concat([X_test_bkg, X_test_EFT_0, X_test_EFT_1])

Y_test_LV = pd.concat([Y_test_bkg, Y_test_LV_0, Y_test_LV_1])
Y_test_DH = pd.concat([Y_test_bkg, Y_test_DH_0, Y_test_DH_1])
Y_test_EFT = pd.concat([Y_test_bkg, Y_test_EFT_0, Y_test_EFT_1])

X_test_LV_rn = X_test_LV.pop('RunNumber')
X_test_DH_rn = X_test_DH.pop('RunNumber')
X_test_EFT_rn = X_test_EFT.pop('RunNumber')

X_test_LV_w = X_test_LV.pop('Weight')
X_test_DH_w = X_test_DH.pop('Weight')
X_test_EFT_w = X_test_EFT.pop('Weight')

if model == 'LV':
    X_test = X_test_LV
    W_test = X_test_LV_w
    Y_test = Y_test_LV

elif model == 'DH':
    X_test = X_test_DH
    W_test = X_test_DH_w
    Y_test = Y_test_DH

elif model == 'EFT':
    X_test = X_test_EFT
    W_test = X_test_EFT_w
    Y_test = Y_test_EFT

else:
    print("Model "+model+" not indluded!")
    exit()

if ML =='NN':
    plot_dir = 'Plots_NeuralNetwork/FULL/SIGNIFICANCE/'+model+'/'
    model_dir = 'Models/NN/'
    model_type = 'FULL_WEIGHTED'    
    network = tf.keras.models.load_model(model_dir+model_type)  
    network_pred_label = network.predict(X_test, batch_size = 2048, use_multiprocessing = True, verbose = 1).ravel()

elif ML == 'BDT':
    plot_dir = 'Plots_XGBoost/FULL/SIGNIFICANCE/'+model+'/'
    model_dir = 'Models/XGB/'
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(model_dir+'FULL.txt')

    # y_pred = xgbclassifier.predict(X_test)
    y_pred_prob = model_xgb.predict_proba(X_test)
    network_pred_label = y_pred_prob[:,1]


try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

test = Y_test
pred = network_pred_label


fpr, tpr, thresholds = roc_curve(test, pred, pos_label=1)
roc_auc = auc(fpr,tpr)
lw = 2

weight_test = np.ones(len(test))
weight_test[test==0] = np.sum(weight_test[test==1])/np.sum(weight_test[test==0])

fpr, tpr, thresholds = roc_curve(test, pred, sample_weight = weight_test, pos_label=1)
roc_auc = auc(fpr,tpr)

plt.figure(1)
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC on FULL DM trained network on "+model+" HDS $m_{Z'}$=130 GeV "+chnl+" model")
plt.legend(loc="lower right")
plt.savefig(plot_dir+'ROC_'+chnl+'.pdf')
plt.show()

plt.figure(2, figsize=[10,6])
bkg_pred, bins, patches = plt.hist(pred[test==0], weights = W_test[test==0], bins = 100, facecolor='blue', alpha=0.2, label="Background")
sig_pred, bins, patches = plt.hist(pred[test==1], weights = W_test[test==1], bins = 100, facecolor='red' , alpha=0.2, label="Signal")
plt.xlabel('TF output')
plt.xlim([0,1])
plt.ylabel('Events')
plt.yscale('log')
plt.title("Model output, Full DM trained network on "+model+" HDS $m_{Z'}$=130 GeV "+chnl+" model, validation data")
plt.grid(True)
plt.legend()
plt.savefig(plot_dir+'VAL_'+chnl+'.pdf')
plt.show()

def low_stat_Z(sig, bkg):
    Z = np.sqrt(2*( (sig + bkg)*np.log(1 + sig/bkg) - sig ))
    return Z

Y_axis = [low_stat_Z(sum(sig_pred[50:]), sum(bkg_pred[50:])), 
            low_stat_Z(sum(sig_pred[60:]), sum(bkg_pred[60:])), 
            low_stat_Z(sum(sig_pred[70:]), sum(bkg_pred[70:])),
            low_stat_Z(sum(sig_pred[80:]), sum(bkg_pred[80:])), 
            low_stat_Z(sum(sig_pred[90:]), sum(bkg_pred[90:])), 
            low_stat_Z(sum(sig_pred[99:]), sum(bkg_pred[99:]))]

X_axis = [50, 60, 70, 80, 90, 99]

plt.figure(3, figsize=[10,6])
plt.plot(X_axis, Y_axis, linestyle='--')
plt.scatter(X_axis, Y_axis, label = '$\sigma$ cut')
plt.xlim([0,100])
plt.grid(True)
plt.legend()
plt.ylabel('Expected significance [$\sigma$]')
plt.title("Significance, Full DM trained network on "+model+" HDS $m_{Z'}$=130 GeV "+chnl+" model")
plt.xlabel('TF output bin')
plt.savefig(plot_dir+'EXP_SIG_'+chnl+'.pdf')
plt.show()
