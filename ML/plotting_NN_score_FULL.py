import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)

print(tf.__version__)

model_dir = 'Models/NN/'
save_dir = "../../../storage/racarcam/"
# filename = 'Full_DM_sig.h5'
filename = 'Find_Diboson.h5'

df = pd.read_hdf(save_dir+filename, key='df_tot')

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')   # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')           # Bad variable

df_labels = df_features.pop('Label')

X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
X_train_w = X_train.pop('Weight')
X_test_w = X_test.pop('Weight')

# model_type = 'FULL_UNWEIGHTED'
# model_type = 'FULL_WEIGHTED'
# model_type = 'Diboson_UNWEIGHTED'
model_type = 'Diboson_WEIGHTED_NEW'
network = tf.keras.models.load_model(model_dir+model_type)  
network_pred_label = network.predict(X_test).ravel()

# plot_dir = 'Plots_NeuralNetwork/ALL/UNWEIGHTED/'
# plot_dir = 'Plots_NeuralNetwork/ALL/WEIGHTED/'
# plot_dir = 'Plots_NeuralNetwork/Diboson/UNWEIGHTED/'
plot_dir = 'Plots_NeuralNetwork/Diboson/WEIGHTED_NEW/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

network_pred_label = network.predict(X_test).ravel()
test = Y_test
pred = network_pred_label


fpr, tpr, thresholds = roc_curve(test, pred, pos_label=1)
roc_auc = auc(fpr,tpr)
plt.figure(1)
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Unweighted ROC on full DM dataset')
plt.legend(loc="lower right")
plt.savefig(plot_dir+'ROC_uw.pdf')
plt.show()

weight_test = np.ones(len(Y_test))
weight_test[Y_test==0] = np.sum(weight_test[Y_test==1])/np.sum(weight_test[Y_test==0])
unique, counts = np.unique(pred, return_counts=True)

fpr, tpr, thresholds = roc_curve(test, pred, sample_weight = weight_test, pos_label=1)
roc_auc = auc(fpr,tpr)

plt.figure(2)
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Estimated weight ROC on full DM dataset')
plt.legend(loc="lower right")
plt.savefig(plot_dir+'ROC_we.pdf')
plt.show()

# fpr, tpr, thresholds = roc_curve(test, pred, sample_weight = X_test_w, pos_label=1)
# roc_auc = auc(fpr,tpr)

# plt.figure(2)
# plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([-0.01, 1.02])
# plt.ylim([-0.01, 1.02])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Real weight ROC on full DM dataset')
# plt.legend(loc="lower right")
# plt.savefig(plot_dir+'ROC_wr.pdf')
# plt.show()


plt.figure(3, figsize=[10,6])
n, bins, patches = plt.hist(pred[test==0], bins = 100, facecolor='blue', alpha=0.2, label="Background")
n, bins, patches = plt.hist(pred[test==1], bins = 100, facecolor='red' , alpha=0.2, label="Signal")
plt.xlabel('TF output')
plt.xlim([0,1])
plt.ylabel('Events')
plt.yscale('log')
plt.title('Model output, Full DM dataset, unweighted validation data')
plt.grid(True)
plt.legend()
plt.savefig(plot_dir+'VAL_uw.pdf')
plt.show()


plt.figure(4, figsize=[10,6])
n, bins, patches = plt.hist(pred[test==0], weights = X_test_w[test==0], bins = 100, facecolor='blue', alpha=0.2, label="Background")
n, bins, patches = plt.hist(pred[test==1], weights = X_test_w[test==1], bins = 100, facecolor='red' , alpha=0.2, label="Signal")
plt.xlabel('TF output')
plt.xlim([0,1])
plt.ylabel('Events')
plt.yscale('log')
plt.title('Model output, Full DM dataset, weighted validation data')
plt.grid(True)
plt.legend()
plt.savefig(plot_dir+'VAL_w.pdf')
plt.show()