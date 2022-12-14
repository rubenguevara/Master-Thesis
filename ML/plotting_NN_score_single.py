import os, json
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
filename = "bkgs.h5"
df_bkgs = pd.read_hdf(save_dir+filename, key='df_tot')

DSIDS = os.listdir('Models/NN/DSID/WEIGHTED/')
dm_dict_file = open('DM_DICT.json')
DM_DICT = json.load(dm_dict_file)

for dsid in DSIDS:
    plot_dir = 'Plots_NeuralNetwork/DSID/'+dsid+'/WEIGHTED/'

    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass

    df_dm = pd.read_hdf(save_dir+'DMS/'+dsid+'.h5', key='df_tot')

    df = pd.concat([df_bkgs, df_dm]).sort_index()

    df_features = df.copy()
    df_EventID = df_features.pop('EventID')
    df_CrossSection = df_features.pop('CrossSection')
    df_RunNumber = df_features.pop('RunNumber')
    df_RunPeriod = df_features.pop('RunPeriod')
    df_dPhiCloseMet = df_features.pop('dPhiCloseMet')   # Bad variable
    df_dPhiLeps = df_features.pop('dPhiLeps')           # Bad variable
    
    df_labels = df_features.pop('Label')
    
    X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
    X_train_wgt = X_train.pop('Weight')
    X_test_wgt = X_test.pop('Weight')
    
    network = tf.keras.models.load_model(model_dir+dsid)  
    network_pred_label = network.predict(X_test, batch_size = 4096, use_multiprocessing = True, verbose = 1).ravel()
    test = Y_test
    pred = network_pred_label
    
    weight_test = np.ones(len(Y_test))
    weight_test[Y_test==0] = np.sum(weight_test[Y_test==1])/np.sum(weight_test[Y_test==0])
    
    dsid_title = DM_DICT[dsid]
    fpr, tpr, thresholds = roc_curve(test, pred, sample_weight = weight_test, pos_label=1)
    roc_auc = auc(fpr,tpr)
    plt.figure(1)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.02])
    plt.ylim([-0.01, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for model on '+dsid_title+' dataset')
    plt.legend(loc="lower right")
    plt.savefig(plot_dir+'ROC.pdf')
    plt.show()


    plt.figure(2, figsize=[10,6])
    n, bins, patches = plt.hist(pred[test==0], weights = X_test_wgt[Y_test==0], bins = 100, facecolor='blue', alpha=0.2, label="Background")
    n, bins, patches = plt.hist(pred[test==1], weights = X_test_wgt[Y_test==1], bins = 100, facecolor='red' , alpha=0.2, label="Signal")
    plt.xlabel('TF output')
    plt.xlim([0,1])
    plt.ylabel('Events')
    plt.title('Model output, '+dsid_title+' dataset, validation data')
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.savefig(plot_dir+'VAL.pdf')
    plt.show()
    break

dm_dict_file.close()