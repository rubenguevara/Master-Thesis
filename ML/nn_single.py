import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)

print(tf.__version__)

save_dir = "../../../storage/racarcam/"
filename = "bkgs.h5"

model_dir = 'Models/NN/'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass

def NN_model(inputsize, n_layers, n_neuron, eta, lamda, norm):
    model=tf.keras.Sequential([norm])      
    
    for i in range(n_layers):                                                # Run loop to add hidden layers to the model
        if (i==0):                                                           # First layer requires input dimensions
            model.add(layers.Dense(n_neuron, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamda), input_dim=inputsize))
            
        else:                                                                # Subsequent layers are capable of automatic shape inferencing
            model.add(layers.Dense(n_neuron, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamda)))
    
    model.add(layers.Dense(1, activation='sigmoid'))                         # 1 output - signal or no signal
    sgd=tf.optimizers.SGD(learning_rate=eta)
    
    model.compile(loss=tf.losses.BinaryCrossentropy(),
                optimizer=sgd,
                metrics = [tf.keras.metrics.BinaryAccuracy()])
    return model

df_bkgs = pd.read_hdf(save_dir+filename, key='df_tot')

dm_dict_file = open('DM_DICT.json')
DM_DICT = json.load(dm_dict_file)
already_done = os.listdir('Plots_NeuralNetwork/DSID/')

for file in os.listdir(save_dir+'/DMS'):
    dsid = file.split('.')[0]
    if dsid in already_done: 
        print('Already did DSID', dsid)
        continue
    print('Doing DSID', dsid)
    
    df_dm = pd.read_hdf(save_dir+'DMS/'+dsid+'.h5', key='df_tot')

    df = pd.concat([df_bkgs, df_dm]).sort_index()

    df_features = df.copy()
    df_EventID = df_features.pop('EventID')
    df_CrossSection = df_features.pop('CrossSection')
    df_RunNumber = df_features.pop('RunNumber')
    df_RunPeriod = df_features.pop('RunPeriod')
    
    df_labels = df_features.pop('Label')
    
    X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
    X_train_wgt = X_train.pop('Weight')
    X_test_wgt = X_test.pop('Weight')
    
    normalize = layers.experimental.preprocessing.Normalization()
    # normalize = layers.Normalization()
    normalize.adapt(X_train)

    network = NN_model(X_train.shape[1], 3, 10, 0.1, 1e-3, normalize)
    network.fit(X_train, Y_train, epochs=10, batch_size=100, sample_weight=X_train_wgt) # Correct way to use weights?
    
    network.save(model_dir+'NEW_WEIGHTED_'+dsid)
    
    """""""""""""""
    Plotting
    """""""""""""""
    
    plot_dir = 'Plots_NeuralNetwork/DSID/NEW_WEIGHTED_'+dsid+'/'

    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass
    
    network_pred_label = network.predict(X_test).ravel()
    test = Y_test
    pred = network_pred_label

    dsid_title = DM_DICT[dsid]
    fpr, tpr, thresholds = roc_curve(test, pred, pos_label=1, )
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
    n, bins, patches = plt.hist(pred[test==0], 200, facecolor='blue', alpha=0.2,label="Background")
    n, bins, patches = plt.hist(pred[test==1], 200, facecolor='red' , alpha=0.2, label="Signal")
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