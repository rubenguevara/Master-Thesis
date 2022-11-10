import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)

print(tf.__version__)

save_dir = "../../../storage/racarcam/"
filename = "bkgs.h5"

model_dir = 'Models/NN/DSID/WEIGHTED'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass

def NN_model(inputsize, n_layers, n_neuron, eta, lamda):
    model=tf.keras.Sequential()
    model.add(layers.BatchNormalization())
    for i in range(n_layers):                                                # Run loop to add hidden layers to the model
        if (i==0):                                                           # First layer requires input dimensions
            model.add(layers.Dense(n_neuron, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamda), input_dim=inputsize))
            
        else:                                                                # Subsequent layers are capable of automatic shape inferencing
            model.add(layers.Dense(n_neuron, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamda)))
    
    model.add(layers.Dense(1, activation='sigmoid'))                         # 1 output - signal or no signal
    adam=tf.optimizers.Adam(learning_rate=eta)
    
    model.compile(loss=tf.losses.BinaryCrossentropy(),
                optimizer=adam,
                metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    return model

df_bkgs = pd.read_hdf(save_dir+filename, key='df_tot')

already_done = os.listdir(model_dir)

for file in os.listdir(save_dir+'/DMS'):
    dsid = file.split('.')[0]
    if dsid in already_done: 
        print('Already did DSID', dsid)
        continue  
    dsid = '514562'
    # dsid = '514630'  
    print('Doing DSID', dsid)
    
    df_dm = pd.read_hdf(save_dir+'DMS/'+dsid+'.h5', key='df_tot')

    df = pd.concat([df_bkgs, df_dm]).sort_index()

    df_features = df.copy()
    df_EventID = df_features.pop('EventID')
    df_CrossSection = df_features.pop('CrossSection')
    df_RunNumber = df_features.pop('RunNumber')
    df_RunPeriod = df_features.pop('RunPeriod')
    df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                        # "Bad" variable
    df_dPhiLeps = df_features.pop('dPhiLeps')                                # "Bad" variable
    
    df_labels = df_features.pop('Label')
    
    X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
    X_train_wgt = X_train.pop('Weight')
    X_test_wgt = X_test.pop('Weight')
    
    W_train = np.ones(len(Y_train))
    W_train[Y_train==0] = sum(W_train[Y_train==1])/sum(W_train[Y_train==0])
    W_train = pd.DataFrame(W_train, columns=['Weight'])                      # Has to be a pandas DataFrame or it crashes
    
    network = NN_model(X_train.shape[1], 3, 10, 0.1, 1e-3)
    network.fit(X_train, Y_train, sample_weight=W_train,
                epochs = 10, batch_size = 8192, use_multiprocessing = True)
    
    network.save(model_dir+dsid)
    break
