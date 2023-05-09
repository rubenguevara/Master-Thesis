import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)

print(tf.__version__)

save_dir = "../../../storage/racarcam/"
filename = "bkgs.h5"

model_dir = 'Models/NN/PART/WEIGHTED_TEST_Batch_3/'
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
todo = ['LV_HDS_low_mass', 'DH_HDS_low_mass', 'EFT_HDS_low_mass']
for file in todo:#os.listdir(save_dir+'DM_Parts/'):
    part = str(file).split('.')[0]
    if part in already_done: 
        print('Already did', part)
        continue  
    print('Doing', part)
    
    df_dm = pd.read_hdf(save_dir+'/DM_Parts/'+part+'.h5', key='df_tot')

    df = pd.concat([df_bkgs, df_dm]).sort_index()

    df_features = df.copy()
    df_EventID = df_features.pop('EventID')
    df_CrossSection = df_features.pop('CrossSection')
    df_Dileptons = df_features.pop('Dileptons')
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
    
    
    network = NN_model(X_train.shape[1], 1, 100, 0.1, 1e-3)
    history = network.fit(X_train, Y_train, sample_weight=W_train,
                validation_data = (X_test, Y_test),    
                epochs = 30, batch_size = int(2**24), use_multiprocessing = True)
                                    
    network.save(model_dir+part)
    
    plot_dir = 'Plots_NeuralNetwork/Parts/'+part+'/WEIGHTED_TEST/Batch_3/'

    try:
        os.makedirs(plot_dir)

    except FileExistsError:
        pass

    plt.figure(1)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Model accuracy')
    ax1.plot(history.history['binary_accuracy'], label = 'Train')
    plt.ylabel('Binary accuracy')
    plt.legend()
    ax2.plot(history.history['val_binary_accuracy'], label = 'Test')
    plt.ylabel('Binary accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(plot_dir+'Binary_accuracy.pdf')
    plt.show()

    plt.figure(2)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Model loss')
    ax1.plot(history.history['loss'], label = 'Train')
    plt.ylabel('Loss')
    plt.legend()
    ax2.plot(history.history['val_loss'], label = 'Test')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(plot_dir+'Loss.pdf')
    plt.show()

    plt.figure(3)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Area Under Curve ')
    ax1.plot(history.history['auc'], label = 'Train')
    plt.ylabel('AUC')
    plt.legend()
    ax2.plot(history.history['val_auc'], label = 'Test')
    plt.ylabel('AUC')
    plt.legend()
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(plot_dir+'AUC.pdf')
    plt.show()
    break