import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
print(tf.__version__)

save_dir = "/storage/racarcam/"
bkg_file = save_dir+'bkgs.h5'
sig_file = save_dir+'/DM_Models/DM_Zp_dh.h5'

df_bkg = pd.read_hdf(bkg_file, key='df_tot')
df_sig = pd.read_hdf(sig_file, key='df_tot')
df = pd.concat([df_bkg, df_sig])

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                             # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')                                     # Bad variable

#No padding
df_jet1Pt = df_features.pop('jet1Pt') 
df_jet2Pt = df_features.pop('jet2Pt')
df_jet1Phi = df_features.pop('jet1Phi')
df_jet2Phi = df_features.pop('jet2Phi')
df_jet1Eta = df_features.pop('jet1Eta')
df_jet2Eta = df_features.pop('jet2Eta')



# Signal region search

df_features = df_features.loc[df_features['mll'] > 120]                     

# met_reg = '50-100'
# met_reg = '100-150'
met_reg = '150'

if met_reg == '50-100':
    df_features = df_features.loc[df_features['met'] < 100]                     

elif met_reg == '100-150':
    df_features = df_features.loc[df_features['met'] > 100]                     
    df_features = df_features.loc[df_features['met'] < 150]      
    
elif met_reg == '150':
    df_features = df_features.loc[df_features['met'] > 150]                      

df_labels = df_features.pop('Label')

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=test_size, random_state=42)
X_train_w = X_train.pop('Weight')
W_test = X_test.pop('Weight')

print('Doing SR with mll > 120 and met',met_reg)

W_train = np.ones(len(Y_train))
W_train[Y_train==0] = sum(W_train[Y_train==1])/sum(W_train[Y_train==0])
W_train = pd.DataFrame(W_train, columns=['Weight'])                          

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


model=NN_model(X_train.shape[1], 2, 100, 0.01, 1e-5)
model.fit(X_train, Y_train, sample_weight = W_train, epochs=50, batch_size = int(2**24), verbose = 1, use_multiprocessing = True)
model.save('../../Models/NN/Zp_DH_mll_120_met_'+met_reg)