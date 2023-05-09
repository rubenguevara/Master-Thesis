import os, gc, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
print(tf.__version__)

t0 = time.time()
start = time.asctime(time.localtime())
print('Started', start)

save_dir = "../../../../storage/racarcam/"
filename = "FULL_DM_50MET.h5"

df = pd.read_hdf(save_dir+filename, key='df_tot')                           # Padding missing values on jet- eta & phi

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                             # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')                                     # Bad variable

# No padding
df_jet1Phi = df_features.pop('jet1Phi')
df_jet2Phi = df_features.pop('jet2Phi')
df_jet1Eta = df_features.pop('jet1Eta')
df_jet2PEta = df_features.pop('jet2Eta')
df_jet1Pt = df_features.pop('jet1Pt')
df_jet2Pt = df_features.pop('jet2Pt')

df_labels = df_features.pop('Label')
df_Weights = df_features.pop('Weight')


# df_features.mean().to_pickle("../Data/normie/mean_df.pkl")
# df_features.var().to_pickle("../Data/normie/var_df.pkl")
# df_features.min().to_pickle("../Data/normie/min_df.pkl")
# df_features.max().to_pickle("../Data/normie/max_df.pkl")

# normalized_df = (df_features-df_features.mean())/np.sqrt(df_features.var())
# normalized_df = (df_features-df_features.min())/(df_features.max()-df_features.min())
normalized_df = df_features
normalized_df['Weight'] = df_Weights

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(normalized_df, df_labels, test_size=test_size, random_state=42)
X_train_w = np.asarray(X_train.pop('Weight'))
W_test = X_test.pop('Weight')

N_sig_train = sum(Y_train)
N_bkg_train = len(Y_train) - N_sig_train
# ratio = N_sig_train/N_bkg_train
ratio = N_sig_train/N_bkg_train

print(N_bkg_train, N_sig_train, ratio)

Balance_train = np.ones(len(Y_train))
Balance_train[Y_train==0] = ratio
W_train = pd.DataFrame(Balance_train, columns=['Weight'])

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

model=NN_model(X_train.shape[1], 2, 100, 0.01, 1e-05)
model.fit(X_train, Y_train, sample_weight = W_train, epochs = 50, batch_size = int(2**24), use_multiprocessing = True)
# model.save('../Models/NORMIE/Z_score')
# model.save('../Models/NORMIE/minmax')
# model.save('../Models/NORMIE/NoNorm')
model.save('../Models/NORMIE/BacthNorm')
# model.save('../Models/NORMIE/LayerNorm')