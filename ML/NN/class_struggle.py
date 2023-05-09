import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
print('TensorFlow GPU version',tf.__version__)


def low_stat_Z(sig, bkg):
    Z = np.sqrt(2*( (sig + bkg)*np.log(1 + sig/bkg) - sig ))
    return Z

def stat_unc(prediction, bins, weights):
    binning = np.linspace(0,1,bins+1)
    histo_bins = np.digitize(prediction, binning)
    stat_unc_array = []
    for i in range(1,len(binning)):
        bin_wgt = weights[np.where(histo_bins==i)[0]]
        sow_bin = np.linalg.norm(bin_wgt,2)
        stat_unc_array.append(sow_bin)
    return np.asarray(stat_unc_array)


save_dir = "../../../../storage/racarcam/"
filename = "FULL_DM_50MET.h5"


df = pd.read_hdf(save_dir+filename, key='df_tot') 

model_dir = '../Models/NN/'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                             # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')                                     # Bad variable

df_labels = df_features.pop('Label')

#No padding
df_jet1Pt = df_features.pop('jet1Pt') 
df_jet2Pt = df_features.pop('jet2Pt')
df_jet1Phi = df_features.pop('jet1Phi')
df_jet2Phi = df_features.pop('jet2Phi')
df_jet1Eta = df_features.pop('jet1Eta')
df_jet2Eta = df_features.pop('jet2Eta')

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size = test_size, random_state = 42)
X_train_w = np.asarray(X_train.pop('Weight'))
W_test = X_test.pop('Weight')

# df_mean = pd.read_pickle("../Data/df_values/mean_df.pkl")
# df_var = pd.read_pickle("../Data/df_values/var_df.pkl")

# X_train = (X_train - df_mean)/np.sqrt(df_var)
# X_test = (X_test - df_mean)/np.sqrt(df_var)

# scaler = 1/test_size

# # N_sig_train = sum(X_train_w[Y_train==1])
# # N_bkg_train = sum(X_train_w[Y_train==0])
# # ratio = N_sig_train/N_bkg_train

# # print(N_bkg_train, N_sig_train, ratio)

# N_sig_train = sum(Y_train)
# N_bkg_train = len(Y_train) - N_sig_train
# # ratio = N_sig_train/N_bkg_train
# ratio = N_sig_train/N_bkg_train

# print(N_bkg_train, N_sig_train, ratio)

# Balance_train = np.ones(len(Y_train))
# Balance_train[Y_train==0] = ratio
# # print(X_train_w)
# # print(X_train_w[Y_train==1])
# # X_train_w[Y_train==1] = X_train_w[Y_train==1]*ratio
# # print(X_train_w)
# # print(X_train_w[Y_train==1])

# # class_struggle = {0: 1.0, 1: ratio}

# W_train = Balance_train # X_train_w
# W_train = pd.DataFrame(W_train, columns=['Weight'])
# print(W_train)

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

network=NN_model(X_train.shape[1], 2, 100, 0.01, 1e-05)
print('Should start fitting...')
network.fit(X_train, Y_train,# sample_weight = W_train, 
            epochs = 50, batch_size = int(2**24), use_multiprocessing = True)
# network.save('../Models/NN/CLASS_STRUGGLE_WGT_sig_up')
# network.save('../Models/NN/CLASS_STRUGGLE_WGT_bkg_down')
# network.save('../Models/NN/CLASS_STRUGGLE_bkg_down')
network.save('../Models/NN/NO_NOTHING')
# network.save('../Models/NN/Purely_balancing_bkg_down_evnts')
# network.save('../Models/NN/Purely_weightings')