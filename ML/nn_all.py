import os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from packaging.version import parse
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
vrs = tf.__version__
print(vrs)

parser = argparse.ArgumentParser()
parser.add_argument('--wgt', type=int, default=1, help="Whether to do unweighted training")
args = parser.parse_args()
wgt = args.wgt                                                              # Default is weighted training

if (parse(vrs) != parse('2.5.0')) and (wgt == 1):
    print('For weighted training it should be 2.5.0 !!!') 
    exit()    

save_dir = "../../../storage/racarcam/"
# filename = 'Full_DM_sig.h5'
filename = 'Find_Diboson.h5'

df = pd.read_hdf(save_dir+filename, key='df_tot')

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                             # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')                                     # Bad variable

df_labels = df_features.pop('Label')

X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
X_train_w = X_train.pop('Weight')
X_test_w = X_test.pop('Weight')

W_test = np.ones(len(Y_test))
W_test[Y_test==0] = np.sum(W_test[Y_test==1])/np.sum(W_test[Y_test==0])
W_test = pd.DataFrame(W_test, columns=['Weight'])                            # Has to be a pandas dataframe else it won't work

W_train = np.ones(len(Y_train))
W_train[Y_train==0] = np.sum(W_train[Y_train==1])/np.sum(W_train[Y_train==0])
W_train = pd.DataFrame(W_train, columns=['Weight'])


if wgt == 0:
    print('Doing unweighted training')
else:
    print('Doing weighted training')
    

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

network = NN_model(X_train.shape[1], 3, 10, 0.1, 1e-3)
print('Starting fitting')
if wgt == 0:
    history = network.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs=10, batch_size=8192)

else:                                                                        # Should work??
    history = network.fit(X_train, Y_train, sample_weight = X_train_w,       # OBS ONLY WORKS FOR TENSORFLOW VERSION 2.5.0 
                    validation_data = (X_test, Y_test),            
                    epochs = 10, batch_size = 8192, use_multiprocessing = True)


# else:                                                                        # Shady weighting
#     history = network.fit(X_train, Y_train, sample_weight = W_train, 
#                     validation_data = (X_test, Y_test),              
#                     epochs = 10, batch_size = 8192)

model_dir = 'Models/NN/'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass

if wgt == 0:
    network.save(model_dir+'Diboson_UNWEIGHTED')
else:
    network.save(model_dir+'Diboson_WEIGHTED')

if wgt == 0:
    plot_dir = 'Plots_NeuralNetwork/Diboson/UNWEIGHTED/'
else:
    plot_dir = 'Plots_NeuralNetwork/Diboson/WEIGHTED/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

plt.figure(1)
plt.plot(history.history['binary_accuracy'], label = 'Train')
plt.plot(history.history['val_binary_accuracy'], label = 'Test')
plt.title('Model accuracy')
plt.ylabel('Binary accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plot_dir+'Binary_accuracy.pdf')
plt.show()

plt.figure(2)
plt.plot(history.history['loss'], label = 'Train')
plt.plot(history.history['val_loss'], label = 'Test')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plot_dir+'Loss.pdf')
plt.show()

plt.figure(3)
plt.plot(history.history['auc'], label = 'Train')
plt.plot(history.history['val_auc'], label = 'Test')
plt.title('Area Under Curve ')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plot_dir+'AUC.pdf')
plt.show()
