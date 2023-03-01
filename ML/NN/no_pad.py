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

save_dir = "/storage/racarcam/"
filename = "FULL_Zp_50MET.h5"

df = pd.read_hdf(save_dir+filename, key='df_tot')                           # Padding missing values on jet- eta & phi

pad_variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50']

bad_variables = ['jet1Pt', 'jet1Eta', 'jet1Phi', 'jet2Pt', 'jet2Eta', 'jet2Phi', 'jet3Pt', 'jet3Eta', 'jet3Phi', 'mjj', 'dPhiCloseMet', 'dPhiLeps']


df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_features = df_features.drop(bad_variables, axis=1)

# New padding variables
""" Comment out next line to include the new variables!"""
# df_features = df_features.drop(pad_variables, axis=1)

df_labels = df_features.pop('Label')
df_Weights = df_features.pop('Weight')

print(df_features)
test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=test_size, random_state=42)


N_sig_train = sum(Y_train)
N_bkg_train = len(Y_train) - N_sig_train
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
history = model.fit(X_train, Y_train, sample_weight = W_train, 
                    validation_data = (X_test, Y_test),
                    epochs = 50, batch_size = int(2**23), use_multiprocessing = True)

model.save('../Models/PAD/new_variables')
# model.save('../Models/PAD/no_pad')


plot_dir = '../../Plots/NeuralNetwork/FULL/padding/new_variables/'
# plot_dir = '../../Plots/NeuralNetwork/FULL/padding/no_pad/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass


plt.figure(1)
plt.suptitle('Model accuracy')
plt.plot(history.history['binary_accuracy'], label = 'Train')
plt.ylabel('Binary accuracy')
plt.legend()
plt.plot(history.history['val_binary_accuracy'], label = 'Test')
plt.ylabel('Binary accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plot_dir+'Binary_accuracy.pdf')
plt.show()

plt.figure(2)
plt.suptitle('Area Under Curve ')
plt.plot(history.history['auc'], label = 'Train')
plt.ylabel('AUC')
plt.legend()
plt.plot(history.history['val_auc'], label = 'Test')
plt.ylabel('AUC')
plt.legend()
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plot_dir+'AUC.pdf')
plt.show()

plt.figure(3)
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Model loss')
ax1.plot(history.history['loss'], label = 'Train')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(history.history['val_loss'], label = 'Test')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()
plt.savefig(plot_dir+'Loss.pdf')
plt.show()
