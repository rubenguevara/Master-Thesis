import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
print('TensorFlow GPU version',tf.__version__)


t0 = time.time()
start = time.asctime(time.localtime())
print('Started', start)

save_dir = "../../../../storage/racarcam/"
filename = "FULL_Zp_50MET_no_pad.h5"

def NN_model(inputsize, n_layers, n_neuron, eta, lamda):
    model=tf.keras.Sequential()
    model.add(layers.BatchNormalization())
    for i in range(n_layers):                                                # Run loop to add hidden layers to the model
        if (i==0):                                                           # First layer requires input dimensions
            model.add(layers.Dense(n_neuron, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamda), input_dim=inputsize))
            
            
        else:                                                                # Subsequent layers are capable of automatic shape inferencing
            model.add(layers.Dense(n_neuron, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamda)))
    
    model.add(layers.Dense(1, activation='sigmoid'))                         # 1 output - signal or no signal
    adam=tf.optimizers.SGD(learning_rate=eta)
    
    model.compile(loss=tf.losses.BinaryCrossentropy(),
                optimizer=adam,
                metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    return model


df = pd.read_hdf(save_dir+filename, key='df_tot')

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                             # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')                                     # Bad variable
df_Weight = df_features.pop('Sample_Weight')

variables = ['n_bjetPt20', 'n_bjetPt30', 'n_bjetPt40', 'n_bjetPt50', 'n_bjetPt60', 'n_ljetPt20', 
            'n_ljetPt30', 'n_ljetPt40', 'n_ljetPt50', 'n_ljetPt60', 'jetEtaCentral', 'jetEtaForward', 
            'jet1Pt', 'jet1Eta', 'jet1Phi', 'jet2Pt', 'jet2Eta', 'jet2Phi']


df_features = df_features.drop(variables, axis=1)


df_labels = df_features.pop('Label')

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=test_size, random_state=42)

X_train_w = X_train.pop('Weight')
W_test = X_test.pop('Weight')


L_train = np.ones(len(Y_train))
sow_sig = sum(L_train[Y_train==1])
sow_bkg = sum(L_train[Y_train==0])

Balancer = np.ones(len(Y_train))
Balancer[Y_train==1] = sow_bkg/sow_sig

T_train = abs(X_train_w)*sum(X_train_w)/(sum(abs(X_train_w)))

W_train = T_train*Balancer
W_train = pd.DataFrame(W_train, columns=['Weight'])

model_dir = '../Models/NN/WGTS/'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass

plot_dir = '../../Plots/TESTING/NN/LVB/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass



network = NN_model(X_train.shape[1], 2, 100, 0.01, 1e-05)
print('Should start fitting...')
history = network.fit(X_train, Y_train, sample_weight = W_train, 
                    validation_data = (X_test, Y_test),    
            epochs = 200, batch_size = int(2**24), use_multiprocessing = True)

network.save(model_dir+'ATLAS_WWW')

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
plt.plot(history.history['auc'], label = 'Train')
plt.plot(history.history['val_auc'], label = 'Test')
plt.title('Area Under Curve')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plot_dir+'AUC.pdf')
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history.history['loss'], label = 'Train')
ax2.plot(history.history['val_loss'], label = 'Test')
fig.suptitle('Model Loss ')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()
fig.savefig(plot_dir+'Loss.pdf')
fig.show()


print('==='*20)
t = "{:.2f}".format(int( time.time()-t0 )/60.)
finish = time.asctime(time.localtime())
print('Finished', finish)
print('Total time:', t)
print('==='*20)