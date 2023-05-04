import os, gc, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
print('TensorFlow v.', tf.__version__)

t0 = time.time()
start = time.asctime(time.localtime())
print('Started', start)

save_dir = "/storage/racarcam/"
dm_model = 'DH_HDS'

bkg_file = save_dir+'bkgs_final.h5'
sig_file = save_dir+'/DM_Models/DM_Zp_'+dm_model.lower()+'.h5'

df_bkg = pd.read_hdf(bkg_file, key='df_tot')
df_sig = pd.read_hdf(sig_file, key='df_tot')

df = pd.concat([df_bkg, df_sig])

df_features = df.copy()


extra_variables = ['jetEtaForward50', 'dPhiCloseMet', 'dPhiLeps', 'jet1Phi', 'jet2Phi', 'jet3Phi', 'mjj', 'jet1Eta', 'jet2Eta', 'jet3Eta', 'jet1Pt', 'jet2Pt', 'jet3Pt'
                    ,'n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral']


df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_features = df_features.drop(extra_variables, axis=1)

df_labels = df_features.pop('Label')

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=test_size, random_state=42)
X_train_w = X_train.pop('Weight')
W_test = X_test.pop('Weight')
btr_wgt = np.ones(len(X_train_w))
btr_wgt[Y_train == 0] = X_train_w[Y_train == 0]

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

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="bkg_MC", help="Balancing type")
args = parser.parse_args()

model_type = args.model_type 



model=NN_model(X_train.shape[1], 2, 100, 0.01, 1e-05)
plott_dir = '../../../Plots/NeuralNetwork/Weighting_methods/'

N_sig_train = sum(Y_train)
Balance_train = np.ones(len(Y_train))
if model_type == 'bkg_MC':
    N_bkg_train = len(Y_train) - N_sig_train
    ratio = N_sig_train/N_bkg_train
    Balance_train[Y_train==0] = ratio


elif model_type == 'bkg_exp':
    N_bkg_train = sum(X_train_w[Y_train==0])
    ratio = N_sig_train/N_bkg_train
    Balance_train[Y_train==0] = ratio

elif model_type == 'sig_MC':  
    N_bkg_train = len(Y_train) - N_sig_train
    ratio = N_bkg_train/N_sig_train
    Balance_train[Y_train==1] = ratio
    
elif model_type == 'sig_exp':
    N_bkg_train = sum(X_train_w[Y_train==0])
    ratio = N_bkg_train/N_sig_train
    Balance_train[Y_train==1] = ratio

W_train = pd.DataFrame(Balance_train*btr_wgt, columns=['Weight'])

history = model.fit(X_train, Y_train, sample_weight = W_train, 
            validation_data = (X_test, Y_test),    
            epochs = 50, batch_size = int(2**22), use_multiprocessing = True, verbose = 2)
    
plot_dir = plott_dir + model_type +'/'
try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

model.save('../../Models/NN/Weighting_methods/'+model_type)

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
