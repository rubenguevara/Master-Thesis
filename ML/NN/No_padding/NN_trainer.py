import os, argparse, time, gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
print(tf.__version__)


parser = argparse.ArgumentParser()
parser.add_argument('--try_pad', type=str, default="0", help="Try new variables to pad")
args = parser.parse_args()

try_pad = args.try_pad

dm_model = 'DH_HDS'

save_dir = "/storage/racarcam/"
bkg_file = save_dir+'bkgs_final.h5'
sig_file = save_dir+'/DM_Models/DM_Zp_'+dm_model.lower()+'.h5'

df_bkg = pd.read_hdf(bkg_file, key='df_tot')
df_sig = pd.read_hdf(sig_file, key='df_tot')

df = pd.concat([df_bkg, df_sig])

new_variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50']
paddable_variables = ['jet1Pt', 'jet1Phi', 'jet1Eta', 'jet2Pt', 'jet2Phi', 'jet2Eta', 'jet3Pt', 'jet3Phi', 'jet3Eta', 'mjj', 'dPhiCloseMet', 'dPhiLeps']

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_features = df_features.drop(paddable_variables, axis=1)



t0 = time.time()
start = time.asctime(time.localtime())
print('Started', start)

if try_pad == '0':
    df_features = df_features.drop(new_variables, axis = 1)


df_labels = df_features.pop('Label')

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=test_size, random_state=42)
X_train_w = X_train.pop('Weight')
W_test = X_test.pop('Weight')
scaler = 1/test_size


N_sig_train = sum(Y_train)
N_bkg_train = len(Y_train) - N_sig_train
ratio = N_sig_train/N_bkg_train


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
    adam=tf.optimizers.SGD(learning_rate=eta)
    
    model.compile(loss=tf.losses.BinaryCrossentropy(),
                optimizer=adam,
                metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    return model


def get_Z(model, b_size, cut):
    pred = model.predict(X_test, batch_size = b_size, use_multiprocessing = True, verbose = 0).ravel()
    bkg_pred, bins, patches = plt.hist(pred[Y_test==0], weights = W_test[Y_test==0]*scaler, bins = 100)
    sig_pred, bins, patches = plt.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = 100)
    Z = low_stat_Z(sum(sig_pred[int(cut):]), sum(bkg_pred[int(cut):]))
    plt.clf()
    return Z

def low_stat_Z(sig, bkg):
    Z = np.sqrt(2*( (sig + bkg)*np.log(1 + sig/bkg) - sig ))
    return Z

def grid_search(n_lays, eta, lamda, n_neuron, eps, b_size):
    n_layers = n_lays                                                       # Define number of hidden layers in the model
    epochs = eps                                                            # Number of reiterations over the input data
    batch_size = b_size
    Train_AUC = np.zeros((len(lamda),len(eta),len(n_neuron)))               # Define matrices to store accuracy scores as a function
    Test_AUC = np.zeros((len(lamda),len(eta),len(n_neuron)))                # of learning rate and number of hidden neurons for 
    Train_accuracy = np.zeros((len(lamda),len(eta),len(n_neuron)))          
    Test_accuracy = np.zeros((len(lamda),len(eta),len(n_neuron)))           
    Exp_sig = np.zeros((len(lamda),len(eta),len(n_neuron)))  
    
    for i in range(len(lamda)):                                             # Run loops over hidden neurons and learning rates to calculate 
        for j in range(len(eta)):                                           # accuracy scores 
            for k in range(len(n_neuron)):
                print("lambda:",i+1,"/",len(lamda),", eta:",j+1,"/",len(eta),", n_neuron:",k+1,"/",len(n_neuron), '| With', lamda[i], 'lambda,', eta[j], 'eta and', n_neuron[k], 'neurons')
                
                model=NN_model(X_train.shape[1],n_layers,n_neuron[k],eta[j],lamda[i])
                model.fit(X_train, Y_train, sample_weight = W_train, epochs=epochs, batch_size = batch_size, verbose = 0, use_multiprocessing = True)
                
                train = model.evaluate(X_train, Y_train, sample_weight = W_train, batch_size = batch_size, use_multiprocessing = True, verbose = 0)
                test = model.evaluate(X_test, Y_test, batch_size = batch_size, use_multiprocessing = True, verbose = 0)
        
                Train_accuracy[i,j,k] = train[1]   
                Test_accuracy[i,j,k] = test[1]
                Train_AUC[i,j,k] = train[2]   
                Test_AUC[i,j,k] = test[2]
                
                Z = get_Z(model, batch_size, 85) 
                Exp_sig[i,j,k] = Z
                
                print('%.3f sigma with %.2f%% accuracy and %.3f AUC on testing' %(Z, test[1]*100, test[2]))
                print('            and %.2f%% accuracy and %.3f AUC on training' %(train[1]*100, train[2]))
                
                #To fix memmory leak bug 
                plt.clf()
                tf.keras.backend.clear_session()
                gc.collect()
                
    
    return Train_accuracy, Test_accuracy, Train_AUC, Test_AUC, Exp_sig

print('==='*30)
print('Starting gridsearch', time.asctime(time.localtime()))
n_neuron = [1, 10, 50, 100]                                                  # Define number of neurons per layer
eta = np.logspace(-3, 0, 4)                                                  # Define vector of learning rates (parameter to SGD optimiser)
lamda = np.logspace(-5, -2, 4)                                               # Define hyperparameter
# lamda = 1e-5
hidden_layers = [2, 3, 4, 5]
Train_accuracy, Test_accuracy, Train_AUC, Test_AUC, Exp_sig = grid_search(2, eta, lamda, n_neuron, 50, int(2**20))


np_dir = '../../Data/NN/Padding/'
if try_pad == '0':
    np_dir = np_dir+'No_pad/'
    
else:
    np_dir = np_dir+'New_pad/'

try:
    os.makedirs(np_dir)

except FileExistsError:
    pass

np.save(np_dir+'train_acc', Train_accuracy)
np.save(np_dir+'test_acc', Test_accuracy)
np.save(np_dir+'train_auc', Train_AUC)
np.save(np_dir+'test_auc', Test_AUC)
np.save(np_dir+'exp_sig', Exp_sig)

print('==='*30)
t = "{:.2f}".format(int( time.time()-t0 )/60.)
finish = time.asctime(time.localtime())
print('Finished', finish)
print('Total time:', t)
print('==='*30)

Exp_sig = np.nan_to_num(Exp_sig)
indices = np.where(Exp_sig == np.max(Exp_sig))


model=NN_model(X_train.shape[1], 2, n_neuron[int(indices[2])], eta[int(indices[1])], lamda[int(indices[0])])
history = model.fit(X_train, Y_train, sample_weight = W_train, 
                    validation_data = (X_test, Y_test),
                    epochs=50, batch_size = int(2**22), verbose = 1, use_multiprocessing = True)

model_dir = '../../Models/NN/No_pad/'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass

if try_pad == '0':
    model.save(model_dir+'Without_padding')
    plot_dir = '../../../Plots/NeuralNetwork/Padding/No_pad/'

else:
    model.save(model_dir+'With_padding')
    plot_dir = '../../../Plots/NeuralNetwork/Padding/New_pad/'

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
