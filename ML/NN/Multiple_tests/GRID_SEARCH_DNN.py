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

# means = np.asarray([df['jet1Phi'].mean(), df['jet2Phi'].mean(), df['jet1Eta'].mean(), df['jet2Eta'].mean()])
# np.save('../Data/df_values/means', means)

# df['jet1Phi'] = pd.DataFrame(df['jet1Phi']).replace(10, means[0])
# df['jet2Phi'] = pd.DataFrame(df['jet2Phi']).replace(10, means[1])
# df['jet1Eta'] = pd.DataFrame(df['jet1Eta']).replace(10, means[2])
# df['jet2Eta'] = pd.DataFrame(df['jet2Eta']).replace(10, means[3])

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

normalized_df = (df_features-df_features.mean())/np.sqrt(df_features.var())
normalized_df['Weight'] = df_Weights
df_features.mean().to_pickle("../Data/df_values/mean_df.pkl")
df_features.var().to_pickle("../Data/df_values/var_df.pkl")

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(normalized_df, df_labels, test_size=test_size, random_state=42)
X_train_w = np.asarray(X_train.pop('Weight'))
W_test = np.asarray(X_test.pop('Weight'))

W_train = np.ones(len(Y_train))
W_train[Y_train==0] = sum(W_train[Y_train==1])/sum(W_train[Y_train==0])
W_train = pd.DataFrame(W_train, columns=['Weight'])                          # Has to be a pandas DataFrame or it crashes

scaler = 1/test_size

def NN_model(inputsize, n_layers, n_neuron, eta, lamda):
    model=tf.keras.Sequential()
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

model_dir = '../Models/NN/'

try:
    os.makedirs(model_dir)

except FileExistsError:
    pass

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

def grid_search(n_lays, eta, lamba, n_neuron, eps, b_size):
    lamda = lamba
    epochs = eps                                                            # Number of reiterations over the input data
    batch_size = b_size
    Train_AUC = np.zeros((len(n_lays),len(eta),len(n_neuron)))               # Define matrices to store accuracy scores as a function
    Test_AUC = np.zeros((len(n_lays),len(eta),len(n_neuron)))                # of learning rate and number of hidden neurons for 
    Train_accuracy = np.zeros((len(n_lays),len(eta),len(n_neuron)))          
    Test_accuracy = np.zeros((len(n_lays),len(eta),len(n_neuron)))           
    Exp_sig = np.zeros((len(n_lays),len(eta),len(n_neuron)))  
    
    for i in range(len(n_lays)):                                             # Run loops over hidden neurons and learning rates to calculate 
        for j in range(len(eta)):                                           # accuracy scores 
            for k in range(len(n_neuron)):
                print("layers:",i+1,"/",len(n_lays),", eta:",j+1,"/",len(eta),", n_neuron:",k+1,"/",len(n_neuron), '| With', n_lays[i], 'layers,', eta[j], 'eta and', n_neuron[k], 'neurons')
                
                model=NN_model(X_train.shape[1],n_lays[i],n_neuron[k],eta[j],lamda)
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
n_neuron = [10, 50, 100]                                                  # Define number of neurons per layer
eta = np.logspace(-3, -1, 3)                                                  # Define vector of learning rates (parameter to SGD optimiser)
lamda = 1e-5
n_layers = [3, 4, 5]
# Train_accuracy, Test_accuracy, Train_AUC, Test_AUC, Exp_sig = grid_search(n_layers, eta, lamda, n_neuron, 50, int(2**23))

# np.save('../Data/DNN/train_acc', Train_accuracy)
# np.save('../Data/DNN/test_acc', Test_accuracy)
# np.save('../Data/DNN/train_auc', Train_AUC)
# np.save('../Data/DNN/test_auc', Test_AUC)
# np.save('../Data/DNN/exp_sig', Exp_sig)

# print('==='*30)
# t = "{:.2f}".format(int( time.time()-t0 )/60.)
# finish = time.asctime(time.localtime())
# print('Finished', finish)
# print('Total time:', t)
# print('==='*30)

# Exp_sig = np.nan_to_num(Exp_sig)
# indices = np.where(Exp_sig == np.max(Exp_sig))

# print("Best expected significance:",np.max(Exp_sig))
# print("The parameters are: layers:",n_layers[int(indices[0])],", eta:", eta[int(indices[1])],"and", n_neuron[int(indices[2])],'neurons')
# print("This gives an AUC and Binary Accuracy of %g and %g when training" %(Train_AUC[indices], Train_accuracy[indices]) )
# print("This gives an AUC and Binary Accuracy of %g and %g when testing " %(Test_AUC[indices], Test_accuracy[indices]) )


# model=NN_model(X_train.shape[1], n_layers[int(indices[0])], n_neuron[int(indices[2])], eta[int(indices[1])], lamda)
# model.fit(X_train, Y_train, sample_weight = W_train, epochs = 50, batch_size = int(2**23), use_multiprocessing = True)
# model.save('../Models/NN/BEST_DNN_GRIDDY')


model=NN_model(X_train.shape[1], 11, 100, 0.01, lamda)
model.fit(X_train, Y_train, sample_weight = W_train, epochs = 50, batch_size = int(2**22), use_multiprocessing = True)
model.save('../Models/NN/10_HIDDEN_LAYERS')