import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

print(tf.__version__)

save_dir = "../../../storage/racarcam/"
filename = "Stat_red_DM1_Run2_50MET.h5"

df = pd.read_hdf(save_dir+filename, key='df_tot')

df_features = df.copy()
df_EventNumber = df_features.pop('EventNumber')
df_weight = df_features.pop('weight')
df_RunNumber = df_features.pop('RunNumber')
df_RunPeriod = df_features.pop('RunPeriod')

df_labels = df_features.pop('Label')

df_train, df_test, df_train_label, df_test_label = train_test_split(df_features, df_labels, test_size=0.2, train_size=0.8, random_state=42)

df_train = np.array(df_train)
df_test = np.array(df_test)
df_train_label = np.array(df_train_label)
df_test_label = np.array(df_test_label)


# normalize = layers.experimental.preprocessing.Normalization()
# # normalize = layers.Normalization()
# normalize.adapt(df_train)

def NN_model(inputsize, n_layers, n_neuron, eta, lamda):
    model=tf.keras.Sequential()      
    
    for i in range(n_layers):                                                # Run loop to add hidden layers to the model
        if (i==0):                                                           # First layer requires input dimensions
            model.add(layers.Dense(n_neuron, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamda), input_dim=inputsize))
            
        else:                                                                # Subsequent layers are capable of automatic shape inferencing
            model.add(layers.Dense(n_neuron, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamda)))
    
    model.add(layers.Dense(1, activation='sigmoid'))                         # 1 output - signal or no signal
    sgd=tf.optimizers.SGD(learning_rate=eta)
    
    model.compile(loss=tf.losses.BinaryCrossentropy(),
                optimizer=sgd,
                metrics = [tf.keras.metrics.BinaryAccuracy()])
    return model

network=NN_model(df_train.shape[1], 3, 10 ,0.1 , 1e-3)
network.fit(df_train, df_train_label, epochs=10, batch_size=100)
exit()
def grid_search(n_lays, eta, lamda, n_neuron, eps, b_size):
    n_layers = n_lays                                                        # Define number of hidden layers in the model
    epochs = eps                                                             # Number of reiterations over the input data
    batch_size = b_size
    Train_accuracy = np.zeros((len(lamda),len(eta),len(n_neuron)))           # Define matrices to store accuracy scores as a function
    Test_accuracy = np.zeros((len(lamda),len(eta),len(n_neuron)))            # of learning rate and number of hidden neurons for 

    print('Go for a stroll, this will take a while')
    for i in range(len(lamda)):                                              # Run loops over hidden neurons and learning rates to calculate 
        for j in range(len(eta)):                                            # accuracy scores 
            for k in range(len(n_neuron)):
                
                print("lambda:",i+1,"/",len(lamda),", eta:",j+1,"/",len(eta),", n_neuron:",k+1,"/",len(n_neuron))
                
                higgsmodel=NN_model(higgs_train.shape[1],n_layers,n_neuron[k],eta[j],lamda[i], normalize)
                higgsmodel.fit(higgs_train,higgs_train_label,epochs=epochs,batch_size=batch_size,verbose=0)
                print("Accuracy test")
                
                Train_accuracy[i,j,k]=higgsmodel.evaluate(higgs_train, higgs_train_label)[1]
                Test_accuracy[i,j,k]=higgsmodel.evaluate(higgs_test, higgs_test_label)[1]
    
    max=np.max(Train_accuracy)
    train_indices=np.where(Train_accuracy==max)
    print("Best training accuracy:",max)
    print("The parameters are: lambda:",lamda[int(train_indices[0])],", eta:", eta[int(train_indices[1])],"and n_neuron:",n_neuron[int(train_indices[2])])

    max=np.max(Test_accuracy)
    indices=np.where(Test_accuracy==max)
    print("Best testing accuracy:",max)
    print("The parameters are: lambda:",lamda[int(indices[0])],", eta:", eta[int(indices[1])],"and n_neuron:",n_neuron[int(indices[2])])

    higgs_model=NN_model(higgs_train.shape[1],n_layers,n_neuron[int(indices[2])],eta[int(indices[1])],lamda[int(indices[0])])
    higgs_model.fit(higgs_train,higgs_train_label,epochs=epochs)
    
    return Train_accuracy, Test_accuracy, higgs_model, indices, train_indices


def plot_data(x, y, s, ind, data, title=None):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)
    
    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])  
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]
    
    
    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)
    if s == "l":
        ax.set_ylabel('$\eta$',fontsize=fontsize)
        ax.set_xlabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for $\eta$ and hidden neuron with $\lambda$ = %g' %lamda[int(ind[0])]
        ax.set_title(titlefr)
        
    elif s =="n":
        ax.set_xlabel('$\eta$',fontsize=fontsize)
        ax.set_ylabel('$\lambda$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for $\eta$ and $\lambda$ with %g hidden neurons' %n_neuron[int(ind[2])]
        ax.set_title(titlefr)
        
    elif s =="e":
        ax.set_ylabel('$\lambda$',fontsize=fontsize)
        ax.set_xlabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for $\lambda$ and hidden neuron with $\eta$ = %g' %eta[int(ind[1])]
        ax.set_title(titlefr)
        
    plt.tight_layout()
    titlefig = title.replace(' ', '_')
    plt.savefig('../Plots/'+titlefig)
    
    plt.show()
    
    
n_neuron = np.logspace(0,3,4,dtype=int)                                      # Define number of neurons per layer
eta = np.logspace(-3, 0, 4)                                                  # Define vector of learning rates (parameter to SGD optimiser)
lamda = np.logspace(-5, -2, 4)                                               # Define hyperparameter

Train_accuracy, Test_accuracy, higgs_model, indices, train_ind = grid_search(3, eta, lamda, n_neuron, 10, 100)

np.save('../Data/train', Train_accuracy)
np.save('../Data/test', Test_accuracy)
np.save('../Data/train_ind', train_ind)
np.save('../Data/indices', indices)


#Train_accuracy = np.load('../Data/train.npy')
#Test_accuracy = np.load('../Data/test.npy')
#train_ind = np.load('../Data/train_ind.npy')
#indices = np.load('../Data/indices.npy')

plot_data(n_neuron, eta, "l", train_ind, Train_accuracy[int(train_ind[0]),:,:], 'Training ne')
plot_data(n_neuron, eta, "l", indices, Test_accuracy[int(indices[0]),:,:], 'Testing ne')
plot_data(eta, lamda, "n", train_ind, Train_accuracy[:,:,int(train_ind[2])], 'Training le')
plot_data(eta, lamda, "n", indices, Test_accuracy[:,:,int(indices[2])], 'Testing le')
plot_data(n_neuron, lamda, "e", train_ind, Train_accuracy[:,int(train_ind[1]),:], 'Training nl')
plot_data(n_neuron, lamda, "e", indices, Test_accuracy[:,int(indices[1]),:], 'Testing nl')

# To not load all this all the time
higgs_model.save('../Models/Supervised_Higgs')

# Epoch choosing
norms = layers.Normalization()
norms.adapt(higgs_features)
higgsmodel = NN_model(higgs_features.shape[1], 3, 100, 0.1, 1e-05, norms)    # OBS best TESTING parameters and choose 100 neurons to spare time
history = higgsmodel.fit(higgs_features, higgs_labels, validation_split=0.25, batch_size=100, epochs=50)
plt.plot(history.history['binary_accuracy'], label = 'Train')
plt.plot(history.history['val_binary_accuracy'], label = 'Test')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('../Plots/Accuracy')
plt.show()
plt.plot(history.history['loss'], label = 'Train')
plt.plot(history.history['val_loss'], label = 'Test')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('../Plots/Loss')
plt.show()

""" Result 11.07.2022

...
Best training accuracy: 0.852495014667511
The parameters are: lambda: 1e-05 , eta: 1.0 and n_neuron: 1000
Best testing accuracy: 0.8406199812889099
The parameters are: lambda: 1e-05 , eta: 0.1 and n_neuron: 1000

...
Epoch 10/10
6250/6250 [==============================] - 58s 9ms/step - loss: 0.3602 - binary_accuracy: 0.8492
"""