import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

print(tf.__version__)


save_dir = "../../../storage/racarcam/"
filename = "Stat_red_DM_Run2_50MET.h5"

df = pd.read_hdf(save_dir+filename, key='df_tot')

df_features = df.copy()
df_EventNumber = df_features.pop('EventID')
df_Weight = df_features.pop('Weight')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_RunPeriod = df_features.pop('RunPeriod')

df_labels = df_features.pop('Label')

X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=0.2, train_size=0.8, random_state=42)

normalize = layers.experimental.preprocessing.Normalization()
# normalize = layers.Normalization()
normalize.adapt(X_train)

def NN_model(inputsize, n_layers, n_neuron, eta, lamda, norm):
    model=tf.keras.Sequential([norm])      
    
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

network = NN_model(X_train.shape[1], 3, 10, 0.1, 1e-3, normalize)
network.fit(X_train, Y_train, epochs=10, batch_size=100)

plot_dir = 'Plots_NeuralNetwork/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

network_pred_label = network.predict(X_test).ravel()
test = Y_test
pred = network_pred_label


fpr, tpr, thresholds = roc_curve(test, pred, pos_label=1)
roc_auc = auc(fpr,tpr)
plt.figure(1)
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for model on full DM dataset')
plt.legend(loc="lower right")
plt.savefig(plot_dir+'ROC.pdf')
plt.show()


plt.figure(2, figsize=[10,6])
n, bins, patches = plt.hist(pred[test==0], 200, facecolor='blue', alpha=0.2,label="Background")
n, bins, patches = plt.hist(pred[test==1], 200, facecolor='red' , alpha=0.2, label="Signal")
plt.xlabel('TF output')
plt.xlim([0,1])
plt.ylabel('Events')
plt.title('Model output, Full DM dataset, validation data')
plt.grid(True)
plt.legend()
plt.savefig(plot_dir+'VAL.pdf')
plt.show()


normalize2 = layers.experimental.preprocessing.Normalization()
normalize2.adapt(df_features)
network = NN_model(df_features.shape[1], 3, 10, 0.1, 1e-3, normalize2)
history = network.fit(df_features, df_labels, validation_split=0.25, epochs=10, batch_size=100)

plt.figure(3)
plt.plot(history.history['binary_accuracy'], label = 'Train')
plt.plot(history.history['val_binary_accuracy'], label = 'Test')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plot_dir+'Accuracy.pdf')
plt.show()

plt.figure(4)
plt.plot(history.history['loss'], label = 'Train')
plt.plot(history.history['val_loss'], label = 'Test')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plot_dir+'Loss.pdf')
plt.show()

# network.save('Models/DM_Slider')

