import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import YlGn_r as color
import os

n_neuron = [10, 50, 100]     
eta = np.logspace(-3, -1, 3) 
n_layers = [3, 4, 5]

Train_accuracy = np.load('../Data/DNN/train_acc.npy')
Test_accuracy = np.load('../Data/DNN/test_acc.npy')
Train_AUC = np.load('../Data/DNN/train_auc.npy')
Test_AUC = np.load('../Data/DNN/test_auc.npy')
Exp_sig = np.load('../Data/DNN/exp_sig.npy')
Exp_sig = np.nan_to_num(Exp_sig)
indices = np.where(Exp_sig == np.max(Exp_sig))
print("Best expected significance:",np.max(Exp_sig))
print("The parameters are: layers:",n_layers[int(indices[0])],", eta:", eta[int(indices[1])],"and", n_neuron[int(indices[2])],'neurons')
print("This gives an AUC and Binary Accuracy of %g and %g when training" %(Train_AUC[indices], Train_accuracy[indices]) )
print("This gives an AUC and Binary Accuracy of %g and %g when testing " %(Test_AUC[indices], Test_accuracy[indices]) )

def plot_data(x, y, s, ind, data, title=None):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    if 'AUC' in title:
        vmax = 1.0
    elif 'Accuracy' in title:
        vmax = 100
    else:   
        vmax = np.max(data) + 0.1*np.max(data)
    vmin = np.min(data) - 0.1*np.min(data)
    
    cax = ax.matshow(data.T, interpolation='nearest', vmax=vmax, vmin=vmin, cmap=color)
    
    cbar=fig.colorbar(cax)
    if 'Accuracy' in title:
        cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    elif 'AUC' in title:
        cbar.ax.set_ylabel('AUC',rotation=90,fontsize=fontsize)
    else:
        cbar.ax.set_ylabel('$\sigma$',rotation=90,fontsize=fontsize)

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            if 'AUC' in title:
                c = '%.3f' %data[i,j]  
            elif 'Significance' in title:
                c = '%.3f $\sigma$' %data[i,j]
            else:
                c = "${0:.1f}\\%$".format( data[i,j])  
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]
    
    
    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)
    if s == "l":
        ax.set_xlabel('$\eta$',fontsize=fontsize)
        ax.set_ylabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for $\eta$ and hidden neuron with %g layers' %n_layers[int(ind[0])]
        ax.set_title(titlefr)
        
    elif s =="n":
        ax.set_ylabel('$\eta$',fontsize=fontsize)
        ax.set_xlabel('layers',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for $\eta$ and layers with %g hidden neurons' %n_neuron[int(ind[2])]
        ax.set_title(titlefr)
        
    elif s =="e":
        ax.set_xlabel('layers',fontsize=fontsize)
        ax.set_ylabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for layers and hidden neuron with $\eta$ = %g' %eta[int(ind[1])]
        ax.set_title(titlefr)
        
    plt.tight_layout()
    if 'Significance' in title:
        titlefig = title.replace(' ', '_')+'.pdf'
    else:
        better_saving = title.split(' ')
        titlefig = better_saving[0] +'/' +better_saving[1]+'_'+better_saving[2]+'.pdf'
    plt.savefig('../../Plots/NeuralNetwork/FULL/GRID_layers_eta_neurons/'+titlefig)
    
    plt.show()

try:
    os.makedirs('../../Plots/NeuralNetwork/FULL/GRID_layers_eta_neurons/')

except FileExistsError:
    pass

plot_data(eta, n_neuron, "l", indices, 100*Train_accuracy[int(indices[0]),:,:], 'Accuracy training ne')
plot_data(eta, n_neuron, "l", indices, 100*Test_accuracy[int(indices[0]),:,:], 'Accuracy testing ne')
plot_data(n_layers, eta, "n", indices, 100*Train_accuracy[:,:,int(indices[2])], 'Accuracy training le')
plot_data(n_layers, eta, "n", indices, 100*Test_accuracy[:,:,int(indices[2])], 'Accuracy testing le')
plot_data(n_layers, n_neuron,  "e", indices, 100*Train_accuracy[:,int(indices[1]),:], 'Accuracy training nl')
plot_data(n_layers, n_neuron, "e", indices, 100*Test_accuracy[:,int(indices[1]),:], 'Accuracy testing nl')

plot_data(eta, n_neuron, "l", indices, Train_AUC[int(indices[0]),:,:], 'AUC training ne')
plot_data(eta, n_neuron, "l", indices, Test_AUC[int(indices[0]),:,:], 'AUC testing ne')
plot_data(n_layers, eta, "n", indices, Train_AUC[:,:,int(indices[2])], 'AUC training le')
plot_data(n_layers, eta, "n", indices, Test_AUC[:,:,int(indices[2])], 'AUC testing le')
plot_data(n_layers, n_neuron, "e", indices, Train_AUC[:,int(indices[1]),:], 'AUC training nl')
plot_data(n_layers, n_neuron, "e", indices, Test_AUC[:,int(indices[1]),:], 'AUC testing nl')

plot_data(eta, n_neuron, "l", indices, Exp_sig[int(indices[0]),:,:], 'Significance ne')
plot_data(n_layers, eta, "n", indices, Exp_sig[:,:,int(indices[2])], 'Significance le')
plot_data(n_layers, n_neuron, "e", indices, Exp_sig[:,int(indices[1]),:], 'Significance nl')
