import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import YlGn_r as color

eta = np.logspace(-3, 0, 4)                                                  # Define vector of learning rates (parameter to SGD optimiser)
lamda = np.logspace(-5, -2, 4)                                               # Define hyperparameter
n_neuron = [1, 10, 50, 100]

Train_accuracy = np.load('../Data/NN/train_acc.npy')
Test_accuracy = np.load('../Data/NN/test_acc.npy')
Train_AUC = np.load('../Data/NN/train_auc.npy')
Test_AUC = np.load('../Data/NN/test_auc.npy')
Exp_sig = np.load('../Data/NN/exp_sig.npy')
Exp_sig = np.nan_to_num(Exp_sig)
indices = np.where(Exp_sig == np.max(Exp_sig))
print("Best expected significance:",np.max(Exp_sig))
print("The parameters are: lambda:",lamda[int(indices[0])],", eta:", eta[int(indices[1])],"and", n_neuron[int(indices[2])],'neurons')
print("This gives an AUC and Binary Accuracy of %g and %g when training" %(Train_AUC[indices], Train_accuracy[indices]) )
print("This gives an AUC and Binary Accuracy of %g and %g when testing " %(Test_AUC[indices], Test_accuracy[indices]) )

def plot_data(x, y, s, ind, data, title=None):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    vmax = np.max(data) + 0.1*np.max(data)
    cax = ax.matshow(data.T, interpolation='nearest', vmax=vmax, cmap=color)
    
    cbar=fig.colorbar(cax)
    if 'Accuracy' in title:
        cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    elif 'AUC' in title:
        cbar.ax.set_ylabel('AUC',rotation=90,fontsize=fontsize)
    else:
        cbar.ax.set_ylabel('$\sigma$',rotation=90,fontsize=fontsize)
    # cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    # cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            if title.split(' ')[1] == 'AUC':
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
        titlefr= title[:-2] + 'scores for $\eta$ and hidden neuron with $\lambda$ = %g' %lamda[int(ind[0])]
        ax.set_title(titlefr)
        
    elif s =="n":
        ax.set_ylabel('$\eta$',fontsize=fontsize)
        ax.set_xlabel('$\lambda$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for $\eta$ and $\lambda$ with %g hidden neurons' %n_neuron[int(ind[2])]
        ax.set_title(titlefr)
        
    elif s =="e":
        ax.set_xlabel('$\lambda$',fontsize=fontsize)
        ax.set_ylabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for $\lambda$ and hidden neuron with $\eta$ = %g' %eta[int(ind[1])]
        ax.set_title(titlefr)
        
    plt.tight_layout()
    titlefig = title.replace(' ', '_')+'.pdf'
    plt.savefig('../../Plots/NeuralNetwork/FULL/GRID/'+titlefig)
    
    plt.show()

plot_data(eta, n_neuron, "l", indices, 100*Train_accuracy[int(indices[0]),:,:], 'Accuracy/Training ne')
plot_data(eta, n_neuron, "l", indices, 100*Test_accuracy[int(indices[0]),:,:], 'Accuracy/Testing ne')
plot_data(lamda, eta, "n", indices, 100*Train_accuracy[:,:,int(indices[2])], 'Accuracy/Training le')
plot_data(lamda, eta, "n", indices, 100*Test_accuracy[:,:,int(indices[2])], 'Accuracy/Testing le')
plot_data(lamda, n_neuron,  "e", indices, 100*Train_accuracy[:,int(indices[1]),:], 'Accuracy/Training nl')
plot_data(lamda, n_neuron, "e", indices, 100*Test_accuracy[:,int(indices[1]),:], 'Accuracy/Testing nl')

plot_data(eta, n_neuron, "l", indices, Train_AUC[int(indices[0]),:,:], 'AUC/Training ne')
plot_data(eta, n_neuron, "l", indices, Test_AUC[int(indices[0]),:,:], 'AUC/Testing ne')
plot_data(lamda, eta, "n", indices, Train_AUC[:,:,int(indices[2])], 'AUC/Training le')
plot_data(lamda, eta, "n", indices, Test_AUC[:,:,int(indices[2])], 'AUC/Testing le')
plot_data(lamda, n_neuron, "e", indices, Train_AUC[:,int(indices[1]),:], 'AUC/Training nl')
plot_data(lamda, n_neuron, "e", indices, Test_AUC[:,int(indices[1]),:], 'AUC/Testing nl')

plot_data(eta, n_neuron, "l", indices, Exp_sig[int(indices[0]),:,:], 'Significance ne')
plot_data(lamda, eta, "n", indices, Exp_sig[:,:,int(indices[2])], 'Significance le')
plot_data(lamda, n_neuron, "e", indices, Exp_sig[:,int(indices[1]),:], 'Significance nl')
