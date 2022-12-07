import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import copper

plot_dir = '../Plots/XGBoost/FULL/GRIDSEARCH/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

eta = np.logspace(-3, 0, 4)                                                  
lamda = 1e-5
max_depth = [3, 4, 5, 6]

Train_accuracy = np.load('Data/xgb_no_lmbd/train_acc.npy')
Test_accuracy = np.load('Data/xgb_no_lmbd/test_acc.npy')
Train_AUC = np.load('Data/xgb_no_lmbd/train_auc.npy')
Test_AUC = np.load('Data/xgb_no_lmbd/test_auc.npy')
Exp_sig = np.load('Data/xgb_no_lmbd/exp_sig.npy')

indices = np.where(Exp_sig == np.max(Exp_sig))
print("Best expected significance:",np.max(Exp_sig))
print("The parameters are: depth:",max_depth[int(indices[0])],", eta:", eta[int(indices[1])])
print("This gives an AUC and Binary Accuracy of %g and %g when training" %(Train_AUC[indices], Train_accuracy[indices]) )
print("This gives an AUC and Binary Accuracy of %g and %g when testing " %(Test_AUC[indices], Test_accuracy[indices]) )

def plot_data(x, y, data, title=None):

    # plot results
    fontsize=16

    if 'significance' in title:
        vmin= np.min(data)-0.1*np.min(data)
    else:
        vmin= np.min(data)-0.01*np.min(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', cmap=copper, vmin=vmin)
    
    cbar=fig.colorbar(cax)
    if title.split(' ')[1] == 'AUC':
        cbar.ax.set_ylabel('AUC',rotation=90,fontsize=fontsize)
    elif 'significance' in title:
        cbar.ax.set_ylabel('$\sigma$',rotation=90,fontsize=fontsize)        
    else:
        cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            if title.split(' ')[1] == 'AUC':
                c = '%.3f' %data[j,i]  
            elif 'significance' in title:
                c = '%.3f $\sigma$' %data[j,i]
            else:
                c = "${0:.1f}\\%$".format( data[j,i])  
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]
    
    
    ax.set_xticklabels(['']+y)
    ax.set_yticklabels(['']+x)
    ax.set_xlabel('$\eta$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{tree\\ depth}$',fontsize=fontsize)
    if 'significance' in title:
        titlefr= title + ' for $\eta$ and tree depth with $\lambda$ = %g' %lamda
    else:
        titlefr= title + ' scores for $\eta$ and tree depth with $\lambda$ = %g' %lamda
    ax.set_title(titlefr)
    
    plt.tight_layout()
    titlefig = title.replace(' ', '_')
    plt.savefig(plot_dir+titlefig+'.pdf')
    
    plt.clf()

plot_data(max_depth, eta, Train_accuracy*100, 'Training accuracy')
plot_data(max_depth, eta, Test_accuracy*100, 'Testing accuracy')
plot_data(max_depth, eta, Train_AUC, 'Training AUC')
plot_data(max_depth, eta, Test_AUC, 'Testing AUC')
plot_data(max_depth, eta, Exp_sig, 'Expected significance')