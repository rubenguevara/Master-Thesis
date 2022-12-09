import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import YlGn_r as color

plot_dir = '../Plots/XGBoost/FULL/GRIDSEARCH_13-18/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

eta = np.logspace(-3, 0, 4)                                                  
lamda = 1e-5
max_depth = [13, 14, 15, 16, 17, 18]

np_dir = 'Data/xgb_d_13-18/'

Train_accuracy = np.load(np_dir+'train_acc.npy')
Test_accuracy = np.load(np_dir+'test_acc.npy')
Train_AUC = np.load(np_dir+'train_auc.npy')
Test_AUC = np.load(np_dir+'test_auc.npy')
Exp_sig = np.load(np_dir+'exp_sig.npy')

indices = np.where(Exp_sig == np.max(Exp_sig))
print("Best expected significance:",np.max(Exp_sig))
print("The parameters are: depth:",max_depth[int(indices[0])],", eta:", eta[int(indices[1])])
print("This gives an AUC and Binary Accuracy of %g and %g when training" %(Train_AUC[indices], Train_accuracy[indices]) )
print("This gives an AUC and Binary Accuracy of %g and %g when testing " %(Test_AUC[indices], Test_accuracy[indices]) )

def plot_data(x, y, data, title=None):

    # plot results
    fontsize=16

    if 'significance' in title:
        vmax= np.max(data)+0.1*np.max(data)
    elif 'AUC' in title:
        vmax= 1.0
    else:
        vmax= np.max(data)+0.01*np.max(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data.T, interpolation='nearest', cmap=color, vmax=vmax)
    if len(x) > 5:  
        sh = 0.62
    else: 
        sh = 1.0
    cbar=fig.colorbar(cax, shrink = sh)
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
                c = '%.3f' %data[i,j]  
            elif 'significance' in title:
                c = '%.3f $\sigma$' %data[i,j]
            else:
                c = "${0:.1f}\\%$".format( data[i,j])  
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]
    
    
    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)
    ax.set_ylabel('$\eta$',fontsize=fontsize)
    ax.set_xlabel('$\\mathrm{tree\\ depth}$',fontsize=fontsize)
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