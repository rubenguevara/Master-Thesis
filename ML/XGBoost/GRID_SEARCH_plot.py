import os, argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import YlGn_r as color


parser = argparse.ArgumentParser()
parser.add_argument('--met_reg', type=str, default="50-100", help="MET signal region")
args = parser.parse_args()

met_reg = args.met_reg


plot_dir = '../../Plots/XGBoost/Model_independent_frfr/'+met_reg+'/GRIDSEARCH/'
try:
    os.makedirs(plot_dir+'AUC/')

except FileExistsError:
    pass

try:
    os.makedirs(plot_dir+'Accuracy/')

except FileExistsError:
    pass


np_dir = '/storage/racarcam/Data/XGB_frfr/'+met_reg+'/'

""" Choose variables"""
eta = [0.001, 0.01, 0.1, 1]
lamda = 1e-5
n_estimator = [10, 100, 500, 1000]
max_depth = [3, 4, 5, 6]

Train_accuracy = np.load(np_dir+'train_acc.npy')
Test_accuracy = np.load(np_dir+'test_acc.npy')
Train_AUC = np.load(np_dir+'train_auc.npy')
Test_AUC = np.load(np_dir+'test_auc.npy')
Exp_sig = np.load(np_dir+'exp_sig.npy')


indices = np.where(Exp_sig == np.max(Exp_sig))
print("Best expected significance:",np.max(Exp_sig))
print("The parameters are: depth:",max_depth[int(indices[0])],", number of estimators:", n_estimator[int(indices[1])],", and learning rate:", eta[int(indices[2])])
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
    
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            if 'AUC'in title:
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
    if s == "d":
        ax.set_ylabel('$\eta$',fontsize=fontsize)
        ax.set_xlabel('$\\mathrm{number of\\ estimators}$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for $\eta$ and # estimators with a depth = %g' %max_depth[int(ind[0])]
        ax.set_title(titlefr)
        
    elif s =="e":
        ax.set_ylabel('$\eta$',fontsize=fontsize)
        ax.set_xlabel('depth',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for the depth and $\eta$ with %g estimators' %n_estimator[int(ind[2])]
        ax.set_title(titlefr)
        
    elif s =="lr":
        ax.set_xlabel('depth',fontsize=fontsize)
        ax.set_ylabel('$\\mathrm{number of\\ estimators}$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for depth and # estimators with $\eta$ = %g' %eta[int(ind[1])]
        ax.set_title(titlefr)
        
    plt.tight_layout()
    if 'Significance' in title:
        titlefig = title.replace(' ', '_')+'.pdf'
    else:
        better_saving = title.split(' ')
        titlefig = better_saving[0] +'/' +better_saving[1]+'_'+better_saving[2]+'.pdf'
    
    plt.savefig(plot_dir+titlefig)
    
    plt.show()

plot_data(n_estimator, eta, "d", indices, 100*Train_accuracy[int(indices[0]),:,:], 'Accuracy training elr')
plot_data(n_estimator ,eta, "d", indices, 100*Test_accuracy[int(indices[0]),:,:], 'Accuracy testing elr')
plot_data(max_depth, n_estimator, "lr", indices, 100*Train_accuracy[:,:,int(indices[2])], 'Accuracy training de')
plot_data(max_depth, n_estimator, "lr", indices, 100*Test_accuracy[:,:,int(indices[2])], 'Accuracy testing de')
plot_data(max_depth, eta,  "e", indices, 100*Train_accuracy[:,int(indices[1]),:], 'Accuracy training dlr')
plot_data(max_depth, eta, "e", indices, 100*Test_accuracy[:,int(indices[1]),:], 'Accuracy testing dlr')

plot_data(n_estimator, eta, "d", indices, Train_AUC[int(indices[0]),:,:], 'AUC training elr')
plot_data(n_estimator, eta, "d", indices, Test_AUC[int(indices[0]),:,:], 'AUC testing elr')
plot_data(max_depth, n_estimator, "lr", indices, Train_AUC[:,:,int(indices[2])], 'AUC training de')
plot_data(max_depth, n_estimator, "lr", indices, Test_AUC[:,:,int(indices[2])], 'AUC testing de')
plot_data(max_depth, eta, "e", indices, Train_AUC[:,int(indices[1]),:], 'AUC training dlr')
plot_data(max_depth, eta, "e", indices, Test_AUC[:,int(indices[1]),:], 'AUC testing dlr')

plot_data(n_estimator, eta, "d", indices, Exp_sig[int(indices[0]),:,:], 'Significance elr')
plot_data(max_depth, n_estimator, "lr", indices, Exp_sig[:,:,int(indices[2])], 'Significance de')
plot_data(max_depth, eta, "e", indices, Exp_sig[:,int(indices[1]),:], 'Significance dlr')
