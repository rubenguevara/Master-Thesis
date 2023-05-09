import os, time
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

print(xgb.__version__)

t0 = time.time()
start = time.asctime(time.localtime())
print('Started', start)


save_dir = "/storage/racarcam/"
bkg_file = save_dir+'new_bkgs.h5'
sig_file = save_dir+'/DM_Models/DM_Zp_lv.h5'

df_bkg = pd.read_hdf(bkg_file, key='df_tot')
df_sig = pd.read_hdf(sig_file, key='df_tot')
df = pd.concat([df_bkg, df_sig])


extra_variables = ['n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50', 'dPhiCloseMet', 'dPhiLeps']


df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_features = df_features.drop(extra_variables, axis=1)


# Signal region search

df_features = df_features.loc[df_features['mll'] > 120]                     # First signal region cut
df_features = df_features.loc[df_features['Weight'] > 0]                    # Only positive weights                     

met_reg = '50-100'
# met_reg = '100-150'
# met_reg = '150'

if met_reg == '50-100':
    df_features = df_features.loc[df_features['met'] < 100]                     

elif met_reg == '100-150':
    df_features = df_features.loc[df_features['met'] > 100]                     
    df_features = df_features.loc[df_features['met'] < 150]      
    
elif met_reg == '150':
    df_features = df_features.loc[df_features['met'] > 150]                      


print('Doing SR with mll > 120 and met',met_reg)
df_labels = df_features.pop('Label')

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=test_size, random_state=42)

W_train = X_train.pop('Weight')
W_test = X_test.pop('Weight')


sow_sig = sum(Y_train)
sow_bkg = len(Y_train)- sow_sig

scaler = 1/test_size

model_dir = '../Models/XGB/'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass

plot_dir = '../../Plots/XGBoost/LV_'+met_reg+'_MET/GRIDSEARCH/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass


np_dir = '../Data/LV_'+met_reg+'_MET/xgb_n_est_1e2-1e6/'

try:
    os.makedirs(np_dir)

except FileExistsError:
    pass

def low_stat_Z(sig, bkg):
    Z = np.sqrt(2*( (sig + bkg)*np.log(1 + sig/bkg) - sig ))
    return Z

def grid_search(n_est, eta, lamda, depth):
    Train_AUC = np.zeros((len(depth),len(n_est)))                             # Define matrices to store accuracy scores as a function
    Test_AUC = np.zeros((len(depth),len(n_est)))                              # of learning rate and tree depth to find best hyperparameters 
    Train_accuracy = np.zeros((len(depth),len(n_est)))          
    Test_accuracy = np.zeros((len(depth),len(n_est)))           
    Exp_sig = np.zeros((len(depth),len(n_est)))  
    
    for i in range(len(depth)):                                             # Run loops over max depth and learning rates to calculate 
        for j in range(len(n_est)):      
            print("max depth:",i+1,"/",len(depth),", n_est:",j+1,"/",len(n_est),'With depth =', depth[i], 'and n_est =', int(n_est[j]))
            
            xgbclassifier = xgb.XGBClassifier(
                max_depth=depth[i], 
                use_label_encoder=False,
                n_estimators=int(n_est[j]),
                learning_rate=eta,
                reg_lambda = lamda,
                predictor = 'cpu_predictor',
                tree_method = 'hist',
                scale_pos_weight=sow_bkg/sow_sig,
                objective='binary:logistic',
                eval_metric='auc',
                missing=-999,
                min_child_weight = 1,
                random_state=42,
                verbosity = 1) 

            xgbclassifier.fit(X_train, Y_train, sample_weight = W_train, verbose = True)
            
            acc_train = xgbclassifier.score(X_train, Y_train)
            acc_test = xgbclassifier.score(X_test, Y_test)
            Train_accuracy[i,j] = acc_train   
            Test_accuracy[i,j] = acc_test
            
            pred = xgbclassifier.predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = roc_curve(Y_test, pred, pos_label=1)
            test = auc(fpr,tpr)
            Test_AUC[i,j] = test
            
            pred_train = xgbclassifier.predict_proba(X_train)[:,1]
            fpr, tpr, thresholds = roc_curve(Y_train, pred_train, pos_label=1)
            train = auc(fpr,tpr)
            Train_AUC[i,j] = train   
            
            bkg_pred, bins, patches = plt.hist(pred[Y_test==0], weights = W_test[Y_test==0]*scaler, bins = 100)
            sig_pred, bins, patches = plt.hist(pred[Y_test==1], weights = W_test[Y_test==1]*scaler, bins = 100)
            Z = low_stat_Z(sum(sig_pred[85:]), sum(bkg_pred[85:]))
            print('%.3f sigma with %.2f%% accuracy and %.3f AUC on testing, and %.2f%% accuracy and %.3f AUC on training' %(Z, acc_test*100, test, acc_train*100, train))
            Exp_sig[i,j] = Z
            plt.clf() 

                
    return Train_accuracy, Test_accuracy, Train_AUC, Test_AUC, Exp_sig


print('==='*20)
print('Starting gridsearch', time.asctime(time.localtime()))

""" Choose variables"""
eta = 0.1
lamda = 1e-5
n_estimator = [10, 100, 500, 1000]
max_depth = [3, 4, 5, 6]

Train_accuracy, Test_accuracy, Train_AUC, Test_AUC, Exp_sig = grid_search(n_estimator, eta, lamda, max_depth)

np.save(np_dir+'train_acc', Train_accuracy)
np.save(np_dir+'test_acc', Test_accuracy)
np.save(np_dir+'train_auc', Train_AUC)
np.save(np_dir+'test_auc', Test_AUC)
np.save(np_dir+'exp_sig', Exp_sig)
print('==='*20)
t = "{:.2f}".format(int( time.time()-t0 )/60.)
finish = time.asctime(time.localtime())
print('Finished', finish)
print('Total time:', t)
print('==='*20)



indices = np.where(Exp_sig == np.max(Exp_sig))
print("Best expected significance:",np.max(Exp_sig))
print("The parameters are: depth:",max_depth[int(indices[0])],", number of estimators:", n_estimator[int(indices[1])])
print("This gives an AUC and Binary Accuracy of %g and %g when training" %(Train_AUC[indices], Train_accuracy[indices]) )
print("This gives an AUC and Binary Accuracy of %g and %g when testing " %(Test_AUC[indices], Test_accuracy[indices]) )

model = xgbclassifier = xgb.XGBClassifier(
                    max_depth=max_depth[int(indices[0])], 
                    use_label_encoder=False,
                    n_estimators=int(n_estimator[int(indices[1])]),
                    learning_rate=eta,
                    reg_lambda = lamda,
                    predictor = 'cpu_predictor',
                    tree_method = 'hist',
                    scale_pos_weight=sow_bkg/sow_sig,
                    objective='binary:logistic',
                    eval_metric='auc',
                    missing=-999,
                    min_child_weight = 1,
                    random_state=42,
                    verbosity = 1) 

model.fit(X_train, Y_train, sample_weight = W_train, verbose = True)
model.save_model(model_dir+'best_LV_'+met_reg+'_MET.txt')