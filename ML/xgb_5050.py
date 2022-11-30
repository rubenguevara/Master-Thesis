import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

print(xgb.__version__)


""" Choose wether to train on "even" of "odd" dataset """
do = 'odd'                                                                 

save_dir = "../../../storage/racarcam/"
filename = 'Full_DM_sig.h5'

df = pd.read_hdf(save_dir+filename, key='df_tot')

df_features = df.copy()
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                             # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')                                     # Bad variable

X_even = df_features.iloc[np.where(df_features['EventID']%2 == 0)]
X_odd = df_features.iloc[np.where(df_features['EventID']%2 != 0)]

df_eventid_even = X_even.pop('EventID')
df_eventid_odd = X_odd.pop('EventID')

Y_even = X_even.pop('Label')
Y_odd = X_odd.pop('Label')

W_even = X_even.pop('Weight')
W_odd = X_odd.pop('Weight')


if do == 'even':
    X_train = X_even
    Y_train = Y_even
    W_train = abs(W_even)
    
    X_test = X_odd
    Y_test = Y_odd
    W_test = W_odd

else:
    X_train = X_odd
    Y_train = Y_odd
    W_train = abs(W_odd)
    
    X_test = X_even
    Y_test = Y_even
    W_test = W_even

sum_wsig = sum(W_train[Y_train==1])
sum_wbkg = sum(W_train[Y_train==0])

xgbclassifier = xgb.XGBClassifier(
    max_depth=3, 
    use_label_encoder=False,
    n_estimators=120,
    learning_rate=0.1,
    predictor = 'cpu_predictor',
    tree_method = 'hist',
    scale_pos_weight=sum_wbkg/sum_wsig,
    objective='binary:logistic',
    # eval_metric='error@0.7',
    eval_metric='auc',
    missing=10,
    random_state=42,
    verbosity = 1) 

print('Starting fitting')
xgbclassifier.fit(X_train, Y_train, sample_weight = W_train, verbose = True)#, eval_set = (X_test, Y_test) ) 

model_dir = 'Models/XGB/Odd-Even/'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass
xgbclassifier.save_model(model_dir+'FULL_'+do+'.txt')

# To load
# model_xgb = xgb.XGBClassifier()
# model_xgb.load_model(model_dir+'FULL.txt')

# y_pred = xgbclassifier.predict(X_test)
y_pred_prob = xgbclassifier.predict_proba(X_test)

### Plotting

plot_dir = '../Plots/XGBoost/FULL/'+do+'/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass


plt.figure(1, figsize=[10,6])
bkg_pred, bins, patches = plt.hist(y_pred_prob[:,1][Y_test==0], weights = W_test[Y_test==0]*2, bins = 100, facecolor='blue', alpha=0.2,label="Background")
sig_pred, bins, patches = plt.hist(y_pred_prob[:,1][Y_test==1], weights = W_test[Y_test==1]*2, bins = 100, facecolor='red', alpha=0.2, label="Signal")
plt.xlabel('XGBoost output')
plt.ylabel('Events')
plt.yscale('log')
plt.xlim([0,1])
plt.title('XGBoost output, FULL DM dataset on '+do+' trained network, validation data')
plt.grid(True)
plt.legend()
plt.savefig(plot_dir+'VAL.pdf')
plt.show()

fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob[:,1], pos_label=1)
roc_auc = auc(fpr,tpr)
plt.figure(2)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for XGBoost on FULL DM dataseton '+do+' trained network')
plt.legend(loc="lower right")
plt.savefig(plot_dir+'ROC.pdf')
plt.show()

plt.figure(3, figsize=[10,6])
n, bins, patches = plt.hist(y_pred_prob[:,1][Y_test==0], bins = 100, facecolor='blue', alpha=0.2,label="Background")
n, bins, patches = plt.hist(y_pred_prob[:,1][Y_test==1], bins = 100, facecolor='red', alpha=0.2, label="Signal")
plt.xlabel('XGBoost output')
plt.ylabel('Events')
plt.yscale('log')
plt.xlim([0,1])
plt.title('XGBoost output, FULL DM dataset on '+do+' trained network, unscaled validation data')
plt.grid(True)
plt.legend()
plt.savefig(plot_dir+'VAL_uw.pdf')
plt.show()

def low_stat_Z(sig, bkg):
    Z = np.sqrt(2*( (sig + bkg)*np.log(1 + sig/bkg) - sig ))
    return Z

Y_axis = [low_stat_Z(sum(sig_pred[50:]), sum(bkg_pred[50:])), 
            low_stat_Z(sum(sig_pred[60:]), sum(bkg_pred[60:])), 
            low_stat_Z(sum(sig_pred[70:]), sum(bkg_pred[70:])),
            low_stat_Z(sum(sig_pred[80:]), sum(bkg_pred[80:])), 
            low_stat_Z(sum(sig_pred[90:]), sum(bkg_pred[90:])), 
            low_stat_Z(sum(sig_pred[99:]), sum(bkg_pred[99:]))]

X_axis = [50, 60, 70, 80, 90, 99]

plt.figure(4, figsize=[10,6])
plt.plot(X_axis, Y_axis, linestyle='--')
plt.scatter(X_axis, Y_axis, label = '$\sigma$ cut')
plt.xlim([0,100])
plt.grid(True)
plt.legend()
plt.ylabel('Expected significance [$\sigma$]')
plt.title('Significance, FULL DM dataset on '+do+' trained network')
plt.xlabel('TF output bin')
plt.savefig(plot_dir+'EXP_SIG.pdf')
plt.show()

fig, ax = plt.subplots(1, 1, figsize = [10, 6])
xgb.plot_importance(booster = xgbclassifier, ax=ax)
plt.savefig(plot_dir+'features.pdf')
plt.show()

