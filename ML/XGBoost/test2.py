import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np
print(xgb.__version__)

save_dir = "/storage/racarcam/"
filename = 'FULL_Zp_FINAL.h5'

df = pd.read_hdf(save_dir+filename, key='df_tot')

extra_variables = ['dPhiCloseMet', 'dPhiLeps', 'n_bjetPt20', 'n_ljetPt40', 'jetEtaCentral', 'jetEtaForward50']

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_Dileptons = df_features.pop('Dileptons')
df_RunPeriod = df_features.pop('RunPeriod')
df_features = df_features.drop(extra_variables, axis=1)

df_labels = df_features.pop('Label')

X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
W_train = abs(X_train.pop('Weight'))
W_test = X_test.pop('Weight')

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

xgbclassifier.fit(X_train, Y_train, sample_weight = W_train, verbose = True)#, eval_set = (X_test, Y_test) ) 

y_pred_prob = xgbclassifier.predict_proba(X_test)

### Plotting
plot_dir = '../../Plots/Plot_types/'
try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass


plt.figure(1, figsize=[10,6])
n, bins, patches = plt.hist(y_pred_prob[:,1][Y_test==0], bins = 100, facecolor='blue', alpha=0.2,label="Background")
n, bins, patches = plt.hist(y_pred_prob[:,1][Y_test==1], bins = 100, facecolor='red', alpha=0.2, label="Signal")
plt.xlabel('ML output')
plt.ylabel('Events')
plt.yscale('log')
plt.xlim([0,1])
# plt.title('XGBoost output, FULL DM dataset, weighted validation data')
plt.grid(True)
plt.legend()
plt.savefig(plot_dir+'VAL_unscaled.pdf')
plt.show()

plt.figure(2, figsize=[10,6])
bkg_pred, bins, patches = plt.hist(y_pred_prob[:,1][Y_test==0], weights = W_test[Y_test==0], bins = 100, facecolor='blue', alpha=0.2,label="Background")
sig_pred, bins, patches = plt.hist(y_pred_prob[:,1][Y_test==1], weights = W_test[Y_test==1], bins = 100, facecolor='red', alpha=0.2, label="Signal")
plt.xlabel('ML output')
plt.ylabel('Events')
plt.yscale('log')
plt.xlim([0,1])
plt.grid(True)
plt.legend()
plt.savefig(plot_dir+'VAL.pdf')
plt.show()

fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob[:,1], pos_label=1)
roc_auc = auc(fpr,tpr)
plt.figure(3)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig(plot_dir+'ROC.pdf')
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

X_axis = [0.50, 0.60, 0.70, 0.80, 0.9, 0.99]

plt.figure(4, figsize=[10,6])
plt.plot(X_axis, Y_axis, linestyle='--')
plt.scatter(X_axis, Y_axis, label = '$\sigma$ cut')
plt.xlim([0,1])
plt.grid(True)
plt.legend()
plt.ylabel('Expected significance [$\sigma$]')
plt.xlabel('ML output')
plt.savefig(plot_dir+'EXP_SIG.pdf')
plt.show()