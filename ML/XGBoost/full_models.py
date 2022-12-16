import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

print(xgb.__version__)

save_dir = "../../../storage/racarcam/"
filename = 'Full_DM_sig.h5'

df = pd.read_hdf(save_dir+filename, key='df_tot')

df_features = df.copy()
df_EventID = df_features.pop('EventID')
df_CrossSection = df_features.pop('CrossSection')
df_RunNumber = df_features.pop('RunNumber')
df_RunPeriod = df_features.pop('RunPeriod')
df_dPhiCloseMet = df_features.pop('dPhiCloseMet')                             # Bad variable
df_dPhiLeps = df_features.pop('dPhiLeps')                                     # Bad variable

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

print('Starting fitting')
xgbclassifier.fit(X_train, Y_train, sample_weight = W_train, verbose = True)#, eval_set = (X_test, Y_test) ) 

model_dir = 'Models/XGB/'
try:
    os.makedirs(model_dir)

except FileExistsError:
    pass
xgbclassifier.save_model(model_dir+'FULL.txt')

# To load
# model_xgb = xgb.XGBClassifier()
# model_xgb.load_model(model_dir+'FULL.txt')

# y_pred = xgbclassifier.predict(X_test)
y_pred_prob = xgbclassifier.predict_proba(X_test)

### Plotting

plot_dir = 'Plots_XGBoost/FULL/'

try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass


plt.figure(1, figsize=[10,6])
n, bins, patches = plt.hist(y_pred_prob[:,1][Y_test==0], weights = W_test[Y_test==0], bins = 100, facecolor='blue', alpha=0.2,label="Background")
n, bins, patches = plt.hist(y_pred_prob[:,1][Y_test==1], weights = W_test[Y_test==1], bins = 100, facecolor='red', alpha=0.2, label="Signal")
plt.xlabel('XGBoost output')
plt.ylabel('Events')
plt.yscale('log')
plt.xlim([0,1])
plt.title('XGBoost output, FULL DM dataset, weighted validation data')
plt.grid(True)
plt.legend()
plt.savefig(plot_dir+'VAL_w.pdf')
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
plt.title('ROC for XGBoost on FULL DM dataset')
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
plt.title('XGBoost output, FULL DM dataset, unweighted validation data')
plt.grid(True)
plt.legend()
plt.savefig(plot_dir+'VAL_uw.pdf')
plt.show()


fig, ax = plt.subplots(1, 1, figsize = [10, 6])
xgb.plot_importance(booster = xgbclassifier, ax=ax)
plt.savefig(plot_dir+'features.pdf')
plt.show()