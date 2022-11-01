import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

save_dir = "../../../storage/racarcam/"
filename = "Stat_red_DM1_Run2_50MET.h5"

df = pd.read_hdf(save_dir+filename, key='df_tot')

X = df.copy()
df_EventNumber = X.pop('EventNumber')
W = np.abs(np.array(X.pop('weight')*0.2))
# df_CrossSection = X.pop('CrossSection')
df_RunNumber = X.pop('RunNumber')
df_RunPeriod = X.pop('RunPeriod')

Y = np.array(X.pop('Label'))
X = np.array(X)
size = int(len(Y)*0.2)

X_train = np.concatenate((X[Y==0][0:size],X[Y==1][0:size]), axis=0)
w_train = np.concatenate((W[Y==0][0:size],W[Y==1][0:size]), axis=0)
y_train = np.concatenate((Y[Y==0][0:size],Y[Y==1][0:size]), axis=0)
X_valid = np.concatenate((X[Y==0][size:size*2],X[Y==1][size:size*2]), axis=0)
y_valid = np.concatenate((Y[Y==0][size:size*2],Y[Y==1][size:size*2]), axis=0)
X_eval = np.concatenate((X[Y==0][size*2:size*3],X[Y==1][size*2:size*3]), axis=0)
y_eval = np.concatenate((Y[Y==0][size*2:size*3],Y[Y==1][size*2:size*3]), axis=0)
# Get the sums of the weights for signal and background events 
sum_wsig = sum( w_train[i] for i in range(len(y_train)) if y_train[i] == 1.0  )
sum_wbkg = sum( w_train[i] for i in range(len(y_train)) if y_train[i] == 0.0  )


xgbclassifier = xgb.XGBClassifier(
    max_depth=3, 
    n_estimators=120,
    learning_rate=0.1,
    #n_jobs=4,
    scale_pos_weight=sum_wbkg/sum_wsig,
    objective='binary:logistic',
    missing=10) 

xgbclassifier.fit(X_train, y_train, w_train) 

# Test the BDT performance using the validation dataset
y_pred = xgbclassifier.predict( X_valid ) # The actual signal/background predictions. Note that we don't actually use them
y_pred_prob = xgbclassifier.predict_proba( X_valid ) # The BDT outputs for each event

#  histogram of the BDT outputs
n, bins, patches = plt.hist(y_pred_prob[:,1][y_valid==0], 200, facecolor='blue', alpha=0.2,label="Background")
n, bins, patches = plt.hist(y_pred_prob[:,1][y_valid==1], 200, facecolor='red', alpha=0.2, label="Signal")


plt.figure(1)
plt.xlabel('XGBoost output')
plt.ylabel('Events')
plt.title('XGBoost output, DM dataset, validation data')
plt.grid(True)
plt.legend()
plt.savefig('Plots/VAL.pdf')
plt.show()

fpr, tpr, thresholds = roc_curve(y_valid,y_pred_prob[:,1], pos_label=1)
roc_auc = auc(fpr,tpr)
plt.figure(2)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for xgBoost on DM dataset')
plt.legend(loc="lower right")
plt.savefig('Plots/ROC.pdf')
plt.show()


plt.figure(3)
xgb.plot_importance(xgbclassifier)
plt.rcParams["figure.figsize"] = [15,15]
plt.savefig('Plots/feature.pdf')
plt.show()