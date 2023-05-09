# import xgboost as xgb
# import numpy as np

# X = np.random.rand(10,1)
# y = np.random.rand(10,1)

# xgb_model = xgb.XGBRegressor(
#     tree_method="gpu_hist"
#     )

# xgb_model.fit(X, y)


from Plot_maker import scatter_plot, distribution_MC, distribution_data
import numpy as np
import pandas as pd
import os, itertools
import matplotlib.pyplot as plt
import multiprocessing as mp

wonky_bin_dir = '../Data/wonky-bin/'
data_pred = np.load(wonky_bin_dir+'data_pred.npy')
pred = np.load(wonky_bin_dir+'pred.npy')
X_test = pd.read_pickle(wonky_bin_dir+'X_test.pkl')
Y_test = np.load(wonky_bin_dir+'Y_test.npy')
W_test = np.load(wonky_bin_dir+'W_test.npy')
DSID_test = np.load(wonky_bin_dir+'DSID_test.npy')
df_data = pd.read_pickle(wonky_bin_dir+'df_data.pkl')

plot_dir = '../../Plots/XGBoost/Re-weighted/DH_150_MET/variables/Data/'
# plot_dir = '../../Plots/XGBoost/Re-weighted/DH_150_MET/scatter-plots/MC/'
try:
    os.makedirs(plot_dir)

except FileExistsError:
    pass

variables = df_data.columns
# data_bool = [i>0.3 for i in data_pred]
# data = df_data[data_bool]
for v in variables:
    distribution_data(df_data, v, plot_dir, 0.3, data_pred)
    plt.close('all')

# pairs = set(itertools.combinations(variables, 2))

# def plot(v):
#     # scatter_plot(X_test, pred, 0.3, couples[0], couples[1], plot_dir, do_weights=True, weights=W_test)
#     distribution_MC(X_test, W_test, DSID_test, v, plot_dir, 0.3, pred)
#     plt.close('all')

# with mp.Pool(processes=len(variables)) as pool:
#     pool.map(plot, variables)
# pool.close()

# for p in pairs:
#     scatter_plot(df_data, data_pred, 0.3, p[0], p[1], plot_dir)
#     plt.close('all')

# x_ax = np.linspace(120, 1500, 21)
# y_ax = np.linspace(150, 1000, 21)
# scatter_plot(df_data, data_pred, 0.3, 'lep2Pt', 'lep1Pt', plot_dir, x_axis_1=x_ax, x_axis_2=y_ax)