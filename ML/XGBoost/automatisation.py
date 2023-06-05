import os, time

t0 = time.time()
start = time.asctime(time.localtime())
print('Started', start)
print('==='*40)

model_cmp = ['LV_HDS', 'LV_LDS', 'DH_HDS', 'DH_LDS', 'EFT_HDS', 'EFT_LDS', 'SlepSlep', 'Stop']
for reg in model_cmp: 
    os.system("python3 model_dependent_training.py --dm_model "+reg)
    os.system("python3 test_one_model_new.py --dm_model "+reg+' --channel ee')
    os.system("python3 test_one_model_new.py --dm_model "+reg+' --channel uu')


reg_cmp = ['50-100', '100-150', '150']
for reg in reg_cmp: 
    os.system("python3 GRID_SEARCH.py --met_reg "+reg)
    os.system("python3 GRID_SEARCH_plot.py --met_reg "+reg)

for reg in reg_cmp: 
    for model in model_cmp:
        os.system('python3 model_independent_test_one_model.py --met_reg '+reg+' --dm_model '+model+' --channel ee')
        os.system('python3 model_independent_test_one_model.py --met_reg '+reg+' --dm_model '+model+' --channel uu')

print('==='*40)
t = "{:.2f}".format(int( time.time()-t0 )/60.)
finish = time.asctime(time.localtime())
print('Finished', finish)
print('Total time:', t)
print('==='*40)

"""
To use do 
nohup python3 lazy2.py &
"""