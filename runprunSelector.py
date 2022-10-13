import os, time

t0 = time.time()

mc_cmps = ["mc16a", "mc16d", "mc16e"]
data_cmps =['data15', 'data16', 'data17', 'data18']

for mc_cmp in mc_cmps: 
    os.system("python prunSelector.py --data "+mc_cmp+" --bkgs all_bkg --ml_file Run2")# --isHepp01 1 --doSyst 1")

for data_cmp in data_cmps: 
    os.system("python prunSelector.py --data "+data_cmp+" --bkgs "+data_cmp+" --ml_file DELETE")# --isHepp01 1 --doSyst 1")

t = "{:.2f}".format(int( time.time()-t0 )/60.)
print( "---"*40)
print( "TOTAL time spent: "+str(t)+" min")  
print( "---"*40)