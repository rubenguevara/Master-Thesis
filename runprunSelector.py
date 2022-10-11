import os 

mc_cmps = ["mc16a", "mc16d", "mc16e"]
data_cmps =['data15', 'data16', 'data17', 'data18']

for mc_cmp in mc_cmps: 
    os.system("python prunSelector.py --data "+mc_cmp+" --bkgs all_bkg --ml_file ZMET-"+mc_cmp)# --isHepp01 1 --doSyst 1")

for data_cmp in data_cmps: 
    os.system("python prunSelector.py --data "+data_cmp+" --bkgs "+data_cmp)# --isHepp01 1 --doSyst 1")

