import os, time

t0 = time.time()

mc_cmps = ["mc16a", "mc16d", "mc16e"]
data_cmps =['data15', 'data16', 'data17', 'data18']

ml_file = "Run2"
bkg = "all_bkg"

for mc_cmp in mc_cmps: 
    os.system("python prunSelector.py --data "+mc_cmp+" --bkgs "+bkg+" --ml_file "+ml_file)# --isHepp01 1 --doSyst 1")

for data_cmp in data_cmps: 
    os.system("python prunSelector.py --data "+data_cmp+" --bkgs "+data_cmp+" --ml_file DELETE")# --isHepp01 1 --doSyst 1")

t = "{:.2f}".format(int( time.time()-t0 )/60.)
print( "---"*40)
print( "TOTAL time spent: "+str(t)+" min")  
print( "---"*40)

import ROOT, shutil 
from ROOT import *

# Merge nTuples
save_path = "../../../storage/racarcam/"

working_dir = os.getcwd() 
os.chdir(save_path) 
files = ""; outfiles = ""
for file in os.listdir("."):
    if "-" not in file:
        print("We aren't touching ", file)
        continue
    type = file.split('-')[0]
    extra = file.split('-')[1]
    if type == ml_file:
        outfiles = file.replace("-"+extra, ".root")
        files += (" "+file)

if outfiles!="":
        if os.path.exists(outfiles): 
                print("Final file", outfiles,"exists! Do you want to delete?")
        else: 
                os.system("hadd "+outfiles+files)

os.remove(ml_file+"-mc16a.root")
os.remove(ml_file+"-mc16d.root")
os.remove(ml_file+"-mc16e.root")
