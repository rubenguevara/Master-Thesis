import os, time

t0 = time.time()

mc_cmps = ["mc16a", "mc16d", "mc16e"]
data_cmps =['data15', 'data16', 'data17', 'data18']

sig = 1
data = 0
# data_file = "datafrfr"
data_file = "DELETE"
"""
CANNOT USE UNDERSCORE ON FILE NAMES
"""
if sig == 1:
    # bkg = "SUSY"
    # ml_file = 'SUSY'
    ml_file = 'DELETE'
    bkg = "dm_sig"
    # ml_file = "monoZp"
    
else:
    ml_file = "Run2frfr"
    ml_file = "DELETE"
    bkg = "all_bkg"

for mc_cmp in mc_cmps: 
    os.system("python3 prunSelector.py --data "+mc_cmp+" --bkgs "+bkg+" --ml_file "+ml_file)#+" |& tee -a log_"+mc_cmp+".out")

if data == 1:
    for data_cmp in data_cmps: 
        os.system("python3 prunSelector.py --data "+data_cmp+" --bkgs "+data_cmp+" --ml_file "+data_file)

t = "{:.2f}".format(int( time.time()-t0 )/60.)
print( "---"*40)
print( "TOTAL time spent: "+str(t)+" min")  
print( "---"*40)

import ROOT
from ROOT import *

# Merge nTuples
save_path = "../../../storage/racarcam/"

working_dir = os.getcwd() 
os.chdir(save_path) 
files_mc = ""; outfiles_mc = ""
files_d = ""; outfiles_d = ""
for file in os.listdir("."):
    if "-" not in file:
        print("We aren't touching ", file)
        continue
    type = file.split('-')[0]
    extra = file.split('-')[1]
    if type == ml_file:
        outfiles_mc = file.replace("-"+extra, ".root")
        files_mc += (" "+file)
        
    if 'data' in type:
        outfiles_d = file.replace("-"+extra, ".root")
        files_d += (" "+file)

if outfiles_mc!="":
        if os.path.exists(outfiles_mc): 
                print("Final file", outfiles_mc,"exists! Do you want to delete?")
        else: 
                os.system("hadd "+outfiles_mc+files_mc)

if outfiles_d!="":
        if os.path.exists(outfiles_d): 
                print("Final file", outfiles_d,"exists! Do you want to delete?")
        else: 
                os.system("hadd "+outfiles_d+files_d)

os.remove(ml_file+"-mc16a.root")
os.remove(ml_file+"-mc16d.root")
os.remove(ml_file+"-mc16e.root")

if data == 1: 
    os.remove(data_file+"-data15.root")
    os.remove(data_file+"-data16.root")
    os.remove(data_file+"-data17.root")
    os.remove(data_file+"-data18.root")
