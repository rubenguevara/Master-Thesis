import ROOT
from ROOT import *
import sys, os, time, argparse, shutil 
import multiprocessing as mp

save_dir = "ML_files"
# Merge histograms 
out_path = save_dir+"/"
working_dir = os.getcwd() 
os.chdir(out_path) 

file_a = ""; file_d = ""; file_e = ""
outfile_a = ""; outfile_d = ""; outfile_e = ""
for file in os.listdir("."):
    if ".root" in file.split("-")[1]: 
        print(file)
        continue
    type = file.split('-')[0]
    if type == 'DELETE':
        os.remove(file)
    mcRun = file.split("-")[1]
    extra = "-" + file.split("-")[2] + "-" + file.split("-")[3]
    outfile = file.replace(extra, ".root")
    if mcRun == 'mc16a':
        outfile_a = file.replace(extra, ".root")
        file_a += (" "+file)
    if mcRun == 'mc16d':
        outfile_d = file.replace(extra, ".root")
        file_d += file
    if mcRun == 'mc16e':
        outfile_d = file.replace(extra, ".root")
        file_d += (" "+file)

if outfile_a!="":
    if os.path.exists(outfile_a): 
        os.remove(outfile_a)
    else: 
        os.system("hadd "+outfile_a+file_a)

#shutil.rmtree(file)

os.chdir(working_dir) 