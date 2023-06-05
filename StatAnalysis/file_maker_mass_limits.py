import json, os, re, argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dm_model', type=str, default="DH_HDS", help="Dataset to test")
args = parser.parse_args()

dm_model = args.dm_model

json_file1 = open(dm_model+"_effective_xs.json")
effective_xs = json.load(json_file1)

json_file2 = open(dm_model+"_events_before.json")
events_before = json.load(json_file2)

json_file1.close()
json_file2.close()

directories = [] 
files = 0 
save_path = 'Data/'+dm_model+'_scores/'
for (dirpath, dirnames, filenames) in os.walk(save_path):
    if filenames == []: continue
    directories.append(dirpath)
    files = filenames
    continue

def sort_alphanum(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l

directories = sort_alphanum(directories)
files = sort_alphanum(filenames)

filenames = []
for f in files:
    if 'uu' in f: 
        filenames.append(f)
    elif 'ee' in f: 
        filenames.append(f)

try:
    os.remove(dm_model+".txt")
    print('Updating old file')
except:
    print('Nothing to update')
    
text_file = open(dm_model+'.txt', 'w')
text_file.write('NmassPoints='+str(int(len(directories)))+';  Nchannels=2;  intLum=139;  intLumUncertainty=1.12; \n')

tex_file = open(dm_model+'_LATEX.txt', 'w')
tex_file.write('\midrule\midrule \n')
tex_file.write("$m_{Z'}$ [GeV] & $\sigma B$ [fb] & Channel & $\\varepsilon_{\\text{sig}}$ $[\\times10]$& $N_{\\text{sig}}$ & $N_{\\text{bkg}}$ & $N_{\\text{obs}}$ \\\\\midrule\midrule\n")


dirpath = 0
for dirpath in directories:
    bkg_after_cut_ee = np.load(dirpath+'/'+filenames[0])[-1]
    bkg_after_cut_uu = np.load(dirpath+'/'+filenames[1])[-1]
    data_pred_ee = int(np.load(dirpath+'/'+filenames[2])[-1])
    data_pred_uu = int(np.load(dirpath+'/'+filenames[3])[-1])
    signal_after_cut_ee = np.load(dirpath+'/'+filenames[4])[-1]
    signal_after_cut_uu = np.load(dirpath+'/'+filenames[5])[-1]
    bkg_after_cut_unc_ee = np.load(dirpath+'/'+filenames[6])[-1]
    bkg_after_cut_unc_uu = np.load(dirpath+'/'+filenames[7])[-1]
    signal_after_cut_unc_ee = np.load(dirpath+'/'+filenames[8])[-1]
    signal_after_cut_unc_uu = np.load(dirpath+'/'+filenames[9])[-1]
    
    efficiency_ee = signal_after_cut_ee/events_before[dirpath.split('/')[-1]] * 0.5
    efficiency_uu = signal_after_cut_uu/events_before[dirpath.split('/')[-1]] * 0.5
    
    text_file.write('mass='+dirpath.split('_')[-1]+';  threshold=110;  theoryCrossSection='+str(effective_xs[dirpath.split('/')[-1]])+';\n')
    
    text_file.write('channel="electron";   efficiency='+str(efficiency_ee)+';  efficiencyUncertainty='+str(efficiency_ee*0.2)+
                    ';  background='+str(bkg_after_cut_ee)+';  backgroundUncertainty='+str(bkg_after_cut_unc_ee)+';  Nobs='+str(data_pred_ee)+';\n')
    
    text_file.write('channel="muon";   efficiency='+str(efficiency_uu)+';  efficiencyUncertainty='+str(efficiency_uu*0.2)+
                    ';  background='+str(bkg_after_cut_uu)+';  backgroundUncertainty='+str(bkg_after_cut_unc_uu)+';  Nobs='+str(data_pred_uu)+';\n')
    
    
    tex_file.write("\multirow{2}{*}[-2\\baselineskip]{%g}& \multirow{2}{*}[-2\\baselineskip]{$%.2e$}& $ee$ & $%.2f\pm%.2f$ & $%.2e\pm%.2e$ & $%.1f\pm%.1f$ & %g\\\\ \n" %(int(dirpath.split('_')[-1]), effective_xs[dirpath.split('/')[-1]], efficiency_ee*10, efficiency_ee*0.2*10, signal_after_cut_ee, signal_after_cut_unc_ee, bkg_after_cut_ee, bkg_after_cut_unc_ee, int(data_pred_ee)))   
    tex_file.write("& & $\mu\mu$ & $%.2f\pm%.2f$ & $%.2e\pm%.2e$ & $%.1f\pm%.1f$ & %g\\\\ \midrule\n" %(efficiency_uu*10, efficiency_uu*0.2*10, signal_after_cut_uu, signal_after_cut_unc_uu, bkg_after_cut_uu, bkg_after_cut_unc_uu, int(data_pred_uu)))

tex_file.write('\midrule\midrule')
text_file.write('xtitle="DM mediator mass [GeV]"; ytitle="Cross section [fb]"; yrange=['+str(effective_xs[dm_model+'_mZp_1500']*0.5)+','+str(effective_xs[dm_model+'_mZp_130']*2)+'];')

