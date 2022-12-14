import ROOT as R
import os, sys, json
from EventIDs import IDs
from Plot_Maker import Plot_Maker
from SOW import SOW_bkg, SOW_sig_AFII#, SOW_AFII, SOW_sig_AFII
from collections import OrderedDict


rootdir = '../EventSelector/Histograms_SUSY'
BigDic = {}; variables = []; dsid_list = {}
Types = ["Drell Yan", 'Single Top', "Diboson", "W", "TTbar", "Signal"]
mc_year = ['mc16a']#, 'mc16d', 'mc16e']


for subdir, dirs, files in os.walk(rootdir+'/mc16a'):
    for file in files:
        if '50' in file: continue
        c = os.path.join(subdir, file)
        myfile = R.TFile.Open(c)
        for variable in myfile.GetListOfKeys():
            variables.append(variable.GetName())
        break

for mc in mc_year:
    BigDic[mc] = {}
    dsid_list[mc] = {}
    for type in Types:
        BigDic[mc][type] = {}
        dsid_list[mc][type] = []
        for variable in variables:
                BigDic[mc][type][variable] = []

# DM_MODEL = 514619  # Write dsid of specific model
# sig_var = 'dm_sig'
sig_var = 'SUSY'
filelist = []
exclude = set(['data'])
for subdir, dirs, files in os.walk(rootdir):
    dirs[:] = [d for d in dirs if d not in exclude]
    for file in files:
        dsid = int(file.split('.')[1])
        c = os.path.join(subdir, file)
        myfile = R.TFile.Open(c)
        filelist.append(myfile)
        mc_run = subdir.split('/')[-1]
        AFII = c.split('.')[-2].split('_')[-1]
        if AFII =='AFII': 
            print('Skipping AFII file:',c)
            continue
        if dsid in IDs["DY"]: 
            dsid_list[mc_run]['Drell Yan'].append(str(dsid))
            for variable in variables:
                BigDic[mc_run]["Drell Yan"][variable].append(myfile.Get(variable))
        elif dsid in IDs['Single_top']: 
            dsid_list[mc_run]['Single Top'].append(str(dsid))
            for variable in variables:
                BigDic[mc_run]['Single Top'][variable].append(myfile.Get(variable))
        elif dsid in IDs['TTbar']:
            dsid_list[mc_run]['TTbar'].append(str(dsid))
            if AFII =='AFII': print('here')
            for variable in variables: 
                BigDic[mc_run]["TTbar"][variable].append(myfile.Get(variable))
        elif dsid in IDs['Diboson']: 
            dsid_list[mc_run]['Diboson'].append(str(dsid))
            for variable in variables:
                BigDic[mc_run]["Diboson"][variable].append(myfile.Get(variable))
        elif dsid in IDs['W']: 
            dsid_list[mc_run]['W'].append(str(dsid))
            for variable in variables:
                BigDic[mc_run]["W"][variable].append(myfile.Get(variable))
        elif dsid in IDs[sig_var]:
            # if not dsid == DM_MODEL: continue
            dsid_list[mc_run]['Signal'].append(str(dsid))
            for variable in variables:
                BigDic[mc_run]["Signal"][variable].append(myfile.Get(variable))

data = {}

for variable in variables:
    data[variable] = []

rootdird = rootdir +'/data'
for subdir, dirs, files in os.walk(rootdird):
    for file in files:
        c = os.path.join(subdir, file)
        myfile = R.TFile.Open(c)
        filelist.append(myfile)
        for variable in variables:
            data[variable].append(myfile.Get(variable))
            
sow_susy_file = open('SOW_SUSY.json')
SOW_SUSY = json.load(sow_susy_file)
SOW_a = OrderedDict(list(SOW_bkg['mc16a'].items()) + list(SOW_SUSY['mc16a'].items()) )
# SOW_a = OrderedDict(list(SOW_bkg['mc16a'].items()) + list(SOW_sig_AFII['mc16a'].items()) )
SOW_d = OrderedDict(list(SOW_bkg['mc16d'].items()) + list(SOW_sig_AFII['mc16d'].items()) )
SOW_e = OrderedDict(list(SOW_bkg['mc16e'].items()) + list(SOW_sig_AFII['mc16e'].items()) )
sow_susy_file.close()

Backgrounds = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan', 'Signal']

Colors = {}
Colors["Signal"] = R.TColor.GetColor('#F42069')
Colors["Drell Yan"] = R.TColor.GetColor('#8EDC9D')
Colors['Single Top'] = R.TColor.GetColor('#EF7126')
Colors["TTbar"] = R.TColor.GetColor('#F9E559')
Colors["Diboson"] = R.TColor.GetColor('#6CCECB')
Colors["W"] = R.TColor.GetColor('#218C8D')

save_dir = "..Plots/Data_Analysis/50MET_FULL_SUSY"
try:
    os.makedirs(save_dir)

except FileExistsError:
    pass

stack = {}
stack2 = {}
lep = {}
hist = {}
charge = {}

thist = {}


for vari in variables:
    lep[vari] = vari.split('_')[1]
    charge[vari] = vari.split('_')[2]
    hist[vari] = vari.split('_')[-1]
    if hist[vari] == 'sig':
        hist[vari] = vari.split('_')[-2]+'_'+hist[vari]
    
    stack[vari] = R.THStack()
    stack2[vari] = R.THStack()
    thist[vari] = {}
    for mc in mc_year:
        thist[vari][mc] = {}
        for i in Backgrounds:
            if BigDic[mc][i][vari] == []: 
                print('No events in',i,' on ',mc,'!')
                continue  
                
            id = dsid_list[mc][i][0]
            if mc == 'mc16a':
                w = 36.2/SOW_a[id]
                
            elif mc == 'mc16d':
                w = 44.3/SOW_d[id]
                
            elif mc == 'mc16e':
                w = 58.5/SOW_e[id]
            
            thist[vari][mc][i] = R.TH1D(BigDic[mc][i][vari][0])
            thist[vari][mc][i].Scale(w)
                        
            for j in range(1,len(BigDic[mc][i][vari])): 
                id = dsid_list[mc][i][j]
                if mc == 'mc16a':
                    w = 36.2/SOW_a[id]
                    
                elif mc == 'mc16d':
                    w = 44.3/SOW_d[id]
                    
                elif mc == 'mc16e':
                    w = 58.5/SOW_e[id]
                    
                thist[vari][mc][i].Add(BigDic[mc][i][vari][j], w)


data_hist = {}
for vari in variables:
    data_hist[vari] = R.TH1D(data[vari][0])
    for i in range(1,len(data[vari])):
        data_hist[vari].Add(data[vari][i])


for vari in variables:
    legend = R.TLegend(0.60, 0.62, 0.90, 0.87)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.04)
    legend.SetNColumns(2)
    data_hist[vari].SetMarkerStyle(20)
    data_hist[vari].SetLineColor(R.kBlack)
    for bkg in Backgrounds:
        for mc in mc_year:
            if bkg == 'Signal':    
                thist[vari][mc][bkg].SetFillStyle(0)
                thist[vari][mc][bkg].SetLineWidth(2)
                thist[vari][mc][bkg].SetLineColor(Colors[bkg])
                thist[vari][mc][bkg].SetLineStyle(1)
                stack2[vari].Add(thist[vari][mc][bkg])
                
            else:   
                thist[vari][mc][bkg].SetFillColor(Colors[bkg])
                thist[vari][mc][bkg].SetLineColor(Colors[bkg])
                stack[vari].Add(thist[vari][mc][bkg])      
        legend.AddEntry(thist[vari][mc][bkg], bkg)
    legend.AddEntry(data_hist[vari], 'Data')
    Plot_Maker(stack[vari], legend, lep[vari], charge[vari], hist[vari], data_hist[vari], save_dir, stack2[vari])
