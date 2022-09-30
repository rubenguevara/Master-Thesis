import ROOT as R
import os
from EventIDs import IDs
from Plot_Maker import Plot_Maker
from SOW import SOW_bkg, SOW_sig, SOW_AFII, SOW_sig_AFII
from collections import OrderedDict


rootdir = '../EventSelector/Histograms'
BigDic = {}; variables = []; dsid_list = {}
Types = ["Drell Yan", 'Single Top', "Diboson", "W", "TTbar", "Signal", "Data"]   # First four are back
mc_year = ['mc16a', 'mc16d', 'mc16e']


for subdir, dirs, files in os.walk(rootdir+'/mc16a'):
    for file in files:
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

filelist = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        dsid = int(file.split('.')[-3])
        c = os.path.join(subdir, file)
        myfile = R.TFile.Open(c)
        filelist.append(myfile)
        mc_run = subdir.split('/')[-1]
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
        elif dsid in IDs['sig']: 
            dsid_list[mc_run]['Signal'].append(str(dsid))
            for variable in variables:
                BigDic[mc_run]["Signal"][variable].append(myfile.Get(variable))
        elif dsid in IDs['all_data']: 
            for variable in variables:
                BigDic[mc_run]["Data"][variable].append(myfile.Get(variable))
                print('you got data')

SOW_a = OrderedDict(list(SOW_bkg['mc16a'].items()) + list(SOW_AFII['mc16a'].items()) + list(SOW_sig['mc16a'].items()) + list(SOW_sig_AFII['mc16a'].items()) )
SOW_d = OrderedDict(list(SOW_bkg['mc16d'].items()) + list(SOW_AFII['mc16d'].items()) + list(SOW_sig['mc16d'].items()) + list(SOW_sig_AFII['mc16d'].items()) )
SOW_e = OrderedDict(list(SOW_bkg['mc16e'].items()) + list(SOW_AFII['mc16e'].items()) + list(SOW_sig['mc16e'].items()) + list(SOW_sig_AFII['mc16e'].items()) )

Backgrounds = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan']

Colors = {}
Colors["Drell Yan"] = R.TColor.GetColor('#8EDC9D')
Colors['Single Top'] = R.TColor.GetColor('#EF7126')
Colors["TTbar"] = R.TColor.GetColor('#F9E559')
Colors["Diboson"] = R.TColor.GetColor('#6CCECB')
Colors["W"] = R.TColor.GetColor('#218C8D')


try:
    os.makedirs("Plots")

except FileExistsError:
    pass

skipper = False
stack = {}
stack2 = {}
lep = {}
hist = {}

thist = {}
for vari in variables:
    lep[vari] = vari.split('_')[1]
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
                skipper = True
                print('No events in',i,' on ',mc,'!')
                continue  
                
            id = dsid_list[mc][i][0]
            if mc == 'mc16a':
                w = 1/SOW_a[id]
                
            elif mc == 'mc16d':
                w = 1/SOW_d[id]
            
            elif mc == 'mc16e':
                w = 1/SOW_e[id]
            
            thist[vari][mc][i] = R.TH1D(BigDic[mc][i][vari][0])
            thist[vari][mc][i].Scale(w)
        
            for j in range(1,len(BigDic[mc][i][vari])):
                id = dsid_list[mc][i][j]
                if mc == 'mc16a':
                    w = 1/SOW_a[id]
                    
                elif mc == 'mc16d':
                    w = 1/SOW_d[id]
                
                elif mc == 'mc16e':
                    w = 1/SOW_e[id]
                    
                thist[vari][mc][i].Add(BigDic[mc][i][vari][j], w)
    
for vari in variables:
    legend = R.TLegend(0.64, 0.62, 0.84, 0.87)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.04)
    for bkg in Backgrounds:
        for mc in mc_year:   
            thist[vari][mc][bkg].SetFillColor(Colors[bkg])
            thist[vari][mc][bkg].SetLineColor(Colors[bkg])
            stack[vari].Add(thist[vari][mc][bkg])
                
        legend.AddEntry(thist[vari][mc][bkg], bkg)
    Plot_Maker(stack[vari], legend, lep[vari], hist[vari], logy=True)