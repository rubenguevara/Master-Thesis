import ROOT as R
import os
from Event_IDs import IDs
from Plot_Maker import Plot_Maker

rootdir = '../EventSelector/Histograms/mc16e'
BigDic = {}; variables = []
Types = ["DY_all", "Top", "Diboson", "W", "TTbar", "Signal", "Data"]   # First four are back


for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        c = os.path.join(subdir, file)
        myfile = R.TFile.Open(c)
        for variable in myfile.GetListOfKeys():
            variables.append(variable.GetName())
        break

for type in Types:
    BigDic[type] = {}
    for variable in variables:
            BigDic[type][variable] = []


filelist = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        dsid = int(file.split('.')[-3])
        c = os.path.join(subdir, file)
        myfile = R.TFile.Open(c)
        filelist.append(myfile)
        for variable in variables:
            if dsid in IDs["DY_all"]: 
                BigDic["DY_all"][variable].append(myfile.Get(variable))
                print('You have DY!')
            elif dsid in IDs['Top']: 
                BigDic["Top"][variable].append(myfile.Get(variable))
            elif dsid in IDs['TTbar_dil']: 
                BigDic["TTbar"][variable].append(myfile.Get(variable))
            elif dsid in IDs['Diboson']: 
                BigDic["Diboson"][variable].append(myfile.Get(variable))
            elif dsid in IDs['W']: 
                BigDic["W"][variable].append(myfile.Get(variable))
            elif dsid in IDs['sig']: 
                BigDic["Signal"][variable].append(myfile.Get(variable))
            elif dsid in IDs['all_data']: 
                BigDic["Data"][variable].append(myfile.Get(variable))
                print('you got data')

types = {}

Backgrounds = ['TTbar', "W", "Diboson", 'Top', 'DY_all']

Colors = {}
Colors["DY_all"] = R.TColor.GetColor('#D4F4EC')
Colors["Top"] = R.TColor.GetColor('#FFD8C5')
Colors["Diboson"] = R.TColor.GetColor('#FEA889')
Colors["W"] = R.TColor.GetColor('#70D0C6')
Colors["TTbar"] = R.kViolet

try:
    os.makedirs("Plots")

except FileExistsError:
    pass

stack = {}
stack2 = {}
for vari in variables:
    lep = vari.split('_')[1]
    hist = vari.split('_')[-1]
    if hist == 'sig':
        hist = vari.split('_')[-2]+'_'+hist
    
    legend = R.TLegend(0.64, 0.62, 0.84, 0.87)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.04)
    
    stack[vari] = R.THStack()
    stack2[vari] = R.THStack()
    test = {}
    for i in Backgrounds:
        #print('---'*30)
        #print('Background:',i)
        if BigDic[i][vari] == []: continue  # No event in background!
        test[i] = R.TH1D(BigDic[i][vari][0])
        #print('Entries from data:', BigDic[i][vari][0].GetEntries())
        #print(0,test[i].GetEntries())
        for j in range(1,len(BigDic[i][vari])):
            test[i].Add(BigDic[i][vari][j])
        #    print('Entries from data:', BigDic[i][vari][j].GetEntries())
        #    print(j,test[i].GetEntries())
        if i == 'Signal':    
            test[i].SetFillColor(R.kWhite)
            test[i].SetLineColor(Colors[i])
            test[i].SetLineStyle(10)
            stack2[vari].Add(test[i])
        else:    
            test[i].SetFillColor(Colors[i])
            test[i].SetLineColor(Colors[i])
            stack[vari].Add(test[i])
            
        if not (BigDic[i][vari]==[]): 
            legend.AddEntry(test[i],i)
    Plot_Maker(stack[vari], legend, lep, hist, logy=True)
