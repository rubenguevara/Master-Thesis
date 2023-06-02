import ROOT as R
import os, sys, json
from EventIDs import IDs
from Plot_MakerSR import Plot_Maker
from SOW import SOW_bkg, SOW_sig_AFII#, SOW_AFII, SOW_sig_AFII
from collections import OrderedDict


rootdir = '../EventSelector/Histograms'
# rootdir = '../EventSelector/Histograms_jet_cuts'
susydir = '../EventSelector/Histograms_SUSY'
BigDic = {}; variables = []; dsid_list = {}
Types = ["Drell Yan", 'Single Top', "Diboson", "W", "TTbar", "DH HDS m_{Z'} 130", "LV HDS m_{Z'} 130", "EFT HDS m_{Z'} 130", 'SUSY (m_{#tilde{l}}, m_{#tilde{#chi}_{1}^{0}}) = (90, 1)', '2HDMa (m_{a}, m_{H^{-}}) = (150, 400)']
mc_year = ['mc16a', 'mc16d', 'mc16e']


for subdir, dirs, files in os.walk(rootdir+'/mc16a'):
    for file in files:
        # if '50' in file: continue
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
sig_var = 'SUSY'
filelist = []
# filelist2 = []
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
                histogram = myfile.Get(variable)
                histogram.SetDirectory(0)
                BigDic[mc_run]["Drell Yan"][variable].append(histogram)
        elif dsid in IDs['Single_top']: 
            dsid_list[mc_run]['Single Top'].append(str(dsid))
            for variable in variables:
                histogram = myfile.Get(variable)
                histogram.SetDirectory(0)
                BigDic[mc_run]['Single Top'][variable].append(histogram)
        elif dsid in IDs['TTbar']:
            dsid_list[mc_run]['TTbar'].append(str(dsid))
            if AFII =='AFII': print('here')
            for variable in variables: 
                histogram = myfile.Get(variable)
                histogram.SetDirectory(0)
                BigDic[mc_run]["TTbar"][variable].append(histogram)
        elif dsid in IDs['Diboson']: 
            dsid_list[mc_run]['Diboson'].append(str(dsid))
            for variable in variables:
                histogram = myfile.Get(variable)
                histogram.SetDirectory(0)
                BigDic[mc_run]["Diboson"][variable].append(histogram)
        elif dsid in IDs['W']: 
            dsid_list[mc_run]['W'].append(str(dsid))
            for variable in variables:
                histogram = myfile.Get(variable)
                histogram.SetDirectory(0)
                BigDic[mc_run]["W"][variable].append(histogram)
        # elif dsid in IDs[sig_var]:
        #     # if not dsid == int(id): continue
        #     dsid_list[mc_run]['Signal'].append(str(dsid))
        #     for variable in variables:
        #         BigDic[mc_run]["Signal"][variable].append(myfile.Get(variable))
        elif dsid in [514560, 514561]:
            # if not dsid == DM_MODEL: continue
            dsid_list[mc_run]["DH HDS m_{Z'} 130"].append(str(dsid))
            for variable in variables:
                BigDic[mc_run]["DH HDS m_{Z'} 130"][variable].append(myfile.Get(variable))
        
        elif dsid in [514562, 514563]:
            # if not dsid == DM_MODEL: continue
            dsid_list[mc_run]["LV HDS m_{Z'} 130"].append(str(dsid))
            for variable in variables:
                BigDic[mc_run]["LV HDS m_{Z'} 130"][variable].append(myfile.Get(variable))
        
        elif dsid in [514564, 514565]:
            # if not dsid == DM_MODEL: continue
            dsid_list[mc_run]["EFT HDS m_{Z'} 130"].append(str(dsid))
            for variable in variables:
                BigDic[mc_run]["EFT HDS m_{Z'} 130"][variable].append(myfile.Get(variable))

for subdir, dirs, files in os.walk(susydir):
    dirs[:] = [d for d in dirs if d not in exclude]
    for file in files:
        dsid = int(file.split('.')[1])
        c = os.path.join(subdir, file)
        if '508116' in c: 
            myfile2 = R.TFile.Open(c)# print(c)
        elif '503085' in c: 
            myfile2 = R.TFile.Open(c)
        else: continue
        # print(c)
        # print(dsid)
        # myfile2 = R.TFile.Open(c)
        # filelist2.append(myfile2)
        mc_run = subdir.split('/')[-1]
        AFII = c.split('.')[-2].split('_')[-1]
        if AFII =='AFII': 
            print('Skipping AFII file:',c)
            continue
        if dsid == 503085:
            dsid_list[mc_run]['SUSY (m_{#tilde{l}}, m_{#tilde{#chi}_{1}^{0}}) = (90, 1)'].append(str(dsid))
            for variable in variables:
                histogram = myfile2.Get(variable)
                histogram.SetDirectory(0)
                BigDic[mc_run]["SUSY (m_{#tilde{l}}, m_{#tilde{#chi}_{1}^{0}}) = (90, 1)"][variable].append(histogram)
        elif dsid == 508116:
            dsid_list[mc_run]['2HDMa (m_{a}, m_{H^{-}}) = (150, 400)'].append(str(dsid))
            for variable in variables:
                histogram = myfile2.Get(variable)
                histogram.SetDirectory(0)
                BigDic[mc_run]["2HDMa (m_{a}, m_{H^{-}}) = (150, 400)"][variable].append(histogram)
# exit()
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
            
sow_susy_file = open('SOW_SIG_SUSY.json')
SOW_SUSY = json.load(sow_susy_file)
SOW_a = OrderedDict(list(SOW_bkg['mc16a'].items()) + list(SOW_SUSY['mc16a'].items()) + list(SOW_sig_AFII['mc16a'].items()))
SOW_d = OrderedDict(list(SOW_bkg['mc16d'].items()) + list(SOW_SUSY['mc16d'].items()) + list(SOW_sig_AFII['mc16d'].items()))
SOW_e = OrderedDict(list(SOW_bkg['mc16e'].items()) + list(SOW_SUSY['mc16e'].items()) + list(SOW_sig_AFII['mc16e'].items()))
sow_susy_file.close()

Backgrounds = ["W", "Diboson", 'TTbar', 'Single Top', 'Drell Yan', "DH HDS m_{Z'} 130", "LV HDS m_{Z'} 130", "EFT HDS m_{Z'} 130", 
                'SUSY (m_{#tilde{l}}, m_{#tilde{#chi}_{1}^{0}}) = (90, 1)', '2HDMa (m_{a}, m_{H^{-}}) = (150, 400)']

Colors = {}
Colors["DH HDS m_{Z'} 130"] = R.TColor.GetColor('#F42069')
Colors["LV HDS m_{Z'} 130"] = R.TColor.GetColor('#FFC0CB')
Colors["EFT HDS m_{Z'} 130"] = R.TColor.GetColor('#FF0000')
Colors["SUSY (m_{#tilde{l}}, m_{#tilde{#chi}_{1}^{0}}) = (90, 1)"] = R.TColor.GetColor('#702963')
Colors["2HDMa (m_{a}, m_{H^{-}}) = (150, 400)"] = R.TColor.GetColor('#FAA0A0')
Colors["Drell Yan"] = R.TColor.GetColor('#8EDC9D')
Colors['Single Top'] = R.TColor.GetColor('#EF7126')
Colors["TTbar"] = R.TColor.GetColor('#F9E559')
Colors["Diboson"] = R.TColor.GetColor('#6CCECB')
Colors["W"] = R.TColor.GetColor('#218C8D')

# save_dir = "../Plots/SUSY_TEST"
save_dir = "../Plots/Data_Analysis/SRs"
# save_dir = "../Plots/Data_Analysis/JetSelection"
try:
    os.makedirs(save_dir)

except FileExistsError:
    pass

stack = {}
stackDH = {}
stackLV = {}
stackEFT = {}
stackSUSY = {}
stackStop = {}
hist = {}
isjet = {}
met_region = {}

thist = {}


for vari in variables:
    met_region[vari] = vari.split("_")[1:3]
    isjet[vari] = 0
    if vari.split("_")[-2] == 'jet':
        isjet[vari] = 1
    hist[vari] = vari.split('_')[-1]
    if hist[vari] == 'sig':
        hist[vari] = vari.split('_')[-2]+'_'+hist[vari]

    stack[vari] = R.THStack()
    stackDH[vari] = R.THStack()
    stackLV[vari] = R.THStack()
    stackEFT[vari] = R.THStack()
    stackSUSY[vari] = R.THStack()
    stackStop[vari] = R.THStack()
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
                if len(dsid_list[mc][i]) == 2 and j == 2: break
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
    legend = R.TLegend(0.41, 0.72, 0.9, 0.88)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.025)
    legend.SetNColumns(2)
    data_hist[vari].SetMarkerStyle(20)
    data_hist[vari].SetLineColor(R.kBlack)
    for bkg in Backgrounds:
        for mc in mc_year:
            if bkg == "DH HDS m_{Z'} 130" or bkg == "LV HDS m_{Z'} 130" or bkg == "EFT HDS m_{Z'} 130" or bkg == 'SUSY (m_{#tilde{l}}, m_{#tilde{#chi}_{1}^{0}}) = (90, 1)' or bkg == '2HDMa (m_{a}, m_{H^{-}}) = (150, 400)':
                thist[vari][mc][bkg].SetFillStyle(0)
                thist[vari][mc][bkg].SetLineWidth(2)
                thist[vari][mc][bkg].SetLineColor(Colors[bkg])
                thist[vari][mc][bkg].SetLineStyle(1)
                if bkg == "DH HDS m_{Z'} 130":
                    stackDH[vari].Add(thist[vari][mc][bkg])
                elif bkg == "LV HDS m_{Z'} 130":
                    stackLV[vari].Add(thist[vari][mc][bkg])
                elif bkg == "EFT HDS m_{Z'} 130":
                    stackEFT[vari].Add(thist[vari][mc][bkg])
                elif bkg == "SUSY (m_{#tilde{l}}, m_{#tilde{#chi}_{1}^{0}}) = (90, 1)":
                    stackSUSY[vari].Add(thist[vari][mc][bkg])
                elif bkg == "2HDMa (m_{a}, m_{H^{-}}) = (150, 400)":
                    stackStop[vari].Add(thist[vari][mc][bkg])
                
            else:   
                thist[vari][mc][bkg].SetFillColor(Colors[bkg])
                thist[vari][mc][bkg].SetLineColor(Colors[bkg])
                stack[vari].Add(thist[vari][mc][bkg])      
        legend.AddEntry(thist[vari][mc][bkg], bkg)
    if met_region[vari][0] == 'CtrlReg':
        legend.AddEntry(data_hist[vari], 'Data')
    Plot_Maker(stack[vari], legend, isjet[vari], met_region[vari], hist[vari], data_hist[vari], save_dir, [stackDH[vari], stackLV[vari], stackEFT[vari], stackSUSY[vari], stackStop[vari]])
