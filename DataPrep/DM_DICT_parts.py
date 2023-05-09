import os, json
import pandas as pd
import numpy as np

DM_dsid_dir = "../../../storage/racarcam/DMS/"

DM_DSID = []
for dsid in os.listdir(DM_dsid_dir):
    DM_DSID.append(dsid.split('.')[0])


if os.path.exists("DM_DICT.json"):
    print('Rewriting dictionary')
    os.remove("DM_DICT.json")


DM_DICT = {}
Event_dic = {}
Event_dic['LV'] = {}
Event_dic['DH'] = {}
Event_dic['EFT'] = {}
Event_dic['LV']['HDS'] = {}
Event_dic['DH']['HDS'] = {}
Event_dic['EFT']['HDS'] = {}
Event_dic['LV']['LDS'] = {}
Event_dic['DH']['LDS'] = {}
Event_dic['EFT']['LDS'] = {}
Event_dic['LV']['HDS']['Low mass'] = {}
Event_dic['DH']['HDS']['Low mass'] = {}
Event_dic['EFT']['HDS']['Low mass'] = {}
Event_dic['LV']['LDS']['Low mass'] = {}
Event_dic['DH']['LDS']['Low mass'] = {}
Event_dic['EFT']['LDS']['Low mass'] = {}
Event_dic['LV']['HDS']['Mid mass'] = {}
Event_dic['DH']['HDS']['Mid mass'] = {}
Event_dic['EFT']['HDS']['Mid mass'] = {}
Event_dic['LV']['LDS']['Mid mass'] = {}
Event_dic['DH']['LDS']['Mid mass'] = {}
Event_dic['EFT']['LDS']['Mid mass'] = {}
Event_dic['LV']['HDS']['High mass'] = {}
Event_dic['DH']['HDS']['High mass'] = {}
Event_dic['EFT']['HDS']['High mass'] = {}
Event_dic['LV']['LDS']['High mass'] = {}
Event_dic['DH']['LDS']['High mass'] = {}
Event_dic['EFT']['LDS']['High mass'] = {}

for file in os.listdir('../EventSelector/Histograms_50MET/mc16a'):
    dsid = file.split('.')[1]
    if dsid in DM_DSID:
        name = file.split('.')[-2].split('_')[3:]
        if name[2] == 'mZp': 
            name[2] ="$m_{Z'}$"
        if name[4] == 'mumu': 
            name[4] ="$\mu\mu$"
        name = name[0].upper()+' '+name[1].upper()+' '+name[2]+' '+name[3]+' '+name[4]
        df_dm = pd.read_hdf(DM_dsid_dir+dsid+'.h5', key='df_tot')
        if int(name.split(' ')[3]) <= 600:         
            Event_dic[name.split(' ')[0]][name.split(' ')[1]]['Low mass'][dsid] = np.shape(df_dm)[0]
            DM_DICT[dsid] = name.split(' ')[0]+'_'+name.split(' ')[1]+'_'+'low_mass'
            
        elif int(name.split(' ')[3]) <= 1100:         
            Event_dic[name.split(' ')[0]][name.split(' ')[1]]['Mid mass'][dsid] = np.shape(df_dm)[0]
            DM_DICT[dsid] = name.split(' ')[0]+'_'+name.split(' ')[1]+'_'+'mid_mass'
        
        else:         
            Event_dic[name.split(' ')[0]][name.split(' ')[1]]['High mass'][dsid] = np.shape(df_dm)[0]
            DM_DICT[dsid] = name.split(' ')[0]+'_'+name.split(' ')[1]+'_'+'high_mass'

print('MC events in LV HDS Low mZ models:', sum(Event_dic['LV']['HDS']['Low mass'].values()))
print('MC events in LV LDS Low mZ models:', sum(Event_dic['LV']['LDS']['Low mass'].values()))
print('MC events in DH HDS Low mZ models:', sum(Event_dic['DH']['HDS']['Low mass'].values()))
print('MC events in DH LDS Low mZ models:', sum(Event_dic['DH']['LDS']['Low mass'].values()))
print('MC events in EFT HDS Low mZ models:', sum(Event_dic['EFT']['HDS']['Low mass'].values()))
print('MC events in EFT LDS Low mZ models:', sum(Event_dic['EFT']['LDS']['Low mass'].values()))

print('MC events in LV HDS Mid mZ models:', sum(Event_dic['LV']['HDS']['Mid mass'].values()))
print('MC events in LV LDS Mid mZ models:', sum(Event_dic['LV']['LDS']['Mid mass'].values()))
print('MC events in DH HDS Mid mZ models:', sum(Event_dic['DH']['HDS']['Mid mass'].values()))
print('MC events in DH LDS Mid mZ models:', sum(Event_dic['DH']['LDS']['Mid mass'].values()))
print('MC events in EFT HDS Mid mZ models:', sum(Event_dic['EFT']['HDS']['Mid mass'].values()))
print('MC events in EFT LDS Mid mZ models:', sum(Event_dic['EFT']['LDS']['Mid mass'].values()))

print('MC events in LV HDS High mZ models:', sum(Event_dic['LV']['HDS']['High mass'].values()))
print('MC events in LV LDS High mZ models:', sum(Event_dic['LV']['LDS']['High mass'].values()))
print('MC events in DH HDS High mZ models:', sum(Event_dic['DH']['HDS']['High mass'].values()))
print('MC events in DH LDS High mZ models:', sum(Event_dic['DH']['LDS']['High mass'].values()))
print('MC events in EFT HDS High mZ models:', sum(Event_dic['EFT']['HDS']['High mass'].values()))
print('MC events in EFT LDS High mZ models:', sum(Event_dic['EFT']['LDS']['High mass'].values()))

print('=='*12,'Testing','=='*12)
if dsid in DM_DICT:
    print(dsid, 'is in', DM_DICT.get(dsid))
    
print(DM_DICT)
exit()
json = json.dumps(DM_DICT)

f = open("DM_DICT.json","w")
f.write(json)
f.close()