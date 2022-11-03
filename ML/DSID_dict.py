import os, json

DM_dsid_dir = "../../../storage/racarcam/DMS"

DM_DSID = []
for dsid in os.listdir(DM_dsid_dir):
    DM_DSID.append(dsid.split('.')[0])

if os.path.exists("DM_DICT.json"):
    print('Rewriting dictionary')
    os.remove("DM_DICT.json")
    
DM_DICT = {}
for file in os.listdir('../EventSelector/Histograms/mc16a'):
    dsid = file.split('.')[1]
    if dsid in DM_DSID:
        name = file.split('.')[-2].split('_')[3:]
        if name[2] == 'mZp': 
            name[2] ="$m_{Z'}$"
        if name[4] == 'mumu': 
            name[4] ="$\mu\mu$"
        name = name[0].upper()+' '+name[1].upper()+' '+name[2]+' '+name[3]+' '+name[4]
        DM_DICT[dsid] = name

json = json.dumps(DM_DICT)

f = open("DM_DICT.json","w")
f.write(json)
f.close()

"""
to read

json_file = open('DM_DICT.json')

data = json.load(json_file)
"""