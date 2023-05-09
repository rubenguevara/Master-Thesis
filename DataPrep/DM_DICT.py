import os, json

DM_dsid_dir = "../../../storage/racarcam/DMS/"

dsid_dict = 2        # change to get full dsid name (1) or theory (0)

if dsid_dict == 0:
    filename = "DM_DICT_Zp_all.json"
    print('Only making dictionary of theories')
    
if dsid_dict == 1:
    filename = "DM_DICT_Zp_dsid.json"
    print('Only making dictionary of DSIDS')


if dsid_dict == 2:
    filename = "DM_DICT_Zp_models.json"
    print('Only making dictionary of experimentally different models')

DM_DSID = []
for dsid in os.listdir(DM_dsid_dir):
    DM_DSID.append(dsid.split('.')[0])


if os.path.exists(filename):
    print('Rewriting dictionary')
    os.remove(filename)


DM_DICT = {}

for file in os.listdir('../EventSelector/Histograms/mc16a'):
    dsid = file.split('.')[1]
    if dsid in DM_DSID:
        name = file.split('.')[-2].split('_')[3:]
        new_name = name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]
        if dsid_dict == 0:
            DM_DICT[name[0]] = []
        elif dsid_dict == 1:
            DM_DICT[new_name] = []
        elif dsid_dict == 2:
            DM_DICT[name[0]+'_'+name[1]] = []
        
for file in os.listdir('../EventSelector/Histograms/mc16a'):
    dsid = file.split('.')[1]
    if dsid in DM_DSID:
        name = file.split('.')[-2].split('_')[3:]
        new_name = name[0]+'_'+name[1]+'_'+name[2]+'_'+name[3]
        if new_name.split('_') == name[:4]:
            if dsid_dict == 0:
                DM_DICT[name[0]].append(dsid)
            elif dsid_dict == 1:
                DM_DICT[new_name].append(dsid)
            elif dsid_dict == 2:
                DM_DICT[name[0]+'_'+name[1]].append(dsid)

json = json.dumps(DM_DICT)
print(DM_DICT)
f = open(filename,"w")
f.write(json)
f.close()