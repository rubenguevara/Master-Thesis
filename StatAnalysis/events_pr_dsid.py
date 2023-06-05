import json, argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dm_model', type=str, default="DH_HDS", help="Dataset to test")
args = parser.parse_args()

dm_model = args.dm_model

save_dir = "/storage/racarcam/"
# sig_file = save_dir+'/DM_Models/DM_Zp_'+dm_model.lower()+'.h5'
sig_file = save_dir+'/DM_Models/DM_'+dm_model+'.h5'

df_sig = pd.read_hdf(sig_file, key='df_tot')
df = df_sig[['CrossSection', 'RunPeriod', 'RunNumber']].drop_duplicates()


json_file_s = open('DM_DICT_SUSY_models.json')
S_file = json.load(json_file_s)
events = {}
for dsid in df['RunNumber'].drop_duplicates():
    if 'MET' in S_file[str(dsid)][0]: continue
    events[dsid] = {}
    # print(dsid)
    for mc in df['RunPeriod'].drop_duplicates():
        events[dsid][mc] = []
for dsid, mc, xs in zip(df['RunNumber'], df['RunPeriod'], df['CrossSection']):
    if 'MET' in S_file[str(dsid)][0]: continue
    if xs == None: 
        xs = 0
    if mc == 'mc16a':
        events[dsid][mc].append(xs)
    elif mc == 'mc16d':
            events[dsid][mc].append(xs)
    elif mc == 'mc16e':
            events[dsid][mc].append(xs)

events_run2 = {}
for dsid in events.keys():
    print(dsid)
    if dsid == 503121:
        events_run2[str(dsid)] = events[dsid]['mc16a'][0]*36.2e6 +events[dsid]['mc16d'][0]*44.3e6
        print('xs in', dsid, events[dsid]['mc16a'][0]*1e6 +events[dsid]['mc16d'][0]*1e6)
    else:        
        events_run2[str(dsid)] = events[dsid]['mc16a'][0]*36.2e6 +events[dsid]['mc16d'][0]*44.3e6 +events[dsid]['mc16e'][0]*58.5e6
        print('xs in', dsid, events[dsid]['mc16a'][0]*1e6 +events[dsid]['mc16d'][0]*1e6 +events[dsid]['mc16e'][0]*1e6)

events_before = {}
effective_xs = {}
for i in events_run2.keys():
#     if dm_model.lower() not in i: continue
    evts = events_run2[i]
    key = S_file[i][0].replace(' ', '_')
    events_before[key] = evts
    if i == 503121:
            effective_xs[key] = evts/(36.2+44.3)
    else:
        effective_xs[key] = evts/(36.2+44.3+58.5)
    print('Expected events before cuts in', key, evts)


json_file_s.close()
json1 = json.dumps(events_before)
f1 = open(dm_model+"_events_before.json","w")
f1.write(json1)
f1.close()

json2 = json.dumps(effective_xs)
f2 = open(dm_model+"_effective_xs.json","w")
f2.write(json2)
f2.close()

