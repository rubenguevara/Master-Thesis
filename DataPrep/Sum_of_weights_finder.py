import ROOT as R
import os, json


SOW_SIG = {}
SOW_SIG['mc16a'] = {}
SOW_SIG['mc16d'] = {}
SOW_SIG['mc16e'] = {}

input_dir = '/storage/shared/data/master_students/Ruben/MC_JAN2023/'

mc_runs = ['mc16a', 'mc16d', 'mc16e']
asdsad = 0
for mc in mc_runs:
    for id in os.listdir(input_dir+mc+'/'):
        dsid = id.split('.')[4]
        SOW_SIG[mc][dsid] = []

    for subdir, dirs, files in os.walk(input_dir+mc+'/'):
        for dir in dirs:
            dsid = dir.split('.')[4]
            if dsid != '505944': continue
            sow = 0
            for file in os.listdir(input_dir+mc+'/'+dir):
                tf = R.TFile(input_dir+mc+'/'+dir+'/'+file)
                h = tf.Get("histoEventCount")
                sow += h.GetBinContent(1)
            SOW_SIG[mc][dsid] = sow

print(sow)
print(SOW_SIG)
# print(SOW_SIG['mc16d']['505889'])
# print(SOW_SIG['mc16e']['505889'])

# json = json.dumps(SOW_SIG)

# f = open("SOW_SIG_SUSY.json","w")  # Change name of file if needed
# f.write(json)
# f.close()


# sow_SIG_file = open('SOW_SIG.json')
# SOW_SIG = json.load(sow_SIG_file)
# print(SOW_SIG)
# sow_SIG_file.close()