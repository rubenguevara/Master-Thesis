import ROOT as R
import os, json


SOW_SIG = {}
SOW_SIG['mc16a'] = {}
SOW_SIG['mc16d'] = {}
SOW_SIG['mc16e'] = {}

input_dir = '/storage/shared/data/master_students/Ruben/MC/'

mc_runs = ['mc16a']#, 'mc16d', 'mc16e']

for mc in mc_runs:
    for id in os.listdir(input_dir+mc+'/'):
        dsid = id.split('.')[4]
        SOW_SIG[mc][dsid] = []

    for subdir, dirs, files in os.walk(input_dir+mc+'/'):
        for dir in dirs:
            dsid = dir.split('.')[4]
            sow = 0
            for file in os.listdir(input_dir+mc+'/'+dir):
                tf = R.TFile(input_dir+mc+'/'+dir+'/'+file)
                h = tf.Get("histoEventCount")
                sow += h.GetBinContent(1)
            SOW_SIG[mc][dsid] = sow

print(SOW_SIG)

# json = json.dumps(SOW_SIG)

# f = open("SOW_SIG.json","w")  # Change name of file if needed
# f.write(json)
# f.close()


# sow_SIG_file = open('SOW_SIG.json')
# SOW_SIG = json.load(sow_SIG_file)
# print(SOW_SIG)
# sow_SIG_file.close()