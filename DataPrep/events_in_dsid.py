import json

json_file = open('SOW_bkg.json')
file = json.load(json_file)

periods = ['mc16a', 'mc16d', 'mc16e']
s = 0
for p in periods:
    s += file[p]['700320']

print(s)