import os, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dm_model', type=str, default="DH_HDS", help="Dataset to test")
parser.add_argument('--channel', type=str, default="ee", help="Lepton channel to test")
parser.add_argument('--tanb', type=str, default="1", help="Lepton channel to test")
args = parser.parse_args()

dm_model = args.dm_model
channel = args.channel 
tanb = args.tanb 


save_txt_path = '../Plots/Limits/Model_independent/'+dm_model+'/'

try:
    os.makedirs(save_txt_path)

except FileExistsError:
    pass

try:
    if dm_model == '2HDM': 
        os.remove(save_txt_path+dm_model+'_'+channel+'_tb'+tanb+'.txt')
    else:
        os.remove(save_txt_path+dm_model+'_'+channel+".txt")
    print('Updating old file')
except:
    print('Nothing to update')

if dm_model == '2HDM': 
    text_file = open(save_txt_path+dm_model+'_'+channel+'_tb'+tanb+'.txt', 'w')
    
    SR_textSR1 = open('../Plots/Limits/Model_independent/50-100/'+dm_model+'/'+dm_model+'_'+channel+'_tb'+tanb+'.txt', 'r')
    SR_textSR2 = open('../Plots/Limits/Model_independent/100-150/'+dm_model+'/'+dm_model+'_'+channel+'_tb'+tanb+'.txt', 'r')
    SR_textSR3 = open('../Plots/Limits/Model_independent/150/'+dm_model+'/'+dm_model+'_'+channel+'_tb'+tanb+'.txt', 'r')
else:
    text_file = open(save_txt_path+dm_model+'_'+channel+'.txt', 'w')
    
    SR_textSR1 = open('../Plots/Limits/Model_independent/50-100/'+dm_model+'/'+dm_model+'_'+channel+'.txt', 'r')
    SR_textSR2 = open('../Plots/Limits/Model_independent/100-150/'+dm_model+'/'+dm_model+'_'+channel+'.txt', 'r')
    SR_textSR3 = open('../Plots/Limits/Model_independent/150/'+dm_model+'/'+dm_model+'_'+channel+'.txt', 'r')
text_file.write('m1 m2 sign \n')

SR_textSR1.readline()
SR_textSR2.readline()
SR_textSR3.readline()
for SR1, SR2, SR3 in zip(SR_textSR1, SR_textSR2, SR_textSR3):
    z_SR1 = SR1.split(' ')[-1]
    z_SR2 = SR2.split(' ')[-1]
    z_SR3 = SR3.split(' ')[-1]
    if SR1.split(' ')[0] != SR2.split(' ')[0] and SR1.split(' ')[0] != SR3.split(' ')[0]:   
        if SR1.split(' ')[1] != SR2.split(' ')[1] and SR1.split(' ')[1] != SR3.split(' ')[1]:
            print('Different masses!')
    slep = SR1.split(' ')[0]
    neut = SR1.split(' ')[1]
    Z = np.sqrt( float(z_SR1)**2 + float(z_SR2)**2 + float(z_SR3)**2 )
    text_file.write(slep+' '+neut+' '+str(Z)+'\n')

if dm_model == 'SlepSlep':
    m_y = "'m_{ #tilde{#chi}_{1}^{0}} [GeV]'"

    if channel == 'ee':
        title = "'#tilde{e}#tilde{e} #rightarrow ee #tilde{#chi}_{1}^{0}#tilde{#chi}_{1}^{0}, Combined SRs'"
        m_x = "'m_{ #tilde{e}} [GeV]'"
    if channel == 'uu':
        title = "'#tilde{#mu}#tilde{#mu} #rightarrow #mu#mu #tilde{#chi}_{1}^{0}#tilde{#chi}_{1}^{0}, Combined SRs'"
        m_x = "'m_{ #tilde{#mu}} [GeV]'"
    if channel == 'll':
        m_x = "'m_{ #tilde{l}} [GeV]'"
        title = "'#tilde{l} #tilde{l} #rightarrow ll #tilde{#chi}_{1}^{0}#tilde{#chi}_{1}^{0}, Combined SRs'"

if dm_model == '2HDM':
    m_y = "'m_{a} [GeV]'"
    title = "'tan#beta = "+tanb+", Combined SRs'"
    m_x= "'m_{H^{-}} [GeV]'"


if dm_model == '2HDM':
    print('../software/Scripts/SUSYPheno/bin/munch.py -coord m1:'+m_x+',m2:'+m_y+' -resvars sign -title '+title+' -fn_table '+save_txt_path+dm_model+'_'+channel+'_tb'+tanb+'.txt -cont sign:Significance:1.645:2:2:3 --legend2bottomright')
else: 
    print('../software/Scripts/SUSYPheno/bin/munch.py -coord m1:'+m_x+',m2:'+m_y+' -resvars sign -title '+title+' -fn_table '+save_txt_path+dm_model+'_'+channel+'.txt -cont sign:Significance:1.645:2:2:3 --legend2bottomright')
