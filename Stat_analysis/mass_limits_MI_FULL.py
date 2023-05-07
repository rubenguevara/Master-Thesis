import pyStats, os, argparse
from ROOT import TMath;
import limitPlot;


parser = argparse.ArgumentParser()
parser.add_argument('--dm_model', type=str, default="DH_HDS", help="Dataset to test")
args = parser.parse_args()

dm_model = args.dm_model

dataset = dm_model
save_dir = '../Plots/Limits/Model_independent/'+dataset+'/'

save_txt_path = 'Data/Model_independent/'
try:
    os.makedirs(save_dir)

except FileExistsError:
    pass

limitPlots = {}
limitPlots['combined'] = limitPlot.limitPlot('Combined')
limitPlots['electron'] = limitPlot.limitPlot('Electron')
limitPlots['muon'] = limitPlot.limitPlot('Muon')
#peek in the input file to check which channels are there
inputFile = open(save_txt_path+dataset+'.txt','r');
lines = inputFile.readlines();
for l in lines:

    exec(l); #Input file consists of valid Python statements such as mass=300 etc.

    if 'channel=' in l:
        if not channel in limitPlots.keys():
            limitPlots[channel] = limitPlot.limitPlot(channel);


print( 'Will make the following limit plots:' );
print(limitPlots);

            
countexp = {};
xtitle='Mass';
ytitle='Cross section';
yrange=[-999.0,-999.0];

#read the inputs and fill in the respective limit plots with background levels, observed counts, etc. 
f = open(save_txt_path+dataset+'.txt');
for l in f.readlines():

    exec(l); #Input file consists of valid Python statements such as mass=300 etc. 

    if 'mass=' in l: #Such a line defines a new point in the limit plot
        for channel in limitPlots.keys():
            countexp[channel] = pyStats.countingExperiment(name = 'mass = '+str(mass), intLum = intLum, intLumUnc = intLumUncertainty);
            limitPlots[channel].addPoint(mass, countexp[channel], theoryCrossSection);
    
    if 'channel=' in l: #Such a line gives the inputs for a given channel     
        if 'electron' in channel:
            if not 'SR' in channel: continue
            countexp['electron'].addChannel(name = channel, bkg = background, bkgUnc = backgroundUncertainty, Nobs = Nobs, eff = efficiency, effUnc = efficiencyUncertainty);
        
        if 'muon' in channel:
            if not 'SR' in channel: continue
            countexp['muon'].addChannel(name = channel, bkg = background, bkgUnc = backgroundUncertainty, Nobs = Nobs, eff = efficiency, effUnc = efficiencyUncertainty);
        
        if not 'SR' in channel: continue    
        countexp['combined'].addChannel(name = channel, bkg = background, bkgUnc = backgroundUncertainty, Nobs = Nobs, eff = efficiency, effUnc = efficiencyUncertainty);
        countexp[channel].addChannel(name = channel, bkg = background, bkgUnc = backgroundUncertainty, Nobs = Nobs, eff = efficiency, effUnc = efficiencyUncertainty);

sr = 'Combined SRs'

limitPlots['electron'].calculate();
limitPlots['electron'].drawPlot(xtitle="m_{Z'} [GeV]", ytitle="Cross section [fb]", yrange=[4e-06,2.2e4], filename=save_dir+"mass_exclusion_ee.pdf", lep='e', sr = sr,
                                watermark=True, unc=20, title="Z' "+dataset.split('_')[0]+' '+dataset.split('_')[1]);


limitPlots['muon'].calculate();
limitPlots['muon'].drawPlot(xtitle="m_{Z'} [GeV]", ytitle="Cross section [fb]", yrange=[4e-06,2.2e4], filename=save_dir+"mass_exclusion_uu.pdf", lep='mu', sr = sr,
                                watermark=True, unc=20, title="Z' "+dataset.split('_')[0]+' '+dataset.split('_')[1]);


limitPlots['combined'].calculate();
limitPlots['combined'].drawPlot(xtitle="m_{Z'} [GeV]", ytitle="Cross section [fb]", yrange=[4e-06,2.2e4], filename=save_dir+"mass_exclusion_comb.pdf", sr = sr,
                                watermark=True, unc=20, title="Z' "+dataset.split('_')[0]+' '+dataset.split('_')[1]);