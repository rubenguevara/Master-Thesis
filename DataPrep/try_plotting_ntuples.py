import uproot as up
import ROOT as R 

save_dir = "/storage/racarcam/"


## Customize files here
dm1 = save_dir + "ZpxDMx50METxFR.root"
Run2_bkgs = save_dir + "NewRun2X50MET.root"
filename = 'FULL_Zp_50MET.h5' 
variable = 'mll'
binning = R.RDF.TH1DModel(variable, variable, 50, 10, 1200)
df_a = R.RDataFrame("id_mc16a", Run2_bkgs)
df_d = R.RDataFrame("id_mc16d", Run2_bkgs)
df_e = R.RDataFrame("id_mc16e", Run2_bkgs)
mll = df_a.Histo1D(binning, variable, 'Weights'.GetPtr())
# mll = mll.Scale(df_a['Weight'])

c = R.TCanvas()
mll.Draw()
c.Draw()
# mll.Draw('HIST')
c.SaveAs("test.pdf") 
exit() 

thing = up.open(Run2_bkgs)

tree_a = thing['id_mc16a']
tree_d = thing['id_mc16d']
tree_e = thing['id_mc16e']

mll_a = tree_a.arrays([variable, 'Weight', 'RunNumber', 'RunPeriod'], library="np")
mll_d = tree_d.arrays([variable, 'Weight', 'RunNumber', 'RunPeriod'], library="np")
mll_e = tree_e.arrays([variable, 'Weight', 'RunNumber', 'RunPeriod'], library="np")
print(mll_a)
exit()


df = ROOT.RDF.FromNumpy({'x': x, 'y': y})
# mll = R.TH1D("mll", "mll", 50, 50, 2000)
c = R.TCanvas()
mll_a.Draw()
c.Draw()
# mll.Draw('HIST')
c.SaveAs("test.pdf") 