import ROOT as R
import os

def Plot_Maker(stack, legend, lep, hist, sig=None, logy=False):
    c = R.TCanvas()
    c.SetWindowSize(1000, 800)
    c.cd()
    R.gStyle.SetOptStat(0)
    if logy == True:
        c.SetLogy()
    
    stack.Draw('HIST')
    if not sig ==None:
        sig.Draw('HIST2SAME')
    legend.Draw('SAME')
    
    stack.GetYaxis().SetTitle("Events")
    
    if lep == 'ee':
        lepp = 'e^{+}e^{-}'
    
    elif lep == 'uu':
        lepp = '#mu^{+}#mu^{-}'
        
    if hist =='pt1':
        xaxis = 'p_{T}^{1} [GeV]'
        
    elif hist =='pt2':
        xaxis = 'p_{T}^{2} [GeV]'
        
    elif hist =='eta1':
        xaxis = '#eta_{1}'
        
    elif hist =='eta2':
        xaxis = '#eta_{2}'
    
    elif hist =='mll':
        xaxis = 'm_{ll} [GeV]'
        
    elif hist =='met':
        xaxis = 'E_{T}^{miss} [GeV]'
    
    elif hist =='met_sig':
        xaxis = 'E_{T}^{miss}/#sigma'
        
    elif hist =='mt':
        xaxis = 'm_{T} [GeV]'
    
    elif hist =='et':
        xaxis = 'E_{T} [GeV]'
    
    elif hist =='phi1':
        xaxis = '#phi_{1}'
    
    elif hist =='phi2':
        xaxis = '#phi_{2}'
    
    # Add ATLAS label
    text = R.TLatex()
    text.SetNDC()
    text.SetTextFont(72)
    text.SetTextSize(0.045)
    text.DrawLatex(0.21, 0.85, "ATLAS")
    text.SetTextFont(42)
    text.DrawLatex(0.235, 0.80, lepp)
    # text.DrawLatex(0.21 + 0.087, 0.85, "Open Data")
    # text.SetTextSize(0.04)
    # text.DrawLatex(0.21, 0.80, "#sqrt{{s}} = 13 TeV, {:.1f} fb^{{-1}}".format(lumi / 1000.0))
    # text.DrawLatex(0.27, 0.75, "Z#rightarrow e^{+} e^{-}")
    stack.GetXaxis().SetTitle(xaxis)
    stack.SetMinimum(1e-3)
    if hist == 'eta1' or hist == 'eta2' or hist == 'phi1' or hist == 'phi2': 
        stack.SetMaximum(1e7)
    else:
        stack.SetMaximum(1e3)
    if not sig == None:     
        try:
            os.makedirs("Plots_S")

        except FileExistsError:
            pass
        savepath = 'Plots_S/S_'+lep+'_'+hist+'.pdf'
    else:
        savepath = 'Plots/'+lep+'_'+hist+'.pdf'
    c.SaveAs(savepath) 