import ROOT as R

def Plot_Maker(stack, legend, lep, hist, logy=False):
    c = R.TCanvas()
    c.SetWindowSize(1000, 800)
    c.cd()
    R.gStyle.SetOptStat(0)
    if logy == True:
        c.SetLogy()
    
    stack.Draw('HIST')
    
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
    
    # Add ATLAS label?
    text = R.TLatex()
    text.SetNDC()
    # text.SetTextFont(72)
    # text.SetTextSize(0.045)
    #text.DrawLatex(0.21, 0.85, "ATLAS")
    text.SetTextFont(42)
    text.SetTextSize(0.04)
    # text.DrawLatex(0.21, 0.80, "#sqrt{s} = 13 TeV  #int Ldt = 139 fb^{-1}")
    text.DrawLatex(0.21, 0.85, "#sqrt{s} = 13 TeV, 139 fb^{-1}")
    text.DrawLatex(0.32, 0.80, lepp)
    # text.DrawLatex(0.21 + 0.087, 0.85, "PreLiminary")
    stack.GetXaxis().SetTitle(xaxis)
    stack.SetMinimum(1e-2)
    if hist == 'eta1' or hist == 'eta2' or hist == 'phi1' or hist == 'phi2': 
        stack.SetMaximum(1e6)
    else:
        stack.SetMaximum(2e4)
        
    savepath = 'Plots/'+lep+'_'+hist+'.pdf'
    c.SaveAs(savepath) 