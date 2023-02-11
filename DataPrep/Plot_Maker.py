import ROOT as R

def Plot_Maker(stack, legend, lep, charge, hist, data, dir, sig=None):
    c = R.TCanvas()
    c.SetWindowSize(1000, 800)
    c.Draw()
    R.gStyle.SetOptStat(0)
    
    pad = R.TPad("upper_pad", "", 0, 0.27, 1, 1)
    pad2 = R.TPad("lower_pad", "", 0, 0, 1, 0.33)
    pad.SetTickx(False)
    pad.SetTicky(False)
    pad.SetBottomMargin(0.09)
    pad.SetLogy()
    pad2.SetBottomMargin(0.5)
    pad2.SetTopMargin(0.009)
    pad.Draw()
    pad2.Draw()
    pad.cd()
    
    stack.Draw('HIST')
    data.Draw("E SAME")
    legend.Draw('SAME')
    
    if sig != None:
        sig.Draw('HIST2SAME')
    
    
    sumMC = stack.GetStack().Last()
    sumMC.SetDirectory(0)
    sumMC_err = stack.GetStack().Last().Clone("h_with_error")
    data_clone = data.Clone()
    
    sumMC_err.SetFillStyle(3018)
    sumMC_err.SetFillColor(R.kBlack)
    sumMC_err.Draw("E2SAME")
    
    legend.AddEntry(sumMC_err,"Stat. unc.")
    
    stack.GetYaxis().SetTitle("Events")
    stack.GetYaxis().SetTitleSize(0.05)
    stack.GetYaxis().SetTitleOffset(0.6)
        
    if charge == "SS":    
        
        if lep == 'ee':
            lepp = 'e^{#pm}e^{#pm}'
        
        elif lep == 'uu':
            lepp = '#mu^{#pm}#mu^{#pm}'
            
        elif lep == 'ue':
            lepp = '#mu^{#pm}e^{#pm}'
            
        elif lep == 'eu':
            lepp = 'e^{#pm}#mu^{#pm}'
            
            
    elif charge == "OS":
        if lep == 'ee':
            lepp = 'e^{#pm}e^{#mp}'
        
        elif lep == 'uu':
            lepp = '#mu^{#pm}#mu^{#mp}'
            
        elif lep == 'ue':
            lepp = '#mu^{#pm}e^{#mp}'
            
        elif lep == 'eu':
            lepp = 'e^{#pm}#mu^{#mp}'
        
    else:
        lepp = 'Jets'
        
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
        
    elif hist =='ht':
        xaxis = 'H_{T} [GeV]'
        
    elif hist =='dPhiLeps':
        xaxis = '|#Delta#phi(l_{1}, l_{2})|'
        
    elif hist =='dPhiCloseMet':
        xaxis = '|#Delta#phi(l_{clos}, E_{T}^{miss})|'
        
    elif hist =='dPhiLeadMet':
        xaxis = '|#Delta#phi(l_{lead}, E_{T}^{miss})|'
        
    elif hist =='dPhiLLmet':
        xaxis = '|#Delta#phi(ll, E_{T}^{miss})|'
        
    elif hist =='mt2':
        xaxis = 'm_{T2} [GeV]'
        
    elif hist =='nBJet':
        xaxis = 'Number of B jets'
        
    elif hist =='nLJet':
        xaxis = 'Number of light jets'
        
    elif hist =='nTJet':
        xaxis = 'Total number of jets'
        
    elif hist =='et':
        xaxis = 'E_{T} [GeV]'
        
    elif hist =='phi1':
        xaxis = '#phi_{1}'
        
    elif hist =='phi2':
        xaxis = '#phi_{2}'
        
    elif hist =='bjetsPt20':
        xaxis = 'Num. of b-jet with p_{T} > 20 GeV'
        
    elif hist =='bjetsPt30':
        xaxis = 'Num. of b-jets with p_{T} > 30 GeV'
        
    elif hist =='bjetsPt40':
        xaxis = 'Num. of b-jets with p_{T} > 40 GeV'
        
    elif hist =='bjetsPt50':
        xaxis = 'Num. of b-jets with p_{T} > 50 GeV'
        
    elif hist =='bjetsPt60':
        xaxis = 'Num. of b-jets with p_{T} > 60 GeV'
        
    elif hist =='ljetsPt20':
        xaxis = 'Num. of light-jets with p_{T} > 20 GeV'
        
    elif hist =='ljetsPt30':
        xaxis = 'Num. of light-jets with p_{T} > 30 GeV'
        
    elif hist =='ljetsPt40':
        xaxis = 'Num. of light-jets with p_{T} > 40 GeV'
        
    elif hist =='ljetsPt50':
        xaxis = 'Num. of light-jets with p_{T} > 50 GeV'
        
    elif hist =='ljetsPt60':
        xaxis = 'Num. of light-jets with p_{T} > 60 GeV'
        
    elif hist =='jetEtaCentral':
        xaxis = 'Num. of jets with |#eta| < 2.5'
        
    elif hist =='jetEtaForward':
        xaxis = 'Num. of jets with |#eta| > 2.5'
        
    # Add ATLAS label?
    text = R.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.04)
    text.DrawLatex(0.21, 0.82, "#sqrt{s} = 13 TeV, 139 fb^{-1}")
    text.DrawLatex(0.21, 0.77, "> 50GeV E_{T}^{miss}, " + lepp)
    stack.SetMinimum(1e-2)
    if hist == 'eta1' or hist == 'eta2' or hist == 'phi1' or hist == 'phi2' or hist=='dPhiLeps' or hist=='dPhiCloseMet' or hist=='dPhiLeadMet' or hist=='dPhiLLmet': 
        stack.SetMaximum(5e8)
    else:
        stack.SetMaximum(2e7)
        
    pad2.cd()
    pad2.SetGridy()
    pad2.SetTickx(False)
    pad2.SetTicky(False)
    data_clone.Divide(sumMC)
    data_clone.SetTitle("")
    data_clone.SetMarkerStyle(20)
    data_clone.GetXaxis().SetTitle(xaxis)
    data_clone.GetXaxis().SetTitleOffset(1.3)
    data_clone.GetYaxis().SetTitle("Events / Bkg")
    data_clone.GetYaxis().SetTitleSize(0.09)
    data_clone.GetYaxis().SetTitleOffset(0.3)
    data_clone.GetXaxis().SetLabelSize(0.1)
    data_clone.GetXaxis().SetTitleSize(0.13)
    if hist == 'met' or hist == 'met_sig':
        data_clone.SetMaximum(3)
        data_clone.SetMinimum(0)
    
    elif hist == 'eta1' or hist == 'eta2' or hist == 'phi1' or hist == 'phi2': 
        data_clone.SetMaximum(1.2)
        data_clone.SetMinimum(0.8)
    
    else:
        data_clone.SetMaximum(2)
        data_clone.SetMinimum(0)
    data_clone.Draw("ep")
    
    if not (charge =='SS' or charge =='OS'):
        savepath = dir+'/'+lep+'_'+hist+'.pdf'
    else:
        savepath = dir+'/'+lep+'_'+charge+'_'+hist+'.pdf'
    c.SaveAs(savepath) 