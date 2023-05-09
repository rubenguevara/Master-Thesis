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
        
    if hist =='50METjetEtaCentral':
        xaxis = 'Num. of jets with |#eta| < 2.5'
        
    elif hist =='50METjetEtaForward20':
        xaxis = 'Num. of jets with |#eta| > 2.5 and p_T > 20 GeV'
        
    elif hist =='50METjetEtaForward30':
        xaxis = 'Num. of jets with |#eta| > 2.5 and p_T > 30 GeV'
        
    elif hist =='50METjetEtaForward40':
        xaxis = 'Num. of jets with |#eta| > 2.5 and p_T > 40 GeV'
        
    elif hist =='50METjetEtaForward50':
        xaxis = 'Num. of jets with |#eta| > 2.5 and p_T > 50 GeV'
        
    elif hist =='50METjetEtaCalorimeter20':
        xaxis = 'Num. of jets with |#eta| < 4.5 and p_T > 20 GeV'
        
    elif hist =='50METjetEtaCalorimeter30':
        xaxis = 'Num. of jets with |#eta| < 4.5 and p_T > 30 GeV'
        
    elif hist =='50METjetEtaCalorimeter40':
        xaxis = 'Num. of jets with |#eta| < 4.5 and p_T > 40 GeV'
        
    elif hist =='50METjetEtaCalorimeter50':
        xaxis = 'Num. of jets with |#eta| < 4.5 and p_T > 50 GeV'
        
        
    # Add ATLAS label?
    text = R.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.04)
    text.DrawLatex(0.21, 0.82, "#sqrt{s} = 13 TeV, 139 fb^{-1}")
    if lep == '50':       
        text.DrawLatex(0.21, 0.77, "> 50GeV E_{T}^{miss}, " + charge)
        stack.SetMaximum(2e7)
    elif lep == '25':       
        text.DrawLatex(0.21, 0.77, "> 25GeV E_{T}^{miss}, " + charge)
        stack.SetMaximum(2e8)
    else: 
        stack.SetMaximum(2e8)
    stack.SetMinimum(1e-2)
        
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
    data_clone.SetMaximum(2)
    data_clone.SetMinimum(0)
    data_clone.Draw("ep")
    
    lep_save = lep+"METcut_"
    hist_save = hist[5:]
    savepath = dir+'/'+lep_save+hist_save+'.pdf'
    c.SaveAs(savepath) 