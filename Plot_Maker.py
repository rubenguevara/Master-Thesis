import ROOT as R

def Plot_Maker(stack, legend, lep, hist, data, dir, sig=None):
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
    sumMC.SetFillColor(R.kBlack)
    sumMC.SetLineColor(R.kBlack)
    sumMC_err = stack.GetStack().Last().Clone("h_with_error")
    
    sumMC_err.SetFillStyle(3018)
    sumMC_err.SetFillColor(R.kBlack)
    sumMC_err.Draw("e2same")
    sumMC.Divide(data)
    
    legend.AddEntry(sumMC_err,"Stat. unc.")
    
    stack.GetYaxis().SetTitle("Events")
    stack.GetYaxis().SetTitleSize(0.05)
    stack.GetYaxis().SetTitleOffset(0.6)
    
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
    #text.DrawLatex(0.21, 0.80, "#sqrt{s} = 13 TeV  #int Ldt = 139 fb^{-1}")
    text.DrawLatex(0.21, 0.80, "#sqrt{s} = 13 TeV, 139 fb^{-1}")
    text.DrawLatex(0.27, 0.75, lepp)
    #text.DrawLatex(0.21 + 0.087, 0.85, "PreLiminary")
    #stack.GetXaxis().SetTitle(xaxis)
    stack.SetMinimum(1e-2)
    if hist == 'eta1' or hist == 'eta2' or hist == 'phi1' or hist == 'phi2': 
        stack.SetMaximum(5e10)
    else:
        stack.SetMaximum(2e8)
        
    pad2.cd()
    pad2.SetGridy()
    pad2.SetTickx(False)
    pad2.SetTicky(False)
    sumMC.SetTitle("")
    sumMC.SetMarkerStyle(20)
    sumMC.GetXaxis().SetTitle(xaxis)
    sumMC.GetXaxis().SetTitleOffset(1.3)
    sumMC.GetYaxis().SetTitle("Events / Bkg")
    sumMC.GetYaxis().SetTitleSize(0.09)
    sumMC.GetYaxis().SetTitleOffset(0.3)
    sumMC.GetXaxis().SetLabelSize(0.1)
    sumMC.GetXaxis().SetTitleSize(0.13)
    #sumMC.GetYaxis().SetLabelSize(0.16)
    if hist == 'met' or hist == 'met_sig':
        sumMC.SetMaximum(3)
        sumMC.SetMinimum(0)
    
    elif hist == 'eta1' or hist == 'eta2' or hist == 'phi1' or hist == 'phi2': 
        sumMC.SetMaximum(1.2)
        sumMC.SetMinimum(0.8)
    
    else:
        sumMC.SetMaximum(2)
        sumMC.SetMinimum(0)
    sumMC.Draw("ep")
    
    savepath = dir+'/'+lep+'_'+hist+'.pdf'
    c.SaveAs(savepath) 