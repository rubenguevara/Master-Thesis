import ROOT as R
import os

def Plot_Maker(stack, legend, isjet, met_region, hist, data, dir, sig=None):
    c = R.TCanvas()
    c.SetWindowSize(1200, 1000)
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
    if met_region[0] == 'CtrlReg':
        data.Draw("EP2SAME")
    legend.Draw('SAME')
    
    if sig != None:
        for s in sig:
            s.GetStack().Last().Draw('HIST2SAME')
    
    
    sumMC = stack.GetStack().Last().Clone()
    sumMC.SetDirectory(0)
    sumMC_err = stack.GetStack().Last().Clone("h_with_error")
    sumMC_syst = stack.GetStack().Last().Clone()
    sumMC_syst.SetDirectory(0)
    
    nBins = sumMC.GetNbinsX()
    for i in range(1,nBins+1):
        sumMC_syst.SetBinError(i, 0.2*sumMC_syst.GetBinContent(i))
    
    sumMC_syst_err = sumMC_syst.Clone()
    data_clone = data.Clone()
    
    
    if hist == 'met' and met_region[0] == 'Uncut':
        line = R.TLine(50, 1.5e-4, 50, stack.GetMaximum()*60)
        line.SetLineWidth(2)
        line.SetLineStyle(10)
        line.SetLineColor(R.kViolet+1)
        line.Draw('HIST2SAME')
    
    sumMC_err.SetFillStyle(3004)
    sumMC_err.SetFillColor(R.kRed-4)
    sumMC_err.Draw("E2SAME")
    
    sumMC_syst_err.SetFillStyle(3005)
    sumMC_syst_err.SetFillColor(R.kBlack)
    sumMC_syst_err.Draw("E2SAME")
    
    legend.AddEntry(sumMC_err,"Stat. unc.")
    legend.AddEntry(sumMC_syst_err,"20% syst. unc.")
    
    stack.GetYaxis().SetTitle("Events")
    stack.GetYaxis().SetTitleSize(0.05)
    stack.GetYaxis().SetTitleOffset(0.6)
        
    if isjet == 0:
        lepp = 'll'
    else:
        lepp = 'jet'          
        
    if hist =='pt1':
        xaxis = lepp+' p_{T}^{1} [GeV]'
        
    elif hist =='pt2':
        xaxis = lepp+' p_{T}^{2} [GeV]'
        
    elif hist =='pt3':
        xaxis = lepp+' p_{T}^{3} [GeV]'
        
    elif hist =='eta1':
        xaxis = lepp+' #eta_{1}'
        
    elif hist =='eta2':
        xaxis = lepp+' #eta_{2}'
        
    elif hist =='eta3':
        xaxis = lepp+' #eta_{3}'
        
    elif hist =='mll':
        xaxis = 'm_{ll} [GeV]'
        
    elif hist =='mjj':
        xaxis = 'm_{jj} [GeV]'
        
    elif hist =='met':
        xaxis = 'E_{T}^{miss} [GeV]'
        
    elif hist =='met_sig':
        xaxis = 'E_{T}^{miss}/#sigma'
        
    elif hist =='mt':
        xaxis = 'm_{T} [GeV]'
        
    elif hist =='ht':
        xaxis = 'H_{T} [GeV]'
        
    elif hist =='rt':
        xaxis = 'E_{T}^{miss} / H_{T}'
        
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
        xaxis = 'Number of b-jets'
        
    elif hist =='nLJet':
        xaxis = 'Number of light jets'
        
    elif hist =='nTJet':
        xaxis = 'Total number of jets'
        
    elif hist =='et':
        xaxis = 'E_{T} [GeV]'
        
    elif hist =='phi1':
        xaxis = lepp+' #phi_{1}'
        
    elif hist =='phi2':
        xaxis = lepp+' #phi_{2}'
        
    elif hist =='phi3':
        xaxis = lepp+' #phi_{3}'
        
    elif hist =='bjetsPt20':
        xaxis = 'Number of b-jets with p_{T} #geq 20 GeV'
        
    elif hist =='bjetsPt30':
        xaxis = 'Number of b-jets with p_{T} #geq 30 GeV'
        
    elif hist =='bjetsPt40':
        xaxis = 'Number of b-jets with p_{T} #geq 40 GeV'
        
    elif hist =='bjetsPt50':
        xaxis = 'Number of b-jets with p_{T} #geq 50 GeV'
        
    elif hist =='bjetsPt60':
        xaxis = 'Number of b-jets with p_{T} #geq 60 GeV'
        
    elif hist =='ljetsPt20':
        xaxis = 'Number of light jets with p_{T} #geq 20 GeV'
        
    elif hist =='ljetsPt30':
        xaxis = 'Number of light jets with p_{T} #geq 30 GeV'
        
    elif hist =='ljetsPt40':
        xaxis = 'Number of light jets with p_{T} #geq 40 GeV'
        
    elif hist =='ljetsPt50':
        xaxis = 'Number of light jets with p_{T} #geq 50 GeV'
        
    elif hist =='ljetsPt60':
        xaxis = 'Number of light jets with p_{T} #geq 60 GeV'
        
    
    text = R.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.03)
    
    if met_region[0] == 'Uncut':
        text.DrawLatex(0.15, 0.82, "#sqrt{s} = 13 TeV, 139 fb^{-1}")
        text.DrawLatex(0.15, 0.78, "m_{ll} > 10 GeV")
        
    elif met_region[0] == 'CtrlReg':
        text.DrawLatex(0.15, 0.82, "#sqrt{s} = 13 TeV, 139 fb^{-1}")
        text.DrawLatex(0.15, 0.78,"E_{T}^{miss}> 50 GeV, m_{ll} < 110 GeV")
    elif met_region[0] == 'PreReg':
        text.DrawLatex(0.15, 0.82, "#sqrt{s} = 13 TeV, 139 fb^{-1}")
        text.DrawLatex(0.15, 0.78,"E_{T}^{miss}> 50 GeV")
    else: 
        text.DrawLatex(0.15, 0.82, "#sqrt{s} = 13 TeV, 139 fb^{-1}, m_{ll} > 110 GeV")
        if "-" in met_region[1]:
            mets = met_region[1].split('-')
            text.DrawLatex(0.15, 0.78, mets[1]+" GeV > E_{T}^{miss} > "+mets[0]+" GeV")
        else: 
            text.DrawLatex(0.15, 0.78, "E_{T}^{miss} > "+met_region[1]+" GeV")
    
    stack.SetMinimum(1e-3)
    if hist == 'eta1' or hist == 'eta2' or hist == 'phi1' or hist == 'eta3' or hist == 'phi3' or hist == 'phi2' or hist=='dPhiLeps' or hist=='dPhiCloseMet' or hist=='dPhiLeadMet' or hist=='dPhiLLmet': 
        stack.SetMaximum(stack.GetMaximum()*1e3)
    elif hist == 'met' and "-" in met_region[1]:
        stack.SetMaximum(stack.GetMaximum()*2e3)
    elif hist == 'mll' and met_region[0] == 'CtrlReg':
        stack.SetMaximum(stack.GetMaximum()*2e3)
    else:
        stack.SetMaximum(stack.GetMaximum()*40)
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
    data_clone.SetMaximum(2.05)
    data_clone.SetMinimum(0.1)

    h_max = R.TH1D(sumMC_syst)
    h_min = R.TH1D(sumMC_syst)
    for i in range(1,nBins+1):
        h_max.SetBinContent(i, sumMC_syst.GetBinContent(i) + 0.2*sumMC_syst.GetBinContent(i) + sumMC_err.GetBinError(i))
        h_min.SetBinContent(i, sumMC_syst.GetBinContent(i) - 0.2*sumMC_syst.GetBinContent(i) - sumMC_err.GetBinError(i))
    
    h_max.Divide(h_max,sumMC_syst)
    h_min.Divide(h_min,sumMC_syst)

    resid_up = h_max.Clone('resid_up')
    resid_dwn = h_min.Clone('resid_dwn')

    resid_up.SetLineStyle(R.kDashed);
    resid_up.SetLineColor(R.kBlack);
    resid_up.SetFillColor(R.kGray);
    resid_up.SetFillStyle(3001)
    resid_dwn.SetLineStyle(R.kDashed);
    resid_dwn.SetLineColor(R.kBlack);
    resid_dwn.SetFillColor(10);
    resid_dwn.SetFillStyle(1001)
    

    if met_region[0] != 'CtrlReg':
        data_clone.SetMarkerColor(R.kWhite)
        data_clone.SetLineColor(R.kWhite)
        data_clone.SetFillColor(R.kWhite)
    
    data_clone.Draw("pe0");
    resid_up.Draw("hist][ same");
    resid_dwn.Draw("hist][ same");
    if met_region[0] == 'CtrlReg':
        data_clone.Draw("pe0 same");
    data_clone.Draw("axis same");
    data_clone.Draw("axiG same");
    
    if met_region[0] == 'Uncut':
        save_dir = dir+'/Uncut/'
        
    elif met_region[0] == 'CtrlReg':
        save_dir = dir+'/Control_region/'
    
    
    elif met_region[0] == 'PreReg':
        save_dir = dir+'/Preselection_region/'
    else:
        save_dir = dir+'/'+met_region[1]+'_MET-110_mll/'

    try:
        os.makedirs(save_dir)

    except FileExistsError:
        pass
        
    if hist == 'eta1' or hist == 'eta2' or hist == 'eta3' or hist == 'phi1' or hist == 'phi2' or hist == 'phi3' or hist == 'pt1' or hist == 'pt2' or hist == 'pt3': 
        savepath = save_dir+hist+"_"+lepp+'.pdf'
    else:
        savepath = save_dir+hist+'.pdf'
    
    if met_region[0] == 'CtrlReg':
        if hist == 'met':
            stack.GetXaxis().SetRangeUser(50, 2500)
            data_clone.GetXaxis().SetRangeUser(50, 2500)
        if hist =='mll' or hist=='mt':
            stack.GetXaxis().SetRangeUser(10, 110)
            data_clone.GetXaxis().SetRangeUser(10, 110)
            
    elif met_region[0] == 'PreReg' and hist == 'met':
        stack.GetXaxis().SetRangeUser(50, 2500)
        data_clone.GetXaxis().SetRangeUser(50, 2500)
        
    elif met_region[0] != 'CtrReg' and met_region[0] != 'Uncut' and met_region[0] != 'PreReg':
        if hist =='mll' or hist=='mt':
            stack.GetXaxis().SetRangeUser(110, 3500)
            data_clone.GetXaxis().SetRangeUser(110, 3500)
        elif hist =='met': 
            if '-' in met_region[1]:
                stack.GetXaxis().SetRangeUser(int(mets[0]), int(mets[1]))
                data_clone.GetXaxis().SetRangeUser(int(mets[0]), int(mets[1]))    
            else:
                stack.GetXaxis().SetRangeUser(int(met_region[1]), 2500)
                data_clone.GetXaxis().SetRangeUser(int(met_region[1]), 2500)
    
    c.SaveAs(savepath) 
    