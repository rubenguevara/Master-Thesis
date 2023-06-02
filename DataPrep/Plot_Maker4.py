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
    data.Draw("EP2SAME")
    legend.Draw('SAME')
    
    if sig != None:
        for s in sig:
            s.GetStack().Last().Draw('HIST2SAME')
    
    
    sumMC_stat = stack.GetStack().Last()
    sumMC_stat.SetDirectory(0)
    stat_unc_MC = stack.GetStack().Last().Clone("h_with_error")
    # sumMC_syst = stack.GetStack().Last()
    data_clone = data.Clone()
    data_syst = data.Clone()
    data_syst2 = data.Clone()
    
    
    if hist == 'met' and met_region[0] == 'Uncut':
        line = R.TLine(50, 1.5e-4, 50, stack.GetMaximum()*30)
        line.SetLineWidth(2)
        line.SetLineStyle(10)
        line.SetLineColor(R.kViolet+1)
        line.Draw('HIST2SAME')
    
    stat_unc_MC.SetFillStyle(3004)
    stat_unc_MC.SetFillColor(R.kBlack)
    stat_unc_MC.Draw("E2SAME")
    
    # sumMC_syst.SetFillStyle(3005)
    # # sumMC_syst.SetFillStyle(3018)
    # sumMC_syst.SetFillColor(R.kBlack)
    # sumMC_syst.Draw("E2SAME")
    
    # sumMC_syst2.SetFillStyle(3005)
    # # sumMC_syst.SetFillStyle(3018)
    # sumMC_syst2.SetFillColor(R.kBlack)
    # sumMC_syst2.Draw("E2SAME")
    
    legend.AddEntry(stat_unc_MC,"Stat. unc.")
    # legend.AddEntry(sumMC_syst,"Syst. unc.")
    
    stack.GetYaxis().SetTitle("Events")
    stack.GetYaxis().SetTitleSize(0.05)
    stack.GetYaxis().SetTitleOffset(0.6)
        
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
        
    elif hist =='bjetsPt20':
        xaxis = 'Number of b-jets with p_{T} > 20 GeV'
        
    elif hist =='bjetsPt30':
        xaxis = 'Number of b-jets with p_{T} > 30 GeV'
        
    elif hist =='bjetsPt40':
        xaxis = 'Number of b-jets with p_{T} > 40 GeV'
        
    elif hist =='bjetsPt50':
        xaxis = 'Number of b-jets with p_{T} > 50 GeV'
        
    elif hist =='bjetsPt60':
        xaxis = 'Number of b-jets with p_{T} > 60 GeV'
        
    elif hist =='ljetsPt20':
        xaxis = 'Number of light jets with p_{T} > 20 GeV'
        
    elif hist =='ljetsPt30':
        xaxis = 'Number of light jets with p_{T} > 30 GeV'
        
    elif hist =='ljetsPt40':
        xaxis = 'Number of light jets with p_{T} > 40 GeV'
        
    elif hist =='ljetsPt50':
        xaxis = 'Number of light jets with p_{T} > 50 GeV'
        
    elif hist =='ljetsPt60':
        xaxis = 'Number of light jets with p_{T} > 60 GeV'
        
    elif hist =='phi1':
        xaxis = lepp+' #phi_{1}'
        
    elif hist =='phi2':
        xaxis = lepp+' #phi_{2}'
        
    elif hist =='phi3':
        xaxis = lepp+' #phi_{3}'
        
    # Add ATLAS label?
    text = R.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.03)
    
    if met_region[0] == 'Uncut':
        text.DrawLatex(0.15, 0.82, "#sqrt{s} = 13 TeV, 139 fb^{-1}")
        
    elif met_region[0] == 'CtrlReg':
        text.DrawLatex(0.15, 0.82, "#sqrt{s} = 13 TeV, 139 fb^{-1}")
        text.DrawLatex(0.15, 0.78,"> 50 GeV E_{T}^{miss}")
    else: 
        text.DrawLatex(0.15, 0.82, "#sqrt{s} = 13 TeV, 139 fb^{-1}, > 120 GeV m_{ll}")
        if "-" in met_region[1]:
            mets = met_region[1].split('-')
            text.DrawLatex(0.15, 0.78, mets[1]+" GeV > E_{T}^{miss} > "+mets[0]+" GeV")
        else: 
            text.DrawLatex(0.15, 0.78, met_region[1]+" GeV > E_{T}^{miss}")
    
    stack.SetMinimum(1e-3)
    if hist == 'eta1' or hist == 'eta2' or hist == 'phi1' or hist == 'eta3' or hist == 'phi3' or hist == 'phi2' or hist=='dPhiLeps' or hist=='dPhiCloseMet' or hist=='dPhiLeadMet' or hist=='dPhiLLmet': 
        stack.SetMaximum(stack.GetMaximum()*1e3)
    else:
        stack.SetMaximum(stack.GetMaximum()*10)
    pad2.cd()
    pad2.SetGridy()
    pad2.SetTickx(False)
    pad2.SetTicky(False)
    data_clone.Divide(sumMC_stat)
    # data_syst.Divide(sumMC_syst)
    data_syst.SetMarkerSize(0)
    data_syst.SetLineWidth(0)
    data_syst.SetFillColor(R.kGray)
    # data_syst.SetFillStyle(3004)
    # data_syst2.Divide(sumMC_syst2)
    data_syst2.SetLineWidth(0)
    data_syst2.SetMarkerSize(0)
    data_syst2.SetFillColor(R.kGray)
    # data_syst2.SetFillStyle(3004)
    
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
        
    # data_syst.SetTitle("")
    # data_syst.SetMarkerStyle(20)
    # data_syst.GetXaxis().SetTitle(xaxis)
    # data_syst.GetXaxis().SetTitleOffset(1.3)
    # data_syst.GetYaxis().SetTitle("Events / Bkg")
    # data_syst.GetYaxis().SetTitleSize(0.09)
    # data_syst.GetYaxis().SetTitleOffset(0.3)
    # data_syst.GetXaxis().SetLabelSize(0.1)
    # data_syst.GetXaxis().SetTitleSize(0.13)
    # if hist == 'met' or hist == 'met_sig':
    #     data_syst.SetMaximum(3)
    #     data_syst.SetMinimum(0)
    
    # elif hist == 'eta1' or hist == 'eta2' or hist == 'phi1' or hist == 'phi2': 
    #     data_syst.SetMaximum(1.2)
    #     data_syst.SetMinimum(0.8)
    
    # else:
    #     data_syst.SetMaximum(2)
    #     data_syst.SetMinimum(0)
    data_syst.SetMaximum(1.5)
    data_syst.SetMinimum(0.5)
    # data_syst.Divide(sumMC_syst)
    # data_syst.SetMarkerSize(0)
    # data_syst.SetLineWidth(0)
    # data_syst.SetFillColor(R.kGray)
    # data_syst2.Divide(sumMC_syst2)
    # data_syst2.SetLineWidth(0)
    # data_syst2.SetMarkerSize(0)
    # data_syst2.SetFillColor(R.kGray)
    
    # data_syst.Draw('EHIST')
    # data_syst2.Draw('EHIST2SAME')
    data_clone.Draw("ep")
    
    if met_region[0] == 'Uncut':
        save_dir = dir+'/Uncut/'
        
    elif met_region[0] == 'CtrlReg':
        save_dir = dir+'/Control_region/'
    else:
        save_dir = dir+'/'+met_region[1]+'_MET-120_mll/'

    try:
        os.makedirs(save_dir)

    except FileExistsError:
        pass
        
    if hist == 'eta1' or hist == 'eta2' or hist == 'eta3' or hist == 'phi1' or hist == 'phi2' or hist == 'phi3' or hist == 'pt1' or hist == 'pt2' or hist == 'pt3': 
        savepath = save_dir+hist+"_"+lepp+'.pdf'
    else:
        savepath = save_dir+hist+'.pdf'
    
    
    if met_region[0] == 'CtrlReg' and hist == 'met':
        stack.GetXaxis().SetRangeUser(50, 2500)
        data_clone.GetXaxis().SetRangeUser(50, 2500)
        
    elif met_region[0] != 'CtrReg' and met_region[0] != 'Uncut':
        if hist =='mll' or hist=='mt':
            stack.GetXaxis().SetRangeUser(120, 3500)
            data_clone.GetXaxis().SetRangeUser(120, 3500)
        elif hist =='met': 
            if '-' in met_region[1]:
                stack.GetXaxis().SetRangeUser(int(mets[0]), int(mets[1]))
                data_clone.GetXaxis().SetRangeUser(int(mets[0]), int(mets[1]))    
            else:
                stack.GetXaxis().SetRangeUser(int(met_region[1]), 2500)
                data_clone.GetXaxis().SetRangeUser(int(met_region[1]), 2500)
    
    c.SaveAs(savepath) 
    