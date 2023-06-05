import os;
import multiprocessing as mp;
from time import sleep;
from array import array;
import ROOT;

"""
Class that defines a simple collection of limit setting results,
namely the observed limit together with the expected limit and
its 1- and 2-sigma fluctuations (green and yellow bands)
"""
class limitResult:

    def __init__(self):

        self.observedLimit = -999.0;

        self.expectedLimitMinus2sigma = -999.0;
        self.expectedLimitMinus1sigma = -999.0;
        self.expectedLimit = -999.0;
        self.expectedLimitPlus1sigma = -999.0;
        self.expectedLimitPlus2sigma = -999.0;


"""
Class for a limit plot with typically some theory parameter on the x-axis (like a mass).
For each point along the x-axis, a counting experiment object must be provided. The class
then handles the parallel execution of the calculation of observed and expected limits
(with 1- and 2-sigma fluctuations) as well as the actual plotting in the conventional
style with green and yellow bands for the 1- and 2-sigma fluctuations around the expected
limit.
"""
class limitPlot:

    def __repr__(self):
        return 'Limit plot "'+self.name+'"';

    def __str__(self):

        retVal = '';
        retVal += '==========================\n==========================\n';
        retVal += 'Limit plot "'+self.name+'"\n'
        retVal += '==========================\n==========================\n';
        retVal += '\nThe plot consists of the following points:\n\n'
        for i in range(len(self.xvalues)):
            retVal += '=============================================';
            retVal += '\nx-axis value: '+str(self.xvalues[i])+'\n';
            if self.theoryPredictions[i] != -999.0:
                retVal += 'theory prediction: '+str(self.theoryPredictions[i])+'\n';
            retVal += str(self.statModels[i]);
            if self.results[i].observedLimit != -999.0:
                retVal += '---------------------------\n';
                retVal += 'Limit setting results:\n';
                retVal += 'Observed limit: '+str(self.results[i].observedLimit)+'\n';
                retVal += "Expected limit and bands:\n"
                retVal += "  -2sigma                 -1sigma                 median                +1sigma                +2sigma\n";
                retVal += str(self.results[i].expectedLimitMinus2sigma)+"    ";
                retVal += str(self.results[i].expectedLimitMinus1sigma)+"    ";
                retVal += str(self.results[i].expectedLimit)+"    ";
                retVal += str(self.results[i].expectedLimitPlus1sigma)+"    ";
                retVal += str(self.results[i].expectedLimitPlus2sigma)+"    \n";
        return retVal;

    #Constructor
    def __init__(self, name='myLimitPlot'):

        self.xvalues = [];
        self.statModels = [];
        self.results = [];
        self.theoryPredictions = [];
        self.name = name;
        self.numCores = int(os.getenv('SLURM_JOB_CPUS_PER_NODE', os.cpu_count()));

    #Function to add a point for a given x-axis value (i.e. typically a given mass).
    #The "model" input is a counting experiment object for the given point. The
    #"theoryPrediction" can be optionally given as the thoretical prediction of the
    #signal cross section at that mass, so that the conventional "theory curve" can be
    #included in the limit plot. The intersection of the theory curve and the limit
    #curve then defines a limit on the x-axis parameter, i.e. typically a mass limit.
    def addPoint(self, xvalue, model, theoryPrediction = -999.0):

        self.xvalues.append(xvalue);
        self.statModels.append(model);
        self.results.append(limitResult());
        self.theoryPredictions.append(theoryPrediction);

    #Function to calculate all limit setting results for one point
    def calculatePoint(self, i, outputDict, NsplitPseudos):

        self.statModels[i].numCores = NsplitPseudos;
        
        result = limitResult();
        result.observedLimit = self.statModels[i].getBayesianLimit();
        m2sig,m1sig,expected,p1sig,p2sig = self.statModels[i].getBayesianExpectedLimit();
        result.expectedLimitMinus2sigma = m2sig;
        result.expectedLimitMinus1sigma = m1sig;
        result.expectedLimit = expected;
        result.expectedLimitPlus1sigma = p1sig;
        result.expectedLimitPlus2sigma = p2sig;
        outputDict[i] = result;

    #Function to calculate all points. The parallelization can be controlled with the
    #input parameters, where "massPointsBatchSize" sets the number of points to be
    #calculated in parallel, and "NsplitPseudos" sets the number of parallel processes
    #for parallelization over pseudo-experiments for each point.
    def calculate(self, massPointsBatchSize = -1, NsplitPseudos = -1):

        #By default, run all mass points in parallel
        if massPointsBatchSize == -1:
            massPointsBatchSize = len(self.statModels);
            
        #Check if there is some additional capacity to split the pseudo-experiments
        if NsplitPseudos == -1:
            NsplitPseudos = max(1,round(self.numCores/massPointsBatchSize));
        
        procs = [];
        outputDict = mp.Manager().dict();
        for i in range(len(self.statModels)):
            
            p = mp.Process(target=self.calculatePoint, args=(i, outputDict, NsplitPseudos));
            procs.append(p);
            p.start();

            while len(procs) == massPointsBatchSize:
                sleep(0.1);
                for proc in procs:
                    if not proc.is_alive():
                        procs.remove(proc);

        for p in procs:
            p.join();

        for i in range(len(self.results)):
            self.results[i] = outputDict[i];

    #Function to dump results to a plain text file that can be plotted using the simple script "limitPlot.cpp"
    def dumpToFile(self, filename='limits.txt'):

        fout = open(filename,'w');

        fout.write(str(len(self.xvalues))+'\n');
        
        for i in range(len(self.xvalues)):
            fout.write(str(self.xvalues[i])+" ");
            fout.write(str(self.theoryPredictions[i])+" ");
            fout.write(str(self.results[i].observedLimit)+" ");
            fout.write(str(self.results[i].expectedLimitMinus2sigma)+" ");
            fout.write(str(self.results[i].expectedLimitMinus1sigma)+" ");
            fout.write(str(self.results[i].expectedLimit)+" ");
            fout.write(str(self.results[i].expectedLimitPlus1sigma)+" ");
            fout.write(str(self.results[i].expectedLimitPlus2sigma)+"\n");

    #Function that does the actual drawing
    def drawPlot(self, xtitle='Mass', ytitle='Cross section', yrange=[-999.0,-999.0], xrange=[131,1499], filename='limitplot.png', watermark=False, lep="lep", unc=0, title='DM exclusion', sr = None, syst_unc = None):

        N = len(self.xvalues);
        xvals = array('d',self.xvalues);
        exclusion = array('d',[]);
        exclusionM2S = array('d',[]);
        exclusionM1S = array('d',[]);
        exclusionExp = array('d',[]);
        exclusionP1S = array('d',[]);
        exclusionP2S = array('d',[]);

        for result in self.results:
            exclusion.append(result.observedLimit);
            exclusionM2S.append(result.expectedLimitMinus2sigma);
            exclusionM1S.append(result.expectedLimitMinus1sigma);
            exclusionExp.append(result.expectedLimit);
            exclusionP1S.append(result.expectedLimitPlus1sigma);
            exclusionP2S.append(result.expectedLimitPlus2sigma);

        exclusionG = ROOT.TGraph(N,xvals,exclusion);
        exclusionM2SG = ROOT.TGraph(N,xvals,exclusionM2S);
        exclusionM1SG = ROOT.TGraph(N,xvals,exclusionM1S);
        exclusionExpG = ROOT.TGraph(N,xvals,exclusionExp);
        exclusionP1SG = ROOT.TGraph(N,xvals,exclusionP1S);
        exclusionP2SG = ROOT.TGraph(N,xvals,exclusionP2S);

        global globalStuff;
        globalStuff.append(exclusionG);
        globalStuff.append(exclusionM2SG);
        globalStuff.append(exclusionM1SG);
        globalStuff.append(exclusionExpG);
        globalStuff.append(exclusionP1SG);
        globalStuff.append(exclusionP2SG);
        blue = (85, 205, 252)
        red = (247, 168, 184)
        exclusionM2SG.SetLineColor(ROOT.TColor.GetColor(*blue));
        exclusionM1SG.SetLineColor(ROOT.TColor.GetColor(*red));
        exclusionExpG.SetLineStyle(2);
        exclusionP2SG.SetLineColor(ROOT.TColor.GetColor(*blue));
        exclusionP1SG.SetLineColor(ROOT.TColor.GetColor(*red));
        exclusionG.SetMarkerStyle(8);

        fillM2S = ROOT.TGraph(2*N);
        for i in range(N):
            fillM2S.SetPoint(i,xvals[i],exclusionM1S[i]);
            fillM2S.SetPoint(N+i,xvals[N-i-1],exclusionM2S[N-i-1]);
        fillM2S.SetFillColor(ROOT.TColor.GetColor(*blue));

        globalStuff.append(fillM2S);

        fill1S = ROOT.TGraph(2*N);
        for i in range(N):
            fill1S.SetPoint(i,xvals[i],exclusionP1S[i]);
            fill1S.SetPoint(N+i,xvals[N-i-1],exclusionM1S[N-i-1]);
        fill1S.SetFillColor(ROOT.TColor.GetColor(*red));

        globalStuff.append(fill1S);

        fillP2S = ROOT.TGraph(2*N);
        for i in range(N):
            fillP2S.SetPoint(i,xvals[i],exclusionP2S[i]);
            fillP2S.SetPoint(N+i,xvals[N-i-1],exclusionP1S[N-i-1]);
        fillP2S.SetFillColor(ROOT.TColor.GetColor(*blue));

        globalStuff.append(fillP2S);

        if self.theoryPredictions[0] != -999.0:
            xsec = array('d',self.theoryPredictions);
            xsecG = ROOT.TGraph(N,xvals,xsec);
            xsecG005 = ROOT.TGraph(N,xvals,xsec);
            xsecG01 = ROOT.TGraph(N,xvals,xsec);
            for i in range(xsecG01.GetN()):
                xsecG005.GetY()[i] *= 25;
                xsecG01.GetY()[i] *= 100;
            globalStuff.append(xsecG);
            globalStuff.append(xsecG005);
            globalStuff.append(xsecG01);
        
        #ATLAS style
        # ROOT.gROOT.LoadMacro("utils.h");
        # ROOT.gROOT.SetStyle("ATLAS");
        # ROOT.atlasStyle.SetTitleSize(0.06,"Y");
        # ROOT.gROOT.ForceStyle();
        
        #Labels and rangesi
        if lep == 'e':
            exclusionG.GetHistogram().SetYTitle("#sigma(pp#rightarrowZ'+#chi#chi)#timesBR(Z'#rightarrowee) [fb]");
        elif lep == 'mu':
            exclusionG.GetHistogram().SetYTitle("#sigma(pp#rightarrowZ'+#chi#chi)#timesBR(Z'#rightarrow#mu#mu) [fb]");
        else:
            exclusionG.GetHistogram().SetYTitle("#sigma(pp#rightarrowZ'+#chi#chi)#timesBR(Z'#rightarrowll) [fb]");
    
        exclusionG.GetHistogram().GetYaxis().SetLabelSize(0.03)
        exclusionG.GetHistogram().GetXaxis().SetLabelSize(0.03)
        exclusionG.GetHistogram().SetXTitle(xtitle);
        exclusionG.GetHistogram().GetYaxis().SetTitleSize(0.04)
        exclusionG.GetHistogram().GetXaxis().SetTitleSize(0.04)

        if yrange[0] != -999.0:
            exclusionG.GetHistogram().GetYaxis().SetRangeUser(yrange[0],yrange[1]);
        exclusionG.GetHistogram().GetXaxis().SetRangeUser(xrange[0],xrange[1]);


        #Actual drawing
        c = ROOT.TCanvas("c", "", 1100, 900);
        c.SetLogy();

        xsecG005.SetLineStyle(4);
        xsecG01.SetLineStyle(8);
        xsecG.SetLineStyle(7);
        xsecG005.SetLineColor(ROOT.kRed);
        exclusionG.SetMarkerSize(0);
        exclusionG.SetLineColor(ROOT.kWhite);
        xsecG01.SetLineColor(ROOT.kRed);
        xsecG.SetLineColor(ROOT.kRed);
        xsecG005.SetMarkerSize(0);
        xsecG01.SetMarkerSize(0);
        xsecG.SetMarkerSize(0);
        globalStuff.append(c);
        exclusionG.SetTitle('')
        exclusionG.Draw('ALP');
        fillM2S.Draw("f");
        fill1S.Draw("f");
        fillP2S.Draw("f");
        
        exclusionM2SG.Draw("L same");
        exclusionM1SG.Draw("L same");
        exclusionP1SG.Draw("L same");
        exclusionP2SG.Draw("L same");
        exclusionExpG.Draw("L same");
        # exclusionG.Draw("LP same");

        if self.theoryPredictions[0] != -999.0:
            xsecG.Draw("C* same");
            xsecG005.Draw("C* same");
            xsecG01.Draw("C* same");
        
        ROOT.gStyle.SetLegendBorderSize(0)
        leg = ROOT.TLegend(0.64,0.66,0.86,0.88);
        # leg.AddEntry(exclusionG,"Excluded at 95% CL","PL");
        leg.AddEntry(exclusionExpG,"Expected limit","L");
        leg.AddEntry(fill1S,"Expected #pm 1#sigma","F");
        leg.AddEntry(fillP2S,"Expected #pm 2#sigma","F");
        if self.theoryPredictions[0] != -999.0:
            leg.AddEntry(xsecG,"Theory, g_{l} = 0.01","PL");
            leg.AddEntry(xsecG005,"Theory, g_{l} = 0.05","PL");
            leg.AddEntry(xsecG01,"Theory, g_{l} = 0.1","PL");
        leg.Draw();
        
        if watermark == True:
            text = ROOT.TLatex();
            text.SetNDC();
            text.SetTextFont(42);
            text.SetTextSize(0.032);
            text.DrawLatex(0.135, 0.84, "#sqrt{s} = 13 TeV, 139 fb^{-1}");
            
            if 'DH' in title:
                text.DrawLatex(0.135, 0.80, "Dark Higgs ("+title.split(' ')[-1]+")");
            elif 'LV' in title:
                text.DrawLatex(0.135, 0.80, "Light Vector ("+title.split(' ')[-1]+")");
            elif 'EFT' in title:
                text.DrawLatex(0.135, 0.80, "Effective Field Theory ("+title.split(' ')[-1]+")");
            
            if sr != None:
                if 'DH' in title:
                    text.DrawLatex(0.135, 0.80, "Dark Higgs ("+title.split(' ')[-1]+"), "+sr);
                elif 'LV' in title:
                    text.DrawLatex(0.135, 0.80, "Light Vector ("+title.split(' ')[-1]+"), "+sr);
                elif 'EFT' in title:
                    text.DrawLatex(0.135, 0.80, "Effective Field Theory ("+title.split(' ')[-1]+"), "+sr);

            if syst_unc != None:
                if syst_unc == 0:
                    text.DrawLatex(0.135, 0.80, "Dark Higgs ("+title.split(' ')[-1]+"), Stat. unc. only");
                    
                else:
                    text.DrawLatex(0.135, 0.80, "Dark Higgs ("+title.split(' ')[-1]+"), "+str(int(syst_unc*100))+"% Syst. unc.");
            
            
            if lep == "e":
                text.DrawLatex(0.135, 0.76, "Electron channel, 95% CL");
            elif lep == "mu":
                text.DrawLatex(0.135, 0.76, "Muon channel, 95% CL");
            else:
                text.DrawLatex(0.135, 0.76, "Dilepton channel, 95% CL");
            
        globalStuff.append(leg);
        
        c.SaveAs(filename);
        c.Draw();

#this is just to prevent that things go out of scope and get deleted
globalStuff = [];