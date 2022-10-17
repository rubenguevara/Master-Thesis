#define EventSelector_cxx

#include "EventSelector.h"
#include <TH2.h>
#include <TStyle.h>
#include "TLorentzVector.h"
#include <fstream>
#include <TString.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <string> 
#include <algorithm> 
#include <iterator> 
#include "CalcGenericMT2/src/MT2_ROOT.h"
#include "MatrixMethod/MatrixMethod.cxx"
#include "MatrixMethod/MMefficiencies.cxx"
#include "makeMYTree.h"
// #include "makeMYTree.cxx"

TLorentzVector l1, l2, l3, met_lor, ll, l1_truth, l2_truth; 

// Options 
Int_t isData = 0, isMC = 0; 
Int_t doTruth = 0, doCutflow = 0, doSyst = 0, doFakes = 0, doLoose = 0, isRecast = 0, isAFII = 0, makeTree = 0; 

// Sample and channel stuff
Int_t DSID, prev_DSID = 0, yr; 
Int_t Incl_Sherpa[] = {364100, 364101, 364102, 364103, 364104, 364105, 364106, 364107, 364108, 364114, 364115, 364117, 364118, 364120};
Int_t Incl_Sh2211[] = {700320, 700321, 700322, 700323, 700324, 700325, 700326, 700327, 700328}; 
Int_t Incl_Powheg[] = {361106, 361107}; 
Int_t TTbar_samples[] = {410470, 410472}; //, 410633, 410634, 410635, 410636, 410637}; 
Int_t STincl_samples[] = {410644, 410645, 410658, 410659}; 
Int_t STdilep_samples[] = {410648, 410649}; 
Int_t Diboson_samples[] = {363356, 363358, 363359, 363360, 363489, 364250, 364253, 364254, 364255}; 
Int_t Zee_samples[] = {364114, 364115, 364116, 364117, 364118, 364119, 364120, 364121, 364122, 364123, 364124, 364125, 364126, 364127, 366309, 366310, 366312, 366313, 366315}; 
Int_t Sh2211_samples[] = {700320, 700321, 700322, 700323, 700324, 700325, 700326, 700327, 700328, 700452, 700453, 700454, 700455, 700456, 700457, 700458, 700459, 700460}; 
Int_t Sh221_samples[] = {364100, 364101, 364102, 364103, 364104, 364105, 364106, 364107, 364108, 364109, 364110, 364111, 364112, 364113, 364114, 364115, 364116, 364117, 364118, 364119, 364120, 364121, 364122, 364123, 364124, 364125, 364126, 364127, 366300, 366301, 366302, 366303, 366304, 366305, 366306, 366307, 366308, 366309, 366310, 366312, 366313, 366315}; 
Int_t isInclusive_Sherpa = 0, isInclusive_Powheg = 0, isInclusive_Sh2211 = 0, isTop = 0, isTTbar = 0, isSTincl = 0, isSTdilep = 0, isDiboson = 0, isSh2211 = 0, isSh221 =0, isZee = 0, isWB = 0, isSystChannel = 0, isTreeChannel = 0; 
TString  dileptons, final_name, filepath, period, name_1; 
TString option, dataset,  ml_file, file_dsid, file_nr;
vector<TString> channel_names, of_channel_names, all_channels, passed_channels, all_METcuts, passed_METcuts, all_METrelcuts, passed_METrelcuts, all_METsigcuts, passed_METsigcuts, variations, SFvariations, weightVariations, TopVariations, WBVariations, blinded_channels, syst_channel_names, syst_channels;  
map<TString, Int_t> weight_indices, weight_indices_Sh2211; 

// Kinematic variables
Float_t mll, met, met_phi, met_sig, met_sig_rel, met_ht, dphi, dR_ll, dphi_ll, ptll, ht, met_rel, dphi_rel, met_res, met_soft, met_elec, met_muon, met_jet, pt_lep1, pt_lep2;
Float_t met_truth, met_rel_truth, dphi_rel_truth, mll_truth, dphi_truth;
Int_t njets, ncjets, nfjets, nljets, nbjets, n_bjet77, n_bjet85;

// Weights and SFs
Float_t wgt, wgt_mc, wgt_pu, wgt_kf, wgt_ttbar, wgt_nlo_ew, wgt_xs, wgt_lsf, wgt_jsf, wgt_tsf, g_eff, xs, wgt_jet, wgt_bjet, wgt_fakes, LLwgt_noSF, wgt_lsf_lep1Loose, wgt_lsf_lep1Tight, wgt_lsf_lep2Loose, wgt_lsf_lep2Tight, wgt_lsf_lep3Loose, wgt_lsf_lep3Tight; 
Float_t nb2fb = 1.E6, pb2fb = 1.E3; 

// Cutflow variables 
Long64_t nevents = 0, sample_nevents = 0, eventID = 0; 
Int_t n_trig_e = 0, n_jetclean_e = 0, n_twolep_e = 0, n_loose_e = 0, n_pt_e = 0, n_eta_e =0, n_mll70_e = 0, n_mll120_e = 0, n_met100_e = 0, n_mll70_e_TT = 0;  
Int_t n_trig_u = 0, n_jetclean_u = 0, n_twolep_u = 0, n_loose_u = 0, n_mubad = 0, n_pt_u = 0, n_mll70_u = 0, n_mll120_u = 0, n_met100_u = 0, n_mll70_u_TT = 0;  
Int_t passed_pt30 = 0; 
Float_t min_MET = 999; 
Int_t n_2lep_truth = 0, n_not2lep_truth;  

// Weight sums 
Float_t s_mc_e = 0, s_kf_e = 0, s_pu_e = 0, s_xs_e = 0, s_lsf_e = 0, s_trig_e = 0, s_tt_e = 0, s_wgt_e = 0;     
Float_t s_mc_u = 0, s_kf_u = 0, s_pu_u = 0, s_xs_u = 0, s_lsf_u = 0, s_trig_u = 0, s_tt_u = 0, s_wgt_u = 0;     
Float_t s_mc_x_pu = 0, s_mc = 0; 
Float_t sw_bin1 = 0, sw_bin2 = 0, sw_bin3 = 0; 
Float_t test_sow = 0; 

Float_t m_e = 0.511e-3; // Electron mass (GeV) 
Float_t m_u = 0.106; // Muon mass (GeV) 

//
//Matrix Method
//
// Flag for whether leptons pass Tight
Bool_t tight1, tight2, tight3;
//Loose/Tight event counts (used per event, so one of them will be 1, the others 0)
Double_t NTT,NTl,NlT,Nll;
//Fake and real efficiencies
Double_t f1,f2,r1,r2;
//Tool to read different kinds of efficiencies from files (fake effs, real effs, 1D/2D, systematics variations...)
MMefficiencies MMeffs;
//Matrix Method tool from Eirik
MatrixMethod MM;

void EventSelector::Begin(TTree * /*tree*/)
{

  time_t t; // t passed as argument in function time()
  struct tm * tt; // decalring variable for localtime()
  time (&t); //passing argument to time()
  tt = localtime(&t);
  cout << "Process started: " << asctime(tt);

  // Read options 
  option = GetOption();
  TObjArray *my_option = option.Tokenize("_");
  dataset = ((TObjString *)(my_option->At(0)))->String();
  doTruth = atoi(((TObjString *)(my_option->At(1)))->String());
  doCutflow = atoi(((TObjString *)(my_option->At(2)))->String());
  doSyst = atoi(((TObjString *)(my_option->At(3)))->String());
  doFakes = atoi(((TObjString *)(my_option->At(4)))->String());
  doLoose = atoi(((TObjString *)(my_option->At(5)))->String());
  isRecast = atoi(((TObjString *)(my_option->At(6)))->String());
  isAFII = atoi(((TObjString *)(my_option->At(7)))->String());
  ml_file = ((TObjString *)(my_option->At(8)))->String();
  file_dsid = ((TObjString *)(my_option->At(9)))->String();
  file_nr = ((TObjString *)(my_option->At(10)))->String();
  delete my_option; 

  string stringy_option(dataset); 
  if(stringy_option.find("data")!=string::npos){isData=1;}else{isMC=1;}

  TH1::SetDefaultSumw2();

  // Define channels to consider 
  channel_names = {"incl"};

  all_channels = {};
  for(const auto & chn:channel_names){
    all_channels.push_back("ee_SS_"+chn); 
    all_channels.push_back("uu_SS_"+chn); 
    all_channels.push_back("eu_SS_"+chn); 
    all_channels.push_back("ue_SS_"+chn);
    all_channels.push_back("ee_OS_"+chn); 
    all_channels.push_back("uu_OS_"+chn); 
    all_channels.push_back("eu_OS_"+chn); 
    all_channels.push_back("ue_OS_"+chn); 
  }

  // Define histograms
  TH1::AddDirectory(false); //prevents that the program hangs while cleaning up after the Python script is done
  TString h_name, filename2; 
  for( const auto & chn:all_channels ){
      h_name = chn; 
      h_mll[h_name] = new TH1D("h_"+h_name+"_mll", h_name+"_mll", 74, 20, 3500);
      h_pt1[h_name] = new TH1D("h_"+h_name+"_pt1", h_name+"_pt1", 74, 20, 3500);  
      h_pt2[h_name] = new TH1D("h_"+h_name+"_pt2", h_name+"_pt2", 74, 20, 3500);  
      h_eta1[h_name] = new TH1D("h_"+h_name+"_eta1", h_name+"_eta1", 50, -3, 3);  
      h_eta2[h_name] = new TH1D("h_"+h_name+"_eta2", h_name+"_eta2", 50, -3, 3);  
      h_met[h_name] = new TH1D("h_"+h_name+"_met", h_name+"_met", 74, 0, 2500); 
      h_mt[h_name] = new TH1D("h_"+h_name+"_mt", h_name+"_mt", 74, 20, 3500);
      h_mt2[h_name] = new TH1D("h_"+h_name+"_mt2", h_name+"_mt2", 74, 20, 1500);
      h_ht[h_name] = new TH1D("h_"+h_name+"_ht", h_name+"_ht", 74, 20, 3500);
      h_met_sig[h_name] = new TH1D("h_"+h_name+"_met_sig", h_name+"_met_sig", 74, 0, 100);
      h_et[h_name] = new TH1D("h_"+h_name+"_et", h_name+"_et", 74, 20, 3000);
      h_phi1[h_name] = new TH1D("h_"+h_name+"_phi1", h_name+"_phi1", 50, -M_PI, M_PI);   
      h_phi2[h_name] = new TH1D("h_"+h_name+"_phi2", h_name+"_phi2", 50, -M_PI, M_PI);  
      h_dPhiLeps[h_name] = new TH1D("h_"+h_name+"_dPhiLeps", h_name+"_dPhiLeps", 50, -M_PI, M_PI); 
      h_dPhiLepMet[h_name] = new TH1D("h_"+h_name+"_dPhiLepMet", h_name+"_dPhiLepMet", 50, -M_PI, M_PI); 
      h_dPhiLLmet[h_name] = new TH1D("h_"+h_name+"_dPhiLLmet", h_name+"_dPhiLLpmet", 50, -M_PI, M_PI);  
      h_nBJet[h_name] = new TH1D("h_jet_nBJet","jet_nBJet", 8, 0, 7); 
      h_nLJet[h_name] = new TH1D("h_jet_nLJet","jet_nLJet", 8, 0, 7); 
      h_nTJet[h_name] = new TH1D("h_jet_nTJet","jet_nTJet", 8, 0, 7); 
      h_jetpt1[h_name] = new TH1D("h_jet_pt1", "jet_pt1", 74, 20, 3500);  
      h_jetpt2[h_name] = new TH1D("h_jet_pt2", "jet_pt2", 74, 20, 3500);  
      h_jeteta1[h_name] = new TH1D("h_jet_eta1", "jet_eta1", 50, -3, 3);  
      h_jeteta2[h_name] = new TH1D("h_jet_eta2", "jet_eta2", 50, -3, 3);  
      h_jetphi1[h_name] = new TH1D("h_jet_phi1", "jet_phi1", 50, -M_PI, M_PI);   
      h_jetphi2[h_name] = new TH1D("h_jet_phi2", "jet_phi2", 50, -M_PI, M_PI);  
      
  }
  filename2 = "../../../storage/racarcam/ML_files/"+ml_file+"-"+dataset+"-"+file_dsid+"-"+file_nr+".root";   // ML FILE
  MY = new makeMYTree(dataset,"central",filename2,"");
}

void EventSelector::SlaveBegin(TTree * /*tree*/){}

Bool_t EventSelector::Process(Long64_t entry){


  fReader.SetLocalEntry(entry);
  if(isMC){
    fReader_MC.SetLocalEntry(entry); 
  }
  if(doTruth){
    fReader_sig.SetLocalEntry(entry); 
  }
  if(!doSyst){
    fReader_nom.SetLocalEntry(entry); 
  }
  if(doSyst){
    fReader_sys.SetLocalEntry(entry);
  }

  nevents++; eventID = *event;

  if(nevents%1000000 == 0){cout << nevents/1000000 << " million events processed" << endl; } 
  // if(nevents>10){return kTRUE;} 
  
  
  if( isData ){

    DSID = *run; 
    if(prev_DSID == 0){ prev_DSID = *run; sample_nevents=0;}
    prev_DSID = DSID; 

    TObjArray *path = (TString(((TChain*)(EventSelector::fChain))->GetFile()->GetName())).Tokenize(".");
    
    
    period = ((TObjString *)(path->At(6)))->String();  //4
    TString file_nr = ((TObjString *)(path->At(11)))->String();  //9

    final_name = period+file_nr;
    delete path;

  } 
  else{
    DSID = *run; 
    if(prev_DSID == 0){ prev_DSID = *run; } 
    if( DSID != prev_DSID ){
      if(doCutflow==1){PrintCutflow(prev_DSID);}
      sample_nevents=0;
      WriteToFile(Form("%d", prev_DSID), dataset, final_name); 
      s_mc_x_pu = 0; s_mc = 0; 
      sw_bin1 = 0; sw_bin2 = 0; sw_bin3 = 0; 
      
    } 
    prev_DSID = DSID; 
    
    TObjArray *path = (TString(((TChain*)(EventSelector::fChain))->GetFile()->GetName())).Tokenize(".");
    name_1 = ((TObjString *)(path->At(7)))->String(); //5
    TString file_nr = ((TObjString *)(path->At(11)))->String();//9
    string name_2(name_1);
    name_2.erase(name_2.end()-10, name_2.end()); 
    if(isAFII){ final_name = TString(name_2)+"_AFII"+file_nr; } 
    else{ final_name = TString(name_2)+file_nr; }
    delete path;

    isInclusive_Sh2211 = (std::find(std::begin(Incl_Sh2211), std::end(Incl_Sh2211), DSID) != std::end(Incl_Sh2211));
    if(isInclusive_Sh2211 && *bornMass/1000.>120){ return kTRUE; } // For inclusive Sherpa 2.2.11 samples: only keep events with mll<120 GeV. 
  }
  sample_nevents++; 

  if(isMC){ yr = *year; } 
  if(isData){ 
    if(dataset=="data15"){yr=2015;} 
    if(dataset=="data16"){yr=2016;} 
    if(dataset=="data17"){yr=2017;} 
    if(dataset=="data18"){yr=2018;} 
  } 

  //==================================//
  // Triggers, isolation and cleaning //
  //==================================//

  Int_t el_trig = 0, mu_trig = 0;  

  if(yr==2015){ 
    if( *trigger_HLT_2e12_lhloose_L12EM10VH ){ el_trig = 1; n_trig_e++; }
    if( *trigger_HLT_mu26_imedium || *trigger_HLT_mu50 ){ mu_trig = 1; n_trig_u++;}
  }
  if(yr==2016){ 
    if( *trigger_HLT_2e17_lhvloose_nod0 ){ el_trig = 1; n_trig_e++; }
    if( *trigger_HLT_mu26_ivarmedium || *trigger_HLT_mu50 ){ mu_trig = 1; n_trig_u++;}
  }
  if(yr==2017){
    if( *trigger_HLT_2e24_lhvloose_nod0 ){el_trig = 1; n_trig_e++;}
    if((isData && (*run<326834 || *run>328393)) || isMC ){ // (isMC && *systName=="" && doSyst && (*randomRunNumber<326834 || *randomRunNumber>328393)) ){
      if( *trigger_HLT_2e17_lhvloose_nod0_L12EM15VHI ){el_trig = 1; n_trig_e++;}
    }  
    if( *trigger_HLT_mu26_ivarmedium || *trigger_HLT_mu50 ){ mu_trig = 1; n_trig_u++;}
  }
  if(yr==2018){
    if( *trigger_HLT_2e24_lhvloose_nod0 || *trigger_HLT_2e17_lhvloose_nod0_L12EM15VHI ){el_trig = 1; n_trig_e++;}
    if( *trigger_HLT_mu26_ivarmedium || *trigger_HLT_mu50 ){ mu_trig = 1; n_trig_u++;}
  }

  if( el_trig == 0 && mu_trig == 0 ){ return kTRUE; } 

  if(*jetCleaning_eventClean == 0){ return kTRUE; } 
  if(doCutflow==1){
    if(mu_trig==1){ n_jetclean_u++;} 
    if(el_trig==1){ n_jetclean_e++;} 
  }

  Int_t n_loose_mu = 0; vector<Int_t> loose_mu = {}; 
  Int_t n_loose_mu_f = 0; vector<Int_t> loose_mu_f = {}; 
  Int_t n_mu_bad = 0; 
  for(Int_t i = 0; i<*n_mu; i++){
    if(fabs(mu_d0sig[i])<3. && fabs(mu_z0sinTheta[i])<0.5){n_loose_mu++; loose_mu.push_back(i);} 
    if(fabs(mu_d0sig[i])>3. && fabs(mu_z0sinTheta[i])<0.5){n_loose_mu_f++; loose_mu_f.push_back(i);} 
    if(mu_isBad[i]){n_mu_bad++; } 
  }
  for(Int_t i = 0; i<*n_jet; i++){
    float jetpt = jet_pt[i];}
  Int_t n_loose_el = 0; vector<Int_t> loose_el = {}; 
  for(Int_t i = 0; i<*n_el; i++){
    if(fabs(el_d0sig[i])<5. && fabs(el_z0sinTheta[i])<0.5){n_loose_el++; loose_el.push_back(i);}
  }

  Int_t n_loose_lep = n_loose_el + n_loose_mu;
  Int_t lep1=-999, lep2=-999, lep3=-999;
  dileptons = ""; 
  if(n_loose_lep==2){ 
    if(n_loose_mu == 2){ 
      if(mu_trig==0){ return kTRUE; } 
      dileptons = "uu"; 
      lep1 = loose_mu[0]; lep2 = loose_mu[1]; n_loose_u++;} 
    else if(n_loose_el == 2){ 
      if(el_trig==0){ return kTRUE; } 
      dileptons = "ee"; 
      lep1 = loose_el[0]; lep2 = loose_el[1]; n_loose_e++;} 
    else if(n_loose_el == 1 && n_loose_mu == 1){                             //HERE
      if(el_trig == 0 && mu_trig == 0){ return kTRUE; } 
      if(el_pt[loose_el[0]]>mu_pt[loose_mu[0]]){
      dileptons = "eu"; 
      lep1 = loose_el[0]; lep2 = loose_mu[0]; n_loose_u++; n_loose_e++;} 
      else if(el_pt[loose_el[0]]<mu_pt[loose_mu[0]]){
      dileptons = "ue"; 
      lep1 = loose_mu[0]; lep2 = loose_el[0]; n_loose_u++; n_loose_e++;} }
  }

  if(dileptons==""){ return kTRUE; } 

  // Bad muon veto 
  //if(*n_mu_baseline_bad!=0){ return kTRUE; n_mubad++;} 
  if(n_mu_bad!=0){ return kTRUE; n_mubad++;} 
  
  //Check whether the leptons pass the Tight selection (as defined in the Matrix Method)
  if (dileptons == "ee") {    
    tight1 = el_passIso[lep1] && el_passLHID[lep1];
    tight2 = el_passIso[lep2] && el_passLHID[lep2];
  }
  if (dileptons == "uu") {    
    tight1 = mu_passIso[lep1];
    tight2 = mu_passIso[lep2];
  }
  if (dileptons == "eu") {    
    tight1 = el_passIso[lep1] && el_passLHID[lep1];
    tight2 = mu_passIso[lep2];
  }
  if (dileptons == "ue") {    
    tight1 = mu_passIso[lep1];
    tight2 = el_passIso[lep2] && el_passLHID[lep2];
  }

  if(!(tight1 && tight2)){ return kTRUE; } 
  
  //============//
  // Kinematics //
  //============//

  Int_t isOS = 0; 
  if(dileptons == "ee"){ 
    l1.SetPtEtaPhiM(el_pt[lep1]/1000., el_eta[lep1], el_phi[lep1], m_e);  
    l2.SetPtEtaPhiM(el_pt[lep2]/1000., el_eta[lep2], el_phi[lep2], m_e);  
    if(el_charge[lep1]!=el_charge[lep2]){ isOS = 1; } 
  } 
  if(dileptons == "uu"){
    l1.SetPtEtaPhiM(mu_pt[lep1]/1000., mu_eta[lep1], mu_phi[lep1], m_u);  
    l2.SetPtEtaPhiM(mu_pt[lep2]/1000., mu_eta[lep2], mu_phi[lep2], m_u);  
    if(mu_charge[lep1]!=mu_charge[lep2]){ isOS = 1; } 
  } 
  if(dileptons == "eu"){
    l1.SetPtEtaPhiM(el_pt[lep1]/1000., el_eta[lep1], el_phi[lep1], m_e);  
    l2.SetPtEtaPhiM(mu_pt[lep2]/1000., mu_eta[lep2], mu_phi[lep2], m_u);  
    if(el_charge[lep1]!=mu_charge[lep2]){ isOS = 1; } 
  } 
  if(dileptons == "ue"){
    l1.SetPtEtaPhiM(mu_pt[lep1]/1000., mu_eta[lep1], mu_phi[lep1], m_u);  
    l2.SetPtEtaPhiM(el_pt[lep2]/1000., el_eta[lep2], el_phi[lep2], m_e);  
    if(mu_charge[lep1]!=el_charge[lep2]){ isOS = 1; } 
  } 

  TLorentzVector totalj, ajet, j1, j2;
  n_bjet77 = 0; n_bjet85 = 0; nljets = 0; nbjets = 0;  
  for(Int_t i = 0; i<*n_jet; i++){
    if(*n_jet >= 2){
      j1.SetPtEtaPhiM(jet_pt[0]/1000., jet_eta[0], jet_phi[0], jet_m[0]/1000.);
      j2.SetPtEtaPhiM(jet_pt[1]/1000., jet_eta[1], jet_phi[1], jet_m[1]/1000.);}
    ajet.SetPtEtaPhiM(jet_pt[i]/1000., jet_eta[i], jet_phi[i], jet_m[i]/1000.);
    
    if(jet_DL1r_score[i]>2.195){n_bjet77++;} 
    if(jet_DL1r_score[i]>0.665){n_bjet85++; nbjets++; } 
    if(jet_DL1r_score[i]<0.665){nljets++;} 

    if(i==0)totalj = ajet;
    else totalj += ajet;
  }

  ll = l1+l2; 
  mll = ll.M(); 
  met = *met_tst_et/1000.;
  met_lor.SetPtEtaPhiE(met, 0.0, *met_tst_phi, 0.0);
  met_sig = *met_tst_significance;

  //=================//
  // Final selection //
  //=================//

  if(dileptons=="ee"){
    if(l1.Pt() < 25 || l2.Pt() < 25 ){return kTRUE;}
    passed_pt30 = 0; 
    if(l1.Pt() > 30 && l1.Pt() > 30 ){passed_pt30=1;}
    n_pt_e++;
    if( fabs(el_cl_etaBE2[lep1])<1.52 && fabs(el_cl_etaBE2[lep1])>1.37 ){return kTRUE;} 
    if( fabs(el_cl_etaBE2[lep2])<1.52 && fabs(el_cl_etaBE2[lep2])>1.37 ){return kTRUE;} 
    n_eta_e++; 
  }
  if(dileptons=="uu"){
    if(l1.Pt() < 27 || l2.Pt() < 20 ){return kTRUE;}
    passed_pt30 = 0; 
    if(l1.Pt() > 30 && l1.Pt() > 30 ){passed_pt30=1;}
    n_pt_u++;  
  }
  if(dileptons=="eu"){
    if(l1.Pt() < 25 || l2.Pt() < 20 ){return kTRUE;}
    passed_pt30 = 0; 
    if(l1.Pt() > 30 && l1.Pt() > 30 ){passed_pt30=1;}
    n_pt_e++; n_pt_u++;  
    if( fabs(el_cl_etaBE2[lep1])<1.52 && fabs(el_cl_etaBE2[lep1])>1.37 ){return kTRUE;} 
    n_eta_e++; 
  }
  if(dileptons=="ue"){
    if(l1.Pt() < 27 || l2.Pt() < 25 ){return kTRUE;}
    passed_pt30 = 0; 
    if(l1.Pt() > 30 && l1.Pt() > 30 ){passed_pt30=1;}
    n_pt_u++;  n_pt_e++;  
    if( fabs(el_cl_etaBE2[lep2])<1.52 && fabs(el_cl_etaBE2[lep2])>1.37 ){return kTRUE;} 
    n_eta_e++;
  }

  if(mll < 10){ return kTRUE; }

  passed_channels = {}; isTreeChannel=0; 

  // if(dileptons=="ee" && met>50){passed_channels.push_back("ee_incl");}   //add met cut?
  // if(dileptons=="uu" && met>50){passed_channels.push_back("uu_incl");} 
  if(dileptons=="ee" && isOS == 0){passed_channels.push_back("ee_SS_incl");}   
  if(dileptons=="uu" && isOS == 0){passed_channels.push_back("uu_SS_incl");} 
  if(dileptons=="eu" && isOS == 0){passed_channels.push_back("eu_SS_incl");}   
  if(dileptons=="ue" && isOS == 0){passed_channels.push_back("ue_SS_incl");} 
  if(dileptons=="ee" && isOS == 1){passed_channels.push_back("ee_OS_incl");}   
  if(dileptons=="uu" && isOS == 1){passed_channels.push_back("uu_OS_incl");} 
  if(dileptons=="eu" && isOS == 1){passed_channels.push_back("eu_OS_incl");}   
  if(dileptons=="ue" && isOS == 1){passed_channels.push_back("ue_OS_incl");} 

  //==============// 
  // Event weight //
  //==============//

  if(isData){
    //Standard weight of 1 for data
    wgt = 1.0; 
  } 
  else{ 
    wgt_mc = *mconly_weight; 
    // Pileup weight
    wgt_pu = *pu_weight;  
    // k-factor
    wgt_kf = *kF_weight; 
    // ttbar reweighting 
    wgt_ttbar = *ttbarNNLOweight;
    // NLO EW correction
    if(isSh2211){ wgt_nlo_ew = mconly_weights[295]/wgt_mc; } 
    else{ wgt_nlo_ew = 1.0; }
    //wgt_nlo_ew = 1.0; 
    // jet weight
    wgt_jet = *jvt_signal_jets_SF; 
    // b-jet weight
    wgt_bjet = *btag_signal_jets_SF; 
    // Cross section  
    xs = *xsec; 
    // Filter efficiency 
    g_eff = *geneff; 
    if( xs < 0 || g_eff < 0 ){ cout << "Error in cross section or filter efficiency!!!" << endl; } 
    // Total cross section weight 
    wgt_xs = xs*wgt_kf*nb2fb*g_eff; 
    // Lepton and trigger scale factor 
    if(dileptons.Contains("ee")){ // ee
      wgt_lsf = *evsf_signal_nominal_EL;
      wgt_lsf *= el_SF_cid[lep1]; 
      wgt_lsf *= el_SF_cid[lep2]; 
    }  
    if(dileptons.Contains("uu")){ // uu
      wgt_lsf = *evsf_signal_nominal_MU;   
    } 
    if(dileptons.Contains("eu")){ // eu
      wgt_lsf = *evsf_signal_nominal_MU;
      wgt_lsf *= *evsf_signal_nominal_EL;    
      wgt_lsf *= el_SF_cid[lep1];
    } 
    if(dileptons.Contains("ue")){ // ue
      wgt_lsf = *evsf_signal_nominal_MU;
      wgt_lsf *= *evsf_signal_nominal_EL;        
      wgt_lsf *= el_SF_cid[lep2];
    } 

    if ( !(tight1 && tight2) ){ wgt = 0.0; } 
    else{ wgt = wgt_mc*wgt_pu*wgt_xs*wgt_lsf*wgt_ttbar*wgt_nlo_ew*wgt_jet*wgt_bjet; } 
    //cout << wgt << endl; 
  
  }

  //=================//
  // Fill histograms // 
  //=================//

  for(const auto & chn:passed_channels){
    TString this_name = chn; 
    h_pt1[this_name]->Fill(l1.Pt(), wgt); 
    h_pt2[this_name]->Fill(l2.Pt(), wgt); 
    h_eta1[this_name]->Fill(l1.Eta(), wgt); 
    h_eta2[this_name]->Fill(l2.Eta(), wgt); 
    h_mll[this_name]->Fill(mll, wgt); 
    h_met[this_name]->Fill(met, wgt);  
    h_met_sig[this_name]->Fill(met_sig, wgt); 
    h_mt[this_name]->Fill(ll.Mt(), wgt);
    h_mt2[this_name]->Fill(ComputeMT2(l1,l2,met_lor,0.,0.).Compute(), wgt);
    h_ht[this_name]->Fill((ll+totalj).Pt(), wgt);
    h_et[this_name]->Fill(ll.Et(), wgt);   
    h_phi1[this_name]->Fill(l1.Phi(), wgt); 
    h_phi2[this_name]->Fill(l2.Phi(), wgt); 
    h_dPhiLeps[this_name]->Fill(l1.DeltaPhi(l2), wgt); 
    h_dPhiLLmet[this_name]->Fill(ll.DeltaPhi(met_lor), wgt); 
    h_nBJet[this_name]->Fill(nbjets, wgt);
    h_nLJet[this_name]->Fill(nljets, wgt);
    h_nTJet[this_name]->Fill(nbjets + nljets, wgt);
    h_jetpt1[this_name]->Fill(j1.Pt(), wgt);
    h_jetpt2[this_name]->Fill(j2.Pt(), wgt);
    h_jeteta1[this_name]->Fill(j1.Eta(), wgt);
    h_jeteta2[this_name]->Fill(j2.Eta(), wgt);
    h_jetphi1[this_name]->Fill(j1.Phi(), wgt);
    h_jetphi2[this_name]->Fill(j2.Phi(), wgt);
    if(l1.Pt() > l2.Pt()){
      h_dPhiLepMet[this_name]->Fill(l1.DeltaPhi(met_lor), wgt);}
    else{
      h_dPhiLepMet[this_name]->Fill(l2.DeltaPhi(met_lor), wgt);}
  }
  
  // ML FILE
  MY->bMY_channel = (DSID);  
  MY->bMY_weight = (wgt);  
  MY->bMY_lep1Pt = (l1.Pt());  
  MY->bMY_lep1Eta = (l1.Eta());  
  MY->bMY_lep1Phi = (l1.Phi()); 
  MY->bMY_lep1Et = (l1.Et());
  MY->bMY_lep2Pt = (l2.Pt());  
  MY->bMY_lep2Eta = (l2.Eta());  
  MY->bMY_lep2Phi = (l2.Phi()); 
  MY->bMY_lep2Et = (l2.Et());
  MY->bMY_jetB = (nbjets);  
  MY->bMY_jetLight = (nljets);   
  MY->bMY_jetTot = (nbjets+nljets);  
  if(*n_jet >=2){
  MY->bMY_jet1Pt = (j1.Pt());  
  MY->bMY_jet1Eta = (j1.Eta());  
  MY->bMY_jet1Phi = (j1.Phi());
  MY->bMY_jet2Pt = (j2.Pt());  
  MY->bMY_jet2Eta = (j2.Eta());  
  MY->bMY_jet2Phi = (j2.Phi());}
  else{
  MY->bMY_jet1Pt = (0);  
  MY->bMY_jet1Eta = (10);  
  MY->bMY_jet1Phi = (10);
  MY->bMY_jet2Pt = (0);  
  MY->bMY_jet2Eta = (10);  
  MY->bMY_jet2Phi = (10);}

  MY->bMY_met_Et = (*met_tst_et/1000);  
  // MY->bMY_met_Phi = (*met_tst_phi);  
  MY->bMY_mll = (mll);
  MY->bMY_EventNumber = (*event);
  MY->bMY_RunNumber = (*run);  
  // MY->bMY_Label = (label); // HOW  
  MY->bMY_RunPeriod = (dataset);  

  MY->WriteTree();
  return kTRUE;
}

void EventSelector::SlaveTerminate(){}

void EventSelector::Terminate()
{ 
  if(isData){  
    WriteToFile(period, dataset, final_name); 
  }
  else{ WriteToFile(Form("%d", DSID), dataset, final_name); } 
  cout << "Events processed: " << nevents << endl; 
  //cout << "Minimum MET for events with METsig>5.: " << min_MET << endl; 

  delete MY;  //ML FILE
}


void EventSelector::WriteToFile(TString fileid, TString data_type, TString name)
{
  
  // Make outfile   
  TString outputdir, filename, filename2, filename_syst, filename_fakes, h_name; 
  if(fileid.Contains("period")){ 
    filename = "hist."+data_type+"."+name+".root"; 
  } 
  else{ 
    filename = "hist."+fileid+"."+name+".root"; 
  }

  outputdir="Histograms"; 
  
  TFile file(outputdir+"/"+data_type+"/"+fileid+"/"+filename, "RECREATE"); 
  
  // Write nominal histograms 
  for( const auto & chn:all_channels ){
    h_name = chn; 
    h_pt1[h_name]->Write(); 
    h_pt2[h_name]->Write(); 
    h_eta1[h_name]->Write(); 
    h_eta2[h_name]->Write(); 
    h_mll[h_name]->Write(); 
    h_met[h_name]->Write(); 
    h_met_sig[h_name]->Write(); 
    h_mt[h_name]->Write(); 
    h_mt2[h_name]->Write(); 
    h_ht[h_name]->Write(); 
    h_et[h_name]->Write(); 
    h_phi1[h_name]->Write(); 
    h_phi2[h_name]->Write();
    h_dPhiLeps[h_name]->Write();
    h_dPhiLLmet[h_name]->Write();
    h_dPhiLepMet[h_name]->Write();
    h_nBJet[h_name]->Write();
    h_nLJet[h_name]->Write();
    h_nTJet[h_name]->Write();
    h_jetpt1[h_name]->Write();
    h_jetpt2[h_name]->Write();
    h_jeteta1[h_name]->Write();
    h_jeteta2[h_name]->Write();
    h_jetphi1[h_name]->Write();
    h_jetphi2[h_name]->Write();
    }

  // Reset histograms
  for( const auto & chn:all_channels ){ 
    h_name = chn; 
    h_pt1[h_name]->Reset(); 
    h_pt2[h_name]->Reset(); 
    h_eta1[h_name]->Reset(); 
    h_eta2[h_name]->Reset(); 
    h_mll[h_name]->Reset();
    h_met[h_name]->Reset(); 
    h_met_sig[h_name]->Reset(); 
    h_mt[h_name]->Reset(); 
    h_mt2[h_name]->Reset(); 
    h_ht[h_name]->Reset(); 
    h_et[h_name]->Reset(); 
    h_phi1[h_name]->Reset(); 
    h_phi2[h_name]->Reset(); 
    h_dPhiLeps[h_name]->Reset(); 
    h_dPhiLepMet[h_name]->Reset(); 
    h_dPhiLLmet[h_name]->Reset(); 
    h_nBJet[h_name]->Reset(); 
    h_nLJet[h_name]->Reset(); 
    h_nTJet[h_name]->Reset(); 
    h_jetpt1[h_name]->Reset(); 
    h_jetpt2[h_name]->Reset(); 
    h_jeteta1[h_name]->Reset(); 
    h_jeteta2[h_name]->Reset(); 
    h_jetphi1[h_name]->Reset(); 
    h_jetphi2[h_name]->Reset(); 
  }
  
  cout << "Done with file: " << name << endl; 

  time_t t; // t passed as argument in function time()
  struct tm * tt; // decalring variable for localtime()
  time (&t); //passing argument to time()
  tt = localtime(&t);
  cout << "Time: "<< asctime(tt);  

}


void EventSelector::PrintCutflow(Int_t fileid){

  cout.precision(17); 
  cout << "===============================================" << endl; 
  cout << "CUTFLOW AND WEIGHT SUMS" << endl; 
  cout << "Run number / DSID: " << fileid << endl; 
  cout << "Total number of events: " << sample_nevents << endl; 
  cout << "==================" << endl; 
  cout << "MUON CHANNEL" << endl; 
  cout << "Passed muon triggers: " << n_trig_u << endl; 
  cout << "Jet cleaning: " << n_jetclean_u << endl; 
  cout << "Exactly two loose muons: " << n_loose_u << endl; 
  cout << "Bad muon veto: " << n_mubad << endl; 
  cout << "pT > 30 GeV: " << n_pt_u << endl; 
  cout << "mll > 70 GeV: " << n_mll70_u << endl; 
  cout << "mll > 70 GeV (Tight+Tight): " << n_mll70_u_TT << endl; 
  cout << "mll < 120 GeV: " << n_mll120_u << endl; 
  cout << "met > 100 GeV: " << n_met100_u << endl; 
  cout << "-----------------" << endl; 
  cout << "MC weights: " << s_mc_u << endl; 
  cout << "kFactors: " << s_kf_u << endl; 
  cout << "Pile-up: " << s_pu_u << endl; 
  cout << "TTbar weight: " << s_tt_u << endl; 
  cout << "XS weight: " << s_xs_u << endl; 
  cout << "Lepton SF: " << s_lsf_u << endl; 
  cout << "Trigger SF: " << s_trig_u << endl; 
  cout << "Combined: " << s_wgt_u << endl; 
  cout << "==================" << endl; 
  cout << "ELECTRON CHANNEL" << endl; 
  cout << "Passed electron triggers: " << n_trig_e << endl; 
  cout << "Jet cleaning: " << n_jetclean_e << endl; 
  cout << "Exactly two loose electrons: " << n_loose_e << endl; 
  cout << "pT > 30 GeV: " << n_pt_e << endl; 
  cout << "Eta crack veto: " << n_eta_e << endl; 
  cout << "mll > 70 GeV: " << n_mll70_e << endl; 
  cout << "mll > 70 GeV (Tight+Tight): " << n_mll70_e_TT << endl; 
  cout << "-----------------" << endl; 
  cout << "MC weights: " << s_mc_e << endl; 
  cout << "kFactors: " << s_kf_e << endl; 
  cout << "Pile-up: " << s_pu_e << endl; 
  cout << "TTbar weight: " << s_tt_e << endl; 
  cout << "XS weight: " << s_xs_e << endl; 
  cout << "Lepton SF: " << s_lsf_e << endl; 
  cout << "Trigger SF: " << s_trig_e << endl; 
  cout << "Combined: " << s_wgt_e << endl; 

  sample_nevents = 0; 
  n_trig_e = 0; n_jetclean_e = 0; n_twolep_e = 0; n_loose_e = 0; n_pt_e = 0; n_eta_e = 0; n_mll70_e = 0; n_mll70_e_TT = 0; 
  n_trig_u = 0; n_jetclean_u = 0; n_twolep_u = 0; n_loose_u = 0; n_mubad = 0; n_pt_u = 0; n_mll70_u = 0; n_mll70_u_TT = 0;
  s_mc_e = 0; s_kf_e = 0; s_pu_e = 0; s_lsf_e = 0; s_trig_e = 0; s_tt_e = 0; s_wgt_e = 0;     
  s_mc_u = 0; s_kf_u = 0; s_pu_u = 0; s_lsf_u = 0; s_trig_u = 0; s_tt_u = 0; s_wgt_u = 0;     

} 

vector<Float_t> EventSelector::GetLogBinning(Int_t nbins, Float_t xmin, Float_t xmax){

  Float_t logmin = log10(xmin);
  Float_t logmax = log10(xmax);
  Float_t logbinwidth = (logmax-logmin)/nbins;

  vector<Float_t> xbins;
  xbins.push_back(xmin);
  for(int i=1; i<nbins+1; i++){
    xbins.push_back( TMath::Power(10, (logmin + i*logbinwidth)) );
  }

  return xbins;
}

vector<Float_t> EventSelector::GetSRBinning(Float_t xmin, Float_t xmax){
  
  vector<Float_t> xbins = GetLogBinning(55, xmin, 2000); // "Standard" log binning up to 650 GeV 

  xbins.push_back(5000); 

  /*
  Float_t bin_low = 2000; 
  Float_t bin_high; 

  while(bin_low<xmax){
    if(bin_low<1000){ bin_high = bin_low+50; } 
    if(bin_low>=1000 && bin_low<2000){ bin_high = bin_low+100; } 
    if(bin_low>=2000){ bin_high = 5000; } 
    xbins.push_back( bin_high );
    bin_low = bin_high; 
  }
  */ 
  
  return xbins;
}

map<TString, Float_t> EventSelector::GetLepSFVariations(Long64_t entry){

  map<TString, Float_t> lep_SF; 
  //lep_SF["EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1up"] = *evsf_signal0_syst_EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1up;
  lep_SF["EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1up"] = *evsf_signal0_syst_EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1up; 
  lep_SF["EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1up"] = *evsf_signal0_syst_EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1up;  
  lep_SF["EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1up"] =  *evsf_signal0_syst_EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1up; 
  lep_SF["EL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR__1up"] =  *evsf_signal0_syst_EL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR__1up;
  lep_SF["EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR__1up"] =  *evsf_signal0_syst_EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR__1up;
  lep_SF["MUON_EFF_BADMUON_SYS__1up"] =  *evsf_signal0_syst_MUON_EFF_BADMUON_SYS__1up;
  lep_SF["MUON_EFF_ISO_STAT__1up"] =  *evsf_signal0_syst_MUON_EFF_ISO_STAT__1up;
  lep_SF["MUON_EFF_ISO_SYS__1up"] =  *evsf_signal0_syst_MUON_EFF_ISO_SYS__1up;
  lep_SF["MUON_EFF_RECO_STAT__1up"] =  *evsf_signal0_syst_MUON_EFF_RECO_STAT__1up;
  lep_SF["MUON_EFF_RECO_SYS__1up"] = *evsf_signal0_syst_MUON_EFF_RECO_SYS__1up;
  lep_SF["MUON_EFF_TTVA_STAT__1up"] = *evsf_signal0_syst_MUON_EFF_TTVA_STAT__1up;
  lep_SF["MUON_EFF_TTVA_SYS__1up"] = *evsf_signal0_syst_MUON_EFF_TTVA_SYS__1up;
  lep_SF["MUON_EFF_TrigStatUncertainty__1up"] = *evsf_signal0_syst_MUON_EFF_TrigStatUncertainty__1up;
  lep_SF["MUON_EFF_TrigSystUncertainty__1up"] = *evsf_signal0_syst_MUON_EFF_TrigSystUncertainty__1up;

  //lep_SF["EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1down"] = *evsf_signal0_syst_EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1down;
  lep_SF["EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1down"] = *evsf_signal0_syst_EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1down; 
  lep_SF["EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1down"] = *evsf_signal0_syst_EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1down;  
  lep_SF["EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1down"] =  *evsf_signal0_syst_EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1down; 
  lep_SF["EL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR__1down"] =  *evsf_signal0_syst_EL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR__1down;
  lep_SF["EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR__1down"] =  *evsf_signal0_syst_EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR__1down;
  lep_SF["MUON_EFF_BADMUON_SYS__1down"] =  *evsf_signal0_syst_MUON_EFF_BADMUON_SYS__1down;
  lep_SF["MUON_EFF_ISO_STAT__1down"] =  *evsf_signal0_syst_MUON_EFF_ISO_STAT__1down;
  lep_SF["MUON_EFF_ISO_SYS__1down"] =  *evsf_signal0_syst_MUON_EFF_ISO_SYS__1down;
  lep_SF["MUON_EFF_RECO_STAT__1down"] =  *evsf_signal0_syst_MUON_EFF_RECO_STAT__1down;
  lep_SF["MUON_EFF_RECO_SYS__1down"] = *evsf_signal0_syst_MUON_EFF_RECO_SYS__1down;
  lep_SF["MUON_EFF_TTVA_STAT__1down"] = *evsf_signal0_syst_MUON_EFF_TTVA_STAT__1down;
  lep_SF["MUON_EFF_TTVA_SYS__1down"] = *evsf_signal0_syst_MUON_EFF_TTVA_SYS__1down;
  lep_SF["MUON_EFF_TrigStatUncertainty__1down"] = *evsf_signal0_syst_MUON_EFF_TrigStatUncertainty__1down;
  lep_SF["MUON_EFF_TrigSystUncertainty__1down"] = *evsf_signal0_syst_MUON_EFF_TrigSystUncertainty__1down;

  return lep_SF; 

}

map<TString, Float_t> EventSelector::GetCidSFVariations(Long64_t entry){

  map<TString, Float_t> cid_SF; 
  cid_SF["EL_CHARGEID_STAT__1down"] = *SF_cid_syst_EL_CHARGEID_STAT__1down; 
  cid_SF["EL_CHARGEID_STAT__1up"] = *SF_cid_syst_EL_CHARGEID_STAT__1up; 
  cid_SF["EL_CHARGEID_SYStotal__1down"] = *SF_cid_syst_EL_CHARGEID_SYStotal__1down; 
  cid_SF["EL_CHARGEID_SYStotal__1up"] = *SF_cid_syst_EL_CHARGEID_SYStotal__1up; 
  cid_SF["EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1down"] = *SF_cid_syst_EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1down; 
  cid_SF["EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1up"] = *SF_cid_syst_EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1up;  
  
  return cid_SF; 

}

map<TString, Float_t> EventSelector::GetJetSFVariations(Long64_t entry){

  map<TString, Float_t> jet_SF; 

  jet_SF["FT_EFF_B_systematics__1down"]= *btag_signal_jets_SF_syst_FT_EFF_B_systematics__1down;     
  jet_SF["FT_EFF_B_systematics__1up"]= *btag_signal_jets_SF_syst_FT_EFF_B_systematics__1up;       
  jet_SF["FT_EFF_C_systematics__1down"]= *btag_signal_jets_SF_syst_FT_EFF_C_systematics__1down;     
  jet_SF["FT_EFF_C_systematics__1up"]= *btag_signal_jets_SF_syst_FT_EFF_C_systematics__1up;       
  jet_SF["FT_EFF_Light_systematics__1down"]= *btag_signal_jets_SF_syst_FT_EFF_Light_systematics__1down;
  jet_SF["FT_EFF_Light_systematics__1up"]= *btag_signal_jets_SF_syst_FT_EFF_Light_systematics__1up; 
  jet_SF["JET_JvtEfficiency__1down"]= *jvt_signal_jets_SF_syst_JET_JvtEfficiency__1down; 
  jet_SF["JET_JvtEfficiency__1up"]= *jvt_signal_jets_SF_syst_JET_JvtEfficiency__1up;     
  jet_SF["JET_fJvtEfficiency__1down"]= *fjvt_signal_jets_SF_syst_JET_fJvtEfficiency__1down; 
  jet_SF["JET_fJvtEfficiency__1up"]= *fjvt_signal_jets_SF_syst_JET_fJvtEfficiency__1up;
  
  return jet_SF; 

}



vector<Double_t> EventSelector::GetEffBinning(){

  vector<Double_t> xbins;
  xbins.push_back(0.0);

  Double_t binedge = 10.0;
  xbins.push_back(binedge);

  while(binedge < 10000.0){

    Double_t delta;
    if (binedge < 50.0)
      delta = 1.0;
    else if (binedge < 100.0)
      delta = 1.0;
    else if (binedge < 120.0)
      delta = 2.5;
    else if (binedge < 200.0)
      delta = 10.0;
    else if (binedge < 1000.0)
      delta = 50.0;
    else
      delta = 1000.0;

    binedge += delta;
    xbins.push_back(binedge);
  }

  return xbins;
}

