/////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Jul 28 13:33:11 2020 by ROOT version 6.20/06
// from TTree nominal/
// found on file: minitree_nominal_final.root
//////////////////////////////////////////////////////////

#ifndef EventSelector_h
#define EventSelector_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TObjString.h>
#include <TSelector.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <TH2.h>
#include "makeMYTree.h"

// Headers needed by this particular selector
#include <vector>

class EventSelector : public TSelector {
public :
  TTreeReader     fReader;  //!the tree reader
  TTreeReader     fReader_MC; //! tree reader for MC specific branches  
  TTreeReader     fReader_sig; //! tree reader for signal specific branches (temporary)  
  TTreeReader     fReader_nom; //! tree reader for branches only in nominal samples
  TTreeReader     fReader_sys; //! tree reader for branches only relevant with systematics 
  TTree          *fChain = 0;   //!pointer to the analyzed TTree or TChain
  makeMYTree     *MY;

  // 1D histograms I am interested in plotting
  map<TString, TH1*> h_pt1; //!
  map<TString, TH1*> h_pt2; //!    
  map<TString, TH1*> h_eta1; //!
  map<TString, TH1*> h_eta2; //!    
  map<TString, TH1*> h_phi1; //!
  map<TString, TH1*> h_phi2; //!    
  map<TString, TH1*> h_met; //! 
  map<TString, TH1*> h_met_sig; //! 
  map<TString, TH1*> h_mll; //!  
  map<TString, TH1*> h_mt; //!    
  map<TString, TH1*> h_mt2; //!    
  map<TString, TH1*> h_ht; //!    
  map<TString, TH1*> h_et; //!
  map<TString, TH1*> h_dPhiLeps; //!
  map<TString, TH1*> h_dPhiLLmet; //!
  map<TString, TH1*> h_dPhiCloseMet; //!
  map<TString, TH1*> h_dPhiLeadMet; //!
  map<TString, TH1*> h_nBJet; //!
  map<TString, TH1*> h_nLJet; //!
  map<TString, TH1*> h_nTJet; //!
  map<TString, TH1*> h_jetpt1; //!
  map<TString, TH1*> h_jetpt2; //!
  map<TString, TH1*> h_jeteta1; //!
  map<TString, TH1*> h_jeteta2; //!
  map<TString, TH1*> h_jetphi1; //!
  map<TString, TH1*> h_jetphi2; //!

  map<TString, TH1*> h_n_bjetsPt20; //!
  map<TString, TH1*> h_n_bjetsPt30; //!
  map<TString, TH1*> h_n_bjetsPt40; //!
  map<TString, TH1*> h_n_bjetsPt50; //!
  map<TString, TH1*> h_n_bjetsPt60; //!
  map<TString, TH1*> h_n_ljetsPt20; //!
  map<TString, TH1*> h_n_ljetsPt30; //!
  map<TString, TH1*> h_n_ljetsPt40; //!
  map<TString, TH1*> h_n_ljetsPt50; //!
  map<TString, TH1*> h_n_ljetsPt60; //!
  map<TString, TH1*> h_jetEtaCentral; //!
  map<TString, TH1*> h_jetEtaForward; //!

  // Readers to access the data (delete the ones you do not need).
  TTreeReaderValue<Float_t> mconly_weight = {fReader, "mconly_weight"};
  TTreeReaderArray<float> mconly_weights = {fReader, "mconly_weights"};
  TTreeReaderValue<Float_t> kF_weight = {fReader, "kF_weight"};
  TTreeReaderValue<Float_t> ttbarNNLOweight = {fReader, "ttbarNNLOweight"};
  TTreeReaderValue<Float_t> xsec = {fReader, "xsec"};
  TTreeReaderValue<Float_t> geneff = {fReader, "geneff"};
  TTreeReaderValue<Float_t> pu_weight = {fReader, "pu_weight"};
  TTreeReaderValue<Int_t> year = {fReader, "year"};
  TTreeReaderValue<Int_t> run = {fReader, "run"};
  TTreeReaderValue<ULong64_t> event = {fReader, "event"};

  // Jet SF systematics 
  TTreeReaderValue<Float_t> btag_signal_jets_SF_syst_FT_EFF_B_systematics__1down = {fReader_sys, "btag_signal_jets_SF_syst_FT_EFF_B_systematics__1down"};
  TTreeReaderValue<Float_t> btag_signal_jets_SF_syst_FT_EFF_B_systematics__1up = {fReader_sys, "btag_signal_jets_SF_syst_FT_EFF_B_systematics__1up"};
  TTreeReaderValue<Float_t> btag_signal_jets_SF_syst_FT_EFF_C_systematics__1down = {fReader_sys, "btag_signal_jets_SF_syst_FT_EFF_C_systematics__1down"};
  TTreeReaderValue<Float_t> btag_signal_jets_SF_syst_FT_EFF_C_systematics__1up = {fReader_sys, "btag_signal_jets_SF_syst_FT_EFF_C_systematics__1up"};
  TTreeReaderValue<Float_t> btag_signal_jets_SF_syst_FT_EFF_Light_systematics__1down = {fReader_sys, "btag_signal_jets_SF_syst_FT_EFF_Light_systematics__1down"};
  TTreeReaderValue<Float_t> btag_signal_jets_SF_syst_FT_EFF_Light_systematics__1up = {fReader_sys, "btag_signal_jets_SF_syst_FT_EFF_Light_systematics__1up"};
  TTreeReaderValue<Float_t> jvt_signal_jets_SF_syst_JET_JvtEfficiency__1down = {fReader_sys, "jvt_signal_jets_SF_syst_JET_JvtEfficiency__1down"};
  TTreeReaderValue<Float_t> jvt_signal_jets_SF_syst_JET_JvtEfficiency__1up = {fReader_sys, "jvt_signal_jets_SF_syst_JET_JvtEfficiency__1up"};
  TTreeReaderValue<Float_t> fjvt_signal_jets_SF_syst_JET_fJvtEfficiency__1down = {fReader_sys, "fjvt_signal_jets_SF_syst_JET_fJvtEfficiency__1down"};
  TTreeReaderValue<Float_t> fjvt_signal_jets_SF_syst_JET_fJvtEfficiency__1up = {fReader_sys, "fjvt_signal_jets_SF_syst_JET_fJvtEfficiency__1up"};


  TTreeReaderValue<Float_t> btag_signal_jets_SF = {fReader, "btag_signal_jets_SF"};
  TTreeReaderValue<Float_t> jvt_signal_jets_SF = {fReader, "jvt_signal_jets_SF"};
  TTreeReaderValue<Float_t> bornMass = {fReader, "bornMass"};
  TTreeReaderValue<Int_t> jetCleaning_eventClean = {fReader, "jetCleaning_eventClean"};
  TTreeReaderValue<Float_t> evsf_signal_nominal_EL = {fReader, "evsf_signal_nominal_EL"};
  TTreeReaderValue<Float_t> evsf_signal_nominal_MU = {fReader, "evsf_signal_nominal_MU"};

  // Lepton SF systematics 
  TTreeReaderValue<Float_t> SF_cid_syst_EL_CHARGEID_STAT__1down = {fReader_sys, "SF_cid_syst_EL_CHARGEID_STAT__1down"};
  TTreeReaderValue<Float_t> SF_cid_syst_EL_CHARGEID_STAT__1up = {fReader_sys, "SF_cid_syst_EL_CHARGEID_STAT__1up"};
  TTreeReaderValue<Float_t> SF_cid_syst_EL_CHARGEID_SYStotal__1down = {fReader_sys, "SF_cid_syst_EL_CHARGEID_SYStotal__1down"};
  TTreeReaderValue<Float_t> SF_cid_syst_EL_CHARGEID_SYStotal__1up = {fReader_sys, "SF_cid_syst_EL_CHARGEID_SYStotal__1up"};
  TTreeReaderValue<Float_t> SF_cid_syst_EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1down = {fReader_sys, "SF_cid_syst_EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1down"};
  TTreeReaderValue<Float_t> SF_cid_syst_EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1up = {fReader_sys, "SF_cid_syst_EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1down = {fReader_sys, "evsf_signal0_syst_EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1up = {fReader_sys, "evsf_signal0_syst_EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1down = {fReader_sys, "evsf_signal0_syst_EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1up = {fReader_sys, "evsf_signal0_syst_EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1down = {fReader_sys, "evsf_signal0_syst_EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1up = {fReader_sys, "evsf_signal0_syst_EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_EL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR__1down = {fReader_sys, "evsf_signal0_syst_EL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_EL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR__1up = {fReader_sys, "evsf_signal0_syst_EL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR__1down = {fReader_sys, "evsf_signal0_syst_EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR__1up = {fReader_sys, "evsf_signal0_syst_EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_BADMUON_SYS__1down = {fReader_sys, "evsf_signal0_syst_MUON_EFF_BADMUON_SYS__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_BADMUON_SYS__1up = {fReader_sys, "evsf_signal0_syst_MUON_EFF_BADMUON_SYS__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_ISO_STAT__1down = {fReader_sys, "evsf_signal0_syst_MUON_EFF_ISO_STAT__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_ISO_STAT__1up = {fReader_sys, "evsf_signal0_syst_MUON_EFF_ISO_STAT__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_ISO_SYS__1down = {fReader_sys, "evsf_signal0_syst_MUON_EFF_ISO_SYS__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_ISO_SYS__1up = {fReader_sys, "evsf_signal0_syst_MUON_EFF_ISO_SYS__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_RECO_STAT__1down = {fReader_sys, "evsf_signal0_syst_MUON_EFF_RECO_STAT__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_RECO_STAT__1up = {fReader_sys, "evsf_signal0_syst_MUON_EFF_RECO_STAT__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_RECO_SYS__1down = {fReader_sys, "evsf_signal0_syst_MUON_EFF_RECO_SYS__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_RECO_SYS__1up = {fReader_sys, "evsf_signal0_syst_MUON_EFF_RECO_SYS__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_TTVA_STAT__1down = {fReader_sys, "evsf_signal0_syst_MUON_EFF_TTVA_STAT__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_TTVA_STAT__1up = {fReader_sys, "evsf_signal0_syst_MUON_EFF_TTVA_STAT__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_TTVA_SYS__1down = {fReader_sys, "evsf_signal0_syst_MUON_EFF_TTVA_SYS__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_TTVA_SYS__1up = {fReader_sys, "evsf_signal0_syst_MUON_EFF_TTVA_SYS__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_TrigStatUncertainty__1down = {fReader_sys, "evsf_signal0_syst_MUON_EFF_TrigStatUncertainty__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_TrigStatUncertainty__1up = {fReader_sys, "evsf_signal0_syst_MUON_EFF_TrigStatUncertainty__1up"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_TrigSystUncertainty__1down = {fReader_sys, "evsf_signal0_syst_MUON_EFF_TrigSystUncertainty__1down"};
  TTreeReaderValue<Float_t> evsf_signal0_syst_MUON_EFF_TrigSystUncertainty__1up = {fReader_sys, "evsf_signal0_syst_MUON_EFF_TrigSystUncertainty__1up"};

  // MET variables 
  TTreeReaderValue<Float_t> met_tst_et = {fReader, "met_tst_et"};
  TTreeReaderValue<Float_t> met_tst_phi = {fReader, "met_tst_phi"};
  TTreeReaderValue<Float_t> met_tst_significance = {fReader, "met_tst_significance"}; 

  // Muon variables 
  TTreeReaderArray<float> mu_pt = {fReader, "mu_pt"};
  TTreeReaderArray<float> mu_eta = {fReader, "mu_eta"};
  TTreeReaderArray<float> mu_phi = {fReader, "mu_phi"};
  TTreeReaderArray<float> mu_charge = {fReader, "mu_charge"};
  TTreeReaderArray<int> mu_passIso = {fReader, "mu_passIso"};
  TTreeReaderArray<float> mu_SF_iso = {fReader, "mu_SF_iso"};
  TTreeReaderArray<float> mu_SF_rec = {fReader, "mu_SF_rec"};
  TTreeReaderArray<int> mu_isBad = {fReader, "mu_isBad"};
  
  // Electron variables 
  TTreeReaderArray<float> el_pt = {fReader, "el_pt"};
  TTreeReaderArray<float> el_eta = {fReader, "el_eta"};
  TTreeReaderArray<float> el_cl_etaBE2 = {fReader, "el_cl_etaBE2"};
  TTreeReaderArray<float> el_phi = {fReader, "el_phi"};
  TTreeReaderArray<int> el_passLHID = {fReader, "el_passLHID"};
  TTreeReaderArray<int> el_passIso = {fReader, "el_passIso"};
  TTreeReaderArray<float> el_SF_cid = {fReader, "el_SF_cid"};
  TTreeReaderArray<float> el_charge = {fReader, "el_charge"};
  
  // Jet variables 
  TTreeReaderArray<float> jet_pt = {fReader, "jet_pt"};
  TTreeReaderArray<float> jet_eta = {fReader, "jet_eta"};
  TTreeReaderArray<float> jet_phi = {fReader, "jet_phi"};
  TTreeReaderArray<float> jet_m = {fReader, "jet_m"};
  TTreeReaderArray<float> jet_DL1r_score = {fReader, "jet_DL1r_score"};
  TTreeReaderArray<float> jet_jvt = {fReader, "jet_jvt"};
  TTreeReaderArray<float> jet_fjvt = {fReader, "jet_fjvt"};
  

  //Trigger selector non-susy
  TTreeReaderValue<Int_t> trigger_HLT_2e12_lhloose_L12EM10VH = {fReader, "trigger_HLT_2e12_lhloose_L12EM10VH"};
  TTreeReaderValue<Int_t> trigger_HLT_2e17_lhvloose_nod0 = {fReader, "trigger_HLT_2e17_lhvloose_nod0"};
  TTreeReaderValue<Int_t> trigger_HLT_2e17_lhvloose_nod0_L12EM15VHI = {fReader, "trigger_HLT_2e17_lhvloose_nod0_L12EM15VHI"};
  TTreeReaderValue<Int_t> trigger_HLT_2e24_lhvloose_nod0 = {fReader, "trigger_HLT_2e24_lhvloose_nod0"};
  TTreeReaderValue<Int_t> trigger_HLT_e120_lhvloose_nod0 = {fReader, "trigger_HLT_e120_lhvloose_nod0"};
  TTreeReaderValue<Int_t> trigger_HLT_e140_lhvloose_nod0 = {fReader, "trigger_HLT_e140_lhvloose_nod0"};
  TTreeReaderValue<Int_t> trigger_HLT_e200_etcut = {fReader, "trigger_HLT_e200_etcut"};
  TTreeReaderValue<Int_t> trigger_HLT_e26_lhvloose_nod0_L1EM20VH = {fReader, "trigger_HLT_e26_lhvloose_nod0_L1EM20VH"};
  TTreeReaderValue<Int_t> trigger_HLT_e300_etcut = {fReader, "trigger_HLT_e300_etcut"};
  TTreeReaderValue<Int_t> trigger_HLT_e60_lhvloose_nod0 = {fReader, "trigger_HLT_e60_lhvloose_nod0"};
  TTreeReaderValue<Int_t> trigger_HLT_mu26_imedium = {fReader, "trigger_HLT_mu26_imedium"};
  TTreeReaderValue<Int_t> trigger_HLT_mu26_ivarmedium = {fReader, "trigger_HLT_mu26_ivarmedium"};
  TTreeReaderValue<Int_t> trigger_HLT_mu50 = {fReader, "trigger_HLT_mu50"};
  TTreeReaderArray<float> mu_d0sig = {fReader, "mu_d0sig"};
  TTreeReaderArray<float> mu_z0sinTheta = {fReader, "mu_z0sinTheta"};
  TTreeReaderArray<float> el_d0sig = {fReader, "el_d0sig"};
  TTreeReaderArray<float> el_z0sinTheta = {fReader, "el_z0sinTheta"};
  TTreeReaderValue<Int_t> n_el = {fReader, "n_el"};
  TTreeReaderValue<Int_t> n_mu = {fReader, "n_mu"};


  // // New variables
  // TTreeReaderValue<Int_t> pass_ee_trig = {fReader, "pass_ee_trig"};  
  // TTreeReaderValue<Int_t> pass_uu_trig = {fReader, "pass_uu_trig"};
  // TTreeReaderValue<Int_t> pass_e_trig = {fReader, "pass_e_trig"};
  // TTreeReaderValue<Int_t> pass_u_trig = {fReader, "pass_u_trig"};
  // TTreeReaderArray<int> mu_passTTVA = {fReader, "mu_passTTVA"};
  // TTreeReaderArray<int> el_passTTVA = {fReader, "el_passTTVA"};
  
  EventSelector(TTree * /*tree*/ =0) { }
  virtual ~EventSelector() { }
  virtual Int_t   Version() const { return 2; }
  virtual void    Begin(TTree *tree);
  virtual void    SlaveBegin(TTree *tree);
  virtual void    Init(TTree *tree);
  virtual Bool_t  Notify();
  virtual Bool_t  Process(Long64_t entry);
  virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
  virtual void    SetOption(const char *option) { fOption = option; }
  virtual void    SetObject(TObject *obj) { fObject = obj; }
  virtual void    SetInputList(TList *input) { fInput = input; }
  virtual TList  *GetOutputList() const { return fOutput; }
  virtual void    SlaveTerminate();
  virtual void    Terminate();
  void WriteToFile(TString fileid, TString option, TString name);
  void PrintCutflow(Int_t fileid); 
  vector<Float_t> GetLogBinning(Int_t nbins, Float_t xmin, Float_t xmax); 
  vector<Float_t> GetSRBinning(Float_t xmin, Float_t xmax); 
  map<TString, Float_t> GetLepSFVariations(Long64_t entry); 
  map<TString, Float_t> GetCidSFVariations(Long64_t entry); 
  map<TString, Float_t> GetJetSFVariations(Long64_t entry); 
  vector<Double_t> GetEffBinning(); 
  ClassDef(EventSelector,0);

};

#endif

#ifdef EventSelector_cxx
void EventSelector::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the reader is initialized.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

  TString option = GetOption(); 
  TObjArray *my_option = option.Tokenize("_");
  int doTruth = atoi(((TObjString *)(my_option->At(1)))->String());
  int doSyst = atoi(((TObjString *)(my_option->At(3)))->String());

  fReader.SetTree(tree);
  if(option.Contains("mc")){ fReader_MC.SetTree(tree); }
  if(doTruth){ fReader_sig.SetTree(tree); }
  if(!doSyst){ fReader_nom.SetTree(tree); } 
  if(doSyst){ fReader_sys.SetTree(tree); } 
  fChain = tree; 
}

Bool_t EventSelector::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

  return kTRUE;
}


#endif // #ifdef EventSelector_cxx
