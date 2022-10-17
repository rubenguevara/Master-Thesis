//------------------------------------------
//   makeMYTrees.h
//   definitions for core analysis
//
//
//   author: Lukas Marti
//-------------------------------------------
#ifndef makeMYTree_h
#define makeMYTree_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TString.h>
#include <TH1F.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "TLorentzVector.h"
#include "TParameter.h"
//#include "makeMYTree/MultiLepEvent.h"


class makeMYTree  {

 public:

  // Constructor
  // - MCID => will be used to name the tree
  // - syst => will be used to name the file
  // That way all files are hadd'able
  // Optional arguments fileName and treeName can override default naming convention
  makeMYTree(TString MCID, TString syst, TString fileName="", TString treeName="");
  ~makeMYTree();

  TFile* file; 
  TTree* tree;
  TString mymcid;

  std::vector<Double_t> bMY_MM_weight;
  std::vector<Double_t> bMY_syst_MM_up;
  std::vector<Double_t> bMY_syst_MM_down;
  std::vector<TString>  bMY_MM_key;

  Int_t                                      bMY_channel;                                               
  Float_t                                    bMY_weight;                                     
  Float_t                                    bMY_lep1Pt;                    // was vector<float> untill jet m                            
  Float_t                                    bMY_lep1Eta;                                                
  Float_t                                    bMY_lep1Phi;                                                
  Float_t                                    bMY_lep1Et;                                              
  Float_t                                    bMY_lep2Pt;                                                 
  Float_t                                    bMY_lep2Eta;                                                
  Float_t                                    bMY_lep2Phi;                                                
  Float_t                                    bMY_lep2Et;                                                  
  Float_t                                    bMY_jet1Pt;                                                 
  Float_t                                    bMY_jet1Eta;                                                
  Float_t                                    bMY_jet1Phi;                                                 
  Float_t                                    bMY_jet2Pt;                                                 
  Float_t                                    bMY_jet2Eta;                                                
  Float_t                                    bMY_jet2Phi;                                              
  Float_t                                    bMY_jetB;                                             
  Float_t                                    bMY_jetLight;                                             
  Float_t                                    bMY_jetTot;                                              
  Float_t                                    bMY_met_Et;                                               
  // Float_t                                    bMY_met_Sign;                                              
  // Float_t                                    bMY_met_Phi;                                              
  // Float_t                                    bMY_met_Et_loose;                                          
  // Float_t                                    bMY_met_Et_tighter;                                       
  // Float_t                                    bMY_met_Et_tenacious;                                     
  // Float_t                                    bMY_met_Phi_loose;                                        
  // Float_t                                    bMY_met_Phi_tighter;                                       
  // Float_t                                    bMY_met_Phi_tenacious;                                     
  Float_t                                    bMY_mll;                                                   
  ULong64_t                                  bMY_EventNumber;                                           
  Int_t                                      bMY_RunNumber;                                             
  TString                                    bMY_RunPeriod;                                             

                                               

  //virtual void InitializeOutput(TFile** file, TString filename,TTree** tree, TString treename );
  void ClearOutputBranches();

  void setSumOfMcWeights(double sumOfMcWeights);


  void WriteTree();

};
#endif // #ifdef makeMYTrees_cxx