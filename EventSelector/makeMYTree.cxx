#define makeMYTree_cxx
#include "makeMYTree.h"
#include "TParameter.h"
#include "TString.h"

using namespace std;



makeMYTree::makeMYTree(TString MCID, TString syst, TString fileName, TString treeName)
{
  if(fileName==""){
    fileName = syst+"_"+MCID+".root";
  }
  if(treeName==""){
    treeName = "id_" + MCID;
  }
  // Setup a TTree in a output file
  file = TFile::Open(fileName, "RECREATE");
  file->cd();
  //tree = new TTree("id_"+MCID, "id_"+MCID);
  tree = new TTree(treeName, treeName);
  tree->SetAutoSave(10000000);
  TTree::SetBranchStyle(1);
  tree->SetDirectory(file);

  mymcid = MCID;

  tree->Branch("EventID",                                            &bMY_EventID);  
  tree->Branch("lep1Pt",                                             &bMY_lep1Pt);                                            
  tree->Branch("lep1Eta",                                            &bMY_lep1Eta);                                           
  tree->Branch("lep1Phi",                                            &bMY_lep1Phi);                                         
  tree->Branch("lep2Pt",                                             &bMY_lep2Pt);                                            
  tree->Branch("lep2Eta",                                            &bMY_lep2Eta);                                           
  tree->Branch("lep2Phi",                                            &bMY_lep2Phi);                                            
  tree->Branch("jetB",                                               &bMY_jetB);                                                    
  tree->Branch("jetLight",                                           &bMY_jetLight);                                                    
  tree->Branch("jetTot",                                             &bMY_jetTot);                                                    
  tree->Branch("jet1Pt",                                             &bMY_jet1Pt);                                            
  tree->Branch("jet1Eta",                                            &bMY_jet1Eta);                                           
  tree->Branch("jet1Phi",                                            &bMY_jet1Phi);                                           
  tree->Branch("jet2Pt",                                             &bMY_jet2Pt);                                            
  tree->Branch("jet2Eta",                                            &bMY_jet2Eta);                                           
  tree->Branch("jet2Phi",                                            &bMY_jet2Phi);                                   
  tree->Branch("mll",                                                &bMY_mll);                                                
  tree->Branch("met",                                                &bMY_met);                                           
  tree->Branch("met_sig",                                            &bMY_met_sig);                                           
  tree->Branch("ht",                                                 &bMY_ht);                                            
  tree->Branch("mt",                                                 &bMY_mt);                                           
  tree->Branch("mt2",                                                &bMY_mt2);                                           
  tree->Branch("et",                                                 &bMY_et);                                                  
  tree->Branch("dPhiLLMet",                                          &bMY_dPhiLLMet);                                           
  tree->Branch("dPhiCloseMet",                                       &bMY_dPhiCloseMet);                                           
  tree->Branch("dPhiLeadMet",                                        &bMY_dPhiLeadMet);                                
  tree->Branch("dPhiLeps",                                           &bMY_dPhiLeps);          
  tree->Branch("Weight",                                             &bMY_Weight);                                         
  tree->Branch("Dileptons",                                          &bMY_Dileptons);                                   
  tree->Branch("CrossSection",                                       &bMY_CrossSection);                                     
  tree->Branch("RunNumber",                                          &bMY_RunNumber);                                        
  tree->Branch("RunPeriod",                                          &bMY_RunPeriod);                 

  ClearOutputBranches();
}


void makeMYTree::ClearOutputBranches(void)
{

  // bMY_lep1Pt.clear();                                                  
  // bMY_lep1Eta.clear();                                                 
  // bMY_lep1Phi.clear();                                                 
  // bMY_lep1Et.clear();
  // bMY_lep2Pt.clear();                                                  
  // bMY_lep2Eta.clear();                                                 
  // bMY_lep2Phi.clear();                                                 
  // bMY_lep2Et.clear();                                                   
  // bMY_jetPt.clear();                                                  
  // bMY_jetEta.clear();                                                 
  // bMY_jetPhi.clear();                                                 
  // bMY_jetM.clear();                                                 


  
  return;
}





//-----------------------------------------------------------------------------------------------------------
makeMYTree::~makeMYTree()
{
    // Write out the output tree and close the output file
  file->Write();
  file->Close();
  delete file;
}


void makeMYTree::WriteTree()
{
  //file->cd();
    tree->Fill();
    //tree->Write();
    //file->Write();
    //file->Close();
    ClearOutputBranches();
}




void makeMYTree::setSumOfMcWeights(double sumOfMcWeights)
{
    // Define histogram
    TH1D *sumwhist = new TH1D("sumOfMcWeights_"+mymcid,"sumOfMcWeights_"+mymcid,1,0.,1.);

    // Fill histogram
    sumwhist -> Fill( 0. , sumOfMcWeights ) ;

    // Write intLumi to file
    file->cd();
    sumwhist->SetDirectory(file);
    sumwhist->Write();
    sumwhist->SetDirectory(0);

    delete sumwhist;
}



