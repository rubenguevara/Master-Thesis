import ROOT
from ROOT import *
import sys, os, time, argparse, shutil 
import multiprocessing as mp

gEnv.SetValue("TFile.AsyncPrefetching", 1)

NCORES = mp.cpu_count()

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="mc16a", help="Data to consider (For MC: mc16a/d/e, for data: data15/16/17/18 or all_data)")
parser.add_argument('--bkgs', type=str, default="", help="Backgrounds IDs to run over")
parser.add_argument('--isHepp01', type=int, default=0, help="Run on hepp01?")
parser.add_argument('--maxCPUs', type=int, default=50, help="How many CPUs to run on")
parser.add_argument('--prod_date', type=str, default="june2022", help="Which ntuple production round to use")

parser.add_argument('--doSyst', type=int, default=0, help="Run on systematics?")
parser.add_argument('--doTruth', type=int, default=0, help="Do truth stuff?")
parser.add_argument('--doCutflow', type=int, default=0, help="Do cutflow?")
parser.add_argument('--doFakes', type=int, default=0, help="Do fakes?")
parser.add_argument('--doLoose', type=int, default=0, help="Do loose selections?")

parser.add_argument('--ml_file', type=str, default="myfile", help="Filename to save all events")

args = parser.parse_args()

data = args.data # Should be "data15/16/17/18" or "mc16a/d/e" 
bkgs = args.bkgs 
isHepp01 = args.isHepp01
prod_date = args.prod_date
maxCPUs = args.maxCPUs

doTruth = args.doTruth
doCutflow = args.doCutflow 
doSyst = args.doSyst
doFakes = args.doFakes 
doLoose = args.doLoose

ml_file = args.ml_file

# Set input and output paths 
input_dir = '../../../storage/shared/data/master_students/ZMET/Nominal_june2022/'
# input_dir = '../../../storage/shared/data/master_students/Ruben/'
hist_path = "Histograms"

if 'mc' in data: 
        input_dir = input_dir+'MC/'+data+'/' 
        # input_dir = input_dir+'MC_JAN2023/'+data+'/' 
if 'data' in data: 
        input_dir = input_dir+'Data/'+data+'/'

if doSyst == 1:
        input_dir = input_dir.replace("Nominal", "Systematics_july2022") 
if isHepp01 == 1:
        input_dir = input_dir.replace("scratch3", "scratch")
if doSyst == 0 and isHepp01==1: 
        input_dir = input_dir.replace("Nominal","Nominal_"+prod_date) 
        
# DSIDs corresponding to the --bkgs argument         
IDs = {} 
IDs["DYmumu_incl"] = [364100, 364101, 364102, 364103, 364104, 364105, 364106, 364107, 364108, 364109, 364110, 364111, 364112, 364113]
IDs["DYee_incl"] = [364114, 364115, 364116, 364117, 364118, 364119, 364120, 364121, 364122, 364123, 364124, 364125, 364126, 364127]
IDs["DY_Sh2211_incl"] = [700320, 700321, 700322, 700323, 700324, 700325, 700326, 700327, 700328] 
IDs["DY_Sh2211_high"] = [700452, 700453, 700454, 700455, 700456, 700457, 700458, 700459, 700460] 
IDs["DY_Sh2211_low"] = [700467, 700468, 700469, 700470, 700471, 700472, 700473, 700474, 700475] 
IDs["DYmumu_100"] = [366300, 366301, 366302, 366303, 366304, 366305, 366306, 366307, 366308]
IDs["DYee_100"] = [366309, 366310, 366312, 366313, 366315]
IDs["DYtautau"] = [364128, 364129, 364130, 364131, 364132, 364133, 364134, 364135, 364136, 364137, 364138, 364139, 364140, 364141] 
IDs["DYee_PH"] = [301000, 301001, 301002, 301003, 301004, 301005, 301006, 301007, 301008, 301009, 301010, 301011, 301012, 301013, 301014, 301015, 301016, 301017, 301018, 361106]
IDs["DYmumu_PH"] = [301020, 301021, 301022, 301023, 301024, 301025, 301026, 301027, 301028, 301029, 301030, 301031, 301032, 301033, 301034, 301035, 301036, 301037, 301038, 361107] 
IDs["Top"] = [410470, 410472, 410644, 410645, 410648, 410649, 410656, 410657, 410658, 410659, 410465, 410482, 410558, 411038, 411039, 412003] #, 410633, 410634, 410635, 410636, 410637] 
IDs["Top_MF"] = [410472, 410633, 410634, 410635, 410636, 410637] 
IDs["TTbar_dil"] = [410472]
IDs["TTbar_MET"] = [407345, 407346, 407347]
IDs["Top_AFII"] = [410465, 410472, 410482, 410558, 410648, 410649, 411038, 411039, 412003] 
IDs["TTbar_nonallhad"] = [410470]
IDs["ST"] = [410644, 410645,410646, 410647, 410648, 410649, 410656, 410657, 410658, 410659]
IDs["Diboson"] = [363356, 363358, 363359, 363360, 363489, 364250, 364253, 364254, 364255]
IDs["Wmunu"] = [364156, 364157, 364158, 364159, 364160, 364161, 364162, 364163, 364164, 364165, 364166, 364167, 364168, 364169]
IDs["Wenu"] = [364170, 364171, 364172, 364173, 364174, 364175, 364176, 364177, 364178, 364179, 364180, 364181, 364182, 364183]
IDs["Wtaunu"] = [364184, 364185, 364186, 364187, 364188, 364189, 364190, 364191, 364192, 364193, 364194, 364195, 364196, 364197]
IDs["DY_low"] = [364198, 364199, 364200, 364201, 364202, 364203, 364204, 364205, 364206, 364207, 364208, 364209, 364210, 364211, 364212, 364213, 364214, 364215]
IDs["sig_high"] = [500339, 500340, 500341, 500342, 500343, 500345, 500346, 500347, 500348, 500349, 500350, 500351, 500352, 500353, 500354, 500355, 500356, 500357, 500358, 500359, 500360, 500361, 500362, 500363, 500364, 500365, 500366, 500367, 500368, 500369, 500370, 500344]
IDs["sig_low"] = [506745, 506746, 506747, 506748, 506749, 506750, 506751, 506752, 506753, 506754, 506755, 506756, 506757, 506758, 506759, 506760, 506761, 506762, 506763, 506764, 506765, 506766, 506767, 506768] 
IDs["sig_AFII"] = [500339, 500340, 500341, 500342, 500343, 500344, 500345, 500346, 500347, 500348, 500349, 500350, 500351, 500352, 500353, 500354, 500355, 500356, 500357, 500358, 500359, 500360, 500361, 500362,
                500363, 500364, 500365, 500366, 500367, 500368, 500369, 500370, 514560, 514561, 514562, 514563, 514564, 514565, 514566, 514567, 514568, 514569, 514570, 514571, 514572, 514573, 514574, 514575,
                514576, 514577, 514578, 514579, 514580, 514581, 514582, 514583, 514584, 514585, 514586, 514587, 514588, 514589, 514590, 514591, 514592, 514593, 514594, 514595, 514596, 514597, 514598, 514599,
                514600, 514601, 514602, 514603, 514604, 514605, 514606, 514607, 514608, 514609, 514610, 514611, 514612, 514613, 514614, 514615, 514616, 514617, 514618, 514619, 514620, 514621, 514622, 514623,
                514624, 514625, 514626, 514627, 514628, 514629, 514630, 514631, 514632, 514633, 514634, 514635, 514636, 514637, 514638, 514639, 514640, 514641, 514642, 514643, 514644, 514645, 514646, 514647,
                514648, 514649, 514650, 514651, 514652, 514653, 514654, 514655, 514656, 514657, 514658, 514659, 514660, 514661, 514662, 514663, 514664, 514665, 514666, 514667, 514668, 514669, 514670, 514671,
                514672, 514673, 514674, 514675, 514676, 514677, 514678, 514679, 514680, 514681, 514682, 514683, 514684, 514685, 514686, 514687, 514688, 514689, 514690, 514691, 514692, 514693, 514694, 514695,
                514696, 514697, 514698, 514699, 514700, 514701, 514702, 514703, 514704, 514705, 514706, 514707]

IDs['DY'] =[700320, 700321, 700322, 700323, 700324, 700325, 700326, 700327, 700328, 700452, 700453, 700454, 700455, 700456, 700457, 700458, 700459, 700460]
IDs['TTbar'] = [410472]
IDs['Single_top'] = [410644, 410645, 410648, 410649, 410658, 410659] 

IDs["DY_all"] = IDs["DYmumu_incl"] + IDs["DYee_incl"] + IDs["DYmumu_100"] + IDs["DYee_100"] + IDs["DYtautau"] #+ IDs["DY_low"] #+ IDs["DYee_PH"] + IDs["DYmumu_PH"] 
IDs["DY_Sh2211"] = IDs["DY_Sh2211_incl"]# + IDs["DY_Sh2211_high"]
#IDs["DY"] = IDs["DYmumu_incl"] + IDs["DYee_incl"] + IDs["DYmumu_100"] + IDs["DYee_100"] + IDs["DYee_PH"] + IDs["DYmumu_PH"] 
IDs["DY_PH"] = IDs["DYee_PH"] + IDs["DYmumu_PH"]
IDs["W"] = IDs["Wmunu"] + IDs["Wenu"] + IDs["Wtaunu"] 
IDs["fakeMC"] = IDs["TTbar_nonallhad"]+IDs["W"] 
IDs["sig"] = IDs["sig_high"] + IDs["sig_low"] 

IDs["all_bkg"] = IDs["DY"]+ IDs['Single_top']+IDs["Diboson"] +IDs["W"] + IDs["TTbar"]
IDs["all_MC"] = IDs["all_bkg"] + IDs["sig_AFII"]

IDs["dm_sig"] = [514560, 514561, 514562, 514563, 514564, 514565, 514566, 514567, 514568, 514569, 514570, 514571, 514572, 514573, 514574, 514575,
                514576, 514577, 514578, 514579, 514580, 514581, 514582, 514583, 514584, 514585, 514586, 514587, 514588, 514589, 514590, 514591, 514592, 514593, 514594, 514595, 514596, 514597, 514598, 514599,
                514600, 514601, 514602, 514603, 514604, 514605, 514606, 514607, 514608, 514609, 514610, 514611, 514612, 514613, 514614, 514615, 514616, 514617, 514618, 514619, 514620, 514621, 514622, 514623,
                514624, 514625, 514626, 514627, 514628, 514629, 514630, 514631, 514632, 514633, 514634, 514635, 514636, 514637, 514638, 514639, 514640, 514641, 514642, 514643, 514644, 514645, 514646, 514647,
                514648, 514649, 514650, 514651, 514652, 514653, 514654, 514655, 514656, 514657, 514658, 514659, 514660, 514661, 514662, 514663, 514664, 514665, 514666, 514667, 514668, 514669, 514670, 514671,
                514672, 514673, 514674, 514675, 514676, 514677, 514678, 514679, 514680, 514681, 514682, 514683, 514684, 514685, 514686, 514687, 514688, 514689, 514690, 514691, 514692, 514693, 514694, 514695,
                514696, 514697, 514698, 514699, 514700, 514701, 514702, 514703, 514704, 514705, 514706, 514707]

IDs["data15"] = ["D", "E", "F", "G", "H", "J"]
IDs["data16"] = ["A", "B", "C", "D", "E", "F", "G", "I", "K", "L"]
IDs["data17"] = ["B", "C", "D", "E", "F", "H", "I", "K"]
IDs["data18"] = ["B", "C", "D", "F", "I", "K", "L", "M", "O", "Q"]

IDs['all_data'] = IDs["data15"] + IDs["data16"] + IDs["data17"] + IDs["data18"]

IDs["sig_test"] = [500340]
IDs["sig_test_el"] = [500339]
IDs["sig_test_mu"] = [500340]
IDs["sig_low_test"] = [506745]
IDs["bkg_test"] = [364187]
IDs["top_test"] = [410644]
IDs["db_test"] = [363360]
IDs["data_test"] = ["E"]

# IDs['SUSY'] = [503054, 503059, 503063, 503067, 503072, 503075, 503076, 503078, 503079, 503082, 503084, 503085, 503086, 503087, 503095, 503097, 503100, 503103, 503104, 503105, 503107, 503111, 503112, 503118, 
#                 503120, 503121, 503125, 503129, 503136, 503139, 503142, 503144, 503146, 503150, 503151, 505886, 505887, 505890, 505891, 505895, 505900, 505904, 505905, 505907, 505909, 505911, 505913, 505916, 
#                 505918, 505920, 505924, 505925, 505926, 505928, 505929, 505930, 505932, 505933, 505937, 505940, 505941, 505944, 508056, 508058, 508060, 508062, 508064, 508066, 508068, 508070, 508072, 508074, 
#                 508076, 508078, 508080, 508082, 508084, 508086, 508088, 508090, 508092, 508094, 508096, 508098, 508100, 508102, 508104, 508106, 508108, 508110, 508112, 508114, 508116, 508118, 508120, 508122, 
#                 508124, 508126, 508128, 508130, 508132, 508134, 508136, 508138, 508140, 508142, 508144, 508146, 508148, 508150, 508152, 508154, 508156, 508158, 508160, 508162, 508164, 508166, 508168, 508170, 
#                 508172, 508174, 508176, 508178, 508180, 508182, 508184, 508186, 508188, 508190, 508192, 508194, 508196, 508198, 508200, 508202, 508204, 508206, 508208, 508210, 508212, 508214, 508216, 508218, 
#                 508220, 508222, 508224, 508226, 508228, 508230, 508232, 508234, 508236, 508238, 508240, 508242, 508244, 508246, 508248, 508250, 508252, 508254, 508256, 508258, 508260, 508262, 508264, 508266, 
#                 508268, 508270, 508272, 508274, 508276, 508278, 508280, 508282, 508284, 508286, 508288, 508290, 508292, 508294, 508296, 508298, 508300, 508302, 508304, 508306, 508308, 508310, 508312, 508314, 
#                 508316, 508318, 508320, 508322, 508324, 508326, 508328, 508330, 508332, 508334, 508336, 508338, 508340, 508342, 508344, 508346, 508348, 508350, 508352, 508354, 508356, 508358, 508360, 508362, 
#                 508364, 508366, 508368, 508370, 508372, 508374, 508376, 508378, 508380, 508382, 508384, 508386, 508388, 508390, 508392, 508394, 508396, 508398, 508400, 508402, 508404, 508406, 508408, 508410, 
#                 508412, 508414, 508416, 508418, 508420, 508422, 508424, 508426, 508428, 508430, 508432]

IDs['SUSY'] = [503054, 503055, 503056, 503057, 503058, 503059, 503060, 503061, 503062, 503063, 503064, 503065, 503066, 503067, 503068, 503069, 503070, 503071, 503072, 503073, 503074, 503075, 503076, 503077, 
                503078, 503079, 503080, 503081, 503082, 503083, 503084, 503085, 503086, 503087, 503088, 503089, 503090, 503091, 503092, 503093, 503094, 503095, 503096, 503097, 503098, 503099, 503100, 503101, 
                503102, 503103, 503104, 503105, 503106, 503107, 503108, 503109, 503110, 503111, 503112, 503113, 503114, 503115, 503116, 503117, 503118, 503119, 503120, 503121, 503122, 503123, 503124, 503125, 
                503126, 503127, 503128, 503129, 503130, 503131, 503132, 503133, 503134, 503135, 503136, 503137, 503138, 503139, 503140, 503141, 503142, 503143, 503144, 503145, 503146, 503147, 503148, 503149, 
                503150, 505886, 505887, 505888, 505889, 505890, 505891, 505892, 505893, 505894, 505895, 505896, 505897, 505898, 505899, 505900, 505901, 505902, 505903, 505904, 505905, 505906, 505907, 505908, 
                505909, 505910, 505911, 505912, 505913, 505914, 505915, 505916, 505917, 505918, 505919, 505920, 505921, 505922, 505923, 505924, 505925, 505926, 505927, 505928, 505929, 505930, 505931, 505932, 
                505933, 505934, 505935, 505936, 505937, 505938, 505939, 505940, 505941, 505942, 505943, 505944, 508056, 508058, 508060, 508062, 508064, 508066, 508068, 508070, 508072, 508074, 508076, 508078, 
                508080, 508082, 508084, 508086, 508088, 508090, 508092, 508094, 508096, 508098, 508100, 508102, 508104, 508106, 508108, 508110, 508112, 508114, 508116, 508118, 508120, 508122, 508124, 508126, 
                508128, 508130, 508132, 508134, 508136, 508138, 508140, 508142, 508144, 508146, 508148, 508150, 508152, 508154, 508156, 508158, 508160, 508162, 508164, 508166, 508168, 508170, 508172, 508174, 
                508176, 508178, 508180, 508182, 508184, 508186, 508188, 508190, 508192, 508194, 508196, 508198, 508200, 508202, 508204, 508206, 508208, 508210, 508212, 508214, 508216, 508218, 508220, 508222, 
                508224, 508226, 508228, 508230, 508232, 508234, 508236, 508238, 508240, 508242, 508244, 508246, 508248, 508250, 508252, 508254, 508256, 508258, 508260, 508262, 508264, 508266, 508268, 508270, 
                508272, 508274, 508276, 508278, 508280, 508282, 508284, 508286, 508288, 508290, 508292, 508294, 508296, 508298, 508300, 508302, 508304, 508306, 508308, 508310, 508312, 508314, 508316, 508318, 
                508320, 508322, 508324, 508326, 508328, 508330, 508332, 508334, 508336, 508338, 508340, 508342, 508344, 508346, 508348, 508350, 508352, 508354, 508356, 508358, 508360, 508362, 508364, 508366, 
                508368, 508370, 508372, 508374, 508376, 508378, 508380, 508382, 508384, 508386, 508388, 508390, 508392, 508394, 508396, 508398, 508400, 508402, 508404, 508406, 508408, 508410, 508412, 508414,
                508416, 508418, 508420, 508422, 508424, 508426, 508428, 508430, 508432]


# IDs['NEW'] = []
# for file in os.listdir(input_dir):                                        # To find new DSIDS print and copy array 
#         dsid = int(file.split('.')[4])
#         IDs['NEW'].append(dsid)
# print(IDs['NEW'])
# exit()

# Make output directories         
if not os.path.exists(hist_path):
        os.makedirs(hist_path)
if not os.path.exists(hist_path+'/'+data+'/'):
        os.makedirs(hist_path+'/'+data)

save_dir = "../../../storage/racarcam/ML_files"
try:
        os.makedirs(save_dir)

except FileExistsError: 
        pass

for ID in IDs[bkgs]: 
        this_path = ""
        if 'data' in data: 
                this_path = hist_path+'/'+data+'/period'+ID
        else: 
                this_path = hist_path+'/'+data+'/'+str(ID)
        if os.path.exists(this_path):
                shutil.rmtree(this_path)
        os.makedirs(this_path)
        
def wait_completion(pids, sleep=30):
    #Wait until completion of one of the launched jobs
    time.sleep(sleep)
    while True:
        for pid in pids:
            if not pid.is_alive():
                try:
                    print( ">>> Completed: "+str(pid.name)+" ("+str(pid.pid)+")" )
                except AttributeError:
                    print( ">>> Completed: "+str(pid.pid) )
                pids.remove(pid)
                return
        #print( "Waiting for completion of jobs..." )
        time.sleep(sleep) # wait before retrying
    return
        
def wait_all(pids):
    for p in pids:
        p.join()    
    print( "All jobs finished!" )
    return


def parallel_exec(target, args_list, nproc=int(NCORES/2), name=None, sleep = 2):
        if nproc >= NCORES:
                nproc = NCORES - 1

        tot_events = 0
        pids = []
        print( "Number of CPUs being used: " +str(nproc)) 
        for a in args_list:
                if len(pids) >= nproc:
                        wait_completion(pids, sleep=sleep)
                p = mp.Process(target=target, args=(a,))        
                n = str(a) 
                print( ">>> Started: "+str(n) )
                p.name = n
                pids.append(p)
                p.start()
                #if len(pids)==1:
                #        time.sleep(10) 
        wait_all(pids)    
        return

def count_events(DSID): 
        myChain1 = TChain('nominal')
        Input_Dir = '' 
        if 'data' in str(DSID): 
                Input_Dir = input_dir+data
        else: 
                Input_Dir = input_dir
        for folder in os.listdir(Input_Dir): 
                if not '.root' in folder: continue 
                if 'mc' in data and not str(DSID) in folder: continue
                if 'data' in data and not "period"+DSID in folder: continue
                for filename in os.listdir(Input_Dir+folder): 
                        myChain1.Add(Input_Dir+folder+'/'+filename)  

        nevents = myChain1.GetEntries()
        return nevents 


def runDSID(DSID): 
        myChain = TChain('nominal')
        Input_Dir = '' 
        if 'data' in str(DSID): 
                Input_Dir = input_dir+DSID+'/'
        else: 
                Input_Dir = input_dir

        isAFII = 0;
        for folder in os.listdir(Input_Dir): 
                if not '.root' in folder: continue 
                if not str(DSID) in folder: continue
                #if not '366313' in folder: continue
                #if not '410644' in folder: continue
                print(folder ) 
                if "BKGAFII" in folder:
                        isAFII = 1
                for filename in os.listdir(Input_Dir+folder): 
                        myChain.Add(Input_Dir+folder+'/'+filename)  

        entries = myChain.GetEntries()
        print(str(entries)+" events to process for "+str(DSID))  

        isRecast = 0;
        if bkgs == 'recast':
                isRecast = 1;

        option = ''
        if 'data' in str(DSID): 
                option = DSID+"_"+str(doTruth)+"_"+str(doCutflow)+"_"+str(doSyst)+"_"+str(doFakes)+"_"+str(doLoose)+"_"+str(isRecast)+"_"+str(isAFII)
        else: 
                option = data+"_"+str(doTruth)+"_"+str(doCutflow)+"_"+str(doSyst)+"_"+str(doFakes)+"_"+str(doLoose)+"_"+str(isRecast)+"_"+str(isAFII)

        myChain.Process("EventSelector.C+", option)
        print("Done with " +str(DSID) )
        return 


def runFile(filename): 

        myChain = TChain("nominal")
        myChain.Add(input_dir+filename)  

        entries = myChain.GetEntries()
        print( str(entries)+" events to process for "+filename) 

        isRecast = 0;
        if bkgs == 'recast':
                isRecast = 1;

        isAFII = 0;
        if "BKGAFII" in filename:
                isAFII = 1

        dsid = filename.split("/")[-2].split(".")[4] 
        fil_nr = filename.split("/")[-1].split(".")[-3]
        
        option = data+"_"+str(doTruth)+"_"+str(doCutflow)+"_"+str(doSyst)+"_"+str(doFakes)+"_"+str(doLoose)+"_"+str(isRecast)+"_"+str(isAFII)+"_"+ml_file+"_"+dsid+fil_nr
        
        
        myChain.Process("EventSelector.C+", option)

        return 


if __name__ == '__main__':

        args_list = []

        for folder in os.listdir(input_dir): 
                if not '.root' in folder: continue 
                dsid = folder.split(".")[4]
                if 'mc' in data and not int(dsid) in IDs[bkgs]: continue 
                if 'data' in data and not dsid[-1] in IDs[bkgs]: continue  
                for filename in os.listdir(input_dir+folder): 
                        #if not "00005" in filename: continue 
                        #print folder+'/'+filename
                        args_list.append(folder+'/'+filename) 

        ncpu = mp.cpu_count()
        ncores = 0
        if isHepp01==1: 
                if len(args_list) > maxCPUs: 
                        ncores = maxCPUs 
                else: 
                        ncores = len(args_list) 
        else: 
                if len(args_list)>ncpu: 
                        ncores = ncpu-1
                else: 
                        ncores = len(args_list) 

        print("Number of files to process: " +str(len(args_list) ))

        t0 = time.time()
        gROOT.ProcessLine(".L makeMYTree.cxx+") # ML FILE
        gROOT.ProcessLine(".L EventSelector.C+");
        parallel_exec(runFile, args_list, ncores)
        # Merge histograms 
        out_path = hist_path+"/"+data+"/"
        working_dir = os.getcwd() 
        os.chdir(out_path) 
        for folder in os.listdir("."):
                if '.root' in folder: continue 
                if 'mc' in data and not int(folder) in IDs[bkgs]: continue 
                if 'data' in data and not folder[-1] in IDs[bkgs]: continue
                if len(os.listdir(folder))==0:
                        print("Found empty folder!! DSID %s" %folder) 
                        continue 
                file_string = ""; file_string_r = ""; file_string_l = ""; file_string_rl = "";
                file_string_AFII = ""; file_string_r_AFII = ""; file_string_l_AFII = ""; file_string_rl_AFII = "";
                outfile = ""; outfile_r = ""; outfile_l = ""; outfile_rl = "";
                outfile_AFII = ""; outfile_r_AFII = ""; outfile_l_AFII = ""; outfile_rl_AFII = "";
                n_files = 0; n_files_AFII = 0; 
                for f in os.listdir(folder):
                        if "syst" in f: continue 
                        file_nr = (f.split('.')[2]).split('_')[-1]
                        if (".real." in f) or ("fakes" in f):
                                if "AFII" in f:
                                        outfile_r_AFII = f.replace("_"+file_nr, "") 
                                        file_string_r_AFII+=(" "+folder+"/"+f)
                                else: 
                                        outfile_r = f.replace("_"+file_nr, "") 
                                        file_string_r+=(" "+folder+"/"+f)
                        elif ".loose." in f:
                                if "AFII" in f:
                                        n_files_AFII+=1 
                                        outfile_l_AFII = f.replace("_"+file_nr, "") 
                                        file_string_l_AFII+=(" "+folder+"/"+f)
                                else: 
                                        n_files+=1 
                                        outfile_l = f.replace("_"+file_nr, "") 
                                        file_string_l+=(" "+folder+"/"+f)
                        elif (".real_loose." in f) or (".fakes_loose." in f):
                                if "AFII" in f:
                                        outfile_rl_AFII = f.replace("_"+file_nr, "") 
                                        file_string_rl_AFII+=(" "+folder+"/"+f)
                                else: 
                                        outfile_rl = f.replace("_"+file_nr, "") 
                                        file_string_rl+=(" "+folder+"/"+f)
                        else: 
                                if "AFII" in f:
                                        n_files_AFII+=1 
                                        outfile_AFII = f.replace("_"+file_nr, "") 
                                        file_string_AFII+=(" "+folder+"/"+f)
                                else: 
                                        outfile = f.replace("_"+file_nr, "") 
                                        file_string+=(" "+folder+"/"+f)
                                        n_files+=1 
                                        
                if outfile_r!="":
                        if os.path.exists(outfile_r): 
                                os.remove(outfile_r)
                        if n_files == 1:
                                shutil.move(file_string_r.replace(" ", ""), outfile_r)
                        else: 
                                os.system("hadd "+outfile_r+file_string_r)
                if outfile_l!="": 
                        if os.path.exists(outfile_l): 
                                os.remove(outfile_l)
                        if n_files == 1:
                                shutil.move(file_string_l.replace(" ", ""), outfile_l)
                        else: 
                                os.system("hadd "+outfile_l+file_string_l)
                if outfile_rl!="": 
                        if os.path.exists(outfile_rl): 
                                os.remove(outfile_rl)
                        if n_files == 1:
                                shutil.move(file_string_rl.replace(" ", ""), outfile_rl)
                        else: 
                                os.system("hadd "+outfile_rl+file_string_rl)
                if outfile!="": 
                        if os.path.exists(outfile): 
                                os.remove(outfile)
                        if n_files == 1:
                                shutil.move(file_string.replace(" ", ""), outfile)
                        else: 
                                os.system("hadd "+outfile+file_string)

                # For AFII samples 
                if outfile_r_AFII!="": 
                        if os.path.exists(outfile_r_AFII): 
                                os.remove(outfile_r_AFII)
                        if n_files_AFII == 1:
                                shutil.move(file_string_r_AFII.replace(" ", ""), outfile_r_AFII)
                        else: 
                                os.system("hadd "+outfile_r_AFII+file_string_r_AFII)
                if outfile_l_AFII!="": 
                        if os.path.exists(outfile_l_AFII): 
                                os.remove(outfile_l_AFII)
                        if n_files_AFII == 1:
                                shutil.move(file_string_l_AFII.replace(" ", ""), outfile_l_AFII)
                        else: 
                                os.system("hadd "+outfile_l_AFII+file_string_l_AFII)
                if outfile_rl_AFII!="": 
                        if os.path.exists(outfile_rl_AFII): 
                                os.remove(outfile_rl_AFII)
                        if n_files_AFII == 1:
                                shutil.move(file_string_rl_AFII.replace(" ", ""), outfile_rl_AFII)
                        else: 
                                os.system("hadd "+outfile_rl_AFII+file_string_rl_AFII)
                if outfile_AFII!="":
                        print (outfile_AFII )
                        if os.path.exists(outfile_AFII): 
                                os.remove(outfile_AFII)
                        if n_files_AFII == 1:
                                shutil.move(file_string_AFII.replace(" ", ""), outfile_AFII)
                        else: 
                                os.system("hadd "+outfile_AFII+file_string_AFII)

                if not doSyst: 
                        shutil.rmtree(folder)

        if doSyst: 
                out_path = hist_path+"/"+data+"/"
                working_dir = os.getcwd()
                os.chdir(out_path) 
                for folder in os.listdir("."):
                        if '.root' in folder: continue 
                        if 'mc' in data and not int(folder) in IDs[bkgs]: continue 
                        if 'data' in data and not folder[-1] in IDs[bkgs]: continue  
                        if len(os.listdir(folder))==0:
                                print("Found empty folder!! DSID %s" %folder) 
                                continue 
                        file_string_s = ""; file_string_s_AFII = "";   
                        outfile_s = "";  outfile_s_AFII = ""; 
                        n_files = 0; n_files_AFII = 0; 
                        for f in os.listdir(folder):
                                if doFakes and not "syst_real" in f: continue 
                                if not doFakes and not ".syst." in f: continue 
                                #print f
                                file_nr = (f.split('.')[2]).split('_')[-1]
                                if "AFII" in f:
                                        outfile_s_AFII = f.replace("_"+file_nr, "") 
                                        file_string_s_AFII+=(" "+folder+"/"+f)
                                        n_files_AFII+=1
                                else: 
                                        outfile_s = f.replace("_"+file_nr, "") 
                                        file_string_s+=(" "+folder+"/"+f)
                                        n_files+=1 
                                        
                        if outfile_s!="": 
                                if os.path.exists(outfile_s): 
                                        os.remove(outfile_s)
                                if n_files == 1:
                                        shutil.move(file_string_s.replace(" ", ""), outfile_s)
                                else: 
                                        os.system("hadd "+outfile_s+file_string_s)

                        # For AFII samples 
                        if outfile_s_AFII!="": 
                                if os.path.exists(outfile_s_AFII): 
                                        os.remove(outfile_s_AFII)
                                if n_files_AFII == 1:
                                        shutil.move(file_string_s_AFII.replace(" ", ""), outfile_s_AFII)
                                else: 
                                        os.system("hadd "+outfile_s_AFII+file_string_s_AFII)

                        shutil.rmtree(folder)
                
        os.chdir("../")

        if 'data' in data: 
                if not os.path.exists('data/'):
                        os.makedirs('data')
                file_string = ""; file_string_r = ""; file_string_l = ""; file_string_rl = "";   
                outfile = ""; outfile_r = ""; outfile_l = ""; outfile_rl = ""; 
                for f in os.listdir(data): 
                        period = f.split('.')[2]
                        if "fakes" in f:
                                outfile_r = "data/"+f.replace("."+period, "") 
                                file_string_r+=(" "+data+"/"+f)
                        elif ".loose." in f:
                                outfile_l = "data/"+f.replace("."+period, "") 
                                file_string_l+=(" "+data+"/"+f)
                        elif ".fakes_loose." in f:
                                outfile_rl = "data/"+f.replace("."+period, "") 
                                file_string_rl+=(" "+data+"/"+f)
                        else: 
                                outfile = "data/"+f.replace("."+period, "") 
                                file_string+=(" "+data+"/"+f)
                        
                if outfile_r!="": 
                        if os.path.exists(outfile_r): 
                                os.remove(outfile_r)
                        os.system("hadd "+outfile_r+file_string_r)
                if outfile_l!="": 
                        if os.path.exists(outfile_l): 
                                os.remove(outfile_l)
                        os.system("hadd "+outfile_l+file_string_l)
                if outfile_rl!="": 
                        if os.path.exists(outfile_rl): 
                                os.remove(outfile_rl)
                        os.system("hadd "+outfile_rl+file_string_rl)
                if outfile!="": 
                        if os.path.exists(outfile): 
                                os.remove(outfile)
                        os.system("hadd "+outfile+file_string)

                shutil.rmtree(data)

        os.chdir(working_dir) 
        # Merge nTuples
        save_path = save_dir+"/"

        working_dir = os.getcwd() 
        os.chdir(save_path) 

        file_a = ""; file_d = ""; file_e = ""
        outfile_a = ""; outfile_d = ""; outfile_e = ""
        file_15 = ""; file_16 = ""; file_17 = ""; file_18 = ""
        outfile_15 = ""; outfile_16 = ""; outfile_17 = ""; outfile_18 = ""
        for file in os.listdir("."):
                type = file.split('-')[0]
                if type == 'DELETE':
                        os.remove(file)
                        continue
                mcRun = file.split("-")[1]
                extra = "-" + file.split("-")[2] + "-" + file.split("-")[3]
                if mcRun == 'mc16a':
                        outfile_a = "../"+file.replace(extra, ".root")
                        file_a += (" "+file)
                if mcRun == 'mc16d':
                        outfile_d = "../"+file.replace(extra, ".root")
                        file_d += (" "+file)
                if mcRun == 'mc16e':
                        outfile_e = "../"+file.replace(extra, ".root")
                        file_e += (" "+file)
                if mcRun == 'data15':
                        outfile_15 = "../"+file.replace(extra, ".root")
                        file_15 += (" "+file)
                if mcRun == 'data16':
                        outfile_16 = "../"+file.replace(extra, ".root")
                        file_16 += (" "+file)
                if mcRun == 'data17':
                        outfile_17 = "../"+file.replace(extra, ".root")
                        file_17 += (" "+file)
                if mcRun == 'data18':
                        outfile_18 = "../"+file.replace(extra, ".root")
                        file_18 += (" "+file)

        if outfile_a!="":
                if os.path.exists(outfile_a): 
                        print("Final file", file,"exists! Do you want to delete?")
                else: 
                        os.system("hadd "+outfile_a+file_a)

        if outfile_d!="":
                if os.path.exists(outfile_d): 
                        print("Final file", file,"exists! Do you want to delete?")
                else: 
                        os.system("hadd "+outfile_d+file_d)

        if outfile_e!="":
                if os.path.exists(outfile_e): 
                        print("Final file", file,"exists! Do you want to delete?")
                else: 
                        os.system("hadd "+outfile_e+file_e)
        
        if outfile_15!="":
                if os.path.exists(outfile_15): 
                        print("Final file", file,"exists! Do you want to delete?")
                else: 
                        os.system("hadd "+outfile_15+file_15)

        if outfile_16!="":
                if os.path.exists(outfile_16): 
                        print("Final file", file,"exists! Do you want to delete?")
                else: 
                        os.system("hadd "+outfile_16+file_16)

        if outfile_17!="":
                if os.path.exists(outfile_17): 
                        print("Final file", file,"exists! Do you want to delete?")
                else: 
                        os.system("hadd "+outfile_17+file_17)

        if outfile_18!="":
                if os.path.exists(outfile_18): 
                        print("Final file", file,"exists! Do you want to delete?")
                else: 
                        os.system("hadd "+outfile_18+file_18)

        shutil.rmtree(save_path)
        
        os.chdir(working_dir) 
        print( "================================================")
        tot_events = 0
        for ID in IDs[bkgs]:
                tot_events += count_events(ID)  
        print ("Total number of events processed: "+('{:,}'.format(tot_events)))
        t = "{:.2f}".format(int( time.time()-t0 )/60.)
        print( "Time spent: "+str(t)+" min")  
        print( "================================================")
