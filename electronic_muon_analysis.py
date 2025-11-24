"""
FASER Data Quality File Maker
_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

A script which produces the Data Quality files for the 
FASER electronic muon neutrino analysis

Slimmed down version of this code
https://gitlab.cern.ch/tboeckh/FASERRDFAnalysis/-/tree/electronic-neutrino-analysis-2023?ref_type=heads

Adapted to work with the new p0011/p0012 2024 data
"""

import os
import glob
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict

import ROOT
import uproot
import numpy as np
from tqdm import tqdm



def get_run_number_lumi_dict(path_to_grls: str) -> Dict[int, float]:
    """
    Parse the .csv files in the Good Run List (GRL) directory to map the run number to the recorded luminosity
    args:
        path_to_grls: str - path to directory containing the .json GRL files
    """

    grl_csvs = glob.glob(f"{path_to_grls}/*.csv")

    if len(grl_csvs) == 0:
        print(f"Error: No GRLS .csv found on path {path_to_grls}!")
        raise OSError("No files found")    

    run_lumi_dict = {}

    for fpath in grl_csvs:
        with open(fpath, 'r') as f:
            for i, line in enumerate(f):
                if i == 0: continue
                if line.startswith('#'): continue

                spline = line.split(',')
                run_number = int(spline[0])
                lumi_rec = float(spline[3])

                run_lumi_dict[run_number] = lumi_rec
    
    return run_lumi_dict


def make_excluded_times_cut(path_to_grls: str) -> str:
    """
    Function to parse the .json files in the Good Run List (GRL) directory to find excluded time periods in otherwise good runs
    Function parses these time periods to construct a cut string which can be applied as a filter in a ROOT.RDataFrame to 
    remove the bad time periods

    args:
        path_to_grls: str - path to directory containing the .json GRL files
    
    returns:
        cut_str: str - a string which can be used with ROOT.RDataFrame::Filter to filter out excluded time periods from runs

    raises:
        OSError if no .json file are found in `path_to_grls` directory
    """

    grl_jsons = glob.glob(f"{path_to_grls}/*.json")

    if len(grl_jsons) == 0:
        print(f"Error: No GRLS .json found on path {path_to_grls}!")
        raise OSError("No files found")

    excluded_times = {}

    n_excluded_times = 0
    for grl_file in grl_jsons:    
        with open(grl_file, 'r') as f:
            grl_dict = json.load(f)
            
            for run_number, run_info in grl_dict.items():

                if "excluded_list" not in run_info.keys(): continue

                excluded_times[run_number] = run_info['excluded_list']
                n_excluded_times += len(run_info['excluded_list'])
                # print(f"Info: Found {len(run_info['excluded_list'])} excluded periods for run {run_number}")

    if n_excluded_times == 0: return ""

    print(f"Info: Applying cuts to remove {n_excluded_times} excluded periods")

    cut_str = ""
    for run_number, exclusion_list in excluded_times.items():
        for i, exclusion_info in enumerate(exclusion_list):
            start_time = exclusion_info['start_utime']
            stop_time = exclusion_info['stop_utime']

            cut_str += f"((eventTime >= {start_time}) && (eventTime <= {stop_time}) && (run == {run_number}))"
            if n_excluded_times > 1:
                cut_str += " || "
            
    return cut_str.rstrip(" || ")


def make_good_times_cut(path_to_grls: str) -> str:
    """
    Function to parse the .json files in the Good Run List (GRL) directory to find the stable time periods in good runs
    Function parses these time periods to construct a cut string which can be applied as a filter in a ROOT.RDataFrame to 
    select for the stable periods.

    args:
        path_to_grls: str - path to directory containing the .json GRL files
    
    returns:
        cut_str: str - a string which can be used with ROOT.RDataFrame::Filter to filter out select for time periods from runs


    raises:
        OSError if no .json file are found in `path_to_grls` directory
    """

    grl_jsons = glob.glob(f"{path_to_grls}/*.json")

    if len(grl_jsons) == 0:
        print(f"Error: No GRLS .json found on path {path_to_grls}!")
        raise OSError("No files found")

    good_times = {}

    n_good_times = 0
    for grl_file in grl_jsons:   
        with open(grl_file, 'r') as f:
            grl_dict = json.load(f)
            
            for run_number, run_info in grl_dict.items():

                good_times[run_number] = run_info['stable_list']
                n_good_times += len(run_info['stable_list'])

    cut_str = ""
    for run_number, stable_list in good_times.items():
        
        for i, stable_info in enumerate(stable_list):
            start_time = stable_info['start_utime']
            stop_time = stable_info['stop_utime']

            cut_str += f"((eventTime >= {start_time}) && (eventTime <= {stop_time}) && (run == {run_number}))"

            if n_good_times > 1:
                cut_str += " || "
            
    return cut_str.rstrip(" || ")



def validate_file_list(file_list) -> List[str]:
    """
    File check while loops through input files and checks that they can be opened and that they contain the `nt` tree

    args: 
        file_list: List[str] - list of files to check
    
    returns:
        good_files: List[str] - list of files which are openable and contain the `nt` tree
    """

    bad_files = []
    for fpath in tqdm(file_list):
        try:
            data = uproot.open(fpath)
        except Exception as e:
            print(f"Error: Unable to open {fpath}")
            bad_files.append(fpath)
            continue
        
        key_found = False
        for key in data.keys():
            if 'nt' in key: 
                key_found = True
                break
        
        if not key_found:
            bad_files.append(fpath)
            print(f"Error: Unable to open {fpath}. Does not contain 'nt' tree. Available keys are {data.keys()}")
    
    good_files = [fpath for fpath in file_list if fpath not in bad_files]

    return good_files



def check_df_and_apply_alias(df: ROOT.RDataFrame, column_name: str, column_alias: str) -> ROOT.RDataFrame:
    """
    Function that checks whether `column_name` exists in an RDataFrame and if it does not, create `column` name 
    by aliasing it to `column_alias`

    args:
        df: ROOT.RDataframe - dataframe to apply aliases to
        column_name: str - the column name to create via alias if not present in df
        column_alias: str - the column that will be aliased to `column_name`

    returns:
        df: ROOT.RDataFrame - dataframe with the aliases applied (if required)
    """

    if column_name not in df.GetColumnNames():
        df = df.Alias(column_name, column_alias)
        print(f"Info: Aliasing {column_name} -> {column_alias}")

    return df


def alias_p0012_data(df: ROOT.RDataFrame) -> ROOT.RDataFrame:
    """
    The names of some variables changed with the introduction of prompt reco tag p0012 (in August 2024) 
    This function creates an alias which maps the new names back onto the old ones

    args: 
        df: ROOT.RDataframe - dataframe to apply aliases to
    
    returns:
        df: ROOT.RDataFrame - dataframe with the aliases applied (if required)
    """
    

    veto_prefix_map = {
    "VetoSt10_": "Veto0_",
    "VetoSt11_": "Veto1_",
    "VetoSt20_": "Veto2_",
    "VetoSt21_": "Veto3_",
    }

    veto_variables = [
    # "time",
    # "peak",
    # "width",
    "charge",
    "raw_peak",
    "raw_charge",
    "baseline",
    "baseline_rms",
    "status"]

    calo_prefix_map = {
        "Calo0_": "CaloLo0_",
        "Calo1_": "CaloLo1_",
        "Calo2_": "CaloLo2_",
        "Calo3_": "CaloLo3_",
    }

    calo_variables = [
    "nMIP",
    "E_dep",
    "E_EM",
    # "time",
    "peak",
    "width",
    "charge",
    "raw_peak",
    "raw_charge",
    "baseline",
    "baseline_rms",
    "status"]

    for old_prefix, new_prefix in veto_prefix_map.items():
        for varname in veto_variables:
            df = check_df_and_apply_alias(df, old_prefix+varname, new_prefix+varname)  

    for old_prefix, new_prefix in calo_prefix_map.items():
        for varname in calo_variables:
            df = check_df_and_apply_alias(df, old_prefix+varname, new_prefix+varname)  

    return df


def alias_r0022_data(df: ROOT.RDataFrame, has_veto11) -> ROOT.RDataFrame:
    """
    The names of some variables changed with the introduction of prompt reco tag r0022 (in August 2025) 
    This function creates an alias which maps the new names back onto the old ones

    args: 
        df: ROOT.RDataframe - dataframe to apply aliases to
    
    returns:
        df: ROOT.RDataFrame - dataframe with the aliases applied (if required)
    """
    

    veto_prefix_map = {
    "VetoSt10_": "Veto10_",
    "VetoSt11_": "Veto11_",
    "VetoSt20_": "Veto20_",
    "VetoSt21_": "Veto21_",
    }
    if not has_veto11:
        veto_prefix_map = {
        "VetoSt10_": "Veto10_",
        "VetoSt11_": "Veto10_", # fudge to get code to run on 2022/23 data
        "VetoSt20_": "Veto20_",
        "VetoSt21_": "Veto21_",
        }


    veto_variables = [
    # "time",
    # "peak",
    # "width",
    "charge",
    "raw_peak",
    "raw_charge",
    "baseline",
    "baseline_rms",
    "status"]

    calo_prefix_map = {
        "Calo0_": "CaloLo0_",
        "Calo1_": "CaloLo1_",
        "Calo2_": "CaloLo2_",
        "Calo3_": "CaloLo3_",
    }

    calo_variables = [
    "nMIP",
    "E_dep",
    "E_EM",
    # "time",
    "peak",
    "width",
    "charge",
    "raw_peak",
    "raw_charge",
    "baseline",
    "baseline_rms",
    "status"]

    for old_prefix, new_prefix in veto_prefix_map.items():
        for varname in veto_variables:
            df = check_df_and_apply_alias(df, old_prefix+varname, new_prefix+varname)  

    for old_prefix, new_prefix in calo_prefix_map.items():
        for varname in calo_variables:
            df = check_df_and_apply_alias(df, old_prefix+varname, new_prefix+varname)  

    return df



def build_dataframe(file_list: List[str], tree: str='nt', is_mc=False) -> ROOT.RDataFrame:
    """
    Function which constructs a ROOT RDataFrame from file(s) in `file_list`
    Applys the neccessary column definitions, aliases and data quality cuts

    args:
        file_list: List[str] - list of files to load
        tree: str - the name of the tree in the NTuples to read (default='nt')

    returns:
        df: ROOT.RDataFrame - the filtered dataframe with columns defined
    """

    df = ROOT.RDataFrame(tree, file_list)
    # ROOT.RDF.Experimental.AddProgressBar(df)

    #* Apply aliases to map p0012 variable names to their old ones
    # df = alias_p0012_data(df)
    has_veto11 = True
    if args.run < 1.2e4: 
        has_veto11 = False 


    df = alias_r0022_data(df, has_veto11)

    #* Allow shorter use of vecops functions in strings
    #* e.g. DeltaPhi rather than ROOT::VecOps::DeltaPhi 
    ROOT.gInterpreter.Declare("using namespace ROOT::VecOps;")

    #* C++ defines (must not rely on anything defined below)
    ROOT.gInterpreter.Declare('#include "RDFDefines.h"')

    #* Data quality cuts
    if not is_mc:
        good_times_cut_str = make_good_times_cut(args.grl_path)           # Select for the periods of stable running
        df = df.Define("GoodTimes", good_times_cut_str)
        df = df.Filter("GoodTimes", "Good times")

        # print("good_times_cut_str", good_times_cut_str)
        
        exlcuded_times_cut_str = make_excluded_times_cut(args.grl_path)   # Some runs have certain time periods excluded. These periods are recorded in the GRL json files.
        if exlcuded_times_cut_str != "":
            df = df.Define("ExcludedTimes", exlcuded_times_cut_str)
            df = df.Filter("!ExcludedTimes", "Excluded times")
        
    # print("exlcuded_times_cut_str", exlcuded_times_cut_str)

    df = df.Filter("(Timing0_status & 4) == 0 && (Timing1_status & 4) == 0 && (Timing2_status & 4) == 0 && (Timing3_status & 4) == 0 ", "No timing saturation")
    df = df.Filter("distanceToCollidingBCID == 0", "Colliding") #! Commented out due to buggy nature in p0011/p0012. Remove this if running over 2022-2023 or the new 2024 data when it becomes available
    df = df.Filter("(TAP & 4) != 0", "Timing Trigger")

    #* Definitions
    df = df.Define("NTracks", "Track_nLayers.size()")
    df = df.Define("NPosTracks", "Track_nLayers[Track_charge > 0].size()")
    df = df.Define("NNegTracks", "Track_nLayers[Track_charge < 0].size()")
    df = df.Define("Track_nHits", "Track_nDoF + 5")        
    df = df.Define("Track_chi2_per_dof", "Track_Chi2/Track_nDoF")
    df = df.Define("GoodTracks", "Track_nLayers >= 7 && Track_chi2_per_dof < 25 && Track_nHits >= 12 && Track_pz0 > 20000" )
    df = df.Define("NGoodTracks", "Track_nLayers[GoodTracks].size()")
    df = df.Define("NGoodPosTracks", "Track_nLayers[GoodTracks && Track_charge > 0].size()")
    df = df.Define("NGoodNegTracks", "Track_nLayers[GoodTracks && Track_charge < 0].size()")    
    df = df.Define("Track_pz_charge0", "Track_pz0 * Track_charge")
    df = df.Define("Track_theta_x1", "asin(Track_px1/Track_p1)")
    df = df.Define("Track_theta_y1", "asin(Track_py1/Track_p1)")
    df = df.Define("Track_pt0", "sqrt(Track_px0*Track_px0 + Track_py0*Track_py0)")
    df = df.Define("Track_theta0", "asin(Track_pt0/Track_p0)")
    df = df.Define("Track_phi0", "acos(Track_px0/Track_pt0)")
    df = df.Define("Track_eta0", "-log(tan(Track_theta0/2))")
    df = df.Define("Track_theta_x0", "asin(Track_px0/Track_p0)")
    df = df.Define("Track_theta_y0", "asin(Track_py0/Track_p0)")
    df = df.Define("Track_theta_x0_pos", "Track_theta_x0[Track_charge > 0]")
    df = df.Define("Track_theta_x0_neg", "Track_theta_x0[Track_charge < 0]")
    df = df.Define("Track_theta_y0_pos", "Track_theta_y0[Track_charge > 0]")
    df = df.Define("Track_theta_y0_neg", "Track_theta_y0[Track_charge < 0]")
    df = df.Define("Track_x0_pos", "Track_x0[Track_charge > 0]")
    df = df.Define("Track_x0_neg", "Track_x0[Track_charge < 0]")
    df = df.Define("Track_y0_pos", "Track_y0[Track_charge > 0]")
    df = df.Define("Track_y0_neg", "Track_y0[Track_charge < 0]")

    df = df.Define("Timing_charge_bottom", "Timing0_raw_charge + Timing1_raw_charge")
    df = df.Define("Timing_charge_top", "Timing2_raw_charge + Timing3_raw_charge")
    df = df.Define("Timing_charge_total", "Timing_charge_top + Timing_charge_bottom")
    
    df = df.Define("hitsVetoNu0", "VetoNu0_raw_charge > 40")
    df = df.Define("hitsVetoNu1", "VetoNu1_raw_charge > 40")
    
    df = df.Define("hitsVeto10", "Veto10_raw_charge > 40")
    df = df.Define("hitsVeto11", "Veto11_raw_charge > 40")
    df = df.Define("hitsVeto20", "Veto20_raw_charge > 40")
    df = df.Define("hitsVeto21", "Veto21_raw_charge > 40")

    df = df.Define(f"hitsTiming", "((Track_Y_atTrig[0] > 20 && Timing_charge_top > 20) || \
                                           (Track_Y_atTrig[0] < -20 && Timing_charge_bottom > 20) || \
                                           (Track_Y_atTrig[0] > -20 && Track_Y_atTrig[0] < 20 && Timing_charge_total > 20))")
    
    df = df.Define("hitsTiming0", "Timing0_status == 0")
    df = df.Define("hitsTiming1", "Timing1_status == 0")
    df = df.Define("hitsTiming2", "Timing2_status == 0")
    df = df.Define("hitsTiming3", "Timing3_status == 0")
    
    df = df.Define("hitsPreshower0", "Preshower0_raw_charge > 2.5")
    df = df.Define("hitsPreshower1", "Preshower1_raw_charge > 2.5")

    df = df.Define("hitsCaloLo0", "CaloLo0_status == 0")
    df = df.Define("hitsCaloLo1", "CaloLo1_status == 0")
    df = df.Define("hitsCaloLo2", "CaloLo2_status == 0")
    df = df.Define("hitsCaloLo3", "CaloLo3_status == 0")

    # Brian says that the double peaks in the CaloHi channel are coming from muons hitting the PMTs rather than energy deposits
    # He suggests requiring that the CaloLo signal is at least 10x  higher than the CaloHi signal
    df = df.Define("hitsCaloHi0", "(CaloHi0_status == 0) && (CaloLo0_raw_charge > 10 * CaloHi0_raw_charge)")
    df = df.Define("hitsCaloHi1", "(CaloHi1_status == 0) && (CaloLo1_raw_charge > 10 * CaloHi1_raw_charge)")
    df = df.Define("hitsCaloHi2", "(CaloHi2_status == 0) && (CaloLo2_raw_charge > 10 * CaloHi2_raw_charge)")
    df = df.Define("hitsCaloHi3", "(CaloHi3_status == 0) && (CaloLo3_raw_charge > 10 * CaloHi3_raw_charge)")

    df = df.Define("LeadTrack_Idx", "ROOT::VecOps::ArgMax(Track_pz0)")
    df = df.Define("Track_rVetoNu", "pow(Track_X_atVetoNu[LeadTrack_Idx]*Track_X_atVetoNu[LeadTrack_Idx] + Track_Y_atVetoNu[LeadTrack_Idx]*Track_Y_atVetoNu[LeadTrack_Idx], 0.5)")
    # df = df.Define("Track_rVetoStation1", "pow(Track_X_atVetoStation1[LeadTrack_Idx]*Track_X_atVetoStation1[LeadTrack_Idx] + Track_Y_atVetoStation1[LeadTrack_Idx]*Track_Y_atVetoStation1[LeadTrack_Idx], 0.5)")
    # df = df.Define("Track_rVetoStation2", "pow(Track_X_atVetoStation2[LeadTrack_Idx]*Track_X_atVetoStation2[LeadTrack_Idx] + Track_Y_atVetoStation2[LeadTrack_Idx]*Track_Y_atVetoStation2[LeadTrack_Idx], 0.5)")
    df = df.Define("Track_rIFT", "Radius(Track_X_atVetoStation2, Track_Y_atVetoStation2)")
    df = df.Define("Track_xAvg", "(Track_X_atVetoNu + Track_X_atVetoStation1 + Track_X_atVetoStation2) / 3")
    df = df.Define("Track_yAvg", "((Track_Y_atVetoNu + Track_Y_atVetoStation1 + Track_Y_atVetoStation2) / 3 ) - 76.8 + 12.3") # y_LOS = 76.8 mm, delta_y = 12.3 mm
    df = df.Define("Track_Theta", "ROOT::VecOps::atan(ROOT::VecOps::sqrt(Track_xAvg*Track_xAvg + Track_yAvg*Track_yAvg) / (475.4 * 1000))") # dZ = 475.4 m (distance to ATLAS IP)
    # df = df.Define("Track_Theta", "Theta(Track_px0, Track_py0, Track_pz0)")
    # df = df.Define("Timing_charge_bottom", "Timing0_charge + Timing1_charge")
    # df = df.Define("Timing_charge_top", "Timing2_charge + Timing3_charge")
    # df = df.Define("Timing_charge_total", "Timing_charge_top + Timing_charge_bottom")
    df = df.Define("LeadTrack_pz0", "Track_pz0[LeadTrack_Idx] / 1000")
    df = df.Define("LeadTrack_Theta", "Track_Theta[LeadTrack_Idx]")
    df = df.Define("LeadTrack_Eta", "-ROOT::VecOps::log(ROOT::VecOps::tan(Track_Theta / 2))[LeadTrack_Idx]")
    

    #* Selection cuts
    df = df.Filter("(Timing0_status & 4) == 0 && (Timing1_status & 4) == 0 && (Timing2_status & 4) == 0 && (Timing3_status & 4) == 0 ", "No timing saturation")
    df = df.Filter("distanceToCollidingBCID == 0", "Colliding")
    df = df.Filter("(inputBits & 0x8) == 0x8 || (inputBits & 0x10) == 0x10 || (inputBits & 0x20) == 0x20 || (inputBits & 0x40) == 0x40", "Trigger")

    cut_veto_station_charge = ""
    # for veto_station_name in ["VetoSt10_charge", "VetoSt11_charge", "VetoSt20_charge", "VetoSt21_charge"]: # VetoSt11 was missing in 2023 (not enough DAQ channels available so was disconnected)
    for veto_station_name in ["VetoSt20_charge", "VetoSt21_charge"]: # Don't use VetoSt10 and VetoSt11 since we want to allow for neutrinos to interact in the lead shield region
        if veto_station_name in df.GetColumnNames():
            cut_veto_station_charge += f"{veto_station_name} > 40 && "
    cut_veto_station_charge = cut_veto_station_charge.rstrip(" && ")
    df = df.Filter(cut_veto_station_charge, "VetoStation Charge")

    df = df.Filter("longTracks >= 1", "At least one long track")
    df = df.Filter("((Track_Y_atTrig[LeadTrack_Idx] > 20 && Timing_charge_top > 20) || \
                (Track_Y_atTrig[LeadTrack_Idx] < -20 && Timing_charge_bottom > 20) || \
                (abs(Track_Y_atTrig[LeadTrack_Idx]) < 20 && Timing_charge_total > 20))", "Timing Station Charge")
    df = df.Filter("Preshower0_charge > 2.5 && Preshower1_charge > 2.5", "Preshower Charge")
    df = df.Filter("Track_nLayers[LeadTrack_Idx] >= 7", "Leading track has at >= 7 layers")
    df = df.Filter("Track_Chi2[LeadTrack_Idx] / Track_nDoF[LeadTrack_Idx] < 15", "Leading track has chi2/ndof < 15")
    df = df.Filter("Track_r_atMaxRadius[LeadTrack_Idx] < 95", "Track R at max radius < 95 mm")
    df = df.Filter("Track_rIFT[LeadTrack_Idx] < 95", "Track R at IFT < 95 mm")
    df = df.Filter("Track_rVetoNu < 120", "Track R at veto nu < 120 mm")
    df = df.Filter("LeadTrack_Theta < 0.025 || Track_pz0[LeadTrack_Idx] > 100 * 1000", "Track theta < 25 mrad or track pz0 < 100")
    

    return df


def parse_input_filelists(input_file_list_dir):

    txt_files = glob.glob(f"{input_file_list_dir}/*.txt")

    file_dict = {}

    for fpath in txt_files:
        with open(fpath, 'r') as f:
            for line in f:
                if line.startswith("#"): continue

                print(line)
                the_file_path = line.strip().strip("\n")
                the_file_name = os.path.basename(the_file_path)
                the_run_number = the_file_name.split("-")[2]
                the_run_number = int(the_run_number)

                if the_run_number in file_dict.keys():
                    file_dict[the_run_number].append(the_file_path)
                else:
                    file_dict[the_run_number] = [the_file_path]    
    return file_dict


def main(args: argparse.Namespace) -> None:

    #* Enable multithreading
    ROOT.ROOT.EnableImplicitMT()

    #* Parse input files
    all_files_dict = parse_input_filelists(args.input_file_list_dir)
    file_list = all_files_dict[args.run]

    if len(file_list) <= 0:
        print("Error: Found no files to run over")
        return 1

    print(f"Info: Running over {len(file_list)} files for run {args.run}")
    for file in file_list:
        print(f"    - {file}")

    #* Get lumi dict
    lumi_dict = {}
    lumi_dict = get_run_number_lumi_dict(args.grl_path)
    run_lumi = lumi_dict.get(args.run, None)
    print(f"Info: Run {args.run} luminosity = {run_lumi:.3f} /pb")

    #* Construct dataframe
    df = build_dataframe(file_list, is_mc=args.is_mc)

    #* Make output file (and output directory if needs be)
    output_file = f"{args.run}.root"


    # print(df.GetColumnNames())

    df.Snapshot("nt", output_file, columnList=["run", "eventID", "LeadTrack_pz0", "LeadTrack_Theta"])
    
    wf_files = []
    with open("waveform-file-list.txt", "r") as f:
        for line in f:
            if str(args.run) in line:
                wf_files.append(line.strip())

    os.system("export EOS_MGM_URL=root://eosuser.cern.ch")

    wf_df = ROOT.RDataFrame("tree", wf_files)

    df_joined = df.Join(wf_df, {"eventID"}, {"eventID"}, "df_", "wf_df_")



    #* Move output file to output directory
    os.makedirs(args.output_file_dir, exist_ok=True)
    os.makedirs(f"{args.output_file_dir}/logs", exist_ok=True) # just in case the log directory doesn't exist
    print(f"Info: transferring output file: {output_file} -> {args.output_file_dir}/{output_file}")
    shutil.move(output_file, f"{args.output_file_dir}/{output_file}")
    

    #* Print cutflow
    print("\nInfo: Cutflow Report:")
    cutReport = df.Report()
    cutReport.Print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=int, help="Run to select")
    parser.add_argument("--input_file_list_dir", "-i", help="directory to txt files containing the available NTuple paths", default=f"{os.getcwd()}/faser_filelists")
    parser.add_argument("--output_file_dir", "-o", type=str, default="output", help = "Output file directory")
    parser.add_argument("--is_mc", "-m", action="store_true" , help = "Output file directory")
    parser.add_argument("--grl_path", "-g", type=str, default="/cvmfs/faser.cern.ch/repo/sw/runlist/v9", help = "Path to directory containing GRL files in the .json format")
    args = parser.parse_args()

    for key in vars(args):
        print(f"\t {key:<30}: {getattr(args, key)}")

    # Make sure all path args are absolute paths so condor doens't get lost
    args.input_file_list_dir = os.path.abspath(args.input_file_list_dir)
    args.output_file_dir = os.path.abspath(args.output_file_dir)
    args.grl_path = os.path.abspath(args.grl_path)


    main(args)