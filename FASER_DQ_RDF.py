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


def book_per_run_hists(df: ROOT.RDataFrame, run_number: int=None, lumi: float=None) -> List:
    """
    Function to book histograms which need to be evaluated for each run seperately
    
    args:
        df: ROOT.RDataFrame - the dataframe used to fill the histograms
        run_number: int (optional) - the run number to select for if given
        lumi: float (optional) - the luminosity of the run in fb; if given then weight histograms by /lumi

    returns:
       per_run_histos: [List] - List of histograms to be filled
    """

    if run_number:
        df_this_run = df.Filter(f"run == {run_number}", f"Run: {run_number}")
    else:
        df_this_run = df

    # For the lumi weighting, only do the RDF.Count if we need to for performance
    weight = 1
    nevents = 1
    if lumi:
        nevents = df_this_run.Count().GetValue()
        weight = lumi
        if nevents == 0:
            print(f"Warning: nevents = 0, setting to 1")
            nevents = 1
        print(f"Info: Weight = {weight / nevents}")

    df_this_run = df_this_run.Define("weight", f"{weight / nevents}")

    GeV = 1000
    per_run_histos = []
    # per_run_histos.append(df_this_run.Histo1D(("CaloEnergyEMZoom", "CaloEnergyEMZoom;CaloEnergyEMZoom;Events", 1000, 0*GeV, 100*GeV), f"Calo_total_E_EM_fudged", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_theta_x0", "Track_theta_x0;Track_theta_x0;Events", 50, -0.01, 0.01), f"Track_theta_x0", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_theta_y0", "Track_theta_x0;Track_theta_x0;Events", 50, -0.005, 0.005), f"Track_theta_y0", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_theta_x1", "Track_theta_x1;Track_theta_x1;Events", 50, -0.01, 0.01), f"Track_theta_x1", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_theta_y1", "Track_theta_y1;Track_theta_y1;Events", 50, -0.005, 0.005), f"Track_theta_y1", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_theta_y1", "Track_theta_y1;Track_theta_y1;Events", 50, -0.005, 0.005), f"Track_theta_y1", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_pz_charge0", "Track_pz_charge0;Track_pz_charge0;Events", 100, -500*GeV, 500*GeV), f"Track_pz_charge0", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_Chi2", "Track_Chi2;Track_Chi2;Events", 50, 0, 50), f"Track_Chi2", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_Chi2_2", "Track_Chi2_2;Track_Chi2_2;Events", 50, 0, 500), f"Track_Chi2", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_chi2_per_dof", "Track_chi2_per_dof;Track_chi2_per_dof;Events", 50,0,25), f"Track_chi2_per_dof", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_nDoF", "Track_nDoF;Track_nDoF;Events", 20, 0, 20), f"Track_nDoF", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_x0", "Track_x0;Track_x0;Events", 100, -100, 100), f"Track_x0", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_y0", "Track_y0;Track_y0;Events", 100, -100, 100), f"Track_y0", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_theta_x0_pos", "Track_theta_x0_pos;Track_theta_x0_pos;Events", 50, -0.01, 0.01), f"Track_theta_x0_pos", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_theta_x0_neg", "Track_theta_x0_neg;Track_theta_x0_neg;Events", 50, -0.01, 0.01), f"Track_theta_x0_neg", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_theta_y0_pos", "Track_theta_y0_pos;Track_theta_y0_pos;Events", 50, -0.005, 0.005), f"Track_theta_y0_pos", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Track_theta_y0_neg", "Track_theta_y0_neg;Track_theta_y0_neg;Events", 50, -0.005, 0.005), f"Track_theta_y0_neg", "weight"))

    per_run_histos.append(df_this_run.Histo1D(("VetoNu0_charge", "VetoNu0_charge;VetoNu0_charge;Events", 50, 0.01, 300.0), f"VetoNu0_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("VetoNu1_charge", "VetoNu1_charge;VetoNu1_charge;Events", 50, 0.01, 300.0), f"VetoNu1_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("VetoSt10_charge", "VetoSt10_charge;VetoSt10_charge;Events", 50, 0.01, 300.0), f"VetoSt10_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("VetoSt20_charge", "VetoSt20_charge;VetoSt20_charge;Events", 50, 0.01, 300.0), f"VetoSt20_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("VetoSt21_charge", "VetoSt21_charge;VetoSt21_charge;Events", 50, 0.01, 300.0), f"VetoSt21_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Timing0_charge", "Timing0_charge;Timing0_charge;Events", 50, 1.0, 80.0), f"Timing0_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Timing1_charge", "Timing1_charge;Timing1_charge;Events", 50, 1.0, 80.0), f"Timing1_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Timing2_charge", "Timing2_charge;Timing2_charge;Events", 50, 1.0, 80.0), f"Timing2_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Timing3_charge", "Timing3_charge;Timing3_charge;Events", 50, 1.0, 80.0), f"Timing3_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Preshower0_charge", "Preshower0_charge;Preshower0_charge;Events", 50, 0.01, 15.0), f"Preshower0_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Preshower1_charge", "Preshower1_charge;Preshower1_charge;Events", 50, 0.01, 15.0), f"Preshower1_charge", "weight"))

    df_hits_vetonu0 = df_this_run.Filter("hitsVetoNu0", "hitsVetoNu0")
    df_hits_vetonu1 = df_this_run.Filter("hitsVetoNu1", "hitsVetoNu0")
    per_run_histos.append(df_hits_vetonu0.Histo1D(("VetoNu0_triggertime", "VetoNu0_triggertime;VetoNu0_triggertime;Events", 100, -50, 25), f"VetoNu0_triggertime", "weight"))
    per_run_histos.append(df_hits_vetonu0.Histo1D(("VetoNu0_localtime", "VetoNu0_localtime;VetoNu0_localtime;Events", 100, 100, 200), f"VetoNu0_localtime", "weight"))
    per_run_histos.append(df_hits_vetonu0.Histo1D(("VetoNu0_bcidtime", "VetoNu0_bcidtime;VetoNu0_bcidtime;Events", 100, 0, 30), f"VetoNu0_bcidtime", "weight"))
    
    per_run_histos.append(df_hits_vetonu1.Histo1D(("VetoNu1_triggertime", "VetoNu1_triggertime;VetoNu1_triggertime;Events", 100, -50, 25), f"VetoNu1_triggertime", "weight"))
    per_run_histos.append(df_hits_vetonu1.Histo1D(("VetoNu1_localtime", "VetoNu1_localtime;VetoNu1_localtime;Events", 100, 100, 200), f"VetoNu1_localtime", "weight"))
    per_run_histos.append(df_hits_vetonu1.Histo1D(("VetoNu1_bcidtime", "VetoNu1_bcidtime;VetoNu1_bcidtime;Events", 100, 0, 30), f"VetoNu1_bcidtime", "weight"))
    
    df_hits_veto10 = df_this_run.Filter("hitsVeto10")
    df_hits_veto11 = df_this_run.Filter("hitsVeto11")
    df_hits_veto20 = df_this_run.Filter("hitsVeto20")
    df_hits_veto21 = df_this_run.Filter("hitsVeto21")
    per_run_histos.append(df_hits_veto10.Histo1D(("VetoSt10_triggertime", "VetoSt10_triggertime;VetoSt10_triggertime;Events", 100, -100, 50), f"Veto10_triggertime", "weight"))
    per_run_histos.append(df_hits_veto10.Histo1D(("VetoSt10_localtime", "VetoSt10_localtime;VetoSt10_localtime;Events", 100, 120, 225), f"Veto10_localtime", "weight"))
    per_run_histos.append(df_hits_veto10.Histo1D(("VetoSt10_bcidtime", "VetoSt10_bcidtime;VetoSt10_bcidtime;Events", 100, 0, 30), f"Veto10_bcidtime", "weight"))

    per_run_histos.append(df_hits_veto11.Histo1D(("VetoSt11_triggertime", "VetoSt11_triggertime;VetoSt11_triggertime;Events", 100, -100, 50), f"Veto11_triggertime", "weight"))
    per_run_histos.append(df_hits_veto11.Histo1D(("VetoSt11_localtime", "VetoSt11_localtime;VetoSt11_localtime;Events", 100, 120, 225), f"Veto11_localtime", "weight"))
    per_run_histos.append(df_hits_veto11.Histo1D(("VetoSt11_bcidtime", "VetoSt11_bcidtime;VetoSt11_bcidtime;Events", 100, 0, 30), f"Veto11_bcidtime", "weight"))    

    per_run_histos.append(df_hits_veto20.Histo1D(("VetoSt20_triggertime", "VetoSt20_triggertime;VetoSt20_triggertime;Events", 100, -100, 50), f"Veto20_triggertime", "weight"))
    per_run_histos.append(df_hits_veto20.Histo1D(("VetoSt20_localtime", "VetoSt20_localtime;VetoSt20_localtime;Events", 100, 120, 225), f"Veto20_localtime", "weight"))
    per_run_histos.append(df_hits_veto20.Histo1D(("VetoSt20_bcidtime", "VetoSt20_bcidtime;VetoSt20_bcidtime;Events", 100, 0, 30), f"Veto20_bcidtime", "weight"))

    per_run_histos.append(df_hits_veto21.Histo1D(("VetoSt21_triggertime", "VetoSt21_triggertime;VetoSt21_triggertime;Events", 100, -100, 50), f"Veto21_triggertime", "weight"))
    per_run_histos.append(df_hits_veto21.Histo1D(("VetoSt21_localtime", "VetoSt21_localtime;VetoSt21_localtime;Events", 100, 120, 225), f"Veto21_localtime", "weight"))
    per_run_histos.append(df_hits_veto21.Histo1D(("VetoSt21_bcidtime", "VetoSt21_bcidtime;VetoSt21_bcidtime;Events", 100, 0, 30), f"Veto21_bcidtime", "weight"))
    
    df_hits_timing0 = df_this_run.Filter("hitsTiming0")
    df_hits_timing1 = df_this_run.Filter("hitsTiming1")
    df_hits_timing2 = df_this_run.Filter("hitsTiming2")
    df_hits_timing3 = df_this_run.Filter("hitsTiming3")
    per_run_histos.append(df_hits_timing0.Histo1D(("Timing0_triggertime", "Timing0_triggertime;Timing0_triggertime;Events", 100, -75, 50), f"Timing0_triggertime", "weight"))
    per_run_histos.append(df_hits_timing0.Histo1D(("Timing0_localtime", "Timing0_localtime;Timing0_localtime;Events", 100, 150, 225), f"Timing0_localtime", "weight"))
    per_run_histos.append(df_hits_timing0.Histo1D(("Timing0_bcidtime", "Timing0_bcidtime;Timing0_bcidtime;Events", 100, 0, 30), f"Timing0_bcidtime", "weight"))
    
    per_run_histos.append(df_hits_timing1.Histo1D(("Timing1_triggertime", "Timing1_triggertime;Timing1_triggertime;Events", 100, -75, 50), f"Timing1_triggertime", "weight"))
    per_run_histos.append(df_hits_timing1.Histo1D(("Timing1_localtime", "Timing1_localtime;Timing1_localtime;Events", 100, 150, 225), f"Timing1_localtime", "weight"))
    per_run_histos.append(df_hits_timing1.Histo1D(("Timing1_bcidtime", "Timing1_bcidtime;Timing1_bcidtime;Events", 100, 0, 30), f"Timing1_bcidtime", "weight"))
    
    per_run_histos.append(df_hits_timing2.Histo1D(("Timing2_triggertime", "Timing2_triggertime;Timing2_triggertime;Events", 100, -75, 50), f"Timing2_triggertime", "weight"))
    per_run_histos.append(df_hits_timing2.Histo1D(("Timing2_localtime", "Timing2_localtime;Timing2_localtime;Events", 100, 150, 225), f"Timing2_localtime", "weight"))
    per_run_histos.append(df_hits_timing2.Histo1D(("Timing2_bcidtime", "Timing2_bcidtime;Timing2_bcidtime;Events", 100, 0, 30), f"Timing2_bcidtime", "weight"))
    
    per_run_histos.append(df_hits_timing3.Histo1D(("Timing3_triggertime", "Timing3_triggertime;Timing3_triggertime;Events", 100, -75, 50), f"Timing3_triggertime", "weight"))
    per_run_histos.append(df_hits_timing3.Histo1D(("Timing3_localtime", "Timing3_localtime;Timing3_localtime;Events", 100, 150, 225), f"Timing3_localtime", "weight"))
    per_run_histos.append(df_hits_timing3.Histo1D(("Timing3_bcidtime", "Timing3_bcidtime;Timing3_bcidtime;Events", 100, 0, 30), f"Timing3_bcidtime", "weight"))

    df_hits_preshower0 = df_this_run.Filter("hitsPreshower0")
    df_hits_preshower1 = df_this_run.Filter("hitsPreshower1")
    per_run_histos.append(df_hits_preshower0.Histo1D(("Preshower0_triggertime", "Preshower0_triggertime;Preshower0_triggertime;Events", 100, -75, 50), f"Preshower0_triggertime", "weight"))
    per_run_histos.append(df_hits_preshower0.Histo1D(("Preshower0_localtime", "Preshower0_localtime;Preshower0_localtime;Events", 100, 150, 225), f"Preshower0_localtime", "weight"))
    per_run_histos.append(df_hits_preshower0.Histo1D(("Preshower0_bcidtime", "Preshower0_bcidtime;Preshower0_bcidtime;Events", 100, 0, 30), f"Preshower0_bcidtime", "weight"))
    
    per_run_histos.append(df_hits_preshower1.Histo1D(("Preshower1_triggertime", "Preshower1_triggertime;Preshower1_triggertime;Events", 100, -75, 50), f"Preshower1_triggertime", "weight"))
    per_run_histos.append(df_hits_preshower1.Histo1D(("Preshower1_localtime", "Preshower1_localtime;Preshower1_localtime;Events", 100, 150, 225), f"Preshower1_localtime", "weight"))
    per_run_histos.append(df_hits_preshower1.Histo1D(("Preshower1_bcidtime", "Preshower1_bcidtime;Preshower1_bcidtime;Events", 100, 0, 30), f"Preshower1_bcidtime", "weight"))
    
    df_hits_calolo0 = df_this_run.Filter("hitsCaloLo0")
    df_hits_calolo1 = df_this_run.Filter("hitsCaloLo1")
    df_hits_calolo2 = df_this_run.Filter("hitsCaloLo2")
    df_hits_calolo3 = df_this_run.Filter("hitsCaloLo3")
    per_run_histos.append(df_hits_calolo0.Histo1D(("CaloLo0_triggertime", "CaloLo0_triggertime;CaloLo0_triggertime;Events", 100, -75, 50), f"CaloLo0_triggertime", "weight"))
    per_run_histos.append(df_hits_calolo0.Histo1D(("CaloLo0_localtime", "CaloLo0_localtime;CaloLo0_localtime;Events", 100, 150, 225), f"CaloLo0_localtime", "weight"))
    per_run_histos.append(df_hits_calolo0.Histo1D(("CaloLo0_bcidtime", "CaloLo0_bcidtime;CaloLo0_bcidtime;Events", 100, 0, 30), f"CaloLo0_bcidtime", "weight"))

    per_run_histos.append(df_hits_calolo1.Histo1D(("CaloLo1_triggertime", "CaloLo1_triggertime;CaloLo1_triggertime;Events", 100, -75, 50), f"CaloLo1_triggertime", "weight"))
    per_run_histos.append(df_hits_calolo1.Histo1D(("CaloLo1_localtime", "CaloLo1_localtime;CaloLo1_localtime;Events", 100, 150, 225), f"CaloLo1_localtime", "weight"))
    per_run_histos.append(df_hits_calolo1.Histo1D(("CaloLo1_bcidtime", "CaloLo1_bcidtime;CaloLo1_bcidtime;Events", 100, 0, 30), f"CaloLo1_bcidtime", "weight"))

    per_run_histos.append(df_hits_calolo2.Histo1D(("CaloLo2_triggertime", "CaloLo2_triggertime;CaloLo2_triggertime;Events", 100, -75, 50), f"CaloLo2_triggertime", "weight"))
    per_run_histos.append(df_hits_calolo2.Histo1D(("CaloLo2_localtime", "CaloLo2_localtime;CaloLo2_localtime;Events", 100, 150, 225), f"CaloLo2_localtime", "weight"))
    per_run_histos.append(df_hits_calolo2.Histo1D(("CaloLo2_bcidtime", "CaloLo2_bcidtime;CaloLo2_bcidtime;Events", 100, 0, 30), f"CaloLo2_bcidtime", "weight"))

    per_run_histos.append(df_hits_calolo3.Histo1D(("CaloLo3_triggertime", "CaloLo3_triggertime;CaloLo3_triggertime;Events", 100, -75, 50), f"CaloLo3_triggertime", "weight"))
    per_run_histos.append(df_hits_calolo3.Histo1D(("CaloLo3_localtime", "CaloLo3_localtime;CaloLo3_localtime;Events", 100, 150, 225), f"CaloLo3_localtime", "weight"))
    per_run_histos.append(df_hits_calolo3.Histo1D(("CaloLo3_bcidtime", "CaloLo3_bcidtime;CaloLo3_bcidtime;Events", 100, 0, 30), f"CaloLo3_bcidtime", "weight"))
    
    df_hits_calohi0 = df_this_run.Filter("hitsCaloHi0")
    df_hits_calohi1 = df_this_run.Filter("hitsCaloHi1")
    df_hits_calohi2 = df_this_run.Filter("hitsCaloHi2")
    df_hits_calohi3 = df_this_run.Filter("hitsCaloHi3")
    per_run_histos.append(df_hits_calohi0.Histo1D(("CaloHi0_triggertime", "CaloHi0_triggertime;CaloHi0_triggertime;Events", 100,  -75, 50), f"CaloHi0_triggertime", "weight"))
    per_run_histos.append(df_hits_calohi0.Histo1D(("CaloHi0_localtime", "CaloHi0_localtime;CaloHi0_localtime;Events", 100, 150, 225), f"CaloHi0_localtime", "weight"))
    per_run_histos.append(df_hits_calohi0.Histo1D(("CaloHi0_bcidtime", "CaloHi0_bcidtime;CaloHi0_bcidtime;Events", 100, 0, 30), f"CaloHi0_bcidtime", "weight"))

    per_run_histos.append(df_hits_calohi1.Histo1D(("CaloHi1_triggertime", "CaloHi1_triggertime;CaloHi1_triggertime;Events", 100,  -75, 50), f"CaloHi1_triggertime", "weight"))
    per_run_histos.append(df_hits_calohi1.Histo1D(("CaloHi1_localtime", "CaloHi1_localtime;CaloHi1_localtime;Events", 100, 150, 225), f"CaloHi1_localtime", "weight"))
    per_run_histos.append(df_hits_calohi1.Histo1D(("CaloHi1_bcidtime", "CaloHi1_bcidtime;CaloHi1_bcidtime;Events", 100, 0, 30), f"CaloHi1_bcidtime", "weight"))

    per_run_histos.append(df_hits_calohi2.Histo1D(("CaloHi2_triggertime", "CaloHi2_triggertime;CaloHi2_triggertime;Events", 100,  -75, 50), f"CaloHi2_triggertime", "weight"))
    per_run_histos.append(df_hits_calohi2.Histo1D(("CaloHi2_localtime", "CaloHi2_localtime;CaloHi2_localtime;Events", 100, 150, 225), f"CaloHi2_localtime", "weight"))
    per_run_histos.append(df_hits_calohi2.Histo1D(("CaloHi2_bcidtime", "CaloHi2_bcidtime;CaloHi2_bcidtime;Events", 100, 0, 30), f"CaloHi2_bcidtime", "weight"))

    per_run_histos.append(df_hits_calohi3.Histo1D(("CaloHi3_triggertime", "CaloHi3_triggertime;CaloHi3_triggertime;Events", 100,  -75, 50), f"CaloHi3_triggertime", "weight"))
    per_run_histos.append(df_hits_calohi3.Histo1D(("CaloHi3_localtime", "CaloHi3_localtime;CaloHi3_localtime;Events", 100, 150, 225), f"CaloHi3_localtime", "weight"))    
    per_run_histos.append(df_hits_calohi3.Histo1D(("CaloHi3_bcidtime", "CaloHi3_bcidtime;CaloHi3_bcidtime;Events", 100, 0, 30), f"CaloHi3_bcidtime", "weight"))


    per_run_histos.append(df_this_run.Histo1D(("Calo0_charge", "Calo0_charge;Calo0_charge;Events", 100, 0.01, 4.0), f"Calo0_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Calo1_charge", "Calo1_charge;Calo1_charge;Events", 100, 0.01, 4.0), f"Calo1_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Calo2_charge", "Calo2_charge;Calo2_charge;Events", 100, 0.01, 4.0), f"Calo2_charge", "weight"))
    per_run_histos.append(df_this_run.Histo1D(("Calo3_charge", "Calo3_charge;Calo3_charge;Events", 100, 0.01, 4.0), f"Calo3_charge", "weight"))

    event_times = np.array(df_this_run.AsNumpy(["eventTime"])["eventTime"])
    per_run_histos.append(df_this_run.Histo1D(("eventTime", "eventTime;eventTime;Events", 100, np.amin(event_times)-1, np.amax(event_times)+1), f"eventTime", "weight"))

    return per_run_histos


def book_yield_hists(df: ROOT.RDataFrame, run_number: int) -> List:

    yield_hists = []
    runs = [run_number]

    nruns = int(max(runs) - min(runs) + 1)
    rmin =  min(runs)
    rmax =  max(runs) + 1

    yield_hists.append(df.Histo1D(("Yield", "Yield;Yield;Events", nruns, rmin, rmax), "run"))
    yield_hists.append(df.Histo1D(("TrkYield", "TrkYield;TrkYield;Events", nruns, rmin, rmax), "run", "NTracks"))
    yield_hists.append(df.Histo1D(("PosTrkYield", "PosTrkYield;PosTrkYield;Events", nruns, rmin, rmax), "run", "NPosTracks"))
    yield_hists.append(df.Histo1D(("NegTrkYield", "NegTrkYield;NegTrkYield;Events", nruns, rmin, rmax), "run", "NNegTracks"))
    
    yield_hists.append(df.Histo1D(("GoodTrkYield", "GoodTrkYield;GoodTrkYield;Events", nruns, rmin, rmax), "run", "NGoodTracks"))
    yield_hists.append(df.Histo1D(("GoodPosTrkYield", "GoodPosTrkYield;GoodPosTrkYield;Events", nruns, rmin, rmax), "run", "NGoodPosTracks"))
    yield_hists.append(df.Histo1D(("GoodNegTrkYield", "GoodNegTrkYield;GoodNegTrkYield;Events", nruns, rmin, rmax), "run", "NGoodNegTracks"))
    return yield_hists


def build_dataframe(file_list: List[str], tree: str='nt') -> ROOT.RDataFrame:
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
    df = df.Define("GoodTracks", "Track_nLayers >= 7 && Track_chi2_per_dof < 25 && Track_nHits >= 12 && Track_pz0 > 20000 && RemoveDuplicates(Track_p0)" )
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
    df = build_dataframe(file_list)
    yield_hists = book_yield_hists(df, args.run)
    run_hists = book_per_run_hists(df, args.run, lumi=run_lumi)

    #* Make output file (and output directory if needs be)
    output_file = f"{args.run}.root"
    file = ROOT.TFile(output_file, "RECREATE")
    tree = ROOT.TTree("dq", "Data Quality")
    print(f"Info: Writing output to {output_file}")
    
    #* Write out run number and lumi for convenience
    lumi_branch = ROOT.std.vector("float")()
    lumi_branch.push_back(run_lumi)
    tree.Branch("lumi", lumi_branch)

    run_num_branch = ROOT.std.vector("int")()
    run_num_branch.push_back(args.run)
    tree.Branch("run_number", run_num_branch)


    #* Write histograms
    for h in tqdm(yield_hists, desc="Info: Filling yield hists: "):
        h.Write()
    for h in tqdm(run_hists, desc="Info: Filling run hists: "):
        h.Write()

    #* Close the file
    tree.Fill()
    tree.Write()
    file.Close()
    print(f"Info: Wrote output to {output_file}")

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
    parser.add_argument("--grl_path", "-g", type=str, default="/cvmfs/faser.cern.ch/repo/sw/runlist/v8", help = "Path to directory containing GRL files in the .json format")
    args = parser.parse_args()

    for key in vars(args):
        print(f"\t {key:<30}: {getattr(args, key)}")

    # Make sure all path args are absolute paths so condor doens't get lost
    args.input_file_list_dir = os.path.abspath(args.input_file_list_dir)
    args.output_file_dir = os.path.abspath(args.output_file_dir)
    args.grl_path = os.path.abspath(args.grl_path)


    main(args)