#!/bin/bash

export EOS_MGM_URL=root://eospublic.cern.ch
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/setup.sh
run=$1
output_dir=$2

cp /home/ppd/bewilson/FASER_2024_DQ_Analyis/RDFDefines.h .
python3 /home/ppd/bewilson/FASER_2024_DQ_Analyis/FASER_DQ_RDF.py $run -o $output_dir -i /home/ppd/bewilson/FASER_2024_DQ_Analyis/faser_filelists_back -g /home/ppd/bewilson/FASER_2024_DQ_Analyis/v9