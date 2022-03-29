#potential script for converting pandas dataframe to rootfile
import subprocess
import os.path
from root_numpy import root2array, array2root, fill_hist, array2tree
from rootpy.tree import Tree, TreeModel, FloatCol
from rootpy.io import root_open
import argparse
from array import array
import numpy as np
import pandas as pd
import ROOT as r
import inspect
pd.options.mode.chained_assignment = None  # default='warn'

parser = argparse.ArgumentParser(description='converting pandas dataframe to rootfile ')
parser.add_argument("-f", "--filenames", dest="filenames", nargs="*", default=['./testfiles/bla.root'])
parser.add_argument("-t", "--treename", dest="treename", default='Events')
parser.add_argument("-o", "--outdir", dest="outdir", default='./rootfiles/')
parser.add_argument("-d", "--data", dest="isdata", default='False')
args = parser.parse_args()


filetype='pkl' #replace with proper filetype

for f in args.filenames: #get proper outname if files are eos files
    outf=f
    f=f.replace("/eos/uscms/","root://cmseos.fnal.gov/")
    print 'prepping input file', f, '...'
    outname = outf[outf.rfind('/')+1:]
    outname = outname.strip('.'+filetype)

    #load dataframe
    data = pd.DataFrame(f) #not sure about the syntax here
    print '# input events:', len(data)
    if len(data)==0:
        print 'no skimmed events. skipping'
        continue

    #here is where you can add branches to the tree, skim the selection to include fewer events, etc

    #fill dataframe to rootfile
    array2root(data.to_records(index=False), filename=args.outdir+outname, treename=args.treename, mode='RECREATE')

print 'Wrote', args.outdir+outname

#run using python dataframe2root.py -f /dir/files* 
