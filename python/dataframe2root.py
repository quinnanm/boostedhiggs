#script for converting pandas dataframe to rootfile
#import fastparquet
import os
import uproot
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
# from root_numpy import root2array, array2root, fill_hist, array2tree
# from rootpy.tree import Tree, TreeModel, FloatCol
# import ROOT as r
pd.options.mode.chained_assignment = None  # default='warn'

parser = argparse.ArgumentParser(description='converting pandas dataframe to rootfile ')
parser.add_argument("-p", "--proc", dest="proc", default='parquet') #specify file type like QCD
parser.add_argument("-y", "--year", dest="year", default='2016')
parser.add_argument("-t", "--treename", dest="treename", default='Events')
parser.add_argument("-d", "--data", dest="isdata", default='False')
args = parser.parse_args()

indir = '/eos/uscms/store/user/fmokhtar/boostedhiggs/Apr4_'+args.year
filetype='.parquet' 
outdir = './rootfiles/'
print(args.proc+' must be in file')

for subdir, dirs, files in os.walk(indir):
    for file in files:
        #load files
        if ('had.parquet' in file) and (args.proc in subdir):
            f=subdir+'/'+file
            # f=f.replace("/eos/uscms/","root://cmseos.fnal.gov//")
            outf=f
            print('prepping input file', f, '...')
            outname = outf[outf.rfind(args.year+'/')+5:]
            outname = outdir+outname.strip('.'+filetype)+'.root'
            outname = outname.replace('/outfiles/', '_')
            
            #load parquet into dataframe
            print('loading dataframe...')
            table = pq.read_table(f)
            data = table.to_pandas()
            print('# input events:', len(data)  )
            if len(data)==0:
                print('no skimmed events. skipping')
                continue

            #here is where you can add branches to the tree, skim the selection to include fewer events, etc   

            #fill dataframe to rootfile
            # array2root(data.to_records(index=False), filename=outname, treename=args.treename, mode='RECREATE') #dont use, requires root
            file = uproot.recreate(outname) 
            file[args.treename]=uproot.newtree(data)
            print('Wrote rootfile ',outname)

