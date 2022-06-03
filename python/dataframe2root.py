# script for converting pandas dataframe to rootfile
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
parser.add_argument("-p", "--process", dest="process", default='process')  # specify file type like QCD
parser.add_argument("-y", "--year", dest="year", default='2016')
parser.add_argument("-t", "--treename", dest="treename", default='Events')
parser.add_argument("-d", "--data", dest="isdata", default='False')
args = parser.parse_args()

indir = '/eos/uscms/store/user/cmantill/boostedhiggs/May7_2017'
filetype = '.parquet'
outdir = './rootfiles/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

print(args.process + ' must be in file')

for subdir, dirs, files in os.walk(indir):
    for file in files:
        # load files
        if ('had.parquet' in file) and ('QCD' in subdir):
            f = subdir + '/' + file

            # load parquet into dataframe
            print('loading dataframe...')
            table = pq.read_table(f)
            data = table.to_pandas()
            print('# input events:', len(data))
            if len(data) == 0:
                print('no skimmed events. skipping')
                continue

            outname = outdir + file[:-8] + '.root'  # the slice removes the .parquet extension (to replace it with a .root extension)
            with uproot.recreate(outdir + file[:-8] + '.root') as file:
                file[args.treename] = pd.DataFrame(data)

            print('Wrote rootfile ', outname)
