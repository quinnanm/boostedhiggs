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
parser.add_argument("-p",   "--proc",     dest="proc", default='parquet')  # specify file type like QCD
parser.add_argument("-t",   "--treename", dest="treename", default='Events')
parser.add_argument("-d",   "--data", dest="isdata", default='False')
parser.add_argument("-ch",  "--ch",   dest="ch", default='had')
parser.add_argument("-dir", "--dir",  dest="dir", default='Apr20_2016')
args = parser.parse_args()


if __name__ == "__main__":
    """
    e.g. run as:
    python -u dataframe2root.py --ch='had' --proc='QCD' --dir='Apr20_2016'
    """

    year = args.dir[-4:]
    indir = '/eos/uscms/store/user/fmokhtar/boostedhiggs/' + args.dir
    filetype = '.parquet'

    # make directory to hold rootfiles
    outdir = './rootfiles/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outdir = './rootfiles/' + args.proc + '/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outdir = './rootfiles/' + args.proc + '/' + args.ch + '/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(f'processing {args.ch} channel')
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            # load files
            if (f'{args.ch}.parquet' in file) and (args.proc in subdir):
                f = subdir + '/' + file
                # f=f.replace("/eos/uscms/","root://cmseos.fnal.gov//")
                outf = f
                print('prepping input file', f, '...')
                outname = outf[outf.rfind(year + '/') + 5:]
                outname = outdir + outname.strip('.' + filetype) + '.root'
                outname = outname.replace('/outfiles/', '_')

                # load parquet into dataframe
                print('loading dataframe...')
                table = pq.read_table(f)
                data = table.to_pandas()
                print('# input events:', len(data))
                if len(data) == 0:
                    print('no skimmed events. skipping')
                    continue

                print(data.keys())
                # here is where you can add branches to the tree, skim the selection to include fewer events, etc

                # fill dataframe to rootfile
                # array2root(data.to_records(index=False), filename=outname, treename=args.treename, mode='RECREATE') #dont use, requires root
                with uproot.recreate(outname) as file:
                    if args.ch != 'had':
                        file[args.treename] = pd.DataFrame(data['fj_pt'])
                    else:
                        file[args.treename] = pd.DataFrame(data['fj0_pt'])

                print('Wrote rootfile ', outname)

                # you can further do hadd name_merged.root files*.root to merge the files per sample
