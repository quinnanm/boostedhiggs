#a python script for properly generating datacards according to SR categories
import os
import sys
import math
import numpy as np
import argparse
import uproot
import types
from collections import OrderedDict
from ROOT import gROOT,TFile,TTree,TH1D
import ROOT as r
from datacardrefs import datacardrefs
from histmaker import histmaker
gROOT.SetBatch(True)

# can run datacards for example using python autodatacards.py -y 2017

parser = argparse.ArgumentParser(description='Datacard information')
oparser.add_argument("-o", "--outdir",   dest="outdir",   default='./datacards/',                      help="directory to put datacards in")
parser.add_argument("-y", "--year",     dest="year",     default='2016',                               help="year")
parser.add_argument("-f", "--shapefile",    dest="shapefile",    default='shapehists.root',            help="file where shape histograms are located")
parser.add_argument("-r", "--remakehists",    dest="remakehists",    default='True',                   help="regenerate shape histograms")
args = parser.parse_args()

class autodatacards:
    def __init__(self, args):
        self.year      = args.year
        self.shapefile = args.shapefile
        self.outdir    = args.outdir
        
        #enabling/disabling automcstats:
        self.usemcbinuncs = True
        
        #for generating histograms or not
        remakehists=False
        if args.remakehists=='True':
            remakehists=True
        self.remakehists = remakehists

        #for running all 3 years or just one 
        fullR2=False
        if args.year=='RunII':
            fullR2=True
        if not fullR2:
            years = [args.year]
        elif fullR2:
            years = ['2016', '2017', '2018']
        self.years = years

        #setup directories
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir); print('Making output directory ', self.outdir)

    #organizes set of bins for datacards and sets up card names and output directory
    def cardsetup(self, year, startbinnum):        
        #set up datacardrefs
        refs = datacardrefs(year)
        name = 'hwwcard' #name in your datacards
   
        # get name of output shape rootfile
        shaperoot = self.outdir+'/'+name+'_'+year+'_'+self.shapefile
        cardname = self.outdir+'/'+name

        #print info on the datacards you are making
        #this is also text to be put in each datacard in the set
        header='#----------------------------------------------#\n'+'#generating datacards. name: '+name+', year'+year+'\n'
        header+='\n'+'#number of cards in set: '+str(len(refs.binsels.keys()))+'\n'
        print(header)

        #generate shape histograms:
        hm = histmaker(year, shaperoot)
        if (not os.path.exists(self.shaperoot)) or self.remakehists:
            print('generating histograms...')
            histfile = r.TFile(shaperoot, "RECREATE")
            histfile.Close()
            del histfile
            yields = hm.makehists()

        shapeline = 'shapes * * '+name+'_'+year+'_'+self.shapefile+' $CHANNEL_$PROCESS $CHANNEL_$PROCESS_$SYSTEMATIC'
        print(shapeline)

        #make cards:
        binnum = startbinnum
        for chanid, binsel in refs.binsels.iteritems():
            label=chanid+'_'+year
            sel = '('+binsel+' &&  '+refs.mcsel
            makecard(self, cardname, label, year, sel, binnum, header, shapeline, refs, yields)
            binnum+=1
        return binnum

    #calculates rates and statistical uncertainties
    def getRate(self, procfile, cutstr, year, treename='Events'):
        print(procfile, treename)
        tree = r.TFile(procfile).Get(treename)
        htmp = r.TH1F('htmp','htmp',1,0.0,1.0)
        tree.Project('htmp','catBDTdisc_nosys',cutstr)
        errorVal = r.Double(0)
        rate = htmp.IntegralAndError(0,3,errorVal)
        rate = 0.0;rateunc = 1.0 
        if rate>0.0: #nonzero
            rateunc += (errorVal/rate)
        return round(rate,3), round(rateunc,3)
        
    def uncline(self, proc, uncval, uncname, unctype, binlabel, index, nprocs):
        rowname = binlabel+'_'+proc+'_'+uncname
        line = rowname+'      '+unctype
        row = ['-']*nprocs
        row[index] = str(uncval)
        for u in row:
            line+=('    '+u)
        line += '\n'
        return line

    def unctable(self, processes, unclist, uncname, unctype, binlabel):
        #unctype is lnN or shape...
        #unclist is the list of uncertainties for each process
        rownames = []
        for proc in processes:
            rowname = binlabel+'_'+proc+'_'+uncname
            rownames.append(rowname)
        # loop and create the table
        table = ''
        for i in range(len(processes)):
            row = ['-']*len(processes)
            row[i] = str(unclist[i])
            line = rownames[i]+'      '+unctype
            for u in row:
                line+=('    '+u)
            line += '\n'
            table+=line
            # print(table)
        return table

    #creates a single datacard
    def makecard(self, cardname, binlabel, year, cutstr, nbin, header, shapeline, refs, yields):
        #binlabel is cut#_bin#_year
        #cardfile is for example datacards/RTht5_cut1bin5_datacard.txt
        cardfile = cardname+'_'+binlabel+'_datacard'+str(nbin)+'.txt'
        procs = refs.processes
        nprocs = len(procs)

        imax = 1#number of final states analyzed (signals)
        jmax = nprocs-1 #number of processes(incl signal)-1
        kmax = '*' #* makes the code figure it out. number of independent systematical uncertainties 
        unctype  = 'lnN'

        print('\n'+binlabel)
        print('writing', cardfile,'selection:',cutstr)

        # get yields and statistical uncertainties for each process in this bin:
        rates=[]; stats=[]
        for proc in procs:
            r=0.0;u=0.0
            yld = yields[binlabel+'_'+proc]
            r=yld[0]; u=yld[1]
            rates.append(r); stats.append(1.0+(u/r))
            print(proc, r, u)
        obs  = int(round(sum(rates)))
        print('TOTAL B: ', round(sum(rates[1:])))

        # put together uncertanties for each process
        shapeuncs = [str(1.0)]*nprocs
        stat_table  = self.unctable(procs, stats, 'stat', 'lnN', binlabel)

        shape_table=''
        syst_table=''
        
        #norm uncertainty example:
        # lumi_line ='lumi_'+year+'   lnN    '+refs.lumiunc+'    '+refs.lumiunc+'    '+refs.lumiunc+'    -    \n'
        # syst_table+=lumi_line

        #shape uncertainties - leaving now to show correlation differences
        corr =[]
        for proc, systlist in refs.systs.iteritems():
            for sys in systlist: 
                if sys not in corr:
                    corr.append(sys)
        for sys in corr:
            if sys=='isr' or sys=='fsr' or sys=='ME' or sys=='pdf':
                shape_table += sys+'  shape     1.0    -    -    -  \n'  #need to add the right symbol for procs #correlated between years
            elif sys=='btagLF' or sys=='btagHF' or sys=='btagCFerr1' or sys=='btagCFerr2':
                shape_table += sys+'  shape     1.0    1.0    1.0    - \n'
            else: 
                shape_table += sys+'_'+year+'   shape     1.0   1.0    1.0    - \n'

        rateline = ''
        procline = ''
        numline = ''
        binuncline = ''
        if usemcbinuncs:
            thr = 0
            includesig = 0
            binuncline = binlabel+' autoMCStats '+str(thr)+' '+str(includesig)
            print(binuncline)

        #write datacard to file

        f = open(cardfile, "w")
        print('## card: '+cardfile+' number in set: '+str(nbin), file=f)
        print('##selection: '+cutstr+'*prefirewgt*\n', file=f)
        print(header, file=f)
        print('## number of signals (i), backgrounds (j), and uncertainties (k)', file=f)
        print("---", file=f)
        print("imax: "+str(imax), file=f)
        print("jmax: "+str(jmax), file=f)
        print("kmax: "+str(kmax), file=f)
        print("---\n", file=f)
        # if useshape:
        print('## input the bdt shape histograms', file=f)
        print("---", file=f)
        print(shapeline, file=f)
        print("---\n", file=f)
        print('## list the bin label and the number of (data) events observed', file=f)
        print("---", file=f)
        print("bin  "+binlabel, file=f)
        print("observation "+str(obs), file=f)
        print("---\n", file=f)
        print('## expected events for signal and all backgrounds in the bins', file=f)
        print("---", file=f)
        print("bin    "+(("    "+binlabel)*nprocs), file=f)
        for i,p in enumerate(procs):
            procline+=("    "+p)
            numline+=("        "+str(i))
        print("process"+procline, file=f)
        print("process"+numline, file=f)
        for r in rates:
            rateline+=("    "+str(r))
        print("rate   "+rateline, file=f)   
        print("---\n", file=f)
        # if not nounc:
        print('## list the independent sources of uncertainties, and give their effect (syst. error) on each process and bin', file=f)
        print("---", file=f)
        print('## statistical uncertainties\n', file=f)
        print(stat_table, file=f)
        # if not nounc:
        print("---", file=f)
        print('## systematic uncertainties\n', file=f)
        print(syst_table, file=f)
        # if useshape:
        print("---", file=f)
        print('## shape uncertainties\n', file=f)
        print(shape_table, file=f)
        print("---\n", file=f)
        print(binuncline, file=f)
        print("---\n", file=f)
    
        f.close()

####################################################################
# run cards

cards = autodatacards(self)
startbinnum=0
print('how many years?', len(cards.years))
for year in cards.years:
    print('year', year, 'startbinnum', startbinnum)
    lastbinnum = cards.cardsetup(self, year, startbinnum)
    print('lastbinnum', lastbinnum)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
    startbinnum=lastbinnum

