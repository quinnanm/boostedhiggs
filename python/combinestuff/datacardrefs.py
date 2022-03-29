# revamped datacard dictionary
# before running I recommend saving dataframes as rootfiles with systematics appended to variable names as _nosys _jerUp _jes_Down etc
import numpy as np
from collections import OrderedDict
import ROOT as r

class datacardrefs:
    def __init__(self, year):
        self.year = year
        
        #shape systematics included  
        systs = {'proc1':['syst1', 'syst2'], 
                 'proc2':['syst1', 'syst2'],
                 'proc3':['syst1', 'syst2']}
        self.systs=systs

        #treename for accessing events
        self.tn = 'Events'
        #variable you are using for histograms/templates:
        self.var = 'mass'

        #histogram settings
        self.edges = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95,  1.0], dtype=np.double)
        self.nbins = 10
        self.selrange = [0.0, 1.0]

        lumi = {'2016':'35.92',
                '2017':'41.53',
                '2018':'59.74'}
        self.lumi = lumi[year]

        
        self.mcprocs = ['proc1', 'proc2', 'proc3']
        self.processes = ['proc1', 'proc2', 'proc3']
        # self.DDprocname = 'DDBKG'
        # self.processes=['proc1', 'proc2', 'proc3', self.DDprocname]
        
        self.datafile = 'none'
        #leaving as example of rootfile location
        self.procfiles={'proc1' :'/eos/uscms/store/user/lpcstop/noreplica/mequinna/NanoAODv6_trees/'+year+'/MC/0lep/maintreeswithsysts/TTTT_'+year+'_merged_0lep_tree.root',
                        'proc2'  :'/eos/uscms/store/user/lpcstop/noreplica/mequinna/NanoAODv6_trees/'+year+'/MC/0lep/maintreeswithsysts/TTX_'+year+'_merged_0lep_tree.root',
                        'proc3':'/eos/uscms/store/user/lpcstop/noreplica/mequinna/NanoAODv6_trees/'+year+'/MC/0lep/maintreeswithsysts/minorMC_'+year+'_merged_0lep_tree.root'}
        
        metfilters   = ' && (goodverticesflag && haloflag && HBHEflag && HBHEisoflag && ecaldeadcellflag && badmuonflag)'
        if year=='2016':
            metfilters =  ' && (goodverticesflag && haloflag && HBHEflag && HBHEisoflag && ecaldeadcellflag && badmuonflag && eeBadScFilterflag)'
        self.metfilters = metfilters

        self.binsels =self.getbinsels('nosys')
        self.mcsel = self.getsels('nosys',True)
        self.datasel = self.getsels('nosys',False)

    #leaving as example
    def getsels(self, systyp='nosys', isMC=True):
        wgtstr = '*weight*btagSF_nosys*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi
        #preselection:
        basesel = ' nfunky_leptons==0 && ht_nosys>=700 && nbjets_nosys>=3 && njets_nosys>=9'
        binsels = self.getbinsels('nosys')

        systsels = {'pileup'   : basesel+self.metfilters+')*weight*btagSF_nosys*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*puWeight',
                    'trigger' : basesel+self.metfilters+')*weight*btagSF_nosys*puWeight*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*trigSF_',
                    'btagHF'   : basesel+self.metfilters+')*weight*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*btagSF_btagHF',
                    'btagLF'   : basesel+self.metfilters+')*weight*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*btagSF_btagLF',
                    'isr'      : basesel+self.metfilters+')*weight*btagSF_nosys*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*isr_',
                    'fsr'      : basesel+self.metfilters+')*weight*btagSF_nosys*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*fsr_',
                    'ME'      : basesel+self.metfilters+')*weight*btagSF_nosys*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*MErn_',
                    'pdf'      : basesel+self.metfilters+')*weight*btagSF_nosys*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*pdfrn_',
                    'btagHFstats1'   : basesel+self.metfilters+')*weight*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*btagSF_btagHFstats1',
                    'btagLFstats1'   : basesel+self.metfilters+')*weight*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*btagSF_btagLFstats1',
                    'btagHFstats2'   : basesel+self.metfilters+')*weight*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*btagSF_btagHFstats2',
                    'btagLFstats2'   : basesel+self.metfilters+')*weight*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*btagSF_btagLFstats2',
                    'btagCFerr1'   : basesel+self.metfilters+')*weight*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*btagSF_btagCFerr1',
                    'btagCFerr2'   : basesel+self.metfilters+')*weight*puWeight*trigSF_nosys*bsWSF_nosys*bsTopSF_nosys*'+self.lumi+'*btagSF_btagCFerr2',
                    'DeepAK8TopSF' : basesel+self.metfilters+')*weight*btagSF_nosys*puWeight*trigSF_nosys*bsWSF_nosys*'+self.lumi+'*bsTopSF_DeepAK8TopSF_',
                    'DeepAK8WSF' : basesel+self.metfilters+')*weight*btagSF_nosys*puWeight*trigSF_nosys*bsTopSF_nosys*'+self.lumi+'*bsWSF_DeepAK8WSF_',
        } 
        

        if systyp =='nosys':
            if not isMC:
                sel = basesel+self.metfilters+')'#trigcorr
                return sel
            elif isMC:
                sel = basesel+self.metfilters+')'+wgtstr
                return sel

        elif systyp not in ['jer','jes']:
            systselup = systsels[systyp]+'Up'
            systseldown = systsels[systyp]+'Down'
            return [systselup, systseldown], [binsels, binsels]

        elif systyp in ['jer','jes']:
            sel= []; binsels=[]
            updown=['Up','Down']
            for ud in updown:
                syst=systyp+ud
                wgtstr = '*weight*btagSF_'+syst+'*puWeight*trigSF_'+syst+'*bsWSF_'+syst+'*bsTopSF_'+syst+'*'+self.lumi
                basesel = ' nfunky_leptons==0 && ht_'+syst+'>=700 && nbjets_'+syst+'>=3 && njets_'+syst+'>=9'
                binsel = self.getbinsels(syst)
                systsel = basesel+self.metfilters+')'+wgtstr
                sel.append(systsel)
                binsels.append(binsel)
            return sel, binsels

    # leaving for now as example of how SR categories are set up
    def getbinsels(self, syst):
        binsels = {'RT1BT0htbin0':'nrestops_'+syst+'==1 && nbstops_'+syst+'==0 && ht_'+syst+'>=700 && ht_'+syst+'<800',
                   'RT1BT0htbin1':'nrestops_'+syst+'==1 && nbstops_'+syst+'==0 && ht_'+syst+'>=800 && ht_'+syst+'<900', 
                   'RT1BT0htbin2':'nrestops_'+syst+'==1 && nbstops_'+syst+'==0 && ht_'+syst+'>=900 && ht_'+syst+'<1000', 
                   'RT1BT0htbin3':'nrestops_'+syst+'==1 && nbstops_'+syst+'==0 && ht_'+syst+'>=1000 && ht_'+syst+'<1100',
                   'RT1BT0htbin4':'nrestops_'+syst+'==1 && nbstops_'+syst+'==0 && ht_'+syst+'>=1100 && ht_'+syst+'<1200',
                   'RT1BT0htbin5':'nrestops_'+syst+'==1 && nbstops_'+syst+'==0 && ht_'+syst+'>=1200 && ht_'+syst+'<1300',
                   'RT1BT0htbin6':'nrestops_'+syst+'==1 && nbstops_'+syst+'==0 && ht_'+syst+'>=1300 && ht_'+syst+'<1500', 
                   'RT1BT0htbin7':'nrestops_'+syst+'==1 && nbstops_'+syst+'==0 && ht_'+syst+'>=1500',
                   'RT1BT1htbin0':'nrestops_'+syst+'==1 && nbstops_'+syst+'>=1 && ht_'+syst+'<1500',
                   'RT1BT1htbin1':'nrestops_'+syst+'==1 && nbstops_'+syst+'>=1 && ht_'+syst+'>=1500',
                   'RT2BTALLhtbin0':'nrestops_'+syst+'>=2 && nbstops_'+syst+'>=0 && ht_'+syst+'<1200',
                   'RT2BTALLhtbin1':'nrestops_'+syst+'>=2 && nbstops_'+syst+'>=0 && ht_'+syst+'>=1200'}
        return binsels
    

    def getyield(self, hist, verbose=False):
        errorVal = r.Double(0)
        minbin=0
        maxbin=hist.GetNbinsX()+1
        hyield = hist.IntegralAndError(minbin, maxbin, errorVal)
        if verbose:
            print('yield:', round(hyield, 3), '+/-', round(errorVal, 3), '\n')
        return hyield,  errorVal
    
    def fixref(self, rootfile, hist):
        hist.SetDirectory(0)
        rootfile.Close()

    def makeroot(self, infile, treename, options="read"):
        rfile = r.TFile(infile, options)
        tree = rfile.Get(treename)
        return rfile, tree
