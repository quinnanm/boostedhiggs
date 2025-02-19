#based on: https://github.com/jennetd/hbb-coffea/blob/stocknano/fits-combine8/hbb-stxs/allyears/stxs.py
#https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/main/data/tutorials/tutorial_unfolding_2023/scripts/make_XSplot.py
import os
import subprocess
import re
import json
import ROOT as rt
import numpy as np
rt.gStyle.SetOptStat(0)

#POILIST to be used:
POIList = {'ggF': ['r_ggH_pt200_300', 'r_ggH_pt300_450', 'r_ggH_pt450_inf'],
           'VBF': ['r_qqH_hww_mjj_1000_Inf']}

class UnfoldingPlot:

    def __init__(self, combinecards='templates/v13/datacards_unfolding', poilist=POIList):

        self.cards_dir = combinecards
        self.multidimresults = "multidimresults.txt"
        self.poilist = poilist
        self.xsecfile = "./xsecs.json"
        
        #run the fit if needed
        print('combine cards directory: '+combinecards)
        print('getting multidimfit results...')
        self.runfit(rerunfit=True)
        self.poivals = self.parsemultidimresult(self.poilist)

        #get the SM cross sections and calculate unfolded differential ones
        #add code to generate json file
        print('getting cross section values...')
        self.makesmxsecjson(rerun=False)
        self.xsecvals, self.smvals = self.getxsecs(self.poivals, self.xsecfile)


    #runmultidimfit
    def runfit(self, rerunfit=True):
        # Define the path to the shell script
        rununfoldingplot = "./rununfoldingplot.sh"

        # Check if the text file exists in the same directory
        if not os.path.isfile(self.multidimresults) or rerunfit:
            print("Running the shell script ./rununfoldingplot.sh...")
    
            # Run the shell script with the cards dir as an argument
            try:
                subprocess.run([rununfoldingplot, '--cardsdir', self.cards_dir], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing shell script: {e}")
        else:
            print(f"{self.multidimresults} found. No need to run the shell script.")

            
    #get processes and values from multidimresult
    def parsemultidimresult(self, poilist, verbose=False):

        POIvals = {}
        
        with open(self.multidimresults, 'r') as file:
            lines = file.readlines()

        # pattern in multidimfit  r_ggH_pt200_300 :    +1.000   -1.000/+4.796 (68%)
        pattern = re.compile(r"(?P<param>\w+) :\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)/([+-]?\d+\.\d+)")

        # Iterate over each line and look for the parameters defined in POIList
        for line in lines:
            match = pattern.search(line)
            if match:
                param = match.group('param')  # Extract the parameter name
                best_fit = float(match.group(2))  # Extract the best fit value
                lower_err = float(match.group(3))  # Extract the lower uncertainty
                upper_err = float(match.group(4))  # Extract the upper uncertainty

                # Check if this parameter is in POIList
                for key, params in POIList.items():
                    if param in params:
                        POIvals[param] = [best_fit, lower_err, upper_err]
                        if verbose:
                            print(f"stored fit results {param}: {POIvals[param]}")
        
        return POIvals

    
    #calculates xsecs from smxsecs in json file
    def getxsecs(self, poivals, xsecfile, verbose=True):

        XSvals = {}
        SMvals = {}
        
        print(self.xsecfile)
        with open(xsecfile) as f:
            smxs = json.load(f)
        
        for key, val in poivals.items():
            if verbose:
                print(f"proc: {key}, poival: {val}, smxs: {smxs['nom'][key]} ")
            nomxs = val[0]*smxs["nom"][key] #poi r value * sm xsec
            downxs = val[1]*smxs["nom"][key]
            upxs = val[2]*smxs["nom"][key]
            
            XSvals[key] = [nomxs, downxs, upxs]
            SMvals[key] = [smxs["nom"][key], smxs["down"][key], smxs["up"][key]]
            if verbose:
                print(f"calculated xsecs {key}: {XSvals[key]}")
                print(f"saved sm xsecs {key}: {SMvals[key]}")

        return XSvals, SMvals


    def makesmxsecjson(self, rerun=False, verbose=True):
        #make sure to print cards dir to json file to check that it matches and autoreruns if not
        if not os.path.isfile(self.multidimresults) or rerun:
            print(f"creating xsecs json file {self.xsecsfile} from datacards {self.cards_dir}...")
        
    def makeplot(self, showplot=False, verbose=True):

        c = rt.TCanvas("stxs", "stxs", 800, 600)
        pad1 = rt.TPad("pad1","pad1",0,.4,0.6,1);
        pad2 = rt.TPad("pad2","pad2",0,0,0.6,.4);
        pad3 = rt.TPad("pad3","pad3",0.6,.4,1,1);
        pad4 = rt.TPad("pad4","pad4",0.6,0,1,.4);

        pad1.SetBottomMargin(0.05)
        pad1.SetTopMargin(0.1)
        pad1.SetBorderMode(0)
        pad2.SetTopMargin(0.00001)
        pad2.SetBottomMargin(0.3)
        pad2.SetBorderMode(0)
        pad3.SetBottomMargin(0.05)
        pad3.SetTopMargin(0.1)
        pad3.SetBorderMode(0)
        pad4.SetTopMargin(0.00001)
        pad4.SetBottomMargin(0.3)
        pad4.SetBorderMode(0)

        pad1.SetLeftMargin(0.15)
        pad2.SetLeftMargin(0.15)
        pad3.SetRightMargin(0.15)
        pad4.SetRightMargin(0.15)
        pad1.SetRightMargin(0.02)
        pad2.SetRightMargin(0.02)
        pad3.SetLeftMargin(0.02)
        pad4.SetLeftMargin(0.02)
        pad1.Draw()
        pad2.Draw()
        pad3.Draw()
        pad4.Draw()

        textsize1 = 19/(pad1.GetWh()*pad1.GetAbsHNDC());
        textsize2 = 1.5*textsize1
        textsize3 = textsize1
        textsize4 = textsize2
        
        pad1.cd()
        pad1.SetLogy()

        lumi = 138
        tag1 = rt.TLatex(0.46, 0.92, "%.0f fb^{-1} (13 TeV)" % lumi)
        tag1.SetNDC()
        tag1.SetTextFont(42)
        tag2 = rt.TLatex(0.17, 0.92, "CMS")
        tag2.SetNDC()
        tag2.SetTextFont(62)
        tag3 = rt.TLatex(0.26, 0.92, "H(WW)")
        tag3.SetNDC()
        tag3.SetTextFont(42)

        tag4 = rt.TLatex(0.2, 0.82, "ggF")
        tag4.SetNDC()
        tag4.SetTextFont(42)
        tag5 = rt.TLatex(0.08, 0.82, "VBF")
        tag5.SetNDC()
        tag5.SetTextFont(42)

        h1 = rt.TH1D("dummy1","",3,-0.5,2.5)
        binnames = ['[200,300]', '[300,450]', '[450,#infty)']
        for b in binnames:
            h1.Fill(b,0)
        
        h1.SetLineColor(0)
        h1.SetLineWidth(3)
        h1.GetXaxis().SetTitleSize(0)
        h1.GetXaxis().SetLabelSize(0)
        h1.GetYaxis().SetTitle('#sigma_{obs} [fb]')
        h1.GetYaxis().SetTitleSize(textsize1)
        h1.GetYaxis().SetLabelSize(textsize1)
        h1.GetYaxis().SetTitleOffset(2*pad1.GetAbsHNDC())
        h1.GetYaxis().SetRangeUser(0.01, 10000)
        h1.GetXaxis().SetTitle("p_{T}^{H} [GeV]")
        h1.GetXaxis().CenterTitle(True)
        h1.GetXaxis().SetTitleOffset(2.5*pad1.GetAbsHNDC())
        h1.Draw()

        # Fill SM xsec+uncertainties
        ggf_smcenter = []
        ggf_smup = []
        ggf_smdo = []

        print(self.smvals)
        for key, smval in self.smvals.items():
            if 'ggH' in key:
                ggf_smcenter += [smval[0]]
                ggf_smdo += [smval[0]*smval[1]] #nom*down
                ggf_smup += [smval[0]*smval[2]] #nom*up

        x1 = np.linspace(0,3,4)
        w1 = np.array([0.5 for i in x1])

        g1 = rt.TGraphAsymmErrors(3,x1,np.array(ggf_smcenter),w1,w1,np.array(ggf_smdo),np.array(ggf_smup))
        g1.SetFillColor(4)
        g1.SetFillStyle(3003)
        g1.SetLineColor(4)
        g1.SetLineWidth(1)
        g1.Draw("2same")
        g1.Draw("pesame")

        #fill calculated unfolded xsecs
        ggf_center = []
        ggf_up = []
        ggf_do = []

        for key, xsval in self.xsecvals.items():
            if 'ggH' in key:
                ggf_center += [xsval[0]]
                ggf_do += [-1.0*xsval[1]]
                ggf_up += [xsval[2]]

        x = np.linspace(0,3,4)
        w = np.zeros(len(x))
        print(ggf_center)
        gggf = rt.TGraphAsymmErrors(3, #points (1 for ggf)
                                    np.array(x, dtype='float64'), #x values
                                    np.array(ggf_center, dtype='float64'), #y values
                                    np.array(w, dtype='float64'), #x up error=0
                                    np.array(w, dtype='float64'), #x down error=0
                                    np.array(ggf_do, dtype='float64'),#y down errors 
                                    np.array(ggf_up, dtype='float64') #y up errors 
                                    )
        
        gggf.SetMarkerColor(1)
        gggf.SetMarkerStyle(20)
        gggf.SetLineColor(1)
        gggf.SetLineWidth(3)
        gggf.Draw("pesame")
        
        tag2.SetTextSize(textsize1)
        tag2.Draw()
        tag3.SetTextSize(textsize1)
        tag3.Draw()
        tag4.SetTextSize(textsize1)
        tag4.Draw()

        #vbf pad
        pad3.cd()
        pad3.SetLogy()

        h2 = rt.TH1D("dummy2","",2,2.5,4.5)

        binnames = ['[1000,#infty)']
        for b in binnames:
            h2.Fill(b,0)

        h2.SetLineColor(0)
        h2.SetLineWidth(3)
        h2.GetXaxis().SetTitleSize(0)
        h2.GetXaxis().SetLabelSize(0)
        h2.GetXaxis().SetTitle("m_{jj}^{gen} [GeV]")
        h2.GetXaxis().CenterTitle(True)
        h2.GetXaxis().SetTitleOffset(2.5*pad3.GetAbsHNDC())
        h2.GetYaxis().SetRangeUser(0.01, 10000)
        h2.GetYaxis().SetTitleSize(0)
        h2.GetYaxis().SetLabelSize(0)
        h2.Draw()

        #fill SM vbf uncertainties
        vbf_smcenter = []
        vbf_smup = []
        vbf_smdo = []
        for key, smval in self.smvals.items():
            if 'qqH' in key:
                vbf_smcenter += [smval[0]]
                vbf_smdo += [smval[0]*smval[1]] #nom*down
                vbf_smup += [smval[0]*smval[2]] #nom*up

        x2 = np.array([3])
        w2 = np.array([0.5])

        print(vbf_smcenter)
        g2 = rt.TGraphAsymmErrors(1, #points (1 for vbf)
                                  np.array(x2, dtype='float64'), #x values
                                  np.array(vbf_smcenter, dtype='float64'), #y values
                                  np.array(w2, dtype='float64'), #x up error=0
                                  np.array(w2, dtype='float64'), #x down error=0
                                  np.array(vbf_smdo, dtype='float64'),#y down errors 
                                  np.array(vbf_smup, dtype='float64') #y up errors 
                                  )

        g2.SetFillColor(94)
        g2.SetFillStyle(3003)
        g2.SetLineColor(94)
        g2.SetMarkerColor(94)
        g2.SetLineWidth(1)
        g2.Draw("2same")
        g2.Draw("pesame")

        #fill calculated unfolded xsecs
        vbf_center = []
        vbf_up = []
        vbf_do = []
        for key, xsval in self.xsecvals.items():
            if 'qqH' in key:
                vbf_center += [xsval[0]]
                vbf_do += [-1.0*xsval[1]]
                vbf_up += [xsval[2]]

        x = np.linspace(3,4,1)
        w = np.zeros(len(x))
        gvbf = rt.TGraphAsymmErrors(1, #points (1 for vbf)
                                    np.array(x, dtype='float64'), #x values
                                    np.array(vbf_center, dtype='float64'), #y values
                                    np.array(w, dtype='float64'), #x up error=0
                                    np.array(w, dtype='float64'), #x down error=0
                                    np.array(vbf_do, dtype='float64'),#y down errors 
                                    np.array(vbf_up, dtype='float64') #y up errors 
                                    )

        gvbf.SetMarkerColor(1)
        gvbf.SetMarkerStyle(20)
        gvbf.SetLineColor(1)
        gvbf.SetLineWidth(3)
        gvbf.Draw("psame")

        tag1.SetTextSize(textsize3)
        tag1.Draw()
        tag5.SetTextSize(textsize3)
        tag5.Draw()

        pad1.cd()

        leg = rt.TLegend(0.44, 0.65, 0.82, 0.87)
        leg.SetBorderSize(0)
        #    leg.SetTextFont(42)
        leg.SetTextSize(textsize3)
        leg.SetFillColor(rt.kWhite)
        leg.SetLineColor(rt.kWhite)
        leg.SetLineStyle(0)
        leg.SetFillStyle(0)
        leg.SetLineWidth(0)

        histo = h2.Clone("histo")
        histo.SetMarkerColor(1)
        histo.SetLineColor(1)
        histo.SetLineWidth(3)
        histo.SetMarkerStyle(20)
        
        leg.AddEntry(histo,"Observed (stat #oplus syst)","pe")
        leg.AddEntry(g1,"ggF (HJMINLO)","f")
        leg.AddEntry(g2,"VBF (POWHEG+HC)","f")

        leg.Draw("same")

        pad2.cd()

        h3 = h1.Clone("dummy3")
        h3.Reset()
        h3.GetYaxis().SetTitle("#sigma_{obs} / #sigma_{SM}")
        h3.GetYaxis().SetTitleOffset(2*pad2.GetAbsHNDC())
        h3.GetYaxis().SetRangeUser(-5,16)
        h3.GetXaxis().SetTitleSize(textsize2)
        h3.GetXaxis().SetLabelSize(1.3*textsize2)
        h3.GetYaxis().SetTitleSize(textsize2)
        h3.GetYaxis().SetLabelSize(textsize2)
        h3.Draw()

        #ratio sm xsecs ggf AND vbf
        ggf_smuperrs = []
        ggf_smdoerrs = []
        vbf_smdoerrs = []
        vbf_smuperrs = []
        for key, smval in self.smvals.items():
            if 'ggH' in key:
                ggf_smdoerrs += [smval[1]] #down
                ggf_smuperrs += [smval[2]] #up
            elif 'qqH' in key:
                vbf_smdoerrs += [smval[1]] #down
                vbf_smuperrs += [smval[2]] #up

        g3 = rt.TGraphAsymmErrors(3,x1,np.ones(3),w1,w1,np.array(ggf_smdoerrs),np.array(ggf_smuperrs))
        g3.SetFillColor(4)
        g3.SetFillStyle(3003)
        g3.SetLineColor(4)
        g3.SetLineWidth(1)
        g3.Draw("2same")
        g3.Draw("pesame")

        #ratio plot ggf and vbf
        vbf_rcenter = []
        vbf_rup = []
        vbf_rdo = []        
        ggf_rcenter = []
        ggf_rup = []
        ggf_rdo = []        
        #fill calculated unfolded xsecs
        for key, poival in self.poivals.items():
            if 'ggH' in key:
                ggf_rcenter += [poival[0]]
                ggf_rdo += [-1.0*poival[1]]
                ggf_rup += [poival[2]]
            elif 'qqH' in key:
                vbf_rcenter += [poival[0]]
                vbf_rdo += [-1.0*poival[1]]
                vbf_rup += [poival[2]]

        print('vbf vals')
        print(vbf_rcenter)
        print(vbf_rdo)
        print(vbf_rup)
        x = np.linspace(0,3,4)
        w = np.zeros(len(x))
        print(poival[0])
        grat = rt.TGraphAsymmErrors(3, #points (1 for ggf)
                                    np.array(x, dtype='float64'), #x values
                                    np.array(ggf_rcenter, dtype='float64'), #y values
                                    np.array(w, dtype='float64'), #x up error=0
                                    np.array(w, dtype='float64'), #x down error=0
                                    np.array(ggf_rdo, dtype='float64'),#y down errors 
                                    np.array(ggf_rup, dtype='float64') #y up errors 
                                    )
        
        grat.SetMarkerColor(1)
        grat.SetMarkerStyle(20)
        grat.SetLineColor(1)
        grat.SetLineWidth(3)
        grat.Draw("pesame")

        pad4.cd()

        h4 = h2.Clone("dummy4")
        h4.Reset()
        h4.GetYaxis().SetTitleSize(0)
        h4.GetYaxis().SetLabelSize(0)
        h4.GetYaxis().SetRangeUser(-5,16)
        h4.GetXaxis().SetTitleSize(textsize2)
        h4.GetXaxis().SetLabelSize(1.3*textsize2)
        h4.Draw()

        print('vbf_smdoerrs')
        x =  np.linspace(3,4,1)
        print(vbf_smdoerrs)
        g4  = rt.TGraphAsymmErrors(1, #points (1 for vbf)
                                   np.array(x2, dtype='float64'), #x values
                                   np.ones(1, dtype='float64'), #y values
                                   # np.array(x, dtype='float64'), #y values #wrong, sets default to 3
                                   np.array(w2, dtype='float64'), #x up error=0 
                                   np.array(w2, dtype='float64'), #x down error=0
                                   np.array(vbf_smdoerrs, dtype='float64'),#y down errors 
                                   np.array(vbf_smuperrs, dtype='float64') #y up errors 
                                   )

        g4.SetFillColor(94)
        g4.SetFillStyle(3003)
        g4.SetMarkerColor(94)
        g4.SetLineColor(94)
        g4.SetLineWidth(1)
        g4.Draw("2same")
        g4.Draw("pesame")

        grat2 =  rt.TGraphAsymmErrors(1, #points (1 for vbf)
                                      np.array(x, dtype='float64'), #x values
                                      np.array(vbf_rcenter, dtype='float64'), #y values
                                      np.array(w, dtype='float64'), #x up error=0
                                      np.array(w, dtype='float64'), #x down error=0
                                      np.array(vbf_rdo, dtype='float64'),#y down errors 
                                      np.array(vbf_rup, dtype='float64') #y up errors 
                                      )
        grat2.SetMarkerColor(1)
        grat2.SetMarkerStyle(20)
        grat2.SetLineColor(1)
        grat2.SetLineWidth(3)
        grat2.Draw("psame")
        
        c.Print("stxs.pdf")
        c.Print("stxs.png")
        c.Print("stxs.C")

        if showplot:
            c.Draw()

        
        

        
            
if __name__ == "__main__":

    up = UnfoldingPlot()
    # my_instance.greet()

    #generate the plot
    print('creating the plot...')
    up.makeplot()
