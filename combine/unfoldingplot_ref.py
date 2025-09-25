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

    def __init__(self, combinecards='templates/v17/datacards_unfolding', poilist=POIList):

        self.cards_dir = combinecards
        self.multidimresults = "multidimresults.txt"
        self.poilist = poilist
        self.xsecfile = "./xsecs_new.json"
        
        #run the fit if needed
        print('combine cards directory: '+combinecards)
        print('getting multidimfit results...')
        self.runfit(rerunfit=False)
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

    def makeplot(self, showplot=False, verbose=True,
                top_logy=False, top_yrange=None, ratio_yrange=None):
        import ROOT as rt
        import numpy as np

        # --- bin keys / x positions ---
        ggf_keys = ['r_ggH_pt200_300', 'r_ggH_pt300_450', 'r_ggH_pt450_inf']
        vbf_keys = ['r_qqH_hww_mjj_1000_Inf']

        x1  = np.array([0.0, 1.0, 2.0], dtype='float64')  # ggF bins
        ex1 = np.array([0.5, 0.5, 0.5], dtype='float64')
        x2  = np.array([3.0], dtype='float64')            # one VBF bin
        ex2 = np.array([0.5], dtype='float64')
        z3  = np.zeros(3, dtype='float64')
        z1  = np.zeros(1, dtype='float64')

        # ========== Canvas & pads ==========
        c = rt.TCanvas("stxs", "stxs", 800, 600)
        pad1 = rt.TPad("pad1","pad1",0,.4,0.6,1); pad1.SetBottomMargin(0.05); pad1.SetTopMargin(0.1);  pad1.SetBorderMode(0)
        pad2 = rt.TPad("pad2","pad2",0,0,0.6,.4); pad2.SetTopMargin(0.00001); pad2.SetBottomMargin(0.3); pad2.SetBorderMode(0)
        pad3 = rt.TPad("pad3","pad3",0.6,.4,1,1); pad3.SetBottomMargin(0.05); pad3.SetTopMargin(0.1);  pad3.SetBorderMode(0)
        pad4 = rt.TPad("pad4","pad4",0.6,0,1,.4); pad4.SetTopMargin(0.00001); pad4.SetBottomMargin(0.3); pad4.SetBorderMode(0)

        pad1.SetLeftMargin(0.15); pad2.SetLeftMargin(0.15)
        pad3.SetRightMargin(0.15); pad4.SetRightMargin(0.15)
        pad1.SetRightMargin(0.02); pad2.SetRightMargin(0.02)
        pad3.SetLeftMargin(0.02);  pad4.SetLeftMargin(0.02)

        for p in (pad1,pad2,pad3,pad4): p.Draw()

        # optional log y on the top pads
        if top_logy:
            pad1.SetLogy()
            pad3.SetLogy()

        textsize1 = 19/(pad1.GetWh()*pad1.GetAbsHNDC())
        textsize2 = 1.5*textsize1
        textsize3 = textsize1

        lumi = 138
        tag1 = rt.TLatex(0.46, 0.92, "%.0f fb^{-1} (13 TeV)" % lumi); tag1.SetNDC(); tag1.SetTextFont(42)
        tag2 = rt.TLatex(0.20, 0.82, "CMS"); tag2.SetNDC(); tag2.SetTextFont(62)
        # tag2 = rt.TLatex(0.17, 0.92, "CMS"); tag2.SetNDC(); tag2.SetTextFont(62)
        #tag3 = rt.TLatex(0.26, 0.92, "H(WW)");                         tag3.SetNDC(); tag3.SetTextFont(42)
        tag4 = rt.TLatex(0.20, 0.75, "ggF"); tag4.SetNDC(); tag4.SetTextFont(42)
        tag5 = rt.TLatex(0.08, 0.75, "VBF"); tag5.SetNDC(); tag5.SetTextFont(42)

        def set_bin_labels(h, labels):
            ax = h.GetXaxis()
            for i, lab in enumerate(labels, 1):
                ax.SetBinLabel(i, lab)

        # ========== COLLECT: SM (absolute) ==========
        ggf_sm_y, ggf_sm_eyl, ggf_sm_eyh = [], [], []
        for k in ggf_keys:
            sm, dfac, ufac = self.smvals[k]
            ggf_sm_y.append(sm)
            ggf_sm_eyl.append(sm * max(0.0, 1.0 - dfac))
            ggf_sm_eyh.append(sm * max(0.0, ufac - 1.0))

        vbf_sm_y, vbf_sm_eyl, vbf_sm_eyh = [], [], []
        for k in vbf_keys:
            sm, dfac, ufac = self.smvals[k]
            vbf_sm_y.append(sm)
            vbf_sm_eyl.append(sm * max(0.0, 1.0 - dfac))
            vbf_sm_eyh.append(sm * max(0.0, ufac - 1.0))

        # ========== COLLECT: Observed σ (absolute) ==========
        # Convention: self.xsecvals[k] = [σ_obs, -Δσ_down, +Δσ_up]
        ggf_y, ggf_eyl, ggf_eyh = [], [], []
        for k in ggf_keys:
            nom, d_err, u_err = self.xsecvals[k]
            ggf_y.append(nom); ggf_eyl.append(abs(d_err)); ggf_eyh.append(abs(u_err))

        vbf_y, vbf_eyl, vbf_eyh = [], [], []
        for k in vbf_keys:
            nom, d_err, u_err = self.xsecvals[k]
            vbf_y.append(nom); vbf_eyl.append(abs(d_err)); vbf_eyh.append(abs(u_err))

        # ========== TOP pads (absolute σ) ==========
        # Original behavior: fixed ymax=0.5 and compute ymin from content.
        def y_min_with_margin(values, eyl, eyh, ymax=0.5, margin=0.10):
            lows = [v - lo for v, lo in zip(values, eyl)]
            lo = min(lows)
            span = (ymax - lo) if ymax > lo else 1.0
            return lo - margin*span, ymax

        all_vals = ggf_sm_y + ggf_y + vbf_sm_y + vbf_y
        all_eyl  = ggf_sm_eyl + ggf_eyl + vbf_sm_eyl + vbf_eyl
        all_eyh  = ggf_sm_eyh + ggf_eyh + vbf_sm_eyh + vbf_eyh

        if top_yrange is None:
            ylo, yhi = y_min_with_margin(all_vals, all_eyl, all_eyh, ymax=0.5, margin=0.10)
        else:
            ylo, yhi = top_yrange

        # if log scale requested but ymin ≤ 0, lift it slightly
        if top_logy and ylo <= 0:
            ylo = max(1e-6, ylo + 1e-6)

        # ggF axis (left)
        pad1.cd()
        h1 = rt.TH1D("h_abs_ggf", "", 3, -0.5, 2.5)
        set_bin_labels(h1, ['[200,300]', '[300,450]', '[450,#infty)'])
        h1.SetLineColor(0); h1.SetLineWidth(3)
        h1.GetXaxis().SetTitleSize(0); h1.GetXaxis().SetLabelSize(0)
        h1.GetYaxis().SetTitle('#sigma_{obs} [fb]')
        h1.GetYaxis().SetTitleSize(textsize1); h1.GetYaxis().SetLabelSize(textsize1)
        h1.GetYaxis().SetTitleOffset(2*pad1.GetAbsHNDC())
        h1.GetXaxis().SetTitle("p_{T}^{H} [GeV]"); h1.GetXaxis().CenterTitle(True)
        h1.GetXaxis().SetTitleOffset(2.5*pad1.GetAbsHNDC())
        h1.GetYaxis().SetRangeUser(ylo, yhi)
        h1.Draw("axis")

        # VBF axis (right), no y labels
        pad3.cd()
        h2 = rt.TH1D("h_abs_vbf", "", 1, 2.5, 3.5)
        set_bin_labels(h2, ['[1000,#infty)'])
        h2.SetLineColor(0); h2.SetLineWidth(3)
        h2.GetXaxis().SetTitleSize(0); h2.GetXaxis().SetLabelSize(0)
        h2.GetXaxis().SetTitle("m_{jj}^{gen} [GeV]"); h2.GetXaxis().CenterTitle(True)
        h2.GetXaxis().SetTitleOffset(2.5*pad3.GetAbsHNDC())
        h2.GetYaxis().SetTitleSize(0); h2.GetYaxis().SetLabelSize(0)
        h2.GetYaxis().SetRangeUser(ylo, yhi)
        h2.Draw("axis")

        # SM bands + observed points
        pad1.cd()
        g1 = rt.TGraphAsymmErrors(3, x1, np.array(ggf_sm_y,'f8'), ex1, ex1,
                                np.array(ggf_sm_eyl,'f8'), np.array(ggf_sm_eyh,'f8'))
        g1.SetFillColor(4); g1.SetFillStyle(3003); g1.SetLineColor(4); g1.SetLineWidth(1)
        g1.Draw("2 same"); g1.Draw("pe same")

        gggf = rt.TGraphAsymmErrors(3, x1, np.array(ggf_y,'f8'), z3, z3,
                                    np.array(ggf_eyl,'f8'), np.array(ggf_eyh,'f8'))
        gggf.SetMarkerColor(1); gggf.SetMarkerStyle(20); gggf.SetLineColor(1); gggf.SetLineWidth(3)
        gggf.Draw("pe same")

        tag2.SetTextSize(textsize2); tag2.Draw()
        # tag3.SetTextSize(textsize1); tag3.Draw()
        tag4.SetTextSize(textsize1); tag4.Draw()

        pad3.cd()
        g2 = rt.TGraphAsymmErrors(1, x2, np.array(vbf_sm_y,'f8'), ex2, ex2,
                                np.array(vbf_sm_eyl,'f8'), np.array(vbf_sm_eyh,'f8'))
        g2.SetFillColor(94); g2.SetFillStyle(3003); g2.SetLineColor(94); g2.SetMarkerColor(94); g2.SetLineWidth(1)
        g2.Draw("2 same"); g2.Draw("pe same")

        gvbf = rt.TGraphAsymmErrors(1, x2, np.array(vbf_y,'f8'), z1, z1,
                                    np.array(vbf_eyl,'f8'), np.array(vbf_eyh,'f8'))
        gvbf.SetMarkerColor(1); gvbf.SetMarkerStyle(20); gvbf.SetLineColor(1); gvbf.SetLineWidth(3)
        gvbf.Draw("pe same")

        tag1.SetTextSize(textsize3); tag1.Draw()
        tag5.SetTextSize(textsize3); tag5.Draw()

        # legend
        pad1.cd()
        leg = rt.TLegend(0.44, 0.65, 0.82, 0.87)
        leg.SetBorderSize(0); leg.SetTextSize(textsize3); leg.SetFillColor(rt.kWhite)
        leg.SetLineColor(rt.kWhite); leg.SetLineStyle(0); leg.SetFillStyle(0); leg.SetLineWidth(0)
        dummy = h2.Clone("histo"); dummy.SetMarkerColor(1); dummy.SetLineColor(1); dummy.SetLineWidth(3); dummy.SetMarkerStyle(20)
        leg.AddEntry(dummy,"Observed (stat #oplus syst)","pe")
        leg.AddEntry(g1,"ggF (HJMINLO)","f")
        leg.AddEntry(g2,"VBF (POWHEG+HC)","f")
        leg.Draw("same")

        # ========== BOTTOM pads (ratios) ==========
        # SM ratio bands (center 1)
        ggf_ratio_y   = np.ones(3, dtype='float64')
        ggf_ratio_eyl = np.array([max(0.0, 1.0 - self.smvals[k][1]) for k in ggf_keys], dtype='float64')
        ggf_ratio_eyh = np.array([max(0.0, self.smvals[k][2] - 1.0) for k in ggf_keys], dtype='float64')

        vbf_ratio_y   = np.ones(1, dtype='float64')
        vbf_ratio_eyl = np.array([max(0.0, 1.0 - self.smvals[k][1]) for k in vbf_keys], dtype='float64')
        vbf_ratio_eyh = np.array([max(0.0, self.smvals[k][2] - 1.0) for k in vbf_keys], dtype='float64')

        # Observed μ from poivals: [μ_nom, μ_down, μ_up] (down/up stored as magnitudes)
        ggf_mu_y   = np.array([self.poivals[k][0] for k in ggf_keys], dtype='float64')
        ggf_mu_eyl = np.array([abs(self.poivals[k][1]) for k in ggf_keys], dtype='float64')
        ggf_mu_eyh = np.array([abs(self.poivals[k][2]) for k in ggf_keys], dtype='float64')

        vbf_mu_y   = np.array([self.poivals[k][0] for k in vbf_keys], dtype='float64')
        vbf_mu_eyl = np.array([abs(self.poivals[k][1]) for k in vbf_keys], dtype='float64')
        vbf_mu_eyh = np.array([abs(self.poivals[k][2]) for k in vbf_keys], dtype='float64')

        # Original behavior: auto symmetric range around 1
        def ratio_range():
            lows  = list(ggf_ratio_y - ggf_ratio_eyl) + list(vbf_ratio_y - vbf_ratio_eyl) + list(ggf_mu_y - ggf_mu_eyl) + list(vbf_mu_y - vbf_mu_eyl)
            highs = list(ggf_ratio_y + ggf_ratio_eyh) + list(vbf_ratio_y + vbf_ratio_eyh) + list(ggf_mu_y + ggf_mu_eyh) + list(vbf_mu_y + vbf_mu_eyh)
            lo, hi = min(lows), max(highs)
            R = max(abs(1.0 - lo), abs(hi - 1.0))
            return 1.0 - 1.1*R, 1.0 + 1.1*R

        if ratio_yrange is None:
            rlo, rhi = ratio_range()
        else:
            rlo, rhi = ratio_yrange

        pad2.cd()
        h3 = rt.TH1D("h_rat_ggf","",3,-0.5,2.5)
        set_bin_labels(h3, ['[200,300]', '[300,450]', '[450,#infty)'])
        h3.GetYaxis().SetTitle("#sigma_{obs} / #sigma_{SM}")
        h3.GetYaxis().SetTitleOffset(2*pad2.GetAbsHNDC())
        h3.GetXaxis().SetTitleSize(textsize2); h3.GetXaxis().SetLabelSize(1.3*textsize2)
        h3.GetYaxis().SetTitleSize(textsize2); h3.GetYaxis().SetLabelSize(textsize2)
        h3.GetYaxis().SetRangeUser(rlo, rhi)
        h3.Draw("axis")

        g3 = rt.TGraphAsymmErrors(3, x1, ggf_ratio_y, ex1, ex1, ggf_ratio_eyl, ggf_ratio_eyh)
        g3.SetFillColor(4); g3.SetFillStyle(3003); g3.SetLineColor(4); g3.SetLineWidth(1)
        g3.Draw("2 same"); g3.Draw("pe same")

        grat = rt.TGraphAsymmErrors(3, x1, ggf_mu_y, z3, z3, ggf_mu_eyl, ggf_mu_eyh)
        grat.SetMarkerColor(1); grat.SetMarkerStyle(20); grat.SetLineColor(1); grat.SetLineWidth(3)
        grat.Draw("pe same")

        pad4.cd()
        h4 = rt.TH1D("h_rat_vbf","",1,2.5,3.5)
        set_bin_labels(h4, ['[1000,#infty)'])
        h4.GetYaxis().SetTitleSize(0); h4.GetYaxis().SetLabelSize(0)
        h4.GetYaxis().SetRangeUser(rlo, rhi)
        h4.GetXaxis().SetTitleSize(textsize2); h4.GetXaxis().SetLabelSize(1.3*textsize2)
        h4.Draw("axis")

        g4  = rt.TGraphAsymmErrors(1, x2, vbf_ratio_y, ex2, ex2, vbf_ratio_eyl, vbf_ratio_eyh)
        g4.SetFillColor(94); g4.SetFillStyle(3003); g4.SetMarkerColor(94); g4.SetLineColor(94); g4.SetLineWidth(1)
        g4.Draw("2 same"); g4.Draw("pe same")

        grat2 = rt.TGraphAsymmErrors(1, x2, vbf_mu_y, z1, z1, vbf_mu_eyl, vbf_mu_eyh)
        grat2.SetMarkerColor(1); grat2.SetMarkerStyle(20); grat2.SetLineColor(1); grat2.SetLineWidth(3)
        grat2.Draw("pe same")

        pad1.cd(); tag2.SetTextSize(textsize2); tag2.Draw(); #tag3.SetTextSize(textsize1); tag3.Draw()
        pad3.cd(); tag1.SetTextSize(textsize3); tag1.Draw(); tag5.SetTextSize(textsize3); tag5.Draw()
        pad1.cd(); leg.Draw("same")

        c.Print("stxs.pdf"); c.Print("stxs.png"); c.Print("stxs.C")
        if showplot:
            c.Draw()

        
            
if __name__ == "__main__":

    up = UnfoldingPlot()
    # my_instance.greet()

    #generate the plot
    print('creating the plot...')
    #up.makeplot()
    # up.makeplot(top_logy=True, top_yrange=(1.0, 10000.0), ratio_yrange=(-5.0, 15.0))
    up.makeplot(top_logy=False, top_yrange=(-3000.0, 2000.0), ratio_yrange=(-11, 11))
