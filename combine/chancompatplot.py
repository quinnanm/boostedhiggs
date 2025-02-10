import ROOT as rt
# from RootIterator import RootIterator

# from optparse import OptionParser
import argparse

poiMap = {'r':"#mu"}
        
catDict = {}
catDict['muonCR','r'] = '#splitline{Combined}{   #scale[0.8]{#mu_{H} = %.1f_{#minus%.1f}^{+%.1f}}}'
catDict['SR1','r'] = '#splitline{qqH}{    #scale[0.8]{#mu = %.1f_{#minus%.1f}^{+%.1f}}}'
catDict['SR2','r'] = '#splitline{ggH [250, 350] GeV}{    #scale[0.8]{#mu = %.1f_{#minus%.1f}^{+%.1f}}}'
catDict['SR3','r'] = '#splitline{ggH [350, 500] GeV}{    #scale[0.8]{#mu = %.1f_{#minus%.1f}^{+%.1f}}}'
catDict['SR4','r'] = '#splitline{ggH [500, Inf] GeV}{    #scale[0.8]{#mu = %.1f_{#minus%.1f}^{+%.1f}}}'

# catDict['muonCR','r'] = '#splitline{Combined}{   #scale[0.8]{#mu_{H} = %.1f_{#minus%.1f}^{+%.1f}}}'
# catDict['cat1','r'] = '#splitline{[450, 500] GeV}{    #scale[0.8]{#mu_{H} = %.1f_{#minus%.1f}^{+%.1f}}}'
# catDict['cat2','r'] = '#splitline{[500, 550] GeV}{    #scale[0.8]{#mu_{H} = %.1f_{#minus%.1f}^{+%.1f}}}'
# catDict['cat3','r'] = '#splitline{[550, 600] GeV}{    #scale[0.8]{#mu_{H} = %.1f_{#minus%.1f}^{+%.1f}}}'
# catDict['cat4','r'] = '#splitline{[600, 675] GeV}{    #scale[0.8]{#mu_{H} = %.1f_{#minus%.1f}^{+%.1f}}}'
# catDict['cat5','r'] = '#splitline{[675, 800] GeV}{    #scale[0.8]{#mu_{H} = %.1f_{#minus%.1f}^{+%.1f}}}'
# catDict['cat6','r'] = '#splitline{[800, 1000] GeV}{     #scale[0.8]{#mu_{H} = %.1f_{#minus%.1f}^{+%.1f}}}'

def cccPlot(poi = "r", rMin =-10, rMax=15, filename="ccc_r.pdf"):
    c1 = rt.TCanvas("c1")
    c1.SetLeftMargin(0.4)
    c1.SetBottomMargin(0.12)
    c1.SetGridx(1)

    if (rt.gFile == 0):
        print("No input file open")
        sys.exit()

    fit_nominal   = rt.gFile.Get("fit_nominal")
    fit_alternate = rt.gFile.Get("fit_alternate")

    if (fit_nominal == 0 or fit_alternate == 0):
        print("Input file ", gFile.GetName(), " does not contain fit_nominal or fit_alternate")
        sys.exit()

    rFit = fit_nominal.floatParsFinal().find(poi)
    if (rFit == 0):
        print ("Nominal fit does not contain parameter ", poi)
        sys.exit()

    prefix =  "_ChannelCompatibilityCheck_%s_"%poi

    nChann = 0
    iter_alternate = fit_alternate.floatParsFinal().createIterator()
    obj = iter_alternate.Next()
    while obj:
        print(obj.GetName())
        if prefix in obj.GetName():
            nChann += 1
        obj = iter_alternate.Next()
        
    # for a in RootIterator(fit_alternate.floatParsFinal()):
    #     if prefix in a.GetName():
    #         nChann+=1

    frame = rt.TH2F("frame",";%s;"%poiMap[poi],1,rMin,rMax,nChann,0,nChann)

    iChann = 0
    points = rt.TGraphAsymmErrors(nChann)
    
    # for a in sorted(RootIterator(fit_alternate.floatParsFinal())):
    #     if (rt.TString(a.GetName()).Index(prefix) == 0):
    #         ri = a
    #         channel = a.GetName()
    #         channel = channel.replace(prefix,"")            
    #         print(channel)
    #         if channel=='muonCR':
    #             # put at some dummy value
    #             points.SetPoint(iChann,       100, iChann+0.5)
    #             points.SetPointError(iChann, -1, 1, 0, 0)
    #         else:
    #             points.SetPoint(iChann,       ri.getVal(), iChann+0.5)
    #             points.SetPointError(iChann, -ri.getAsymErrorLo(), ri.getAsymErrorHi(), 0, 0)
    #         iChann+=1
    #         if channel=='muonCR':
    #             frame.GetYaxis().SetBinLabel(iChann, (catDict[channel,options.poi]%(rFit.getVal(),-rFit.getAsymErrorLo(), rFit.getAsymErrorHi())).replace('-','#minus'))
    #         else:
    #             frame.GetYaxis().SetBinLabel(iChann, (catDict[channel,options.poi]%(ri.getVal(),-ri.getAsymErrorLo(), ri.getAsymErrorHi())).replace('-','#minus'))

    iter_alternate = fit_alternate.floatParsFinal().createIterator()
    obj = iter_alternate.Next()
    while obj:
        if obj.GetName().startswith(prefix):
            channel = obj.GetName().replace(prefix, "")
            print(channel)
            if channel == 'muonCR':
                points.SetPoint(iChann, 100, iChann + 0.5)
                points.SetPointError(iChann, -1, 1, 0, 0)
            else:
                points.SetPoint(iChann, obj.getVal(), iChann + 0.5)
                points.SetPointError(iChann, -obj.getAsymErrorLo(), obj.getAsymErrorHi(), 0, 0)
            
            if channel == 'muonCR':
                label = (catDict[channel, poi] % (rFit.getVal(), -rFit.getAsymErrorLo(), rFit.getAsymErrorHi())).replace('-', '#minus')
            else:
                label = (catDict[channel, poi] % (obj.getVal(), -obj.getAsymErrorLo(), obj.getAsymErrorHi())).replace('-', '#minus')
            print(label)
            frame.GetYaxis().SetBinLabel(iChann + 1, label)
            iChann += 1
            
        obj = iter_alternate.Next()
    
    points.SetLineColor(rt.kRed)
    points.SetLineWidth(3)
    points.SetMarkerStyle(21)
    frame.GetXaxis().SetTitleSize(0.05)
    frame.GetXaxis().SetLabelSize(0.04)
    frame.GetYaxis().SetLabelSize(0.06)
    frame.Draw()
    
    rt.gStyle.SetOptStat(0)

    globalFitBand = rt.TBox(rFit.getVal()+rFit.getAsymErrorLo(), 0, rFit.getVal()+rFit.getAsymErrorHi(), nChann)
    globalFitBand.SetFillColor(rt.kGreen)
    globalFitBand.SetLineStyle(0)
    globalFitBand.Draw('')

    globalFitLine = rt.TLine(rFit.getVal(), 0, rFit.getVal(), nChann)
    globalFitLine.SetLineWidth(4)
    globalFitLine.SetLineColor(214)
    globalFitLine.Draw('')

    points.Draw("PZ SAME")
    
    l = rt.TLatex()
    l.SetTextAlign(11)
    l.SetTextSize(0.045)
    l.SetNDC()
    l.SetTextFont(62)
    l.DrawLatex(0.41,0.85,"CMS")
    l.SetTextFont(52)
    l.DrawLatex(0.41,0.8,"Preliminary")
    # tag1 = rt.TLatex(0.67,0.92,"%.1f fb^{-1} (13 TeV)"%(35.9))
    # tag1.SetNDC()
    # tag1.SetTextFont(42)
    # tag1.SetTextSize(0.045)
    # tag1.Draw()
    
    c1.RedrawAxis()
    c1.Print(filename)
    c1.Print(filename.replace('.pdf','.C'))

# if __name__ == '__main__':
    #  parser = OptionParser()
    # parser.add_option('-b', action='store_true', dest='noX', default=False, help='no X11 windows')
    # parser.add_option('--rMin',dest='rMin', default=-10 ,type='float',help='minimum of r (signal strength) in profile likelihood plot')
    # parser.add_option('--rMax',dest='rMax', default=15,type='float',help='maximum of r (signal strength) in profile likelihood plot')  
    
    # parser.add_option('-P','--poi'   ,action='store',type='string',dest='poi'   ,default='r', help='poi name')  

    
    # (options, args) = parser.parse_args()
    # rt.gROOT.SetBatch()
    # f = rt.TFile.Open('higgsCombineHWW.ChannelCompatibilityCheck.mH125.root'%options.poi)
    # cccPlot(options.poi, options.rMin, options.rMax, "ccc_"+options.poi+".pdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Channel Compatibility Check plots")
    parser.add_argument('-b', action='store_true', dest='noX', default=True, help='no X11 windows')
    parser.add_argument('--rMin', dest='rMin', default=-2, type=float, help='minimum of r (signal strength) in profile likelihood plot')
    parser.add_argument('--rMax', dest='rMax', default=3, type=float, help='maximum of r (signal strength) in profile likelihood plot')  
    parser.add_argument('-P', '--poi', action='store', type=str, dest='poi', default='r', help='POI name')  

    args = parser.parse_args()

    # Set ROOT to batch mode if no X11 window is required
    if args.noX:
        rt.gROOT.SetBatch()

    # Open the ROOT file
    filename = f'higgsCombineHWW.ChannelCompatibilityCheck.mH125.root'
    f = rt.TFile.Open(filename)

    if not f or f.IsZombie():
        print(f"Error: Cannot open file {filename}")
        sys.exit(1)

    cccPlot(args.poi, args.rMin, args.rMax, f"ccc_{args.poi}.pdf")
