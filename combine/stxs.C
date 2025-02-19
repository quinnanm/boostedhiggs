#ifdef __CLING__
#pragma cling optimize(0)
#endif
void stxs()
{
//=========Macro generated from canvas: stxs/stxs
//=========  (Tue Feb 18 21:10:34 2025) by ROOT version 6.30/07
   TCanvas *stxs = new TCanvas("stxs", "stxs",0,0,800,600);
   gStyle->SetOptStat(0);
   stxs->SetHighLightColor(2);
   stxs->Range(0,0,1,1);
   stxs->SetFillColor(0);
   stxs->SetBorderMode(0);
   stxs->SetBorderSize(2);
   stxs->SetFrameBorderMode(0);
  
// ------------>Primitives in pad: pad1
   TPad *pad1__0 = new TPad("pad1", "pad1",0,0.4,0.6,1);
   pad1__0->Draw();
   pad1__0->cd();
   pad1__0->Range(-1.042169,-2.352941,2.572289,4.705882);
   pad1__0->SetFillColor(0);
   pad1__0->SetBorderMode(0);
   pad1__0->SetBorderSize(2);
   pad1__0->SetLogy();
   pad1__0->SetLeftMargin(0.15);
   pad1__0->SetRightMargin(0.02);
   pad1__0->SetBottomMargin(0.05);
   pad1__0->SetFrameBorderMode(0);
   pad1__0->SetFrameBorderMode(0);
   
   TH1D *dummy1__1 = new TH1D("dummy1__1","",3,-0.5,2.5);
   dummy1__1->SetMinimum(0.01);
   dummy1__1->SetMaximum(10000);
   dummy1__1->SetEntries(3);
   dummy1__1->SetStats(0);
   dummy1__1->SetLineColor(0);
   dummy1__1->SetLineWidth(3);
   dummy1__1->GetXaxis()->SetTitle("p_{T}^{H} [GeV]");
   dummy1__1->GetXaxis()->SetBinLabel(1,"[200,300]");
   dummy1__1->GetXaxis()->SetBinLabel(2,"[300,450]");
   dummy1__1->GetXaxis()->SetBinLabel(3,"[450,#infty)");
   dummy1__1->GetXaxis()->CenterTitle(true);
   dummy1__1->GetXaxis()->SetLabelFont(42);
   dummy1__1->GetXaxis()->SetLabelSize(0);
   dummy1__1->GetXaxis()->SetTitleSize(0);
   dummy1__1->GetXaxis()->SetTitleOffset(1.5);
   dummy1__1->GetXaxis()->SetTitleFont(42);
   dummy1__1->GetYaxis()->SetTitle("#sigma_{obs} [fb]");
   dummy1__1->GetYaxis()->SetLabelFont(42);
   dummy1__1->GetYaxis()->SetLabelSize(0.0553613);
   dummy1__1->GetYaxis()->SetTitleSize(0.0553613);
   dummy1__1->GetYaxis()->SetTitleOffset(1.2);
   dummy1__1->GetYaxis()->SetTitleFont(42);
   dummy1__1->GetZaxis()->SetLabelFont(42);
   dummy1__1->GetZaxis()->SetTitleOffset(1);
   dummy1__1->GetZaxis()->SetTitleFont(42);
   dummy1__1->Draw("");
   
   Double_t Graph0_fx3001[3] = { 0, 1, 2 };
   Double_t Graph0_fy3001[3] = { 128.8848, 40.74342, 7.040057 };
   Double_t Graph0_felx3001[3] = { 0.5, 0.5, 0.5 };
   Double_t Graph0_fely3001[3] = { 122.3632, 38.68181, 6.68383 };
   Double_t Graph0_fehx3001[3] = { 0.5, 0.5, 0.5 };
   Double_t Graph0_fehy3001[3] = { 135.4063, 42.80504, 7.396284 };
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(3,Graph0_fx3001,Graph0_fy3001,Graph0_felx3001,Graph0_fehx3001,Graph0_fely3001,Graph0_fehy3001);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   
   TH1F *Graph_Graph03001 = new TH1F("Graph_Graph03001","Graph",100,-0.8,2.8);
   Graph_Graph03001->SetMinimum(0.3206042);
   Graph_Graph03001->SetMaximum(290.6846);
   Graph_Graph03001->SetDirectory(nullptr);
   Graph_Graph03001->SetStats(0);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   Graph_Graph03001->SetLineColor(ci);
   Graph_Graph03001->GetXaxis()->SetLabelFont(42);
   Graph_Graph03001->GetXaxis()->SetTitleOffset(1);
   Graph_Graph03001->GetXaxis()->SetTitleFont(42);
   Graph_Graph03001->GetYaxis()->SetLabelFont(42);
   Graph_Graph03001->GetYaxis()->SetTitleFont(42);
   Graph_Graph03001->GetZaxis()->SetLabelFont(42);
   Graph_Graph03001->GetZaxis()->SetTitleOffset(1);
   Graph_Graph03001->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph03001);
   
   grae->Draw("2");
   
   Double_t Graph0_fx3002[3] = { 0, 1, 2 };
   Double_t Graph0_fy3002[3] = { 128.8848, 40.74342, 7.040057 };
   Double_t Graph0_felx3002[3] = { 0.5, 0.5, 0.5 };
   Double_t Graph0_fely3002[3] = { 122.3632, 38.68181, 6.68383 };
   Double_t Graph0_fehx3002[3] = { 0.5, 0.5, 0.5 };
   Double_t Graph0_fehy3002[3] = { 135.4063, 42.80504, 7.396284 };
   grae = new TGraphAsymmErrors(3,Graph0_fx3002,Graph0_fy3002,Graph0_felx3002,Graph0_fehx3002,Graph0_fely3002,Graph0_fehy3002);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   
   TH1F *Graph_Graph03002 = new TH1F("Graph_Graph03002","Graph",100,-0.8,2.8);
   Graph_Graph03002->SetMinimum(0.3206042);
   Graph_Graph03002->SetMaximum(290.6846);
   Graph_Graph03002->SetDirectory(nullptr);
   Graph_Graph03002->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph03002->SetLineColor(ci);
   Graph_Graph03002->GetXaxis()->SetLabelFont(42);
   Graph_Graph03002->GetXaxis()->SetTitleOffset(1);
   Graph_Graph03002->GetXaxis()->SetTitleFont(42);
   Graph_Graph03002->GetYaxis()->SetLabelFont(42);
   Graph_Graph03002->GetYaxis()->SetTitleFont(42);
   Graph_Graph03002->GetZaxis()->SetLabelFont(42);
   Graph_Graph03002->GetZaxis()->SetTitleOffset(1);
   Graph_Graph03002->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph03002);
   
   grae->Draw("pe");
   
   Double_t Graph1_fx3003[3] = { 0, 1, 2 };
   Double_t Graph1_fy3003[3] = { -81.32629, -82.54618, 0.8096066 };
   Double_t Graph1_felx3003[3] = { 0, 0, 0 };
   Double_t Graph1_fely3003[3] = { 590.2922, 114.0816, 18.22671 };
   Double_t Graph1_fehx3003[3] = { 0, 0, 0 };
   Double_t Graph1_fehy3003[3] = { 733.2255, 90.49114, 21.38769 };
   grae = new TGraphAsymmErrors(3,Graph1_fx3003,Graph1_fy3003,Graph1_felx3003,Graph1_fehx3003,Graph1_fely3003,Graph1_fehy3003);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph13003 = new TH1F("Graph_Graph13003","Graph",100,0,2.2);
   Graph_Graph13003->SetMinimum(0.7253437);
   Graph_Graph13003->SetMaximum(725.3437);
   Graph_Graph13003->SetDirectory(nullptr);
   Graph_Graph13003->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph13003->SetLineColor(ci);
   Graph_Graph13003->GetXaxis()->SetLabelFont(42);
   Graph_Graph13003->GetXaxis()->SetTitleOffset(1);
   Graph_Graph13003->GetXaxis()->SetTitleFont(42);
   Graph_Graph13003->GetYaxis()->SetLabelFont(42);
   Graph_Graph13003->GetYaxis()->SetTitleFont(42);
   Graph_Graph13003->GetZaxis()->SetLabelFont(42);
   Graph_Graph13003->GetZaxis()->SetTitleOffset(1);
   Graph_Graph13003->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph13003);
   
   grae->Draw("pe");
   TLatex *   tex = new TLatex(0.17,0.92,"CMS");
   tex->SetNDC();
   tex->SetTextSize(0.0553613);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.26,0.92,"H(WW)");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.0553613);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.2,0.82,"ggF");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.0553613);
   tex->SetLineWidth(2);
   tex->Draw();
   
   TLegend *leg = new TLegend(0.44,0.65,0.82,0.87,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetTextSize(0.0553613);
   leg->SetLineColor(0);
   leg->SetLineStyle(0);
   leg->SetLineWidth(0);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   TLegendEntry *entry=leg->AddEntry("histo","Observed (stat #oplus syst)","pe");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(3);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph0","ggF (HJMINLO)","f");
   entry->SetFillColor(4);
   entry->SetFillStyle(3003);
   entry->SetLineColor(4);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph","VBF (POWHEG+HC)","f");
   entry->SetFillColor(94);
   entry->SetFillStyle(3003);
   entry->SetLineColor(94);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   leg->Draw();
   pad1__0->Modified();
   stxs->cd();
  
// ------------>Primitives in pad: pad2
   TPad *pad2__1 = new TPad("pad2", "pad2",0,0,0.6,0.4);
   pad2__1->Draw();
   pad2__1->cd();
   pad2__1->Range(-1.042169,-14.00013,2.572289,16.0003);
   pad2__1->SetFillColor(0);
   pad2__1->SetBorderMode(0);
   pad2__1->SetBorderSize(2);
   pad2__1->SetLeftMargin(0.15);
   pad2__1->SetRightMargin(0.02);
   pad2__1->SetTopMargin(1e-05);
   pad2__1->SetBottomMargin(0.3);
   pad2__1->SetFrameBorderMode(0);
   pad2__1->SetFrameBorderMode(0);
   
   TH1D *dummy3__2 = new TH1D("dummy3__2","",3,-0.5,2.5);
   dummy3__2->SetMinimum(-5);
   dummy3__2->SetMaximum(16);
   dummy3__2->SetStats(0);
   dummy3__2->SetLineColor(0);
   dummy3__2->SetLineWidth(3);
   dummy3__2->GetXaxis()->SetTitle("p_{T}^{H} [GeV]");
   dummy3__2->GetXaxis()->SetBinLabel(1,"[200,300]");
   dummy3__2->GetXaxis()->SetBinLabel(2,"[300,450]");
   dummy3__2->GetXaxis()->SetBinLabel(3,"[450,#infty)");
   dummy3__2->GetXaxis()->CenterTitle(true);
   dummy3__2->GetXaxis()->SetLabelFont(42);
   dummy3__2->GetXaxis()->SetLabelSize(0.1079545);
   dummy3__2->GetXaxis()->SetTitleSize(0.08304196);
   dummy3__2->GetXaxis()->SetTitleOffset(1.5);
   dummy3__2->GetXaxis()->SetTitleFont(42);
   dummy3__2->GetYaxis()->SetTitle("#sigma_{obs} / #sigma_{SM}");
   dummy3__2->GetYaxis()->SetLabelFont(42);
   dummy3__2->GetYaxis()->SetLabelSize(0.08304196);
   dummy3__2->GetYaxis()->SetTitleSize(0.08304196);
   dummy3__2->GetYaxis()->SetTitleOffset(0.8);
   dummy3__2->GetYaxis()->SetTitleFont(42);
   dummy3__2->GetZaxis()->SetLabelFont(42);
   dummy3__2->GetZaxis()->SetTitleOffset(1);
   dummy3__2->GetZaxis()->SetTitleFont(42);
   dummy3__2->Draw("");
   
   Double_t Graph0_fx3004[3] = { 0, 1, 2 };
   Double_t Graph0_fy3004[3] = { 1, 1, 1 };
   Double_t Graph0_felx3004[3] = { 0.5, 0.5, 0.5 };
   Double_t Graph0_fely3004[3] = { 0.9494, 0.9494, 0.9494 };
   Double_t Graph0_fehx3004[3] = { 0.5, 0.5, 0.5 };
   Double_t Graph0_fehy3004[3] = { 1.0506, 1.0506, 1.0506 };
   grae = new TGraphAsymmErrors(3,Graph0_fx3004,Graph0_fy3004,Graph0_felx3004,Graph0_fehx3004,Graph0_fely3004,Graph0_fehy3004);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   
   TH1F *Graph_Graph03004 = new TH1F("Graph_Graph03004","Graph",100,-0.8,2.8);
   Graph_Graph03004->SetMinimum(0.04554);
   Graph_Graph03004->SetMaximum(2.2506);
   Graph_Graph03004->SetDirectory(nullptr);
   Graph_Graph03004->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph03004->SetLineColor(ci);
   Graph_Graph03004->GetXaxis()->SetLabelFont(42);
   Graph_Graph03004->GetXaxis()->SetTitleOffset(1);
   Graph_Graph03004->GetXaxis()->SetTitleFont(42);
   Graph_Graph03004->GetYaxis()->SetLabelFont(42);
   Graph_Graph03004->GetYaxis()->SetTitleFont(42);
   Graph_Graph03004->GetZaxis()->SetLabelFont(42);
   Graph_Graph03004->GetZaxis()->SetTitleOffset(1);
   Graph_Graph03004->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph03004);
   
   grae->Draw("2");
   
   Double_t Graph0_fx3005[3] = { 0, 1, 2 };
   Double_t Graph0_fy3005[3] = { 1, 1, 1 };
   Double_t Graph0_felx3005[3] = { 0.5, 0.5, 0.5 };
   Double_t Graph0_fely3005[3] = { 0.9494, 0.9494, 0.9494 };
   Double_t Graph0_fehx3005[3] = { 0.5, 0.5, 0.5 };
   Double_t Graph0_fehy3005[3] = { 1.0506, 1.0506, 1.0506 };
   grae = new TGraphAsymmErrors(3,Graph0_fx3005,Graph0_fy3005,Graph0_felx3005,Graph0_fehx3005,Graph0_fely3005,Graph0_fehy3005);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   
   TH1F *Graph_Graph03005 = new TH1F("Graph_Graph03005","Graph",100,-0.8,2.8);
   Graph_Graph03005->SetMinimum(0.04554);
   Graph_Graph03005->SetMaximum(2.2506);
   Graph_Graph03005->SetDirectory(nullptr);
   Graph_Graph03005->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph03005->SetLineColor(ci);
   Graph_Graph03005->GetXaxis()->SetLabelFont(42);
   Graph_Graph03005->GetXaxis()->SetTitleOffset(1);
   Graph_Graph03005->GetXaxis()->SetTitleFont(42);
   Graph_Graph03005->GetYaxis()->SetLabelFont(42);
   Graph_Graph03005->GetYaxis()->SetTitleFont(42);
   Graph_Graph03005->GetZaxis()->SetLabelFont(42);
   Graph_Graph03005->GetZaxis()->SetTitleOffset(1);
   Graph_Graph03005->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph03005);
   
   grae->Draw("pe");
   
   Double_t Graph1_fx3006[3] = { 0, 1, 2 };
   Double_t Graph1_fy3006[3] = { -0.631, -2.026, 0.115 };
   Double_t Graph1_felx3006[3] = { 0, 0, 0 };
   Double_t Graph1_fely3006[3] = { 4.58, 2.8, 2.589 };
   Double_t Graph1_fehx3006[3] = { 0, 0, 0 };
   Double_t Graph1_fehy3006[3] = { 5.689, 2.221, 3.038 };
   grae = new TGraphAsymmErrors(3,Graph1_fx3006,Graph1_fy3006,Graph1_felx3006,Graph1_fehx3006,Graph1_fely3006,Graph1_fehy3006);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph13006 = new TH1F("Graph_Graph13006","Graph",100,0,2.2);
   Graph_Graph13006->SetMinimum(-6.2379);
   Graph_Graph13006->SetMaximum(6.0849);
   Graph_Graph13006->SetDirectory(nullptr);
   Graph_Graph13006->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph13006->SetLineColor(ci);
   Graph_Graph13006->GetXaxis()->SetLabelFont(42);
   Graph_Graph13006->GetXaxis()->SetTitleOffset(1);
   Graph_Graph13006->GetXaxis()->SetTitleFont(42);
   Graph_Graph13006->GetYaxis()->SetLabelFont(42);
   Graph_Graph13006->GetYaxis()->SetTitleFont(42);
   Graph_Graph13006->GetZaxis()->SetLabelFont(42);
   Graph_Graph13006->GetZaxis()->SetTitleOffset(1);
   Graph_Graph13006->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph13006);
   
   grae->Draw("pe");
   pad2__1->Modified();
   stxs->cd();
  
// ------------>Primitives in pad: pad3
   TPad *pad3__2 = new TPad("pad3", "pad3",0.6,0.4,1,1);
   pad3__2->Draw();
   pad3__2->cd();
   pad3__2->Range(2.475904,-2.352941,3.680723,4.705882);
   pad3__2->SetFillColor(0);
   pad3__2->SetBorderMode(0);
   pad3__2->SetBorderSize(2);
   pad3__2->SetLogy();
   pad3__2->SetLeftMargin(0.02);
   pad3__2->SetRightMargin(0.15);
   pad3__2->SetBottomMargin(0.05);
   pad3__2->SetFrameBorderMode(0);
   pad3__2->SetFrameBorderMode(0);
   
   TH1D *dummy2__3 = new TH1D("dummy2__3","",1,2.5,3.5);
   dummy2__3->SetMinimum(0.01);
   dummy2__3->SetMaximum(10000);
   dummy2__3->SetEntries(1);
   dummy2__3->SetStats(0);
   dummy2__3->SetLineColor(0);
   dummy2__3->SetLineWidth(3);
   dummy2__3->GetXaxis()->SetTitle("m_{jj}^{gen} [GeV]");
   dummy2__3->GetXaxis()->SetBinLabel(1,"[1000,#infty)");
   dummy2__3->GetXaxis()->SetRange(1,2);
   dummy2__3->GetXaxis()->CenterTitle(true);
   dummy2__3->GetXaxis()->SetLabelFont(42);
   dummy2__3->GetXaxis()->SetLabelSize(0);
   dummy2__3->GetXaxis()->SetTitleSize(0);
   dummy2__3->GetXaxis()->SetTitleOffset(1.5);
   dummy2__3->GetXaxis()->SetTitleFont(42);
   dummy2__3->GetYaxis()->SetLabelFont(42);
   dummy2__3->GetYaxis()->SetLabelSize(0);
   dummy2__3->GetYaxis()->SetTitleSize(0);
   dummy2__3->GetYaxis()->SetTitleFont(42);
   dummy2__3->GetZaxis()->SetLabelFont(42);
   dummy2__3->GetZaxis()->SetTitleOffset(1);
   dummy2__3->GetZaxis()->SetTitleFont(42);
   dummy2__3->Draw("");
   
   Double_t Graph0_fx3007[1] = { 3 };
   Double_t Graph0_fy3007[1] = { 100.8066 };
   Double_t Graph0_felx3007[1] = { 0.5 };
   Double_t Graph0_fely3007[1] = { 98.66949 };
   Double_t Graph0_fehx3007[1] = { 0.5 };
   Double_t Graph0_fehy3007[1] = { 102.9638 };
   grae = new TGraphAsymmErrors(1,Graph0_fx3007,Graph0_fy3007,Graph0_felx3007,Graph0_fehx3007,Graph0_fely3007,Graph0_fehy3007);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(94);
   grae->SetFillStyle(3003);
   grae->SetLineColor(94);
   grae->SetMarkerColor(94);
   
   TH1F *Graph_Graph03007 = new TH1F("Graph_Graph03007","Graph",100,2.4,3.6);
   Graph_Graph03007->SetMinimum(1.92339);
   Graph_Graph03007->SetMaximum(223.9338);
   Graph_Graph03007->SetDirectory(nullptr);
   Graph_Graph03007->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph03007->SetLineColor(ci);
   Graph_Graph03007->GetXaxis()->SetLabelFont(42);
   Graph_Graph03007->GetXaxis()->SetTitleOffset(1);
   Graph_Graph03007->GetXaxis()->SetTitleFont(42);
   Graph_Graph03007->GetYaxis()->SetLabelFont(42);
   Graph_Graph03007->GetYaxis()->SetTitleFont(42);
   Graph_Graph03007->GetZaxis()->SetLabelFont(42);
   Graph_Graph03007->GetZaxis()->SetTitleOffset(1);
   Graph_Graph03007->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph03007);
   
   grae->Draw("2");
   
   Double_t Graph0_fx3008[1] = { 3 };
   Double_t Graph0_fy3008[1] = { 100.8066 };
   Double_t Graph0_felx3008[1] = { 0.5 };
   Double_t Graph0_fely3008[1] = { 98.66949 };
   Double_t Graph0_fehx3008[1] = { 0.5 };
   Double_t Graph0_fehy3008[1] = { 102.9638 };
   grae = new TGraphAsymmErrors(1,Graph0_fx3008,Graph0_fy3008,Graph0_felx3008,Graph0_fehx3008,Graph0_fely3008,Graph0_fehy3008);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(94);
   grae->SetFillStyle(3003);
   grae->SetLineColor(94);
   grae->SetMarkerColor(94);
   
   TH1F *Graph_Graph03008 = new TH1F("Graph_Graph03008","Graph",100,2.4,3.6);
   Graph_Graph03008->SetMinimum(1.92339);
   Graph_Graph03008->SetMaximum(223.9338);
   Graph_Graph03008->SetDirectory(nullptr);
   Graph_Graph03008->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph03008->SetLineColor(ci);
   Graph_Graph03008->GetXaxis()->SetLabelFont(42);
   Graph_Graph03008->GetXaxis()->SetTitleOffset(1);
   Graph_Graph03008->GetXaxis()->SetTitleFont(42);
   Graph_Graph03008->GetYaxis()->SetLabelFont(42);
   Graph_Graph03008->GetYaxis()->SetTitleFont(42);
   Graph_Graph03008->GetZaxis()->SetLabelFont(42);
   Graph_Graph03008->GetZaxis()->SetTitleOffset(1);
   Graph_Graph03008->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph03008);
   
   grae->Draw("pe");
   
   Double_t Graph1_fx3009[1] = { 3 };
   Double_t Graph1_fy3009[1] = { 160.8873 };
   Double_t Graph1_felx3009[1] = { 0 };
   Double_t Graph1_fely3009[1] = { 80.04043 };
   Double_t Graph1_fehx3009[1] = { 0 };
   Double_t Graph1_fehy3009[1] = { 117.8429 };
   grae = new TGraphAsymmErrors(1,Graph1_fx3009,Graph1_fy3009,Graph1_felx3009,Graph1_fehx3009,Graph1_fely3009,Graph1_fehy3009);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph13009 = new TH1F("Graph_Graph13009","Graph",100,2.9,4.1);
   Graph_Graph13009->SetMinimum(61.05855);
   Graph_Graph13009->SetMaximum(298.5185);
   Graph_Graph13009->SetDirectory(nullptr);
   Graph_Graph13009->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph13009->SetLineColor(ci);
   Graph_Graph13009->GetXaxis()->SetLabelFont(42);
   Graph_Graph13009->GetXaxis()->SetTitleOffset(1);
   Graph_Graph13009->GetXaxis()->SetTitleFont(42);
   Graph_Graph13009->GetYaxis()->SetLabelFont(42);
   Graph_Graph13009->GetYaxis()->SetTitleFont(42);
   Graph_Graph13009->GetZaxis()->SetLabelFont(42);
   Graph_Graph13009->GetZaxis()->SetTitleOffset(1);
   Graph_Graph13009->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph13009);
   
   grae->Draw("p");
      tex = new TLatex(0.46,0.92,"138 fb^{-1} (13 TeV)");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.0553613);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.08,0.82,"VBF");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.0553613);
   tex->SetLineWidth(2);
   tex->Draw();
   pad3__2->Modified();
   stxs->cd();
  
// ------------>Primitives in pad: pad4
   TPad *pad4__3 = new TPad("pad4", "pad4",0.6,0,1,0.4);
   pad4__3->Draw();
   pad4__3->cd();
   pad4__3->Range(2.475904,-14.00013,3.680723,16.0003);
   pad4__3->SetFillColor(0);
   pad4__3->SetBorderMode(0);
   pad4__3->SetBorderSize(2);
   pad4__3->SetLeftMargin(0.02);
   pad4__3->SetRightMargin(0.15);
   pad4__3->SetTopMargin(1e-05);
   pad4__3->SetBottomMargin(0.3);
   pad4__3->SetFrameBorderMode(0);
   pad4__3->SetFrameBorderMode(0);
   
   TH1D *dummy4__4 = new TH1D("dummy4__4","",1,2.5,3.5);
   dummy4__4->SetMinimum(-5);
   dummy4__4->SetMaximum(16);
   dummy4__4->SetStats(0);
   dummy4__4->SetLineColor(0);
   dummy4__4->SetLineWidth(3);
   dummy4__4->GetXaxis()->SetTitle("m_{jj}^{gen} [GeV]");
   dummy4__4->GetXaxis()->SetBinLabel(1,"[1000,#infty)");
   dummy4__4->GetXaxis()->SetRange(1,2);
   dummy4__4->GetXaxis()->CenterTitle(true);
   dummy4__4->GetXaxis()->SetLabelFont(42);
   dummy4__4->GetXaxis()->SetLabelSize(0.1079545);
   dummy4__4->GetXaxis()->SetTitleSize(0.08304196);
   dummy4__4->GetXaxis()->SetTitleOffset(1.5);
   dummy4__4->GetXaxis()->SetTitleFont(42);
   dummy4__4->GetYaxis()->SetLabelFont(42);
   dummy4__4->GetYaxis()->SetLabelSize(0);
   dummy4__4->GetYaxis()->SetTitleSize(0);
   dummy4__4->GetYaxis()->SetTitleFont(42);
   dummy4__4->GetZaxis()->SetLabelFont(42);
   dummy4__4->GetZaxis()->SetTitleOffset(1);
   dummy4__4->GetZaxis()->SetTitleFont(42);
   dummy4__4->Draw("");
   
   Double_t Graph0_fx3010[1] = { 3 };
   Double_t Graph0_fy3010[1] = { 1 };
   Double_t Graph0_felx3010[1] = { 0.5 };
   Double_t Graph0_fely3010[1] = { 0.9788 };
   Double_t Graph0_fehx3010[1] = { 0.5 };
   Double_t Graph0_fehy3010[1] = { 1.0214 };
   grae = new TGraphAsymmErrors(1,Graph0_fx3010,Graph0_fy3010,Graph0_felx3010,Graph0_fehx3010,Graph0_fely3010,Graph0_fehy3010);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(94);
   grae->SetFillStyle(3003);
   grae->SetLineColor(94);
   grae->SetMarkerColor(94);
   
   TH1F *Graph_Graph03010 = new TH1F("Graph_Graph03010","Graph",100,2.4,3.6);
   Graph_Graph03010->SetMinimum(0.01908);
   Graph_Graph03010->SetMaximum(2.22142);
   Graph_Graph03010->SetDirectory(nullptr);
   Graph_Graph03010->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph03010->SetLineColor(ci);
   Graph_Graph03010->GetXaxis()->SetLabelFont(42);
   Graph_Graph03010->GetXaxis()->SetTitleOffset(1);
   Graph_Graph03010->GetXaxis()->SetTitleFont(42);
   Graph_Graph03010->GetYaxis()->SetLabelFont(42);
   Graph_Graph03010->GetYaxis()->SetTitleFont(42);
   Graph_Graph03010->GetZaxis()->SetLabelFont(42);
   Graph_Graph03010->GetZaxis()->SetTitleOffset(1);
   Graph_Graph03010->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph03010);
   
   grae->Draw("2");
   
   Double_t Graph0_fx3011[1] = { 3 };
   Double_t Graph0_fy3011[1] = { 1 };
   Double_t Graph0_felx3011[1] = { 0.5 };
   Double_t Graph0_fely3011[1] = { 0.9788 };
   Double_t Graph0_fehx3011[1] = { 0.5 };
   Double_t Graph0_fehy3011[1] = { 1.0214 };
   grae = new TGraphAsymmErrors(1,Graph0_fx3011,Graph0_fy3011,Graph0_felx3011,Graph0_fehx3011,Graph0_fely3011,Graph0_fehy3011);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(94);
   grae->SetFillStyle(3003);
   grae->SetLineColor(94);
   grae->SetMarkerColor(94);
   
   TH1F *Graph_Graph03011 = new TH1F("Graph_Graph03011","Graph",100,2.4,3.6);
   Graph_Graph03011->SetMinimum(0.01908);
   Graph_Graph03011->SetMaximum(2.22142);
   Graph_Graph03011->SetDirectory(nullptr);
   Graph_Graph03011->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph03011->SetLineColor(ci);
   Graph_Graph03011->GetXaxis()->SetLabelFont(42);
   Graph_Graph03011->GetXaxis()->SetTitleOffset(1);
   Graph_Graph03011->GetXaxis()->SetTitleFont(42);
   Graph_Graph03011->GetYaxis()->SetLabelFont(42);
   Graph_Graph03011->GetYaxis()->SetTitleFont(42);
   Graph_Graph03011->GetZaxis()->SetLabelFont(42);
   Graph_Graph03011->GetZaxis()->SetTitleOffset(1);
   Graph_Graph03011->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph03011);
   
   grae->Draw("pe");
   
   Double_t Graph1_fx3012[1] = { 3 };
   Double_t Graph1_fy3012[1] = { 1.596 };
   Double_t Graph1_felx3012[1] = { 0 };
   Double_t Graph1_fely3012[1] = { 0.794 };
   Double_t Graph1_fehx3012[1] = { 0 };
   Double_t Graph1_fehy3012[1] = { 1.169 };
   grae = new TGraphAsymmErrors(1,Graph1_fx3012,Graph1_fy3012,Graph1_felx3012,Graph1_fehx3012,Graph1_fely3012,Graph1_fehy3012);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph13012 = new TH1F("Graph_Graph13012","Graph",100,2.9,4.1);
   Graph_Graph13012->SetMinimum(0.6057);
   Graph_Graph13012->SetMaximum(2.9613);
   Graph_Graph13012->SetDirectory(nullptr);
   Graph_Graph13012->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph13012->SetLineColor(ci);
   Graph_Graph13012->GetXaxis()->SetLabelFont(42);
   Graph_Graph13012->GetXaxis()->SetTitleOffset(1);
   Graph_Graph13012->GetXaxis()->SetTitleFont(42);
   Graph_Graph13012->GetYaxis()->SetLabelFont(42);
   Graph_Graph13012->GetYaxis()->SetTitleFont(42);
   Graph_Graph13012->GetZaxis()->SetLabelFont(42);
   Graph_Graph13012->GetZaxis()->SetTitleOffset(1);
   Graph_Graph13012->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph13012);
   
   grae->Draw("p");
   pad4__3->Modified();
   stxs->cd();
   stxs->Modified();
   stxs->SetSelected(stxs);
}
