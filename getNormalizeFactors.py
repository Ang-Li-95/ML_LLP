import ROOT
import sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--MC', help="background MC file")
parser.add_argument('--data', help="data file")
parser.add_argument('--tkdir', default='mfvTrackHistosExtraLooseMET',
                     help="directory in root file that we want to get factors from")
parser.add_argument('--vtxdir', default='mfvVertexHistosExtraLooseMET',
                     help="directory in root file that we want to get factors from")
options = parser.parse_args()
fdata = ROOT.TFile(options.data)
fmc = ROOT.TFile(options.MC)
tk_vars = ['h_leading_track_pt', 'h_leading_track_eta', 'h_leading_track_phi', 'h_leading_track_dxybs', 'h_leading_track_dxybs_sig', 'h_leading_track_dz', 'h_leading_track_dz_sig']
vtx_vars = ['h_sv_all_ntracks', 'h_sv_all_bsbs2ddist', 'h_sv_all_bs2derr']
tk_factors_data = []
tk_factors_mc = []
vtx_factors_data = []
vtx_factors_mc = []
# for tracks
for v in tk_vars:
  hdata = fdata.Get(options.tkdir+'/'+v)
  hmc = fmc.Get(options.tkdir+'/'+v)
  tk_factors_data.append([hdata.GetMean(), hdata.GetRMS()])
  tk_factors_mc.append([hmc.GetMean(), hmc.GetRMS()])
for v in vtx_vars:
  hdata = fdata.Get(options.vtxdir+'/'+v)
  hmc = fmc.Get(options.vtxdir+'/'+v)
  vtx_factors_data.append([hdata.GetMean(), hdata.GetRMS()])
  vtx_factors_mc.append([hmc.GetMean(), hmc.GetRMS()])

print("--------TRACKS----------")
print("### MC ###")
print(*tk_factors_mc, sep=',\n')
print("### data ###")
print(*tk_factors_data, sep=',\n')
print("--------VERTEX----------")
print("### MC ###")
print(*vtx_factors_mc, sep=',\n')
print("### data ###")
print(*vtx_factors_data, sep=',\n')
