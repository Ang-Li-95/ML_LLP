from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uproot
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import awkward as ak
from utils.parameterSet import *
normalize_factors_tk = []
normalize_factors_vtx = []
fns_xsec_UL = {
    "qcdht0100": 2.366e+07,
    "qcdht0200": 1.550e+06,
    "qcdht0300": 3.245e+05,
    "qcdht0500": 3.032e+04,
    "qcdht0700": 6.430e+03,
    "qcdht1000": 1.118e+03,
    "qcdht1500": 1.080e+02,
    "qcdht2000": 2.201e+01,
    "wjetstolnuht0100": 1.255e+03,
    "wjetstolnuht0200": 3.364e+02,
    "wjetstolnuht0400": 4.526e+01,
    "wjetstolnuht0600": 1.099e+01,
    "wjetstolnuht0800": 4.924,
    "wjetstolnuht1200": 1.157,
    "wjetstolnuht2500": 2.623e-02,
    "wjetstolnuht0400sum": 4.526e+01,
    "wjetstolnuht0600sum": 1.099e+01,
    "wjetstolnuht0800sum": 4.924,
    "wjetstolnuht1200sum": 1.157,
    "wjetstolnuht2500sum": 2.623e-02,
    "zjetstonunuht0100": 2.659e+02,
    "zjetstonunuht0200": 7.297e+01,
    "zjetstonunuht0400": 9.917,
    "zjetstonunuht0600": 2.410,
    "zjetstonunuht0800": 1.080,
    "zjetstonunuht1200": 2.521e-01,
    "zjetstonunuht2500": 5.633e-03,
    "ww": 7.587e+01,
    "wz": 2.756e+01,
    "zz": 1.214e+01,
    "ttbar": 7.572e+02,
    "st_tchan_antitop": 6.793e+01,
    "st_tchan_top": 1.134e+02,
    "st_tw_antitop": 3.251e+01,
    "st_tw_top": 3.245e+01,
    #"mfv_splitSUSY_tau000000100um_M2000_1800": 1e-03,
    #"mfv_splitSUSY_tau000000100um_M2000_1900": 1e-03,
    #'mfv_splitSUSY_tau000000100um_M2400_2300': 1e-03,
    #"mfv_splitSUSY_tau000000300um_M2000_1800": 1e-03,
    #"mfv_splitSUSY_tau000000300um_M2000_1900": 1e-03,
    #'mfv_splitSUSY_tau000000300um_M2400_2300': 1e-03,
    #"mfv_splitSUSY_tau000001000um_M2000_1800": 1e-03,
    #"mfv_splitSUSY_tau000001000um_M2000_1900": 1e-03,
    #'mfv_splitSUSY_tau000001000um_M2400_2300': 1e-03,
    #'mfv_splitSUSY_tau000001000um_M1200_1100': 1e-03,
    #'mfv_splitSUSY_tau000001000um_M1400_1200': 1e-03,
    #"mfv_splitSUSY_tau000010000um_M2000_1800": 1e-03,
    #"mfv_splitSUSY_tau000010000um_M2000_1900": 1e-03,
    #'mfv_splitSUSY_tau000010000um_M2400_2300': 1e-03,
    #'mfv_splitSUSY_tau000010000um_M1200_1100': 1e-03,
    #'mfv_splitSUSY_tau000010000um_M1400_1200': 1e-03,
    #'mfv_splitSUSY_tau000100000um_M2000_1800': 1e-03,
    #'mfv_splitSUSY_tau000100000um_M2000_1900': 1e-03,
    #'mfv_splitSUSY_tau001000000um_M2000_1800': 1e-03,
    #'mfv_splitSUSY_tau001000000um_M2000_1900': 1e-03,
    #'mfv_splitSUSY_tau010000000um_M2000_1800': 1e-03,
    #'mfv_splitSUSY_tau010000000um_M2000_1900': 1e-03,
    "mfv_splitSUSY_tau000000100um_M1400_1200" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M1400_1300" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M1600_1400" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M1600_1500" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M1800_1600" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M1800_1700" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M2000_1800" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M2000_1900" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M2200_2000" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M2200_2100" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M2400_2200" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M2400_2300" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M2600_2400" : 1e-03,
    "mfv_splitSUSY_tau000000100um_M2600_2500" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M1400_1200" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M1400_1300" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M1600_1400" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M1600_1500" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M1800_1600" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M1800_1700" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M2000_1800" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M2000_1900" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M2200_2000" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M2200_2100" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M2400_2200" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M2400_2300" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M2600_2400" : 1e-03,
    "mfv_splitSUSY_tau000000300um_M2600_2500" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M1400_1200" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M1400_1300" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M1600_1400" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M1600_1500" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M1800_1600" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M1800_1700" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M2000_1800" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M2000_1900" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M2200_2000" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M2200_2100" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M2400_2200" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M2400_2300" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M2600_2400" : 1e-03,
    "mfv_splitSUSY_tau000001000um_M2600_2500" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M1400_1200" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M1400_1300" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M1600_1400" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M1600_1500" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M1800_1600" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M1800_1700" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M2000_1800" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M2000_1900" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M2200_2000" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M2200_2100" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M2400_2200" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M2400_2300" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M2600_2400" : 1e-03,
    "mfv_splitSUSY_tau000003000um_M2600_2500" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M1400_1200" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M1400_1300" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M1600_1400" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M1600_1500" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M1800_1600" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M1800_1700" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M2000_1800" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M2000_1900" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M2200_2000" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M2200_2100" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M2400_2200" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M2400_2300" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M2600_2400" : 1e-03,
    "mfv_splitSUSY_tau000010000um_M2600_2500" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M1400_1200" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M1400_1300" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M1600_1400" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M1600_1500" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M1800_1600" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M1800_1700" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M2000_1800" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M2000_1900" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M2200_2000" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M2200_2100" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M2400_2200" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M2400_2300" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M2600_2400" : 1e-03,
    "mfv_splitSUSY_tau000030000um_M2600_2500" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M1400_1200" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M1400_1300" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M1600_1400" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M1600_1500" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M1800_1600" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M1800_1700" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M2000_1800" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M2000_1900" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M2200_2000" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M2200_2100" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M2400_2200" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M2400_2300" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M2600_2400" : 1e-03,
    "mfv_splitSUSY_tau000100000um_M2600_2500" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M1400_1200" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M1400_1300" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M1600_1400" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M1600_1500" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M1800_1600" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M1800_1700" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M2000_1800" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M2000_1900" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M2200_2000" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M2200_2100" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M2400_2200" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M2400_2300" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M2600_2400" : 1e-03,
    "mfv_splitSUSY_tau000300000um_M2600_2500" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M1400_1200" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M1400_1300" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M1600_1400" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M1600_1500" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M1800_1600" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M1800_1700" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M2000_1800" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M2000_1900" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M2200_2000" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M2200_2100" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M2400_2200" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M2400_2300" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M2600_2400" : 1e-03,
    "mfv_splitSUSY_tau001000000um_M2600_2500" : 1e-03,
    'WminusHToSSTobbbb_tau100um_M55': 1e-03,
    'WminusHToSSTobbbb_tau10mm_M55': 1e-03,
    'WminusHToSSTobbbb_tau1mm_M55': 1e-03,
    'WminusHToSSTodddd_tau100um_M55': 1e-03,
    'WminusHToSSTodddd_tau10mm_M55': 1e-03,
    'WminusHToSSTodddd_tau1mm_M55': 1e-03,
    'mfv_neu_tau000100um_M0400' : 1e-03,
    'mfv_neu_tau001000um_M0400' : 1e-03,
    'mfv_neu_tau010000um_M0400' : 1e-03,
    'mfv_neu_tau000100um_M1600' : 1e-03,
    'mfv_neu_tau001000um_M1600' : 1e-03,
    'mfv_neu_tau010000um_M1600' : 1e-03,
    'mfv_splitSUSY_tau000010000um_M2000_1950': 1e-03,
    'mfv_splitSUSY_tau000010000um_M2000_1980': 1e-03,
    'hists': 1,
    'mltree': 1,
}

fns_xsec_EOY = {
    "qcdht0200": 1.547e+06,
    "qcdht0300": 3.226E+05,
    "qcdht0500": 2.998E+04,
    "qcdht0700": 6.351E+03,
    "qcdht1000": 1.096E+03,
    "qcdht1500": 99.0,
    "qcdht2000": 20.2,
    "wjetstolnusum": 5.28E+04,
    #"wjetstolnu": 5.28E+04,
    #"wjetstolnuext": 5.28E+04,
    "zjetstonunuht0100": 302.8,
    "zjetstonunuht0200": 92.59,
    "zjetstonunuht0400": 13.18,
    "zjetstonunuht0600": 3.257,
    "zjetstonunuht0800": 1.49,
    "zjetstonunuht1200": 0.3419,
    "zjetstonunuht2500": 0.005146,
    "ttbar": 832,
    "ttbarht0600": 1.821,
    "ttbarht0800": 0.7532,
    "ttbarht1200": 0.1316,
    "ttbarht2500": 0.001407,
    "mfv_splitSUSY_tau000000000um_M2000_1800": 1e-03,
    "mfv_splitSUSY_tau000000000um_M2000_1900": 1e-03,
    'mfv_splitSUSY_tau000000000um_M2400_2300': 1e-03,
    "mfv_splitSUSY_tau000000300um_M2000_1800": 1e-03,
    "mfv_splitSUSY_tau000000300um_M2000_1900": 1e-03,
    'mfv_splitSUSY_tau000000300um_M2400_2300': 1e-03,
    "mfv_splitSUSY_tau000001000um_M2000_1800": 1e-03,
    "mfv_splitSUSY_tau000001000um_M2000_1900": 1e-03,
    'mfv_splitSUSY_tau000001000um_M2400_2300': 1e-03,
    'mfv_splitSUSY_tau000001000um_M1200_1100': 1e-03,
    'mfv_splitSUSY_tau000001000um_M1400_1200': 1e-03,
    "mfv_splitSUSY_tau000010000um_M2000_1800": 1e-03,
    "mfv_splitSUSY_tau000010000um_M2000_1900": 1e-03,
    'mfv_splitSUSY_tau000010000um_M2400_2300': 1e-03,
    'mfv_splitSUSY_tau000010000um_M1200_1100': 1e-03,
    'mfv_splitSUSY_tau000010000um_M1400_1200': 1e-03,
}

def GetXsec(sample):
    nsample = sample[:sample.find('_',-7,-1)]
    if isUL:
        xsecs = fns_xsec_UL
    else:
        xsecs = fns_xsec_EOY
    if nsample not in xsecs:
        raise ValueError("Sample {} not available!!!".format(nsample))
    return xsecs[nsample]

def GetXsecList(fns):
    if isUL:
        fns_xsec = fns_xsec_UL
    else:
        fns_xsec = fns_xsec_EOY
    xsecs = []
    for fn in fns:
        nfn = fn[:fn.find('_',-7,-1)]
        #nfn = fn
        assert(nfn in fns_xsec)
        xsecs.append(fns_xsec[nfn])
    return xsecs

def GetNormWeight(fns, fn_dir, isData, int_lumi=1):
    if isData:
      return [1]*len(fns)
    xsecs = GetXsecList(fns)
    nevents = GetNevtsList(fns, fn_dir)
    assert(len(xsecs)==len(nevents))
    normweights = []
    for i in range(len(xsecs)):
        normweights.append((xsecs[i]*int_lumi)/nevents[i])
    return normweights

def GetNevts(f):
    nevt = f['mfvWeight/h_sums'].values[f['mfvWeight/h_sums'].xlabels.index('sum_nevents_total')]
    return nevt

def GetNevtsList(fns, fn_dir):
    nevents = []
    for fn in fns:
        f = uproot.open(fn_dir+fn+'.root')
        nevt = GetNevts(f)
        nevents.append(nevt)
        del f
    return nevents

def GetLoadFactor(fn,f,lumi):
    '''
    To make the fraction of background similar actual case (xsec normalization), 
    calculate the factor so that (Number_selected_events)*LoadFactor 
    represent the number of selected events from given sample at given luminosity
    '''
    nevt = GetNevts(f) # total number of events before selection
    xsec = GetXsec(fn)
    return xsec*lumi/nevt

def GetDataAndLabel(fns, split, isSignal, cut="", lumi=200000, maxsigevt=1000, isTrainData=False):
    assert(not (isSignal and isTrainData))
    tk_train = []
    tk_val = []
    tk_test = []
    vtx_train = []
    vtx_val = []
    vtx_test = []
    for fn in fns:
        print("Loading sample {}...".format(fn))
        f = uproot.open(fndir+fn+'.root')
        if not isTrainData:
          loadfactor = GetLoadFactor(fn, f, lumi)
        f = f["mfvJetTreer/tree_DV"]
        if len(f['evt'].array())==0:
          print( "no events!!!")
          continue
        variables = ['tk_pt', 'tk_eta', 'tk_phi', 'tk_dxybs','tk_dxybs_sig','tk_dz','tk_dz_sig','met_pt','vtx_ntk', 'vtx_dBV', 'vtx_dBVerr', 'metnomu_pt']
        matrix = f.arrays(variables, namedecode="utf-8")
        # apply cuts
        evt_select = (matrix['metnomu_pt']>80) & (matrix['metnomu_pt']<=200) & (matrix['vtx_ntk']>2) & (matrix['vtx_dBVerr']<0.0025)
        # blind signal region
        if isTrainData:
          evt_select = evt_select & (matrix['vtx_ntk']<5) 

        for v in matrix:
          matrix[v] = matrix[v][evt_select]
        if len(matrix['met_pt'])==0:
          print("no event after selection")
          continue
        
        # define the max number of events to pick and the number of train/val/test events to use
        train_idx = -1
        val_idx = -1
        nevt_total = len(matrix['met_pt'])
        if not isTrainData:
            nevt = int(loadfactor*nevt_total)
            if nevt>nevt_total:
                nevt = nevt_total
            if isSignal:
                nevt = min(nevt_total, maxsigevt)
        if isTrainData:
            nevt = nevt_total

        print("  {} events in file, {} are used".format(nevt_total, nevt))
            
        train_idx = int(nevt*split[0])
        val_idx = int(nevt*(split[0]+split[1]))
        
        # train
        m_tk = np.array([matrix[v][:train_idx] for v in mlvar_tk])
        m_vtx = np.array([matrix[v][:train_idx] for v in mlvar_vtx]).T
        #m_vtx = np.reshape(m_vtx, m_vtx.shape+(1,))
        m = zeropadding(m_tk, No)
        if len(m)>0:
            tk_train.append(m)
            vtx_train.append(m_vtx)
        
        # val
        m_tk = np.array([matrix[v][train_idx+1:val_idx] for v in mlvar_tk])
        m_vtx = np.array([matrix[v][train_idx+1:val_idx] for v in mlvar_vtx]).T
        #m_vtx = np.reshape(m_vtx, m_vtx.shape+(1,))
        m = zeropadding(m_tk, No)
        if len(m)>0:
            tk_val.append(m)
            vtx_val.append(m_vtx)

        # test
        m_tk = np.array([matrix[v][val_idx+1:nevt] for v in mlvar_tk])
        m_vtx = np.array([matrix[v][val_idx+1:nevt] for v in mlvar_vtx]).T
        #m_vtx = np.reshape(m_vtx, m_vtx.shape+(1,))
        m = zeropadding(m_tk, No)
        if len(m)>0:
            tk_test.append(m)
            vtx_test.append(m_vtx)

    tk_train = np.concatenate(tk_train)
    vtx_train = np.concatenate(vtx_train)
    tk_val = np.concatenate(tk_val)
    vtx_val = np.concatenate(vtx_val)
    tk_test = np.concatenate(tk_test)
    vtx_test = np.concatenate(vtx_test)

    if not isSignal:
        label_train = np.zeros((vtx_train.shape[0], 1))
        label_val = np.zeros((vtx_val.shape[0], 1))
        label_test = np.zeros((vtx_test.shape[0], 1))
    elif isSignal:
        label_train = np.ones((vtx_train.shape[0], 1))
        label_val = np.ones((vtx_val.shape[0], 1))
        label_test = np.ones((vtx_test.shape[0], 1))
        
    return (tk_train, vtx_train, label_train), (tk_val, vtx_val, label_val), (tk_test, vtx_test, label_test)

def removeNaN(datasets):
    idx1 = ~(pd.isnull(datasets[0])).any(axis=(1,2))
    idx2 = ~(pd.isnull(datasets[1])).any(axis=(1))
    idx = np.logical_and(idx1,idx2)
    datasets_new = (datasets[0][idx], datasets[1][idx], datasets[2][idx])
    #datasets[0] = datasets[0][idx]
    #datasets[1] = datasets[1][idx]
    #datasets[2] = datasets[2][idx]
    return datasets_new

def importData(split,year,isMC,normalize=True,shuffle=True,maxsigevt=1000):
    '''
    import training/val/testing data from root file normalize, padding and shuffle if needed
    split: [train, val, test] fraction
    returns data_train/val/test, which are tuples, structure:
      (data, ntk, label, met, data_no_normalized)
    '''
    if not isMC:
      fn_d = fns_data[year]
      train_dat, val_dat, test_dat = GetDataAndLabel(fn_d, split, False, maxsigevt=maxsigevt, isTrainData=True)
      train_dat = removeNaN(train_dat)
      val_dat = removeNaN(val_dat)
      test_dat = removeNaN(test_dat)
      print("Training REAL data: {0} events as background".format(len(train_dat[0])))
      nitems = len(train_dat)
      data_train = [None]*(nitems)
      data_val = [None]*(nitems)
      data_test = [None]*(nitems)
      for i in range(nitems):
          data_train[i] = np.concatenate([train_dat[i]])
          data_val[i] = np.concatenate([val_dat[i]])
          data_test[i] = np.concatenate([test_dat[i]])
    elif isMC:
      fn_s = fns_signal.copy()
      for fi in range(len(fn_s)):
        fn_s[fi] += year
      fn_b = fns_bkg.copy()
      for fi in range(len(fn_b)):
        fn_b[fi] += year
      train_sig, val_sig, test_sig = GetDataAndLabel(fn_s, split, True, maxsigevt=maxsigevt, isTrainData=False)
      train_bkg, val_bkg, test_bkg = GetDataAndLabel(fn_b, split, False, maxsigevt=maxsigevt, isTrainData=False)
      train_sig = removeNaN(train_sig)
      val_sig = removeNaN(val_sig)
      test_sig = removeNaN(test_sig)
      train_bkg = removeNaN(train_bkg)
      val_bkg = removeNaN(val_bkg)
      test_bkg = removeNaN(test_bkg)
      #sig_bkg_weight = float(len(train_bkg[0]))/len(train_sig[0])
      print("Training MC data: {0} signals {1} backgrounds".format(len(train_sig[0]), len(train_bkg[0])))
      nitems = len(train_sig)
      data_train = [None]*(nitems)
      data_val = [None]*(nitems)
      data_test = [None]*(nitems)
      for i in range(nitems):
          data_train[i] = np.concatenate([train_sig[i], train_bkg[i]])
          data_val[i] = np.concatenate([val_sig[i], val_bkg[i]])
          data_test[i] = np.concatenate([test_sig[i], test_bkg[i]])
    
    
    if shuffle:
        shuffler = np.random.permutation(len(data_train[0]))
        for i in range(nitems):
            data_train[i] = data_train[i][shuffler]
        
        shuffler = np.random.permutation(len(data_val[0]))
        for i in range(nitems):
            data_val[i] = data_val[i][shuffler]
        
        shuffler = np.random.permutation(len(data_test[0]))
        for i in range(nitems):
            data_test[i] = data_test[i][shuffler]

    if normalize:
      for i in range(2):
        data_train[i] = normalizedata(data_train[i], year, isMC)
        data_val[i] = normalizedata(data_val[i], year, isMC)
        data_test[i] = normalizedata(data_test[i], year, isMC)
    
    if isMC:
      num_sig = len(train_sig[0])
      num_bkg = len(train_bkg[0])
    else:
      num_sig = 0
      num_bkg = len(train_dat[0])

    return data_train, data_val, data_test, (num_sig, num_bkg)

def zeropadding(matrix, l):
    '''
    make the number of object the same for every event, zero padding those
    df: np.array of data
    l: expected length of each event (# objects)
    '''
    m_mod = []
    for i in range(matrix.shape[1]):
        # transfer df to matrix for each event
        m = np.array([matrix[:,i][v] for v in range(len(mlvar_tk))])
        sortedidx = np.argsort(m[0,:])[::-1] # this step orders pt of tracks
        m = m[:,sortedidx]
        if m.shape[1]<l:
            idx_mod = l-m.shape[1]
            pad = np.zeros((m.shape[0],idx_mod))
            m_mod.append(np.concatenate((m,pad), axis=1))
        else:
            m_mod.append(m[:,0:l])
    return np.array(m_mod)

def normalizedata(data, year, isMC):
  if isUL:
    if isMC:
      normalize_factors_tk = normalize_factors_tk_UL[year+"MC"]
      normalize_factors_vtx = normalize_factors_vtx_UL[year+"MC"]
    else:
      normalize_factors_tk = normalize_factors_tk_UL[year]
      normalize_factors_vtx = normalize_factors_vtx_UL[year]
  else:
    normalize_factors_tk = normalize_factors_tk_EOY
    normalize_factors_vtx = normalize_factors_vtx_EOY

  if len(data.shape)==3:
    n_features_data = Ds
    
    for i in range(n_features_data):
        mean = normalize_factors_tk[i][0]
        stddev = normalize_factors_tk[i][1]
        #print("normalize {} with mean {} stddev {}".format(mlvar_tk[i],mean,stddev))
        data[:,i,:][data[:,i,:]!=0] = (data[:,i,:][data[:,i,:]!=0]-mean)*(1.0/(stddev))
  elif len(data.shape)==2:
    for i in range(Dv_ori):
        mean = normalize_factors_vtx[i][0]
        stddev = normalize_factors_vtx[i][1]
        #print("normalize {} with mean {} stddev {}".format(mlvar_vtx[i],mean,stddev))
        data[:,i] = (data[:,i]-mean)*(1.0/(stddev))
  return data

def getRmatrix(mini_batch_num=100):
    # Set Rr_data, Rs_data, Ra_data and X_data
    Rr_data=np.zeros((mini_batch_num,No,Nr),dtype=float)
    Rs_data=np.zeros((mini_batch_num,No,Nr),dtype=float)
    Ra_data=np.ones((mini_batch_num,Dr,Nr),dtype=float)
    cnt=0
    for i in range(No):
        for j in range(No):
            if(i!=j):
                Rr_data[:,i,cnt]=1.0
                Rs_data[:,j,cnt]=1.0
                cnt+=1
    return Rr_data, Rs_data, Ra_data

def getRmatrix_dR2(jets):
    n_evt = len(jets)
    # Set Rr_data, Rs_data, Ra_data and X_data
    Rr_data=np.zeros((n_evt,No,Nr),dtype=float)
    Rs_data=np.zeros((n_evt,No,Nr),dtype=float)
    Ra_data=np.ones((n_evt,Dr,Nr),dtype=float)
    cnt=0
    for i in range(No):
        for j in range(No):
            if(i!=j):
                # use mask to get rid the padded non-existing jets
                mask = np.multiply(jets[:,0,i],jets[:,0,j])==0
                dR2 = np.sum(np.square(jets[:,0:2,i]-jets[:,0:2,j]),axis=1)
                dR2[mask] = -1
                dR2_inverse = (1e-03)/dR2
                dR2_inverse[mask] = 0
                Rr_data[:,i,cnt]=dR2_inverse
                Rs_data[:,j,cnt]=dR2_inverse
                cnt+=1
    R_sum = np.sum(Rr_data,axis=(1,2))
    #Rr_data = Rr_data/R_sum
    #Rs_data = Rs_data/R_sum
    for i in range(len(R_sum)):
        if R_sum[i]==0:
            continue
        Rr_data[i] = Rr_data[i]/R_sum[i]
        Rs_data[i] = Rs_data[i]/R_sum[i]
    return Rr_data, Rs_data, Ra_data

