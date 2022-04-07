#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import argparse
import sys
import copy
import uproot
import ROOT
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import numpy as np
from array import array
from utils.parameterSet import *
from utils.utilities import *
from utils.plot_setting import *

METNoMu_avai = True
B_info = True
doSignal = False
doBackground = False
doData = True
#fndir_plot = '/uscms/home/ali/nobackup/LLP/crabdir/MLTreeULV5_keeptkMETm/'
fndir_plot = '/uscms/home/ali/nobackup/LLP/crabdir/MLTreeULV11METm/'
#m_path = './model_1119_ntk_1/'
m_path = './model_0406_ntk_ULV11_3/'
save_plot_path='./UL0406_ULV11_ntk_3_1/'
if not os.path.exists(save_plot_path):
      os.makedirs(save_plot_path)

print("processing {} with UL mode {}".format(fndir_plot, isUL))

variables = ['evt', 'weight', 'met_pt', 'met_phi', 'nsv', 
             'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 
             'tk_pt', 'tk_eta', 'tk_phi', 
             'tk_dxybs', 'tk_dxybs_sig', 'tk_dxybs_err', 'tk_dz', 'tk_dz_sig', 'tk_dz_err', 
             'vtx_ntk', 'vtx_dBV', 'vtx_dBVerr', 'vtx_mass_track', 'vtx_mass_jet', 'vtx_mass_trackjet',
             'vtx_njets', 'vtx_nbjets_loose', 'vtx_nbjets_medium', 'vtx_nbjets_tight',
             'vtx_nbtks_loose', 'vtx_nbtks_medium', 'vtx_nbtks_tight',
             'vtx_tk_pt', 'vtx_tk_eta', 'vtx_tk_phi',
             'vtx_tk_dxy', 'vtx_tk_dxy_err', 'vtx_tk_nsigmadxy', 'vtx_tk_dz', 'vtx_tk_dz_err', 'vtx_tk_nsigmadz']
vars_plot = [
  'weight','met_pt','met_phi','nsv','MLScore',
  'jet_pt', 'jet_eta', 'jet_phi',
  'vtx_ntk','vtx_dBV','vtx_dBVerr', 'vtx_mass_track', 'vtx_mass_jet', 'vtx_mass_trackjet',
  'vtx_njets', 'vtx_nbjets_loose', 'vtx_nbjets_medium', 'vtx_nbjets_tight',
  'vtx_nbtks_loose', 'vtx_nbtks_medium', 'vtx_nbtks_tight',
  'tk_pt', 'tk_eta', 'tk_phi', 'tk_dxybs', 'tk_dxybs_sig', 'tk_dxybs_err', 'tk_dz', 'tk_dz_sig', 'tk_dz_err',
  'vtx_tk_pt','vtx_tk_eta','vtx_tk_phi', 'vtx_tk_dxy', 'vtx_tk_dxy_err', 'vtx_tk_nsigmadxy', 'vtx_tk_dz', 'vtx_tk_dz_err', 'vtx_tk_nsigmadz',
            ]
if METNoMu_avai:
  variables += ['metnomu_pt', 'metnomu_phi']
  vars_plot += ['metnomu_pt', 'metnomu_phi']
if B_info:
  variables += ['nbtag_jet', 'n_gen_bquarks', 'jet_ntrack', 'jet_btag', 'jet_flavor', 'gen_bquarks_pt', 'gen_bquarks_eta', 'gen_bquarks_phi']
  vars_plot += ['nbtag_jet', 'n_gen_bquarks', 'jet_ntrack', 'jet_btag', 'jet_flavor', 'gen_bquarks_pt', 'gen_bquarks_eta', 'gen_bquarks_phi']

def GetData(fns, variables, cut=""):
    ML_inputs_tk = []
    ML_inputs_vtx = []
    phys_variables = []
    for fn in fns:
        #print(fn)
        print("opening {}...".format(fndir_plot+fn+'.root'))
        f = uproot.open(fndir_plot+fn+'.root')
        f = f["mfvJetTreer/tree_DV"]
        if len(f['evt'].array())==0:
          print( "no events!!!")
          continue
        phys = f.arrays(variables, namedecode="utf-8")
        del f
        evt_select = (phys['vtx_ntk']>0) & (phys['metnomu_pt']>=200) & (phys['vtx_dBVerr']<0.0025)
        #evt_select = (phys['vtx_ntk']>0) & (phys['metnomu_pt']>=100) & (phys['metnomu_pt']<200) & (phys['vtx_dBVerr']<0.0025)
        for v in phys:
          phys[v] = np.array(phys[v][evt_select])
        if len(phys['evt'])==0:
            print("no events after selection!")
            continue
        m_tk = np.array([phys[v] for v in mlvar_tk])
        m_vtx = np.array([phys[v] for v in mlvar_vtx]).T
        m_tk = zeropadding(m_tk, No)
        m_tk = normalizedata(m_tk)
        m_vtx = normalizedata(m_vtx)
        ML_inputs_tk.append(m_tk)
        ML_inputs_vtx.append(m_vtx)
        phys_variables.append(phys)
        
    return ML_inputs_tk, ML_inputs_vtx, phys_variables

def calcMLscore(ML_inputs_tk, ML_inputs_vtx, model_path='./', model_name="test_model.meta"):
    batch_size=4096
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path+model_name)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        MLscores = []
        for iML in range(len(ML_inputs_tk)):
            ML_input_tk = ML_inputs_tk[iML]
            ML_input_vtx = ML_inputs_vtx[iML]
            evt=0
            outputscore = []
            while evt<len(ML_input_tk):
                if evt+batch_size <= len(ML_input_tk):
                    batch_input_tk = ML_input_tk[evt:evt+batch_size]
                    #batch_input_vtx = ML_input_vtx[evt:evt+batch_size]
                else:
                    batch_input_tk = ML_input_tk[evt:]
                    #batch_input_vtx = ML_input_vtx[evt:]
                evt += batch_size
                Rr, Rs, Ra = getRmatrix(len(batch_input_tk))
                ML_output = sess.run(['INscore:0'],feed_dict={'O:0':batch_input_tk,'Rr:0':Rr,'Rs:0':Rs,'Ra:0':Ra})
                outputscore.append(ML_output[0])
            outputscore = np.concatenate(outputscore)
            MLscores.append(outputscore)
    return MLscores

def makehist(data,weight,var,apply_weight=True):
    assert(var in plot_setting)
    assert(var in plot_vars_titles)
    label = plot_vars_titles[var]
    setting = plot_setting[var]
    if 'binlist' in setting:
      h = ROOT.TH1F(label[0],";".join(label), len(setting['binlist'])-1, setting['binlist'])
    else:
      h = ROOT.TH1F(label[0],";".join(label),setting['bins'],setting['range'][0],setting['range'][1])
    for i in range(len(data)):
      if apply_weight:
        h.Fill(data[i],weight[i])
      else:
        h.Fill(data[i])
    return h

def make2dhist(datax,datay,weight,varx,vary):
    assert(len(datax)==len(datay))
    assert(varx in plot_setting)
    assert(varx in plot_vars_titles)
    assert(vary in plot_setting)
    assert(vary in plot_vars_titles)
    labelx = plot_vars_titles[varx][1]
    labely = plot_vars_titles[vary][1]
    settingx = plot_setting[varx]
    settingy = plot_setting[vary]
    h = ROOT.TH2F("h_"+varx+"_"+vary,";"+labelx+";"+labely,settingx['bins'],settingx['range'][0],settingx['range'][1],settingy['bins'],settingy['range'][0],settingy['range'][1])
    for i in range(len(datax)):
      h.Fill(datax[i],datay[i],weight[i])
    return h

def plotcategory(f,dirname,vars_name,data,weight):
    f.cd()
    f.mkdir(dirname)
    f.cd(dirname)
    #hists = []
    for v in vars_name:
      if v=="weight":
        hist = makehist(data[v], weight[v], v, apply_weight=False)
      else:
        hist = makehist(data[v], weight[v], v)
      hist.Write()
    hist2d = make2dhist(data['vtx_dBV'],data['vtx_dBVerr'],weight['vtx_dBV'],'vtx_dBV','vtx_dBVerr')
    ntk_ML = make2dhist(data['vtx_ntk'],data['MLScore'],weight['vtx_ntk'],'vtx_ntk','MLScore')
    dbv_ML = make2dhist(data['vtx_dBV'],data['MLScore'],weight['vtx_dBV'],'vtx_dBV','MLScore')
    dbverr_ML = make2dhist(data['vtx_dBVerr'],data['MLScore'],weight['vtx_dBVerr'],'vtx_dBVerr','MLScore')
    hist2d.Write()
    ntk_ML.Write()
    dbv_ML.Write()
    dbverr_ML.Write()
    return

def MLoutput(signals, sig_fns, backgrounds, bkg_fns, isData):
    weights = GetNormWeight(bkg_fns, fndir_plot, isData,int_lumi=40610.0)
    MLoutput_bkg = []
    w_bkg = []
    for i in range(len(bkg_fns)):
        # w includes event-level weight and xsec normalizetion
        w_bkg.append(backgrounds[i]['weight']*weights[i])
        #w = backgrounds[i]['weight']*weights[i]
        MLoutput_bkg.append(backgrounds[i]['MLScore'])
    MLoutput_bkg = np.concatenate(MLoutput_bkg, axis=None)
    w_bkg = np.concatenate(w_bkg, axis=None)
    
    MLoutput_sig = []
    for i in range(len(sig_fns)):
        #MLoutput_sig.append(signals[i]['MLScore'])
        #MLoutput_sig = np.concatenate(MLoutput_sig, axis=None)
        #w_sig = np.ones(MLoutput_sig.shape)
        MLoutput_sig = signals[i]['MLScore']
        w_sig = signals[i]['weight']
        comparehists([MLoutput_sig, MLoutput_bkg], [w_sig, w_bkg], ['signal', 'background'], 
                     [sig_fns[i], 'MLscore', 'fraction of events'],True, '_sig_bkg_compare'+sig_fns[i], bins=50, range=(0,1))
    #compare.show()
    #return compare


def getPlotData(phys_vars, vars_name, idx, fns, isData):
    '''
    this function produced 1d arrays with weights for pyplot hist
    used for combine different source of background samples with different weight (can be event level)
    phys_vars: data of different variables
    vars_name: variables that to be combined
    idx: indices of events that is going to be used
    fns: root filenames of all those samples 
    '''

    weights = GetNormWeight(fns, fndir_plot, isData, int_lumi=40610.0)
    plot_w = {}
    plot_data = {}

    single_vars = []
    multi_vars = []
    nested_vars = []
    for v in vars_name:
      if v in plot_vars_single:
        single_vars.append(v)
      elif v in plot_vars_multi:
        multi_vars.append(v)
      elif v in plot_vars_nestedarray:
        nested_vars.append(v)
      else:
        raise ValueError("variable {} doesn't belong to any variable type!".format(v))
    
    for i in range(len(fns)):
        # w includes event-level weight and xsec normalizetion
        if len(phys_vars[i]['weight'][idx[i]])==0:
            continue
        w = phys_vars[i]['weight'][idx[i]]*weights[i]
        for v in single_vars:
            if v in plot_data:
                plot_data[v].append(phys_vars[i][v][idx[i]])
                plot_w[v].append(w)
            else:
                phys_vars[i][v][idx[i]].shape
                plot_data[v] = [phys_vars[i][v][idx[i]]]
                plot_w[v] = [w]
        
        for v in multi_vars:
            var = phys_vars[i][v][idx[i]]
            # make w the same dimension as variables
            w_extended = []
            for ievt in range(len(w)):
                w_extended.append([w[ievt]]*len(var[ievt]))
            var_flattern = np.concatenate(var)
            w_extended = np.concatenate(w_extended)
            if v in plot_data:
                plot_data[v].append(var_flattern)
                plot_w[v].append(w_extended)
            else:
                plot_data[v] = [var_flattern]
                plot_w[v] = [w_extended]
           
        for v in nested_vars:
            var = phys_vars[i][v][idx[i]]
            # flattern variable data and make w the same dimensions
            w_extended = []
            var_flattern = []
            for ievt in range(len(w)):
                var_ievt_array = np.concatenate(var[ievt], axis=None)
                w_extended.append([w[ievt]]*len(var_ievt_array))
                var_flattern.append(var_ievt_array)
            w_extended = np.concatenate(w_extended)
            var_flattern = np.concatenate(var_flattern, axis=None)
            if v in plot_data:
                plot_data[v].append(var_flattern)
                plot_w[v].append(w_extended)
            else:
                plot_data[v] = [var_flattern]
                plot_w[v] = [w_extended]

    for v in vars_name:
        if v in plot_data:
          plot_data[v] = np.concatenate(plot_data[v], axis=None)
          plot_w[v] = np.concatenate(plot_w[v], axis=None)
        else:
          plot_data[v] = np.array([])
          plot_w[v] = np.array([])
    
    return plot_data, plot_w

def makeplotfile(fns,newfn,isSignal,isData,MLscore_threshold_high=0.4,MLscore_threshold_low=0.4):
    print("ML cut high: {}  ---  ML cut low: {}".format(MLscore_threshold_high, MLscore_threshold_low))
    fnew = ROOT.TFile(save_plot_path+newfn+".root","RECREATE")
    #MLscore_threshold = 0.4
    #MLscore_threshold_high = 0.4
    #MLscore_threshold_low = 0.3
    ML_inputs_tk, ML_inputs_vtx, phys_vars = GetData(fns, variables)
    assert(len(fns)==len(ML_inputs_tk))
    assert(len(fns)==len(ML_inputs_vtx))
    assert(len(fns)==len(phys_vars))
    ML_outputs = calcMLscore(ML_inputs_tk, ML_inputs_vtx, model_path=m_path)
    for i in range(len(fns)):
        phys_vars[i]['MLScore'] = ML_outputs[i]
    idx_highML = []
    idx_lowML = []
    idx_all = []
    ntk_idx = {
      '3trk':[],
      '4trk':[],
      '5trk':[]
    } # 3-trk, 4-trk, >=5-trk
    for out in ML_outputs:
        highML = out>MLscore_threshold_high
        idx_highML.append(np.reshape(highML, len(highML)))
        lowML = out<=MLscore_threshold_low
        idx_lowML.append(np.reshape(lowML, len(lowML)))
        allML = np.array([True]*len(lowML))
        idx_all.append(allML)

    for i in range(len(fns)):
        max_ntk = phys_vars[i]['vtx_ntk']
        ntk_3 = max_ntk==3
        ntk_4 = max_ntk==4
        ntk_5 = max_ntk>=5
        ntk_idx['3trk'].append(np.reshape(ntk_3, len(ntk_3)))
        ntk_idx['4trk'].append(np.reshape(ntk_4, len(ntk_5)))
        ntk_idx['5trk'].append(np.reshape(ntk_5, len(ntk_5)))

    data_highML, weight_highML = getPlotData(phys_vars, vars_plot, idx_highML, fns, isData)
    data_lowML, weight_lowML = getPlotData(phys_vars, vars_plot, idx_lowML, fns, isData)
    data_all, weight_all = getPlotData(phys_vars, vars_plot, idx_all, fns, isData)
    plotcategory(fnew,"highML_inclusive",vars_plot,data_highML,weight_highML)
    plotcategory(fnew,"lowML_inclusive",vars_plot,data_lowML,weight_lowML)
    plotcategory(fnew,"allML_inclusive",vars_plot,data_all,weight_all)

    for intk in ntk_idx:
      pick_idx_incl = []
      pick_idx_high = []
      pick_idx_low = []
      for iidx in range(len(idx_highML)):
        pick_idx_incl.append(ntk_idx[intk][iidx])
        pick_idx_high.append(idx_highML[iidx] & ntk_idx[intk][iidx])
        pick_idx_low.append(idx_lowML[iidx] & ntk_idx[intk][iidx])
      data_ntk_incl, weight_ntk_incl = getPlotData(phys_vars, vars_plot, pick_idx_incl, fns, isData)
      data_highML, weight_highML = getPlotData(phys_vars, vars_plot, pick_idx_high, fns, isData)
      data_lowML, weight_lowML = getPlotData(phys_vars, vars_plot, pick_idx_low, fns, isData)
      plotcategory(fnew,"inclusive_"+intk,vars_plot,data_ntk_incl,weight_ntk_incl)
      plotcategory(fnew,"highML_"+intk,vars_plot,data_highML,weight_highML)
      plotcategory(fnew,"lowML_"+intk,vars_plot,data_lowML,weight_lowML)

    fnew.Close()

    # print number of events in each region
    weights = GetNormWeight(fns, fndir_plot, isData, int_lumi=40610.0)
    cut_var = 'vtx_ntk'
    cut_val_high = 5
    cut_val_val = 4
    # total_sum/var = [A,B,C,D] representing regions
    region_names = ['5tk high', '5tk low', '4tk high', '4tk low', '3tk high', '3tk low']
    total_sum = [0,0,0,0,0,0]
    total_var = [0,0,0,0,0,0]
    for i in range(len(fns)):
        w = phys_vars[i]['weight']
        cut_var_array = phys_vars[i][cut_var]
        cut_region = [
            (idx_highML[i]) & (cut_var_array>=cut_val_high), # A
            (idx_lowML[i]) & (cut_var_array>=cut_val_high),  # B
            (idx_highML[i]) & (cut_var_array>=cut_val_val) & (cut_var_array<cut_val_high),
            (idx_lowML[i]) & (cut_var_array>=cut_val_val) & (cut_var_array<cut_val_high),
            (idx_highML[i]) & (cut_var_array<cut_val_val),  # C
            (idx_lowML[i]) & (cut_var_array<cut_val_val),   # D
        ]
        for iregion in range(len(cut_region)):
            w_region = w[cut_region[iregion]]
            nevt_region = np.sum(w_region)*weights[i]
            nevt_raw = len(w_region)
            nevt_variance_region = nevt_region*weights[i]
            total_sum[iregion] += nevt_region
            total_var[iregion] += nevt_variance_region
            if not isData:
              print("sample {} in region {} : {} +- {} raw: {}".format(fns[i],region_names[iregion],nevt_region,np.sqrt(nevt_variance_region),nevt_raw))
            
    if not(isData or isSignal):
      print("Summing together: ")
      for iregion in range(len(region_names)):
          print("Region {}: {} +- {}".format(region_names[iregion],total_sum[iregion],np.sqrt(total_var[iregion])))
    


def main():
    fns = [
      #"qcdht0100_2017",
      #"qcdht0200_2017",
      #"qcdht0300_2017",
      "qcdht0500_2017",
      "qcdht0700_2017",
      "qcdht1000_2017",
      "qcdht1500_2017",
      "qcdht2000_2017",
      "wjetstolnuht0100_2017",
      "wjetstolnuht0200_2017",
      "wjetstolnuht0400_2017",
      "wjetstolnuht0600_2017",
      "wjetstolnuht0800_2017",
      "wjetstolnuht1200_2017",
      "wjetstolnuht2500_2017",
      #"zjetstonunuht0100_2017",
      "zjetstonunuht0200_2017",
      "zjetstonunuht0400_2017",
      "zjetstonunuht0600_2017",
      "zjetstonunuht0800_2017",
      "zjetstonunuht1200_2017",
      "zjetstonunuht2500_2017",
      "ww_2017",
      "wz_2017",
      "zz_2017",
      "st_tchan_antitop_2017",
      "st_tchan_top_2017",
      "st_tw_antitop_2017",
      "st_tw_top_2017",
      "ttbar_2017",
    ]
    if doBackground:
      #makeplotfile(fns,"background_METtrigger",False)
      makeplotfile(fns,"background_METtrigger",False,False,0.4,0.4)
      #for bkg_fn in fns:
      #  makeplotfile([bkg_fn],bkg_fn+"_lowMET_bquark",False)

    sig_fns = ['mfv_splitSUSY_tau000000100um_M2000_1800_2017',
               'mfv_splitSUSY_tau000000100um_M2000_1900_2017',
               'mfv_splitSUSY_tau000000100um_M2400_2300_2017',
               'mfv_splitSUSY_tau000000300um_M2000_1800_2017',
               'mfv_splitSUSY_tau000000300um_M2000_1900_2017',
               'mfv_splitSUSY_tau000000300um_M2400_2300_2017',
               'mfv_splitSUSY_tau000001000um_M2000_1800_2017',
               'mfv_splitSUSY_tau000001000um_M2000_1900_2017',
               'mfv_splitSUSY_tau000001000um_M2400_2300_2017',
               'mfv_splitSUSY_tau000001000um_M1200_1100_2017',
               'mfv_splitSUSY_tau000001000um_M1400_1200_2017',
               'mfv_splitSUSY_tau000010000um_M2000_1800_2017',
               'mfv_splitSUSY_tau000010000um_M2000_1900_2017',
               'mfv_splitSUSY_tau000010000um_M2400_2300_2017',
               'mfv_splitSUSY_tau000010000um_M1200_1100_2017',
               'mfv_splitSUSY_tau000010000um_M1400_1200_2017',
               'mfv_splitSUSY_tau000100000um_M2000_1800_2017',
               'mfv_splitSUSY_tau000100000um_M2000_1900_2017',
               #'mfv_splitSUSY_tau001000000um_M2000_1800_2017',
               #'mfv_splitSUSY_tau001000000um_M2000_1900_2017',
               #'mfv_splitSUSY_tau010000000um_M2000_1800_2017',
               #'mfv_splitSUSY_tau010000000um_M2000_1900_2017',
              ]
    if doSignal:
      for sig_fn in sig_fns:
        makeplotfile([sig_fn],sig_fn+"_METtrigger",True,False,0.4,0.4)
    fns_data = [
      'MET2017B',
      'MET2017C',
      'MET2017D',
      'MET2017E',
      'MET2017F',
    ]
    if doData:
      makeplotfile(fns_data,"data_METtrigger",False,True,0.4,0.4)

main()


