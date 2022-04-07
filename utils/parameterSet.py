
mlvar_tk = ['tk_pt', 'tk_eta', 'tk_phi', 'tk_dxybs','tk_dxybs_sig','tk_dz','tk_dz_sig']
mlvar_vtx = ['vtx_ntk', 'vtx_dBV', 'vtx_dBVerr']

normalize_factors_tk_EOY = [
  [7.13, 18],
  [0.1876, 1.115],
  [0.0108, 1.816],
  [3.025e-06, 0.03052],
  [0.01902, 5.188],
  [0.2271, 3.668],
  [54.45, 958.4],
]
normalize_factors_tk_UL = [
  #ULV5
  #[5.625, 13.37],
  #[0.1027, 1.201],
  #[-0.0194, 1.814],
  #[0.001582, 0.04679],
  #[0.0234, 3.762],
  #[-0.245, 3.486],
  #[-57.99, 716.7],
  #ULV11 MC
  #[7.333, 17.92],
  #[0.1716, 1.117],
  #[0.03046, 1.803],
  #[8.38e-05, 0.0224],
  #[-0.005048, 4.235],
  #[-0.6606, 3.715],
  #[-121.7, 852.5],
  #ULV11 MC no MET cut
  #[6.36, 14.34],
  #[0.1506, 1.145],
  #[-0.01217, 1.823],
  #[-0.0006342, 0.0313],
  #[0.06847, 4.031],
  #[-0.5231, 3.406],
  #[-88.08, 743.8],
  #ULV11 data
  #[6.911, 18.47],
  #[0.1158, 1.103],
  #[-0.01123, 1.809],
  #[0.0004163, 0.02455],
  #[0.05917, 4.423],
  #[0.0524, 3.434],
  #[13.43, 690.7],
  #ULV11 data no MET cut
  #[5.804, 16.67],
  #[0.1364, 1.147],
  #[-0.01353, 1.816],
  #[0.0001037, 0.02479],
  #[0.01197, 4.173],
  #[0.1075, 3.563],
  #[20.25, 688.1],
  #ULV11 MC no MET njet cut
  [5.519, 13.03],
  [0.09625, 1.197],
  [-0.03469, 1.809],
  [0.001684, 0.04679],
  [0.03939, 3.697],
  [-0.2593, 3.443],
  [-57.78, 686.6],
  #ULV11 data no MET njet cut
  #[5.708, 16.47],
  #[0.1376, 1.15],
  #[-0.01476, 1.817],
  #[8.22e-05, 0.02479],
  #[0.01199, 4.146],
  #[0.1208, 3.569],
  #[21.5, 683.3],
]
normalize_factors_vtx_EOY = [
  [3.634, 0.3758],
  [0.02555, 0.02167],
  [0.001634, 0.0004853],
]
normalize_factors_vtx_UL = [
  #ULV5
  #[3.101, 0.3274],
  #[0.02427, 0.01203],
  #[0.001624, 0.0003752],
  #ULV11 MC
  #[3.195, 0.4725],
  #[0.02491, 0.0294],
  #[0.0017, 0.0004682],
  #ULV11 MC no MET cut
  #[3.137, 0.3913],
  #[0.02527, 0.01691],
  #[0.001844, 0.0004894],
  #ULV11 data
  #[3.211, 0.5193],
  #[0.02529, 0.015],
  #[0.001773, 0.0004394],
  #ULV11 data no MET cut
  #[3.183, 0.4385],
  #[0.02793, 0.02858],
  #[0.00185, 0.0004259],
  #ULV11 MC no MET njet cut
  [3.11, 0.3554],
  [0.02394, 0.01664],
  [0.001918, 0.0004637],
  #ULV11 data no MET njet cut
  #[3.18, 0.4351],
  #[0.02808, 0.02844],
  #[0.001861, 0.0004274],
]

No = 50
Ds = len(mlvar_tk)
Nr = No*(No-1)
Dp = 20
Dv = len(mlvar_vtx)
Dv_ori = len(mlvar_vtx)
Dr = 1
De = 20
#lambda_dcorr = 0.5
lambda_param = 0.005
lambda_dcorr_met = 0
lr = 0.0005
use_dR = False
num_epochs=100

isUL = True
#fndir = "/uscms/home/ali/nobackup/LLP/crabdir/MLTreeULV4_norefit_keeptkMETm/"
#fndir = "root://cmseos.fnal.gov//store/user/ali/MLTree_trackdxysig0p5_ULV4_norefit_keeptkMETm/"
#fndir = "/uscms/home/ali/nobackup/LLP/crabdir/MLTreeULV3_keeptkMETm/"
fndir = "/uscms/home/ali/nobackup/LLP/crabdir/MLTreeULV11METm/"
dir_model = "./model_0406_ntk_ULV11_3/"

#fns_bkg = [
#    "qcdht0200_2017",
#    "qcdht0300_2017",
#    "qcdht0500_2017",
#    "qcdht0700_2017",
#    "qcdht1000_2017",
#    "qcdht1500_2017",
#    "qcdht2000_2017",
#    "wjetstolnusum_2017",
#    "zjetstonunuht0100_2017",
#    "zjetstonunuht0200_2017",
#    "zjetstonunuht0400_2017",
#    "zjetstonunuht0600_2017",
#    "zjetstonunuht0800_2017",
#    "zjetstonunuht1200_2017",
#    "zjetstonunuht2500_2017",
#    "ttbar_2017",
#]
#fns_signal = [
#    "mfv_splitSUSY_tau000000000um_M2000_1800_2017",
#    "mfv_splitSUSY_tau000000000um_M2000_1900_2017",
#    "mfv_splitSUSY_tau000000300um_M2000_1800_2017",
#    "mfv_splitSUSY_tau000000300um_M2000_1900_2017",
#    "mfv_splitSUSY_tau000001000um_M2000_1800_2017",
#    "mfv_splitSUSY_tau000001000um_M2000_1900_2017",
#    "mfv_splitSUSY_tau000010000um_M2000_1800_2017",
#    "mfv_splitSUSY_tau000010000um_M2000_1900_2017",
#]

fns_bkg = [
    "qcdht0100_2017",
    "qcdht0200_2017",
    "qcdht0300_2017",
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
    "zjetstonunuht0100_2017",
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
fns_signal = [
    "mfv_splitSUSY_tau000000100um_M2000_1800_2017",
    "mfv_splitSUSY_tau000000100um_M2000_1900_2017",
    "mfv_splitSUSY_tau000000100um_M2400_2300_2017",
    "mfv_splitSUSY_tau000000300um_M2000_1800_2017",
    "mfv_splitSUSY_tau000000300um_M2000_1900_2017",
    "mfv_splitSUSY_tau000000300um_M2400_2300_2017",
    "mfv_splitSUSY_tau000001000um_M2000_1800_2017",
    "mfv_splitSUSY_tau000001000um_M2000_1900_2017",
    "mfv_splitSUSY_tau000001000um_M1200_1100_2017",
    "mfv_splitSUSY_tau000001000um_M1400_1200_2017",
    "mfv_splitSUSY_tau000001000um_M2400_2300_2017",
    "mfv_splitSUSY_tau000010000um_M2000_1800_2017",
    "mfv_splitSUSY_tau000010000um_M2400_2300_2017",
    #"mfv_splitSUSY_tau000010000um_M1200_1100_2017",
    "mfv_splitSUSY_tau000010000um_M1400_1200_2017",
    "mfv_splitSUSY_tau000010000um_M2000_1900_2017",
]

