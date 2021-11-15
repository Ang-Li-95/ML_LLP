
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
  [4.907, 12.21],
  [0.1238, 1.195],
  [-0.03527, 1.801],
  [-8.377e-05, 0.02875],
  [0.0009776, 5.025],
  [0.1099, 3.532],
  [19.47, 757.4],
  #[7.084, 17.73],
  #[0.1627, 1.148],
  #[-0.01556, 1.807],
  #[-0.0004281, 0.03207],
  #[-0.03078, 5.274],
  #[0.3174, 3.391],
  #[61.16, 833.6],
]
normalize_factors_vtx_EOY = [
  [3.634, 0.3758],
  [0.02555, 0.02167],
  [0.001634, 0.0004853],
]
normalize_factors_vtx_UL = [
  [3.496, 0.7984],
  [0.6714, 0.5107],
  [0.01499, 0.008985],
  #[3.61, 0.3493],
  #[0.02523, 0.03307],
  #[0.001679, 0.0004475],
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
lambda_param = 0.001
lambda_dcorr_met = 0
lr = 0.0005
use_dR = False
num_epochs=50

isUL = True
#fndir = "root://cmseos.fnal.gov//store/user/ali/MLTreeV43keeptkMETm/MLTreeV43keeptkMETm/"
fndir = "/uscms/home/ali/nobackup/LLP/crabdir/MLTreeULV3_keeptkMETm/"
#fndir = 'root://cmseos.fnal.gov//store/user/ali/MLTreeULV1_keeptkMETm/MLTreeULV1_keeptkMETm/'
dir_model = "./model_1115_ntk_2/"

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
    "qcdht0200_2017",
    "qcdht0300_2017",
    "qcdht0500sum_2017",
    "qcdht0700_2017",
    "qcdht1000_2017",
    "qcdht1500_2017",
    "qcdht2000_2017",
    "wjetstolnu_2017",
    "zjetstonunuht0100_2017",
    "zjetstonunuht0200_2017",
    "zjetstonunuht0400_2017",
    "zjetstonunuht0600_2017",
    "zjetstonunuht0800_2017",
    "zjetstonunuht1200_2017",
    "zjetstonunuht2500_2017",
    "ttbar_2017",
]
fns_signal = [
    "mfv_splitSUSY_tau000000100um_M2000_1800_2017",
    "mfv_splitSUSY_tau000000100um_M2000_1900_2017",
    "mfv_splitSUSY_tau000000300um_M2000_1800_2017",
    "mfv_splitSUSY_tau000000300um_M2000_1900_2017",
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

