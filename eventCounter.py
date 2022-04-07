import ROOT
import ctypes
from uncertainties import ufloat
#fn_path = 'UL1115_ULV11_ntk_1/'
fn_path = 'UL0406_ULV11_ntk_3/'
if __name__ == '__main__':
  #dirs = ['inclusive_SR',
  #        'inclusive_VR',
  #        'inclusive_CR',
  #       ]
  dirs = ['inclusive_5trk',
          'inclusive_4trk',
          'inclusive_3trk',
         ]
  fns = [
    'background',
    'mfv_splitSUSY_tau000000100um_M2000_1800_2017',
    'mfv_splitSUSY_tau000000100um_M2400_2300_2017',
    'mfv_splitSUSY_tau000000300um_M2000_1800_2017',
    'mfv_splitSUSY_tau000000300um_M2400_2300_2017',
    'mfv_splitSUSY_tau000001000um_M1200_1100_2017',
    'mfv_splitSUSY_tau000001000um_M1400_1200_2017',
    'mfv_splitSUSY_tau000001000um_M2000_1800_2017',
    'mfv_splitSUSY_tau000001000um_M2400_2300_2017',
    'mfv_splitSUSY_tau000010000um_M1200_1100_2017',
    'mfv_splitSUSY_tau000010000um_M1400_1200_2017',
    'mfv_splitSUSY_tau000010000um_M2000_1800_2017',
    'mfv_splitSUSY_tau000010000um_M2400_2300_2017',
  ]
  fn_common = '_METtrigger.root'
  MLhigh = 0.4
  MLlow = 0.4
  nevts = {
    'inclusive_5trk': [],
    'inclusive_4trk': [],
    'inclusive_3trk': [],
  }
  for fn in fns:
    f = ROOT.TFile(fn_path+fn+fn_common)
    print(fn)
    for d in dirs:
      h = f.Get(d+'/MLScore')
      #h.Sumw2()
      binhigh = h.GetXaxis().FindBin(MLhigh)
      binlow = h.GetXaxis().FindBin(MLlow)-1
      nevt_error_high = ctypes.c_double(0)
      nevt_error_low = ctypes.c_double(0)
      nevt_high = h.IntegralAndError(binhigh,100000,nevt_error_high)
      nevt_low = h.IntegralAndError(0,binlow,nevt_error_low)
      nevts[d] = [ufloat(nevt_high,nevt_error_high.value),ufloat(nevt_low,nevt_error_low.value)]
    for d in dirs:
      print("{:<30}: {:15.8f}".format("region "+d+" highML",nevts[d][0]))
      print("{:<30}: {:15.8f}".format("region "+d+" lowML",nevts[d][1]))
      #if not 'mfv' in fn:
      if 1:
        print("{:<30}: {:15.8f}".format("highML/lowML ratio", nevts[d][0]/nevts[d][1]))
        print("{:<30}: {:15.8f}".format("predicted highML", nevts[d][1]*(nevts['inclusive_3trk'][0]/nevts['inclusive_3trk'][1])))
    

