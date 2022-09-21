import ROOT
import ctypes
from uncertainties import ufloat
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="directory of histograms")
parser.add_argument('--year', help="year to look at")
parser.add_argument('--weight', type=float, default=1, help="overall weight applied on the number of events")
args = parser.parse_args()
fn_path = args.dir
if len(fn_path)!=0 and fn_path[-1]!='/':
  fn_path = fn_path+'/'
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
    #'data',
    #'background',
    'mfv_splitSUSY_tau000010000um_M2000_1950_',
    'mfv_splitSUSY_tau000010000um_M2000_1980_',
    #'mfv_splitSUSY_tau000000100um_M2000_1800_',
    #'mfv_splitSUSY_tau000000100um_M2000_1900_',
    #'mfv_splitSUSY_tau000000300um_M2000_1800_',
    #'mfv_splitSUSY_tau000000300um_M2000_1900_',
    #'mfv_splitSUSY_tau000001000um_M2000_1800_',
    #'mfv_splitSUSY_tau000001000um_M2000_1900_',
    #'mfv_splitSUSY_tau000010000um_M2000_1800_',
    #'mfv_splitSUSY_tau000010000um_M2000_1900_',
    #'mfv_splitSUSY_tau000100000um_M2000_1800_',
    #'mfv_splitSUSY_tau000100000um_M2000_1900_',

    #'mfv_splitSUSY_tau000000100um_M2000_1800_',
    #'mfv_splitSUSY_tau000000100um_M2000_1900_',
    #'mfv_splitSUSY_tau000000300um_M2000_1800_',
    #'mfv_splitSUSY_tau000000300um_M2000_1900_',
    #'mfv_splitSUSY_tau000001000um_M2000_1800_',
    #'mfv_splitSUSY_tau000001000um_M2000_1900_',
    #'mfv_splitSUSY_tau000003000um_M2000_1800_',
    #'mfv_splitSUSY_tau000003000um_M2000_1900_',
    #'mfv_splitSUSY_tau000010000um_M2000_1800_',
    #'mfv_splitSUSY_tau000010000um_M2000_1900_',
    #'mfv_splitSUSY_tau000030000um_M2000_1800_',
    #'mfv_splitSUSY_tau000030000um_M2000_1900_',
    #'mfv_splitSUSY_tau000100000um_M2000_1800_',
    #'mfv_splitSUSY_tau000100000um_M2000_1900_',
    #'mfv_splitSUSY_tau000300000um_M2000_1800_',
    #'mfv_splitSUSY_tau000300000um_M2000_1900_',
    #'mfv_splitSUSY_tau001000000um_M2000_1800_',
    #'mfv_splitSUSY_tau001000000um_M2000_1900_',

  ]
  fn_common = '_METtrigger_'+args.year+'.root'
  MLhigh = 0.2
  MLlow = 0.2
  nevts = {
    'inclusive_5trk': [],
    'inclusive_4trk': [],
    'inclusive_3trk': [],
  }
  for fn in fns:
    if 'mfv' in fn:
      fn_common = args.year+'_METtrigger.root'
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
      nevts[d] = [args.weight*ufloat(nevt_high,nevt_error_high.value),args.weight*ufloat(nevt_low,nevt_error_low.value)]
    for d in dirs:
      if fn=='data' and d=='inclusive_5trk':
        continue
      print("{:<30}: {:5.3f}".format("region "+d+" highML",nevts[d][0]))
      print("{:<30}: {:5.3f}".format("region "+d+" lowML",nevts[d][1]))
      if not 'mfv' in fn:
      #if 1:
        print("{:<30}: {:5.3f}".format("highML/lowML ratio", nevts[d][0]/nevts[d][1]))
        print("{:<30}: {:5.3f}".format("predicted highML", nevts[d][1]*(nevts['inclusive_3trk'][0]/nevts['inclusive_3trk'][1])))
    

