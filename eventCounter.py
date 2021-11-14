import ROOT
import ctypes
from uncertainties import ufloat
fn_path = 'UL1112_highMET_ntk/'
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
      print("  region {} highML: {:15.8f}".format(d,nevts[d][0]))
      print("  region {}  lowML: {:15.8f}".format(d,nevts[d][1]))
      print("          highML/lowML ratio: {:15.8f}".format(nevts[d][0]/nevts[d][1]))
      print("            predicted highML: {:15.8f}".format(nevts[d][1]*(nevts['inclusive_3trk'][0]/nevts['inclusive_3trk'][1])))
    

