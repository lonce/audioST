# kernel: pyaudio27
from __future__ import print_function
import numpy as np
from PIL import TiffImagePlugin
from PIL import Image

import json # http://python-guide-pt-br.readthedocs.io/en/latest/scenarios/json/

#------------------------------------------------
# For reading/writing single-channel tiff spectroggams in [0,1] / single channel log spectrograms in dB for display
#------------------------------------------------
# Standard (but optional) names for the info: 
# tiffinfo={"scale" : scale, "shift" : shift,
#            "sr" : srate, "n_fft" : fftsize, "hop_length" : fftHop, 
#           "linFreqBins" : LIN_FREQ_BINS, "lowRow" : LOW_ROW, "logFreqBins" : LOG_FREQ_BINS}
#------------------------------------------------

def logSpect2Tiff(outimg, fname, lwinfo=None):
    """ 
    Single channel spectrogram to tiff file normed to [0,1] 
    """
    info = TiffImagePlugin.ImageFileDirectory()
            
    scale = 80.
    shift = float(np.amax(outimg))

    lwinfo= lwinfo or {}
    lwinfo["scale"]=scale;
    lwinfo["shift"]=shift;

    #just chose a tag index that appears to be unused: https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml
    info[666]=json.dumps(lwinfo)


    #shift to put max at 0
    shiftimg = outimg-shift 
    #scale to map [-80, 0] to  [-1,0]
    outimg = [x / scale for x in outimg]
    #shift to [0,1]
    outimg = [x +1. for x in outimg]
    #clip anything below 0 (anything originally below -80dB)
    outimg = np.maximum(outimg, 0) # clip below 0
    #print('logSpect2Tiff: writing image with min ' + str(np.amin(outimg)) + ', and max ' + str(np.amax(outimg)))
    savimg = Image.fromarray(np.flipud(outimg))
    savimg.save(fname, tiffinfo=info)
    return info[666] # just in case you want it for some reason

def Tiff2LogSpect(fname) :
    """Read tif images, and expand to original scale, return single channel image"""
    img = Image.open(fname)
    #print('Tiff2LogSpect: image min is ' + str(np.amin(img)) + ', and image max is ' + str(np.amax(img)))


    lwinfo=json.loads(img.tag[666][0])
    try :
        scale=lwinfo["scale"]
    except :
        scale=80.

    try :
        shift=lwinfo["shift"]
    except :
        shift=0.

    outimg = np.asarray(img, dtype=np.float32)
    outimg = outimg-1.
    outimg = outimg*scale #  [x *scale for x in outimg] # 
    outimg = outimg + shift #  [x +shift for x in outimg] # 
    return (np.flipud(outimg), lwinfo)

def Tiff2MagSpect(fname) :
    logmag, lwinfo = Tiff2LogSpect(fname)
    return (np.power(10, logmag/20.) , lwinfo)
    


    