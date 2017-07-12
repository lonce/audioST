# kernel: pyaudio27
from __future__ import print_function
import numpy as np
from PIL import TiffImagePlugin
from PIL import Image

#------------------------------------------------
# For reading/writing single-channel tiff spectroggams in [0,1] / single channel log spectrograms in dB for display
#------------------------------------------------

def logSpect2Tiff(outimg, fname, scaleinfo=None):
    """ 
    Single channel spectrogram to tiff file normed to [0,1] 
    """
    info = TiffImagePlugin.ImageFileDirectory()
    
    scale = 80.
    shift = np.amax(outimg)
    
    if (scaleinfo == None) :
        info[270] = str(scale) + ',' + str(shift)
    else :
        info[270] = scaleinfo
    
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
    return info[270] # just in case you want it for some reason

def Tiff2LogSpect(fname) :
    """Read tif images, and expand to original scale, return single channel image"""
    img = Image.open(fname)
    #print('Tiff2LogSpect: image min is ' + str(np.amin(img)) + ', and image max is ' + str(np.amax(img)))
    try :
        img.tag[270] = img.tag[270]
    except :
        print('Tiff2LogSpect: no img.tag[207], no scale adjustment for ' +  fname)
        img.tag[270] = '80., 0'
        
    it = img.tag[270][0]
    sscale, sshift = it.split(',')
    #print('Tiff2LogSpect: img.tag info says scale is ' + sscale + ', and shift is ' + sshift)
    scale = float(sscale)
    shift = float(sshift)
    #outimg = [x -1 for x in img]
    outimg = np.asarray(img, dtype=np.float32)
    outimg = outimg-1.
    outimg = outimg*scale #  [x *scale for x in outimg] # 
    outimg = outimg + shift #  [x +shift for x in outimg] # 
    return (np.flipud(outimg), img.tag[270])

def Tiff2MagSpect(fname) :
    logmag, tag = Tiff2LogSpect(fname)
    return np.power(10, logmag/20.)


    