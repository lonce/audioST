import os 
import re

import numpy as np
import math
import tiffspect

import librosa
import librosa.display

import matplotlib.pyplot as plt

K_SPECTDIR = '/home/lonce/tflow/DATA-SETS/ESC-50-spect'
k_soundsPerClass=125 # must divide the total number of sounds evenly!

#============================================

def weightedCentroid(spect) :
    """
    param: spect - a magnitude spectrum
    Returns the spectral centroid averaged over frames, and weighted by the rms of each frame
    """
    cent = librosa.feature.spectral_centroid(S=spect)
    rms = librosa.feature.rmse(S=spect)
    avg = np.sum(np.multiply(cent, rms))/np.sum(rms)
    return avg

def log2mag(S) : 
    """ Get your log magnitude spectrum back to magnitude"""
    return np.power(10, np.divide(S,20.))

def spectFile2Centroid(fname) :
    """ Our spect files are in log magnitude, and in tiff format"""
    D1, _ = tiffspect.Tiff2LogSpect(fname)
    D2 = log2mag(D1)
    return weightedCentroid(D2)
#============================================

# Next, some utilities for managing files
#----------------------------------------

def fullpathfilenames(directory): 
    '''Returns the full path to all files living in directory (the leaves in the directory tree)
    '''
    fnames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(directory)) for f in fn]
    return fnames

def esc50files(directory, regexString) :
    filenames = fullpathfilenames(directory)
    return [fname for fname in filenames if re.match(regexString, fname)]

def addClass2Filename(fname, cname, action="move") : 
    newname = re.sub('.tif', '._'+ str(cname) + '_.tif', fname)
    if (action == "move") :
        os.rename(fname, newname)
    else :
        print(newname)
    
def filestats (filenames, func) :
    stats = [[fname, func(fname)] for fname in filenames]
    return stats

#============================================


def createBalancedClassesWithFunc(topDirectory, regexString, func, numPerClass, action="move") :
    """
    Groups files in topDirectory matching regexString by the single number returned by func.
    Each group will have numPerClass files in it (the total number of files must be divisible by numPerClass)
    Renames them using their group index, gidx: origFilename.tif -> origFilename._gidx_.tif
    if action="move, files are renames. Otherwise, the new names are just printed to console.
    """
    wholelist=esc50files(topDirectory, regexString)
    stats = filestats(wholelist, func)
    stats_ordered = sorted(stats, key=lambda a_entry: a_entry[1])
    classes=np.array(stats_ordered)[:,0].reshape(-1, numPerClass)
    for i in range(len(classes)) :
        for j in range(len(classes[i])) :
            addClass2Filename(classes[i,j],i, action)

    return stats, stats_ordered #returns stuff just for viewing 

#--------------------------------------------------------------------------------
#if you got yourself in trouble, and need to remove all the secondary classnames:
def removeAllSecondaryClassNames(directory) :
    """Revomve ALL the 2ndary class names (of the form ._cname_) from ALL files in the directory restoring them to their original"""
    for fname in fullpathfilenames(directory) :
        m = re.match('.*?(\._.*?_)\.tif$', fname)  #grabs the string of all secondary classes if there is a seq of them
        if (m) :
            newname = re.sub(m.group(1), '', fname)
            print('Will move ' + fname + '\n to ' + newname)
            os.rename(fname, newname)
        else :
            print('do nothing with ' + fname)

#============================================

# DO IT
stats, stats_ordered  = createBalancedClassesWithFunc(K_SPECTDIR, '.*/([1-5]).*', spectFile2Centroid, k_soundsPerClass, action="print")
stats, stats_ordered  = createBalancedClassesWithFunc(K_SPECTDIR, '.*/([1-5]).*', spectFile2Centroid, k_soundsPerClass, action="move")

