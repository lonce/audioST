import os
import numpy as np
import matplotlib.pyplot as plt
# https://github.com/librosa/librosa
import librosa
import librosa.display

import scipy

from PIL import TiffImagePlugin
from PIL import Image
import tiffspect

# Set some project parameters
K_SR = 22050
K_FFTSIZE = 512 # also used for window length where that parameter is called for
K_HOP = 128
K_DUR = 5.0 # make all files this duration
K_FRAMEMULTIPLEOF = 4 # some programs like to have convinent dimensions for conv and decimation
                        # the last columns of a matrix are removed if necessary to satisfy
                        # 1 means any number of frames will work

# location of subdirectories of ogg files organized by category
K_OGGDIR = '/home/lonce/tflow/DATA-SETS/ESC-50'
# location to write the wav files (converted from ogg)
K_WAVEDIR = '/home/lonce/tflow/DATA-SETS/ESC-50-wave'
# location to write the spectrogram files (converted from wave files)
K_SPECTDIR = '/home/lonce/tflow/DATA-SETS/ESC-50-spect'

#===============================================

def get_subdirs(a_dir):
    """ Returns a list of sub directory names in a_dir """ 
    return [name for name in os.listdir(a_dir)
            if (os.path.isdir(os.path.join(a_dir, name)) and not (name.startswith('.')))]

def listDirectory(directory, fileExtList):                                        
    """Returns list of file info objects in directory that extension in the list fileExtList - include the . in your extension string"""
    fnameList = [os.path.normcase(f)
                for f in os.listdir(directory)
                    if (not(f.startswith('.')))]            
    fileList = [os.path.join(directory, f) 
               for f in fnameList
                if os.path.splitext(f)[1] in fileExtList]  
    return fileList , fnameList

def dirs2labelfile(parentdir, labelfile):
    """takes subdirectories of parentdir and writes them, one per line, to labelfile"""
    namelist = get_subdirs(parentdir)
    #with open(labelfile, mode='wt', encoding='utf-8') as myfile:
    with open(labelfile, mode='wt') as myfile:
        myfile.write('\n'.join(namelist))

# ===============================================

def stereo2mono(data) :
    """ Combine 2D array into a single array, averaging channels """ 
    """ Deprecated, since we use librosa for this now. """ 
    print('converting stereo data of shape ' + str(data.shape))
    outdata=np.ndarray(shape=(data.shape[0]), dtype=np.float32)
    if data.ndim != 2 :
        print('You are calling stero2mono on a non-2D array')
    else : 
        print('    converting stereo to mono, with outdata shape = ' + str(outdata.shape))
        for idx in range(data.shape[0]) :
            outdata[idx] = (data[idx,0]+data[idx,1])/2
    return outdata

# ===============================================

def esc50Ogg2Wav (topdir, outdir, dur, srate) :
    """ 
        Creates regularlized wave files for the ogg files in the ESC-50 dataset. 
        Creates class folders for the wav files in outdir with the same structure found in topdir.
        
        Parameters
            topdir - the ESC-50 dir containing class folders. 
            outdir - the top level directory to write wave files to (written in to class subfolders)
            dur - (in seconds) all files will be truncated or zeropadded to have this duration given the srate
            srate - input files will be resampled to srate as they are read in before being saved as wav files
    """ 
    sample_length = int(dur * srate)
    try:
        os.stat(outdir)  # test for existence
    except:
        os.mkdir(outdir) # create if necessary
        
    subdirs = get_subdirs(topdir)
    for subdir in subdirs :
        try:
            os.stat(outdir + '/'  + subdir) # test for existence
        except:
            os.mkdir(outdir + '/' + subdir) # create if necessary
            print('creating ' + outdir + '/'  + subdir)
    
        fullpaths, _ = listDirectory(topdir + '/' + subdir, '.ogg') 
        for idx in range(len(fullpaths)) : 
            fname = os.path.basename(fullpaths[idx])
            # librosa.load resamples to sr, clips to duration, combines channels. 
            audiodata, samplerate = librosa.load(fullpaths[idx], sr=srate, mono=True, duration=dur) # resamples if necessary (some esc-50 files are in 48K)
            # just checking ..... 
            if (samplerate != srate) :
                print('You got a sound file ' + subdir  +  '/' +  fname + ' with sample rate ' + str(samplerate) + '!')
                print(' ********* BAD SAMPLE RATE ******** ')
            if (audiodata.ndim != 1) :
                print('You got a sound file ' + subdir  +  '/' +  fname + ' with ' + str(audiodata.ndim) + ' channels!')
                audiodata = stereo2mono(audiodata)
            if (len(audiodata) > sample_length) :
                print('You got a long sound file ' + subdir  +  '/' +  fname + ' with shape ' + str(audiodata.shape) + '!')
                audiodata = np.resize(audiodata, sample_length)
                # print('  ..... and len(audiodata) = ' + str(len(audiodata)) + ', while sample_length is sposed to be ' + str(sample_length))
                print('trimming data to shape ' + str(audiodata.shape))
            if (len(audiodata) < sample_length) :
                print('You got a short sound file ' + subdir  +  '/' +  fname + ' with shape ' + str(audiodata.shape) + '!')
                audiodata = np.concatenate([audiodata, np.zeros((sample_length-len(audiodata)))])
                print('      zero padding data to shape ' + str(audiodata.shape))
            # write the file out as a wave file
            librosa.output.write_wav(outdir + '/' + subdir + '/' + os.path.splitext(fname)[0] + '.wav', audiodata, samplerate)

# ===============================================



def wav2spect(fname, srate, fftSize, fftHop, dur=None, showplt=False, dcbin=True, framesmulitpleof=1) :
    try:
        audiodata, samplerate = librosa.load(fname, sr=srate, mono=True, duration=dur) 
    except:
        print('can not read ' + fname)
        return
    
    S = np.abs(librosa.stft(audiodata, n_fft=fftSize, hop_length=fftHop, win_length=fftSize,  center=False))

    if (dcbin ==  False) :
        S = np.delete(S, (0), axis=0)  # delete freq 0 row
            #note: a pure DC input signal bleeds into bin 1, too.
    
    #trim the non-mulitple fat if necessary
    nr, nc = S.shape  
    fat = nc%framesmulitpleof
    for num in range(0,fat):
        S = np.delete(S, (nc-1-num), axis=1)
        
        
    D = librosa.amplitude_to_db(S, ref=np.max)
    
    if showplt : # Dangerous for long runs - it opens a new figure for each file!
        librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=srate, hop_length=fftHop)
        plt.colorbar(format='%+2.0f dB')
        plt.title(showplt)
        plt.show(block=True)
                
    return D
# ===============================================

def esc50Wav2Spect(topdir, outdir, dur, srate, fftSize, fftHop, showplt=False, dcbin=True) :
    """ 
        Creates spectrograms for subfolder-labeled wavfiles. 
        Creates class folders for the spectrogram files in outdir with the same structure found in topdir.
        
        Parameters
            topdir - the dir containing class folders containing wav files. 
            outdir - the top level directory to write wave files to (written in to class subfolders)
            dur - (in seconds) all files will be truncated or zeropadded to have this duration given the srate
            srate - input files will be resampled to srate as they are read in before being saved as wav files
    """ 
    
    try:
        os.stat(outdir)  # test for existence
    except:
        os.mkdir(outdir) # create if necessary
    
    subdirs = get_subdirs(topdir)
    count = 0
    for subdir in subdirs :
        try:
            os.stat(outdir + '/'  + subdir) # test for existence
        except:
            os.mkdir(outdir + '/' + subdir) # create if necessary
            print('creating ' + outdir + '/'  + subdir)
    
        fullpaths, _ = listDirectory(topdir + '/' + subdir, '.wav') 
        
        for idx in range(len(fullpaths)) : 
            fname = os.path.basename(fullpaths[idx])
            # librosa.load resamples to sr, clips to duration, combines channels. 
            #
            #try:
            #    audiodata, samplerate = librosa.load(fullpaths[idx], sr=srate, mono=True, duration=dur) 
            #except:
            #    print('can not read ' + fname)
            #    
            #S = np.abs(librosa.stft(audiodata, n_fft=fftSize, hop_length=fftHop, win_length=fftSize,  center=False))
            #
            #if (! dcbin) :
            #    S = np.delete(S, (0), axis=0)  # delete freq 0 row
            ##print('esc50Wav2Spect" Sfoo max is ' + str(np.max(Sfoo)) +  ', and Sfoo sum is ' + str(np.sum(Sfoo)) + ', and Sfoo min is ' + str(np.min(Sfoo)))
            #
            #
            #D = librosa.amplitude_to_db(S, ref=np.max)
            D = wav2spect(fullpaths[idx], srate, fftSize, fftHop, dur=dur, dcbin=True, showplt=False, framesmulitpleof=K_FRAMEMULTIPLEOF)
            
            #plt.title(str(count) + ':  ' + subdir + '/' + os.path.splitext(fname)[0]) 
            
            tiffspect.logSpect2Tiff(D, outdir + '/' + subdir + '/' + os.path.splitext(fname)[0] + '.tif')
            
            print(str(count) + ': ' + subdir + '/' + os.path.splitext(fname)[0])
            count +=1
            
# ===============================================

# DO IT
#esc50Ogg2Wav(K_OGGDIR, K_WAVEDIR, K_DUR, K_SR)
#esc50Wav2Spect(K_WAVEDIR, K_SPECTDIR, K_DUR, K_SR, K_FFTSIZE, K_HOP, dcbin=True) 
dirs2labelfile(K_SPECTDIR, K_SPECTDIR + '/labels.text')


