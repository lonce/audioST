import numpy as np
# https://github.com/librosa/librosa
import librosa
import librosa.display
import scipy
import tiffspect
import math
import argparse

FLAGS = None
# ------------------------------------------------------
# get any args provided on the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('tifFile', type=str, help='stored log mag spectrogram'  ) 
parser.add_argument('--sr', type=int, help='samplerate', default=22050 ) 
parser.add_argument('--fftSize', type=int, help='fft size, and window size', default=1024  ) 
parser.add_argument('--hopSize', type=int, help='size of frame hop through sample file', default=256 ) 
parser.add_argument('--glSteps', type=int, help='number of Griffin&Lim iterations following SPSI', default=50 ) 
parser.add_argument('--wavFile', type=str, help='output audio file. Unspecified means just play audio', default="wavOut.wav" ) 

FLAGS, unparsed = parser.parse_known_args()
print('\n FLAGS parsed :  {0}'.format(FLAGS))

def spsi(msgram, fftsize, hop_length) :
    """
    Takes a 2D spectrogram ([freqs,frames]), the fft legnth (= widnow length) and the hope size (both in units of samples).
    Returns an audio signal.
    """
    numBins, numFrames  = msgram.shape
    y_out=np.zeros(numFrames*hop_length+fftsize-hop_length)
        
    m_phase=np.zeros(numBins);      
    m_win=scipy.signal.hanning(fftsize, sym=True)  # assumption here that hann was used to create the frames of the spectrogram
    
    #processes one frame of audio at a time
    for i in range(numFrames) :
            m_mag=msgram[:, i] 
            for j in range(1,numBins-1) : 
                if(m_mag[j]>m_mag[j-1] and m_mag[j]>m_mag[j+1]) : #if j is a peak
                    alpha=m_mag[j-1];
                    beta=m_mag[j];
                    gamma=m_mag[j+1];
                    denom=alpha-2*beta+gamma;
                    
                    if(denom!=0) :
                        p=0.5*(alpha-gamma)/denom;
                    else :
                        p=0;
                        
                    phaseRate=2*math.pi*(j-1+p)/fftsize;    #adjusted phase rate
                    m_phase[j]= m_phase[j] + hop_length*phaseRate; #phase accumulator for this peak bin
                    peakPhase=m_phase[j];
                    
                    # If actual peak is to the right of the bin freq
                    if (p>0) :
                        # First bin to right has pi shift
                        bin=j+1;
                        m_phase[bin]=peakPhase+math.pi;
                        
                        # Bins to left have shift of pi
                        bin=j-1;
                        while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until you reach the trough
                            m_phase[bin]=peakPhase+math.pi;
                            bin=bin-1;
                        
                        #Bins to the right (beyond the first) have 0 shift
                        bin=j+2;
                        while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                            m_phase[bin]=peakPhase;
                            bin=bin+1;
                            
                    #if actual peak is to the left of the bin frequency
                    if(p<0) :
                        # First bin to left has pi shift
                        bin=j-1;
                        m_phase[bin]=peakPhase+math.pi;

                        # and bins to the right of me - here I am stuck in the middle with you
                        bin=j+1;
                        while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                            m_phase[bin]=peakPhase+math.pi;
                            bin=bin+1;
                        
                        # and further to the left have zero shift
                        bin=j-2;
                        while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until trough
                            m_phase[bin]=peakPhase;
                            bin=bin-1;
                            
                #end ops for peaks
            #end loop over fft bins with

            magphase=m_mag*np.exp(1j*m_phase)  #reconstruct with new phase (elementwise mult)
            
            magphase[numBins-1] = 0 #remove dc and nyquist
            magphase[0]=0;
            if not len(magphase) % 2 :
                #ACTUALLY, I'm not sure what it means to get a mag spectrum that has an even number of rows. Has the nyquist bin been removed?
                m_recon=np.concatenate([magphase, [0], np.flip(np.conjugate(magphase[0:numBins-1]), 0)])
            else :
                m_recon=np.concatenate([magphase,np.flip(np.conjugate(magphase[1:numBins-1]), 0)]) 
            
            #overlap and add
            m_recon=np.real(np.fft.ifft(m_recon))*m_win
            y_out[i*hop_length:i*hop_length+fftsize]+=m_recon
            
    return y_out


D, _ = tiffspect.Tiff2LogSpect(FLAGS.tifFile)#('spectrograms/BeingRural_short.tif')
magD = np.power(10, D/20)
y_out = spsi(magD, fftsize=FLAGS.fftSize, hop_length=FLAGS.hopSize)

if FLAGS.glSteps != 0 :
	p = np.angle(librosa.stft(y_out, FLAGS.fftSize, center=False))
	for i in range(FLAGS.glSteps):
	    S = magD * np.exp(1j*p)
	    y_out = librosa.istft(S, center=True) # Griffin Lim, assumes hann window, 1/4 window hop size ; librosa only does one iteration?
	    p = np.angle(librosa.stft(y_out, FLAGS.fftSize, center=True))

scalefactor = np.amax(np.absolute(y_out))
print('scaling peak sample, ' + str(scalefactor) + ' to 1')
y_out/=scalefactor
librosa.output.write_wav(FLAGS.wavFile, y_out, FLAGS.sr)


