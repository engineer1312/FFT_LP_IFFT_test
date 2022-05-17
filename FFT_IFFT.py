# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:28:24 2020

@author: hekic
"""

import pandas as pd
from numpy.fft import rfft, fft, fftfreq, rfftfreq
import matplotlib.pyplot as plt
import numpy as np

print('I made change here')
Fs=500 #Sampling rate, is/should be known from the experimental campaign

#READING HDF5 FILE 
data=pd.read_hdf(r'2019_05_10_12_06_51_416000.hdf5')

#Plot of the signal
signal=data['S04'] #D02 is choosen sensor (D stands for displacement sensor)
plt.plot(signal)

#signal length should be even, else last data point is neglected 
'''It would work widd odd length as well, but length of the signal after
inverse FFT will be different than the length of the original signal'''

if signal.shape[0] % 2 != 0:
    print('Length of the original signal is odd, last data point will be cut')
    signal=signal[:-1]

#FFT with rfftfreq function
signallength=signal.shape[0]
frq=rfftfreq(n=signallength,d=1/Fs) #one side frequency range [0,FNyquist] Array of length n//2 + 1 containing the sample frequencies.
Y=rfft(signal)/signallength #fft computing and normalisation
Yabs=2*abs(Y)
fig=plt.figure()
ax=plt.subplot(111)
ax.plot(frq,Yabs)
ax.set_xlim(1/Fs,30)
ax.set_ylim(0,Yabs[1]*1.05)
plt.show()

#%%
#HAND MADE LOW-PASS FILTER
cutofffreq=10 #if this is Fs/2 then LP filter wont change anything
YLowPass=np.zeros(len(Y),dtype='complex')

for i in range(len(frq)):
    if frq[i]>=cutofffreq:
        YLowPass[i]=0
    else:
        YLowPass[i]=Y[i]
        
YabsLowPass=2*(abs(YLowPass))

plt.plot(frq,YabsLowPass)
plt.xlim(1/Fs,30)
plt.ylim(0,YabsLowPass[1]*1.05)
plt.show()

#%%    
#Inverse IFFT
YInverseLP=signallength*np.fft.irfft(YLowPass)
plt.plot(signal.index,signal,label='original') #original signal
plt.plot(signal.index,YInverseLP,label='LP filtered') #filtered signal
plt.legend()

#%%





















#%%
#ARCHIVE

#HIGH-PASS FILTER - usually it is not necessary
cutofffreq=0.01 #if this is 0 than LP filter wont change anything -> I use this just to make the 0th value in the frequency spectrum zero
YHighPass=np.zeros(len(Y),dtype='complex')
for i in range(len(frq)):
    if frq[i]*Fs<=cutofffreq:
        YHighPass[i]=0
    else:
        YHighPass[i]=Y[i]
        
YabsHighPass=2*(abs(YHighPass))

#IFFT
YInverseLP=signallength*np.fft.irfft(YHighPass)
plt.plot(signal.index,YInverseLP)

#Same FFT with rfftfreq function
frq=rfftfreq(signallength) #one side frequency range
Y=rfft(signal)/signallength #fft computing and normalisation
Yabs=pd.DataFrame(2*abs(Y))
plt.plot(frq*Fs,Yabs.iloc[:,0])
plt.xlim(1/Fs,20)
plt.ylim(0,0.04)
plt.show()
