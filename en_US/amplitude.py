# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 21:21:28 2016

@author: Rodrigo
"""

## Parte 6
## Convolução de teste entre a Ricker que o artigo menciona e os traços do bloco selecionado

import scipy
import numpy as np
import matplotlib.pyplot as plt

import rickerwav
import contrastimped
import noisegen

#{{{
def amplitudesWang16(angle, frequency):
    """Gera um array 2D com amplitudes sinteticas geradas pela convolucao da wavelet com as reflectivitys"""
    rkr = rickerwav.generateRickerWang16(angle, frequency)
    reflectivity = contrastimped.generateConstrastImpedancyWang16()
    seismicAmplitude = np.zeros(shape=(200,800))
    for i in range(reflectivity.shape[0]):
        ref = reflectivity[i,:].ravel()
        cnv = np.convolve(rkr, ref, mode='same')
        seismicAmplitude[i,:] = cnv
    # XXX: comentei a normalizaco por valor absoluto maximo
    # seismicAmplitude = seismicAmplitude / np.max(np.abs(seismicAmplitude))
    return seismicAmplitude.T
#}}}

def amplitudesWithNoiseWang16(percent, angle=-30., frequency=30.): #{
    seismicAmplitude = amplitudesWang16(angle, frequency)
    maxamp = np.max(np.abs(seismicAmplitude))
    amplitudesWithNoise = noisegen.applyNoiseToImage(seismicAmplitude, percent*maxamp, True)
    # XXX: comentei a normalization por valor absoluto maximo
    # amplitudesWithNoise = amplitudesWithNoise / np.max(np.abs(amplitudesWithNoise))
    return amplitudesWithNoise
#}

def displayAmplitudesWang16(ang, freq): #{
    seismicAmplitude = amplitudesWang16(ang, freq)
    lims = (np.min(seismicAmplitude), np.max(seismicAmplitude))
    cmap = 'spectral'
    imgplot = plt.imshow(seismicAmplitude, clim=lims, cmap=cmap)
    plt.colorbar()
    plt.show(imgplot)
#}

def plotTimeFreqSignal(ySignal1, ySignal2, freqSignal, duration):
    Ts = 1.0/freqSignal; # sampling interval
    t = np.arange(0,duration,Ts) # time vector
    
    n = len(ySignal1) # length of the signal
    k = np.arange(n)
    T = n/freqSignal
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range
    Y = np.fft.fft(ySignal1)/n # fft computing and normalization
    Y = Y[range(n/2)]

    n2 = len(ySignal2) # length of the signal
    k2 = np.arange(n2)
    T2 = n2/freqSignal
    frq2 = k2/T2 # two sides frequency range
    frq2 = frq[range(n2/2)] # one side frequency range
    Y2 = np.fft.fft(ySignal2)/n2 # fft computing and normalization
    Y2 = Y2[range(n2/2)]


    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t,ySignal1,t,ySignal2)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frq,abs(Y), 'r', frq, abs(Y2),'g') # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()


def displayAmplitudesWithNoise(ruido = 0.05, angle = -30., frequency = 30.):#{
    nonoise = amplitudesWang16(angle, frequency)
    seismicAmplitudeR = amplitudesWithNoiseWang16(ruido, angle, frequency)
    diffce = seismicAmplitudeR - nonoise
    lims = (np.min(seismicAmplitudeR), np.max(seismicAmplitudeR))
    cmap = 'gray'
    imgplot = plt.imshow(seismicAmplitudeR, clim=lims, cmap=cmap)
    plt.colorbar()
    plt.show(imgplot)
    lims2 = (np.min(diffce), np.max(diffce))
    cmap = 'spectral'
    imgplot2 = plt.imshow(diffce, clim=lims2, cmap=cmap)
    plt.colorbar()
    plt.show(imgplot2)
    # exibicao de um trace de amplitudes
    plt.plot(seismicAmplitudeR[:,1],linewidth=3.0)
    plt.show()
    # calculo da fft da amplitude
    sig = seismicAmplitudeR[:,1]
    freq = 1000.0
    duration = 0.8
    
    genReflect = contrastimped.generateConstrastImpedancyWang16()
    sinal2 = genReflect[1,:].ravel()
    sinal2[abs(sinal2) < 1e-2] = 0.0
    
    nonoise2 = amplitudesWang16(angle, frequency)
    sinal2 = nonoise2[:,1]

    plotTimeFreqSignal(sig, sinal2, freq, duration)
#}


if __name__ == "__main__":
    displayAmplitudesWang16(-30., 30.)
    displayAmplitudesWithNoise()
