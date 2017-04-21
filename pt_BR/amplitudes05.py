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

import ricker01
import contrasteImpedancia
import geracaoRuido02

#{{{
def amplitudesArtigo(angulo, frequencia):
    """Gera um array 2D com amplitudes sinteticas geradas pela convolucao da wavelet com as refletividades"""
    rickerArtigo = ricker01.geraRickerDeAcordoComOArtigo2(angulo, frequencia)
    refletividade = contrasteImpedancia.geraConstrasteImpedancia200TracosModelo()
    amplitudeSismica = np.zeros(shape=(200,800))
    for i in range(refletividade.shape[0]):
        ref = refletividade[i,:].ravel()
        cnv = np.convolve(rickerArtigo, ref, mode='same')
        amplitudeSismica[i,:] = cnv
    # XXX: comentei a normalizaco por valor absoluto maximo
    # amplitudeSismica = amplitudeSismica / np.max(np.abs(amplitudeSismica))
    return amplitudeSismica.T
#}}}

def amplitudesDoArtigoComRuido2(percentual, angulo=-30., frequencia=30.): #{
    amplitudeSismica = amplitudesArtigo(angulo, frequencia)
    maxamp = np.max(np.abs(amplitudeSismica))
    amplitudesComRuido = geracaoRuido02.applyNoiseToImage(amplitudeSismica, percentual*maxamp, True)
    # XXX: comentei a normalizacao por valor absoluto maximo
    # amplitudesComRuido = amplitudesComRuido / np.max(np.abs(amplitudesComRuido))
    return amplitudesComRuido
#}

def exibirAmplitudesArtigo(ang, freq): #{
    amplitudeSismica = amplitudesArtigo(ang, freq)
    lims = (np.min(amplitudeSismica), np.max(amplitudeSismica))
    cmap = 'spectral'
    imgplot = plt.imshow(amplitudeSismica, clim=lims, cmap=cmap)
    plt.colorbar()
    plt.show(imgplot)
#}

def plotTimeFreqSignal(ySinal1, ySinal2, freqSinal, tempoMaximo):
    Ts = 1.0/freqSinal; # sampling interval
    t = np.arange(0,tempoMaximo,Ts) # time vector
    
    n = len(ySinal1) # length of the signal
    k = np.arange(n)
    T = n/freqSinal
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range
    Y = np.fft.fft(ySinal1)/n # fft computing and normalization
    Y = Y[range(n/2)]

    n2 = len(ySinal2) # length of the signal
    k2 = np.arange(n2)
    T2 = n2/freqSinal
    frq2 = k2/T2 # two sides frequency range
    frq2 = frq[range(n2/2)] # one side frequency range
    Y2 = np.fft.fft(ySinal2)/n2 # fft computing and normalization
    Y2 = Y2[range(n2/2)]


    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t,ySinal1,t,ySinal2)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frq,abs(Y), 'r', frq, abs(Y2),'g') # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()


def exibirAmplitudesComRuido(ruido = 0.05, angulo = -30., frequencia = 30.):#{
    semRuido = amplitudesArtigo(angulo, frequencia)
    amplitudeSismicaR = amplitudesDoArtigoComRuido2(ruido, angulo, frequencia)
    diferenca = amplitudeSismicaR - semRuido
    lims = (np.min(amplitudeSismicaR), np.max(amplitudeSismicaR))
    cmap = 'gray'
    imgplot = plt.imshow(amplitudeSismicaR, clim=lims, cmap=cmap)
    plt.colorbar()
    plt.show(imgplot)
    lims2 = (np.min(diferenca), np.max(diferenca))
    cmap = 'spectral'
    imgplot2 = plt.imshow(diferenca, clim=lims2, cmap=cmap)
    plt.colorbar()
    plt.show(imgplot2)
    # exibicao de um traco de amplitudes
    plt.plot(amplitudeSismicaR[:,1],linewidth=3.0)
    plt.show()
    # calculo da fft da amplitude
    sig = amplitudeSismicaR[:,1]
    freq = 1000.0
    tempoMaximo = 0.8
    
    genReflect = contrasteImpedancia.geraConstrasteImpedancia200TracosModelo()
    sinal2 = genReflect[1,:].ravel()
    sinal2[abs(sinal2) < 1e-2] = 0.0
    
    semRuido2 = amplitudesArtigo(angulo, frequencia)
    sinal2 = semRuido2[:,1]

    plotTimeFreqSignal(sig, sinal2, freq, tempoMaximo)
#}


if __name__ == "__main__":
    exibirAmplitudesArtigo(-30., 30.)
    exibirAmplitudesComRuido()
