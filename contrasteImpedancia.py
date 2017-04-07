# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:23:14 2016

@author: Rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

import modeloMarmousi203

def constrateImpedancia(velocidade, densidade): #{
    res = np.zeros(velocidade.shape)
    velprox = velocidade[1:]
    velprev = velocidade[:-1]
    densprox = densidade[1:]
    densprev = densidade[:-1]
    prox = velprox * densprox
    prev = velprev * densprev
    res[1:] = (prox - prev) / (prox + prev)

    maxabs = np.max(np.abs(res))
    res = res / maxabs

    return res
#}

def geraConstrasteImpedancia200TracosModelo(): #{
    selectedTraces = modeloMarmousi203.selecionarTracosExemplo()[::10,:]
    selectedDensities = modeloMarmousi203.selecionarTracosDensidadeExemplo()[::10,:]
    
    
    selectedTraces2 = np.zeros(shape=(selectedTraces.shape[0], 800), dtype=np.float32)
    selectedDensities2 = np.zeros(shape=(selectedDensities.shape[0], 800), dtype=np.float32)

    xnew = np.linspace(0.0,1.0,800)
    #print "xnew.shape:", xnew.shape
    for i in range(selectedTraces.shape[0]):
        xold = np.linspace(0.0,1.0,selectedTraces.shape[1])
        velref = selectedTraces[i,:].ravel()
        f = scipy.interpolate.interp1d(xold,velref,kind='nearest')
        selectedTraces2[i,:] = f(xnew)

        xoldD = np.linspace(0.0,1.0,selectedDensities.shape[1])
        denref = selectedDensities[i,:].ravel()
        f = scipy.interpolate.interp1d(xoldD,denref,kind='nearest')
        selectedDensities2[i,:] = f(xnew)


    #densidade = np.asarray(selectedTraces2.shape[1]*[1.0], dtype=np.float64)
    #print "densidade.shape:", densidade.shape
    genReflect = np.zeros(shape=(selectedTraces2.shape[0],selectedTraces2.shape[1]), dtype=np.float64)
    #print "genReflect.shape:", genReflect.shape
    for i in range(selectedTraces.shape[0]):
        #print "selectedTraces2[i,:].ravel().shape:", selectedTraces2[i,:].ravel().shape
        #print "densidade.ravel().shape:", densidade.ravel().shape
        genReflect[i,:] = constrateImpedancia(selectedTraces2[i,:].ravel(), selectedDensities2[i,:].ravel())
    return genReflect
#}

def plotTimeFreqSignal(ySinal, freqSinal, tempoMaximo):
    Ts = 1.0/freqSinal; # sampling interval
    t = np.arange(0,tempoMaximo,Ts) # time vector
    
    n = len(ySinal) # length of the signal
    k = np.arange(n)
    T = n/freqSinal
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range
    Y = np.fft.fft(ySinal)/n # fft computing and normalization
    Y = Y[range(n/2)]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t,ySinal,linewidth=3.0)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frq,abs(Y),'r',linewidth=3.0) # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()


def exibirConstrateImpedancia200TracosModelo(): #{
    genReflect = geraConstrasteImpedancia200TracosModelo()
    #print "Refletividade ", genReflect.shape
    lims4 = (np.min(genReflect), np.max(genReflect))
    cmap4 = 'seismic'
    gr = (genReflect.T)[:,::10]
    for j in range(gr.shape[1]):
        for i in range(gr.shape[0]-3):
            if abs(gr[i+3,j]) > 0.03:
                gr[i:i+4,j] = gr[i+3,j]
    imgplot5 = plt.imshow(gr, clim=lims4, cmap=cmap4, aspect=0.04)
    plt.colorbar()
    plt.show(imgplot5)
    plt.figure(10)
    plt.plot(genReflect[2,:].ravel(), 'b',linewidth=3.0)
    plt.show()
    
    sinal = genReflect[1,:].ravel()
    freq = 1000.0
    tMax = 0.8
    plotTimeFreqSignal(sinal, freq, tMax)
    

#}

if __name__ == "__main__":
    exibirConstrateImpedancia200TracosModelo()
