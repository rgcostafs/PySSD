# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:38:00 2016

@author: Rodrigo
"""

## Passo 2
## Geração de ruído (branco?) gaussiano
## Fonte: http://dspguru.com/dsp/howtos/how-to-generate-white-gaussian-noise

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ricker01 import *

def initRandom(): #{
    random.seed() # vai usar a hora do sistema para inicializar o gerador de numeros pseudo-aleatorios (Mersenne-Twister)
#}
    
def generateOneSampleGaussian(n): #{
    x = 0.0
    for i in range(n):
        x += random.random()
    x = x - n/2.0
    x = x * math.sqrt(12.0 / float(n))
    return x
#}

def generateNSamplesGaussian01(numSamples): #{
    rs = np.random.random(numSamples)
    rs = rs - numSamples/2.0
    rs = rs * math.sqrt(12.0 / float(numSamples))
    return rs
#}

def generateNSamplesGaussian02(numSamples): #{
    return np.random.normal(size=numSamples)
#}

def readAnImage(filename, dtype): #{
    img = mpimg.imread(filename).astype(dtype)
    return img
#}

def applyNoiseToImage(img, intensity, keepRange): #{
    minv = np.min(img)
    maxv = np.max(img)
    sz = img.shape[0] * img.shape[1]
    gnoise = generateNSamplesGaussian02(sz)
    #print "Noise: (min,max) = (%.3f,%.3f)" % (np.min(gnoise), np.max(gnoise))
    gnoise = gnoise * intensity
    gnoise = gnoise.reshape(img.shape)
    res = img + gnoise
    if (keepRange):
        minv2, maxv2 = np.min(res), np.max(res)
        ss = res.ravel()
        for i in range(len(ss)):
            ss[i] = minv + ((ss[i]-minv2)/(maxv2-minv2))*(maxv-minv)
        res = ss.reshape(img.shape)
    return res
#}

def testeGeracaoRuido02(): #{
    img = readAnImage(u'H:\\Projetos\\prisma\\data\\01_man.tiff', np.float32)
    plt.figure(1)
    imgplot = plt.imshow(img)
    imgplot.set_cmap('gray')
    plt.colorbar()
    plt.show(imgplot)    
    plt.figure(2)
    imgplot = plt.imshow(img[200:400,200:400])
    imgplot.set_cmap('gray')
    plt.colorbar()
    plt.show(imgplot)

    plt.figure(3)
    img2 = applyNoiseToImage(img, 10, True)
    imgplot2 = plt.imshow(img2)
    imgplot2.set_cmap('gray')
    plt.colorbar()
    plt.show(imgplot2)
    
    plt.figure(4)
    imgplot2 = plt.imshow(img2[200:400,200:400])
    imgplot2.set_cmap('gray')
    plt.colorbar()
    plt.show(imgplot2)
    
    # Aplicacao de ruido gaussiano numa Ricker
    
    duration = 0.80 # duracao em segundos
    fs = 1000.0 # frequencia de amostragem: 1000Hz, uma amostra a cada 1ms
    samples = int(fs*duration) # numero de amostras = 100 amostras
    t = np.linspace(-0.5, 0.5, samples) # tempo
    
    angulo = -30.0 # 30 graus
    frequenciaDominante = 30.0 # 30Hz eh a frequencia dominante da Ricker
    waveletInicial = rickerShift(t[350:450], frequenciaDominante, angulo * math.pi / 180.0)
    
    plt.figure(3)
    plt.plot(t[350:450], waveletInicial, 'b')
    plt.show()
    
    gnoise = generateNSamplesGaussian02(len(waveletInicial))
    waveletInicial = waveletInicial + 0.05 * gnoise
    
    plt.figure(4)
    plt.plot(t[350:450], waveletInicial, 'b')
    plt.show()
#}

#if __name__ == "__main__":
#    testeGeracaoRuido02()
