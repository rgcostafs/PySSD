# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:31:20 2016

@author: Rodrigo
"""

## Passo 1
## Geração da Ricker com deslocamento de fase

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import sys

def ricker0(t,mf): #{
    a = (math.pi * mf * t)
    a *= a
    return (1.0 - 2.0*a) * math.exp(-a)
#}

def ricker(t,mf): #{
    a = (np.pi * mf * t)
    a *= a
    return (1.0 - 2.0*a) * np.exp(-a)
#}

def ricker2(t, mf): #{
    p = mf * mf * t * t
    parte1 = (1.0 - 2.0*math.pi*math.pi*p)
    parte2 = np.exp(-math.pi*math.pi*p)
    amplitude = parte1*parte2
    return amplitude
#}

def rickerShift2(t,mf,phase): #{
    amp = ricker2(t, mf)
    hsx = scipy.signal.hilbert(amp)
    x2 = math.cos(phase) * np.real(hsx)
    x2 = x2 - math.sin(phase) * np.imag(hsx)
    return x2
#}

def rickerShift(t,mf,phase): #{
    a = (np.pi * mf * t)
    a *= a
    amp = (1.0 - 2.0*a) * np.exp(-a)
    x = scipy.signal.hilbert(amp)
    x2 = np.cos(phase)*np.real(x) - np.sin(phase)*np.imag(x)
    return x2
#}

def rickervec(t,mf): #{
    res = np.zeros(shape=(len(t),))
    for i in range(len(t)):
        res[i] = ricker0(t[i],mf)
    return res
#}

def testeRicker01(): #{
    x = np.linspace(-0.5,0.5,9).astype(np.float64)
    y = ricker(x,10)
    print y
    plt.figure(1)
    plt.plot(x, y, 'r')
    plt.show()
    angle = -30.0 # 30 graus
    y2 = rickerShift(x,30,angle*math.pi/180.0)
    plt.figure(2)
    plt.plot(x, y2, 'g')
    plt.show()
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
#}

#def geraRickerDeAcordoComOArtigo(): #{
#    duration = 0.80 # duracao em segundos
#    samples = 1000 # numero de amostras = 800 amostras
#    t = np.linspace(-duration/2.0, duration/2.0, samples) # tempo
#    angulo = -30.0 # 30 graus
#    frequenciaDominante = 30.0 # 30Hz eh a frequencia dominante da Ricker
#    return rickerShift(t,frequenciaDominante,angulo*math.pi/180.0)
#}

def geraRickerDeAcordoComOArtigo2(angulo = -30.0, frequenciaDominante = 30.0): #{
    duracao = 0.201 # tempo, em segundos, de duracao do pulso
    samples = 201 # 201 amostras
    t = np.linspace(-duracao/2.0, duracao/2.0, samples) # tempo
    rs = rickerShift(t,frequenciaDominante,-angulo*math.pi/180.0)
    rs = rs / np.max(np.abs(rs))
    return rs
#}


#def exibeRickerDeAcordoComOArtigo(): #{
#    plt.figure(2)
#    plt.plot(t, geraRickerDeAcordoComOArtigo(),'r')
#    plt.show()
#}

def exibeRickerDeAcordoComOArtigo2(angulo = -30, frequenciaDominante = 30.0): #{
    plt.figure(2)
    plt.plot(geraRickerDeAcordoComOArtigo2(angulo, frequenciaDominante),'r', linewidth=3.0)
    plt.show()
#}

if __name__ == "__main__":
    #testeRicker01()
    if (len(sys.argv) >= 3):
        exibeRickerDeAcordoComOArtigo2(float(sys.argv[1]), float(sys.argv[2]))
    else:
        for frequencia in range(10,50,10):
            for angulo in range(-90,90,15):
                print "Showing ricker with angle %d and dom. freq. %d\n" % (angulo, frequencia)
                exibeRickerDeAcordoComOArtigo2(float(angulo), float(frequencia))
