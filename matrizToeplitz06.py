# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 21:44:54 2016

@author: Rodrigo
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.linalg as spla

from ricker01 import *
from geracaoRuido02 import *
from modeloMarmousi203 import *
from contrasteImpedancia import *
from amplitudes05 import *


def matrizToeplitzDoSinal(s): #{
    sig = list(s)
    if ((len(sig) % 2) == 0):
        sig.append(sig[-1])
    hl = len(sig) / 2
    res = np.ndarray(shape=(hl+1,hl+1), dtype=np.float32)
    for row in range(hl+1):
        for col in range(hl+1):
            res[row,col] = sig[(col-row) + hl]
    return np.matrix(res)
#}

def montaMatrizToeplitzComRickerArtigo(angulo = -30.0, frequencia = 30.0): #{
    sinal = geraRickerDeAcordoComOArtigo2(angulo, frequencia)
    sinalAumentado = np.zeros(shape=(1599,))
    sinalAumentado[699:900] = sinal
    plt.plot(sinalAumentado)
    plt.show()
    toep = matrizToeplitzDoSinal(sinalAumentado)
    return np.matrix(toep)
#}

def montaMatrizToeplitzComRickerArtigoComRuido(percentual, angulo = -30.0, frequencia = 30.0): #{
    sinal = geraRickerDeAcordoComOArtigo2(angulo, frequencia)
    ruido = np.random.normal(size=len(sinal))
    maxval = np.max(np.abs(sinal))
    sinalAumentado = np.zeros(shape=(1599,))
    sinalAumentado[699:900] = sinal + percentual * ruido * maxval
#    plt.plot(sinalAumentado)
#    plt.show()
    toep = matrizToeplitzDoSinal(sinalAumentado)
    return np.matrix(toep)
#}

#
#def ToeplitzParaAlpha(toeplitzA): #{
#    n = toeplitzA.shape[0]
#    alpha = np.matrix(np.zeros(shape=(2*n-1, 1)))
#    for i in range(n):
#        alpha[n-1-i] = toeplitzA[0,i]
#        alpha[n-1+i] = toeplitzA[i,0]
#    return alpha
##}
#

def ToeplitzParaAlpha2(toeplitzA): #{
    n = toeplitzA.shape[0]
    alpha = np.matrix(np.zeros(shape=(2*n-1, 1)))
    alpha[:n] = toeplitzA[::-1,0]
    alpha[n:] = toeplitzA[0,1:].T
    return alpha
#}

#
#def alphaParaToeplitz(alpha): #{
#    n = alpha.shape[0]
#    s = (n+1)/2
#    toep = np.matrix(np.zeros(shape=(s,s)))
#    for j in range(s): #{
#        for i in range(s): #{
#            p = i-j+s-1
#            toep[i,j] = alpha[p]
#        #}
#    #}
#    return toep
##}
#

def alphaParaToeplitz2(alpha): #{
    s = alpha.shape[0]/2
    return np.matrix(spla.toeplitz(alpha[s::-1],alpha[s:]))
#}


def exibirMatrixToeplitzComRickerArtigo():#{
    cmap = 'gray'
    t = montaMatrizToeplitzComRickerArtigo()
    lims = (np.min(t), np.max(t))
    imgplot = plt.imshow(t, clim=lims, cmap=cmap)
    plt.colorbar()
    plt.show(imgplot)
#}

def alphaParaToeplitz3(mascara, sinal):
    szM = mascara.shape[0]
    szS = sinal.shape[0]
    primL = np.zeros(shape=(szS,))
    primC = np.zeros(shape=(szS,))
    primL[:1+szM/2] = mascara[szM/2::-1,0].T
    primC[:1+szM/2] = mascara[szM/2:,0].T
    matToepl = np.matrix(spla.toeplitz(primC,primL))
    return matToepl


def ToeplitzParaAlpha3(matrizT, mascara):
    szM = max(mascara.shape)
    masc = np.matrix(np.zeros((szM,1)))
    masc[szM/2::-1,0] = (matrizT[0,:1+szM/2]).T
    masc[szM/2:,0] = matrizT[:1+szM/2,0]
    return masc


if __name__ == "__main__":
    exibirMatrixToeplitzComRickerArtigo()
    toep = montaMatrizToeplitzComRickerArtigo()
    print toep.shape
    alpha0 = ToeplitzParaAlpha2(toep)
    toep2 = alphaParaToeplitz2(alpha0)
    print alpha0.shape
    for i in range(13):
        alpha = ToeplitzParaAlpha2(toep)
        toep = alphaParaToeplitz2(alpha)
        print "toep.shape:", toep.shape
        print "alpha.shape:", alpha.shape
        plt.plot(alpha0);
        plt.plot(alpha);
        plt.show();
