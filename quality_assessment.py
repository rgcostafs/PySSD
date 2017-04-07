#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:36:18 2017

@author: rodrigo
"""

import ricker01
from geracaoRuido02 import *
from modeloMarmousi203 import *
from contrasteImpedancia import *
from amplitudes05 import *
#import matrizToeplitz06

import numpy as np
import matplotlib.pyplot as plt
import uncertainty
import scipy.stats.stats as ss
import math

refletividades = geraConstrasteImpedancia200TracosModelo()
refletividadesSelecionadas = refletividades[::10]

observacoes = np.asfortranarray(np.matrix(np.load("observacoes_multi.npy")))
tracosAmplitude = np.matrix(observacoes.copy())

print "refletividadesSelecionadas.shape:", refletividadesSelecionadas.shape

ref = np.reshape(refletividadesSelecionadas.T, (800*20,))
wav = np.load("/home/rodrigo/Projetos/quali/wav_gabarito.npy")
wav = np.reshape(wav, (201,))

#plt.hist(ref, bins=1000)
#plt.yscale('log')
#plt.show()

simp_ref = uncertainty.interarrivalTimes(ref, 0.0099)

ref[abs(ref) < 0.0099] = 0.0

for met in ["OMP", "FistaTSMF", "NLasso"]:
    for n in [15, 20, 25, 30, 35, 40]:
        print "\n*****\nResult for ", met, " ", n, " peaks"
        
        R = np.load("rodadas/Ref%02d_%s_MUL.npy" % (n, met))
        W = np.load("rodadas/Wav%02d_%s_MUL.npy" % (n, met))
        A = np.load("rodadas/Toep%02d_%s_MUL.npy" % (n, met))
                
        W = np.reshape(W, (201,))

        rf = np.reshape(R, (800*20,))     
        rf[abs(rf) < 0.0099] = 0.0
        
        R = np.matrix(R)
        A = np.matrix(A)

        srf = uncertainty.interarrivalTimes(rf,0.0099)
        print "JSD(ref, r) = ", math.sqrt(uncertainty.divergenceShannonJensen(simp_ref, srf))
        print "CCP(ref, r)=", ss.pearsonr(ref, rf)
        d = ref - rf
        d2 = d * d
        print "SUM((ref_i - r_i)**2) = ", np.sum(d2)
        print "DOT(ref, r) = ", np.dot(ref, rf)
            
        print "xxxxx Wavelet analysis xxxxx"
        print "CCP_W(ref, r)=", ss.pearsonr(W, wav)
        d = W - wav
        d2 = d*d
        print "SUM((W_i - wav_i)**2) = ", np.sum(d2)

        print "***** Avaliacao da reconstrucao *****"
        (minv, maxv) = (np.min(tracosAmplitude), np.max(tracosAmplitude))
        erro = tracosAmplitude - A*R
        erro2 = np.linalg.norm(erro.T)**2
        print "erro2:", erro2
        mse = erro2 / float(np.prod(tracosAmplitude.shape))
        print "MSE:", mse
        print "PSNR:", 20. * math.log10((maxv-minv) / math.sqrt(mse))

        potSinal = np.linalg.norm(tracosAmplitude)**2
        print "SNR:", 10. * math.log10(potSinal / mse)
