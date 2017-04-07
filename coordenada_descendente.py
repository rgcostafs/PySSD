import numpy as np
import math

import gradiente_descendente as gd

#{{{ passoCD
# j eh o indice/dimensao que sera otimizado(a)
def passoCD(A, x, b, j):
    xx2 = np.matrix(x.copy())
    x2 = np.matrix(x.copy())
    A2 = np.matrix(A.copy())
    A2[:,j] = 0.0
    x2[j,:] = 0.0
    xx2[j,0] = ((A[:,j].T * (b - A2*x2)) / (A[:,j].T * A[:,j]))
    return xx2
#}}}

#{{{ coordenadaDescendente
def coordenadaDescendente(medidas_A, parametros_x, observacoes_b, opcoes): #{
    usarLog = False
    if (opcoes.has_key("retornar_log") and opcoes["retornar_log"]):
        usarLog = True
        logCusto = []
    limiteIteracoes = 100000 # limito a 100.000 iteracoes
    if (opcoes.has_key("iteracoes") and opcoes["iteracoes"] > 0):
        limiteIteracoes = opcoes["iteracoes"]
    limiarMagParam = 1e-15
    if (opcoes.has_key("limiar_magnitude_parametro") and opcoes["limiar_magnitude_parametro"] > 0.0):
        limiarMagParam = opcoes["limiar_magnitude_parametro"]
    n = parametros_x.shape[0] # quantos parametros == quantas linhas tem no vetor de parametros
    indices_escolhidos = range(n)
    if (opcoes.has_key("indices_escolhidos")):
        indices_escolhidos = opcoes["indices_escolhidos"]
    x1 = parametros_x.copy()
    mags = np.array([0.0] * len(indices_escolhidos))
    for i in xrange(limiteIteracoes): #{
        ie = 0
        for j in indices_escolhidos:
            x2 = passoCD(medidas_A, x1, observacoes_b, j)
            mags[ie] = abs(x2[j,0] - x1[j,0])
            x1[:,:] = x2[:,:]
            ie += 1
            if (usarLog):
                logCusto.append(gd.custoMQO(medidas_A, x1, observacoes_b))
        if (np.max(mags) < limiarMagParam):
            break
        if (i % 10) == 0:
            print "it cd: ", i, " np.max(mags) ", np.max(mags)
    #}
    if (usarLog):
        return (x1, logCusto)
    else:
        return x1
#}}}


