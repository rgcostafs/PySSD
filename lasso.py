import numpy as np
import math

import gradiente_descendente as gd

#{{{ limiarizacaoSuave1
def limiarizacaoSuave1(x, lmbda):
    """soft thresholding operator (Hastie, 2015, cap.2, pag. 15)"""
    return np.sign(x) * np.max(np.abs(x) - lmbda, 0.0)
#}}}

#{{{ limiarizacaoSuave
def limiarizacaoSuave(x, lda, regularizarColuna1 = False):
    """soft thresholding operator (Hastie, 2015, cap.2, pag. 15)"""
    if (lda > 0.0):
        sz = max(x.shape)
        xc = np.array(x.copy())
        tmp = np.matrix(x.copy())
        
        tmp[:,0] = np.reshape(np.sign(xc[:,0]) * np.array(map(lambda k : max(abs(k)-lda, 0.0), xc[:,0])), (sz,1))
        if (not regularizarColuna1):
            tmp[0,0] = xc[0,0]
        return tmp
    else:
        return x
#}}}

#{{{ limiarizacaoSuave2
def limiarizacaoSuave2(x, lda):
    """soft thresholding operator (Hastie, 2015, cap.2, pag. 15)"""
    if (lda > 0.0):
        sz = max(x.shape)
        xc = np.array(x.copy())
        tmp = np.matrix(x.copy())
        tmp[:,0] = np.reshape(np.sign(xc[:,0]) * np.array(map(lambda k : max(abs(k)-lda, 0.0), xc[:,0])), (sz,1))
        return tmp
    else:
        return x
#}}}


#{{{ custoLasso
def custoLasso(A, x, b, lmbda, fator = 1.0):
    mA = np.matrix(A)
    mX = np.matrix(x)
    mB = np.matrix(b)
    diff = mA * mX - mB
    l1normX = np.sum(np.abs(mX))
    cst = fator * (diff.T * diff) + lmbda*l1normX
    return cst[0,0]
#}}}

#{{{ CustoLasso
class CustoLasso:
    def __init__(self, A, b):
        self.mA = np.matrix(A)
        self.mB = np.matrix(b)

    def execute(self, x, lmbda):
        return custoLasso(self.mA, x, self.mB, lmbda)
#}}}

#{{{ penalidadeLasso
def penalidadeLasso(x, lmbda):
    return lmbda * np.sum(np.abs(x))
#}}}

#{{{ penalidadeFusedLasso
def penalidadeFusedLasso(x, lmbda1, lmbda2):
    parte1 = penalidadeLasso(x, lmbda1)
    parte2 = lmnda2 * np.sum(np.abs(x[1:] - x[:-1]))
    return parte1 + parte2
#}}}

#{{{ passoISTA
def passoISTA(medidas_A, parametros_x, observacoes_b, alfa, lmbda, fator = 2.0, regularizarColuna1 = False): #{
    (tt, grad) = gd.passoGradienteDescendente(medidas_A, parametros_x, observacoes_b, alfa, fator)
    return (limiarizacaoSuave(tt, lmbda, regularizarColuna1), grad)
#}}}

#{{{ ISTA
def ISTA(medidas_A, parametros_x, observacoes_b, opcoes):
    x1 = parametros_x.copy()
    usarLog = False
    if (opcoes.has_key("retornar_log") and opcoes["retornar_log"]):
        usarLog = True
        logCusto = []
    limiteIteracoes = 1000000 # limito a 1 milhao de iteracoes
    if (opcoes.has_key("iteracoes") and opcoes["iteracoes"] > 0):
        limiteIteracoes = opcoes["iteracoes"]
    limiarGrad = 1e-15
    if (opcoes.has_key("limiar_magnitude_gradiente") and opcoes["limiar_magnitude_gradiente"] > 0.0):
        limiarGrad = opcoes["limiar_magnitude_gradiente"]
    if (opcoes.has_key("alfa") and opcoes["alfa"] > 0.0):
        alfa = opcoes["alfa"]
    else:
        magFrobeniusNorm = np.linalg.norm(medidas_A.T * medidas_A)
        alfa = 1.0 / (magFrobeniusNorm**2)
    fator = 2.0 # fator de escala a ser usado no custo e no gradiente
    if (opcoes.has_key("fator") and opcoes["fator"] > 0.0):
        fator = opcoes["fator"]
    lmbda = 0.0
    if (opcoes.has_key("lambda") and opcoes["lambda"] > 0.0):
        lmbda = opcoes["lambda"]
    regCol1 = False
    if (opcoes.has_key("regularizar_col_1")):
        regCol1 = opcoes["regularizar_col_1"]

    it = 0
    for i in xrange(limiteIteracoes): #{
        (x1, grad) = passoISTA(medidas_A, x1, observacoes_b, alfa, lmbda, fator, regCol1)
        if (usarLog):
            logCusto.append(custoLasso(medidas_A, x1, observacoes_b, lmbda, 0.5 * fator))
        if (np.linalg.norm(grad) < limiarGrad):
            break
        it += 1
    #}
    print "ISTA executou ", it, "iteracoes"
    if (usarLog):
        return (x1, logCusto)
    else:
        return x1
#}}}

#{{{ passoISTA2
def passoISTA2(medidas_A, parametros_x, observacoes_b, alfa, lmbda, fator = 2.0, regularizarColuna1 = False): #{
    (tt, grad) = gd.passoGradienteDescendente(medidas_A, parametros_x, observacoes_b, alfa, fator)
    return (limiarizacaoSuave(tt, lmbda, regularizarColuna1), grad)
#}}}

#{{{ ISTA2
def ISTA2(medidas_A, parametros_x, observacoes_b, opcoes):
    x1 = parametros_x.copy()
    usarLog = False
    if (opcoes.has_key("retornar_log") and opcoes["retornar_log"]):
        usarLog = True
        logCusto = []
    limiteIteracoes = 1000000 # limito a 1 milhao de iteracoes
    if (opcoes.has_key("iteracoes") and opcoes["iteracoes"] > 0):
        limiteIteracoes = opcoes["iteracoes"]
    limiarGrad = 1e-15
    if (opcoes.has_key("limiar_magnitude_gradiente") and opcoes["limiar_magnitude_gradiente"] > 0.0):
        limiarGrad = opcoes["limiar_magnitude_gradiente"]
    if (opcoes.has_key("alfa") and opcoes["alfa"] > 0.0):
        alfa = opcoes["alfa"]
    else:
        magFrobeniusNorm = np.linalg.norm(medidas_A.T * medidas_A)
        alfa = 1.0 / (magFrobeniusNorm**2)
    fator = 2.0 # fator de escala a ser usado no custo e no gradiente
    if (opcoes.has_key("fator") and opcoes["fator"] > 0.0):
        fator = opcoes["fator"]
    lmbda = 0.0
    if (opcoes.has_key("lambda") and opcoes["lambda"] > 0.0):
        lmbda = opcoes["lambda"]
    calcularIntercept = False
    if (opcoes.has_key("calcular_intercept")):
        calcularIntercept = opcoes["calcular_intercept"]
    
    it = 0
    for i in xrange(limiteIteracoes): #{
        (x1, grad) = passoISTA(medidas_A, x1, observacoes_b, alfa, lmbda, fator, regCol1)
        if (usarLog):
            logCusto.append(custoLasso(medidas_A, x1, observacoes_b, lmbda, 0.5 * fator))
        if (np.linalg.norm(grad) < limiarGrad):
            break
        it += 1
    #}
    print "ISTA executou ", it, "iteracoes"
    if (usarLog):
        return (x1, logCusto)
    else:
        return x1
#}}}


#{{{ passoFISTAConstante (substitui o passoFISTAConstante do metodosOtimizacao)
def passoFISTAConstante(medidas_A, parametros_x, observacoes_b, alfa, lmbda, xk_menos_1, tk, fator = 2.0, regularizarColuna1 = False):
    (xk, grad) = gd.passoGradienteDescendente(medidas_A, parametros_x, observacoes_b, alfa, fator)
    xk = limiarizacaoSuave(xk, lmbda, regularizarColuna1)
    tkp1 = 0.5 * (1.0 + math.sqrt(1.0 + 4.0*tk*tk))
    ykp1 = xk + ((tk-1.0)/tkp1)*(xk - xk_menos_1)
    return (ykp1, xk, tkp1, grad)
#}}}

#{{{ FISTA (substitui o FISTA de metodosOtimizacao)
def FISTA(medidas_A, parametros_x, observacoes_b, opcoes):
    usarLog = False
    if (opcoes.has_key("retornar_log") and opcoes["retornar_log"]):
        usarLog = True
        logCusto = []
    limiteIteracoes = 100000 # limito a 100.000 iteracoes
    if (opcoes.has_key("iteracoes") and opcoes["iteracoes"] > 0):
        limiteIteracoes = opcoes["iteracoes"]
    limiarGrad = 1e-15
    if (opcoes.has_key("limiar_magnitude_gradiente") and opcoes["limiar_magnitude_gradiente"] > 0.0):
        limiarGrad = opcoes["limiar_magnitude_gradiente"]
    if (opcoes.has_key("alfa") and opcoes["alfa"] > 0.0):
        alfa = opcoes["alfa"]
    else:
        magFrobeniusNorm = np.linalg.norm(medidas_A.T * medidas_A)
        alfa = 1.0 / (magFrobeniusNorm**2)
    fator = 2.0 # fator de escala a ser usado no custo e no gradiente
    if (opcoes.has_key("fator") and opcoes["fator"] > 0.0):
        fator = opcoes["fator"]
    lmbda = 0.0
    if (opcoes.has_key("lambda") and opcoes["lambda"] > 0.0):
        lmbda = opcoes["lambda"]
    regCol1 = False
    if (opcoes.has_key("regularizar_col_1")):
        regCol1 = opcoes["regularizar_col_1"]
    
    tk = 1.0
    x1 = parametros_x.copy()
    xkm1 = parametros_x.copy()
    it = 0
    for i in xrange(limiteIteracoes): #{
        (x1, xkm1, tk, grad) = passoFISTAConstante(medidas_A, x1, observacoes_b, alfa, lmbda, xkm1, tk, fator, regCol1)
        if (usarLog):
            logCusto.append(custoLasso(medidas_A, x1, observacoes_b, lmbda, 0.5 * fator))
        if (np.linalg.norm(grad) < limiarGrad):
            break
        it += 1
    #}
    print "FISTA executou ", it, "iteracoes"
    if (usarLog):
        return (x1, logCusto)
    else:
        return x1
#}

