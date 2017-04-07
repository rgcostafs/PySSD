import numpy as np
import math

#{{{ custoMQO Ax = b
def custoMQO(A, x, b, fator = 1.0):
    mA = np.matrix(A)
    mX = np.matrix(x)
    mB = np.matrix(b)
    diff = mA * mX - mB
    return (fator * (diff.T * diff))[0,0]
#}}}

#{{{ classe CustoMQO 
class CustoMQO:
    def __init__(self, A, b):
        self.mA = np.matrix(A)
        self.mB = np.matrix(b)

    def execute(self, x):
        return custoMQO(self.mA, x, self.mB)
#}}}

#{{{ gradienteMQO de Ax = b
def gradienteMQO(A, x, b, fator = 2.0):
    mA = np.matrix(A)
    mX = np.matrix(x)
    mB = np.matrix(b)
    diffs = mA * mX - mB
    return fator * mA.T * diffs
#}}}

#{{{ classe GradienteMQO
class GradienteMQO:
    def __init__(self, A, b): #{
        self.mA = A
        self.mB = b
    #}
    def execute(self, x): #{
        return gradienteMQO(self.mA, x, self.mB)
    #}
#}}}

#{{{ passoGradienteDescendente
def passoGradienteDescendente(medidas_A, parametros_x, observacoes_b, taxa_alfa, fator = 2.0):
    """ Se uma constante estiver presente na funcao alvo da regressao, a primeira coluna devera vir com 1's """
    grad = gradienteMQO(medidas_A, parametros_x, observacoes_b, fator)
    novos_parametros_x = parametros_x - taxa_alfa * grad
    return (novos_parametros_x, grad)
#}}}

#{{{ gradienteDescendente
def gradienteDescendente(medidas_A, parametros_x, observacoes_b, opcoes):
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

    for i in xrange(limiteIteracoes): #{
        (x1, grad) = passoGradienteDescendente(medidas_A, x1, observacoes_b, alfa, fator)
        if (usarLog):
            logCusto.append(custoMQO(medidas_A, x1, observacoes_b, 0.5 * fator))
        if (np.linalg.norm(grad) < limiarGrad):
            break
    #}
    if (usarLog):
        return (x1, logCusto)
    else:
        return x1
#}}}

#{{{ matrizRegularizacaoDiferencaQuadratica
def matrizRegularizacaoDiferencaQuadratica(theta, regularizarPrimeiraColuna = False):
    sp = theta.shape
    RQD = np.matrix(np.zeros(shape=sp))
    shift = 0
    if (regularizarPrimeiraColuna):
        shift = -1
    for k in range(2+shift,sp[0]-1):
        RQD[k] = 2*theta[k] - theta[k-1] - theta[k+1]
    RQD[1+shift] = theta[1+shift] - theta[2+shift]
    RQD[-1] = theta[-1] - theta[-2]
    return RQD
#}}}

#{{{ passoGDRegularizacaoDiferencaQuadratica
def passoGDRegularizacaoDiferencaQuadratica(medidas_A, parametros_x, observacoes_b, alfa, beta, fator = 2.0, regularizarColuna1 = False):
    """Supoe que a matriz de medidas tem sua primeira coluna com 1's"""
    regQDiff = matrizRegularizacaoDiferencaQuadratica(parametros_x, regularizarColuna1)
    grad = gradienteMQO(medidas_A, parametros_x, observacoes_b, fator) + beta*regQDiff
    tt = parametros_x - alfa * grad
    return (tt, grad)
#}}}

#{{{ gradienteDescendenteRegularizacaoDiferencaQuadratica
def gradienteDescendenteRegularizacaoDiferencaQuadratica(medidas_A, parametros_x, observacoes_b, opcoes):
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
        alfa = 1.0 / magFrobeniusNorm
    beta = 1.0 / (parametros_x.shape[0]-1)
    if (opcoes.has_key("beta") and opcoes["beta"] > 0.0):
        beta = opcoes["beta"]
    fator = 2.0 # fator de escala a ser usado no custo e no gradiente
    if (opcoes.has_key("fator") and opcoes["fator"] > 0.0):
        fator = opcoes["fator"]
    regCol1 = False
    if (opcoes.has_key("regularizar_col_1")):
        regCol1 = opcoes["regularizar_col_1"]

    for i in xrange(limiteIteracoes): #{
        (x1, grad) = passoGDRegularizacaoDiferencaQuadratica(medidas_A, parametros_x, observacoes_b, alfa, beta, fator, regCol1)
        if (usarLog):
            logCusto.append(custoMQO(medidas_A, x1, observacoes_b, 0.5 * fator))
        if (np.linalg.norm(grad) < limiarGrad):
            break
    #}
    if (usarLog):
        return (x1, logCusto)
    else:
        return x1
#}}}

#{{{ passoGDRegularizacaoTikhonov
def passoGDRegularizacaoTikhonov(medidas_A, parametros_x, observacoes_b, alfa, beta, fator = 2.0, regularizarColuna1 = False):
    beta1 = min(0.999*parametros_x.shape[0]/alfa, beta)
    betaCol = np.matrix(np.zeros(parametros_x.shape))
    if regularizarColuna1:
        betaCol[:,0] = 2.0*beta1*(1.0/parametros_x.shape[0])*parametros_x[:,0]
    else:
        betaCol[1:,0] = 2.0*beta1*(1.0/(parametros_x.shape[0]-1))*parametros_x[1:,0]
    grad = gradienteMQO(medidas_A, parametros_x, observacoes_b, fator) + betaCol
    tt = parametros_x - alfa * grad
    return (tt, grad)
#}}}

#{{{ gradienteDescendenteRegularizacaoTikhonov
def gradienteDescendenteRegularizacaoTikhonov(medidas_A, parametros_x, observacoes_b, opcoes):
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
        alfa = 1.0 / magFrobeniusNorm
    beta = 1.0 / (parametros_x.shape[0]-1)
    if (opcoes.has_key("beta") and opcoes["beta"] > 0.0):
        beta = opcoes["beta"]
    fator = 2.0 # fator de escala a ser usado no custo e no gradiente
    if (opcoes.has_key("fator") and opcoes["fator"] > 0.0):
        fator = opcoes["fator"]
    regCol1 = False
    if (opcoes.has_key("regularizar_col_1")):
        regCol1 = opcoes["regularizar_col_1"]
    x1 = parametros_x.copy()

    for i in xrange(limiteIteracoes): #{
        (x1, grad) = passoGDRegularizacaoTikhonov(medidas_A, x1, observacoes_b, alfa, beta, fator, regCol1)
        if (usarLog):
            if (regCol1):
                custoreg = beta*np.dot(x1.T, x1)[0,0]
            else:
                custoreg = beta*np.dot(x1[1:,:].T, x1[1:,:])[0,0]
            logCusto.append(custoMQO(medidas_A, x1, observacoes_b, 0.5 * fator) + custoreg)
        if (np.linalg.norm(grad) < limiarGrad):
            break
    #}
    if (usarLog):
        return (x1, logCusto)
    else:
        return x1
#}}}

#{{{ encontreConstanteLipschitz
def encontreConstanteLipschitz(A, theta, b, lmbda, eta):
    assert(eta>1.0)
    ik = 0
    encontrei_L = False
    L_candidato = 1.0
    objCusto = CustoMQO(A,b)
    funcCusto = objCusto.execute
    objGradiente = GradienteMQO(A,b)
    funcGrad = objGradiente.execute

    while (not encontrei_L):
        alpha = 1.0 / L_candidato
        ldaLocal = lmbda / L_candidato
        prox = passoISTA(A, b, theta, alpha, ldaLocal)
        F = custoLasso(A, prox, b, ldaLocal)
        Q = modeloBasicoAproximacaoQuadraticaLasso(funcCusto, funcGrad, L_candidato, prox, theta, ldaLocal)
        if (F > Q):
            ik += 1
            L_candidato = (eta**ik)
        else:
            encontrei_L = True
    return L_candidato
#}}}
