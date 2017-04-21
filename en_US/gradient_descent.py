import numpy as np
import math

#{{{ costOLS Ax = b
def costOLS(A, x, b, factor = 1.0):
    mA = np.matrix(A)
    mX = np.matrix(x)
    mB = np.matrix(b)
    diff = mA * mX - mB
    return (factor * (diff.T * diff))[0,0]
#}}}

#{{{ classe CostOLS 
class CostOLS:
    def __init__(self, A, b):
        self.mA = np.matrix(A)
        self.mB = np.matrix(b)

    def execute(self, x):
        return costOLS(self.mA, x, self.mB)
#}}}

#{{{ gradientOLS de Ax = b
def gradientOLS(A, x, b, factor = 2.0):
    mA = np.matrix(A)
    mX = np.matrix(x)
    mB = np.matrix(b)
    diffs = mA * mX - mB
    return factor * mA.T * diffs
#}}}

#{{{ classe GradientOLS
class GradientOLS:
    def __init__(self, A, b): #{
        self.mA = A
        self.mB = b
    #}
    def execute(self, x): #{
        return gradientOLS(self.mA, x, self.mB)
    #}
#}}}

#{{{ stepGradDescent
def stepGradDescent(meas_A, params_x, obs_b, learning_alpha, factor = 2.0):
    """ Se uma constante estiver presente na funcao alvo da regressao, a primeira coluna devera vir com 1's """
    grad = gradientOLS(meas_A, params_x, obs_b, factor)
    novos_params_x = params_x - learning_alpha * grad
    return (novos_params_x, grad)
#}}}

#{{{ gradientDescent
def gradientDescent(meas_A, params_x, obs_b, options):
    x1 = params_x.copy()
    useLog = False
    if (options.has_key("return_log") and options["return_log"]):
        useLog = True
        costLog = []
    maxIters = 1000000 # limito a 1 milhao de iters
    if (options.has_key("iters") and options["iters"] > 0):
        maxIters = options["iters"]
    thresholdGrad = 1e-15
    if (options.has_key("threshold_mag_gradient") and options["threshold_mag_gradient"] > 0.0):
        thresholdGrad = options["threshold_mag_gradient"]
    if (options.has_key("alpha") and options["alpha"] > 0.0):
        alpha = options["alpha"]
    else:
        magFrobeniusNorm = np.linalg.norm(meas_A.T * meas_A)
        alpha = 1.0 / (magFrobeniusNorm**2)
    factor = 2.0 # factor de escala a ser usado no custo e no gradiente
    if (options.has_key("factor") and options["factor"] > 0.0):
        factor = options["factor"]

    for i in xrange(maxIters): #{
        (x1, grad) = stepGradDescent(meas_A, x1, obs_b, alpha, factor)
        if (useLog):
            costLog.append(costOLS(meas_A, x1, obs_b, 0.5 * factor))
        if (np.linalg.norm(grad) < thresholdGrad):
            break
    #}
    if (useLog):
        return (x1, costLog)
    else:
        return x1
#}}}

#{{{ matrixRegulQuadDifference
def matrixRegulQuadDifference(theta, regul1stCol = False):
    sp = theta.shape
    RQD = np.matrix(np.zeros(shape=sp))
    shift = 0
    if (regul1stCol):
        shift = -1
    for k in range(2+shift,sp[0]-1):
        RQD[k] = 2*theta[k] - theta[k-1] - theta[k+1]
    RQD[1+shift] = theta[1+shift] - theta[2+shift]
    RQD[-1] = theta[-1] - theta[-2]
    return RQD
#}}}

#{{{ stepGDRegulQuadDifference
def stepGDRegulQuadDifference(meas_A, params_x, obs_b, alpha, beta, factor = 2.0, regul1stCol = False):
    """Supoe que a matriz de medidas tem sua primeira coluna com 1's"""
    regQDiff = matrixRegulQuadDifference(params_x, regul1stCol)
    grad = gradientOLS(meas_A, params_x, obs_b, factor) + beta*regQDiff
    tt = params_x - alpha * grad
    return (tt, grad)
#}}}

#{{{ gradientDescentRegularizacaoDiferencaQuadratica
def gradientDescentRegularizacaoDiferencaQuadratica(meas_A, params_x, obs_b, options):
    x1 = params_x.copy()
    useLog = False
    if (options.has_key("return_log") and options["return_log"]):
        useLog = True
        costLog = []
    maxIters = 1000000 # limito a 1 milhao de iters
    if (options.has_key("iters") and options["iters"] > 0):
        maxIters = options["iters"]
    thresholdGrad = 1e-15
    if (options.has_key("threshold_mag_gradient") and options["threshold_mag_gradient"] > 0.0):
        thresholdGrad = options["threshold_mag_gradient"]
    if (options.has_key("alpha") and options["alpha"] > 0.0):
        alpha = options["alpha"]
    else:
        magFrobeniusNorm = np.linalg.norm(meas_A.T * meas_A)
        alpha = 1.0 / magFrobeniusNorm
    beta = 1.0 / (params_x.shape[0]-1)
    if (options.has_key("beta") and options["beta"] > 0.0):
        beta = options["beta"]
    factor = 2.0 # factor de escala a ser usado no custo e no gradiente
    if (options.has_key("factor") and options["factor"] > 0.0):
        factor = options["factor"]
    regCol1 = False
    if (options.has_key("regularize_1st_col")):
        regCol1 = options["regularize_1st_col"]

    for i in xrange(maxIters): #{
        (x1, grad) = stepGDRegulQuadDifference(meas_A, params_x, obs_b, alpha, beta, factor, regCol1)
        if (useLog):
            costLog.append(costOLS(meas_A, x1, obs_b, 0.5 * factor))
        if (np.linalg.norm(grad) < thresholdGrad):
            break
    #}
    if (useLog):
        return (x1, costLog)
    else:
        return x1
#}}}

#{{{ stepGDRegulTikhonov
def stepGDRegulTikhonov(meas_A, params_x, obs_b, alpha, beta, factor = 2.0, regul1stCol = False):
    beta1 = min(0.999*params_x.shape[0]/alpha, beta)
    betaCol = np.matrix(np.zeros(params_x.shape))
    if regul1stCol:
        betaCol[:,0] = 2.0*beta1*(1.0/params_x.shape[0])*params_x[:,0]
    else:
        betaCol[1:,0] = 2.0*beta1*(1.0/(params_x.shape[0]-1))*params_x[1:,0]
    grad = gradientOLS(meas_A, params_x, obs_b, factor) + betaCol
    tt = params_x - alpha * grad
    return (tt, grad)
#}}}

#{{{ gradientDescentRegularizacaoTikhonov
def gradientDescentRegularizacaoTikhonov(meas_A, params_x, obs_b, options):
    useLog = False
    if (options.has_key("return_log") and options["return_log"]):
        useLog = True
        costLog = []
    maxIters = 1000000 # limito a 1 milhao de iters
    if (options.has_key("iters") and options["iters"] > 0):
        maxIters = options["iters"]
    thresholdGrad = 1e-15
    if (options.has_key("threshold_mag_gradient") and options["threshold_mag_gradient"] > 0.0):
        thresholdGrad = options["threshold_mag_gradient"]
    if (options.has_key("alpha") and options["alpha"] > 0.0):
        alpha = options["alpha"]
    else:
        magFrobeniusNorm = np.linalg.norm(meas_A.T * meas_A)
        alpha = 1.0 / magFrobeniusNorm
    beta = 1.0 / (params_x.shape[0]-1)
    if (options.has_key("beta") and options["beta"] > 0.0):
        beta = options["beta"]
    factor = 2.0 # factor de escala a ser usado no custo e no gradiente
    if (options.has_key("factor") and options["factor"] > 0.0):
        factor = options["factor"]
    regCol1 = False
    if (options.has_key("regularize_1st_col")):
        regCol1 = options["regularize_1st_col"]
    x1 = params_x.copy()

    for i in xrange(maxIters): #{
        (x1, grad) = stepGDRegulTikhonov(meas_A, x1, obs_b, alpha, beta, factor, regCol1)
        if (useLog):
            if (regCol1):
                costreg = beta*np.dot(x1.T, x1)[0,0]
            else:
                costreg = beta*np.dot(x1[1:,:].T, x1[1:,:])[0,0]
            costLog.append(costOLS(meas_A, x1, obs_b, 0.5 * factor) + costreg)
        if (np.linalg.norm(grad) < thresholdGrad):
            break
    #}
    if (useLog):
        return (x1, costLog)
    else:
        return x1
#}}}

#{{{ findLipschitzConst
def findLipschitzConst(A, theta, b, lmbda, eta):
    assert(eta>1.0)
    ik = 0
    encontrei_L = False
    L_candidato = 1.0
    objCusto = CostOLS(A,b)
    funcCusto = objCusto.execute
    objGradiente = GradientOLS(A,b)
    funcGrad = objGradiente.execute

    while (not encontrei_L):
        alpha = 1.0 / L_candidato
        ldaLocal = lmbda / L_candidato
        prox = stepISTA(A, b, theta, alpha, ldaLocal)
        F = costLasso(A, prox, b, ldaLocal)
        Q = modelBasicApproxQuadLasso(funcCusto, funcGrad, L_candidato, prox, theta, ldaLocal)
        if (F > Q):
            ik += 1
            L_candidato = (eta**ik)
        else:
            encontrei_L = True
    return L_candidato
#}}}
