import numpy as np
import math

import gradient_descent as gd

#{{{ softThresholding1
def softThresholding1(x, lmbda):
    """soft thresholding operator (Hastie, 2015, cap.2, pag. 15)"""
    return np.sign(x) * np.max(np.abs(x) - lmbda, 0.0)
#}}}

#{{{ softThresholding
def softThresholding(x, lda, regul1stCol = False):
    """soft thresholding operator (Hastie, 2015, cap.2, pag. 15)"""
    if (lda > 0.0):
        sz = max(x.shape)
        xc = np.array(x.copy())
        tmp = np.matrix(x.copy())
        
        tmp[:,0] = np.reshape(np.sign(xc[:,0]) * np.array(map(lambda k : max(abs(k)-lda, 0.0), xc[:,0])), (sz,1))
        if (not regul1stCol):
            tmp[0,0] = xc[0,0]
        return tmp
    else:
        return x
#}}}

#{{{ softThresholding2
def softThresholding2(x, lda):
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

#{{{ costLasso
def costLasso(A, x, b, lmbda, factor = 1.0):
    mA = np.matrix(A)
    mX = np.matrix(x)
    mB = np.matrix(b)
    diff = mA * mX - mB
    l1normX = np.sum(np.abs(mX))
    cst = factor * (diff.T * diff) + lmbda*l1normX
    return cst[0,0]
#}}}

#{{{ CostLasso
class CostLasso:
    def __init__(self, A, b):
        self.mA = np.matrix(A)
        self.mB = np.matrix(b)

    def execute(self, x, lmbda):
        return costLasso(self.mA, x, self.mB, lmbda)
#}}}

#{{{ penaltyLasso
def penaltyLasso(x, lmbda):
    return lmbda * np.sum(np.abs(x))
#}}}

#{{{ penaltyFusedLasso
def penaltyFusedLasso(x, lmbda1, lmbda2):
    parte1 = penaltyLasso(x, lmbda1)
    parte2 = lmnda2 * np.sum(np.abs(x[1:] - x[:-1]))
    return parte1 + parte2
#}}}

#{{{ stepISTA
def stepISTA(meas_A, params_x, obs_b, alpha, lmbda, factor = 2.0, regul1stCol = False): #{
    (tt, grad) = gd.stepGradDescent(meas_A, params_x, obs_b, alpha, factor)
    return (softThresholding(tt, lmbda, regul1stCol), grad)
#}}}

#{{{ ISTA
def ISTA(meas_A, params_x, obs_b, options):
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
    lmbda = 0.0
    if (options.has_key("lambda") and options["lambda"] > 0.0):
        lmbda = options["lambda"]
    regCol1 = False
    if (options.has_key("regularize_1st_col")):
        regCol1 = options["regularize_1st_col"]

    it = 0
    for i in xrange(maxIters): #{
        (x1, grad) = stepISTA(meas_A, x1, obs_b, alpha, lmbda, factor, regCol1)
        if (useLog):
            costLog.append(costLasso(meas_A, x1, obs_b, lmbda, 0.5 * factor))
        if (np.linalg.norm(grad) < thresholdGrad):
            break
        it += 1
    #}
    print "ISTA ran ", it, "iters"
    if (useLog):
        return (x1, costLog)
    else:
        return x1
#}}}

#{{{ stepISTA2
def stepISTA2(meas_A, params_x, obs_b, alpha, lmbda, factor = 2.0, regul1stCol = False): #{
    (tt, grad) = gd.stepGradDescent(meas_A, params_x, obs_b, alpha, factor)
    return (softThresholding(tt, lmbda, regul1stCol), grad)
#}}}

#{{{ ISTA2
def ISTA2(meas_A, params_x, obs_b, options):
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
    lmbda = 0.0
    if (options.has_key("lambda") and options["lambda"] > 0.0):
        lmbda = options["lambda"]
    calcularIntercept = False
    if (options.has_key("calcular_intercept")):
        calcularIntercept = options["calcular_intercept"]
    
    it = 0
    for i in xrange(maxIters): #{
        (x1, grad) = stepISTA(meas_A, x1, obs_b, alpha, lmbda, factor, regCol1)
        if (useLog):
            costLog.append(costLasso(meas_A, x1, obs_b, lmbda, 0.5 * factor))
        if (np.linalg.norm(grad) < thresholdGrad):
            break
        it += 1
    #}
    print "ISTA ran ", it, "iters"
    if (useLog):
        return (x1, costLog)
    else:
        return x1
#}}}


#{{{ stepFISTAConst (substitui o stepFISTAConst do Main_Lobbes_OMP_TSMF)
def stepFISTAConst(meas_A, params_x, obs_b, alpha, lmbda, xk_menos_1, tk, factor = 2.0, regul1stCol = False):
    (xk, grad) = gd.stepGradDescent(meas_A, params_x, obs_b, alpha, factor)
    xk = softThresholding(xk, lmbda, regul1stCol)
    tkp1 = 0.5 * (1.0 + math.sqrt(1.0 + 4.0*tk*tk))
    ykp1 = xk + ((tk-1.0)/tkp1)*(xk - xk_menos_1)
    return (ykp1, xk, tkp1, grad)
#}}}

#{{{ FISTA (substitui o FISTA de Main_Lobbes_OMP_TSMF)
def FISTA(meas_A, params_x, obs_b, options):
    useLog = False
    if (options.has_key("return_log") and options["return_log"]):
        useLog = True
        costLog = []
    maxIters = 100000 # limito a 100.000 iters
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
    lmbda = 0.0
    if (options.has_key("lambda") and options["lambda"] > 0.0):
        lmbda = options["lambda"]
    regCol1 = False
    if (options.has_key("regularize_1st_col")):
        regCol1 = options["regularize_1st_col"]
    
    tk = 1.0
    x1 = params_x.copy()
    xkm1 = params_x.copy()
    it = 0
    for i in xrange(maxIters): #{
        (x1, xkm1, tk, grad) = stepFISTAConst(meas_A, x1, obs_b, alpha, lmbda, xkm1, tk, factor, regCol1)
        if (useLog):
            costLog.append(costLasso(meas_A, x1, obs_b, lmbda, 0.5 * factor))
        if (np.linalg.norm(grad) < thresholdGrad):
            break
        it += 1
    #}
    print "FISTA ran ", it, "iters"
    if (useLog):
        return (x1, costLog)
    else:
        return x1
#}

