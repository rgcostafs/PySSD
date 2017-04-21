import numpy as np
import math

import gradient_descent as gd

#{{{ stepCD
# j eh o indice/dimensao que sera otimizado(a)
def stepCD(A, x, b, j):
    xx2 = np.matrix(x.copy())
    x2 = np.matrix(x.copy())
    A2 = np.matrix(A.copy())
    A2[:,j] = 0.0
    x2[j,:] = 0.0
    xx2[j,0] = ((A[:,j].T * (b - A2*x2)) / (A[:,j].T * A[:,j]))
    return xx2
#}}}

#{{{ coordinateDescent
def coordinateDescent(meas_A, params_x, obs_b, options): #{
    useLog = False
    if (options.has_key("return_log") and options["return_log"]):
        useLog = True
        costLog = []
    maxIters = 100000 # limito a 100.000 iters
    if (options.has_key("iters") and options["iters"] > 0):
        maxIters = options["iters"]
    thresMagParam = 1e-15
    if (options.has_key("thres_mag_param") and options["thres_mag_param"] > 0.0):
        thresMagParam = options["thres_mag_param"]
    n = params_x.shape[0] # quantos parametros == quantas linhas tem no vetor de parametros
    chosen_indices = range(n)
    if (options.has_key("chosen_indices")):
        chosen_indices = options["chosen_indices"]
    x1 = params_x.copy()
    mags = np.array([0.0] * len(chosen_indices))
    for i in xrange(maxIters): #{
        ie = 0
        for j in chosen_indices:
            x2 = stepCD(meas_A, x1, obs_b, j)
            mags[ie] = abs(x2[j,0] - x1[j,0])
            x1[:,:] = x2[:,:]
            ie += 1
            if (useLog):
                costLog.append(gd.costOLS(meas_A, x1, obs_b))
        if (np.max(mags) < thresMagParam):
            break
#        if (i % 10) == 0:
#            print "it cd: ", i, " np.max(mags) ", np.max(mags)
    #}
    if (useLog):
        return (x1, costLog)
    else:
        return x1
#}}}


