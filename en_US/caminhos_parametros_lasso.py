import numpy as np
import math
import random
import spams # SParse Modeling Software
import time
import sys
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import time

import coordinate_descent as cd
import gradient_descent as gd
import lasso_cd as lsscd
import lasso as lss
import normalization as nrm

#{{{ FISTASpams
def FISTASpams(matMeasX, y, theta, lmbda, iters): #{
    print matMeasX.shape
    print y.shape
    print theta.shape
    print lmbda
    print iters

    myfloat = np.float64

    param = {'numThreads' : 1,
             'verbose' : False,
             'lambda1' : lmbda, # parametro de regularizacao. como a reg. aqui e L1, e o lambda que multiplica a parcela de regularizacao da equacao de custo a minimizar
             'it0' : 10, # a cada quantas iters se calcula a lacuna de dualidade
             'max_it' : iters, # maximo de iters 
             'L0' : 0.01,  # Parametro L inicial do FISTA (nao entendi ainda)
             'tol' : 1e-8, # tolerancia para criterio de parada. relativo a mudanca nos parametros ou lacuna de dualidade
             'intercept' : False, # Se verdadeiro, nao regulariza a ultima linha do vetor W de pesos/parametros
             'pos' : False, # Se verdadeiro, adiciona restricoes de positividade nos coeficientes
             'loss': 'square', # Usando perda quadratica
             'regul': 'l1'} # regularizacoo l1: Lasso

    X = np.asfortranarray(matMeasX)
#    X = np.asfortranarray(X - np.tile(np.mean(X,0),(X.shape[0],1)),dtype=myfloat)
#    X = spams.normalize(X)
    
    Y = np.asfortranarray(y)
#    Y = np.asfortranarray(Y - np.tile(np.mean(Y,0),(Y.shape[0],1)),dtype=myfloat)
#    Y = spams.normalize(Y)

    W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=myfloat,order="FORTRAN")
    
    (W, optim_info) = spams.fistaFlat(Y,X,W0,True,**param)
    return W
#}
#}}}

#{{{ paramsPathLasso
def paramsPathLasso(meas_A, obs_b, subdivs): #{
    maximoLambda = np.max(np.abs(meas_A.T * obs_b))

    ultimoThetaFISTACD = np.matrix(np.zeros(shape=(meas_A.shape[1],1)))
    ultimoThetaISTACD = np.matrix(np.zeros(shape=(meas_A.shape[1],1)))
    ultimoThetaISTA_antCD = np.matrix(np.zeros(shape=(meas_A.shape[1],1)))
    ultimoThetaSPAMS = np.matrix(np.zeros(shape=(meas_A.shape[1],1)))
    ultimoThetaISTA = np.matrix(np.zeros(shape=(meas_A.shape[1],1)))
    ultimoThetaFISTA = np.matrix(np.zeros(shape=(meas_A.shape[1],1)))

    valoresTheta = np.matrix(np.zeros(shape=(meas_A.shape[1], subdivs+1)))
    valoresThetaSPAMS = np.matrix(np.zeros(shape=(meas_A.shape[1], subdivs+1)))
    valoresThetaISTACD = np.matrix(np.zeros(shape=(meas_A.shape[1], subdivs+1)))
    valoresThetaISTACD_ant = np.matrix(np.zeros(shape=(meas_A.shape[1], subdivs+1)))
    valoresThetaFISTACD = np.matrix(np.zeros(shape=(meas_A.shape[1], subdivs+1)))
    valoresThetaISTA = np.matrix(np.zeros(shape=(meas_A.shape[1], subdivs+1)))
    valoresThetaFISTA = np.matrix(np.zeros(shape=(meas_A.shape[1], subdivs+1)))

    opts_lasso_cd = {"return_log" : False, "iters" : 3000, \
                    "thres_mag_param" : 1e-4, \
                    "lambda" : 0.0, \
                    "regularize_1st_col" : True } # forcei regularizacao da constante para comparar com o SPAMS, mas isso deve ser revisto

    opts_lasso_gd = {"return_log" : False, "iters" : 3000, \
                    "threshold_mag_gradient" : 1e-6, \
                    "lambda" : 0.0, \
                    "regularize_1st_col" : True } # forcei regularizacao da constante para comparar com o SPAMS, mas isso deve ser revisto

    # TODO: Verificar se o ISTA esta iterando sobre todos os indices do vetor de parametros antes de devolver resultado

    for i in range(subdivs+1):
        print "PATH_STEP i:", i
        lmbda = (i / float(subdivs))*maximoLambda
        print "lambda:", lmbda
        opts_lasso_cd["lambda"] = lmbda/meas_A.shape[0]
        opts_lasso_gd["lambda"] = lmbda/meas_A.shape[0]

        # restart para ISTA e FISTA
        ultimoThetaISTA = np.matrix(np.zeros(shape=(meas_A.shape[1],1)))
        ultimoThetaFISTA = np.matrix(np.zeros(shape=(meas_A.shape[1],1)))

        novoThetaISTA = ultimoThetaISTA.copy()
        novoThetaFISTA = ultimoThetaISTA.copy()
        novoThetaFISTACD = ultimoThetaISTA.copy()
        novoThetaISTACD = ultimoThetaISTA.copy()
        novoThetaISTACD_ant = ultimoThetaISTA.copy()
        novoThetaSPAMS = FISTASpams(meas_A, obs_b, ultimoThetaSPAMS, lmbda, 1000)

        valoresThetaISTA[:,i] = novoThetaISTA[:,0]
        valoresThetaFISTA[:,i] = novoThetaFISTA[:,0]
        valoresThetaFISTACD[:,i] = novoThetaFISTACD[:,0]
        valoresThetaISTACD[:,i] = novoThetaISTACD[:,0]
        valoresThetaISTACD_ant[:,i] = novoThetaISTACD_ant[:,0]
        valoresThetaSPAMS[:,i] = novoThetaSPAMS[:]

        ultimoThetaISTA[:,:] = novoThetaISTA[:,:]
        ultimoThetaFISTA[:,:] = novoThetaFISTA[:,:]
        ultimoThetaFISTACD[:,:] = novoThetaFISTACD[:,:]
        ultimoThetaISTACD[:,:] = novoThetaISTACD[:,:]
        ultimoThetaISTA_antCD[:,:] = novoThetaISTACD_ant[:,:]
        ultimoThetaSPAMS[:,0] = novoThetaSPAMS[:]

        errof = lss.costLasso(meas_A, novoThetaFISTACD, obs_b, lmbda)
        erroi = lss.costLasso(meas_A, novoThetaISTACD, obs_b, lmbda)
        erroiA = lss.costLasso(meas_A, novoThetaISTACD_ant, obs_b, lmbda)
        errors = lss.costLasso(meas_A, novoThetaSPAMS, obs_b, lmbda)
        errorsIG = lss.costLasso(meas_A, novoThetaISTA, obs_b, lmbda)
        errorsFG = lss.costLasso(meas_A, novoThetaFISTA, obs_b, lmbda)
        print "errors (fista,ista,ista_ant,spams,ista,fista): %.3f  %.3f  %.3f %.3f %.3f %.3f" % (errof, erroi, erroiA, errors, errorsIG, errorsFG)

    return (valoresThetaFISTACD, valoresThetaISTACD, valoresThetaISTACD_ant, valoresThetaSPAMS, valoresThetaISTA, valoresThetaFISTA)
#}
#}}}


#{{{ paramsPathLasso
def paramsPathLasso_SPAMS(meas_A, obs_b, subdivs): #{
    maximoLambda = np.max(np.abs(meas_A.T * obs_b))

    ultimoThetaSPAMS = np.matrix(np.zeros(shape=(meas_A.shape[1],1)))
    
    valoresThetaSPAMS = np.matrix(np.zeros(shape=(meas_A.shape[1], subdivs+1)))

    for i in range(subdivs+1):
        print "PATH_STEP i:", i
        lmbda = (i / float(subdivs))*maximoLambda
        print "lambda:", lmbda
        novoThetaSPAMS = FISTASpams(meas_A, obs_b, ultimoThetaSPAMS, lmbda, 1000)
        valoresThetaSPAMS[:,i] = novoThetaSPAMS[:]
        ultimoThetaSPAMS[:,0] = novoThetaSPAMS[:]

    return valoresThetaSPAMS
#}
#}}}



#{{{ teste_paramsPathLasso
def teste_paramsPathLasso(): #{   
    print "\nTesting solution paths"
    print "Test Lasso CD"
    # Geracao do dado sintetico de teste
    # Sao gerados 10 pontos para se fazer o ajuste de uma curva de ate 3 grau
    SZ = 10
    POT = 4 # grau maximo = 4-1 = 3
    x = SZ * np.random.random(SZ)
    x = np.sort(x)
    y = (1. + 0.5*np.random.random(SZ)) * x ** (1. + 0.5*np.random.random(SZ)) + \
        (1. + 0.2 * np.random.random(SZ))

    # Montagem da matriz de medidas e da matriz de observacoes como se fossemos resolver minimos quadrados
    A = np.matrix(np.zeros(shape=(SZ,POT)))
    b = np.matrix(np.zeros(shape=(SZ,1)))
    for r in range(10):
        b[r] = y[r]
        for c in range(A.shape[1]):
            A[r,c] = x[r]**c
 
    # ignore1stCol: a parcela constante nao sera normalizada e nem sera regularizada
    # (A1, tamsA) = nrm.normalizationMatrizNormaUnitaria(A, ignore1stCol=True)
    (A1, meansA, stdsA) = nrm.normalizationMatrizNormalPadrao(A, ignore1stCol=True)
    params_x = np.matrix(np.zeros(shape=(POT,1)))

    # salvando as matrizes para possivel uso posterior
    #sio.savemat('teste_caminho_lasso_cd.mat', {"A": A, "b": b, "Anorm": A1, "tamsA": tamsA})
    #np.savez_compressed('teste_caminho_lasso_cd', A=A, b=b, A1=A1, tamsA=tamsA)

    print "original matrix A:"
    print A
    print "normalized matrix A1:"
    print A1
    print "observations b:"
    print b
    print "parameters x:"
    print params_x

    t0 = time.time()
    (valoresThetaFISTACD, valoresThetaISTACD, valoresThetaISTA_antCD, valoresThetaSPAMS, valsISTA, valsFISTA) = paramsPathLasso(A1, b, 100)
    elapsed = time.time() - t0
    print "Elapsed Time:", elapsed

    for i in xrange(valoresThetaISTACD.shape[0]):
        rw = np.zeros(shape=(valoresThetaISTACD.shape[1],))
        rw[:] = valoresThetaISTACD[i,:]
        plt.plot(rw)
    plt.show()

    for i in xrange(valoresThetaSPAMS.shape[0]):
        plt.plot(range(valoresThetaSPAMS.shape[1]), valoresThetaSPAMS[i,:])
    plt.show()
#}
#}}}


#{{{ teste_paramsPathLasso_TSMF
def teste_paramsPathLasso_TSMF(): #{   
    print "\nTesting solution paths"
    print "Test Lasso CD TSMF"
    # Geracao do dado sintetico de teste
    # Sao gerados 10 pontos para se fazer o ajuste de uma curva de ate 3 grau
    SZ = 10
    POT = 4 # grau maximo = 4-1 = 3
    x = SZ * np.random.random(SZ)
    x = np.sort(x)
    y = (1. + 0.5*np.random.random(SZ)) * x ** (1. + 0.5*np.random.random(SZ)) + \
        (1. + 0.2 * np.random.random(SZ))

    # Montagem da matriz de medidas e da matriz de observacoes como se fossemos resolver minimos quadrados
    A = np.matrix(np.zeros(shape=(SZ,POT)))
    b = np.matrix(np.zeros(shape=(SZ,1)))
    for r in range(10):
        b[r] = y[r]
        for c in range(A.shape[1]):
            A[r,c] = x[r]**c

    ############################################################# 
    # Sobrescrevendo com uma matriz de Toeplitz com uma wavelet #
    #############################################################

    A = np.matrix(np.load("matrizToeplitzA.npy"))
    b = np.matrix(np.load("observacao.npy"))
    POT = A.shape[1]
    SZ = b.shape[0]

    # ignore1stCol: a parcela constante nao sera normalizada e nem sera regularizada
    # (A1, tamsA) = nrm.normalizationMatrizNormaUnitaria(A, ignore1stCol=True)
    #(A1, meansA, stdsA) = nrm.normalizationMatrizNormalPadrao(A, ignore1stCol=True)
    (A1, meansA, stdsA) = nrm.normalizationMatrizNormalPadrao(A, ignore1stCol=False)
    params_x = np.matrix(np.zeros(shape=(POT,1)))

    # salvando as matrizes para possivel uso posterior
    #sio.savemat('teste_caminho_lasso_cd.mat', {"A": A, "b": b, "Anorm": A1, "tamsA": tamsA})
    #np.savez_compressed('teste_caminho_lasso_cd', A=A, b=b, A1=A1, tamsA=tamsA)

    print "original matrix A:"
    print A
    print "normalized matrix A1:"
    print A1
    print "observations b:"
    print b
    print "parameters x:"
    print params_x

    t0 = time.time()
    (valoresThetaFISTACD, valoresThetaISTACD, valoresThetaISTA_antCD, valoresThetaSPAMS, valsISTA, valsFISTA) = paramsPathLasso(A1, b, 100)
    elapsed = time.time() - t0
    print "Elapsed Time:", elapsed

    fmt = ['r-','g-','b-','k-','c-','m-','y-']

    arow = np.array(valoresThetaSPAMS.shape[1] * [0.0])
    for i in xrange(valoresThetaSPAMS.shape[0]):
        arow[:] = valoresThetaSPAMS[i,:]
        plt.plot(arow, fmt[i%7])
        if (i % 10)==0:
            print "%d/%d" % (i, valoresThetaSPAMS.shape[0])
    plt.show()
#}
#}}}


if __name__ == "__main__":
    teste_paramsPathLasso_TSMF()

