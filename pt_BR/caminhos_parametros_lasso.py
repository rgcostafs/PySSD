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

import coordenada_descendente as cd
import gradiente_descendente as gd
import lasso_cd as lsscd
import lasso as lss
import normalizacao as nrm

#{{{ FISTASpams
def FISTASpams(matrizMedidasX, y, theta, lmbda, iteracoes): #{
    print matrizMedidasX.shape
    print y.shape
    print theta.shape
    print lmbda
    print iteracoes

    myfloat = np.float64

    param = {'numThreads' : 1,
             'verbose' : False,
             'lambda1' : lmbda, # parametro de regularizacao. como a reg. aqui e L1, e o lambda que multiplica a parcela de regularizacao da equacao de custo a minimizar
             'it0' : 10, # a cada quantas iteracoes se calcula a lacuna de dualidade
             'max_it' : iteracoes, # maximo de iteracoes 
             'L0' : 0.01,  # Parametro L inicial do FISTA (nao entendi ainda)
             'tol' : 1e-8, # tolerancia para criterio de parada. relativo a mudanca nos parametros ou lacuna de dualidade
             'intercept' : False, # Se verdadeiro, nao regulariza a ultima linha do vetor W de pesos/parametros
             'pos' : False, # Se verdadeiro, adiciona restricoes de positividade nos coeficientes
             'loss': 'square', # Usando perda quadratica
             'regul': 'l1'} # regularizacoo l1: Lasso

    X = np.asfortranarray(matrizMedidasX)
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

#{{{ caminhosParametrosLasso
def caminhosParametrosLasso(medidas_A, observacoes_b, subdivisoes): #{
    maximoLambda = np.max(np.abs(medidas_A.T * observacoes_b))

    ultimoThetaFISTACD = np.matrix(np.zeros(shape=(medidas_A.shape[1],1)))
    ultimoThetaISTACD = np.matrix(np.zeros(shape=(medidas_A.shape[1],1)))
    ultimoThetaISTA_antCD = np.matrix(np.zeros(shape=(medidas_A.shape[1],1)))
    ultimoThetaSPAMS = np.matrix(np.zeros(shape=(medidas_A.shape[1],1)))
    ultimoThetaISTA = np.matrix(np.zeros(shape=(medidas_A.shape[1],1)))
    ultimoThetaFISTA = np.matrix(np.zeros(shape=(medidas_A.shape[1],1)))

    valoresTheta = np.matrix(np.zeros(shape=(medidas_A.shape[1], subdivisoes+1)))
    valoresThetaSPAMS = np.matrix(np.zeros(shape=(medidas_A.shape[1], subdivisoes+1)))
    valoresThetaISTACD = np.matrix(np.zeros(shape=(medidas_A.shape[1], subdivisoes+1)))
    valoresThetaISTACD_ant = np.matrix(np.zeros(shape=(medidas_A.shape[1], subdivisoes+1)))
    valoresThetaFISTACD = np.matrix(np.zeros(shape=(medidas_A.shape[1], subdivisoes+1)))
    valoresThetaISTA = np.matrix(np.zeros(shape=(medidas_A.shape[1], subdivisoes+1)))
    valoresThetaFISTA = np.matrix(np.zeros(shape=(medidas_A.shape[1], subdivisoes+1)))

    opcoes_lasso_cd = {"retornar_log" : False, "iteracoes" : 3000, \
                    "limiar_magnitude_parametro" : 1e-4, \
                    "lambda" : 0.0, \
                    "regularizar_col_1" : True } # forcei regularizacao da constante para comparar com o SPAMS, mas isso deve ser revisto

    opcoes_lasso_gd = {"retornar_log" : False, "iteracoes" : 3000, \
                    "limiar_magnitude_gradiente" : 1e-6, \
                    "lambda" : 0.0, \
                    "regularizar_col_1" : True } # forcei regularizacao da constante para comparar com o SPAMS, mas isso deve ser revisto

    # TODO: Verificar se o ISTA esta iterando sobre todos os indices do vetor de parametros antes de devolver resultado

    for i in range(subdivisoes+1):
        print "CAMINHO i:", i
        lmbda = (i / float(subdivisoes))*maximoLambda
        print "lambda:", lmbda
        opcoes_lasso_cd["lambda"] = lmbda/medidas_A.shape[0]
        opcoes_lasso_gd["lambda"] = lmbda/medidas_A.shape[0]

        # restart para ISTA e FISTA
        ultimoThetaISTA = np.matrix(np.zeros(shape=(medidas_A.shape[1],1)))
        ultimoThetaFISTA = np.matrix(np.zeros(shape=(medidas_A.shape[1],1)))

        novoThetaISTA = ultimoThetaISTA.copy()
        novoThetaFISTA = ultimoThetaISTA.copy()
        novoThetaFISTACD = ultimoThetaISTA.copy()
        novoThetaISTACD = ultimoThetaISTA.copy()
        novoThetaISTACD_ant = ultimoThetaISTA.copy()
        novoThetaSPAMS = FISTASpams(medidas_A, observacoes_b, ultimoThetaSPAMS, lmbda, 1000)

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

        errof = lss.custoLasso(medidas_A, novoThetaFISTACD, observacoes_b, lmbda)
        erroi = lss.custoLasso(medidas_A, novoThetaISTACD, observacoes_b, lmbda)
        erroiA = lss.custoLasso(medidas_A, novoThetaISTACD_ant, observacoes_b, lmbda)
        erros = lss.custoLasso(medidas_A, novoThetaSPAMS, observacoes_b, lmbda)
        errosIG = lss.custoLasso(medidas_A, novoThetaISTA, observacoes_b, lmbda)
        errosFG = lss.custoLasso(medidas_A, novoThetaFISTA, observacoes_b, lmbda)
        print "erros (fista,ista,ista_ant,spams,ista,fista): %.3f  %.3f  %.3f %.3f %.3f %.3f" % (errof, erroi, erroiA, erros, errosIG, errosFG)

    return (valoresThetaFISTACD, valoresThetaISTACD, valoresThetaISTACD_ant, valoresThetaSPAMS, valoresThetaISTA, valoresThetaFISTA)
#}
#}}}


#{{{ caminhosParametrosLasso
def caminhosParametrosLasso_SPAMS(medidas_A, observacoes_b, subdivisoes): #{
    maximoLambda = np.max(np.abs(medidas_A.T * observacoes_b))

    ultimoThetaSPAMS = np.matrix(np.zeros(shape=(medidas_A.shape[1],1)))
    
    valoresThetaSPAMS = np.matrix(np.zeros(shape=(medidas_A.shape[1], subdivisoes+1)))

    for i in range(subdivisoes+1):
        print "CAMINHO i:", i
        lmbda = (i / float(subdivisoes))*maximoLambda
        print "lambda:", lmbda
        novoThetaSPAMS = FISTASpams(medidas_A, observacoes_b, ultimoThetaSPAMS, lmbda, 1000)
        valoresThetaSPAMS[:,i] = novoThetaSPAMS[:]
        ultimoThetaSPAMS[:,0] = novoThetaSPAMS[:]

    return valoresThetaSPAMS
#}
#}}}



#{{{ teste_caminhosParametrosLasso
def teste_caminhosParametrosLasso(): #{   
    print "\nTestando caminhos das solucoes"
    print "Teste Lasso CD"
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
 
    # ignorar1aColuna: a parcela constante nao sera normalizada e nem sera regularizada
    # (A1, tamsA) = nrm.normalizacaoMatrizNormaUnitaria(A, ignorar1aColuna=True)
    (A1, mediasA, stdsA) = nrm.normalizacaoMatrizNormalPadrao(A, ignorar1aColuna=True)
    parametros_x = np.matrix(np.zeros(shape=(POT,1)))

    # salvando as matrizes para possivel uso posterior
    #sio.savemat('teste_caminho_lasso_cd.mat', {"A": A, "b": b, "Anorm": A1, "tamsA": tamsA})
    #np.savez_compressed('teste_caminho_lasso_cd', A=A, b=b, A1=A1, tamsA=tamsA)

    print "matriz original A:"
    print A
    print "matriz normalizada A1:"
    print A1
    print "matriz das observacoes b:"
    print b
    print "matriz dos parametros x:"
    print parametros_x

    t0 = time.time()
    (valoresThetaFISTACD, valoresThetaISTACD, valoresThetaISTA_antCD, valoresThetaSPAMS, valsISTA, valsFISTA) = caminhosParametrosLasso(A1, b, 100)
    elapsed = time.time() - t0
    print "Tempo:", elapsed

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


#{{{ teste_caminhosParametrosLasso_TSMF
def teste_caminhosParametrosLasso_TSMF(): #{   
    print "\nTestando caminhos das solucoes"
    print "Teste Lasso CD TSMF"
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

    # ignorar1aColuna: a parcela constante nao sera normalizada e nem sera regularizada
    # (A1, tamsA) = nrm.normalizacaoMatrizNormaUnitaria(A, ignorar1aColuna=True)
    #(A1, mediasA, stdsA) = nrm.normalizacaoMatrizNormalPadrao(A, ignorar1aColuna=True)
    (A1, mediasA, stdsA) = nrm.normalizacaoMatrizNormalPadrao(A, ignorar1aColuna=False)
    parametros_x = np.matrix(np.zeros(shape=(POT,1)))

    # salvando as matrizes para possivel uso posterior
    #sio.savemat('teste_caminho_lasso_cd.mat', {"A": A, "b": b, "Anorm": A1, "tamsA": tamsA})
    #np.savez_compressed('teste_caminho_lasso_cd', A=A, b=b, A1=A1, tamsA=tamsA)

    print "matriz original A:"
    print A
    print "matriz normalizada A1:"
    print A1
    print "matriz das observacoes b:"
    print b
    print "matriz dos parametros x:"
    print parametros_x

    t0 = time.time()
    (valoresThetaFISTACD, valoresThetaISTACD, valoresThetaISTA_antCD, valoresThetaSPAMS, valsISTA, valsFISTA) = caminhosParametrosLasso(A1, b, 100)
    elapsed = time.time() - t0
    print "Tempo:", elapsed

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
    teste_caminhosParametrosLasso_TSMF()

