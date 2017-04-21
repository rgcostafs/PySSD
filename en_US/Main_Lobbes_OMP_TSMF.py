# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 20:59:40 2016

@author: Rodrigo
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import spams # SParse Modeling Software
import time
import sys
import os

import scipy.io as sio
import scipy

from scipy import signal

import simulacao
import gradient_descent as gd
import lasso as lss
import coordinate_descent as cd
import lasso_cd
import normalization as nrm


def initThetaOMP(matMeasurementsA, y, K): #{
    mm = matMeasurementsA.copy()
    szs = []
    for j in range(mm.shape[1]):
        sz = np.linalg.norm(mm[:,j])
        szs.append(sz)
        mm[:,j] = mm[:,j] / sz
    D = np.asfortranarray(mm, dtype=np.float64)
    X = np.asfortranarray(y, dtype=np.float64)
    res = spams.omp(X, D, L=K, eps=1.0e-20, return_reg_path = False, numThreads = 2)
    res = res.todense()
    for j in range(mm.shape[1]):
        if szs[j] != 0.0:
            res[j] = res[j] / szs[j]
    return res
#}


def FusedLassoFistaFlatSpams(A, obs_Y, params_theta, lambda1, lambda2, lambda3): #{

    #(matA, szA) = normalizationMatrizNorma1(A) 
    #(matA, maxA) = normalizationMatrizMaximo1(A) 
    matA = A

    myfloat = np.float64

    if ((lambda2 == 0.0) and (lambda3 == 0.0)): # lasso puro
        print "Lasso..."
        param = {'numThreads' : 2,
                 'verbose' : False,
                 'lambda1' : lambda1, # parametro de regularização aplicado à norma L1 das diffces dos parametros consecutivos
                 'lambda2' : 0.0,     # parametro de regularização aplicado à norma L1 do vetor dos parametros!
                 'lambda3' : 0.0, # parametro de regularização aplicado à norma L2 do vetor dos parametros
                 'it0' : 10, # a cada quantas iterações se calcula a lacuna de dualidade
                 'max_it' : 200, # maximo de iterações 
                 'L0' : 0.01,  # Parametro L inicial do FISTA (não entendi ainda)
                 'tol' : 1e-4, # tolerancia para critério de parada. relativo à mudança nos parâmetros ou lacuna de dualidade
                 'intercept' : False, # Se verdadeiro, não regulariza a última linha do vetor W de pesos/parametros
                 'pos' : False, # Se verdadeiro, adiciona restrições de positividade nos coeficientes
                 'loss': 'square', # Usando perda quadrática
                 'regul': 'l1'} # regularização lasso - l1
    elif ((lambda2 == 0.0) and (lambda3 > 0.0)): # elastic-net
        print "Elastic Net..."
        param = {'numThreads' : 2,
             'verbose' : False,
             'lambda1' : lambda1, # parametro de regularização aplicado à norma L1 das diffces dos parametros consecutivos
             'lambda2' : lambda3, # parametro de regularização aplicado à norma L1 do vetor dos parametros!
             'lambda3' : 0.0, # parametro de regularização aplicado à norma L2 do vetor dos parametros
             'it0' : 10, # a cada quantas iterações se calcula a lacuna de dualidade
             'max_it' : 200, # maximo de iterações 
             'L0' : 0.01,  # Parametro L inicial do FISTA (não entendi ainda)
             'tol' : 1e-4, # tolerancia para critério de parada. relativo à mudança nos parâmetros ou lacuna de dualidade
             'intercept' : False, # Se verdadeiro, não regulariza a última linha do vetor W de pesos/parametros
             'pos' : False, # Se verdadeiro, adiciona restrições de positividade nos coeficientes
             'loss': 'square', # Usando perda quadrática
             'regul': 'elastic-net'} # regularização elastic-net
    else: # fused-lasso
        print "Fused Lasso..."
        param = {'numThreads' : 2,
             'verbose' : False,
             'lambda1' : lambda2, # parametro de regularização aplicado à norma L1 das diffces dos parametros consecutivos
             'lambda2' : lambda1, # parametro de regularização aplicado à norma L1 do vetor dos parametros!
             'lambda3' : lambda3, # parametro de regularização aplicado à norma L2 do vetor dos parametros
             'it0' : 10, # a cada quantas iterações se calcula a lacuna de dualidade
             'max_it' : 200, # maximo de iterações 
             'L0' : 0.01,  # Parametro L inicial do FISTA (não entendi ainda)
             'tol' : 1e-4, # tolerancia para critério de parada. relativo à mudança nos parâmetros ou lacuna de dualidade
             'intercept' : False, # Se verdadeiro, não regulariza a última linha do vetor W de pesos/parametros
             'pos' : False, # Se verdadeiro, adiciona restrições de positividade nos coeficientes
             'loss': 'square', # Usando perda quadrática
             'regul': 'fused-lasso'} # regularização elastic-net


    X = np.asfortranarray(matA)
    Y = np.asfortranarray(obs_Y)

    W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=myfloat,order="FORTRAN")
    
    # faco a copia do valor anteriormente obtido
    W0[:] = params_theta[:]
    
    print "shapes: X: ", X.shape, "Y:", Y.shape

    (W, optim_info) = spams.fistaFlat(Y,X,W0,True,**param)

#    for j in range(W.shape[0]):
#        W[j,:] = W[j,:] * szA[j]

    return W
#}


def costSparseFactorizationToeplitz(matToeplitzA, matRefletR, matObsY, lmbda, beta, beta1, baseWavelet): #{
    diff = matToeplitzA * matRefletR - matObsY
    frobNorm2 = np.linalg.norm(diff)**2 
    l1normReflet = np.sum(np.abs(matRefletR))
    alphas = toeplitzMat.toeplitz2Alpha3(matToeplitzA, baseWavelet)
    l1normAlphas = np.sum(np.abs(alphas))
    diffAlphas = alphas[1:] - alphas[:-1]
    l1diffAlphas = np.sum(np.abs(diffAlphas))
    cst = 0.5*frobNorm2 + lmbda*l1normReflet + beta*l1normAlphas + beta1*l1diffAlphas
    return cst
#}


def localDiagMat(lado, posicao): #{
    assert(abs(posicao) <= (lado-1))
    Ik = np.matrix(np.zeros(shape=(lado,lado)))
    for p in range(lado):
        i = p - posicao
        j = posicao + i
        if ((i>=0) and (j>=0) and (i<lado) and (j<lado)):
            Ik[i,j] = 1.0
    return Ik
#}


def vecPosRk(pos, Rk): #{
    lado = Rk.shape[0]
    Ik = localDiagMat(lado, pos)
    vec = Ik * Rk
    vec = np.matrix(vec.T.flatten()).T
    return vec
#}

#{{{ FistaTSMF
def FistaTSMF(A, x, b, sparsK, thres):
    r0 = np.matrix(x)
    rk = r0.copy()
    t0 = 1.
    z0 = r0.copy()
    k = 0
    mi0 = 0.99 / (np.linalg.norm(A)**2)
    converged = False
    for k in xrange(10001):
        k += 1
        mik = mi0
        e = b - A*r0
        uk = A.T * e
        bk = r0 + mik*uk
        bkArr = np.abs(np.array(bk))
        bkASort = np.argsort(bkArr[:,0])[::-1]
        lmbdaMi = (bkArr[bkASort[sparsK]])[0]
        zk = np.zeros(shape=b.shape)
        zk[bkASort[:sparsK]] = bk[bkASort[:sparsK]]
        zk = np.sign(zk) * (np.abs(zk) - 0.5 * lmbdaMi)
        tk = 0.5 * (1. + math.sqrt(1.0 + 4.0*t0*t0))
        zk = np.matrix(np.reshape(zk, r0.shape))
        rk = zk + ((t0 - 1.0)/tk) * (zk - z0)
        diff = rk - r0
        r0 = rk.copy()
        z0 = zk.copy()
        t0 = tk
        if (np.linalg.norm(diff) < thres):
            break
    #}
    print "iters:", k
    return rk
#}}}

#{{{ TSMF2
def TSMF2(matObservationsY, sparsityK, toeplitzA0, wavelet0, lambda1 = 1.0, lambda2 = 0.1, lambda3 = 0.01, diretorio = "./", ref = [], algor = "FistaTSMF"):
    # uma observacao por COLUNA, como nos metodos convencionais
    print "####### TSMF 2 : Algorithm %s #######" % (algor)
    k = 0
    custoAnterior = 1e30
    convergiu = False
    yTil = np.matrix(matObservationsY.T.flatten()).T
    numTracos = matObservationsY.shape[1] # o numero de traces eh o numero de colunas na observations
    # print "Numero de traces:", numTracos
    # inicializa a matriz de reflectivitys SEM REFLETORES
    matrizRefletividade = np.matrix(np.zeros(shape=matObservationsY.shape))
    nl = np.prod(matrizRefletividade.shape)
    # a quantidade de colunas da matriz RkTil é a quantidade de linhas da wavelet (tamanho)
    nc = max(wavelet0.shape)
    hf = matrizRefletividade.shape[0]
    alphaKmenos1 = toeplitzMat.toeplitz2Alpha3(toeplitzA0, wavelet0)
    #print "verificando validade das informacoes de wavelet"
    #print "formato da matriz alphaKmenos1:", alphaKmenos1.shape
    #print "formato da matriz de Toeplitz passada:", toeplitzA0.shape
    #print "formato da wavelet0:", wavelet0.shape
    # ajuste dos valores da wavelet (desfeito para FistaTSMF, feito para OMP sem reajuste)
    maxv = np.max(np.abs(alphaKmenos1))
    alphaKmenos1 = alphaKmenos1 / maxv
    #print "ref.shape:", ref.shape
    matrizToeplitzA = toeplitzMat.alpha2Toeplitz3(alphaKmenos1, ref)
    #print "Formato da matriz antes de processar:", matrizToeplitzA.shape
    np.savez(diretorio+"argumentos", numTracos=numTracos, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, wavelet0=wavelet0, referencia=ref)

    options = {"return_log" : False, "iters" : 5000, \
              "thres_mag_param" : 1e-3, \
              "threshold_mag_gradient" : 1e-3, \
              "regularize_1st_col" : True, \
              "num_amostras_esparsidade" : 10}

    costLog = []

    cacheLambda = np.zeros(shape=(numTracos,))
    cacheIndices = {}
    cacheSolucoes = {}

    while (not convergiu): #{
        print "iteration ", k
        Rk = np.matrix(np.zeros(shape=matrizRefletividade.shape))

        # normalização forçada
        # (matrizToeplitzA, szA) = nrm.normalizationMatrizMaximo1(matrizToeplitzA)

        

        for j in range(numTracos): #{
            print "Processing trace ", j
            yj = matObservationsY[:,j].copy()
            rj = matrizRefletividade[:,j].copy()
            
            #print matrizToeplitzA.shape, "antes de salvar!"
            #np.save("matrizToeplitzA", matrizToeplitzA)
            #print matrizToeplitzA.shape, "depois de salvar!"
            #np.save("observacao", yj)

            if (algor == "FistaTSMF"):
                rjk = FistaTSMF(matrizToeplitzA, rj, yj, sparsityK, 1e-4)
            elif (algor == "OMP"):
                rjk = initThetaOMP(matrizToeplitzA, yj, sparsityK) # funciona bem quando a quantidade de picos coincide com a
            elif (algor == "NLasso"):
                ((b0, rjk), cacheLambda[j]) = lasso_cd.lobbes(matrizToeplitzA, rj, yj, sparsityK, options, cacheLambda[j])

            # reequilibrio dos coeficientes restantes
            # fazendo ajuste posterior via MQO #{ # OMP nao precisa disso!
            applyAdjustment = True
            if (algor == "OMP"):
                applyAdjustment = False

            if applyAdjustment: #{
                refletMin = np.min(np.abs(rjk)) # o valor minimo pode ser zero
                #print "tentando aplicar MQO depois da selecao de variaveis"
                params_x = rjk.copy()
                indices_selecionados = []
                for i in xrange(params_x.shape[0]):
                    if abs(params_x[i,0]) > refletMin:
                        indices_selecionados.append(i)
                    else:
                        params_x[i,0] = 0.
    
                if cacheIndices.has_key(j):
                    if (cmp(cacheIndices[j], indices_selecionados) == 0):
                        params_x = cacheSolucoes[j].copy()
                
                cacheIndices[j] = indices_selecionados    

                options_cd = { "return_log" : False, "iters" : 300, \
                              "thres_mag_param" : 1e-4, \
                              "chosen_indices": indices_selecionados }
                t0 = time.time()
                ajuste_rjk = cd.coordinateDescent(matrizToeplitzA, params_x, yj, options_cd)
                et0 = time.time() - t0
    #            print "resultado do AJUSTE pos-selecao deconvolucao (antes: vermelho; depois: azul) %.3f s" % et0
    #            plt.plot(rjk,'r-')
    #            plt.plot(ajuste_rjk,'b-')
    #            plt.show()
    #           
                cacheSolucoes[j] = ajuste_rjk.copy()
 
                rjk = ajuste_rjk.copy()
            # } # fim ajuste posterior
            else:
                print "Rodando OMP. Nao faz ajuste posterior dos parametros"


            Rk[:,j] = rjk
            #np.save(diretorio+"refletores_it_%d_tr_%d" % (k,j), Rk) 
        #} # end for

        RkTil = np.matrix(np.zeros(shape=(nl,nc)))
        for i in range(-nc/2,1+nc/2):
            RkTil[:,i+nc/2] = vecPosRk(i, Rk)

        #np.savez(diretorio+("RK_it_%03d" % k), RkTil=RkTil, Rk=Rk)
        #print "formato de RkTil:", RkTil.shape
        
        # Forçar o recorte
        # Nesta versao está com 201 amostras
        windowSize = 201
        alphaKMenos1Invertida = alphaKmenos1.copy()[::-1,:]
        # print "formato de alphaKMenos1Recortada:", alphaKMenos1Invertida.shape

        alphaKTemp = FusedLassoFistaFlatSpams(RkTil, yTil, alphaKMenos1Invertida, lambda1, lambda2, lambda3)
        alphaK = alphaKmenos1.copy() # somente copia a forma
        alphaK[:,:] = alphaKTemp.copy()[::-1, :] # agora copia o conteudo

        # normalization forcada do alphaK. Com OMP tem efeito positivo.
        maxv = np.max(np.abs(alphaK))
        alphaK = alphaK / maxv
        #np.save(diretorio+("wavelet_it_%03d" % k), alphaK)
        ##plt.plot(alphaK, 'c-')
        ##plt.show()

        Ak = toeplitzMat.alpha2Toeplitz3(alphaK, ref)
    
        cst = costSparseFactorizationToeplitz(matrizToeplitzA, Rk, matObservationsY, lambda1, lambda2, lambda3, baseWavelet=alphaK)
        convergiu = abs(cst - custoAnterior) < 1e-4
        # print "Custo depois de 'otimizar' a matriz de Toeplitz: ", cst
        
        costLog.append(cst)
        if (not convergiu):
            custoAnterior = cst
            matrizToeplitzA = Ak.copy()
            matrizRefletividade = Rk.copy()
            alphaKmenos1 = alphaK.copy()
            k += 1
            if (k==50):
                break
    #}
    print "! %d iterations" % k
    np.savez(diretorio+"resultados", matrizRefletividade=matrizRefletividade, matrizToeplitzA=matrizToeplitzA, matObservationsY=matObservationsY)
    return(matrizToeplitzA, matrizRefletividade, costLog)
#}}}


import rickerwav
from noisegen import *
from marmousi2model import *
from contrastimped import *
from amplitude import *
import toeplitzMat

#{{{ Test de execucao do procedimento de Wang et al. (2016)
def mainTest():
    reflectivitys = generateConstrastImpedancyWang16()
    show200TracesContrastImpedModel()
    reflectivitysSelecionadas = reflectivitys[::10]
    ruido = 0.05
    angle = -45.0
    freq = 30.0
    print "Using ricker with angle %.2f and dominant freq %.2f" % (angle, freq)
    showRickerWang16_2(angle, freq)
    displayAmplitudesWithNoise(ruido, angle, freq)
    #observacoes = np.asfortranarray(amplitudesWithNoiseWang16(ruido, angle, freq)[:,::10])
    #np.save("observacoes_multi", observacoes) # deixar salvando somente na primeira rodada.
    observacoes = np.asfortranarray(np.matrix(np.load("observacoes_multi.npy")))
    # print "forma das observacoes: ", observacoes.shape

    ref = generateConstrastImpedancyWang16()[2]
    np.save("reflectivity_2", ref)
    # print "Salvei a reflectivity"
    
    ric = generateRickerWang16(-30., 30.)
    np.save("wav_gabarito", ric)
    # print "Salvei a wavelet gabarito"
    
    # NAO EH NECESSARIO normalizar as observacoes
    #tracesAmplitude = np.matrix(observacoes.copy().reshape(len(observacoes),1)) # para o caso de usar somente 1 trace
    tracesAmplitude = np.matrix(observacoes.copy()) # para o caso de usar varios traces
    # (tracesAmplitude, szA1) = nrm.normalizationMatrizNormaUnitaria(tracesAmplitude)

    # traces de amplitude
    # cmap = 'spectral'
    # lims = (np.min(tracesAmplitude), np.max(tracesAmplitude))
    # imgplot = plt.imshow(tracesAmplitude.T, clim=lims, cmap=cmap)
    # plt.colorbar()
    # plt.show(imgplot)
    
    l1 =  max(0.2, 2.*ruido)
    l2 =  max(0.1, ruido)
    l3 =  max(0.02, ruido / 5.)

    dt = time.localtime()
    print "#####\nExecution params:"
    print "time: ", time.time()
    strinstante = "%04d_%02d_%02d_%02d_%02d_%02d" % (dt.tm_year, dt.tm_mon, dt.tm_mday, dt.tm_hour, dt.tm_min, dt.tm_sec)
    os.makedirs(strinstante, 0755)
    print "date/time: ", strinstante
    print "l1: ", l1
    print "l2: ", l2
    print "l3: ", l3
    strinstante += "/"

    # usando uma wavelet inicial de fase 0
    wavelet0 = rickerwav.generateRickerWang16(angle = 0.0, frequencyDominante = 30.0)
    wavMat = np.matrix(np.array(wavelet0).reshape(len(wavelet0), 1))
    matToeplitzInicial = toeplitzMat.alpha2Toeplitz3(wavMat, ref)

    # A matriz características é formada pela multiplicacao da matriz de Toeplitz e da matriz de reflectivity
    # Todas as caracteristicas tem a mesma faixa de valores e o mesmo peso. Portanto, nao eh necessario normalizar.
    print "this the initial wavelet"
    plt.plot(wavelet0,'b-',linewidth=3.0)
    plt.show()
    print "showing reference refletivities"
    plt.plot(ref,'r-',linewidth=3.0)
    plt.show()
    print "showing amplitudes (observations) that result from convolving the wavelet and the refletivities"

    # se estiver usando varios traces de entrada
    cmap = 'gray'
    lims = (np.min(observacoes), np.max(observacoes))
    imgplot = plt.imshow(observacoes, clim=lims, cmap=cmap, aspect=0.04)
    plt.colorbar()
    plt.show(imgplot)

    # se estiver usando somente um trace
#    plt.plot(observacoes, 'k-',linewidth=2.0)
#    plt.show()

    # teste = "FistaTSMF_MUL"
    # algor = "FistaTSMF"
    #teste = "OMP_MUL"
    #algor = "OMP"
    teste = "NLasso_MUL"
    algor = "NLasso"

    for picos in [15,20, 25, 30, 35, 40]: #{
        dt = time.localtime()
        print "#####\nExecution params:"
        print "time: ", time.time()
        strinstante = "%04d_%02d_%02d_%02d_%02d_%02d" % (dt.tm_year, dt.tm_mon, dt.tm_mday, dt.tm_hour, dt.tm_min, dt.tm_sec)
        os.makedirs(strinstante, 0755)
        print "date/time: ", strinstante
        print "l1: ", l1
        print "l2: ", l2
        print "l3: ", l3
        strinstante += "/"

        print "running procedure trying to find ", picos, " peaks"
        print "method being tested:", teste
        (A,R,costLog) = TSMF2(tracesAmplitude, picos, matToeplitzInicial, wavelet0, lambda1 = l1, lambda2 = l2, lambda3 = l3, diretorio = strinstante, ref = ref, algor = algor)
        plt.plot(costLog)
        plt.show()
        #print "Para ver se as matrizes de resultado estão realmente corretas"
        #print A.shape
        #print R.shape        
        
        
        print "showing resulting wavelet"
        alphas = toeplitzMat.toeplitz2Alpha3(A, wavelet0)
        plt.plot(alphas,linewidth=2.0)
        plt.show()
        
        np.save("Toep%02d_%s" % (picos, teste), A)
        np.save("Ref%02d_%s" % (picos, teste), R)
        np.save("Wav%02d_%s" % (picos, teste), alphas)
        
        print "comparing original and obtainded reflectivity"
        print "reference in red, obtained in blue"
        plt.plot(ref,'r-',linewidth=2.0)
        plt.plot(R,'b-',linewidth=2.0)
        plt.show()
        
        print "comparing original and obtained amplitudes (red: original; blue: extracted)"
        plt.plot(tracesAmplitude,'r-',linewidth=2.0)
        plt.plot(A*R,'b-',linewidth=2.0)
        plt.show()
        
        (minv, maxv) = (np.min(tracesAmplitude), np.max(tracesAmplitude))
        erro = tracesAmplitude - A*R
        erro2 = np.linalg.norm(erro.T)**2
        print "error2:", erro2
        mse = erro2 / float(np.prod(tracesAmplitude.shape))
        print "MSE:", mse
        print "PSNR:", 20. * math.log10((maxv-minv) / math.sqrt(mse))
    #}

    cmap = 'spectral'
    lims = (np.min(A), np.max(A))
    imgplot = plt.imshow(A, clim=lims, cmap=cmap)
    plt.colorbar()
    plt.show(imgplot)
    
    cmap = 'spectral'
    lims = (np.min(R), np.max(R))
    imgplot = plt.imshow(R, clim=lims, cmap=cmap)
    plt.colorbar()
    plt.show(imgplot)
#}}}

## {{ Principal:
if __name__ == "__main__":
    mainTest()
## }}     
