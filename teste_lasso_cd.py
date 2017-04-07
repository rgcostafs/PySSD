import coordenada_descendente as cd
import gradiente_descendente as gd
import lasso_cd as lsscd
import lasso as lss
import normalizacao as nrm

import numpy as np
import scipy.io as sio
import scipy.linalg as spla
import time
import matplotlib.pyplot as plt
import math
import spams

def alphaParaToeplitz3(mascara, sinal):
    szM = max(mascara.shape)
    szS = max(sinal.shape)
    primL = np.zeros(shape=(szS,))
    primC = np.zeros(shape=(szS,))
    primL[:1+szM/2] = mascara[szM/2::-1]
    primC[:1+szM/2] = mascara[szM/2:]
    matToepl = np.matrix(spla.toeplitz(primC,primL))
    return matToepl


def ToeplitzParaAlpha3(matrizT, mascara):
    szM = max(mascara.shape)
    masc = np.matrix(np.zeros((szM,1)))
    masc[szM/2::-1,0] = (matrizT[0,:1+szM/2]).T
    masc[szM/2:,0] = matrizT[:1+szM/2,0]
    return masc


#{{{ teste do Lasso CD
def teste_LassoCD():
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
    (Anorm, tamsA) = nrm.normalizacaoMatrizNormaUnitaria(A, ignorar1aColuna=True)
    parametros_x = np.matrix(np.zeros(shape=(POT,1)))

    # salvando as matrizes para possivel uso posterior
    sio.savemat('teste_lasso_cd.mat', {"A": A, "b": b, "Anorm": Anorm, "tamsA": tamsA})
    np.savez_compressed('teste_lasso_cd', A=A, b=b, Anorm=Anorm, tamsA=tamsA)

    # matriz com mais pontos para fazer o grafico
    xmin = np.min(A[:,1])
    xmax = np.max(A[:,1])
    xs = np.linspace(xmin,xmax,100)
    A2 = np.matrix(np.zeros(shape=(100,4)))
    for r in range(len(xs)):
        for c in range(A.shape[1]):
            A2[r,c] = xs[r]**c
    for j in range(1, Anorm.shape[1]):
        A2[:,j] /= tamsA[j]

    print "matriz original A:"
    print A
    print "matriz normalizada Anorm):"
    print Anorm
    print "matriz das observacoes b:"
    print b
    print "matriz dos parametros x:"
    print parametros_x
    
    lmb = 0.0001

    opcoes_lasso = {"retornar_log" : True, "iteracoes" : 10000, \
                    "limiar_magnitude_parametro" : 1e-8, \
                    "lambda" : lmb, \
                    "regularizar_col_1" : False }

    t0 = time.time()
    (novos_parametros_ista_cd, cst_ista_cd) = lsscd.ISTA_CD(Anorm, parametros_x, b, opcoes_lasso)
    et0 = time.time() - t0
    print "Resultado ISTA CD:"
    print novos_parametros_ista_cd
    print "Erro MQO:", gd.custoMQO(Anorm, novos_parametros_ista_cd, b)
    print "Erro Lasso ISTA CD:", lss.custoLasso(Anorm, novos_parametros_ista_cd, b, lmb)
    print "Tempo:", et0
 
    t0 = time.time()
    (novos_parametros_fista_cd, cst_fista_cd) = lsscd.FISTA_CD(Anorm, parametros_x, b, opcoes_lasso)
    et0 = time.time() - t0
    print "Resultado FISTA CD:"
    print novos_parametros_fista_cd
    print "Erro MQO:", gd.custoMQO(Anorm, novos_parametros_fista_cd, b)
    print "Erro Lasso FISTA CD:", lss.custoLasso(Anorm, novos_parametros_fista_cd, b, lmb)
    print "Tempo:", et0
   
    t0 = time.time()
    novos_parametros_xen = (np.linalg.inv(Anorm.T * Anorm)*Anorm.T) * b
    elapsed = time.time() - t0
    print "Equacao normal:"
    print novos_parametros_xen
    print "Erro:", gd.custoMQO(Anorm, novos_parametros_xen, b)
    print "Tempo:", elapsed
 
    t0 = time.time()
    novos_parametros_xen = (np.linalg.inv(A.T * A)*A.T) * b
    elapsed = time.time() - t0
    print "Equacao normal sem normalizacao:"
    print novos_parametros_xen
    print "Erro:", gd.custoMQO(A, novos_parametros_xen, b)
    print "Tempo:", elapsed
   
    plt.figure(0)
    plt.plot(x,y,'ko',x,Anorm*novos_parametros_ista_cd,'kx',xs,A2*novos_parametros_ista_cd,'g-')
    plt.plot(x,Anorm*novos_parametros_fista_cd,'yx',xs,A2*novos_parametros_fista_cd,'bx')
    plt.plot(x,Anorm*novos_parametros_xen,'bx',xs,A2*novos_parametros_xen,'r-')
    plt.title("ISTA CD (verde) vs. FISTA CD (azul) vs. Equacao Normal (vermelho)")
    plt.show()

    # exibicao do progresso dos custos com as iteracoes
    plt.plot(cst_ista_cd,'b-')
    plt.plot(cst_fista_cd, 'r-')
    plt.xscale('log')
    plt.title("Custo ISTA CD (azul) x FISTA CD (vermelho)")
    plt.show()

    # tentativa de fazer uma deconvolucao simples usando somente regressao linear
    mascara = np.array([0,0,1,0,-1,0,0])
    sinal = np.array([0,0,-1,1,0,0,0])
    toep = alphaParaToeplitz3(mascara, sinal)
    matSinal = np.matrix(np.reshape(sinal,(max(sinal.shape),1)))
    parametros_x = np.matrix(np.zeros(shape=(max(sinal.shape),1)))
    convol = toep * matSinal
    print convol

    opcoes_lasso = {"retornar_log" : True, "iteracoes" : 10000, \
                    "limiar_magnitude_parametro" : 1e-15, \
                    "lambda" : 0.001, \
                    "regularizar_col_1" : False }
    t0 = time.time()
    (deconvolucao_parametros_x_ista, cst) = lsscd.ISTA_CD(toep, parametros_x, convol, opcoes_lasso)
    et0 = time.time() - t0
    print "resultado da deconvolucao ISTA:", deconvolucao_parametros_x_ista
    print "Tempo:", et0

    t0 = time.time()
    (deconvolucao_parametros_x_fista, cst) = lsscd.FISTA_CD(toep, parametros_x, convol, opcoes_lasso)
    et0 = time.time() - t0
    print "resultado da deconvolucao FISTA:", deconvolucao_parametros_x_fista
    print "Tempo:", et0

    print "tentando aplicar MQO depois da selecao de variaveis"
    parametros_x = deconvolucao_parametros_x_ista.copy()
    indices_selecionados = []
    for i in xrange(parametros_x.shape[0]):
        if abs(parametros_x[i,0]) > 0.001:
            indices_selecionados.append(i)
        else:
            parametros_x[i,0] = 0.

    opcoes_cd = { "retornar_log" : True, "iteracoes" : 10000, \
                  "limiar_magnitude_parametro" : 1e-15, \
                  "indices_escolhidos": indices_selecionados }
    t0 = time.time()
    (ajuste_pos_decon, cst) = cd.coordenadaDescendente(toep, parametros_x, convol, opcoes_cd)
    et0 = time.time() - t0
    print "resultado do AJUSTE da deconvolucao ISTA:", ajuste_pos_decon
    print "Tempo:", et0
#}}}

#{{{ teste do Lasso CD
def teste_LassoCD_2():
    print "Teste Lasso CD - Normalizacao - Denormalizacao"
    # Geracao do dado sintetico de teste
    # Sao gerados 10 pontos para se fazer o ajuste de uma curva de ate 3 grau

    # Montagem da matriz de medidas e da matriz de observacoes como se fossemos resolver minimos quadrados
    A = np.matrix(np.array([1,1,2,1, 1,3,4,1, 1,5,7,1, 1,7,9,1, 1,9,12,1, 0,0,0,1]).reshape(6,4))
    b = np.matrix(np.array([3,5,8,11,12,0]).reshape(6,1))
    mb = np.mean(b[:-1,0]) 

    ma = np.zeros(A.shape[1])
    ma[1:-1] = np.mean(A[:-1,1:-1], axis=0)  
    
    # matriz de centralizacao do dado
    T = np.matrix(np.eye(A.shape[1]))
    T[-1,1:-1] = -ma[1:-1]

    print "T"
    print T

    AT = A * T # aplicacao da transformacao de centralizacao das colunas/dimensoes das amostras

    ms = np.zeros(A.shape[1])
    S = np.matrix(np.eye(A.shape[1]))
    for j in xrange(1,A.shape[1]-1):
        ms[j] = np.sqrt((AT[:-1,j].T * AT[:-1,j]) / (AT.shape[0]-1))
        # ms[j] = np.sqrt(AT[:,j].T * AT[:,j])
        S[j,j] = 1.0 / ms[j]
    
    print "S"
    print S

    ATS = AT * S

    print "ATS"
    print ATS

    ATS2 = ATS[:-1,1:-1]

    bc = b - mb

    dotsAb = np.zeros(A.shape[1])
    for i in xrange(A.shape[1]-1):
        dotsAb[i] = abs(ATS[:-1,i].T * bc[:-1,0])
    dotsAb2 = np.zeros(ATS2.shape[1])
    for i in xrange(ATS2.shape[1]):
        dotsAb2[i] = abs(ATS2[:,i].T * bc[:-1,0])

    print "Produtos escalares de ATS2 por bc"
    print dotsAb2
    lambdaMax = max(dotsAb)/(ATS.shape[0]-1)
    # lambdaMax *= 0.99
    print "Maximo lambda usando ATS?>", lambdaMax
    print "Maximo lambda usando ATS2?>", max(dotsAb2)/ATS2.shape[0]
    lambdaMax = max(dotsAb2)/ATS2.shape[0]

    parametros_x = np.matrix(np.zeros(shape=(2,1)))

    res = np.linalg.inv(ATS.T * ATS) * ATS.T * bc # este eh o resultado pela equacao normal, mas precisa devolver para o espaco de parametros original

    TS = T * S
    resOrig = TS * res

    print "Resultado da equacao normal:"
    print resOrig
    print "Eliminando a ultima coordenada"
    resOrig[-1,0] = 0.
    print resOrig
    print "Erro MQO (eq. normal):", gd.custoMQO(A, resOrig, b)

    print "matriz original A:"
    print A
    print "matriz normalizada Anorm):"
    print ATS
    print "matriz das observacoes b:"
    print b
    print "matriz das observacoes centralizadas:"
    print bc
    print "matriz dos parametros x:"
    print parametros_x
    
    lmb = 0.0001
    
    #indices = [0,1,2]
    opcoes_lasso = {"retornar_log" : True, "iteracoes" : 10000, \
                    "limiar_magnitude_parametro" : 1e-8, \
                    #"lambda" : lambdaMax, \
                    "lambda" : lmb, \
                    "regularizar_col_1" : True }
                    #"indices_escolhidos" : indices }

    t0 = time.time()
    (novos_parametros_ista_cd, cst_ista_cd) = lsscd.ISTA_CD(ATS2, parametros_x, bc[:-1,:], opcoes_lasso)
    et0 = time.time() - t0
    print "Resultado ISTA CD:"
    print novos_parametros_ista_cd

    beta0 = mb - np.matrix(ma[1:-1]) * novos_parametros_ista_cd
    print "Intercept:"
    print beta0

    print "Voltando ao espaco das variaveis:"
    parametros_k = np.matrix(np.zeros(shape=(4,1)))
    parametros_k[0] = beta0
    parametros_k[1] = novos_parametros_ista_cd[0]
    parametros_k[2] = novos_parametros_ista_cd[1]
    parametros_k[3] = -beta0

    resOrig1 = TS * parametros_k
    print resOrig1

    parametros_x = np.matrix(np.zeros(shape=(4,1)))
    t0 = time.time()
    (novos_parametros_ista_cd, cst_ista_cd) = lsscd.ISTA_CD(ATS, parametros_x, bc, opcoes_lasso)
    et0 = time.time() - t0
    print "Resultado ISTA CD:"
    print novos_parametros_ista_cd

    print "Erro MQO:", gd.custoMQO(ATS, novos_parametros_ista_cd, bc)
    print "Erro Lasso ISTA CD:", lss.custoLasso(ATS, novos_parametros_ista_cd, bc, lmb)
    print "Tempo:", et0
    print "Devolvendo os parametros para o espaco original:"
    novos_parametros_ista_cd_orig = TS * novos_parametros_ista_cd
    novos_parametros_ista_cd_orig[-1,0] = 0.
    print novos_parametros_ista_cd_orig
    print "Erro MQO (original):", gd.custoMQO(A, novos_parametros_ista_cd_orig, b)

    opcoes_lasso["iteracoes"] = 30000
    t0 = time.time()
    (novos_parametros_ista, cst_ista) = lss.ISTA(ATS, parametros_x, bc, opcoes_lasso)
    et0 = time.time() - t0
    print "Resultado ISTA:"
    print novos_parametros_ista
    print "Erro MQO:", gd.custoMQO(ATS, novos_parametros_ista, bc)
    print "Erro Lasso ISTA:", lss.custoLasso(ATS, novos_parametros_ista, bc, lmb)
    print "Tempo:", et0
    print "Devolvendo os parametros para o espaco original:"
    novos_parametros_ista_orig = TS * novos_parametros_ista
    novos_parametros_ista_orig[-1,0] = 0.
    print novos_parametros_ista_orig
    print "Erro MQO (original):", gd.custoMQO(A, novos_parametros_ista_orig, b)

    
 
#}}}


def teste_LassoCD_3():
    print "Teste Lasso CD - Normalizacao - Denormalizacao"
    # Geracao do dado sintetico de teste
    # Sao gerados 10 pontos para se fazer o ajuste de uma curva de ate 3 grau

    # Montagem da matriz de medidas e da matriz de observacoes como se fossemos resolver minimos quadrados
    # A coluna do intercept foi removida de proposito
#    A = np.matrix(np.array([1,2, 3,4, 5,7, 7,9, 9,12]).reshape(5,2))
#    b = np.matrix(np.array([3,5,8,11,12]).reshape(5,1))
    
    A = np.matrix(np.load("matrizToeplitzA.npy"))
    b = np.matrix(np.load("observacao.npy"))

    ma = np.zeros(A.shape[1])
    ma = np.mean(A, axis=0)  

    mb = np.mean(b) 
    bc = b - mb
    print "bc"
    print bc.shape

    print "Dado de entrada:"
    print A.shape
    print b.shape
 
    # Adicionei a coluna do intercept para chamar usar a equacao normal
#    AInter = np.matrix(np.zeros(15).reshape(5,3))
#    AInter[:,0] = 1
#    AInter[:,1:] = A
#   
#    print "Resultado usando equacao normal de cara:"
#    res = np.linalg.inv(AInter.T * AInter)*AInter.T*b
#    print res

    # aumentei a matriz para conter uma ultima coluna com 1's
    A2 = np.matrix(np.ones(shape=(A.shape[0],A.shape[1]+1)))
    A2[:,:-1] = A

    # matriz de centralizacao das caracteristicas do dado
    T = np.matrix(np.eye(A2.shape[1]))
    T[-1,:-1] = -ma

    print "T"
    print T.shape

    # aplicacao da transformacao de centralizacao das colunas/dimensoes das amostras
    A2T = A2 * T 
    print "A2T"
    print A2T.shape

    ms = np.ones(A2T.shape[1])
    S = np.matrix(np.eye(A2T.shape[1]))
    for j in xrange(A2T.shape[1]-1):
        ms[j] = np.sqrt((A2T[:,j].T * A2T[:,j]) / A2T.shape[0])
        S[j,j] = 1.0 / ms[j]
    
    print "S"
    print S.shape

    A3TS = A2T * S

    print "A3TS"
    print A3TS.shape

    TS = T * S
    print "TS"
    print TS.shape

    dotsATSbc = np.zeros(A3TS.shape[1])
    for i in xrange(A3TS.shape[1]-1):
        a = A3TS[:,i].T * bc[:,0]
        dotsATSbc[i] = abs(a)

    print "Produtos escalares de A3TS por bc"
    print dotsATSbc
    ## usar assim para o ISTA_CD
    #lambdaMax = max(dotsATSbc)/A3TS.shape[0]
    ## usar assim para o FISTASpams
    lambdaMax = max(dotsATSbc)

    #lambdaMax2 = lambdaMax/(2*A3TS.shape[0])
    #print "Outro valor que pode fazer...", lambdaMax2

    print "Maximo lambda usando ATS>", lambdaMax

    A4 = A3TS[:,:-1]
    print "A4"
    print A4.shape

    parametros_x = np.matrix(np.zeros(shape=(A4.shape[1],1)))

    lmb = 0.0001
    
    opcoes_lasso = {"retornar_log" : False, "iteracoes" : 1000, \
                    "limiar_magnitude_parametro" : 1e-3, \
                    "limiar_magnitude_gradiente" : 1e-3, \
                     "lambda" : lmb, \
                    "regularizar_col_1" : True }

    TSc = TS[:,:-1]

    caminho = []
    n = parametros_x.shape[0]
    for i in xrange(n):
        caminho.append([])

    t0 = time.time()
    #qtdIters = []
    totIters = 0
    deltaTempo = 0
    for i in xrange(101):
        opcoes_lasso["lambda"] = (i / 100.0) * lambdaMax
        t1 = time.time()
        novpar = lsscd.ISTA_CD(A4, parametros_x, bc, opcoes_lasso)
        #novpar = lsscd.FISTASpams(A4, bc, parametros_x, opcoes_lasso["lambda"], 1000)
        print "novpar.shape", novpar.shape
        novos_parametros_ista_cd = novpar.copy()
        et1 = time.time() - t1
        print "novpar"
        #print novpar.shape
        #t1 = time.time()
        #(novos_parametros_ista_cd, cst_ista_cd) = lsscd.ISTA_CD(A4, parametros_x, bc, opcoes_lasso)
        #et1 = time.time() - t1
        deltaTempo += et1
        
        parametros_x = novos_parametros_ista_cd.copy()
        #qtdIters.append(len(cst_ista_cd))
        #totIters += len(cst_ista_cd)
        #print "Resultado ISTA CD:"
        #print novos_parametros_ista_cd
        #print "Trazendo o vetor de parametros de volta para o espaco original"
        res = TSc * novos_parametros_ista_cd
        print "res.shape", res.shape
        #print res
        #print "Calculando o intercept"
        b0 = mb - ma*res[:-1,0]
        #print b0
        #print "Gerando solucao final, com o primeiro elemento sendo o intercept"
        res[1:] = res[:-1]
        res[0] = b0
        #print res
        for j in xrange(n):
            caminho[j].append(res[j,0])
        
        print i,"/",(101-1)
    et0 = time.time() - t0
    print "Tempo:", et0
    print "Tempo especifico:", deltaTempo
    #print "Quantidade de iteracoes", totIters
    #plt.plot(qtdIters)
    #plt.show()

    fmt = ['r-','g-','b-','k-','c-','m-','y-']

    for i in xrange(n):
        plt.plot(caminho[i], fmt[i%7])
    plt.show()
#}}}


if __name__ == "__main__":
    teste_LassoCD_3()

