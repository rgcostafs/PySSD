import numpy as np
import math
import lasso
import coordenada_descendente as cd
import gradiente_descendente as gd
import matplotlib.pyplot as plt
import spams # SParse Modeling Software

#{{{ passoCDLimiarizacao2
def passoCDLimiarizacao2(A, x, b, j, lmbda, regCol1 = False):
    tt2 = np.matrix(x.copy())
    t2 = np.matrix(x.copy())
    A2 = A.copy()
    A2[:,j] = 0.0
    t2[j,:] = 0.0
    tt2[j,0] = (A[:,j].T * (b - A2*t2)) / (A[:,j].T * A[:,j])
    if (lmbda > 0.0):
        sinal = 0.
        if (tt2[j,0] > 0.0):
            sinal = 1.
        elif (tt2[j,0] < 0.0):
            sinal = -1.
        if ((j != 0) or regCol1):
            tt2[j,0] = sinal * max(0.0, abs(tt2[j,0]) -  lmbda)
    
    return tt2
#}}}

#{{{ ISTA_CD (substitui o ISTA_CD_2 do metodosOtimizacao
def ISTA_CD(medidas_A, parametros_x, observacoes_b, opcoes):
    usarLog = False
    if (opcoes.has_key("retornar_log") and opcoes["retornar_log"]):
        usarLog = True
        logCusto = []
    limiteIteracoes = 100000 # limito a 100.000 de iteracoes
    if (opcoes.has_key("iteracoes") and opcoes["iteracoes"] > 0):
        limiteIteracoes = opcoes["iteracoes"]
    limiarMagParam = 1e-15
    if (opcoes.has_key("limiar_magnitude_parametro") and opcoes["limiar_magnitude_parametro"] > 0.0):
        limiarMagParam = opcoes["limiar_magnitude_parametro"]
    fator = 2.0 # fator de escala a ser usado no custo e no gradiente
    if (opcoes.has_key("fator") and opcoes["fator"] > 0.0):
        fator = opcoes["fator"]
    lmbda = 0.0
    if (opcoes.has_key("lambda") and opcoes["lambda"] > 0.0):
        lmbda = opcoes["lambda"]
    regCol1 = False
    if (opcoes.has_key("regularizar_col_1")):
        regCol1 = opcoes["regularizar_col_1"]
    n = parametros_x.shape[0] # quantos parametros == quantas linhas tem no vetor de parametros
    indices_escolhidos = range(n)
    if (opcoes.has_key("indices_escolhidos")):
        indices_escolhidos = opcoes["indices_escolhidos"]
    x1 = parametros_x.copy()
    mags = np.array([0.0] * n)
    it = 0
    for i in xrange(limiteIteracoes): #{
        xant = x1.copy()
        for j in indices_escolhidos: #{
            x2 = passoCDLimiarizacao2(medidas_A, x1, observacoes_b, j, lmbda, regCol1)
            mags[j] = abs(x2[j,0] - x1[j,0])
            x1[:,:] = x2[:,:]
        #}
        diff = x1 - xant
        grad = gd.gradienteMQO(medidas_A, x1, observacoes_b)
        dotp = diff.T * grad
        print "dotp ISTA:", dotp

        xant = x1.copy()

        if (usarLog):
            logCusto.append(lasso.custoLasso(medidas_A, x1, observacoes_b, lmbda, 0.5*fator))
        if (np.max(mags) < limiarMagParam):
            break
        print it, " e: ", np.max(mags)
        it += 1
    #}
    #print "ISTA_CD executou", it, "iteracoes"
    if (usarLog):
        return (x1, logCusto)
    else:
        return x1
#}}}

#{{{ passoFISTAConstanteCD (substitui passoFISTAConstante_CD_Loop)
def passoFISTAConstanteCD(medidas_A, parametros_xk, observacoes_b, lmbda, xk_menos_1, tk, indices_escolhidos, regCol1=False):
    jk = 0
    xk = parametros_xk.copy()
    xk2 = parametros_xk.copy()

    for j in indices_escolhidos:
        xk2 = cd.passoCD(medidas_A, xk, observacoes_b, j)
        xk[:,:] = xk2[:,:]

    xk = lasso.limiarizacaoSuave(xk2, lmbda, regCol1)
    tkp1 = 0.5 * (1.0 + math.sqrt(1.0 + 4.0*tk*tk))
    ykp1 = xk + ((tk-1.0)/tkp1)*(xk - xk_menos_1)

    cst0 = lasso.custoLasso(medidas_A,xk,observacoes_b,lmbda)
    cst1 = lasso.custoLasso(medidas_A,ykp1,observacoes_b,lmbda)

    if (cst1 < cst0):
        print "melhorou"
    else:
        print "piorou"
        ykp1 = xk.copy()

    return (ykp1, xk, tkp1)
#}}}

#{{{ FISTA_CD (substitui FISTA_CD_Loop)
def FISTA_CD(medidas_A, parametros_x, observacoes_b, opcoes):
    # configuracao do metodo
    usarLog = False
    if (opcoes.has_key("retornar_log") and opcoes["retornar_log"]):
        usarLog = True
        logCusto = []
    limiteIteracoes = 100000 # limito a 100.000 de iteracoes
    if (opcoes.has_key("iteracoes") and opcoes["iteracoes"] > 0):
        limiteIteracoes = opcoes["iteracoes"]
    limiarMagParam = 1e-15
    if (opcoes.has_key("limiar_magnitude_parametro") and opcoes["limiar_magnitude_parametro"] > 0.0):
        limiarMagParam = opcoes["limiar_magnitude_parametro"]
    fator = 2.0 # fator de escala a ser usado no custo e no gradiente
    if (opcoes.has_key("fator") and opcoes["fator"] > 0.0):
        fator = opcoes["fator"]
    lmbda = 0.0
    if (opcoes.has_key("lambda") and opcoes["lambda"] > 0.0):
        lmbda = opcoes["lambda"]
    regCol1 = False
    if (opcoes.has_key("regularizar_col_1")):
        regCol1 = opcoes["regularizar_col_1"]
    
    n = parametros_x.shape[0] # quantos parametros == quantas linhas tem no vetor de parametros
    indices_escolhidos = range(n)
    if (opcoes.has_key("indices_escolhidos")):
        indices_escolhidos = opcoes["indices_escolhidos"]
    x1 = parametros_x.copy()
    # execucao FISTA
    tk = 1.0
    xkm1 = parametros_x.copy()
    yk = parametros_x.copy()
    it = 0
    for i in xrange(limiteIteracoes): #{
        (yk2, xkm1, tk) = passoFISTAConstanteCD(medidas_A, yk, observacoes_b, lmbda, xkm1, tk, indices_escolhidos, regCol1)
        if (usarLog):
            logCusto.append(lasso.custoLasso(medidas_A, yk2, observacoes_b, lmbda, 0.5*fator))
        mags = np.abs(yk2 - yk)
        yk[:,:] = yk2[:,:]
        if (np.max(mags) < limiarMagParam):
            break
        it += 1
        print it, " e:", np.max(mags)
    #}
    print "FISTA_CD executou", it, "iteracoes"
    if (usarLog):
        return (yk, logCusto)
    else:
        return yk
#}}}

#{{{ FISTA_CD (substitui FISTA_CD_Loop)
def FISTA_CD_2(medidas_A, parametros_x, observacoes_b, opcoes):
    # configuracao do metodo
    usarLog = False
    if (opcoes.has_key("retornar_log") and opcoes["retornar_log"]):
        usarLog = True
        logCusto = []
    limiteIteracoes = 100000 # limito a 100.000 de iteracoes
    if (opcoes.has_key("iteracoes") and opcoes["iteracoes"] > 0):
        limiteIteracoes = opcoes["iteracoes"]
    limiarMagParam = 1e-15
    if (opcoes.has_key("limiar_magnitude_parametro") and opcoes["limiar_magnitude_parametro"] > 0.0):
        limiarMagParam = opcoes["limiar_magnitude_parametro"]
    fator = 2.0 # fator de escala a ser usado no custo e no gradiente
    if (opcoes.has_key("fator") and opcoes["fator"] > 0.0):
        fator = opcoes["fator"]
    lmbda = 0.0
    if (opcoes.has_key("lambda") and opcoes["lambda"] > 0.0):
        lmbda = opcoes["lambda"]
    regCol1 = False
    if (opcoes.has_key("regularizar_col_1")):
        regCol1 = opcoes["regularizar_col_1"]
    
    n = parametros_x.shape[0] # quantos parametros == quantas linhas tem no vetor de parametros
    indices_escolhidos = range(n)
    if (opcoes.has_key("indices_escolhidos")):
        indices_escolhidos = opcoes["indices_escolhidos"]
    x1 = parametros_x.copy()
    # execucao FISTA
    tk = 1.0
    xkm1 = parametros_x.copy()
    yk = parametros_x.copy()
    it = 0
    for i in xrange(limiteIteracoes): #{
        (yk2, xkm1, tk) = passoFISTAConstanteCD(medidas_A, yk, observacoes_b, lmbda, xkm1, tk, indices_escolhidos, regCol1)
        if (usarLog):
            logCusto.append(lasso.custoLasso(medidas_A, yk2, observacoes_b, lmbda, 0.5*fator))
        mags = np.abs(yk2 - yk)
        yk[:,:] = yk2[:,:]
        if (np.max(mags) < limiarMagParam):
            break
        it += 1
        print it, " e:", np.max(mags)
    #}
    print "FISTA_CD executou", it, "iteracoes"
    if (usarLog):
        return (yk, logCusto)
    else:
        return yk
#}}}

#{{{ iteracaoCoordinateDescentST
def iteracaoCoordinateDescentST(A, y, theta, j, lmbda):
    tt2 = np.matrix(theta.copy())
    t2 = np.matrix(theta.copy())
    A2 = A.copy()
    A2[:,j] = 0.0
    t2[j,:] = 0.0
    ttj = (A[:,j].T * (y - A2*t2)) / (A[:,j].T * A[:,j])[0,0]
    sinal = 0.
    if (ttj > 0.0):
        sinal = 1.
    elif (ttj < 0.0):
        sinal = -1.
    tt2[j,0] = sinal * max(0.0, abs(ttj) -  lmbda) 
    return (tt2, (j+1)%(theta.shape[0]))
#}}}

#
#def passoISTACD_2(matrizMedidasA, y, theta, lmbda, j): #{
#    (tt2, j_prox) = iteracaoCoordinateDescentST(matrizMedidasA, y, theta, j, lmbda)
#    return (tt2, j_prox)
##}
#
#def ISTA_CD_2(matrizMedidasX, y, theta, lmbda, iteracoes): #{
#    j = 0
#    theta0 = theta.copy()
#    for i in range(iteracoes + theta.shape[0] - (iteracoes % theta.shape[0])):
#        (theta1, j) = passoISTACD_2(matrizMedidasX, y, theta0, lmbda, j)
#        if ((np.linalg.norm(theta1-theta0) < 1e-8) and (j == 0)):
#            print "ISTA_CD_2 executou", i/theta.shape[0], "iteracoes"
#            break
#        theta0[:,:] = theta1[:,:]
#    return theta1
#}
#

#{{{ normalizacao baseada em Hastie, Tibshirani e Wainwright (2015)
def normalizacaoHTW(medidas_A, observacoes_b):
    """ 
    1. Calcula as matrizes de transformacao de A de tal forma que a nova configucacao
    tenha, em cada coluna j (1/N)\sum_{i=1}^{N}{a_{ij}} = 0 e 
    (1/N)\sum_{i=1}^{N}{{a_{ij}}^2} = 1. 

    2. Calcula, com base na normalizacao, qual o valor maximo do parametro de regularizacao
    da formula do Lasso capaz de zerar todos os coeficientes (nao-intercept) do vetor de 
    resposta

    3. Transforma a matriz de medidas original numa versao com normalizacao

    4. Transforma a matriz de observacoes original numa versao centralizada

    Supoe que esta matriz NAO tem a coluna do intercept comum a resolucao de outros 
    problemas de aprendizado de maquina.
    """
    medias_A = np.mean(medidas_A, axis=0)
    media_b = np.mean(observacoes_b)

    bCentrado = observacoes_b - media_b
    A2 = np.matrix(np.ones(shape=(medidas_A.shape[0],medidas_A.shape[1]+1)))
    A2[:,:-1] = medidas_A
    
    # matriz de centralizacao das caracteristicas do dado
    T = np.matrix(np.eye(A2.shape[1]))
    T[-1,:-1] = -medias_A

    A2T = A2 * T

    ms = np.ones(A2T.shape[1])
    S = np.matrix(np.eye(A2T.shape[1]))
    for j in xrange(A2T.shape[1]-1):
        ms[j] = np.sqrt((A2T[:,j].T * A2T[:,j]) / A2T.shape[0])
        S[j,j] = 1.0 / ms[j]
    
    A3TS = A2T * S
    A4 = A3TS[:,:-1]

    dotsATSbc = np.zeros(A3TS.shape[1])
    for i in xrange(A3TS.shape[1]-1):
        dotsATSbc[i] = abs(A3TS[:,i].T * bCentrado[:,0])
    lambdaMax = max(dotsATSbc)/A3TS.shape[0]
    
    print "lambdaMax"
    print lambdaMax

    TS = T*S
    TSc = TS[:,:-1]

    return (A4, medias_A, bCentrado, media_b, TSc, lambdaMax)
#}}}

#{{{ desnormalizacao baseada em Hastie, Tibshirani e Wainwright (2015)
def desnormalizacaoHTW(medias_A, media_b, TSc, solucaoEspacoNormalizado):
    res = TSc * solucaoEspacoNormalizado
    b0 = media_b - medias_A*res[:-1,0]
    return (b0, res[:-1,0])
#}}}

#{{{ FISTASpams
def FISTASpams(matrizMedidasX, y, theta, lmbda, iteracoes):
    myfloat = np.float64

    param = {'numThreads' : 2,
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

    X = np.asfortranarray(matrizMedidasX, dtype=np.float64)
    print "X.shape"
    print X.shape
    Y = np.asfortranarray(np.matrix(y), dtype=np.float64)
    print "Y.shape"
    print Y.shape
    W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=myfloat,order="FORTRAN")
    
    (W, optim_info) = spams.fistaFlat(Y,X,W0,True,**param)
    return W
#}}}

#{{{ corteEsparsidadeFISTACD
def corteEsparsidadeFISTACD(medidas_A, parametros_x, observacoes_b, esparsidade, opcoes, cacheLambda):
    (A4, medias_A, bCentrado, media_b, TSc, lambdaMax) = normalizacaoHTW(medidas_A, observacoes_b)
    
    numPassosIntermediarios = opcoes["num_amostras_esparsidade"]
    deltaPassosIntermediarios = lambdaMax / (numPassosIntermediarios + 1.0)

    pontos_x = []
    pontos_y = []
    opcoes["retornar_log"] = False
    opcoes["regularizar_col_1"] = True
    encontreiEsparsidade = False
    solucaoEncontrada = []
    lambdaSuperior = 0.0
    espSup = 0
    lambdaInferior = 1.0
    espInf = 0
    parametrosSuperior = []
    lmbda = 0.
    if (abs(cacheLambda) <= 1e-10): #{ # eh ~ zero
        for i in range(numPassosIntermediarios+1): #{
            print "processando passo (esparsidade FISTA CD)", i
            lmbda = i * deltaPassosIntermediarios
            opcoes["lambda"] = lmbda
            if i==0:
                # testando...
                # forcando uma regressao ridge com baixa penalidade para acelerar o processo e para garantir inversa
                novos_parametros = np.linalg.inv(A4.T * A4 + 1e-5 * np.eye(A4.shape[1])) * A4.T * bCentrado
                #opcoes["iteracoes"] = 100
                #novos_parametros = cd.coordenadaDescendente(A4, parametros_x, bCentrado, opcoes)
            else:
                A5 = np.asfortranarray(A4)
                bCentrado2 = np.asfortranarray(bCentrado)
                novos_parametros = FISTASpams(A5, bCentrado2, parametros_x, lmbda * A4.shape[0], 1000)
                #opcoes["iteracoes"] = 2000
                #novos_parametros = ISTA_CD(A4, parametros_x, bCentrado, opcoes)
            parametros_x = novos_parametros.copy()
            (b0, res) = desnormalizacaoHTW(medias_A, media_b, TSc, novos_parametros)
            #res = parametros_x.copy()
            esp = np.sum(np.abs(res) > 1e-15)
            if (esp == esparsidade):
                encontreiEsparsidade = True
                solucaoEncontrada = (b0, res)
                break
            if (esp > esparsidade):
                lambdaSuperior = lmbda
                espSup = esp
                parametrosSuperior = parametros_x.copy()
            else:
                lambdaInferior = lmbda
                espInf = esp
                break
            print "esparsidade ", esp, "com lambda ", lmbda
        #} # for
    #} # if
    else: #{
        print "Cache lambda:", cacheLambda
        A5 = np.asfortranarray(A4)
        bCentrado2 = np.asfortranarray(bCentrado)
        lambdaSuperior = cacheLambda
        lambdaInferior = cacheLambda
        encontreiIntervalo = False
        encontreiMenor = False
        encontreiMaior = False
        it = 0
        while (not encontreiIntervalo): #{
            lmbda = 0.5 * (lambdaSuperior + lambdaInferior)
            novos_parametros = FISTASpams(A5, bCentrado2, parametros_x, lmbda * A4.shape[0], 1000)
            parametros_x = novos_parametros.copy()
            (b0, res) = desnormalizacaoHTW(medias_A, media_b, TSc, novos_parametros)
            esp = np.sum(np.abs(res) > 1e-15)
            if (esp == esparsidade):
                encontreiEsparsidade = True
                solucaoEncontrada = (b0, res)
                if (it == 0):
                    print ">>>>> Solucao aproveitada!"
                else:
                    print ">> Solucao encontrada por aproveitamento com ", it, "iteracoes"
                break
            if (esp > esparsidade):
                encontreiMaior = True
                lambdaSuperior = lmbda
                if encontreiMenor:
                    encontreiIntervalo = True
                else:
                    lambdaInferior = 0.9*lambdaInferior + 0.1*lambdaMax # devo aumentar o valor de lambda para achatar mais parametros
                espSup = esp
                print "! Encontrei esparsidade SUPERIOR", espSup, "com lambda", lambdaSuperior
                parametrosSuperior = parametros_x.copy()
            else:
                encontreiMenor = True
                lambdaInferior = lmbda
                if encontreiMaior:
                    encontreiIntervalo = True
                else:
                    lambdaSuperior = 0.9 * lambdaSuperior
                espInf = esp
                print "Encontrei esparsidade INFERIOR", espInf, "com lambda", lambdaInferior
            it += 1
        #}
    #}

    if (encontreiEsparsidade):
        return (solucaoEncontrada, lmbda)
    else:
        print "devo fazer busca binaria ate achar a esparsidade que preciso (%.3f, %.3f) <-> (%d,%d)" % (lambdaSuperior,lambdaInferior,espSup, espInf)
        parametros_x = parametrosSuperior.copy()
        limiteIteracoes = 100
        it = 0
        
        while (not encontreiEsparsidade): #{
            lmbda = (0.5*(lambdaSuperior + lambdaInferior))
            print "processando lambda:", lmbda
            opcoes["lambda"] = lmbda
            
            novos_parametros = FISTASpams(A4, bCentrado, parametros_x, lmbda * A4.shape[0], 1000)
            #novos_parametros = FISTA_CD(A4, parametros_x, bCentrado, opcoes)
            parametros_x = novos_parametros.copy()
            (b0, res) = desnormalizacaoHTW(medias_A, media_b, TSc, novos_parametros)
            esp = np.sum(np.abs(res) > 1e-15)
            if (esp == esparsidade):
                encontreiEsparsidade = True
                solucaoEncontrada = (b0, res)
                break
            if (esp > esparsidade):
                print "foi maior", esp," lambda = ", lmbda
                segundaMelhorSolucao = (b0, res)
                if abs(lambdaSuperior - lmbda) <= 1e-12:
                    print "retornando segunda melhor por precisao numerica"
                    return segundaMelhorSolucao
                lambdaSuperior = lmbda
            else:
                print "foi menor", esp," lambda = ", lmbda
                lambdaInferior = lmbda
            it += 1
            if (it >= limiteIteracoes):
                print "vai retornar a segunda melhor"
                return (segundaMelhorSolucao, lmbda)
        #} // while
    return (solucaoEncontrada, lmbda)
#}}}

