import coordenada_descendente as cd
import gradiente_descendente as gd
import normalizacao as nrm

import numpy as np
import scipy.io as sio
import scipy.linalg as spla
import time
import matplotlib.pyplot as plt


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


#{{{ teste do CD
def teste_CD():
    print "Teste CD"
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
    # (Anorm, tamsA) = nrm.normalizacaoMatrizNormaUnitaria(A, ignorar1aColuna=True)
    (Anorm, mediasA, stdsA) = nrm.normalizacaoMatrizNormalPadrao(A, ignorar1aColuna=True)
    parametros_x = np.matrix(np.zeros(shape=(POT,1)))

    # salvando as matrizes para possivel uso posterior
    #sio.savemat('teste_cd.mat', {"A": A, "b": b, "Anorm": Anorm, "tamsA": tamsA})
    #np.savez_compressed('teste_cd', A=A, b=b, Anorm=Anorm, tamsA=tamsA)    

    # matriz com mais pontos para fazer o grafico
    xmin = np.min(A[:,1])
    xmax = np.max(A[:,1])
    xs = np.linspace(xmin,xmax,100)
    A2 = np.matrix(np.zeros(shape=(100,4)))
    for r in range(len(xs)):
        for c in range(A.shape[1]):
            A2[r,c] = xs[r]**c
    for j in range(1, Anorm.shape[1]):
        A2[:,j] = (A2[:,j] - mediasA[j]) / stdsA[j]
        #A2[:,j] /= tamsA[j]

    print "matriz original A:"
    print A
    print "matriz normalizada Anorm):"
    print Anorm
    print "matriz das observacoes b:"
    print b
    print "matriz dos parametros x:"
    print parametros_x
    
    opcoes_cd = {"retornar_log" : True, "iteracoes" : 10000, \
                 "limiar_magnitude_parametro" : 1e-8 }
    t0 = time.time()
    (novos_parametros_x, cst) = cd.coordenadaDescendente(Anorm, parametros_x, b, opcoes_cd)
    et0 = time.time() - t0
    print "Resultado:"
    print novos_parametros_x
    print "Erro:", gd.custoMQO(Anorm, novos_parametros_x, b)
    print "Tempo:", et0
    
    t0 = time.time()
    novos_parametros_xen = (np.linalg.inv(Anorm.T * Anorm)*Anorm.T) * b
    elapsed = time.time() - t0
    print "Equacao normal:"
    print novos_parametros_xen
    print "Erro:", gd.custoMQO(Anorm, novos_parametros_xen, b)
    print "Tempo:", elapsed

    
    plt.figure(0)
    plt.plot(x,y,'ko',x,Anorm*novos_parametros_x,'kx',xs,A2*novos_parametros_x,'g-')
    plt.plot(x,Anorm*novos_parametros_xen,'bx',xs,A2*novos_parametros_xen,'r-')
    plt.title("CD vs. Equacao Normal")
    plt.show()

    # exibicao do progresso dos custos com as iteracoes
    plt.plot(cst)
    plt.xscale('log')
    plt.title("Custo CD")
    plt.show()

    # tentativa de fazer uma deconvolucao simples usando somente regressao linear
    mascara = np.array([0,0,1,0,-1,0,0])
    sinal = np.array([0,0,-1,1,0,0,0])
    toep = alphaParaToeplitz3(mascara, sinal)
    matSinal = np.matrix(np.reshape(sinal,(max(sinal.shape),1)))
    parametros_x = np.matrix(np.zeros(shape=(max(sinal.shape),1)))
    convol = toep * matSinal
    print convol
    opcoes_cd = {"retornar_log" : True, "iteracoes" : 10000, 
                 "limiar_magnitude_parametro" : 1e-15 }
    t0 = time.time()
    (deconvolucao_parametros_x, cst) = cd.coordenadaDescendente(toep, parametros_x, convol, opcoes_cd)
    et0 = time.time() - t0
    print "resultado da deconvolucao:", deconvolucao_parametros_x

    #vendo se a equacao normal resolve
    deconv_eq_norm = (np.linalg.inv(toep.T * toep)*toep.T) * convol
    print "resultado da deconvolucao (EqNorm):", deconv_eq_norm

#}}}

if __name__ == "__main__":
    teste_CD()

