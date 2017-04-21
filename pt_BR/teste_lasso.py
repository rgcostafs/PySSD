import coordenada_descendente as cd
import gradiente_descendente as gd
import lasso as lss
import normalizacao as nrm

import numpy as np
import scipy.io as sio
import time
import matplotlib.pyplot as plt

#{{{ teste do Lasso (ISTA e FISTA, sem backtracking)
def teste_Lasso():
    print "Teste Lasso"
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
    sio.savemat('teste_lasso.mat', {"A": A, "b": b, "Anorm": Anorm, "tamsA": tamsA})
    np.savez_compressed('teste_lasso', A=A, b=b, Anorm=Anorm, tamsA=tamsA)    

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
    
    alfa = 1.0 / (np.linalg.norm(Anorm)**2)
    lmb = 0.1

    opcoes_lasso = {"retornar_log" : True, "iteracoes" : 100000, \
                    "limiar_magnitude_gradiente" : 1e-10, \
                    "alfa" : alfa, "lambda" : lmb, \
                    "regularizar_col_1" : False }

    t0 = time.time()
    (novos_parametros_ista, cst_ista) = lss.ISTA(Anorm, parametros_x, b, opcoes_lasso)
    et0 = time.time() - t0
    print "Resultado ISTA:"
    print novos_parametros_ista
    print "Erro MQO:", gd.custoMQO(Anorm, novos_parametros_ista, b)
    print "Erro Lasso ISTA:", lss.custoLasso(Anorm, novos_parametros_ista, b, lmb)
    print "Tempo:", et0
 
    t0 = time.time()
    (novos_parametros_fista, cst_fista) = lss.FISTA(Anorm, parametros_x, b, opcoes_lasso)
    et0 = time.time() - t0
    print "Resultado FISTA:"
    print novos_parametros_fista
    print "Erro MQO:", gd.custoMQO(Anorm, novos_parametros_fista, b)
    print "Erro Lasso FISTA:", lss.custoLasso(Anorm, novos_parametros_fista, b, lmb)
    print "Tempo:", et0
   
    t0 = time.time()
    novos_parametros_xen = (np.linalg.inv(Anorm.T * Anorm)*Anorm.T) * b
    elapsed = time.time() - t0
    print "Equacao normal:"
    print novos_parametros_xen
    print "Erro:", gd.custoMQO(Anorm, novos_parametros_xen, b)
    print "Tempo:", elapsed

    
    plt.figure(0)
    plt.plot(x,y,'ko',x,Anorm*novos_parametros_ista,'kx',xs,A2*novos_parametros_ista,'g-')
    plt.plot(x,Anorm*novos_parametros_fista,'yx',xs,A2*novos_parametros_fista,'bx')
    plt.plot(x,Anorm*novos_parametros_xen,'bx',xs,A2*novos_parametros_xen,'r-')
    plt.title("ISTA (verde) vs. FISTA (azul) vs. Equacao Normal (vermelho)")
    plt.show()

    # exibicao do progresso dos custos com as iteracoes
    plt.plot(cst_ista,'b-')
    plt.plot(cst_fista, 'r-')
    plt.xscale('log')
    plt.title("Custo ISTA (azul) x FISTA (vermelho)")
    plt.show()

#}}}

#{{{ teste do Lasso (ISTA e FISTA)
def teste_Lasso_2():
    print "Teste Lasso"
    # Geracao do dado sintetico de teste
    # Sao gerados 10 pontos para se fazer o ajuste de uma curva de ate 3 grau
    SZ = 10
    POT = 4 # grau maximo = 4
    x = SZ * np.random.random(SZ)
    x = np.sort(x)
    y = (1. + 0.5*np.random.random(SZ)) * x ** (1. + 0.5*np.random.random(SZ)) + \
        (1. + 0.2 * np.random.random(SZ))

    # Montagem da matriz de medidas e da matriz de observacoes como se fossemos resolver minimos quadrados
    A = np.matrix(np.zeros(shape=(SZ,POT)))
    b = np.matrix(np.zeros(shape=(SZ,1)))
    for r in range(10):
        b[r] = y[r]
        for c in range(1,POT+1): # 
            A[r,c] = x[r]**c
 
    # ignorar1aColuna: a parcela constante nao sera normalizada e nem sera regularizada
    (Anorm, medA, szA) = nrm.normalizacaoCentralizaNorma1(A, ignorar1aColuna=False)
    parametros_x = np.matrix(np.zeros(shape=(POT,1)))

    meanB = np.mean(b)

    bCent = b - np.mean(b)

    # salvando as matrizes para possivel uso posterior
    sio.savemat('teste_lasso.mat', {"A": A, "b": b, "Anorm": Anorm, "tamsA": tamsA})
    np.savez_compressed('teste_lasso', A=A, b=b, Anorm=Anorm, tamsA=tamsA)    

    # matriz com mais pontos para fazer o grafico
    xmin = np.min(A[:,1])
    xmax = np.max(A[:,1])
    xs = np.linspace(xmin,xmax,100)
    A2 = np.matrix(np.zeros(shape=(100,4)))
    for r in range(len(xs)):
        for c in range(A.shape[1]):
            A2[r,c] = xs[r]**c
    for j in range(1, Anorm.shape[1]):
        A2[:,j] -= medA[j]
        A2[:,j] /= szA[j]

    print "matriz original A:"
    print A
    print "matriz normalizada Anorm):"
    print Anorm
    print "matriz das observacoes b:"
    print b
    print "matriz das observacoes b centralizadas:"
    print bCent
    print "matriz dos parametros x:"
    print parametros_x
    
    alfa = 1.0 / (np.linalg.norm(Anorm)**2)
    lmb = 0.1

    opcoes_lasso = {"retornar_log" : True, "iteracoes" : 100000, \
                    "limiar_magnitude_gradiente" : 1e-10, \
                    "alfa" : alfa, "lambda" : lmb, \
                    "regularizar_col_1" : False }

    t0 = time.time()
    (novos_parametros_ista, cst_ista) = lss.ISTA2(Anorm, parametros_x, bCent, opcoes_lasso)
    et0 = time.time() - t0
    print "Resultado ISTA:"
    print novos_parametros_ista
    print "Erro MQO:", gd.custoMQO(Anorm, novos_parametros_ista, bCent)
    print "Erro Lasso ISTA:", lss.custoLasso(Anorm, novos_parametros_ista, bCent, lmb)
    print "Tempo:", et0
 
    t0 = time.time()
    (novos_parametros_fista, cst_fista) = lss.FISTA2(Anorm, parametros_x, bCent, opcoes_lasso)
    et0 = time.time() - t0
    print "Resultado FISTA:"
    print novos_parametros_fista
    print "Erro MQO:", gd.custoMQO(Anorm, novos_parametros_fista, bCent)
    print "Erro Lasso FISTA:", lss.custoLasso(Anorm, novos_parametros_fista, bCent, lmb)
    print "Tempo:", et0
   
    t0 = time.time()
    novos_parametros_xen = (np.linalg.inv(Anorm.T * Anorm)*Anorm.T) * b
    elapsed = time.time() - t0
    print "Equacao normal:"
    print novos_parametros_xen
    print "Erro:", gd.custoMQO(Anorm, novos_parametros_xen, b)
    print "Tempo:", elapsed
    
    plt.figure(0)
    plt.plot(x,y,'ko',x,Anorm*novos_parametros_ista,'kx',xs,A2*novos_parametros_ista,'g-')
    plt.plot(x,Anorm*novos_parametros_fista,'yx',xs,A2*novos_parametros_fista,'bx')
    plt.plot(x,Anorm*novos_parametros_xen,'bx',xs,A2*novos_parametros_xen,'r-')
    plt.title("ISTA (verde) vs. FISTA (azul) vs. Equacao Normal (vermelho)")
    plt.show()

    # exibicao do progresso dos custos com as iteracoes
    plt.plot(cst_ista,'b-')
    plt.plot(cst_fista, 'r-')
    plt.xscale('log')
    plt.title("Custo ISTA (azul) x FISTA (vermelho)")
    plt.show()

#}}}


if __name__ == "__main__":
    teste_Lasso()

