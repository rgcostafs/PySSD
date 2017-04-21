import gradiente_descendente as gd
import normalizacao as nrm

import numpy as np
import scipy.io as sio
import time
import matplotlib.pyplot as plt

#{{{ teste do GD, Ridge/Tikhonov
def teste_GD():
    print "Teste GD"
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
    sio.savemat('teste_gd.mat', {"A": A, "b": b, "Anorm": Anorm, "tamsA": tamsA})
    np.savez_compressed('teste_gd', A=A, b=b, Anorm=Anorm, tamsA=tamsA)    

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

    opcoes_gd = {"retornar_log" : True, "iteracoes" : 100000, \
                 "limiar_magnitude_gradiente" : 1e-10, \
                 "alfa" : alfa }
    t0 = time.time()
    (novos_parametros_x, cst) = gd.gradienteDescendente(Anorm, parametros_x, b, opcoes_gd)
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

    opcoes_gd["beta"] = 0.5
    opcoes_gd["regularizar_col_1"] = False
    t0 = time.time()
    (novos_parametros_ridge, cst_ridge) = gd.gradienteDescendenteRegularizacaoTikhonov(Anorm, parametros_x, b, opcoes_gd)
    elapsed = time.time() - t0
    print "Ridge/Tikhonov:"
    print novos_parametros_ridge
    print "Erro:", gd.custoMQO(Anorm, novos_parametros_ridge, b)
    print "Tempo:", elapsed    
    
    plt.figure(0)
    plt.plot(x,y,'ko',x,Anorm*novos_parametros_x,'kx',xs,A2*novos_parametros_x,'g-')
    plt.plot(x,Anorm*novos_parametros_xen,'bx',xs,A2*novos_parametros_xen,'r-')
    plt.title("Gradiente vs. Equacao Normal")
    plt.show()

    plt.figure(0)
    plt.plot(x,y,'ko',x,Anorm*novos_parametros_ridge,'kx',xs,A2*novos_parametros_ridge,'g-')
    plt.plot(x,Anorm*novos_parametros_xen,'bx',xs,A2*novos_parametros_xen,'r-')
    plt.title("Ridge/Tikhonov vs. Equacao Normal")
    plt.show()    

    # exibicao do progresso dos custos com as iteracoes
    plt.plot(cst)
    plt.xscale('log')
    plt.title("Custo GD")
    plt.show()

    plt.plot(cst_ridge)
    plt.xscale('log')
    plt.title("Custo Ridge")
    plt.show()   

#}}}

#{{{
def teste_GD_2():
    print "Teste GD"
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
    (Anorm, mA, sA) = nrm.normalizacaoMatrizNormalPadrao(A, ignorar1aColuna=True)

    print "Matrizes da normalizacao"
    print mA
    print sA    

    parametros_x = np.matrix(np.zeros(shape=(POT,1)))

    print "matriz original A:"
    print A
    print "matriz normalizada Anorm):"
    print Anorm
    print "matriz das observacoes b:"
    print b
    print "matriz dos parametros x:"
    print parametros_x
    
    alfa = 1.0 / (np.linalg.norm(Anorm)**2)

    opcoes_gd = {"retornar_log" : True, "iteracoes" : 100000, \
                 "limiar_magnitude_gradiente" : 1e-10, \
                 "alfa" : alfa }
    t0 = time.time()
    (novos_parametros_x, cst) = gd.gradienteDescendente(Anorm, parametros_x, b, opcoes_gd)
    et0 = time.time() - t0
    print "Resultado:"
    print novos_parametros_x
    print "Erro:", gd.custoMQO(Anorm, novos_parametros_x, b)
    print "Tempo:", et0
    
    t0 = time.time()
    novos_parametros_xen = (np.linalg.inv(A.T * A)*A.T) * b
    elapsed = time.time() - t0
    print "Equacao normal:"
    print novos_parametros_xen
    print "Erro:", gd.custoMQO(A, novos_parametros_xen, b)
    print "Tempo:", elapsed
#}}}



if __name__ == "__main__":
    teste_GD_2()
