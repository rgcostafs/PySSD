import numpy as np

#{{{ normalizacaoMatrizNormalPadrao coloca os dados com media 0 e variancia 1
def normalizacaoMatrizNormalPadrao(matObservacoes, ignorar1aColuna = False):
    res = matObservacoes.copy()
    meanCols = np.array([0.0]*matObservacoes.shape[1])
    stdCols = np.array([0.0]*matObservacoes.shape[1])
    inicio = 0
    if (ignorar1aColuna):
        inicio = 1
    for j in range(inicio,matObservacoes.shape[1]):
        meanCols[j] = np.mean(matObservacoes[:,j])
        stdCols[j] = np.std(matObservacoes[:,j])
        res[:,j] = (matObservacoes[:,j] - meanCols[j]) / stdCols[j]
    return (res, meanCols, stdCols)
#}}}

#{{{ normalizacaoCentralizaNorma1 coloca os dados com media 0 e norma 1 (
def normalizacaoCentralizaNorma1(matObservacoes, ignorar1aColuna = False):
    res = matObservacoes.copy()
    meanCols = np.array(np.mean(matObservacoes, axis=0)).flatten()
    inicio = 0
    if (ignorar1aColuna):
        inicio = 1
    res[:,inicio:] = res[:,inicio:] - meanCols[inicio:]
    szCols = np.linalg.norm(res, axis=0)
    # TODO: tratar o caso de tamanho 0
    res[:,inicio:] = res[:,inicio:] / szCols[inicio:]
    return (res, meanCols, szCols)
#}}}

#{{{ normalizacaoMatrizNormaUnitaria divide cada coluna pela sua norma euclideana
def normalizacaoMatrizNormaUnitaria(matObservacoes, ignorar1aColuna = False):
    res = matObservacoes.copy()
    szCols = np.array([0.0]*matObservacoes.shape[1], dtype=np.float64)
    inicio = 0
    if (ignorar1aColuna):
        inicio = 1
    for j in range(inicio,matObservacoes.shape[1]):
        szCols[j] = np.linalg.norm(matObservacoes[:,j])
        res[:,j] = matObservacoes[:,j] / szCols[j]
    return (res, szCols)
#}}}

#{{{ normalizacaoMatrizMaximo1 divide cada coluna pelo seu valor de modulo maximo
def normalizacaoMatrizMaximo1(matObservacoes, ignorar1aColuna = False):
    res = matObservacoes.copy()
    maxCols = np.array([0.0]*matObservacoes.shape[1], dtype=np.float64)
    inicio = 0
    if (ignorar1aColuna):
        inicio = 1
    for j in range(inicio,matObservacoes.shape[1]):
        maxCols[j] = np.max(np.abs(matObservacoes[:,j]))
        res[:,j] = matObservacoes[:,j] / maxCols[j]
    return (res, maxCols)
#}}}


