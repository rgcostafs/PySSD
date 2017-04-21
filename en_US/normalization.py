import numpy as np

#{{{ normalizationMatrizNormalPadrao coloca os dados com media 0 e variancia 1
def normalizationMatrizNormalPadrao(observations, ignore1stCol = False):
    res = observations.copy()
    meanCols = np.array([0.0]*observations.shape[1])
    stdCols = np.array([0.0]*observations.shape[1])
    inicio = 0
    if (ignore1stCol):
        inicio = 1
    for j in range(inicio,observations.shape[1]):
        meanCols[j] = np.mean(observations[:,j])
        stdCols[j] = np.std(observations[:,j])
        res[:,j] = (observations[:,j] - meanCols[j]) / stdCols[j]
    return (res, meanCols, stdCols)
#}}}

#{{{ normalizationCentralizaNorma1 coloca os dados com media 0 e norma 1 (
def normalizationCentralizaNorma1(observations, ignore1stCol = False):
    res = observations.copy()
    meanCols = np.array(np.mean(observations, axis=0)).flatten()
    inicio = 0
    if (ignore1stCol):
        inicio = 1
    res[:,inicio:] = res[:,inicio:] - meanCols[inicio:]
    szCols = np.linalg.norm(res, axis=0)
    # TODO: tratar o caso de tamanho 0
    res[:,inicio:] = res[:,inicio:] / szCols[inicio:]
    return (res, meanCols, szCols)
#}}}

#{{{ normalizationMatrizNormaUnitaria divide cada coluna pela sua norma euclideana
def normalizationMatrizNormaUnitaria(observations, ignore1stCol = False):
    res = observations.copy()
    szCols = np.array([0.0]*observations.shape[1], dtype=np.float64)
    inicio = 0
    if (ignore1stCol):
        inicio = 1
    for j in range(inicio,observations.shape[1]):
        szCols[j] = np.linalg.norm(observations[:,j])
        res[:,j] = observations[:,j] / szCols[j]
    return (res, szCols)
#}}}

#{{{ normalizationMatrizMaximo1 divide cada coluna pelo seu valor de modulo maximo
def normalizationMatrizMaximo1(observations, ignore1stCol = False):
    res = observations.copy()
    maxCols = np.array([0.0]*observations.shape[1], dtype=np.float64)
    inicio = 0
    if (ignore1stCol):
        inicio = 1
    for j in range(inicio,observations.shape[1]):
        maxCols[j] = np.max(np.abs(observations[:,j]))
        res[:,j] = observations[:,j] / maxCols[j]
    return (res, maxCols)
#}}}


