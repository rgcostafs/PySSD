# based on the chapter "Uncertainty measures and the concentration of probability density functions", by Helio Lopes and Simone Barbosa
# "Book title"

import math
import numpy as np
import scipy.stats.stats as ss


#{{{ discreteShannonEntropy
def discreteShannonEntropy(samples):
    histog = {}
    sz = float(len(samples))
    for s in samples:
        if histog.has_key(s):
            histog[s] = histog[s] + 1
        else:
            histog[s] = 1
    accum = 0.
    for k in histog.keys():
        prob = histog[k] / sz
        accum += prob * math.log(prob)
    
    return -1.0 * accum # $ -\sum_{i=1}^{n}{p_i \ln(p_i)} $
#}}}

#{{{ interarrivalTimes
def interarrivalTimes(samples, threshold):
    iats = []
    aux = 1
    counting = False
    for s in samples:
        if abs(s) < threshold: # eh nulo
            counting = True
            aux += 1
        else:
            counting = False
            iats.append(aux)
            aux = 1
    if counting:
        iats.append(aux)
    return iats
#}}}

#{{{ normalizedDiscreteShannonEntropy
def normalizedDiscreteShannonEntropy(samples):
    dse = discreteShannonEntropy(samples)
    sz = float(len(samples))
    normFactor = -math.log(1.0 / sz)
    return dse / normFactor
#}}}

#{{{ divergenceShannonJensen
def divergenceShannonJensen(samplesP, samplesQ): # samplesP e samplesQ sao os interarrival times
    samplesPQ = 0.5 * np.concatenate((samplesP, samplesQ))
    dsePQ = discreteShannonEntropy(samplesPQ)
    dseP = discreteShannonEntropy(samplesP)
    dseQ = discreteShannonEntropy(samplesQ)
    return dsePQ - 0.5*dseP - 0.5*dseQ
#}}}

#{{{ sequenceSimplification
def sequenceSimplification(samples, threshold):
    abssams = np.abs(samples)
    seq = np.zeros(samples.shape)
    seq[abssams > threshold] = np.sign(samples[abssams > threshold])
    return seq
#}}}


    
if __name__ == "__main__":
    np.random.seed
    seq1 = np.random.random(800) - 0.5
    seq2 = np.random.random(800) - 0.5
    seq3 = seq1.copy()

    pos = np.random.randint(low=0, high=799, size=100)
    seq3[pos] = np.random.random(100) - 0.5

    seq1 = 2*seq1
    seq2 = 2*seq2
    seq3 = 2*seq3

    ss1 = sequenceSimplification(seq1, 0.8)
    ss2 = sequenceSimplification(seq2, 0.8)
    ss3 = sequenceSimplification(seq3, 0.8)

    iat1 = interarrivalTimes(seq1, 0.8)
    iat2 = interarrivalTimes(seq2, 0.8)
    iat3 = interarrivalTimes(seq3, 0.8)
#
#    diff12 = divergenceShannonJensen(ss1,ss2)
#    diff13 = divergenceShannonJensen(ss1,ss3)
#    diff23 = divergenceShannonJensen(ss2,ss3)
#    
#    print diff12, diff13, diff23
#    print math.sqrt(diff12), math.sqrt(diff13), math.sqrt(diff23)
#
#    diff12 = divergenceShannonJensen(seq1,seq2)
#    diff13 = divergenceShannonJensen(seq1,seq3)
#    diff23 = divergenceShannonJensen(seq2,seq3)
#
#    print diff12, diff13, diff23
#    print math.sqrt(diff12), math.sqrt(diff13), math.sqrt(diff23)
 
    print "Versao corrigida"

    diff12 = divergenceShannonJensen(iat1,iat2)
    diff13 = divergenceShannonJensen(iat1,iat3)
    diff23 = divergenceShannonJensen(iat2,iat3)

    print diff12, diff13, diff23
    print math.sqrt(diff12), math.sqrt(diff13), math.sqrt(diff23)
    
    print "FIM Versao corrigida"
   
    print ss.pearsonr(seq1, seq2)
    print ss.pearsonr(seq1, seq3)
    print ss.pearsonr(seq2, seq3)
    
    print ss.pearsonr(ss1, ss2)
    print ss.pearsonr(ss1, ss3)
    print ss.pearsonr(ss2, ss3)
