import numpy as np
from scipy import stats

def getErrors(bpmES, bpmGT, timesES=None, timesGT=None, PCC=True):
    RMSE = RMSEerror(bpmES, bpmGT, timesES, timesGT)
    MAE = MAEerror(bpmES, bpmGT, timesES, timesGT)
    MAX = MAXError(bpmES, bpmGT, timesES, timesGT)
    if(PCC == True):
        PCC = PearsonCorr(bpmES, bpmGT, timesES, timesGT)
    else:
        PCC = None
    return RMSE, MAE, MAX, PCC

def RMSEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ RMSE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    df = np.zeros(n)
    for j in range(m):
        for c in range(n):
            df[c] += np.power(diff[c,j],2)

    # -- final RMSE
    RMSE = np.sqrt(df/m)
    return RMSE

def MAEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ MAE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff),axis=1)

    # -- final MAE
    MAE = df/m
    return MAE

def MAXError(bpmES, bpmGT, timesES=None, timesGT=None):
    """ MAE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    df = np.max(np.abs(diff),axis=1)

    # -- final MAE
    MAX = df
    return MAX

def PearsonCorr(bpmES, bpmGT, timesES=None, timesGT=None):

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    CC = np.zeros(n)
    for c in range(n):
        # -- corr
        r,p = stats.pearsonr(diff[c,:]+bpmES[c,:],bpmES[c,:])
        CC[c] = r
    return CC

def bpm_diff(bpmES, bpmGT, timesES=None, timesGT=None):
    n,m = bpmES.shape  # n = num channels, m = bpm length

    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES
            
    diff = np.zeros((n,m))
    for j in range(m):
        t = timesES[j]
        i = np.argmin(np.abs(t-timesGT))
        for c in range(n):
            diff[c,j] = bpmGT[i]-bpmES[c,j]
    return diff

def printErrors(RMSE, MAE, MAX, PCC):
    print("\n    * Errors: RMSE = %.2f, MAE = %.2f, MAX = %.2f, PCC = %.2f" %(RMSE,MAE,MAX,PCC))