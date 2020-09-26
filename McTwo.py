import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def calAcc(FS, C):
    """
    Calculate the BAcc of NN(K=1) algorithm.
    FS: Subset of Features, in ndarray format, of size (s, k)
    C: Label in ndarray format, containing only 0 and 1, of size (s,)
    """
    s, _ = FS.shape
    C = C.astype('int')

    NN = KNeighborsClassifier(n_neighbors=1)
    pred = []
    # Leave-One-Out validation
    for i in range(s):
        NN.fit(FS[[x for x in range(s) if x != i]],
               C[[x for x in range(s) if x != i]])
        pred.append(NN.predict(FS[[i]]).tolist()[0])
    pred = np.array(pred)

    BAcc = (np.mean(pred[np.where(C == 0)] == C[np.where(C == 0)]) +
            np.mean(pred[np.where(C == 1)] == C[np.where(C == 1)])) / 2
    
    return BAcc


def McTwo(FR, C):
    """
    FR: Reduced Features from McOne, in ndarray format, of size (s, k)
    C: Label in ndarray format, containing only 0 and 1, of size (s,)
    """
    s, k = FR.shape
    curAcc = -1
    Subset = [-1 for _ in range(k)]
    numSubset = 0  # [0, numSubset) contains the selected features

    curAcc = calAcc(FR[:, 0:1], C)
    Subset[numSubset] = 0
    numSubset += 1

    for i in range(1, k):
        Subset[numSubset] = i
        tempAcc = calAcc(FR[:, Subset[:numSubset+1]], C)
        if tempAcc > curAcc:
            curAcc = tempAcc
            numSubset += 1
            
    return FR[:, Subset[:numSubset]]
