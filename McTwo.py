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
    curSet = set([])
    leftSet = set([x for x in range(k)])

    while True:
        tempAcc, idx = -1, -1
        for x in leftSet:
            tmpAcc = calAcc(FR[:, list(curSet) + [x]], C)
            if tmpAcc > tempAcc:
                tempAcc = tmpAcc
                idx = x
        if tempAcc > curAcc:
            curAcc = tempAcc
            curSet.add(idx)
            leftSet.remove(idx)
        else:
            break

    return FR[:, list(curSet)]
