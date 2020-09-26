import numpy as np
from minepy import MINE
import pandas as pd


def mic(x, y):
    mine = MINE()
    mine.compute_score(x, y)
    return mine.mic()


def McOne(F, C, r):
    """
    F: Features in ndarray format of size (s, k)
    C: Label in ndarray format, containing only 0 and 1, of size (s,)
    r: A pre-set irrelevency threshold
    """
    s, k = F.shape
    micFC = [-1 for _ in range(k)]
    Subset = [-1 for _ in range(k)]
    numSubset = 0  # [0, numSubset) contains the selected features

    for i in range(k):
        micFC[i] = mic(F[:, i], C)
        if micFC[i] >= r:
            Subset[numSubset] = i
            numSubset += 1

    Subset = Subset[0:numSubset]
    Subset.sort(key=lambda x: micFC[x], reverse=True)

    mask = [True for _ in range(numSubset)]
    for e in range(numSubset):
        if mask[e]:
            for q in range(e + 1, numSubset):
                if mask[q] and mic(F[:, Subset[e]], F[:, Subset[q]]) >= micFC[Subset[q]]:
                    mask[q] = False
    FReduce = F[:, np.array(Subset)[mask]]

    return FReduce