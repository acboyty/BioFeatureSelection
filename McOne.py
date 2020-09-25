import numpy as np
from minepy import MINE
import pandas as pd 


def mic(x, y):
    mine = MINE()
    mine.compute_score(x, y) 
    return mine.mic()

# x = np.linspace(0, 1, 1000)
# y = np.sin(10 * np.pi * x) + x
# print(mic(x, y))

def McOne()