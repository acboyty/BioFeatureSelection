import pandas as pd 
import numpy as np
from config import config
import os

for path in os.listdir(config.DATA_PATH):
    # data preparation
    data = pd.read_table(os.path.join(config.DATA_PATH, path), header=None, index_col=0, low_memory=False).transpose().values
    features = data[:, 1:]
    label = data[:, 0]
    for idx, l in enumerate(list(set(label))):
        label[np.where(label == l)] = idx
    print(features.shape, label.shape)

    