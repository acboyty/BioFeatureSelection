import pandas as pd 
import numpy as np
from config import config
import os
from McOne import McOne
from McTwo import McTwo
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


def evaluation(X, y):
    """
    Get the greatest accuracy of SVM, NBayes, Dtree, NN
    """
    y = y.astype('int')
    kf = KFold(n_splits=5)
    mAcc = []
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc1 = np.mean(SVC().fit(X_train, y_train).predict(X_test) == y_test)
        acc2 = np.mean(GaussianNB().fit(X_train, y_train).predict(X_test) == y_test)
        acc3 = np.mean(DecisionTreeClassifier().fit(X_train, y_train).predict(X_test) == y_test)
        acc4 = np.mean(KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train).predict(X_test) == y_test)
        mAcc.append(max(acc1, acc2, acc3, acc4))
    return np.array(mAcc).mean()


for data_name in os.listdir(config.DATA_PATH):
    print(f'Dataset: {data_name}')

    # data preparation
    data = pd.read_table(os.path.join(config.DATA_PATH, data_name), header=None, index_col=0, low_memory=False).transpose().values
    features = data[:, 1:]
    label = data[:, 0]
    for idx, l in enumerate(list(set(label))):
        label[np.where(label == l)] = idx
    # print(f'features.shape: {features.shape}, label.shape: {label.shape}')

    FOne = McOne(features, label, config.r)
    # print(f'FOne.shape: {FOne.shape}')

    FTwo = McTwo(FOne, label)
    # print(f'FTwo.shape: {FTwo.shape}')

    mAcc1 = evaluation(FOne, label)
    mAcc2 = evaluation(FTwo, label)
    print(f'mAcc1: {mAcc1}, mAcc2: {mAcc2}')
    