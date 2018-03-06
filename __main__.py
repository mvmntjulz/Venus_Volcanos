import numpy as np
import os
import sklearn.decomposition
from matplotlib import pylab as plt


def pca(X, y, comp):
    X = X.reshape([225, 1179]).T
    labels = y.T

    pca = sklearn.decomposition.PCA(n_components=comp)
    pca.fit(X)
    X_reduced = pca.transform(X)

    plt.plot(X_reduced[labels == 0, 0], X_reduced[labels == 0, 1], "k.")
    plt.plot(X_reduced[labels == 1, 0], X_reduced[labels == 1, 1], "ro")
    plt.plot(X_reduced[labels == 2, 0], X_reduced[labels == 2, 1], "bo")
    plt.plot(X_reduced[labels == 3, 0], X_reduced[labels == 3, 1], "go")
    plt.plot(X_reduced[labels == 4, 0], X_reduced[labels == 4, 1], "co")

    plt.show()


def load_data():
    chips = np.fromfile(os.path.join("data", "Chips", "exp_A1_Ctrn.sdt"), dtype='uint8', sep="")
    labels = np.fromfile(os.path.join("data", "Chips", "exp_A1_Ltrn.sdt"), dtype='uint8', sep="")
    return chips, labels


if __name__ == "__main__":
    pca(*load_data(), comp=2)
