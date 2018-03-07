import numpy as np
import os
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.metrics
from matplotlib import pylab as plt


def plot(X, y):
    plt.plot(X[y == 0, 0], X[y == 0, 1], "k.")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "ro")
    plt.plot(X[y == 2, 0], X[y == 2, 1], "bo")
    plt.plot(X[y == 3, 0], X[y == 3, 1], "go")
    plt.plot(X[y == 4, 0], X[y == 4, 1], "co")

    plt.show()

def pca(X, y, test, comp=None):
    X, y = reshape_data(X, y)

    pca = sklearn.decomposition.PCA(n_components='mle' if not comp else comp, svd_solver='full' if not comp else 'auto')
    pca.fit(X)

    return pca.transform(X), y



#is scheiße
def pca_knn():
    trn_X, trn_y = pca(*load_data(test=False), test=False)

    # Training phase
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(trn_X, trn_y)

    # Testing phase
    tst_X, tst_y = pca(*load_data(test=True), test=True, comp=trn_X.shape[1])
    y_pred = knn.predict(tst_X)

    cm = sklearn.metrics.confusion_matrix(tst_y, y_pred)
    print(cm)

#is ok
def lda_knn():
    trn_X, trn_y = reshape_data(*load_data(test=False))
    tst_X, tst_y = reshape_data(*load_data(test=True))

    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(trn_X, trn_y)
    trn_X2 = lda.transform(trn_X)

    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=15)
    knn.fit(trn_X2, trn_y)
    y_pred = knn.predict(lda.transform(tst_X))

    print(sklearn.metrics.confusion_matrix(tst_y, y_pred))


def reshape_data(X, y):
    rows = X.shape[0]//225
    X = X.reshape([225, rows]).T
    y = y.T
    return X, y


def load_data(test=False):
    t = "tst" if test else "trn"

    chips = np.fromfile(os.path.join("data", "Chips", "exp_E1_C{}.sdt".format(t)), dtype='uint8', sep="")
    labels = np.fromfile(os.path.join("data", "Chips", "exp_E1_L{}.sdt".format(t)), dtype='uint8', sep="")
    return chips, labels



if __name__ == "__main__":
    pca_knn()
    # lda_knn()