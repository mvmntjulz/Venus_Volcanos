import numpy as np
import os
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.metrics
from sklearn.model_selection import train_test_split
from matplotlib import pylab as plt
from skimage import exposure
import scipy.misc

PATCHES = list()
LABELS = list()


def plot(X, y):
    plt.plot(X[y == '4', 0], X[y == '4', 1], "ko", alpha=0.2)
    plt.plot(X[y == '3', 0], X[y == '3', 1], "go", alpha=0.4)
    plt.plot(X[y == '2', 0], X[y == '2', 1], "bo", alpha=0.4)
    plt.plot(X[y == '1', 0], X[y == '1', 1], "ro", alpha=0.4)
    plt.show()


def pca(X, comp=None):

    pca = sklearn.decomposition.PCA(n_components='mle' if not comp else comp, svd_solver='full' if not comp else 'auto')
    pca.fit(X)

    return pca.transform(X)


def pca_knn():
    trn_X, tst_X, trn_y, tst_y = train_test_split(*load_data(), test_size=0.33, shuffle=True)
    trn_X_2 = pca(trn_X, comp=2)

    #plot(trn_X_2, trn_y)

    # Training phase
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(trn_X_2, trn_y)

    # Testing phase
    tst_X_2 = pca(tst_X, comp=2)
    y_pred = knn.predict(tst_X_2)

    print(sklearn.metrics.confusion_matrix(tst_y, y_pred))
    print(sklearn.metrics.accuracy_score(tst_y, y_pred))


def lda():
    trn_X, tst_X, trn_y, tst_y = train_test_split(*load_data(), test_size=0.33, shuffle=True)

    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(trn_X, trn_y)
    trn_X_2 = lda.transform(trn_X)

    #plot(trn_X_2, trn_y)

    y_pred = lda.predict(tst_X)

    #knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    #knn.fit(trn_X_2, trn_y)
    #y_pred = knn.predict(lda.transform(tst_X))

    print(sklearn.metrics.confusion_matrix(tst_y, y_pred))
    print(sklearn.metrics.accuracy_score(tst_y, y_pred))

def reshape_data(X, y):
    rows = X.shape[0]//225
    X = X.reshape([225, rows]).T
    y = y.T
    return X, y


def load_data():
    labels = np.load(os.path.join("data", "Patches", "labels.npy"))
    patches = np.load(os.path.join("data", "Patches", "patches.npy"))
    return patches, labels


def create_patches_for_image(i):
    patch_size = 50
    padding_size = 100
    path_to_image = os.path.join("data", "Images", "img{}.sdt".format(i))
    path_to_ground_truth = os.path.join("data", "GroundTruths", "img{}.lxyr".format(i))

    # Padding
    image = np.fromfile(path_to_image, dtype='uint8')
    image = image.reshape((1024, 1024))
    new_image = np.zeros((1024 + 2 * padding_size, 1024 + 2 * padding_size), dtype='uint8')
    new_image[padding_size:-padding_size, padding_size:-padding_size] = image
    image = new_image
    #image = exposure.equalize_adapthist(image, clip_limit=0.03)

    with open(path_to_ground_truth) as lines:
        for line in lines:
            line_array = line.split(" ")
            x_pos = int(float(line_array[1])) + 100
            y_pos = int(float(line_array[2])) + 100
            patch = image[y_pos:y_pos + patch_size, x_pos:x_pos + patch_size]
            global LABELS
            LABELS.append(line[0])
            global PATCHES
            PATCHES.append(patch.ravel())


def create_patches():
    for image in range(134):
        create_patches_for_image(image+1)

    global LABELS
    LABELS = np.asarray(LABELS)
    global PATCHES
    PATCHES = np.asarray(PATCHES)

    np.save(os.path.join("data", "Patches", "patches.npy"), PATCHES)
    np.save(os.path.join("data", "Patches", "labels.npy"), LABELS)


if __name__ == "__main__":
    #create_patches()
    #pca_knn()
    lda()


