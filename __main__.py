import numpy as np
import os
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.metrics
import sklearn.datasets
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from matplotlib import pylab as plt
from skimage import exposure
import scipy.misc

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import sklearn.cluster
from sklearn.model_selection import KFold

PATCHES = list()
LABELS = list()
HOG_PATCHES = list()


def plot(ax, X, y):
    ax.plot(X[y == 4, 0], X[y == 4, 1], "ko", alpha=0.4)
    ax.plot(X[y == 3, 0], X[y == 3, 1], "go", alpha=0.4)
    ax.plot(X[y == 2, 0], X[y == 2, 1], "bo", alpha=0.4)
    ax.plot(X[y == 1, 0], X[y == 1, 1], "ro", alpha=0.4)


def pca_knn(trn_X, tst_X, trn_y, tst_y, do_plot=False):
    pca = sklearn.decomposition.KernelPCA(n_components=2)
    pca.fit(trn_X)

    trn_X_2 = pca.transform(trn_X)
    if do_plot:
        plot(plt, trn_X_2, trn_y)
        plt.show()

    # Training phase
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)
    knn.fit(trn_X_2, trn_y)

    # Testing phase
    tst_X_2 = pca.transform(tst_X)
    if do_plot:
        plt.figure()
        plt.title("Train")
        plot(plt, trn_X_2, trn_y)

        plt.figure()
        plt.title("Test - Ground-Truth")
        plot(plt, tst_X_2, tst_y)

    y_pred = knn.predict(pca.transform(tst_X))

    if do_plot:
        plot(plt, tst_X_2, y_pred)
        plt.show()

    #print("PCA:")
    #print(sklearn.metrics.confusion_matrix(tst_y, y_pred))
    #print(sklearn.metrics.accuracy_score(tst_y, y_pred))

    acc = sklearn.metrics.accuracy_score(tst_y, y_pred)
    prec = sklearn.metrics.precision_score(tst_y, y_pred, average='weighted')
    f = sklearn.metrics.f1_score(tst_y, y_pred, average='weighted')

    return pca, acc, prec, f


def lda(trn_X, tst_X, trn_y, tst_y, do_plot=False):
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(trn_X, trn_y)

    trn_X_2 = lda.transform(trn_X)
    tst_X_2 = lda.transform(tst_X)

    if do_plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.set_title("Train")
        plot(ax1, trn_X_2, trn_y)

        ax2 = fig.add_subplot(132)
        ax2.set_title("Test - GT after lda.transform")
        plot(ax2, tst_X_2, tst_y)

    y_pred = lda.predict(tst_X)

    if do_plot:
        ax3 = fig.add_subplot(133)
        ax3.set_title("Test - Post-Predict")
        plot(ax3, tst_X_2, y_pred)

        plt.tight_layout()
        plt.show()

    cm = sklearn.metrics.confusion_matrix(tst_y, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)

    acc = sklearn.metrics.accuracy_score(tst_y, y_pred)
    prec = sklearn.metrics.precision_score(tst_y, y_pred, average='weighted')
    f = sklearn.metrics.f1_score(tst_y, y_pred, average='weighted')

    #print("LDA:")
    #print(cm)
    #print("Acc:", acc)
    #print("Prec:", prec)
    #print("F-Score:", f)

    return lda, acc, prec, f


def reshape_data(X, y):
    rows = X.shape[0]//225
    X = X.reshape([225, rows]).T
    y = y.T
    return X, y


def load_data(balance=False):
    labels = np.load(os.path.join("data", "Patches", "labels.npy"))
    labels = np.asarray(labels, dtype=np.uint8)

    patches = np.load(os.path.join("data", "Patches", "patches.npy"))

    if balance:
        ones = patches[labels == (1 or 2)]
        twos = patches[labels == 2]
        threes = patches[labels == (3 or 4)]
        fours = patches[labels == 4]

        minimum = min(len(ones), len(twos), len(threes), len(fours))
        labels = np.asarray([1 for x in range(minimum)] + [2 for x in range(minimum)] + [3 for x in range(minimum)] + [4 for x in range(minimum)])
        patches = np.concatenate([ones[:minimum,:], twos[:minimum,:], threes[:minimum, :], fours[:minimum, :]])

    return patches, labels


def create_patches_for_image(i, patch_size=16, padding_size= 20, combine=False):
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
            x_pos = int(float(line_array[1])) + padding_size - patch_size//2
            y_pos = int(float(line_array[2])) + padding_size - patch_size//2
            patch = image[y_pos:y_pos + patch_size, x_pos:x_pos + patch_size]
            #plt.imshow(patch)
            #plt.show()

            global LABELS
            if combine:
                if line[0] == '1' or line[0] == '2':
                    LABELS.append(1)
                else:
                    LABELS.append(int(line[0]))
                    #LABELS.append(3)

            else:
                LABELS.append(int(line[0]))

            global PATCHES
            PATCHES.append(patch.ravel())


def create_patches(patch_size=16, padding_size= 20, combine=False):
    for image in range(134):
        create_patches_for_image(image+1, patch_size, padding_size, combine)

    global LABELS
    LABELS = np.asarray(LABELS)
    global PATCHES
    PATCHES = np.asarray(PATCHES)

    np.save(os.path.join("data", "Patches", "patches.npy"), PATCHES)
    np.save(os.path.join("data", "Patches", "labels.npy"), LABELS)


def create_hog_patches(X):
    hog_fv = list()
    size = int(np.sqrt(X.shape[1]))
    for x in X:
        x_q = x.reshape(size, size)
        fv, hi = hog(x_q, visualise=True, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4))
        hog_fv.append(fv)

    return np.asarray(hog_fv)


def adaboost(patches, labels):
    clf = AdaBoostClassifier(n_estimators=1000)
    scores = cross_val_score(clf, patches, labels)
    print(scores.mean())


if __name__ == "__main__":

    # PARAMETERS ========

    # Patches:
    patch_size = 16
    padding_size = 20
    combine = False
    balance = False
    use_hog = False

    # Other:
    mnist = False
    do_plot = False
    n_splits = 10

    # ===================

    create_patches(patch_size, padding_size, combine)

    if mnist:
        data = sklearn.datasets.load_digits()
        X, y = data.data, data.target
    else:
        X, y = load_data(balance)
        X = create_hog_patches(X) if use_hog else X


    kf = KFold(n_splits=n_splits, shuffle=True)
    global_acc = 0
    global_prec = 0
    global_f = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        data = X_train, X_test, y_train, y_test
        #LDA
        model_lda, acc, prec, f = lda(*data, do_plot)

        #PCA
        #model_pca, acc, prec, f = pca_knn(*data, do_plot)

        global_acc += acc
        global_prec += prec
        global_f += f

    global_acc /= kf.get_n_splits(X)
    global_prec /= kf.get_n_splits(X)
    global_f /= kf.get_n_splits(X)

    print("Acc:", global_acc)
    print("Prec:", global_prec)
    print("F-Score:", global_f)

    #data = train_test_split(patches, labels, test_size=0.33, shuffle=True)
    #adaboost(patches, labels)


