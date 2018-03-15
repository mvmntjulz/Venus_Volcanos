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

PATCHES = list()
LABELS = list()
HOG_PATCHES = list()


def plot(ax, X, y):
    ax.plot(X[y == 4, 0], X[y == 4, 1], "ko", alpha=0.2)
    ax.plot(X[y == 3, 0], X[y == 3, 1], "go", alpha=0.4)
    ax.plot(X[y == 2, 0], X[y == 2, 1], "bo", alpha=0.4)
    ax.plot(X[y == 1, 0], X[y == 1, 1], "ro", alpha=0.4)



def pca_knn(trn_X, tst_X, trn_y, tst_y):
    pca = sklearn.decomposition.KernelPCA(n_components=2)
    pca.fit(trn_X)

    trn_X_2 = pca.transform(trn_X)
    plot(trn_X_2, trn_y)

    # Training phase
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)
    knn.fit(trn_X_2, trn_y)

    # Testing phase
    tst_X_2 = pca.transform(tst_X)
    plt.figure()
    plt.title("Train")
    plot(trn_X_2, trn_y)

    plt.figure()
    plt.title("Test - Ground-Truth")
    plot(tst_X_2, tst_y)

    y_pred = knn.predict(pca.transform(tst_X))

    plot(ax, tst_X_2, y_pred)

    plt.show()

    print("LDA:")
    print(sklearn.metrics.confusion_matrix(tst_y, y_pred))
    print(sklearn.metrics.accuracy_score(tst_y, y_pred))


def lda(trn_X, tst_X, trn_y, tst_y):
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(trn_X, trn_y)

    trn_X_2 = lda.transform(trn_X)
    tst_X_2 = lda.transform(tst_X)

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.set_title("Train")
    plot(ax1, trn_X_2, trn_y)

    ax2 = fig.add_subplot(132)
    ax2.set_title("Test - GT after lda.transform")
    plot(ax2, tst_X_2, tst_y)

    y_pred = lda.predict(tst_X)

    ax3 = fig.add_subplot(133)
    ax3.set_title("Test - Post-Predict")
    plot(ax3, tst_X_2, y_pred)

    plt.tight_layout()
    plt.show()

    print("LDA:")
    cm = sklearn.metrics.confusion_matrix(tst_y, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    print(cm)
    print(sklearn.metrics.accuracy_score(tst_y, y_pred))


def reshape_data(X, y):
    rows = X.shape[0]//225
    X = X.reshape([225, rows]).T
    y = y.T
    return X, y


def load_data():
    labels = np.load(os.path.join("data", "Patches", "labels.npy"))
    labels = np.asarray(labels, dtype=np.uint8)

    patches = np.load(os.path.join("data", "Patches", "patches.npy"))

    ones = patches[labels == 1]
    twos = patches[labels == 2]
    threes = patches[labels == 3]
    fours = patches[labels == 4]

    minimum = min(len(ones), len(twos), len(threes), len(fours))
    labels = np.asarray([1 for x in range(minimum)] + [2 for x in range(minimum)] + [3 for x in range(minimum)] + [4 for x in range(minimum)])
    patches = np.concatenate([ones[:minimum,:], twos[:minimum,:], threes[:minimum, :], fours[:minimum, :]])

    return patches, labels


def create_patches_for_image(i):
    patch_size = 16
    padding_size = 20
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


def create_hog_patches(X):
    hog_fv = list()
    size = int(np.sqrt(X.shape[1]))
    for x in X:
        x_q = x.reshape(size, size)
        fv, hi = hog(x_q, visualise=True, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4))
        hog_fv.append(fv)

    return np.asarray(hog_fv)


if __name__ == "__main__":
    create_patches()

    use_hog = False
    mnist = False

    if mnist:
        data = sklearn.datasets.load_digits()
        patches, labels = data.data, data.target
    else:
        patches, labels = load_data()
        patches = create_hog_patches(patches) if use_hog else patches

    data = train_test_split(patches, labels, test_size=0.33, shuffle=True)

    #pca_knn(*data)
    lda(*data)


