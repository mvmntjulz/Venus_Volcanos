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
import copy
import random
from scipy.spatial import distance

WINDOWS = list()
PATCHES = list()
LABELS = list()
HOG_PATCHES = list()


def plot(ax, X, y):
    ax.plot(X[y == 4, 0], X[y == 4, 1], "ko", alpha=0.4)
    ax.plot(X[y == 3, 0], X[y == 3, 1], "go", alpha=0.4)
    #ax.plot(X[y == 2, 0], X[y == 2, 1], "bo", alpha=0.4)
    ax.plot(X[y == 1, 0], X[y == 1, 1], "ro", alpha=0.4)


def pca_knn(trn_X, tst_X, trn_y, tst_y, do_plot=False):
    pca = sklearn.decomposition.PCA(n_components=2)
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

    print("PCA:")
    print(sklearn.metrics.confusion_matrix(tst_y, y_pred))
    print(sklearn.metrics.accuracy_score(tst_y, y_pred))

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
    print(cm)

    return lda, acc, prec, f


def reshape_data(X, y):
    rows = X.shape[0]//225
    X = X.reshape([225, rows]).T
    y = y.T
    return X, y


def load_data(balance=False, win=False):
    if win:
        labels = np.load(os.path.join("data", "Windows", "labels.npy"))
        patches = np.load(os.path.join("data", "Windows", "windows.npy"))
    else:
        labels = np.load(os.path.join("data", "Patches", "labels.npy"))
        patches = np.load(os.path.join("data", "Patches", "patches.npy"))

    labels = np.asarray(labels, dtype=np.uint8)

    if balance:
        ones = patches[labels == 1]
        fours = patches[labels == 4]

        minimum = min(len(ones), len(fours))
        labels = np.asarray([1 for x in range(minimum)] + [4 for x in range(minimum)])
        patches = np.concatenate([ones[:minimum,:], fours[:minimum, :]])

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
            global PATCHES
            if combine:
                if line_array[0] == '1' or line_array[0] == '2' or line_array[0] == '3':
                    LABELS.append(1)
                    PATCHES.append(patch.ravel())
                #else:
                    #LABELS.append(int(line[0]))
                    #LABELS.append(3)
                    #PATCHES.append(patch.ravel())
            else:
                LABELS.append(int(line[0]))


def create_patches(patch_size=16, padding_size= 20, combine=False, with_zeros=False):
    for image in range(134):
        create_patches_for_image(image+1, patch_size, padding_size, combine)

    global LABELS
    LABELS = np.asarray(LABELS)
    global PATCHES
    PATCHES = np.asarray(PATCHES)

    if with_zeros:
        zero_patches, zero_labels = find_zeros(patch_size, len(LABELS[LABELS == 1]), len(LABELS[LABELS == 4]))
        PATCHES = np.vstack((PATCHES, zero_patches))
        LABELS = np.concatenate((LABELS, zero_labels))

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


def get_label_for_coords(x, y, window_size, gt, image=None):
    labels = list()

    for line in gt:
        line_array = line.split(" ")

        # Volcano mid point inside window
        if x < int(float(line_array[1])) < x + window_size and y < int(float(line_array[2])) < y + window_size:
            labels.append(int(float(line_array[0])))
            continue

        # Window mid point within volcano radius
        mid_point = (x + window_size//2, y + window_size//2)
        volcan_point = (int(float(line_array[1])), int(float(line_array[2])))
        if distance.euclidean(mid_point, volcan_point) < float(line_array[3]):
            labels.append(int(float(line_array[0])))

    if len(labels) == 1:
        return labels[0]
    elif len(labels) > 1:
        return -1
    elif len(labels) == 0:
        return 0


def create_windows():
    windows = list()
    windows_0 = list()
    labels = list()
    gt_list = list()

    for i in range(10):
        path_to_image = os.path.join("data", "Images", "img{}.sdt".format(i+1))
        path_to_ground_truth = os.path.join("data", "GroundTruths", "img{}.lxyr".format(i+1))
        image = np.fromfile(path_to_image, dtype='uint8')
        image = image.reshape((1024, 1024))
        window_size = 32

        x_range = range(0, image.shape[1] - window_size + 1, window_size//2)
        y_range = range(0, image.shape[0] - window_size + 1, window_size//2)

        print(i)
        print(len(windows), len(windows_0))

        with open(path_to_ground_truth) as gt:
            for line in gt:
                gt_list.append(line)

        for y_coord in y_range:
            for x_coord in x_range:
                window = image[y_coord:y_coord + window_size, x_coord:x_coord + window_size]
                label = get_label_for_coords(x_coord, y_coord, window_size, gt_list, image)
                if label > 0:
                    labels.append(label)
                    windows.append(window.ravel())
                if label == 0:
                    windows_0.append(window.ravel())

    for i in range(len(labels)):
        rand_index = random.randint(0, len(windows_0))

        windows.append(windows_0[rand_index])
        labels.append(0)

    windows = np.asarray(windows)
    labels = np.asarray(labels)

    np.save(os.path.join("data", "Windows", "windows.npy"), windows)
    np.save(os.path.join("data", "Windows", "labels.npy"), labels)
    return windows, labels


def find_zeros(patch_size, n_ones, n_fours):
    n_zeros = n_ones if n_fours == 0 else (n_ones + n_fours)//2
    zero_patches = list()
    labels = list()

    while len(zero_patches) < n_zeros:
        rdn_image_nbr = random.randint(1, 134)
        image = np.fromfile(os.path.join("data", "Images", "img{}.sdt".format(rdn_image_nbr)), dtype='uint8')
        image = image.reshape((1024, 1024))
        rnd_x = random.randint(0, 1024 - patch_size)
        rnd_y = random.randint(0, 1024 - patch_size)
        candidate = image[rnd_y:rnd_y + patch_size, rnd_x:rnd_x + patch_size]

        with open(os.path.join("data", "GroundTruths", "img{}.lxyr".format(rdn_image_nbr))) as gt:
            too_close = False

            for line in gt:
                line_array = line.split(" ")
                mid_point = (rnd_x + patch_size // 2, rnd_y + patch_size // 2)
                volcan_point = (int(float(line_array[1])), int(float(line_array[2])))
                diagonal = np.sqrt(2 * (patch_size//2)**2)
                if distance.euclidean(mid_point, volcan_point) < float(line_array[3]) + diagonal:
                    too_close = True
                    break
            if not too_close:
                zero_patches.append(candidate.ravel())
                labels.append(0)

    return np.asarray(zero_patches), np.asarray(labels)


if __name__ == "__main__":

    # PARAMETERS ========
    # Patches:
    patch_size = 16
    padding_size = 20
    combine = True
    balance = False
    use_hog = False
    windows = False
    with_zeros = True

    # Other:
    mnist = False
    do_plot = False
    n_splits = 10
    # ===================

    #X, y = create_windows()
    #data = train_test_split(X, y, test_size=0.33, shuffle=True)
    #model_lda, acc, prec, f = lda(*data, do_plot)
    #print("Acc:", acc)
    #print("Prec:", prec)
    #print("F-Score:", f)

    create_patches(patch_size, padding_size, combine, with_zeros)

    if mnist:
        data = sklearn.datasets.load_digits()
        X, y = data.data, data.target
    else:
        X, y = load_data(balance, windows)
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