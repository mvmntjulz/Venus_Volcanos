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
import skimage.morphology
import scipy.misc
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import sklearn.cluster
from sklearn.model_selection import KFold
import copy
import skimage.filters
import random
from scipy.spatial import distance
import matplotlib.patches as patches
from PIL import Image

WINDOWS = list()
PATCHES = list()
LABELS = list()
HOG_PATCHES = list()


def plot(ax, X, y):
    #ax.plot(X[y == 4, 0], X[y == 4, 1], "ko", alpha=0.4)
    #ax.plot(X[y == 3, 0], X[y == 3, 1], "go", alpha=0.4)
    #ax.plot(X[y == 2, 0], X[y == 2, 1], "bo", alpha=0.4)

    ax.plot(X[y == 0, 0], X[y == 0, 1], "ro", alpha=0.4)
    ax.plot(X[y == 1, 0], X[y == 1, 1], "bo", alpha=0.4)


# TODO: PCA and LDA/SVM afterwards
def pca_knn(trn_X, tst_X, trn_y, tst_y, do_plot=False):
    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(trn_X)

    trn_X_2 = pca.transform(trn_X)

    # Training phase
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)
    knn.fit(trn_X_2, trn_y)

    # Testing phase
    tst_X_2 = pca.transform(tst_X)


    if do_plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.set_title("Train")
        plot(ax1, trn_X_2, trn_y)

        ax2 = fig.add_subplot(132)
        ax2.set_title("Test - GT after pca transform")
        plot(ax2, tst_X_2, tst_y)

    y_pred = knn.predict(pca.transform(tst_X))

    if do_plot:
        ax3 = fig.add_subplot(133)
        ax3.set_title("Post-Predict with KNN")
        plot(ax3, tst_X_2, y_pred)

        plt.tight_layout()
        plt.show()


    print("PCA:")
    print(sklearn.metrics.confusion_matrix(tst_y, y_pred))
    print(sklearn.metrics.accuracy_score(tst_y, y_pred))

    acc = sklearn.metrics.accuracy_score(tst_y, y_pred)
    if len(set(LABELS)) == 2:
        prec = sklearn.metrics.precision_score(tst_y, y_pred, average='binary')
        rec = sklearn.metrics.recall_score(tst_y, y_pred, average='binary')
        f = sklearn.metrics.f1_score(tst_y, y_pred, average='binary')
    else:
        prec = sklearn.metrics.precision_score(tst_y, y_pred, average='weighted')
        rec = sklearn.metrics.recall_score(tst_y, y_pred, average='weighted')
        f = sklearn.metrics.f1_score(tst_y, y_pred, average='weighted')

    return pca, knn, acc, prec, rec, f


def lda(trn_X, tst_X, trn_y, tst_y, do_plot=False, patch_size=None):
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
    prec = sklearn.metrics.precision_score(tst_y, y_pred, average='binary')
    rec = sklearn.metrics.recall_score(tst_y, y_pred, average='binary')
    f = sklearn.metrics.f1_score(tst_y, y_pred, average='binary')
    print(cm)

    return lda, acc, prec, rec, f


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


def create_patches_for_image(i, patch_size=16, padding_size= 20, combine=False, manipulate_image=False):
    path_to_image = os.path.join("data", "Images", "img{}.sdt".format(i))
    path_to_ground_truth = os.path.join("data", "GroundTruths", "img{}.lxyr".format(i))

    # Padding
    image = np.fromfile(path_to_image, dtype='uint8')
    image = image.reshape((1024, 1024))
    new_image = np.zeros((1024 + 2 * padding_size, 1024 + 2 * padding_size), dtype='uint8')
    new_image[padding_size:-padding_size, padding_size:-padding_size] = image
    image = new_image
    if manipulate_image:
        image = skimage.exposure.adjust_gamma(image, 5)

    with open(path_to_ground_truth) as lines:
        for line in lines:
            line_array = line.split(" ")
            x_pos = int(float(line_array[1])) + padding_size - patch_size//2
            y_pos = int(float(line_array[2])) + padding_size - patch_size//2
            patch = image[y_pos:y_pos + patch_size, x_pos:x_pos + patch_size]

            global LABELS
            global PATCHES
            if combine:
                if line_array[0] == '1' or line_array[0] == '2' or line_array[0] == '3':
                    LABELS.append(1)
                    PATCHES.append(patch.ravel())
            else:
                LABELS.append(int(line[0]))
                PATCHES.append(patch.ravel())


def create_patches(patch_size=16, padding_size= 20, combine=False, with_zeros=False, manipulate_image=False):
    for image in range(134):
        create_patches_for_image(image+1, patch_size, padding_size, combine, manipulate_image)

    global LABELS
    LABELS = np.asarray(LABELS)
    global PATCHES
    PATCHES = np.asarray(PATCHES)

    if with_zeros:
        zero_patches, zero_labels = find_zeros(patch_size)
        PATCHES = np.vstack((PATCHES, zero_patches))
        LABELS = np.concatenate((LABELS, zero_labels))

    zipped = list(zip(PATCHES, LABELS))
    random.shuffle(zipped)

    PATCHES, LABELS  = zip(*zipped)

    np.save(os.path.join("data", "Patches", "patches.npy"), PATCHES)
    np.save(os.path.join("data", "Patches", "labels.npy"), LABELS)


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


def find_zeros(patch_size):
    distinct_labels = set(LABELS)
    n_zeros = len(LABELS) // len(distinct_labels)

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
                if distance.euclidean(mid_point, volcan_point) < float(line_array[3]) + diagonal and line_array[0] == '1' or line_array[0] == '2':
                    too_close = True
                    break
            if not too_close:
                zero_patches.append(candidate.ravel())
                labels.append(0)

    return np.asarray(zero_patches), np.asarray(labels)


def sliding_window_vote(model_array, image, image_nbr, patch_size):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    x_range = range(0, image.shape[1] - patch_size + 1, patch_size // 2)
    y_range = range(0, image.shape[0] - patch_size + 1, patch_size // 2)

    for y_coord in y_range:
        for x_coord in x_range:
            candidate = image[y_coord:y_coord + patch_size, x_coord:x_coord + patch_size].reshape(1, patch_size * patch_size)

            #plt.figure()
            #plt.imshow(image[y_coord:y_coord + patch_size, x_coord:x_coord + patch_size])
            votes = list()
            for model in model_array:
                if type(model) == tuple:
                    votes.append((model[1].predict(model[0].transform(candidate))[0]))
                else:
                    votes.append(model.predict(candidate)[0])

            label = max(set(votes), key=votes.count)
            if label == 1 and votes.count(1)/len(model_array) >= 0.8:
                rect = patches.Rectangle((x_coord, y_coord), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

    draw_rect_gt(image_nbr, ax)
    plt.show()


def draw_rect_gt(image_nbr, ax):
    with open(os.path.join("data", "GroundTruths", "img{}.lxyr".format(image_nbr))) as gt:
        print("Image used: {}".format(image_nbr))
        for line in gt:
            line_array = line.split(" ")
            if int(line_array[0]) == 1 or int(line_array[0]) == 2 or int(line_array[0]) == 3:
                color = 'w'
            else:
                color = 'orange'

            x_mid = int(float(line_array[1])) - int(float(line_array[3]))
            y_mid = int(float(line_array[2])) - int(float(line_array[3]))
            vulcano_radius = int(float(line_array[3]))
            rect = patches.Rectangle((x_mid, y_mid), 2*vulcano_radius, 2*vulcano_radius, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)


if __name__ == "__main__":

    # PARAMETERS ========
    # Patches:
    patch_size = 32
    padding_size = 100
    combine = True
    balance = False
    use_hog = False
    windows = False
    with_zeros = True
    manipulate_image = False

    # Other:
    do_plot = False
    n_splits = 10
    # ===================

    create_patches(patch_size, padding_size, combine, with_zeros, manipulate_image)
    X, y = load_data(balance, windows)

    kf = KFold(n_splits=n_splits, shuffle=True)
    global_acc = 0
    global_prec = 0
    global_rec = 0
    global_f = 0
    model_array = list()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        data = X_train, X_test, y_train, y_test

        #LDA
        #model_lda, acc, prec, rec, f = lda(*data, do_plot, patch_size)
        #model_array.append(model_lda)

        #PCA
        model_pca, model_knn, acc, prec, rec, f = pca_knn(*data, do_plot)
        model_array.append((model_pca, model_knn))

        global_acc += acc
        global_prec += prec
        global_rec += rec
        global_f += f

    global_acc /= kf.get_n_splits(X)
    global_prec /= kf.get_n_splits(X)
    global_rec /= kf.get_n_splits(X)
    global_f /= kf.get_n_splits(X)

    print("Acc:", global_acc)
    print("Prec:", global_prec)
    print("Rec:", global_rec)
    print("F-Score:", global_f)

    #Sliding Window
    while True:
        i = random.randint(1, 134)
        print(i)
        image_nbr = 66
        test_image = np.fromfile(os.path.join("data", "Images", "img{}.sdt".format(image_nbr)), dtype='uint8')
        test_image = test_image.reshape((1024, 1024))
        if manipulate_image:
            test_image = skimage.exposure.adjust_gamma(test_image, 5)

        sliding_window_vote(model_array, test_image, image_nbr, patch_size)


