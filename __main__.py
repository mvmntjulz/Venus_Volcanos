"""
 Code for Project in FML-Course WT 17/18
 Authors: Max Klingmann and Julien Stern
"""
import itertools
import numpy as np
import os
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.metrics
import sklearn.datasets
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from matplotlib import pylab as plt, colors
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
import sklearn.gaussian_process
import sklearn.neural_network


PATCHES = list()
LABELS = list()
HOG_PATCHES = list()


def plot(ax, X, y):
    ax.plot(X[y == 0, 0], X[y == 0, 1], "ro", alpha=0.4)
    ax.plot(X[y == 4, 0], X[y == 4, 1], "mo", alpha=0.4)
    ax.plot(X[y == 3, 0], X[y == 3, 1], "co", alpha=0.4)
    ax.plot(X[y == 2, 0], X[y == 2, 1], "go", alpha=0.4)
    ax.plot(X[y == 1, 0], X[y == 1, 1], "bo", alpha=0.4)


def pca_classifier(trn_X, tst_X, trn_y, tst_y, do_plot=False, classifier='lda'):
    classfier_title = ""
    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(trn_X)
    trn_X_2 = pca.transform(trn_X)

    # Training ==========================================================
    if classifier == 'lda':
        classifier = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        classfier_title = "LDA"
    elif classifier == 'knn':
        classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)
        classfier_title = "KNN"
    elif classifier == 'qda':
        classifier = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
        classfier_title = "QDA"
    elif classifier == 'nn':
        classifier = sklearn.neural_network.MLPClassifier(solver='adam')
        classfier_title = "Neural Network"

    classifier.fit(trn_X_2, trn_y)

    # Testing ===========================================================
    tst_X_2 = pca.transform(tst_X)

    if do_plot:
        fig = plt.figure()

        Z, xx, yy = create_decision_boundry(trn_X_2, classifier)
        ax1 = fig.add_subplot(131)
        ax1.set_title("Training Set after PCA Transformation")
        ax1.contourf(xx, yy, Z, cmap='RdBu', alpha=.9)
        ax1.contour(xx, yy, Z, [0.5], linewidths=1.5, colors='k')
        ax1.set_xlim(-2000, 1500)
        ax1.set_ylim(-500, 500)
        plot(ax1, trn_X_2, trn_y)

        ax2 = fig.add_subplot(132)
        ax2.set_title("Testing Set after PCA Transformation")
        ax2.contourf(xx, yy, Z, cmap='RdBu', alpha=.9)
        ax2.contour(xx, yy, Z, [0.5], linewidths=1.5, colors='k')
        ax2.set_xlim(-2000, 1500)
        ax2.set_ylim(-500, 500)
        plot(ax2, tst_X_2, tst_y)

        y_pred = classifier.predict(pca.transform(tst_X))

        ax3 = fig.add_subplot(133)
        ax3.set_title("Label Prediction with {}".format(classfier_title))
        ax3.contourf(xx, yy, Z, cmap='RdBu', alpha=.9)
        ax3.contour(xx, yy, Z, [0.5], linewidths=1.5, colors='k')
        ax3.set_xlim(-2000, 1500)
        ax3.set_ylim(-500, 500)
        plot(ax3, tst_X_2, y_pred)

        fig.subplots_adjust(left=0.03, bottom=0.2, right=0.97, top=0.8, wspace=0.09, hspace=0.2)

        cm = sklearn.metrics.confusion_matrix(tst_y, y_pred)
        plot_confusion_matrix(cm, np.unique(tst_y), classfier_title)
        plt.show()
    else:
        y_pred = classifier.predict(pca.transform(tst_X))

    acc = sklearn.metrics.accuracy_score(tst_y, y_pred)
    if len(set(LABELS)) == 2:
        prec = sklearn.metrics.precision_score(tst_y, y_pred, average='binary')
        rec = sklearn.metrics.recall_score(tst_y, y_pred, average='binary')
        f = sklearn.metrics.f1_score(tst_y, y_pred, average='binary')
    else:
        prec = sklearn.metrics.precision_score(tst_y, y_pred, average='weighted')
        rec = sklearn.metrics.recall_score(tst_y, y_pred, average='weighted')
        f = sklearn.metrics.f1_score(tst_y, y_pred, average='weighted')

    return pca, classifier, acc, prec, rec, f


def plot_confusion_matrix(cm, classes, classfier_title, cmap=plt.cm.Blues):
    fig = plt.figure()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix for PCA + {}".format(classfier_title))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_classes_grid(patch_size=32):
    ones = PATCHES[LABELS == 1]
    twos = PATCHES[LABELS == 2]
    threes = PATCHES[LABELS == 3]
    fours = PATCHES[LABELS == 4]

    one_samples = list()
    two_samples = list()
    three_samples = list()
    fours_samples = list()

    for i in range(5):
        one_samples.append(ones[np.random.randint(ones.shape[0]), :])
        two_samples.append(twos[np.random.randint(twos.shape[0]), :])
        three_samples.append(threes[np.random.randint(threes.shape[0]), :])
        fours_samples.append(fours[np.random.randint(fours.shape[0]), :])

    all_samples = np.vstack((one_samples, two_samples, three_samples, fours_samples))
    fig = plt.figure()

    sub_plot_base = "45"
    for i in range(20):
        print(i+1)
        ax = fig.add_subplot(4, 5, i+1)
        ax.imshow(all_samples[i, :].reshape(patch_size, patch_size))
        ax.set_axis_off()
    plt.show()

def create_decision_boundry(X, clf):
    nx, ny = 200, 100
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    return Z, xx, yy

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
    print("Creating patches...")
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
    LABELS = np.asarray(LABELS)
    PATCHES = np.asarray(PATCHES)

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


def sliding_window_vote(model_array, image, image_nbr, patch_size, voting_threshold=0.7):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    x_range = range(0, image.shape[1] - patch_size + 1, patch_size // 2)
    y_range = range(0, image.shape[0] - patch_size + 1, patch_size // 2)

    for y_coord in y_range:
        for x_coord in x_range:
            candidate = image[y_coord:y_coord + patch_size, x_coord:x_coord + patch_size].reshape(1, patch_size * patch_size)
            votes = list()

            for model in model_array:
                if type(model) == tuple:
                    votes.append((model[1].predict(model[0].transform(candidate))[0]))
                else:
                    votes.append(model.predict(candidate)[0])

            label = max(set(votes), key=votes.count)
            if label == 1 and votes.count(1)/len(model_array) >= voting_threshold:
                rect = patches.Rectangle((x_coord, y_coord), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

    draw_rect_gt(image_nbr, ax)
    plt.show()


def draw_rect_gt(image_nbr, ax):
    with open(os.path.join("data", "GroundTruths", "img{}.lxyr".format(image_nbr))) as gt:
        for line in gt:
            line_array = line.split(" ")
            if int(line_array[0]) == 1 or int(line_array[0]) == 2 or int(line_array[0]) == 3:
                color = 'w'
            else:
                color = 'orange'

            x_mid = int(float(line_array[1])) - int(float(line_array[3]))
            y_mid = int(float(line_array[2])) - int(float(line_array[3]))
            volcano_radius = int(float(line_array[3]))
            rect = patches.Rectangle((x_mid, y_mid), 2*volcano_radius, 2*volcano_radius, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)


def sliding_window_for_images(model_array, patch_size, voting_threshold, example_image_nbr, use_example_image=True):
    if use_example_image:
        image_nbr = example_image_nbr
        print("Sliding-Window for image", image_nbr, "...")
        image = load_image(image_nbr, manipulate_image)
        sliding_window_vote(model_array, image, image_nbr, patch_size, voting_threshold)
    else:
        while True:
            image_nbr = random.randint(1, 134)
            print("Sliding-Window for image:", image_nbr, "...")
            image = load_image(image_nbr, manipulate_image)
            sliding_window_vote(model_array, image, image_nbr, patch_size, voting_threshold)


def load_image(image_nbr, manipulate_image=False):
    image = np.fromfile(os.path.join("data", "Images", "img{}.sdt".format(image_nbr)), dtype='uint8')
    image = image.reshape((1024, 1024))
    if manipulate_image:
        image = skimage.exposure.adjust_gamma(image, 5)
    return image


def do_kfold(X, y, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True)
    global_acc = 0
    global_prec = 0
    global_rec = 0
    global_f = 0
    model_array = list()

    print("Training...")
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        data = X_train, X_test, y_train, y_test

        print("Model {}/{}".format(i+1, n_splits))
        model_pca, model_classifier, acc, prec, rec, f = pca_classifier(*data, do_plot, classifier)
        model_array.append((model_pca, model_classifier))

        global_acc += acc
        global_prec += prec
        global_rec += rec
        global_f += f

    global_acc /= kf.get_n_splits(X)
    global_prec /= kf.get_n_splits(X)
    global_rec /= kf.get_n_splits(X)
    global_f /= kf.get_n_splits(X)

    print("Accuracy:", global_acc)
    print("Precision:", global_prec)
    print("Recall:", global_rec)
    print("F1-Score:", global_f)

    return model_array


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

    # Classifier
    classifier = 'lda'  # choose one: lda, qda, knn, nn

    # Plotting
    do_plot = False

    # Sliding Window
    n_splits = 10
    use_example_image = True
    example_image_nbr = 3
    voting_threshold = 0.8
    # ===================

    create_patches(patch_size, padding_size, combine, with_zeros, manipulate_image)
    #plot_classes_grid()

    X, y = load_data(balance, windows)

    model_array = do_kfold(X, y, n_splits)


    #Sliding Window for image/s
    sliding_window_for_images(model_array, patch_size, voting_threshold, example_image_nbr, use_example_image)


