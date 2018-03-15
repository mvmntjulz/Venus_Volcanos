import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import sklearn.discriminant_analysis
from skimage import exposure
from skimage.feature import canny, daisy
from skimage.morphology import reconstruction
from skimage.restoration import denoise_bilateral, denoise_wavelet, denoise_tv_chambolle
from sklearn.model_selection import train_test_split
import skimage.feature
import sklearn.decomposition

PATCH_SIZE=15

def load_data(i):
    image = np.fromfile(os.path.join("data","Images","img{}.sdt".format(i)), dtype=np.uint8) 
    gt = np.loadtxt(os.path.join("data", "GroundTruths", "img{}.lxyr".format(i)))

    return image, gt

def create_patches(image, coords):
    coords = coords.T
    patches = None
    for x, y in coords.T:
        x, y = int(x), int(y)

        ypos = y+50-PATCH_SIZE//2
        xpos = x+50-PATCH_SIZE//2
        patch = image[ypos:ypos+PATCH_SIZE, xpos:xpos+PATCH_SIZE]
        #patch_hog, hi = skimage.feature.hog(patch, pixels_per_cell=(10,10), cells_per_block=(5,5), visualise=True)
        #plt.imshow(hi)
        #plt.show()
        flat_patch = patch.ravel()
        patches = np.vstack([patches, flat_patch]) if patches is not None else flat_patch

    return np.asarray(patches)

def plot(X, y):
    plt.plot(X[y == 4, 0], X[y == 4, 1], "ko", alpha=0.2)
    plt.plot(X[y == 3, 0], X[y == 3, 1], "go", alpha=0.4)
    plt.plot(X[y == 2, 0], X[y == 2, 1], "bo", alpha=0.4)
    plt.plot(X[y == 1, 0], X[y == 1, 1], "ro", alpha=0.4)
    plt.show()


if __name__ == "__main__":
    all_patches = None
    all_labels = None
    for i in range(1,134):
        image, gt = load_data(i)
        image = np.reshape(image, (1024, 1024))
        ni = np.zeros((1024 + 2 * 50, 1024 + 2 * 50), dtype="uint8")
        ni[50:-50, 50:-50] = image
        image = ni
        image = exposure.equalize_hist(image)
        # image = denoise_tv_chambolle(image, multichannel=False)

        plt.imshow(image)
        plt.show()

        #image =
        #plt.imshow(rec)
        #plt.show()

        gt = np.atleast_2d(gt)
        if len(gt) == 0:
            continue

        coords = gt[:,1:3] if len(gt) > 1 else gt[1:3]
        labels = np.asarray(gt[:,0].T if len(gt) > 1 else gt[0], dtype=np.uint8)

        patches = create_patches(image, coords)
        if patches is not np.atleast_2d(patches):
            continue

        all_patches = np.vstack([all_patches, patches]) if all_patches is not None else patches
        all_labels = np.concatenate([all_labels, labels]) if all_labels is not None else labels

    X, tst_X, y, tst_y = train_test_split(all_patches, all_labels, test_size=0.33, shuffle=True)

    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X, y)
    X = lda.transform(X)
    #plot(X, y)
    #plot(lda.transform(tst_X), tst_y)

    y_pred = lda.predict(tst_X)
    #plot(lda.transform(tst_X), y_pred)

    # pca = sklearn.decomposition.PCA(n_components=2)
    # pca.fit(X)
    # X = pca.transform(X)
    # tst_X = pca.transform(tst_X)
    # plot(X, y)

    # knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    # knn.fit(X, y)
    #y_pred = knn.predict(tst_X)



    print(sklearn.metrics.confusion_matrix(tst_y, y_pred))
    print(sklearn.metrics.accuracy_score(tst_y, y_pred))




        



    




