import numpy as np
import os
from matplotlib import pylab as plt


def load_images():
    A = np.fromfile(os.path.join("data", "Images", "img2.sdt"), dtype='int8', sep="")
    A = A.reshape([1024, 1024])
    plt.imshow(A)
    plt.show()


if __name__ == "__main__":
    load_images()
