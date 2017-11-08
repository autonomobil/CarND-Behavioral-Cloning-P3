import numpy as np
import cv2
import pickle
import random
import glob
import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.ndimage import zoom
from scipy.misc import imsave

training_file = './normalized_traindata/traindata.p'

with open(training_file, mode='rb') as dat:
    train = pickle.load(dat)

X_train, y_train = train['features'], train['labels']

# Helper function
def normalize_imgs(imgs, option = 1):
    X_train_normalized = imgs.copy()

    for i, image in enumerate(X_train_normalized):
        if i == 0:
            X_train_normalized = X_train_normalized.astype('float32')
        image = image.astype('float32')

        for color in range(3):
            min_val = image[:,:, color].min()
            min_val = min_val.astype('float32')
            max_val = image[:,:, color].max()
            max_val = max_val.astype('float32')

            val_range = (max_val - min_val)

            ###### Normalize for range 0, 1
            if option == 0:
                image[:,:, color] = (image[:,:,color] - (min_val)) / (val_range)
            ###### Normalize for range -1, 1
            elif option == 1:
                image[:,:, color] = (image[:,:,color] - (val_range/2 + min_val)) / (val_range/2)

            X_train_normalized[i] = image

    return X_train_normalized

mplnorm = mplcolors.Normalize(vmin = -1, vmax = 1)

X_train = normalize_imgs(X_train)

### dump into .p-file
augmentedX = {'features' : X_train, 'labels': y_train}

with open('./normalized_traindata/normalized_traindata.p', 'wb') as handle:
    pickle.dump(augmentedX, handle, protocol=pickle.HIGHEST_PROTOCOL)
