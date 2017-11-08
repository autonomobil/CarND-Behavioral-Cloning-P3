import csv
import numpy as np
import cv2
import pickle
import random
import glob

import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Conv2D, LeakyReLU, Cropping2D, SimpleRNN
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import regularizers
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

dump_pfile = 0
load_pfile = 0

data_paths =['./me_data/driving_log.csv',
            './me_data2/driving_log.csv',
            './data/driving_log.csv']

print("Getting train data ready...")

if load_pfile:
    with open('./normalized_traindata/traindata.p', mode='rb') as dat:
        train = pickle.load(dat)
    X_train, y_train = train['features'], train['labels']

else:
    images = []
    measurements = []

    for path in data_paths:

        log = []
        with open(path) as logfile:
            reader = csv.reader(logfile)
            for line in reader:

            path = line[0]
            filename  = path.split('\\')[-1]
            current_path = './me_data/IMG/' + filename
            image = mpimg.imread(current_path)

            images.append(image)
            images.append(np.fliplr(image))

            steering = float(line[3])
            # create adjusted steering measurements for the side camera images
            measurements.append(steering)
            measurements.append(-steering)

    X_train = np.array(images)
    y_train = np.array(measurements)

if dump_pfile and not load_pfile:
    ### dump into .p-file
    imgs = {'features' : X_train, 'labels': y_train}

    with open('./normalized_traindata/traindata.p', 'wb') as handle:
        pickle.dump(imgs, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("X_train and y_train ready... ")
print("*****TRAINING******")
print()

model = Sequential()
model.add(Cropping2D(cropping=((60,20),(0,0)), input_shape =(160,320,3)))
model.add(Lambda(lambda x: (x/255 - 0.5)*2))

model.add(Conv2D(3, kernel_size = 1, strides = 1, padding="same"))

model.add(Conv2D(64, kernel_size = 7, strides = 1, padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(64, kernel_size = 7, strides = 1, kernel_regularizer=regularizers.l2(0.01)))
model.add(LeakyReLU(alpha=0.1))
model.add(AveragePooling2D())
model.add(Dropout(0.8))

model.add(Conv2D(128, kernel_size = 5, strides = 1, padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(128, kernel_size = 3, strides = 1, kernel_regularizer=regularizers.l2(0.01)))
model.add(LeakyReLU(alpha=0.1))
model.add(AveragePooling2D())
model.add(Dropout(0.8))

model.add(Flatten())

#model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(84))
model.add(LeakyReLU(alpha=0.1))

model.add(Dense(1))

model.compile(loss='mse', optimizer ='adam')
history_object = model.fit(X_train, y_train, validation_split=0.20, shuffle = True, epochs = 10, callbacks=[early_stopping])
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

path = 'C:/Users/morit/CarND-Behavioral-Cloning-P3/models/*.h5'
no =  len(glob.glob(path))

model.save('./models/model{}.h5'.format(no+1))
