# import keras.backend as K
# K.clear_session()

import csv
import numpy as np
import random
from time import localtime, strftime

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Merge, BatchNormalization, Dropout, Activation, Conv2D, LeakyReLU, ELU, Cropping2D, SimpleRNN
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

from helper import get_logs, clean_logs, load_img, augment_img, get_gen_batch_size
from data_augmentation import create_augmen_img_byflip

################################
batch_size = 50
augmented_per_sample = 0
plot_distribution = 1
use_all_perspectives = 1
path_for_trained_model = './models/best-0.0420.h5'

data_paths =['./me_data',
            './me_data2',
            './me_data3',
            './me_data4',
            './me_track2_data',
            './me_track2_data2',
            './data']

logs = get_logs(data_paths)

# clean the logs for steering angle near 0
logs = clean_logs(logs, num_bins =  21, plot_distribution = plot_distribution, factor_max = 1.25)

# split in training and testing
train_logs, valid_logs = train_test_split(logs, test_size=0.2)
print("logs ready")
################################
# defition of generator

# check if center, left,and right should be used or just center image
nb_perspectives = 1
if use_all_perspectives:
    nb_perspectives = 3

def generator(samples, batch_size, augmented_per_sample):
    number_samples = len(samples)
    shuffle(samples)
    shuffle(samples)

    while 1:
        # calculate generator batch size, which smaller than batch because of perspectives, flipping and augmentation
        if augmented_per_sample > 0:
            gen_batch_size = get_gen_batch_size(batch_size, augmented_per_sample, use_all_perspectives)
        else:
            gen_batch_size = get_gen_batch_size(batch_size, 0, 0)

        for offset in range(0, number_samples, gen_batch_size):
            batch = samples[offset:offset + gen_batch_size]
            imgs = []
            steering = []

            for sample in batch:

                #iterate over perspectives if use_all_perspectives = 1
                for pers in range(nb_perspectives):
                    img, angle = load_img(sample, perspective = pers, steering_correction = 0.175 )

                    imgs.append(img)
                    steering.append(angle)

                    img_flipped, angle_flipped = create_augmen_img_byflip(img, angle)
                    imgs.append(img_flipped)
                    steering.append(angle_flipped)

                    if augmented_per_sample > 0:
                        for i in range(augmented_per_sample): # also augmentation for flipped
                            rand = np.random.random() # decide randomly if augment flipped or normal img
                            if rand >= 0.5:
                                new_img = augment_img(img_flipped)
                                imgs.append(new_img)
                                steering.append(angle_flipped)
                            else:
                                new_img = augment_img(img)
                                imgs.append(new_img)
                                steering.append(angle)

            Xtrain = np.array(imgs)
            ytrain = np.array(steering)
            yield shuffle(Xtrain, ytrain)

# create generators for training and validation
train_gen = generator(train_logs, batch_size = batch_size, augmented_per_sample = augmented_per_sample)
valid_gen = generator(valid_logs, batch_size = batch_size, augmented_per_sample = 0)

################################
if path_for_trained_model:
    model = None
    model = load_model(path_for_trained_model)
    print('Previous model {} loaded!'.format(path_for_trained_model))
else:
    # model architecture
    model = Sequential()
    model.add(Cropping2D(cropping=((65,23),(0,0)), input_shape = (160,320,3)))
    model.add(Lambda(lambda x: (x/255 - 0.5)*2))

    model.add(Conv2D(3, kernel_size = 1, strides = 1, padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 7, strides = 1, padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(32, kernel_size = 7, strides = 1, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(AveragePooling2D())
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 5, strides = 1, padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(64, kernel_size = 5, strides = 1, kernel_regularizer=regularizers.l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(AveragePooling2D())
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size = 3, strides = 1, padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(128, kernel_size = 3, strides = 1, kernel_regularizer=regularizers.l2(0.01)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(AveragePooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(8))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer ='adam')

############################
# set batches_per_epoch for the fit_generator
nb_real_train_sample = len(train_logs)
nb_real_valid_sample = len(valid_logs)
print("len(train_logs): ",len(train_logs))
print("len(valid_logs): ",len(valid_logs))

gen_train_batch_size = get_gen_batch_size(batch_size, augmented_per_sample, use_all_perspectives)
batches_per_epoch = int(np.ceil(nb_real_train_sample/gen_train_batch_size))
real_batch_size = gen_train_batch_size*2*(1 + augmented_per_sample)*nb_perspectives
print("factor for train data:", 2*(1 + augmented_per_sample)*nb_perspectives)
print("total number of training samples: ", nb_real_train_sample*2*(1 + augmented_per_sample)*nb_perspectives)
print("real batch size(with flipping and augmentation): ", real_batch_size)
print("batches per epoch: ", batches_per_epoch)

# set validation_steps for the fit_generator
gen_valid_batch_size = get_gen_batch_size(batch_size, 0, 0)
validation_steps = int(np.ceil(nb_real_valid_sample/gen_valid_batch_size))
print("total number of valid samples: ", nb_real_valid_sample*2)
print("validation batches per epoch:", validation_steps)

############################
# create callbacks for checkpoint saving and earlystopping callbacks
checkpoint = ModelCheckpoint(filepath="./models/best-{val_loss:.4f}.h5", monitor='val_loss', verbose=0)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=2)

############################
print()
print("*****TRAINING******")
print()



history_object = model.fit_generator(generator = train_gen,
                                    validation_data = valid_gen,
                                    steps_per_epoch = batches_per_epoch,
                                    use_multiprocessing=False,
                                    validation_steps = validation_steps,
                                    callbacks =[checkpoint, early_stopping],
                                    epochs = 15)

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#save model
model.save('./models/model-{:.4f}.h5'.format(history_object.history['val_loss'][-1]))

# save the summary with clock timestemp
timestemp = strftime("%H-%M-%S", localtime())
with open('./models/summary-{}.txt'.format(timestemp),'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

# del history_object
# del model
# K.clear_session()
