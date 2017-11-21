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
from keras.optimizers import Adam

from helper import get_logs, clean_logs, load_img, augment_img, get_gen_batch_size, resize_img

from data_augmentation import create_augmen_img_byflip

################################
batch_size = 50
augmented_per_sample = 1
plot_distribution = 0
use_all_perspectives = 0
model_loader = 'my_model'

# define input shape for CNN
input_shape = (160,320,3)

# resized_shape = (128,128)

# data folder paths
data_paths =['./me_data',
            './me_data2',
            './me_data3',
            './me_data4',
            './me_track2_data',
            './me_track2_data2',
            './me_track2_data3',
            './data']

# get the cvs logs and load them into a np.array
logs = get_logs(data_paths)

# clean the logs for steering angle near 0, see distribution plot
logs = clean_logs(logs, num_bins =  51, plot_distribution = plot_distribution, factor_max = 1.3)

# split in training and testing
train_logs, valid_logs = train_test_split(logs, test_size=0.2)
print("logs ready")

# check if center, left,and right camera image should be used or just center image
nb_perspectives = 1
if use_all_perspectives:
    nb_perspectives = 3

################################
# defition of generator
def generator(samples, batch_size, augmented_per_sample):
    shuffle(samples)

    while 1:
        # calculate generator batch size, which is smaller than batch because of perspectives, flipping and augmentation
        gen_batch_size = get_gen_batch_size(batch_size, augmented_per_sample, use_all_perspectives)

        for offset in range(0, len(samples), gen_batch_size):
            batch = samples[offset:offset + gen_batch_size]
            imgs = []
            steering = []

            for sample in batch:

                #iterate over perspectives if use_all_perspectives = 1
                for pers in range(nb_perspectives):
                    img, angle = load_img(sample, perspective = pers, steering_correction = 0.175)

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
if model_loader == 'my_model':
    # model architecture
    model = Sequential()
    model.add(Cropping2D(cropping=((70,10),(0,0)), input_shape = input_shape))
    model.add(Lambda(resize_img))
    model.add(Lambda(lambda x: (x/255 - 0.5)*2))
    model.add(Conv2D(3, kernel_size = 1, strides = 1, padding="same"))

    model.add(Conv2D(32, kernel_size = 7, strides = 2, padding="same", kernel_regularizer=regularizers.l2(0.001)))
    model.add(ELU(alpha=0.1))
    model.add(Conv2D(32, kernel_size = 7, strides = 2, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(ELU(alpha=0.1))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size = 5, strides = 1, padding="same", kernel_regularizer=regularizers.l2(0.001)))
    model.add(ELU(alpha=0.1))
    model.add(Conv2D(64, kernel_size = 3, strides = 1, kernel_regularizer=regularizers.l2(0.001)))
    model.add(ELU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(ELU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(ELU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(8))
    model.add(ELU(alpha=0.1))

    model.add(Dense(1))

    optimizer = Adam(lr=1e-4)
    model.compile(optimizer=optimizer, loss='mse')

elif model_loader == 'nvidia':

    model = Sequential()
    model.add(Cropping2D(cropping=((70,10),(0,0)), input_shape = input_shape))
    model.add(Lambda(resize_img))
    model.add(Lambda(lambda x: (x/255 - 0.5)*2))

    # 1x1 convolution layer to automatically determine best color model
    model.add(Conv2D(3, kernel_size = 1, strides = 1, padding="same"))

    # NVIDIA model
    model.add(Conv2D(24, kernel_size = 5, strides = 2))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(36, kernel_size = 5, strides = 2))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(48, kernel_size = 5, strides = 2))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(64, kernel_size = 3, strides = 1))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(64, kernel_size = 3, strides = 1))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(ELU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(ELU(alpha=0.1))

    model.add(Dense(1))
    optimizer = Adam(lr=1e-4)
    model.compile(optimizer=optimizer, loss='mse')

else:
    model = None
    model = load_model(model_loader)
    print('Previous model {} loaded!'.format(model_loader))

############################
# set batches_per_epoch for the fit_generator
nb_real_train_sample = len(train_logs)
nb_real_valid_sample = len(valid_logs)
print("len(train_logs): ",len(train_logs))
print("len(valid_logs): ",len(valid_logs))
print('')

gen_train_batch_size = get_gen_batch_size(batch_size, augmented_per_sample, use_all_perspectives)
batches_per_epoch = int(np.ceil(nb_real_train_sample/gen_train_batch_size))
factor_batch_train = 2*(1 + augmented_per_sample)*nb_perspectives
real_batch_size = gen_train_batch_size * factor_batch_train
print("factor for train data: {} (flipping = 2 * augmented_per_sample = {} * nb_perspectives = {})".format(factor_batch_train,
                                                                                                    1 + augmented_per_sample,
                                                                                                    nb_perspectives))
print("total number of training images: ", nb_real_train_sample * factor_batch_train)
print("real training batch size(with flipping, augmentation and perspectives): ", real_batch_size)
print("train batches per epoch: ", batches_per_epoch)
print('')

# set validation_steps for the fit_generator
gen_valid_batch_size = get_gen_batch_size(batch_size, 0, use_all_perspectives)
validation_steps = int(np.ceil(nb_real_valid_sample/gen_valid_batch_size))
print("factor for valid data(only flipping):", 2*nb_perspectives)
print("real valid batch size(with flipping): ", gen_valid_batch_size*2*nb_perspectives)
print("total number of valid images: ", nb_real_valid_sample*2*nb_perspectives)
print("validation batches per epoch:", validation_steps)

############################
# create callbacks for checkpoint saving and earlystopping callbacks
checkpoint = ModelCheckpoint(filepath="./models/best-{val_loss:.4f}.h5", monitor='val_loss', verbose=0)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=2)

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
