import csv
import numpy as np
import cv2
import random
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from data_augmentation import create_augmen_img_bytransform, create_augmen_img_byflip

def get_logs(data_paths):
    logs = []

    for data_path in data_paths:
        input_file = csv.reader(open(data_path + '/driving_log.csv'))
        for line in input_file:
            line[2] = line[2].replace('\\', '/')
            line[2] = line[2].replace(' ', '')
            line[1] = line[1].replace('\\', '/')
            line[1] = line[1].replace(' ', '')
            line[0] = line[0].replace('\\', '/')
            line[0] = line[0].replace(' ', '')
            logs.append(line)

    logs = np.array(logs)
    return logs


def clean_logs(logs, num_bins = 100, plot_distribution = 0, factor_max = 1.5):
    steering = np.asfarray(logs[:,3])
    #windows....
    print("data cleaning...")
    # Use the angle to determine probability of deleting imgs
    # angle smaller -> probability higher
    hist, bins = np.histogram(steering, num_bins)
    avg_num_imgs= len(steering)/num_bins
    maximum = int(avg_num_imgs*factor_max)

    if plot_distribution:
        plt.figure(figsize=(8,5))
        center = (bins[:-1] + bins[1:]) / 2
        axes = plt.bar(center, hist, align='center', width=  (bins[1] - bins[0])*0.8, log = True)


    too_much = hist - maximum
    delete_this =[]

    for i in range(num_bins):
        if too_much[i] > 0:
            #for j in range(too_much[i]):
            indcs =[]
            indcs = np.array(np.where(np.logical_and(steering >= bins[i], steering < bins[i+1])))
            indcs = indcs[0]
            choices = indcs[np.random.choice(len(indcs), too_much[i], replace=False)]

            delete_this = np.append(delete_this, choices)
    delete_this = np.asarray(delete_this, dtype= int)
    print()
    print("DELETION")
    print("length delete_this", len(delete_this))
    print("log entries before cleaning shape:", logs.shape)
    logs = np.delete(logs, delete_this, axis=0)
    steering = np.asfarray(logs[:,3])
    print("log entries after cleaning shape:", logs.shape)

    if plot_distribution:
        # print histogram again to show a more even distribution of steering
        hist, bins = np.histogram(steering, num_bins)
        axes = plt.bar(center, hist, align='center', width=(bins[1] - bins[0])*0.8, log = True)
        axes = plt.plot((np.min(steering), np.max(steering)), (maximum, maximum), 'k-')
        #plt.yscale('log')
        #axes.set_ylim([100,10000])

        plt.show()


    ######################################################

    return logs


def load_img(sample, perspective, steering_correction = 0.2):
    img_path = sample[perspective]
   #img_path = img_path.replace('\\', '/') #windows....
    img = mpimg.imread(img_path)
    angle = float(sample[3])

    if perspective == 1:
        angle = angle + steering_correction
    elif perspective == 2:
        angle = angle - steering_correction

    return img, angle


def augment_img(img):

    rand = np.random.random() # decide randomly if augment flipped or normal img
    if rand >= 0.5:
        brightness_or_colorshift = 0
    else:
        brightness_or_colorshift = 1
    new_img = create_augmen_img_bytransform(img, range_colorshift = 0.2, brightness_or_colorshift= brightness_or_colorshift, warp_factor= 0, range_zoom = 0.00, range_rotate = 0,  range_shift = 0)
    return new_img


def get_gen_batch_size(desired_batch_size, augmented_per_sample, use_all_perspectives)  :
    if augmented_per_sample > 0:
        gen_batch_size = desired_batch_size //( 2 * (1 + augmented_per_sample))
    else:
        gen_batch_size = desired_batch_size // 2

    if use_all_perspectives:
        gen_batch_size = gen_batch_size//3
    return gen_batch_size
