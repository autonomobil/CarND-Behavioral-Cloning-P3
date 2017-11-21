import csv
import numpy as np
import cv2
import pickle
import random
import glob
import time
import pandas
from sklearn.model_selection import train_test_split
from math import pi

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from helper import *

data_path ='./me_track2_data'

logs = []
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

pers = 0

def angle_line(input_img, angle):
    dim = input_img.shape[:2]
    h = dim[0]
    w = dim[1]
    radius = int(min(h,w) * 0.8)

    deg_angle = 90 - angle * 25
    rad_angle = deg_angle* pi / 180
    deltax = np.cos(rad_angle) *radius
    deltay = np.sin(rad_angle) *radius

    x0 = int(w/2)
    y0 = int(h)

    x1 = int(w/2 + deltax)
    y1 = int(h - deltay)
    linex = (x0,x1)
    liney = (y0,y1)
    return linex, liney

for log in logs[1000:]:
    plt.figure(figsize=(12, 6), facecolor='w', edgecolor='k')
    orig_img, angle = load_img(log, perspective = pers , steering_correction = 0.2)
    plt.subplot(221)
    linex, liney = angle_line(orig_img, angle)
    ax = plt.gca()
    ax.plot(linex, liney, 'r')
    plt.imshow(orig_img)
    plt.title('original, {}'.format(angle))

    img = crop_resize_img(orig_img)
    plt.subplot(222)
    linex, liney = angle_line(img, angle)
    ax = plt.gca()
    ax.plot(linex, liney, 'r')
    plt.imshow(img)
    plt.title('cropped & resized, {}'.format(angle))

    img_flipped, angle_flipped = create_augmen_img_byflip(img, angle)
    plt.subplot(223)
    linex, liney = angle_line(img, angle_flipped)
    ax = plt.gca()
    ax.plot(linex, liney, 'r')
    plt.imshow(img_flipped)
    plt.title('flipped')


    img = augment_img(img)
    plt.subplot(224)
    linex, liney = angle_line(img, angle)
    ax = plt.gca()
    ax.plot(linex, liney, 'r')
    plt.imshow(img)
    plt.title('augmented')

    plt.show()
