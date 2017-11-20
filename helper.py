import csv
import cv2
import numpy as np
import random
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
from data_augmentation import create_augmen_img_bytransform, create_augmen_img_byflip

##########################
def resize_img(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (128, 128))

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

def colormask(img,c_mask_low,c_mask_high, return_cmask = 0 ):
    """
    Apply a hsv colormask on image and then return it
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Threshold the HSV image to get only yellow/white
    c_mask = cv2.inRange(img_hsv, c_mask_low, c_mask_high)

    img_c_masked = cv2.bitwise_and(img, img, mask=c_mask)
    if return_cmask:
        return c_mask

    return img_c_masked

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def dynamic_crop(img, view_field, angle, perspective):

    center = 160 * (1 + angle)
    top_l = center - view_field/2
    top_r = center + view_field/2
    if perspective == 0:
        bottom_left_x = 0
        bottom_left_y = 160
        bottom_left_x2 = 20
        bottom_left_y2 = 160

        carfront_l_x = 120
        carfront_l_y = 140

        carfront_r_x = 180
        carfront_r_y = 140

        bottom_right_x = 300
        bottom_right_y = 160
        bottom_right_x2 = 320
        bottom_right_y2 = 160

    elif perspective > 0:
        bottom_left_x = 0
        bottom_left_y = 140
        bottom_left_x2 = 0
        bottom_left_y2 = 140

        carfront_l_x = 100
        carfront_l_y = 140

        carfront_r_x = 200
        carfront_r_y = 140

        bottom_right_x = 320
        bottom_right_y = 140
        bottom_right_x2 = 320
        bottom_right_y2 = 140

    # top left, top right, bottom left, bottom right
    vertices = np.array([[(bottom_left_x, bottom_left_y),
                            (bottom_left_x2, bottom_left_y2),
                            (carfront_l_x,carfront_l_y),
                            (carfront_r_x,carfront_r_y),
                            (bottom_right_x, bottom_right_y),
                            (bottom_right_x2, bottom_right_y2),
                            (320, 90),
                            (top_r, 70),
                            (top_l, 70),
                            (0, 90)]],
                        dtype=np.int32)
    # mask region
    img = region_of_interest(img, vertices)

    return img

def increase_contrast(img, clipLimit, tileGridSize):

    # # Converting image to LAB Color model
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #
    # # Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)
    #
    # # Applying CLAHE to L-channel
    tileGridSize = (tileGridSize,tileGridSize)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    cl = clahe.apply(l)
    #
    # # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    #
    # # Converting image from LAB Color model to RGB model
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def load_img(sample, perspective, steering_correction = 0.2):
    img_path = sample[perspective]
    img = mpimg.imread(img_path)
    angle = float(sample[3])

    if perspective == 1:
        angle = angle + steering_correction
    elif perspective == 2:
        angle = angle - steering_correction

    return img, angle


def crop_resize_img(img):
    # if dyn_crop:
    #     img = dynamic_crop(img, view_field = 180, angle = angle, perspective = perspective)

    img = img[69:149, :]

    img = cv2.resize(img,(160,160))

    ####################
    # orig_img = img.copy()
    # img = gaussian_blur(img, 3)
    # # define range of color in HSV normal gray
    # lower_gray = np.array([0,0,0])
    # upper_gray = np.array([179,100,255])
    # img = colormask(img,lower_gray,upper_gray)
    # #
    # # white light gray asphalt
    # lower_white= np.array([0,0,170])
    # upper_white = np.array([179,20,255])
    # img_masked2 = colormask(img,lower_white,upper_white)
    #
    # img = weighted_img(img_masked2, img_masked1,0.5,2)
    #
    # img = weighted_img(orig_img, img, 0.5,0.6)
    ##########################

    return img

def augment_img(img):

    rand = np.random.random() # decide randomly if random colorshift or brightness shift
    if rand >= 0.5:
        brightness_or_colorshift = 0
    else:
        brightness_or_colorshift = 1

    new_img = create_augmen_img_bytransform(img,
                                            range_colorshift = 0.2,
                                            brightness_or_colorshift = brightness_or_colorshift,
                                            warp_factor= 0,
                                            range_zoom = 0,
                                            range_rotate = 0,
                                            range_shift = 0)
    return new_img

def movingAverage(avg_array, new_sample):
    """takes in an array of values, a new value and computes the average"""
    avg_new = (sum(avg_array) + new_sample) / (len(avg_array)+1)
    return avg_new;

def get_gen_batch_size(desired_batch_size, augmented_per_sample, use_all_perspectives)  :
    if augmented_per_sample > 0:
        gen_batch_size = desired_batch_size //( 2 * (1 + augmented_per_sample))
    else:
        gen_batch_size = desired_batch_size // 2

    if use_all_perspectives:
        gen_batch_size = gen_batch_size//3
    return gen_batch_size
