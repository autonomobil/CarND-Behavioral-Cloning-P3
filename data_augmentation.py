import numpy as np
import cv2
import random

import matplotlib.pyplot as plt

### Helper functions
def create_augmen_img_byflip(img, steering):

    img = np.fliplr(img)
    steering = -steering
    return img, steering

def create_augmen_img_bytransform(img,  plot = 0, range_colorshift = 0.1, warp_factor= 3, range_zoom = 0.08, range_rotate = 8,  range_shift = 2):

    img.setflags(write=1)
    img = colorshift(img, range_colorshift, brightness_or_colorshift = 0)
    img = warp(img,warp_factor)
    img = zoom(img,range_zoom)
    img = rotate(img, range_rotate)
    img = shift(img, range_shift)

    if plot:
        org_img = img.copy()
        fig, axs = plt.subplots(1,2, figsize=(10, 10))
        axs[0].imshow(org_img)
        axs[0].set_title('original')

        axs[1].imshow(img)
        axs[1].set_title("Augmented")
        plt.show()

    return img

def colorshift(img, range_colorshift, brightness_or_colorshift):
    ######################## brightness or COLORSHIFT
    if range_colorshift > 0:
        if brightness_or_colorshift ==  0:
            rand_shift = np.random.uniform(1 - range_colorshift, 1)
            img = img * rand_shift

        elif brightness_or_colorshift == 1:
            for color in range(3):
                rand_shift = np.random.uniform(1 - range_colorshift, 1)
                img[:,:,color ] = img [:,:,color ] * rand_shift
    return img

def warp(img,warp_factor):
    if warp_factor > 0:
        y, x = img.shape[:2]
        ######################## WARP
        f1 = random.uniform(-1, 1 )
        f2 = random.uniform(-1, 1 )
        f3 = random.uniform(-1, 1 )

        pts1 = np.float32([[0,0],[x,0],[0,y]])
        pts2 = np.float32([[0 + warp_factor*f1*x, 0 + warp_factor*f1*y],[x - warp_factor*f2*x, 0 + warp_factor*f2*y],[0+warp_factor*f3*x, y-warp_factor*f3*y]])

        M = cv2.getAffineTransform(pts1,pts2)
        img = cv2.warpAffine(img,M,(x,y))
    return img

def zoom(img,range_zoom):
    if range_zoom > 0:
        y, x = img.shape[:2]
        ######################## ZOOM
        zoom_factor = random.uniform(1 - range_zoom, 1 + range_zoom)
        newx = (x // zoom_factor )
        newy = (y // zoom_factor )

        deltax = newx - x
        deltay = newy - y

        pts1 = np.float32([[0,0],[x,0],[0,y]])
        pts2 = np.float32([[-deltax,-deltay],[newx,-deltay],[-deltax,newy]])

        M = cv2.getAffineTransform(pts1,pts2)
        img = cv2.warpAffine(img,M,(x,y))
    return img

def rotate(img, range_rotate):
    if range_rotate>0:
        y, x = img.shape[:2]
        ########################## ROTATE
        angle = np.random.randint(-range_rotate, range_rotate)

        M = cv2.getRotationMatrix2D((x/2,y/2),angle,1)
        img = cv2.warpAffine(img,M,(x,y))

    return img

def shift(img, range_shift):
    if range_shift>0:
        y, x = img.shape[:2]
        ####################### move
        dx = np.random.randint(-range_shift, range_shift)
        dy = np.random.randint(-range_shift, range_shift)

        # roll
        img = np.roll(img, dx, 1)
        img = np.roll(img, dy, 0)

        if dx > 0:
            img[:,0:dx,:] = 0
        elif dx < 0:
            img[:,dx:,:] = 0

        if dy > 0:
            img[0:dy,:,:] = 0
        elif dy < 0:
            img[dy:,:,:] = 0

    return img
