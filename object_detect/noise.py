# ---- LauzHack18 ----
# NOISE.PY
# _______________________________________________________________
# Helper functions using opencv 
# Based from PyimageSerach script and modified for our purposes
# by Sylvain C, Fall 2018
# ______________________________________________________________


import numpy as np
import os
import cv2

def minimize(img, sub_face, x, y, w, h):
    if (w, h) != (0, 0):
        # get random parameters per person
        [p1, p2, p3] = np.random.randint(10, 100, size=3)
        # blur people
        sub_face = cv2.GaussianBlur(sub_face,(2*p1+1, 2*p2+1), p3)
        sub_face = cv2.GaussianBlur(sub_face,(25, 25), 100)
        img[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
        
        return img

def noisy(noise_typ,image, param=(0, 0.1)):
    (p1, p2) = param
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = p1
        var = p2
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = p1
        amount = p2
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy