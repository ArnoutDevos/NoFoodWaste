# ---- LauzHack18 ----
# PATCHING.PY
# _______________________________________________________________
# Patching function to divide the image into 224x224 images
# by Sylvain C, Fall 2018
# ______________________________________________________________

import numpy as np
import argparse
import cv2
from noise import noisy, minimize

def patch(image, width=224, height=224, privacy= True):
    
    # get image size
    (h, w) = image.shape[:2]
    
    N_x_grid = w//width+1
    N_y_grid = h//height+1
    
    res = []
    
    face_cascade = cv2.CascadeClassifier('/Users/sylvainchatel/python6258/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/Users/sylvainchatel/python6258/opencv/data/haarcascades/haarcascade_eye.xml')
    
    for i in range(N_x_grid):
        for j in range(N_y_grid):
            X = i*width
            Y = j*height
            title = 'image{}{}'.format(i,j)
            sub_image = image[Y:Y+height, X:X+width]
            
            # detect faces
            gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) != 0:
                print('[SUB-INFO] {} face(s) detected and being minimized'.format(len(faces)))
            
            for (x,y,w,h) in faces:
                #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                minimize(sub_image, sub_image[y:y+h,x:x+w], x, y, w, h)
            
            (hh, ww) = sub_image.shape[:2]
            sub_image = cv2.copyMakeBorder(sub_image,0,224-hh,0,224-ww,cv2.BORDER_CONSTANT,value=[0,0,0])
            sub_image = cv2.resize(sub_image, (224, 224))
            res.append(sub_image)
            title = 'images/image_{}_{}.png'.format(i,j)
            cv2.imwrite(title,sub_image)
    return res, N_y_grid, N_x_grid

                    