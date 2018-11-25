# ---- LauzHack18 ----
# OBJECT_DETECTION2.PY
# _______________________________________________________________
# Face detection and blurring using opencv 
# Based from PyimageSerach script and modified for our purposes
# by Sylvain C, Fall 2018
# ______________________________________________________________

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from noise import noisy, minimize
from patching import patch

# Init people counter
ppl_cnt = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="patch to input image")
ap.add_argument("-p", "--prototxt", required=True, help="patch to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="patch to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.20, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobilNet SSD was trained to detect an
# then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0,255, size=(len(CLASSES),3))
COUNT = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# load our serialized model from disk
print("[INFO] loading model ...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and build an input blob for it resizing to a fixed
# 300x300 pixel image and normalizing it
image = cv2.imread(args["image"])

#pre detection 
img = image
face_cascade = cv2.CascadeClassifier('/Users/sylvainchatel/python6258/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/sylvainchatel/python6258/opencv/data/haarcascades/haarcascade_eye.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
N_faces = len(faces)
print('[INFO] nbr faces : {}'.format(N_faces))

for (x,y,w,h) in faces:
    img = minimize(img, img[y:y+h,x:x+w], x, y, w, h)

# pass blob through network
print("[INFO] computing object detections...")
image = img
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (1000, 1000)), 0.007843, (1000, 1000), 127.5)
net.setInput(blob)
detections = net.forward()

# explicit detections
for i in np.arange(0, detections.shape[2]):
    # get the proba
    proba = detections[0,0,i,2]
    # filter out low proba
    if (proba > args["confidence"]) & (CLASSES[int(detections[0,0,i,1])]== "person"):
        ppl_cnt += 1
        # get label
        index = int(detections[0,0,i,1])
        item_nbr = COUNT[index]+1
        COUNT[index] = COUNT[index] + 1
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype("int")
        # display pred
        label = "{} {}: {:.3f}%".format(CLASSES[index], item_nbr, proba*100)
        print("[INFO] {}".format(label))


# display
print('[INFO] People count = {}'.format(max(ppl_cnt, N_faces)))

# Patch the image in 224^2 images
res, N_y_grid, N_x_grid = patch(image, 224, 224)
print('[INFO] patches 224x224 completed')

#Reompose the overall image
tmp = []
tmp_x = []
for i in range(N_x_grid):
    tmp = np.concatenate(res[i*N_y_grid:(i+1)*N_y_grid], axis = 0)
    tmp_x.append(tmp)
rec_img = np.concatenate(tmp_x[:], axis=1)

cv2.imwrite('images/image_rec.png',rec_img)
