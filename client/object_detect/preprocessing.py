# ---- LauzHack18 ----
# OBJECT_DETECTION2.PY
# _______________________________________________________________
# Face detection and blurring using opencv 
# Based from PyimageSerach script and modified for our purposes
# by Sylvain C, Fall 2018
# ______________________________________________________________

import cv2
import numpy as np
import os

from .noise import minimize
from .patching import patch

HERE = os.path.dirname(__file__)


def preprocess(image, net, face_cascade):
    # Init people counter
    ppl_cnt = 0

    # initialize the list of class labels MobilNet SSD was trained to detect an
    # then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    COUNT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # load the input image and build an input blob for it resizing to a fixed
    # 300x300 pixel image and normalizing it
    # image = cv2.imread(image)

    # pre detection
    img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    N_faces = len(faces)
    print('[INFO] nbr faces : {}'.format(N_faces))

    for (x, y, w, h) in faces:
        img = minimize(img, img[y:y + h, x:x + w], x, y, w, h)

    # pass blob through network
    print("[INFO] computing object detections...")
    image = img
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (1000, 1000)), 0.007843, (1000, 1000), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # explicit detections
    for x in np.arange(0, detections.shape[2]):
    # get the proba
        proba = detections[0, 0, x, 2]
    # filter out low proba
    if (proba > 0.20) & (CLASSES[int(detections[0, 0, x, 1])] == "person"):
        ppl_cnt += 1
    # get label
    index = int(detections[0, 0, x, 1])
    item_nbr = COUNT[index] + 1
    COUNT[index] = COUNT[index] + 1
    box = detections[0, 0, x, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    # display pred
    label = "{} {}: {:.3f}%".format(CLASSES[index], item_nbr, proba * 100)
    print("[INFO] {}".format(label))

    # display
    print('[INFO] People count = {}'.format(max(ppl_cnt, N_faces)))

    # Patch the image in 224^2 images
    res, N_y_grid, N_x_grid = patch(image, 224, 224)
    print('[INFO] patches 224x224 completed')

    # Reompose the overall image
    patches = {}
    tmp_x = []
    for x in range(N_x_grid):
        tmp = np.concatenate(res[x * N_y_grid:(x + 1) * N_y_grid], axis=0)
        for y in range(N_y_grid):
            patches[(x, y)] = res[x * N_y_grid + y]
        tmp_x.append(tmp)
    rec_img = np.concatenate(tmp_x[:], axis=1)

    # cv2.imwrite('images/image_rec.png', rec_img)

    return rec_img, patches, max(ppl_cnt, N_faces)
