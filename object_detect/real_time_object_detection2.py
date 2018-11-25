# ---- LauzHack18 ----
# REAL_TIME_OBJECT_DETECTION.PY
# _______________________________________________________________
# Real time face detection and blurring using opencv 
# Based from PyimageSerach script and modified for our purposes
# by Sylvain C, Fall 2018
# ______________________________________________________________

# packages import
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from noise import noisy, minimize


# import face detection
face_cascade = cv2.CascadeClassifier('/Users/sylvainchatel/python6258/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/sylvainchatel/python6258/opencv/data/haarcascades/haarcascade_eye.xml')


# Build the argument parser for the function
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum rpoba to filter weak detections")
ap.add_argument("-v", "--video", help="path to video")
ap.add_argument("-a", "--min-area",type=int, default=400, help="minimum area size")
args = vars(ap.parse_args())

# Initialize the different classes for MobilleSSD recognition
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "hrose", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0,255, size=(len(CLASSES), 3))

# Load model from OpenCv
print("[INFO] loading model ...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# Start the video stream
print("[INFO] starting video stream on webcam ...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# initialize the frames for the video stream
firstFrame = None

# time for warning print
t = time.time()

# start loop for the recognition
while True:
    #get and resize frame to 400px
    frame = vs.read()
    frame = imutils.resize(frame, width=1600)
    # create grey version of the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21),0)
    if firstFrame is None:
        firstFrame = gray
        continue
    frameDelta = cv2.absdiff(firstFrame, gray)
    thrs = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    thrs = cv2.dilate(thrs, None, iterations=2)
    (cnt, _, _) = cv2.findContours(thrs.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert to blob
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (600,600)),0.007843, (600,600), 127.5)
    # put blob through net
    net.setInput(blob)
    detections = net.forward()
    

    # Loop over detections
    for i in np.arange(0, detections.shape[2]):
        # get proba
        proba = detections[0,0,i,2]
        # filter
        if proba > args["confidence"]:
            index = int(detections[0,0,i,1])
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            # plot box
            label = "{}: {:.2f}%".format(CLASSES[index], proba*100)
            
            
            # Get faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            N_faces = len(faces)
            print('[INFO] nbr faces : {}'.format(N_faces))
            for (x,y,w,h) in faces:
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                frame = minimize(frame, frame[y:y+h,x:x+w], x, y, w, h)
            
            
            #cv2.rectangle(frame, (startX, startY), (endX, endY), 2)
            y = startY -15 if startY -15 > 15 else startY + 15
            #cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[index], 2)
            # Get location of the object
            (sX, sY, eX, eY) = detections[0,0,i,3:7].astype("float")
            (cX, cY) = (float(sX+eX)/2, float(sY+eY)/2)

            if (CLASSES[index] in {"chair", "diningtable"}) and (cY >= 0.5) and (0.33 <= cX <= 0.66):
                    if (time.time() - t > 2):
                        print("[WARNING] {} ahead".format(CLASSES[index]))
                        t = time.time()
    
    
    # check that there is no more face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    N_faces = len(faces)
    print('[SUB-INFO] Still {} face(s)'.format(N_faces))
    for (x,y,w,h) in faces:
        frame = minimize(frame, frame[y:y+h,x:x+w], x, y, w, h)
    
    # Show
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if 'c' was press break
    if key == ord("c"):
        break
    # update fps counter
    fps.update()

# STOP the program
fps.stop()
print("[INFO] elapsed time : {:.2f}".format(fps.elapsed()))
#print("[INFO] app. FPS : {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop
