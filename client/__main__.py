import time

import json
import logging
from io import BytesIO

import cv2
import imutils
import os
import requests
from PIL import Image
from datetime import datetime
from imutils.video import FPS
from imutils.video import VideoStream

from object_detect.preprocessing import preprocess

logging.basicConfig()

output_folder = os.path.join(os.path.dirname(__file__), '..')

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=1)
cap = vs.stream.stream
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
vs.start()
time.sleep(1)

# start the FPS throughput estimator
fps = FPS().start()

HERE = os.path.dirname(__file__)
# load our serialized model from disk
print("[INFO] loading model ...")
net = cv2.dnn.readNetFromCaffe(
    os.path.join(HERE, "object_detect/MobileNetSSD_deploy.prototxt.txt"),
    os.path.join(HERE, "object_detect/MobileNetSSD_deploy.caffemodel"),
)

# load the input image and build an input blob for it resizing to a fixed
# 300x300 pixel image and normalizing it
# image = cv2.imread(image)

# pre detection
face_cascade = cv2.CascadeClassifier(os.path.join(HERE, 'object_detect/haarcascade_frontalface_default.xml'))


def to_bytes(frame):
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # filename = '{}.jpg'.format(datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
    # pil_im.save(os.path.join(output_folder, 'images/{}'.format(filename)))

    byte_io = BytesIO()
    pil_im.save(byte_io, 'jpeg')
    byte_io.seek(0)

    return byte_io


# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    print(frame.shape)

    frame, patches, people = preprocess(frame, net, face_cascade)

    try:
        r = requests.post(
            'http://localhost:8000/food-watch/picture-events',
            data=dict(
                json=json.dumps(dict(
                    patches=[
                        dict(x=x, y=y)
                        for (x, y) in patches.keys()
                    ],
                )),
            ),
            files={
                'patch_{}_{}'.format(x, y): ('{}_{}'.format(x, y), to_bytes(patch))
                for (x, y), patch in patches.items()
            },
        )
        r.raise_for_status()
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as ex:
        logging.exception(ex)

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, height=600)
    (h, w) = frame.shape[:2]

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30_000) & 0xFF
    if key == ord("q"):
        break

    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
