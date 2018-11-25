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

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    print(frame.shape)

    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    filename = '{}.jpg'.format(datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
    pil_im.save(os.path.join(output_folder, 'images/{}'.format(filename)))

    byte_io = BytesIO()
    pil_im.save(byte_io, 'jpeg')
    byte_io.seek(0)

    try:
        r = requests.post(
            'http://localhost:8000/food-watch/picture-events',
            data=dict(
                json=json.dumps(dict(
                    patches=[
                        dict(x=0, y=0),
                    ],
                )),
            ),
            files={
                'patch_0_0': ('picture.jpg', byte_io),
            },
        )
        r.raise_for_status()
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as ex:
        logging.exception(ex)

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(5_000) & 0xFF
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
