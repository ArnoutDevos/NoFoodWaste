import time

import cv2
import imutils
import os
from PIL import Image
from datetime import datetime
from imutils.video import FPS
from imutils.video import VideoStream

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

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    cv2.imshow("Frame", frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(frame)
    pil_im.save(os.path.join(output_folder, 'images/{}.jpg'.format(datetime.now().strftime("%Y%m%d_%H%M%S.%f"))))

    key = cv2.waitKey(150) & 0xFF
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