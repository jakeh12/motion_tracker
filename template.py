#!/usr/bin/env python3

import numpy as np
import cv2

# print out opencv version number
print(cv2.__version__)

# open camera for capture
cap = cv2.VideoCapture(-1)

# capture a frame
ret, frame = cap.read()

# save a frame into a jpg file
cv2.imwrite('frame.jpg', frame)

# release camera
cap.release()

