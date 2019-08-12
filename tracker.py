#!/usr/bin/env python3

import numpy as np
import cv2
import time


NUM_FRAMES = 128
FRAME_SIZE = 128
MOVING_AVG_N = 8
THRESHOLD = 10
GAUSS_BLUR = 3
MIN_AREA = 300


# lists for results
results = []
results_frames = []

# open camera for capture
cap = cv2.VideoCapture(-1)

# capture a throw away frame to turn on the camera
cap.read()

# keep track of last known x and y coordinates
last_x = 0.5
last_y = 0.5

# initialize moving average windows to center values
avg_x = 0.5
avg_y = 0.5
avg_x_window = [0.5] * MOVING_AVG_N
avg_y_window = [0.5] * MOVING_AVG_N

print('capture starting in')
time.sleep(1.0)
print('3...')
time.sleep(1.0)
print('2...')
time.sleep(1.0)
print('1...')
time.sleep(1.0)
print('')
print('capturing...')

start_time = time.time()

# capture and process frames
prev_frame = None
for i in range(0, NUM_FRAMES):
    
    # capture a frame
    ret, frame = cap.read()
   
    # scale down image for faster processing
    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)

    # convert image to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur image to get rid of high frequency noise
    frame = cv2.GaussianBlur(frame, (GAUSS_BLUR, GAUSS_BLUR), 0)

    # skip the very first frame, we need at least two to do processing
    if i == 0:
        prev_frame = frame
        continue

    # calculate difference between current and previous frame
    frame_diff = cv2.absdiff(prev_frame, frame)

    # save current frame into prev_frame for next iteration
    prev_frame = frame

    # perform binary threshold on the frame difference
    frame_diff = cv2.threshold(frame_diff, THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded frame to fill holes
    frame_diff = cv2.dilate(frame_diff, None, iterations=4)

    # find contours
    frame_contours, __ = cv2.findContours(frame_diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort countours by area
    frame_contours_sorted = sorted(frame_contours, key=lambda x: cv2.contourArea(x))

    # prepare frame for drawing centroids
    frame_centroids = frame_diff.copy()
    frame_centroids = cv2.cvtColor(frame_centroids, cv2.COLOR_GRAY2BGR)
    
    # make sure some contour was found and the contour area is bigger than threshold
    if (frame_contours_sorted and cv2.contourArea(frame_contours_sorted[len(frame_contours_sorted)-1]) > MIN_AREA): 

        # compute centroid
        contour_moments = cv2.moments(frame_contours_sorted[len(frame_contours_sorted)-1])
        contour_x = int(contour_moments['m10'] / contour_moments['m00'])
        contour_y = int(contour_moments['m01'] / contour_moments['m00'])
        
        # update last x and y coordinates
        last_x = contour_x / frame.shape[:2][0]
        last_y = contour_y / frame.shape[:2][1]
       
        # update moving average
        avg_x_window = avg_x_window[1:] + [last_x]
        avg_y_window = avg_y_window[1:] + [last_y]
        avg_x = sum(avg_x_window)/MOVING_AVG_N
        avg_y = sum(avg_y_window)/MOVING_AVG_N
    
    # draw tracked point in a frame
    cv2.circle(frame_centroids, (int(avg_x * frame.shape[:2][0]), int(avg_y * frame.shape[:2][1])), 4, (0,0,255), -1)

    # append result lists
    results.append((avg_x, avg_y))
    results_frames.append(frame_centroids.copy())

    
# calculate fps
duration = time.time() - start_time

# print tracking point coordinates
for coords in results:
    print('{:.2}, {:.2}'.format(coords[0], coords[1]))

# save all frames to files
for i, frame in enumerate(results_frames):
    cv2.imwrite('frames/frame_{:04d}.ppm'.format(i), frame)
    
# print fps
print('--------')
print('fps: {:.2f}'.format(NUM_FRAMES/duration))
print('--------')

# release camera
cap.release()

