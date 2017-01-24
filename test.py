# import the necessary packages
from collections import deque
import argparse
import imutils
import cv2

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "ball"
ballLower = (0, 0, 190)
ballUpper = (60, 60, 255)

# list of tracked points
pts = deque(maxlen=args["buffer"])

# check argument
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])

while (1):
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV color space
	frame = imutils.resize(frame, width=600)
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# construct a mask for the color
	mask = cv2.inRange(hsv, ballLower, ballUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > 5:
			# draw the circle
			cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
			pts.appendleft(center)

	# show the frame to our screen and increment the frame counter
	cv2.imshow("Ball Tracker", frame)

	# to exit press 'ESC'
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()


# import numpy as np
# import cv2
# import imutils
#
# cap = cv2.VideoCapture(0)
# ballUpper = (200, 224, 220)
# ballLower = (172, 184, 171)
# # print(cap.get(cv2.cv.CV_CAP_PROP_FPS))
#
#
# while (cap.isOpened()):
# 	ret, frame = cap.read()
# 	cv2.waitKey(1)
# 	frame = imutils.resize(frame, width=480)
# 	frameCopy = frame.copy()
# 	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# 	mask = cv2.inRange(frame, ballLower, ballUpper)
# 	# gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	gray_image_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
# 	hough = cv2.HoughCircles(gray_image_blurred, cv2.HOUGH_GRADIENT, 50, 500, maxRadius=80)
#
# 	if hough is not None:
# 		hough = np.round(hough[0, :]).astype("int")
# 		for (x, y, r) in hough:
# 			cv2.circle(frameCopy, (x, y), r, (0, 255, 0), 4)
# 			cv2.rectangle(frameCopy, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#
# 	# cv2.imshow('frame',frameCopy)
# 	# cv2.imshow('mask',mask)
# 	cv2.waitKey(10)
# 	# cv2.imshow('gray',hough)
# 	cv2.imshow("output", np.hstack([frame, frameCopy]))
# 	cv2.waitKey(10)
#
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break
#
# cap.release()
# cv2.destroyAllWindows()