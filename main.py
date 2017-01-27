import numpy as np
import argparse
import cv2

from time import time

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='path to the video file')
ap.add_argument('-f', '--file', help='path to .csv file')


if __name__ == '__main__':
    args = vars(ap.parse_args())

    start_time = time()

    cap = cv2.VideoCapture(args['video'])
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # upper case - yl3

    out = cv2.VideoWriter('output.avi', fourcc, 20, size, 1)

    print 'Processing video...'

    # take first frame of the video
    ret, frame = cap.read()

    # setup initial location of window
    c, r, w, h = 468, 428, 30, 30
    track_window = (c, r, w, h)

    # set up the ROI for tracking
    hsv = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((100., 100., 150.)))
    mask_inv = cv2.bitwise_not(mask)
    roi_hist = cv2.calcHist([hsv_roi], [0], mask_inv, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    t = time()

    while True:
        ret, frame = cap.read()

        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # Draw it on image
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            out.write(img2)
            k = cv2.waitKey(30) & 0xff

            if time() - t > 15:
                break

    cv2.destroyAllWindows()
    cap.release()
    out.release()

    print 'Processing finished in {finish_time} seconds.'.format(finish_time=int(time() - start_time))