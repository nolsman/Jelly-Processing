import cv2
import numpy as np
from matplotlib import pyplot as plt
# fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=10, detectShadows=False)
fgbg = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=150, detectShadows=False)

mag = []
for n in range(1, 280):
    frame = cv2.imread('night2/Frame_'+str(n)+'.TIF')
    fgmask = fgbg.apply(frame)
    fgmask = cv2.medianBlur(fgmask, 7)
    # th, fgmask = cv2.threshold(fgmask, 50, 0, cv2.THRESH_TOZERO)
    cimg = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    fgmask = cv2.medianBlur(fgmask, 7)
    circles = cv2.HoughCircles(fgmask, cv2.HOUGH_GRADIENT, 1, 30, param1=30, param2=12, minRadius=10, maxRadius=75)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    fgmask = fgmask[0:200, 0:200]
    cv2.imshow('frame1', cimg)
    cv2.imshow('frame2', frame)
    cv2.waitKey(67)
    mag.append(sum(sum(fgmask))/200**2)
mag = mag[5:]
mag -= np.average(mag)
plt.figure(1)
plt.plot(mag)
plt.savefig('test')

t_step = 1/15
spec = np.fft.rfft(mag)
freqs = np.fft.fftfreq(len(mag), t_step)
plt.figure(2)
plt.plot([t*7.5/len(spec) for t in range(len(spec) - 1)], np.log10(np.abs(spec[1:])))
plt.axis([0, 7.5, -1, 1])
plt.savefig('freq')