import numpy as np
import sys
import cv2 as cv
from matplotlib import pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
imgL = cv.imread('left1.png',0)
imgR = cv.imread('right1.png',0)
stereo = cv.StereoBM_create(numDisparities=32, blockSize=9)
disparity = stereo.compute(imgL,imgR)
#disparity = cv.normalize(stereo.compute(imgL, imgR),stereo.compute(imgL, imgR), alpha=0, beta=255, \norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    
#plt.imshow(disparity,'gray')
#print(disparity)
plt.subplot(121),plt.imshow(disparity)
plt.subplot(122),plt.imshow(imgL)
#plt.imshow(imgL,'gray')
plt.show()
cv.imshow('ssss',disparity)

cv.imshow('ddd',imgL)
cv.waitKey(0)


#cv.imshow('gray1',imgL)
#cv.Waitkey(0)
#cv.DestroyAllWindows()
